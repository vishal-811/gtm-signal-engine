"""Pipeline wiring.

These exist because of a real escape: `_finish` read `llm.usage.cache_read_tokens`,
a field that was renamed to `cached_tokens` during the provider swap. It crashed
every single run — dry or live, with or without candidates — and the whole
294-test suite missed it, because every test stubbed the LLM layer and nothing
touched the accounting path that joins the two modules together.

The lesson generalizes: cross-module attribute access is exactly what unit tests
with mocks do not cover.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from signal_engine import llm, pipeline
from signal_engine.filters import FilterReport
from signal_engine.schemas import Article
from signal_engine.schemas import RunStats


class TestUsageContract:
    """pipeline and cli read attributes off llm.usage by name. A rename in
    llm.py that misses a caller is invisible until runtime."""

    @pytest.mark.parametrize(
        "attr", ["input_tokens", "output_tokens", "cached_tokens", "calls", "cost_usd"]
    )
    def test_every_field_the_pipeline_reads_exists(self, attr):
        assert hasattr(llm.usage, attr), (
            f"llm.usage has no {attr!r}; pipeline.py or cli.py reads it and will "
            "crash at runtime"
        )

    def test_runstats_accepts_what_finish_assigns(self):
        # Mirrors the assignments in pipeline._finish exactly.
        stats = RunStats(started_at=datetime.now(timezone.utc))
        stats.input_tokens = llm.usage.input_tokens
        stats.output_tokens = llm.usage.output_tokens
        stats.cache_read_tokens = llm.usage.cached_tokens
        stats.estimated_cost_usd = llm.usage.cost_usd

        assert stats.estimated_cost_usd >= 0

    def test_summary_renders_without_a_configured_rate(self):
        # cost_usd returns 0.0 when rates are unset; summary must still format.
        llm.reset_usage()
        assert "calls" in llm.usage.summary()


class TestFinish:
    """The accounting path runs on every code path, including the early return
    when nothing survives filtering — which is the common case on a quiet day."""

    def _stats(self) -> RunStats:
        return RunStats(started_at=datetime.now(timezone.utc), dry_run=True)

    def test_dry_run_with_no_candidates_completes(self):
        llm.reset_usage()
        result = pipeline._finish(
            self._stats(), [], [], FilterReport(), [], None,
            datetime.now(timezone.utc).date(), True,
        )
        assert result.posted == []
        assert result.stats.finished_at is not None
        assert result.sheet_url is None

    def test_dry_run_never_touches_a_sink(self):
        # Passing sheets_client=None would raise if _finish tried to write.
        llm.reset_usage()
        result = pipeline._finish(
            self._stats(), [], [], FilterReport(), [], None,
            datetime.now(timezone.utc).date(), True,
        )
        assert result.stats.dry_run is True

    def test_token_totals_are_carried_onto_the_run_record(self):
        llm.reset_usage()
        llm.usage.input_tokens = 1234
        llm.usage.output_tokens = 567
        llm.usage.cached_tokens = 89
        try:
            result = pipeline._finish(
                self._stats(), [], [], FilterReport(), [], None,
                datetime.now(timezone.utc).date(), True,
            )
            assert result.stats.input_tokens == 1234
            assert result.stats.output_tokens == 567
            assert result.stats.cache_read_tokens == 89
        finally:
            llm.reset_usage()


def test_extraction_failures_are_counted_not_swallowed(monkeypatch):
    """A failed batch returns [], which downstream cannot tell from 'no news'.

    Regression: a production run lost every article to a gateway error,
    extracted nothing, and exited 0. The cron went green.
    """
    from signal_engine import extract as extract_mod

    extract_mod.reset_failures()

    def boom(_articles):
        raise RuntimeError("upstream returned HTTP 200 but the body is not a chat completion")

    monkeypatch.setattr(extract_mod, "_extract_once", boom)
    arts = [
        Article(title=f"t{i}", url=f"https://n/{i}", source="s", summary="x")
        for i in range(8)
    ]
    events = extract_mod.extract(arts, batch_size=4)

    assert events == []
    assert extract_mod.failures.articles == 8, "every lost article must be counted"
    assert extract_mod.failures.batches == 2
    assert extract_mod.failures.reasons, "the reason must be kept for the run log"


def test_a_quiet_day_records_no_failures(monkeypatch):
    """The counter must not fire when articles simply are not funding news."""
    from signal_engine import extract as extract_mod

    extract_mod.reset_failures()
    monkeypatch.setattr(extract_mod, "_extract_once", lambda a: [])
    extract_mod.extract(
        [Article(title="t", url="https://n/1", source="s", summary="x")],
        batch_size=4,
    )
    assert extract_mod.failures.articles == 0


def test_scoring_failures_are_counted_not_read_as_weak_candidates(monkeypatch):
    """A dead scoring stage must not look like a shortlist of weak companies.

    A failed score leaves composite unset, which above_threshold() reads as
    "below the bar" — identical to a company that was scored and rejected. So a
    provider outage produced an empty shortlist and a green run. Extraction
    already counted its losses; scoring did not.
    """
    from signal_engine import score as score_mod

    score_mod.reset_failures()
    monkeypatch.setattr(
        score_mod, "structured_call",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("gateway 503")),
    )
    cands = [_scoring_candidate(f"Co{i}") for i in range(3)]
    out = score_mod.score_all(cands)

    assert all(c.composite is None for c in out)
    assert score_mod.failures.candidates == 3
    assert "gateway 503" in score_mod.failures.reasons[0]


def test_a_refusal_is_not_counted_as_a_stage_failure(monkeypatch):
    """A refusal is about one company, not a broken stage."""
    from signal_engine import llm, score as score_mod

    score_mod.reset_failures()
    monkeypatch.setattr(
        score_mod, "structured_call",
        lambda **kw: (_ for _ in ()).throw(llm.RefusalError("declined")),
    )
    score_mod.score_all([_scoring_candidate("Co")])
    assert score_mod.failures.candidates == 0


def _scoring_candidate(name: str):
    from signal_engine.schemas import Candidate, FundingEvent

    return Candidate(event=FundingEvent(
        is_funding_announcement=True, company_name=name, company_domain=f"{name}.com",
        round_stage="seed", investors=[], sector="ai", one_line_description="d",
        source_url="https://n", extraction_confidence=0.9))
