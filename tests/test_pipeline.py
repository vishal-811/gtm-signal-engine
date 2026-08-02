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
