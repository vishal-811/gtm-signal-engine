"""Extraction and drafting, with the LLM call stubbed.

The model calls themselves are not tested here — what is tested is the logic
wrapped around them, which is where the correctness risk actually lives: batch
rendering, the source-URL backfill (which can attach the wrong article to the
wrong company), failure isolation, and the outreach word ceiling.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from signal_engine import draft, extract
from signal_engine.llm import RefusalError
from signal_engine.schemas import (
    Article,
    Candidate,
    ExtractionBatch,
    FundingEvent,
    OpeningsResult,
    Outreach,
    ScoreResult,
)

NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)


def _article(n: int) -> Article:
    return Article(
        title=f"Company{n} raises money",
        url=f"https://news.example.com/{n}",
        source="Test Feed",
        published_at=NOW,
        summary=f"Company{n} raised a round.",
    )


def _event(n: int, url: str = "") -> FundingEvent:
    return FundingEvent(
        is_funding_announcement=True,
        company_name=f"Company{n}",
        round_stage="seed",
        sector="devtools",
        one_line_description="x",
        source_url=url,
        extraction_confidence=0.9,
    )


class TestRenderBatch:
    def test_numbers_articles_and_includes_every_field(self):
        rendered = extract._render_batch([_article(1), _article(2)])

        assert 'index="1"' in rendered and 'index="2"' in rendered
        assert "Company1 raises money" in rendered
        assert "https://news.example.com/2" in rendered
        assert "2026-08-01" in rendered

    def test_handles_a_missing_publication_date(self):
        article = _article(1)
        article.published_at = None
        assert "unknown" in extract._render_batch([article])

    def test_handles_an_empty_summary(self):
        article = _article(1)
        article.summary = ""
        assert "no summary provided" in extract._render_batch([article])

    def test_states_the_expected_record_count(self):
        assert "3 articles" in extract._render_batch([_article(i) for i in range(3)])


class TestExtractBatching:
    def test_splits_into_batches_of_the_configured_size(self, monkeypatch):
        seen_batches: list[int] = []

        def fake_call(**kwargs):
            count = kwargs["user"].count("<article ")
            seen_batches.append(count)
            return ExtractionBatch(events=[_event(i) for i in range(count)])

        monkeypatch.setattr(extract, "structured_call", fake_call)
        events = extract.extract([_article(i) for i in range(25)], batch_size=10)

        assert sorted(seen_batches) == [5, 10, 10]
        assert len(events) == 25

    def test_backfills_the_authoritative_source_url(self, monkeypatch):
        # Models paraphrase or truncate URLs; the article's own URL is the one
        # the outreach draft cites, so it must win.
        monkeypatch.setattr(
            extract,
            "structured_call",
            lambda **kw: ExtractionBatch(
                events=[_event(0, url="https://wrong.example.com/hallucinated")]
            ),
        )
        events = extract.extract([_article(7)], batch_size=10)

        assert events[0].source_url == "https://news.example.com/7"

    def test_does_not_backfill_when_the_record_count_disagrees(self, monkeypatch):
        # With a count mismatch, position no longer identifies the article, so
        # backfilling would attach one company's name to another's article.
        monkeypatch.setattr(
            extract,
            "structured_call",
            lambda **kw: ExtractionBatch(
                events=[_event(0, url="https://model.example.com/a")]
            ),
        )
        events = extract.extract([_article(1), _article(2), _article(3)], batch_size=10)

        assert len(events) == 1
        assert events[0].source_url == "https://model.example.com/a"

    def test_a_refused_batch_yields_nothing_and_does_not_raise(self, monkeypatch):
        def refuse(**kwargs):
            raise RefusalError("declined")

        monkeypatch.setattr(extract, "structured_call", refuse)
        assert extract.extract([_article(1)]) == []

    def test_one_failed_batch_does_not_lose_the_others(self, monkeypatch):
        calls = {"n": 0}

        def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("upstream hiccup")
            count = kwargs["user"].count("<article ")
            return ExtractionBatch(events=[_event(i) for i in range(count)])

        monkeypatch.setattr(extract, "structured_call", flaky)
        events = extract.extract([_article(i) for i in range(20)], batch_size=10)

        # One batch of 10 was lost; the other survived.
        assert len(events) == 10

    def test_empty_input_makes_no_calls(self, monkeypatch):
        def explode(**kwargs):
            raise AssertionError("should not call the API for zero articles")

        monkeypatch.setattr(extract, "structured_call", explode)
        assert extract.extract([]) == []


class TestDraft:
    def _candidate(self) -> Candidate:
        candidate = Candidate(event=_event(1), market="sf-bay-area")
        candidate.composite = 4.0
        candidate.openings = OpeningsResult(
            status="verified",
            ats_provider="greenhouse",
            eng_role_count=5,
            sample_titles=["Backend Engineer", "SRE"],
        )
        candidate.score = ScoreResult(criteria=[], key_signal="Five backend roles open.")
        return candidate

    def test_render_offers_exact_titles_the_model_may_cite(self):
        rendered = draft.render_candidate(self._candidate())
        assert "Backend Engineer" in rendered
        assert "Five backend roles open." in rendered

    def test_render_forbids_inventing_a_title_when_none_are_known(self):
        candidate = self._candidate()
        candidate.openings = OpeningsResult(status="unverified")
        rendered = draft.render_candidate(candidate)

        assert "do NOT invent a job title" in rendered
        assert "Backend Engineer" not in rendered

    def test_over_length_body_keeps_its_ending(self, monkeypatch):
        """An over-length draft must not be decapitated.

        Trimming to the word ceiling cut from the end — which is exactly where
        the call to action and the signature are — producing an email that
        stopped mid-sentence and was signed by nobody. Every draft is reviewed
        by a human before sending, so slightly long is strictly better.
        """
        long_body = (
            " ".join(f"w{i}" for i in range(150))
            + " Are you open to reviewing a shortlist? Ankur, Founder, Hire100x"
        )
        monkeypatch.setattr(
            draft,
            "structured_call",
            lambda **kw: Outreach(subject="s", body=long_body, personalization_hook="h"),
        )
        result = draft.draft_one(self._candidate())

        assert result.outreach is not None
        assert result.outreach.body == long_body, "must not be truncated"
        assert result.outreach.body.rstrip().endswith("Hire100x"), (
            "the signature is the part truncation used to remove"
        )
        assert not result.outreach.body.endswith("\u2026")

    def test_body_within_the_ceiling_is_left_alone(self, monkeypatch):
        body = "Short and to the point."
        monkeypatch.setattr(
            draft,
            "structured_call",
            lambda **kw: Outreach(subject="s", body=body, personalization_hook="h"),
        )
        assert draft.draft_one(self._candidate()).outreach.body == body

    def test_a_failed_draft_leaves_the_candidate_usable(self, monkeypatch):
        # The company still belongs in the digest; only the email is missing.
        def boom(**kwargs):
            raise RuntimeError("nope")

        monkeypatch.setattr(draft, "structured_call", boom)
        result = draft.draft_one(self._candidate())

        assert result.outreach is None
        assert result.composite == 4.0

    def test_refusal_is_handled_like_any_other_failure(self, monkeypatch):
        def refuse(**kwargs):
            raise RefusalError("declined")

        monkeypatch.setattr(draft, "structured_call", refuse)
        assert draft.draft_one(self._candidate()).outreach is None

    def test_empty_input(self):
        assert draft.draft_all([]) == []


class TestDraftSystemPrompt:
    def test_includes_the_sender_identity(self, monkeypatch):
        cfg = draft.settings()
        monkeypatch.setattr(cfg, "sender_name", "Vishal", raising=False)
        monkeypatch.setattr(cfg, "sender_title", "Founder", raising=False)
        monkeypatch.setattr(cfg, "sender_email", "v@example.com", raising=False)

        rendered = draft.system_prompt()
        assert "Vishal" in rendered
        assert "Founder" in rendered
        assert "v@example.com" in rendered

    @pytest.mark.parametrize(
        "field,marker",
        [
            ("sender_name", "SENDER_NAME not set"),
            ("sender_title", "SENDER_TITLE not set"),
            ("sender_email", "SENDER_EMAIL not set"),
        ],
    )
    def test_flags_a_missing_field_rather_than_signing_blankly(
        self, field, marker, monkeypatch
    ):
        # A blank signature would ship a cold email signed by nobody; the
        # visible marker makes the misconfiguration obvious in the draft.
        monkeypatch.setattr(draft.settings(), field, "", raising=False)
        assert marker in draft.system_prompt()

    def test_instructs_the_model_not_to_alter_the_signature(self, monkeypatch):
        # The sender address must appear verbatim — an invented or "corrected"
        # variant would put a wrong reply-to on real outreach.
        monkeypatch.setattr(
            draft.settings(), "sender_email", "v@example.com", raising=False
        )
        assert "Do not alter them" in draft.system_prompt()

    def test_carries_the_no_fabrication_rule(self):
        assert "Never state a fact that is not in the input" in draft.system_prompt()


class TestBatchSplitOnRejection:
    """Some endpoints reject a combined request while accepting each article in
    it individually — AgentRouter answers `content-blocked` to certain batches.
    Losing all ten articles to one poisoned neighbour is the difference between
    a full shortlist and a thin one."""

    def test_recognises_a_whole_request_rejection(self):
        assert extract._is_batch_rejection(
            RuntimeError("Error code: 400 - {'code': 'content-blocked'}")
        )

    def test_ignores_ordinary_failures(self):
        # Splitting on a rate limit or auth error would multiply the problem.
        for message in ("rate limit exceeded", "invalid api key", "timeout"):
            assert not extract._is_batch_rejection(RuntimeError(message))

    def test_splitting_recovers_everything_when_only_batches_are_rejected(
        self, monkeypatch
    ):
        # This is AgentRouter's observed behaviour: a combined request is
        # refused while every article in it succeeds on its own.
        def batches_only(**kwargs):
            user = kwargs["user"]
            count = user.count("<article ")
            if count > 1:
                raise RuntimeError("400 content-blocked")
            return ExtractionBatch(events=[_event(0)])

        monkeypatch.setattr(extract, "structured_call", batches_only)
        events = extract.extract([_article(i) for i in range(4)], batch_size=4)

        # Without splitting this would be 0 articles. With it, all 4 survive.
        assert len(events) == 4

    def test_one_poisoned_article_costs_only_itself(self, monkeypatch):
        bad = _article(3)

        def selective(**kwargs):
            user = kwargs["user"]
            if bad.url in user:
                raise RuntimeError("400 content-blocked")
            count = user.count("<article ")
            return ExtractionBatch(events=[_event(i) for i in range(count)])

        monkeypatch.setattr(extract, "structured_call", selective)
        events = extract.extract([_article(1), _article(2), bad, _article(4)],
                                 batch_size=4)

        # The offending article is dropped alone; its neighbours are recovered.
        assert len(events) == 3

    def test_a_single_article_rejection_drops_only_that_article(self, monkeypatch):
        def always_blocked(**kwargs):
            raise RuntimeError("400 content-blocked")

        monkeypatch.setattr(extract, "structured_call", always_blocked)
        assert extract.extract([_article(1), _article(2)], batch_size=2) == []

    def test_non_rejection_errors_do_not_trigger_splitting(self, monkeypatch):
        calls = {"n": 0}

        def boom(**kwargs):
            calls["n"] += 1
            raise RuntimeError("upstream exploded")

        monkeypatch.setattr(extract, "structured_call", boom)
        extract.extract([_article(i) for i in range(4)], batch_size=4)

        # One attempt, not a recursive cascade of retries.
        assert calls["n"] == 1
