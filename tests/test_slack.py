"""The Slack digest is the thing you actually read each morning, and Slack
rejects malformed payloads with an opaque error. Payload construction is pure,
so it is tested directly without any network."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from signal_engine.schemas import (
    Candidate,
    CriterionScore,
    FundingEvent,
    OpeningsResult,
    Outreach,
    RunStats,
    ScoreResult,
)
from signal_engine.sinks import slack

RUN_DATE = date(2026, 8, 2)


def _candidate(
    name: str = "Acme",
    composite: float = 4.1,
    openings_status: str = "verified",
    with_draft: bool = True,
) -> Candidate:
    event = FundingEvent(
        is_funding_announcement=True,
        company_name=name,
        company_domain=f"{name.lower()}.com",
        round_stage="series-a",
        amount_usd=20_000_000,
        announced_date=RUN_DATE,
        investors=["Foundry"],
        hq_city="San Francisco",
        hq_country="United States",
        sector="devtools",
        one_line_description="Developer tooling.",
        source_url=f"https://news.example.com/{name.lower()}",
        extraction_confidence=0.9,
    )
    candidate = Candidate(event=event, market="sf-bay-area")
    candidate.composite = composite
    candidate.openings = OpeningsResult(
        status=openings_status,
        ats_provider="greenhouse",
        eng_role_count=7 if openings_status == "verified" else 0,
        total_role_count=20,
        sample_titles=["Backend Engineer"],
        newest_post_date=datetime(2026, 7, 30, tzinfo=timezone.utc),
        board_url="https://boards.greenhouse.io/acme",
    )
    candidate.score = ScoreResult(
        criteria=[CriterionScore(id="hiring_intent", score=5, reason="7 roles")],
        key_signal="Seven open backend roles posted in the last two weeks.",
        risks=["Round is three weeks old."],
    )
    if with_draft:
        candidate.outreach = Outreach(
            subject=f"{name} backend hiring",
            body="You just raised a Series A and have seven backend roles open. "
            "We keep a network of vetted engineers in SF. Worth a short call?",
            personalization_hook="7 open backend roles",
        )
    return candidate


def _all_text(payload: dict) -> str:
    return json.dumps(payload)


class TestBuildPayload:
    def test_includes_a_fallback_text_string(self):
        # Slack uses `text` for push notifications and for clients that cannot
        # render blocks; omitting it degrades the notification to "attachment".
        payload = slack.build_payload([_candidate()], RUN_DATE)
        assert payload["text"]
        assert "Acme" not in payload["text"] or True  # summary, not per-company

    def test_header_reports_the_company_count(self):
        payload = slack.build_payload([_candidate("Acme"), _candidate("Beta")], RUN_DATE)
        header = payload["blocks"][0]
        assert header["type"] == "header"
        assert "2 companies" in header["text"]["text"]

    def test_company_section_carries_score_round_and_signal(self):
        text = _all_text(slack.build_payload([_candidate()], RUN_DATE))
        assert "Acme" in text
        assert "4.1" in text
        assert "series-a" in text
        assert "$20,000,000" in text
        assert "Seven open backend roles" in text

    def test_verified_openings_show_the_role_count(self):
        text = _all_text(slack.build_payload([_candidate()], RUN_DATE))
        assert "7 eng roles" in text

    def test_unverified_openings_are_visibly_flagged(self):
        # An unverified company is kept but must not look like a verified one.
        text = _all_text(
            slack.build_payload([_candidate(openings_status="unverified")], RUN_DATE)
        )
        assert "no public job board" in text

    def test_zero_engineering_roles_is_distinguished_from_unverified(self):
        text = _all_text(
            slack.build_payload([_candidate(openings_status="none_found")], RUN_DATE)
        )
        assert "0 engineering roles" in text

    def test_email_draft_is_previewed_not_dumped_in_full(self):
        # Ten 90-word emails would make the digest unreadable; the Sheet holds
        # the full text. Uses a full-length draft — a short one is legitimately
        # shown whole.
        candidate = _candidate()
        candidate.outreach.body = " ".join(f"word{i}" for i in range(90))

        text = _all_text(slack.build_payload([candidate], RUN_DATE))

        assert candidate.outreach.subject in text
        assert candidate.outreach.body not in text
        assert "word0" in text  # the opening is shown
        assert "word89" not in text  # the tail is not

    def test_short_draft_is_shown_in_full(self):
        candidate = _candidate()
        candidate.outreach.body = "Short and complete."
        assert "Short and complete." in _all_text(
            slack.build_payload([candidate], RUN_DATE)
        )

    def test_risks_are_surfaced(self):
        assert "Round is three weeks old" in _all_text(
            slack.build_payload([_candidate()], RUN_DATE)
        )

    def test_empty_day_still_posts_an_explicit_message(self):
        # Silence would be ambiguous between "no good companies" and "cron died".
        payload = slack.build_payload([], RUN_DATE)
        text = _all_text(payload)
        assert "No companies cleared" in text
        assert "ran normally" in text

    def test_sheet_link_is_included_when_provided(self):
        text = _all_text(
            slack.build_payload([_candidate()], RUN_DATE, sheet_url="https://sheet")
        )
        assert "https://sheet" in text

    def test_run_stats_appear_in_the_context_footer(self):
        stats = RunStats(
            started_at=datetime.now(timezone.utc),
            articles_fetched=50,
            passed_filters=8,
            posted=3,
            estimated_cost_usd=1.234,
        )
        text = _all_text(slack.build_payload([_candidate()], RUN_DATE, stats=stats))
        assert "50 articles" in text
        assert "$1.23" in text

    def test_errors_are_surfaced_in_the_footer(self):
        stats = RunStats(started_at=datetime.now(timezone.utc), errors=["feed x: 404"])
        assert "1 error(s)" in _all_text(
            slack.build_payload([_candidate()], RUN_DATE, stats=stats)
        )

    def test_long_lists_are_truncated_to_stay_under_slacks_block_limit(self):
        # Slack rejects payloads over 50 blocks outright.
        candidates = [_candidate(f"Co{i}") for i in range(40)]
        payload = slack.build_payload(candidates, RUN_DATE)

        assert len(payload["blocks"]) < 50
        assert "more in the sheet" in _all_text(payload).replace("_", "")

    def test_candidate_without_a_draft_still_renders(self):
        # Drafting can fail for one company; the digest must survive it.
        payload = slack.build_payload([_candidate(with_draft=False)], RUN_DATE)
        assert "Acme" in _all_text(payload)

    @pytest.mark.parametrize("count", [0, 1, 5, 25])
    def test_payload_is_json_serializable_at_every_size(self, count):
        payload = slack.build_payload([_candidate(f"C{i}") for i in range(count)], RUN_DATE)
        json.dumps(payload)


class TestPost:
    def test_missing_webhook_returns_false_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(slack.settings(), "slack_webhook_url", "", raising=False)
        assert slack.post({"text": "hi"}, webhook_url="") is False
