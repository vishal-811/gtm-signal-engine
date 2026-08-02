"""The composite is computed in Python precisely so it can be tested. These
cover the arithmetic and the two ways a model can hand back a malformed score
set — an out-of-range value, and a missing criterion."""

from __future__ import annotations

import pytest

from signal_engine import score
from signal_engine.config import Criterion, Rubric
from signal_engine.schemas import CriterionScore, ScoreResult

RUBRIC = Rubric(
    threshold=3.0,
    criteria=[
        Criterion(id="a", weight=0.5, question="A?", anchors={"5": "best", "0": "worst"}),
        Criterion(id="b", weight=0.3, question="B?"),
        Criterion(id="c", weight=0.2, question="C?"),
    ],
)


def _result(**scores: float) -> ScoreResult:
    return ScoreResult(
        criteria=[
            CriterionScore(id=cid, score=value, reason="because")
            for cid, value in scores.items()
        ],
        key_signal="signal",
    )


class TestComposite:
    def test_weighted_sum(self):
        # 4*0.5 + 3*0.3 + 5*0.2 = 2.0 + 0.9 + 1.0
        assert score.composite(_result(a=4, b=3, c=5), RUBRIC) == pytest.approx(3.9)

    def test_all_zeros_and_all_fives_hit_the_range_ends(self):
        assert score.composite(_result(a=0, b=0, c=0), RUBRIC) == 0.0
        assert score.composite(_result(a=5, b=5, c=5), RUBRIC) == pytest.approx(5.0)

    def test_out_of_range_scores_are_clamped(self):
        # The LLM schema carries no numeric bounds (the structured-output subset
        # does not support them), so a stray 9 must be clamped here rather than
        # blowing past the 5.0 ceiling.
        assert score.composite(_result(a=9, b=5, c=5), RUBRIC) == pytest.approx(5.0)
        assert score.composite(_result(a=-3, b=0, c=0), RUBRIC) == 0.0

    def test_missing_criterion_counts_as_zero_rather_than_rescaling(self):
        # Dropping it from the denominator would let a model raise its own score
        # by simply omitting the criterion it scores worst on.
        assert score.composite(_result(a=5, b=5), RUBRIC) == pytest.approx(4.0)

    def test_unknown_criteria_from_the_model_are_ignored(self):
        result = _result(a=5, b=5, c=5)
        result.criteria.append(CriterionScore(id="invented", score=5, reason="x"))
        assert score.composite(result, RUBRIC) == pytest.approx(5.0)

    def test_empty_criteria_scores_zero(self):
        assert score.composite(ScoreResult(criteria=[], key_signal=""), RUBRIC) == 0.0


class TestThreshold:
    def _candidate(self, composite: float | None):
        from signal_engine.schemas import Candidate, FundingEvent

        event = FundingEvent(
            is_funding_announcement=True,
            company_name="Acme",
            round_stage="seed",
            sector="devtools",
            one_line_description="x",
            source_url="https://x.com",
            extraction_confidence=0.9,
        )
        candidate = Candidate(event=event)
        candidate.composite = composite
        return candidate

    def test_above_threshold(self):
        assert score.above_threshold(self._candidate(3.5), RUBRIC)

    def test_exactly_at_threshold_does_not_pass(self):
        # Documented as "> threshold", not ">=".
        assert not score.above_threshold(self._candidate(3.0), RUBRIC)

    def test_below_threshold(self):
        assert not score.above_threshold(self._candidate(2.9), RUBRIC)

    def test_unscored_candidate_never_passes(self):
        # A scoring failure must not silently promote a company.
        assert not score.above_threshold(self._candidate(None), RUBRIC)


class TestRenderRubric:
    def test_includes_every_criterion_with_its_weight(self):
        rendered = score.render_rubric(RUBRIC)
        for criterion in RUBRIC.criteria:
            assert criterion.id in rendered
        assert "50%" in rendered

    def test_includes_anchor_text(self):
        assert "best" in score.render_rubric(RUBRIC)

    def test_is_deterministic_so_the_prompt_cache_holds(self):
        # Any run-to-run variation in this string silently disables prompt
        # caching for the most expensive stage in the pipeline.
        assert score.render_rubric(RUBRIC) == score.render_rubric(RUBRIC)


class TestRenderCandidate:
    def _candidate_with_openings(self, status: str, **kwargs):
        from signal_engine.schemas import Candidate, FundingEvent, OpeningsResult

        event = FundingEvent(
            is_funding_announcement=True,
            company_name="Acme",
            company_domain="acme.com",
            round_stage="series-a",
            amount_usd=20_000_000,
            sector="devtools",
            one_line_description="Dev tooling.",
            hq_city="San Francisco",
            hq_country="United States",
            source_url="https://x.com/a",
            extraction_confidence=0.9,
        )
        candidate = Candidate(event=event, market="sf-bay-area")
        candidate.openings = OpeningsResult(status=status, **kwargs)
        return candidate

    def test_unverified_openings_are_described_as_weak_evidence(self):
        # The model must not read a missing board as neutral.
        rendered = score.render_candidate(self._candidate_with_openings("unverified"))
        assert "UNVERIFIED" in rendered
        assert "weaker evidence" in rendered

    def test_zero_engineering_roles_is_stated_explicitly(self):
        rendered = score.render_candidate(
            self._candidate_with_openings(
                "none_found", ats_provider="greenhouse", total_role_count=7
            )
        )
        assert "ZERO ENGINEERING ROLES" in rendered
        assert "7" in rendered

    def test_verified_openings_include_counts_and_titles(self):
        rendered = score.render_candidate(
            self._candidate_with_openings(
                "verified",
                ats_provider="lever",
                eng_role_count=6,
                total_role_count=20,
                sample_titles=["Backend Engineer", "SRE"],
            )
        )
        assert "VERIFIED" in rendered
        assert "6" in rendered
        assert "Backend Engineer" in rendered

    def test_includes_the_funding_facts(self):
        rendered = score.render_candidate(self._candidate_with_openings("unverified"))
        assert "$20,000,000" in rendered
        assert "series-a" in rendered
        assert "San Francisco" in rendered


class TestGeographyBlock:
    """The geo filter matches on job-board locations, so the scorer has to see
    the same evidence. Without it the model reads an unstated HQ and scores
    geo_match 0 for a company the pipeline just verified is hiring in SF —
    observed live, costing 0.75 composite points."""

    def _candidate(self, market=None, locations=None):
        from signal_engine.schemas import Candidate, FundingEvent, OpeningsResult

        event = FundingEvent(
            is_funding_announcement=True,
            company_name="Simile",
            round_stage="series-b",
            sector="behavioral AI",
            one_line_description="x",
            hq_city=None,
            source_url="https://x.com/a",
            extraction_confidence=0.9,
        )
        c = Candidate(event=event, market=market)
        c.openings = OpeningsResult(
            status="verified", eng_role_count=7, locations=locations or []
        )
        return c

    def test_states_the_market_as_confirmed(self):
        rendered = score.render_candidate(
            self._candidate(market="sf-bay-area", locations=["Hybrid - San Francisco"])
        )
        assert "CONFIRMED target market: sf-bay-area" in rendered

    def test_shows_the_job_locations_as_the_evidence(self):
        rendered = score.render_candidate(
            self._candidate(market="sf-bay-area", locations=["Hybrid - San Francisco"])
        )
        assert "Hybrid - San Francisco" in rendered

    def test_tells_the_model_not_to_penalise_an_unstated_hq(self):
        rendered = score.render_candidate(
            self._candidate(market="sf-bay-area", locations=["Remote - US"])
        )
        assert "Do not mark this down for an unstated HQ" in rendered

    def test_an_unstated_hq_is_labelled_as_such_not_as_unknown_location(self):
        rendered = score.render_candidate(self._candidate(market="nyc-metro"))
        assert "not stated" in rendered

    def test_no_market_is_reported_honestly(self):
        rendered = score.render_candidate(self._candidate(market=None))
        assert "No target market could be confirmed" in rendered
