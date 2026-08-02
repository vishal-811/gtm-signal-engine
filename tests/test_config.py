"""The rubric is user-edited YAML. A weight typo must fail loudly at startup,
not silently skew every score for a week before anyone notices."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from signal_engine import config
from signal_engine.config import Criterion, GeoConfig, Rubric


def _criterion(cid: str, weight: float) -> Criterion:
    return Criterion(id=cid, weight=weight, question="q?")


class TestRubricValidation:
    def test_accepts_weights_summing_to_one(self):
        rubric = Rubric(
            threshold=3.0,
            criteria=[_criterion("a", 0.6), _criterion("b", 0.4)],
        )
        assert len(rubric.criteria) == 2

    def test_rejects_weights_that_do_not_sum_to_one(self):
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            Rubric(threshold=3.0, criteria=[_criterion("a", 0.6), _criterion("b", 0.3)])

    def test_tolerates_floating_point_dust(self):
        # 0.3 + 0.3 + 0.4 does not sum to exactly 1.0 in binary floating point.
        Rubric(
            threshold=3.0,
            criteria=[
                _criterion("a", 0.3),
                _criterion("b", 0.3),
                _criterion("c", 0.4),
            ],
        )

    def test_rejects_duplicate_criterion_ids(self):
        with pytest.raises(ValidationError, match="duplicate criterion ids"):
            Rubric(
                threshold=3.0,
                criteria=[_criterion("same", 0.5), _criterion("same", 0.5)],
            )

    def test_rejects_empty_criteria(self):
        with pytest.raises(ValidationError, match="at least one criterion"):
            Rubric(threshold=3.0, criteria=[])


class TestShippedConfigFiles:
    """The files in the repo must actually load — a broken default config is a
    setup failure for whoever clones this next."""

    def test_rubric_yaml_loads_and_validates(self):
        rubric = config.rubric()
        assert 0 < rubric.threshold <= 5
        assert rubric.criteria

    def test_every_criterion_has_calibration_anchors(self):
        # Anchor-less criteria produce noisy scores; the rubric docs promise
        # at least the 5 / 3 / 0 points are pinned.
        for criterion in config.rubric().criteria:
            assert len(criterion.anchors) >= 3, f"{criterion.id} needs more anchors"

    def test_feeds_yaml_loads_with_at_least_one_enabled_feed(self):
        assert config.feeds_config().active

    def test_geo_yaml_keeps_the_three_network_hubs_at_tier_1(self):
        markets = {m.id: m for m in config.geo_config().markets}
        for hub in ("sf-bay-area", "nyc-metro", "bengaluru"):
            assert markets[hub].tier == 1, f"{hub} must outrank broad regions"

    def test_geo_yaml_covers_the_wider_regions(self):
        ids = {m.id for m in config.geo_config().markets}
        assert {"us-other", "india-other", "uk"} <= ids

    def test_every_market_id_has_a_rubric_anchor_or_is_deliberate(self):
        # A market with no matching anchor gets scored arbitrarily by the model.
        anchors = " ".join(
            " ".join(c.anchors.values())
            for c in config.rubric().criteria if c.id == "geo_match"
        )
        for market in config.geo_config().markets:
            assert market.id in anchors, (
                f"market {market.id!r} is not named in the geo_match anchors, "
                "so the model has no guidance on how to score it"
            )

    def test_eng_titles_yaml_loads(self):
        titles = config.eng_titles_config()
        assert "engineer" in titles.include
        assert "sales engineer" in titles.exclude


class TestPrompts:
    """Prompts are loaded by filename at runtime, so a rename breaks the
    pipeline at 02:30 UTC rather than in CI unless something checks."""

    @pytest.mark.parametrize("name", ["extract", "score", "draft"])
    def test_every_stage_prompt_exists(self, name):
        assert config.prompt(name).strip()

    # OpenAI caches automatically only at or above 1024 prompt tokens, and a
    # shorter prompt simply never caches — silently, for every call in the run.
    # At ~4 chars/token that is ~4096 characters.
    CACHE_MIN_CHARS = 4096

    @staticmethod
    def _system_prompt(stage: str) -> str:
        """The string actually sent as the system message.

        Not the same as the raw .md file for two stages: `score` appends the
        rendered rubric and `draft` appends the sender block, both at runtime.
        Measuring the file alone would test the wrong thing.
        """
        from signal_engine import draft, score

        return {
            "extract": lambda: config.prompt("extract"),
            "score": score.system_prompt,
            "draft": draft.system_prompt,
        }[stage]()

    @pytest.mark.parametrize("stage", ["extract", "score"])
    def test_high_volume_prompts_clear_the_1024_token_cache_minimum(self, stage):
        # extract and score are the volume drivers, so caching pays for itself
        # there. `extract` currently clears the bar by only ~1%, which is
        # exactly why this test exists: trimming a few lines from that prompt
        # would silently disable caching for the most expensive stage.
        size = len(self._system_prompt(stage))
        assert size > self.CACHE_MIN_CHARS, (
            f"the {stage} system prompt is {size} chars, under the "
            f"~{self.CACHE_MIN_CHARS} needed for OpenAI's 1024-token caching "
            "threshold. Caching would stop working with no error."
        )

    def test_draft_prompt_is_known_to_be_under_the_cache_threshold(self):
        # Documented rather than fixed. The draft stage only runs for companies
        # that already cleared the score threshold (single digits per day), so
        # the lost caching is worth a few cents a month. Padding the prompt to
        # win a cache hit would be cargo-culting. If this ever grows past the
        # threshold the assertion flips and the exemption can be deleted.
        assert len(self._system_prompt("draft")) < self.CACHE_MIN_CHARS

    def test_missing_prompt_raises_a_useful_error(self):
        with pytest.raises(FileNotFoundError, match="Missing prompt file"):
            config.prompt("does-not-exist")

    def test_draft_prompt_forbids_fabrication(self):
        # The single most important rule in the draft prompt: a made-up detail
        # in a cold email is obvious to the recipient and unrecoverable.
        assert "Never state a fact that is not in the input" in config.prompt("draft")

    def test_extract_prompt_warns_against_guessing_domains(self):
        # A wrong domain sends the openings check to another company entirely.
        assert "Do not guess" in config.prompt("extract")


class TestGeoConfig:
    def test_markets_may_omit_country_restrictions(self):
        geo = GeoConfig(markets=[{"id": "x", "label": "X", "aliases": ["somewhere"]}])
        assert geo.markets[0].countries == []
