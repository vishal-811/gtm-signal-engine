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

    def test_geo_yaml_covers_all_three_target_markets(self):
        ids = {m.id for m in config.geo_config().markets}
        assert ids == {"sf-bay-area", "nyc-metro", "bengaluru"}

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

    @pytest.mark.parametrize("name", ["extract", "score", "draft"])
    def test_prompts_clear_the_512_token_cache_minimum(self, name):
        # Claude Opus 5 will not cache a prefix under 512 tokens, and does so
        # silently. At ~4 chars/token, 2400 chars is a safe floor. Falling
        # under it would quietly multiply the bill for the whole run.
        assert len(config.prompt(name)) > 2400, (
            f"prompts/{name}.md is short enough to fall under the prompt-cache "
            "minimum; caching would silently stop working"
        )

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
