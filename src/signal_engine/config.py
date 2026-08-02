"""Environment and YAML configuration, validated at load time.

Anything tunable lives either in the environment (secrets, flags) or in a YAML
file under ``config/`` (feeds, geography, engineering titles, the rubric). Code
should never hardcode a feed URL, a city name, or a rubric weight.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RUBRIC_PATH = PROJECT_ROOT / "rubric.yaml"


class Settings(BaseSettings):
    """Secrets and runtime flags, read from the environment or a local .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    # Configurable because OpenAI renames and retires models frequently.
    # `verify-credentials` prints the ids this key can actually reach.
    openai_model: str = "gpt-5"
    # Per-model rates, used only to estimate the figure in the `runs` tab.
    # Left at zero the pipeline reports tokens and no dollar amount, which
    # beats reporting a confidently wrong one for a model that was swapped.
    openai_input_cost_per_mtok: float = 0.0
    openai_output_cost_per_mtok: float = 0.0

    google_sheet_id: str = ""
    google_service_account_file: str = ""
    google_service_account_json: str = ""

    slack_webhook_url: str = ""

    apollo_enabled: bool = False
    apollo_api_key: str = ""

    sender_name: str = ""
    sender_title: str = ""
    sender_email: str = ""
    sender_company: str = "Hire100x"

    dedupe_window_days: int = 30
    max_event_age_days: int = 14
    max_article_age_hours: int = 36

    def google_credentials_info(self) -> dict[str, Any] | None:
        """Return service-account credentials as a dict, from either source.

        GitHub Actions can't mount a file, so the JSON body is accepted inline.
        A local file path is the friendlier option for development.
        """
        if self.google_service_account_json.strip():
            return json.loads(self.google_service_account_json)
        if self.google_service_account_file.strip():
            path = Path(self.google_service_account_file)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                return json.loads(path.read_text())
        return None


@cache
def settings() -> Settings:
    return Settings()


# ── YAML-backed config ────────────────────────────────────────────────────────


class Feed(BaseModel):
    name: str
    url: str
    enabled: bool = True


class FeedsConfig(BaseModel):
    feeds: list[Feed]

    @property
    def active(self) -> list[Feed]:
        return [f for f in self.feeds if f.enabled]


class MarketConfig(BaseModel):
    """One target market and every string that should resolve to it."""

    id: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)


class GeoConfig(BaseModel):
    markets: list[MarketConfig]
    # Substrings that must NOT match, checked before aliases. Guards against
    # e.g. "South San Francisco" or "New York, Lincolnshire".
    exclusions: list[str] = Field(default_factory=list)


class EngTitlesConfig(BaseModel):
    include: list[str]
    exclude: list[str] = Field(default_factory=list)


class Criterion(BaseModel):
    id: str
    weight: float
    question: str
    anchors: dict[str, str] = Field(default_factory=dict)


class Rubric(BaseModel):
    threshold: float
    criteria: list[Criterion]

    @model_validator(mode="after")
    def _check_weights(self) -> Rubric:
        if not self.criteria:
            raise ValueError("rubric.yaml must define at least one criterion")
        total = sum(c.weight for c in self.criteria)
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"rubric.yaml criterion weights must sum to 1.0, got {total:.3f}. "
                "Adjust the `weight` fields so they add up."
            )
        ids = [c.id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError(f"rubric.yaml has duplicate criterion ids: {ids}")
        return self


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing config file: {path}. Copy it from the repo template or "
            "re-run setup."
        )
    return yaml.safe_load(path.read_text()) or {}


@cache
def feeds_config() -> FeedsConfig:
    return FeedsConfig(**_load_yaml(CONFIG_DIR / "feeds.yaml"))


@cache
def geo_config() -> GeoConfig:
    return GeoConfig(**_load_yaml(CONFIG_DIR / "geo.yaml"))


@cache
def eng_titles_config() -> EngTitlesConfig:
    return EngTitlesConfig(**_load_yaml(CONFIG_DIR / "eng_titles.yaml"))


@cache
def rubric() -> Rubric:
    return Rubric(**_load_yaml(RUBRIC_PATH))


@cache
def prompt(name: str) -> str:
    """Load a prompt template from prompts/<name>.md."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text()


def reset_caches() -> None:
    """Clear all config caches. Used by tests that write temp config files."""
    for fn in (
        settings,
        feeds_config,
        geo_config,
        eng_titles_config,
        rubric,
        prompt,
    ):
        fn.cache_clear()


def is_ci() -> bool:
    return os.getenv("CI", "").lower() in {"1", "true", "yes"}
