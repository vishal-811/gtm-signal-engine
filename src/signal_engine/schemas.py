"""All data shapes for the pipeline, in one place.

Models are split into two groups:

* **LLM output models** — passed to ``client.messages.parse()`` as the response
  schema. These deliberately avoid numeric bounds and string-length limits,
  because the structured-output schema subset does not support them
  (the SDK strips such constraints before sending, then re-validates locally,
  which turns a model's harmless out-of-range answer into a hard failure).
  Ranges are clamped in Python instead — see ``clamp_score``.
* **Internal models** — everything the pipeline passes between stages. These are
  free to use whatever validation we want.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

RoundStage = Literal[
    "pre-seed", "seed", "series-a", "series-b", "series-c", "later", "unknown"
]
Market = Literal["sf-bay-area", "nyc-metro", "bengaluru"]
OpeningsStatus = Literal["verified", "none_found", "unverified"]
AtsProvider = Literal["greenhouse", "lever", "ashby", "smartrecruiters"]

SCORE_MIN = 0.0
SCORE_MAX = 5.0


def clamp_score(value: float) -> float:
    """Constrain a model-supplied criterion score to the rubric's 0-5 range."""
    return max(SCORE_MIN, min(SCORE_MAX, value))


# ── Ingest ────────────────────────────────────────────────────────────────────


class Article(BaseModel):
    """One normalized item from an RSS feed."""

    title: str
    url: str
    source: str
    published_at: datetime | None = None
    summary: str = ""

    def dedupe_key(self) -> str:
        """Stable key for cross-feed deduplication of the same story."""
        from .textutil import normalize_title

        return normalize_title(self.title)


# ── Extraction (LLM output) ───────────────────────────────────────────────────


class FundingEvent(BaseModel):
    """A funding announcement extracted from an article.

    Emitted by the model. Kept flat and constraint-free for structured-output
    compatibility.
    """

    is_funding_announcement: bool = Field(
        description=(
            "True only if this article announces that a specific company raised "
            "a specific round of funding. False for market commentary, fund "
            "launches by VCs, M&A, IPOs, and roundup posts."
        )
    )
    company_name: str = Field(description="The company that raised the money.")
    company_domain: str | None = Field(
        default=None,
        description=(
            "Bare domain of the company website, e.g. 'acme.com'. No scheme, no "
            "'www.'. Null if the article does not state it and you cannot infer "
            "it with confidence."
        ),
    )
    round_stage: RoundStage = Field(description="Funding stage named in the article.")
    amount_usd: int | None = Field(
        default=None,
        description=(
            "Round size in US dollars, converted from other currencies if needed. "
            "Null if undisclosed."
        ),
    )
    announced_date: date | None = Field(
        default=None, description="Date the round was announced (not the article date)."
    )
    investors: list[str] = Field(
        default_factory=list,
        description="Named investors, lead investor first if identifiable.",
    )
    hq_city: str | None = Field(default=None, description="Company headquarters city.")
    hq_country: str | None = Field(
        default=None, description="Company headquarters country."
    )
    sector: str = Field(
        description="Short sector label, e.g. 'AI infrastructure', 'fintech', 'devtools'."
    )
    one_line_description: str = Field(description="What the company does, in one line.")
    source_url: str = Field(description="URL of the article this came from.")
    extraction_confidence: float = Field(
        description=(
            "0.0 to 1.0. How confident you are that the company name, stage, and "
            "amount are correct. Use a low value when the article is a passing "
            "mention rather than a dedicated announcement."
        )
    )


class ExtractionBatch(BaseModel):
    """Wrapper so one model call can process several articles at once."""

    events: list[FundingEvent]


# ── Enrichment (Apollo, optional) ─────────────────────────────────────────────


class Enrichment(BaseModel):
    """Firmographics from Apollo. All fields optional — Apollo may be disabled."""

    employee_count: int | None = None
    industry: str | None = None
    hq_city: str | None = None
    hq_country: str | None = None
    linkedin_url: str | None = None
    website_url: str | None = None
    technologies: list[str] = Field(default_factory=list)
    source: Literal["apollo"] = "apollo"


# ── Openings verification ─────────────────────────────────────────────────────


class JobPosting(BaseModel):
    title: str
    location: str | None = None
    url: str | None = None
    posted_at: datetime | None = None


class OpeningsResult(BaseModel):
    """Outcome of the ATS check for one company.

    ``unverified`` means we could not find a job board at all — the company is
    kept in the shortlist with a visible flag rather than silently dropped,
    because plenty of early-stage teams hire off a Notion page.
    """

    status: OpeningsStatus
    ats_provider: AtsProvider | None = None
    board_token: str | None = None
    board_url: str | None = None
    eng_role_count: int = 0
    total_role_count: int = 0
    sample_titles: list[str] = Field(default_factory=list)
    # Where the engineering roles actually are. This is the geography signal
    # the pipeline filters on: funding articles routinely omit a company's HQ,
    # but a job board always says where it is hiring — and where the roles are
    # is what actually matters for placing engineers.
    locations: list[str] = Field(default_factory=list)
    newest_post_date: datetime | None = None


# ── Scoring ───────────────────────────────────────────────────────────────────


class CriterionScore(BaseModel):
    """One rubric criterion, scored by the model."""

    id: str = Field(description="The criterion id, copied verbatim from the rubric.")
    score: float = Field(description="0 to 5, where 5 is the strongest possible fit.")
    reason: str = Field(description="One sentence justifying the score.")


class ScoreResult(BaseModel):
    """The model's scoring output. The composite total is NOT included here — it is
    computed in Python from these criteria and the rubric weights, so the
    arithmetic is deterministic and testable rather than model-generated."""

    criteria: list[CriterionScore]
    key_signal: str = Field(
        description="The single strongest reason to reach out to this company now."
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Reasons this company might be a poor use of outreach effort.",
    )


# ── Drafting ──────────────────────────────────────────────────────────────────


class Outreach(BaseModel):
    """A drafted cold email. Never sent by this codebase."""

    subject: str = Field(description="Email subject line, under 60 characters.")
    body: str = Field(description="Email body, 90 words maximum, plain text.")
    personalization_hook: str = Field(
        description="The specific fact from the input data that the email opens on."
    )


# ── Pipeline record ───────────────────────────────────────────────────────────


class Candidate(BaseModel):
    """A funding event as it accumulates state through the pipeline.

    Stages attach to this rather than returning new types, so a partially
    processed run is still a valid, inspectable object.
    """

    event: FundingEvent
    market: Market | None = None
    enrichment: Enrichment | None = None
    openings: OpeningsResult | None = None
    score: ScoreResult | None = None
    composite: float | None = None
    outreach: Outreach | None = None

    @property
    def key(self) -> str:
        """Identity for dedupe: domain if known, else normalized company name."""
        from .textutil import normalize_company

        return self.event.company_domain or normalize_company(self.event.company_name)


class RunStats(BaseModel):
    """One row of the `runs` sheet tab — the pipeline's health dashboard."""

    started_at: datetime
    finished_at: datetime | None = None
    articles_fetched: int = 0
    articles_after_dedupe: int = 0
    events_extracted: int = 0
    passed_filters: int = 0
    openings_verified: int = 0
    scored: int = 0
    posted: int = 0
    errors: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    estimated_cost_usd: float = 0.0
    dry_run: bool = True
