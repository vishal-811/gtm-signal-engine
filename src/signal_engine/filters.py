"""Post-extraction filtering: validity, recency, geography, dedupe.

Everything here is a pure function over already-extracted events, so the whole
stage is testable offline and costs nothing to re-run while tuning.

Order matters — the cheap, high-rejection filters run first so the expensive
downstream stages (ATS lookups, scoring) see the smallest possible set.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from .config import MarketConfig, geo_config, settings
from .schemas import Candidate, FundingEvent, Market
from .textutil import normalize_company

log = logging.getLogger(__name__)

# Below this, the model is telling us it was guessing.
MIN_EXTRACTION_CONFIDENCE = 0.6


@dataclass
class FilterReport:
    """Why candidates were dropped, so a thin day is explainable rather than
    mysterious. Surfaced by `run --dry-run` and logged to the `runs` tab."""

    input_count: int = 0
    kept: int = 0
    not_funding: int = 0
    low_confidence: int = 0
    too_old: int = 0
    wrong_geo: int = 0
    duplicate: int = 0
    recently_seen: int = 0
    rejected_examples: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.input_count} events → {self.kept} kept "
            f"(dropped: {self.not_funding} not-funding, "
            f"{self.low_confidence} low-confidence, {self.too_old} stale, "
            f"{self.wrong_geo} out-of-market, {self.duplicate} duplicate, "
            f"{self.recently_seen} recently-posted)"
        )

    def _note(self, event: FundingEvent, reason: str) -> None:
        if len(self.rejected_examples) < 25:
            self.rejected_examples.append(f"{event.company_name}: {reason}")


# ── Geography ─────────────────────────────────────────────────────────────────


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Whole-phrase matcher.

    Word boundaries are what stop "sf" matching "sfo-adjacent" and "york"
    matching "New York" — a plain substring check produces both false positives
    and false negatives here.
    """
    return re.compile(rf"(?<![a-z0-9]){re.escape(_flatten(phrase))}(?![a-z0-9])")


# Compiled once at import; the alias lists are static config.
_MARKET_PATTERNS: dict[str, list[re.Pattern[str]]] = {}
_EXCLUSION_PATTERNS: list[re.Pattern[str]] = []


def _ensure_patterns() -> None:
    global _EXCLUSION_PATTERNS
    if _MARKET_PATTERNS:
        return
    geo = geo_config()
    for market in geo.markets:
        _MARKET_PATTERNS[market.id] = [_phrase_pattern(a) for a in market.aliases]
    _EXCLUSION_PATTERNS = [_phrase_pattern(e) for e in geo.exclusions]


def _country_allows(market: MarketConfig, country: str | None) -> bool:
    """An unknown country never vetoes a match.

    Feeds often omit the country. Rejecting on absence would throw away good
    candidates, and the city alias is already a strong signal on its own.
    """
    if not market.countries:
        return True
    if not country:
        return True
    normalized = country.strip().lower().rstrip(".")
    return any(normalized == c.strip().lower().rstrip(".") for c in market.countries)


_SEPARATORS = re.compile(r"[\s/_|,;·•‐-―-]+")


def _flatten(text: str) -> str:
    """Collapse separator variation so one alias matches every spelling.

    Boards write "Remote - United States", "Remote | US", "Remote/US". Without
    this, the alias "remote united states" matches none of them — the same
    class of bug already fixed for job titles.
    """
    return _SEPARATORS.sub(" ", text.lower()).strip()


def match_market(city: str | None, country: str | None = None) -> Market | None:
    """Resolve a location string to a configured market, or None."""
    _ensure_patterns()
    if not city:
        return None

    haystack = _flatten(city)
    if any(p.search(haystack) for p in _EXCLUSION_PATTERNS):
        return None

    geo = geo_config()
    best: tuple[int, int, Market] | None = None
    for market in geo.markets:
        if not _country_allows(market, country):
            continue
        for alias, pattern in zip(market.aliases, _MARKET_PATTERNS[market.id]):
            if pattern.search(haystack):
                # Rank by tier first, then alias length. Tier keeps a hub
                # ahead of the broad region containing it; length keeps
                # "south san francisco" from losing to a shorter alias.
                score = (-market.tier, len(alias))
                if best is None or score > (best[0], best[1]):
                    best = (score[0], score[1], market.id)
    return best[2] if best else None


# ── Individual predicates ─────────────────────────────────────────────────────


def is_valid_funding(event: FundingEvent) -> bool:
    return event.is_funding_announcement


def is_confident(event: FundingEvent, minimum: float = MIN_EXTRACTION_CONFIDENCE) -> bool:
    return event.extraction_confidence >= minimum


def is_recent(event: FundingEvent, max_age_days: int | None = None) -> bool:
    """Recency check on the announcement date.

    Events with no announcement date are **kept**: the article itself already
    passed the ingest recency window, so a missing field is a gap in the
    extraction, not evidence the round is old.
    """
    if event.announced_date is None:
        return True
    days = max_age_days if max_age_days is not None else settings().max_event_age_days
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    return event.announced_date >= cutoff


# ── Pipeline ──────────────────────────────────────────────────────────────────


def apply(
    events: list[FundingEvent],
    *,
    seen_keys: dict[str, date] | None = None,
    dedupe_window_days: int | None = None,
) -> tuple[list[Candidate], FilterReport]:
    """Run every filter in order and return surviving candidates.

    ``seen_keys`` maps a company key to the date it was last posted, read from
    the `seen` sheet tab. Passing None skips suppression entirely, which is what
    dry runs do so you can see the full unfiltered picture.
    """
    report = FilterReport(input_count=len(events))
    window = (
        dedupe_window_days
        if dedupe_window_days is not None
        else settings().dedupe_window_days
    )
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=window)

    candidates: list[Candidate] = []
    seen_this_run: set[str] = set()

    for event in events:
        if not is_valid_funding(event):
            report.not_funding += 1
            continue

        if not is_confident(event):
            report.low_confidence += 1
            report._note(event, f"confidence {event.extraction_confidence:.2f}")
            continue

        if not is_recent(event):
            report.too_old += 1
            report._note(event, f"announced {event.announced_date}")
            continue

        # Geography is decided later, in apply_geo(), once the openings check
        # has supplied job locations. Filtering on the article's stated HQ here
        # discarded every candidate: funding coverage almost never prints it.
        candidate = Candidate(event=event)
        key = candidate.key

        # Two companies can appear twice in one run when different outlets name
        # them slightly differently and the title dedupe missed it.
        if key in seen_this_run:
            report.duplicate += 1
            continue
        seen_this_run.add(key)

        if seen_keys is not None:
            last_posted = seen_keys.get(key)
            if last_posted is not None and last_posted >= cutoff:
                report.recently_seen += 1
                report._note(event, f"posted {last_posted}, within {window}d window")
                continue

        candidates.append(candidate)

    report.kept = len(candidates)
    log.info("filters: %s", report.summary())
    return candidates, report


def market_from_locations(locations: list[str]) -> Market | None:
    """Resolve a market from a set of job-posting locations.

    A single posting can name several cities ("Hybrid - San Francisco, New
    York City, London"), so every location string is checked and the first
    match wins. Country is deliberately not passed: board locations rarely
    include one, and requiring it would reject "San Francisco, CA".
    """
    for location in locations:
        market = match_market(location)
        if market:
            return market
    return None


@dataclass
class GeoReport:
    """Why candidates were dropped at the geography stage.

    Kept separate from :class:`FilterReport` because this now runs after the
    openings check, and because "we could not determine a location" is a very
    different outcome from "this company is in Berlin" — conflating them hides
    whether the pipeline is working or the data is thin.
    """

    input_count: int = 0
    kept: int = 0
    matched_on_jobs: int = 0
    matched_on_article: int = 0
    out_of_market: int = 0
    location_unknown: int = 0
    unknown_examples: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.input_count} candidates → {self.kept} in-market "
            f"({self.matched_on_jobs} via job locations, "
            f"{self.matched_on_article} via article HQ) · "
            f"{self.out_of_market} elsewhere · "
            f"{self.location_unknown} location unknown"
        )


def apply_geo(candidates: list[Candidate]) -> tuple[list[Candidate], GeoReport]:
    """Resolve each candidate's market and keep only the in-market ones.

    Runs *after* the openings check, because that is where the location data
    actually is. Funding articles routinely omit a company's headquarters —
    measured across a full run, every single extracted round had ``hq_city``
    of None — so filtering on the article alone discarded everything.

    Job locations are preferred over the article's stated HQ: they are more
    reliable, and where the roles are is what matters for placing engineers.
    """
    report = GeoReport(input_count=len(candidates))
    kept: list[Candidate] = []

    for candidate in candidates:
        openings = candidate.openings
        locations = openings.locations if openings else []

        market = market_from_locations(locations)
        if market:
            report.matched_on_jobs += 1
        else:
            market = match_market(candidate.event.hq_city, candidate.event.hq_country)
            if market:
                report.matched_on_article += 1

        if market:
            candidate.market = market
            kept.append(candidate)
            continue

        # Distinguish "definitely elsewhere" from "no signal at all". The first
        # is a correct rejection; a lot of the second means the openings stage
        # is failing to find boards, which is a different problem.
        if locations or candidate.event.hq_city:
            report.out_of_market += 1
        else:
            report.location_unknown += 1
            if len(report.unknown_examples) < 15:
                report.unknown_examples.append(candidate.event.company_name)

    report.kept = len(kept)
    log.info("geo: %s", report.summary())
    return kept, report


def normalized_seen_keys(rows: list[tuple[str, date]]) -> dict[str, date]:
    """Build the suppression map from raw `seen` tab rows.

    Keys are normalized the same way :attr:`Candidate.key` normalizes them, so a
    company logged under a slightly different spelling still suppresses.
    """
    result: dict[str, date] = {}
    for raw_key, seen_on in rows:
        key = _as_key(raw_key)
        existing = result.get(key)
        if existing is None or seen_on > existing:
            result[key] = seen_on
    return result


def _as_key(raw: str) -> str:
    """Interpret a `seen` tab value as either a domain or a company name.

    A dot alone is not enough to identify a domain — "Acme Inc." has one. A
    domain has a dot and no whitespace; anything else is a name and gets the
    same normalization :attr:`Candidate.key` applies, so the two agree.
    """
    stripped = raw.strip()
    if "." in stripped and not any(c.isspace() for c in stripped):
        return stripped.lower()
    return normalize_company(stripped)


def reset_pattern_cache() -> None:
    """Clear compiled geo patterns. Used by tests that swap in a temp config."""
    _MARKET_PATTERNS.clear()
    global _EXCLUSION_PATTERNS
    _EXCLUSION_PATTERNS = []
