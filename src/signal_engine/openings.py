"""Live engineering-openings verification via public ATS job boards.

This is the load-bearing hiring signal in the whole pipeline. A funding round
says a company *has money*; an open backend role posted last Tuesday says they
are *spending it on engineers right now*.

Three providers are supported, each verified against a live board:

===========  ==========================================================
Greenhouse   ``boards-api.greenhouse.io/v1/boards/{token}/jobs``
Lever        ``api.lever.co/v0/postings/{token}?mode=json``
Ashby        ``api.ashbyhq.com/posting-api/job-board/{token}``
===========  ==========================================================

All three are public and unauthenticated. Workable is deliberately absent: its
widget endpoint returned 404 for every token tried *and* for a control, so its
response shape could not be verified. Adding an unverified client would produce
silent ``unverified`` results that look like "no board" rather than "broken
code". Add it when a real Workable board is available to test against.

A company where no board is found is reported ``unverified``, never dropped —
plenty of seed-stage teams hire off a Notion page, and a missing board is
absence of evidence rather than evidence of absence.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import httpx
from dateutil import parser as dateparser

from .config import eng_titles_config
from .schemas import AtsProvider, JobPosting, OpeningsResult
from .textutil import company_slug

log = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; SignalEngine/0.1)"
_TIMEOUT = httpx.Timeout(20.0, connect=8.0)
_MAX_SAMPLE_TITLES = 6

# Careers pages that are worth checking for an embedded board link, in the
# order companies most commonly use.
_CAREERS_PATHS = ("/careers", "/jobs", "/company/careers", "/about/careers", "/careers/")

# Board links as they appear in careers-page HTML. `job-boards.greenhouse.io`
# is Greenhouse's newer host and is easy to miss.
_BOARD_LINK_PATTERNS: list[tuple[AtsProvider, re.Pattern[str]]] = [
    ("greenhouse", re.compile(r"(?:job-)?boards\.greenhouse\.io/(?:embed/job_board\?for=)?([a-z0-9_-]+)", re.I)),
    ("greenhouse", re.compile(r"boards\.greenhouse\.io/embed/job_board/js\?for=([a-z0-9_-]+)", re.I)),
    ("lever", re.compile(r"jobs\.lever\.co/([a-z0-9_-]+)", re.I)),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([a-z0-9_-]+)", re.I)),
]

# Path segments that are not board tokens even though they match the pattern.
_NOT_TOKENS = {"embed", "js", "api", "static", "assets", "www"}


def _parse_dt(value: Any) -> datetime | None:
    """Parse the several date formats the three providers use.

    Greenhouse sends ISO-8601 with an offset, Ashby ISO-8601 UTC, and Lever
    epoch milliseconds. A bad date must never sink an otherwise valid posting,
    so failures return None rather than raising.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # Lever: epoch milliseconds.
        try:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = dateparser.parse(str(value))
    except (ValueError, TypeError, OverflowError):
        return None
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ── Engineering title classification ──────────────────────────────────────────

_include_patterns: list[re.Pattern[str]] = []
_exclude_patterns: list[re.Pattern[str]] = []


_SEPARATORS = re.compile(r"[\s /_‐-―-]+")


def normalize_title_text(title: str) -> str:
    """Flatten separator variation so one config phrase matches every spelling.

    Job boards are inconsistent in ways that silently defeat literal matching:
    "Forward-Deployed Engineer" (hyphen) vs the config's "forward deployed",
    and Vercel's board ships titles containing a non-breaking space. Collapsing
    hyphens, slashes, underscores, en/em dashes, and exotic whitespace to a
    single plain space makes the comparison stable.
    """
    return _SEPARATORS.sub(" ", title.lower()).strip()


def _phrase(p: str) -> re.Pattern[str]:
    """Whole-phrase matcher that tolerates ``-s`` and ``-ing`` on the last word.

    Without the suffix allowance, the exclusion "support engineer" fails to
    match "Support Engineering Manager" — the boundary check sees the ``i`` of
    "-ing" and rejects. That let a whole class of non-IC titles through. One
    rule here beats maintaining both spellings of every phrase in the config.
    """
    normalized = normalize_title_text(p)
    return re.compile(rf"(?<![a-z0-9]){re.escape(normalized)}(?:s|ing)?(?![a-z0-9])")


def _ensure_title_patterns() -> None:
    if _include_patterns:
        return
    cfg = eng_titles_config()
    _include_patterns.extend(_phrase(p) for p in cfg.include)
    _exclude_patterns.extend(_phrase(p) for p in cfg.exclude)


def reset_title_cache() -> None:
    """Clear compiled title patterns. Used by tests that swap the config."""
    _include_patterns.clear()
    _exclude_patterns.clear()


def is_engineering_title(title: str) -> bool:
    """True when a job title is an engineering role Hire100x could fill.

    Exclusions are checked first and win outright: "Sales Engineer" contains
    "engineer" but is not a role this network places, and counting it would
    inflate the hiring-intent score on a company that is only growing revenue
    headcount.
    """
    _ensure_title_patterns()
    normalized = normalize_title_text(title)
    if any(p.search(normalized) for p in _exclude_patterns):
        return False
    return any(p.search(normalized) for p in _include_patterns)


# ── Provider clients ──────────────────────────────────────────────────────────


def _get_json(url: str) -> Any | None:
    try:
        response = httpx.get(
            url,
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        )
    except Exception as exc:  # noqa: BLE001 - a dead board is not a run failure
        log.debug("board fetch failed for %s: %s", url, exc)
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        # Some boards answer 200 with an HTML error page.
        return None


def fetch_greenhouse(token: str) -> list[JobPosting] | None:
    data = _get_json(
        f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=false"
    )
    if not isinstance(data, dict) or "jobs" not in data:
        return None
    postings = []
    for job in data["jobs"]:
        location = job.get("location") or {}
        postings.append(
            JobPosting(
                title=job.get("title", ""),
                location=location.get("name") if isinstance(location, dict) else None,
                url=job.get("absolute_url"),
                # `first_published` is when the role opened; `updated_at` moves
                # on every edit and would make a stale role look fresh.
                posted_at=_parse_dt(job.get("first_published") or job.get("updated_at")),
            )
        )
    return postings


def fetch_lever(token: str) -> list[JobPosting] | None:
    data = _get_json(f"https://api.lever.co/v0/postings/{token}?mode=json")
    if not isinstance(data, list):
        return None
    postings = []
    for job in data:
        categories = job.get("categories") or {}
        postings.append(
            JobPosting(
                title=job.get("text", ""),
                location=categories.get("location"),
                url=job.get("hostedUrl"),
                posted_at=_parse_dt(job.get("createdAt")),
            )
        )
    return postings


def fetch_ashby(token: str) -> list[JobPosting] | None:
    data = _get_json(f"https://api.ashbyhq.com/posting-api/job-board/{token}")
    if not isinstance(data, dict) or "jobs" not in data:
        return None
    postings = []
    for job in data["jobs"]:
        # Ashby returns unlisted drafts alongside live roles.
        if job.get("isListed") is False:
            continue
        postings.append(
            JobPosting(
                title=job.get("title", ""),
                location=job.get("location"),
                url=job.get("jobUrl"),
                posted_at=_parse_dt(job.get("publishedAt")),
            )
        )
    return postings


PROVIDERS: dict[AtsProvider, Callable[[str], list[JobPosting] | None]] = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
    "ashby": fetch_ashby,
}

BOARD_URLS: dict[AtsProvider, str] = {
    "greenhouse": "https://boards.greenhouse.io/{token}",
    "lever": "https://jobs.lever.co/{token}",
    "ashby": "https://jobs.ashbyhq.com/{token}",
}


# ── Board discovery ───────────────────────────────────────────────────────────


def discover_from_careers_page(domain: str) -> tuple[AtsProvider, str] | None:
    """Look for an embedded board link on the company's own careers page.

    Far more reliable than guessing a token, because it finds the token the
    company actually uses — which is frequently not their name (acquisitions,
    rebrands, legal entities).
    """
    for path in _CAREERS_PATHS:
        url = f"https://{domain}{path}"
        try:
            response = httpx.get(
                url,
                timeout=_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            )
        except Exception:  # noqa: BLE001
            continue
        if response.status_code != 200:
            continue

        html = response.text
        for provider, pattern in _BOARD_LINK_PATTERNS:
            for match in pattern.finditer(html):
                token = match.group(1).lower()
                if token and token not in _NOT_TOKENS:
                    log.debug("found %s board '%s' on %s", provider, token, url)
                    return provider, token
    return None


def discover_by_slug(company_name: str, domain: str | None) -> tuple[AtsProvider, str] | None:
    """Try plausible tokens against each provider until a board answers.

    Ordered cheapest-signal-first: the company's own domain label is a better
    guess than a squashed display name.
    """
    candidates: list[str] = []
    if domain:
        candidates.append(domain.split(".")[0].lower())
    slug = company_slug(company_name)
    if slug and slug not in candidates:
        candidates.append(slug)
    hyphenated = re.sub(r"[^a-z0-9]+", "-", company_name.lower()).strip("-")
    if hyphenated and hyphenated not in candidates:
        candidates.append(hyphenated)

    for token in candidates:
        if len(token) < 3:
            continue
        for provider, fetch in PROVIDERS.items():
            postings = fetch(token)
            # An empty list means the board exists but has no roles — a real
            # and meaningful answer. None means no such board.
            if postings is not None:
                log.debug("slug guess hit: %s/%s", provider, token)
                return provider, token
    return None


def discover(
    company_name: str,
    domain: str | None,
    cache: dict[str, tuple[AtsProvider, str]] | None = None,
) -> tuple[AtsProvider, str] | None:
    """Resolve a company to an (provider, token) pair.

    Checks the persistent cache first — board tokens essentially never change,
    and re-resolving costs several HTTP round-trips per company per day.
    """
    if cache and domain and domain in cache:
        return cache[domain]

    if domain:
        found = discover_from_careers_page(domain)
        if found:
            return found

    return discover_by_slug(company_name, domain)


# ── Public entry point ────────────────────────────────────────────────────────


def check(
    company_name: str,
    domain: str | None = None,
    cache: dict[str, tuple[AtsProvider, str]] | None = None,
) -> OpeningsResult:
    """Verify live engineering openings for one company."""
    resolved = discover(company_name, domain, cache)
    if resolved is None:
        return OpeningsResult(status="unverified")

    provider, token = resolved
    postings = PROVIDERS[provider](token)
    if postings is None:
        # The cache pointed at a board that no longer answers.
        return OpeningsResult(status="unverified", ats_provider=provider, board_token=token)

    eng = [p for p in postings if is_engineering_title(p.title)]
    dates = [p.posted_at for p in eng if p.posted_at]

    return OpeningsResult(
        status="verified" if eng else "none_found",
        ats_provider=provider,
        board_token=token,
        board_url=BOARD_URLS[provider].format(token=token),
        eng_role_count=len(eng),
        total_role_count=len(postings),
        sample_titles=[p.title for p in eng[:_MAX_SAMPLE_TITLES]],
        newest_post_date=max(dates) if dates else None,
    )


def check_many(
    companies: Iterable[tuple[str, str | None]],
    cache: dict[str, tuple[AtsProvider, str]] | None = None,
    max_workers: int = 6,
) -> list[OpeningsResult]:
    """Check several companies concurrently.

    Discovery is I/O-bound and slow (up to five careers-page fetches plus token
    guesses per company), so this is where the wall-clock time goes. Kept modest
    to stay polite to the boards.
    """
    items = list(companies)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(lambda c: check(c[0], c[1], cache), items))
