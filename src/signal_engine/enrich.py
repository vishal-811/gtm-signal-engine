"""Optional Apollo firmographic enrichment.

Apollo's raw API is gated to their Organization plan; Basic and Professional
can only integrate through Zapier/Make/CRM connectors. This module is therefore
built as a strictly optional layer: when ``APOLLO_ENABLED`` is false — or when
the API rejects the key — every function is a no-op that returns the candidate
unchanged, and scoring proceeds on the fields it already has.

No code path in the pipeline depends on Apollo succeeding.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import httpx

from .config import settings
from .schemas import Candidate, Enrichment

log = logging.getLogger(__name__)

_ENRICH_URL = "https://api.apollo.io/api/v1/organizations/enrich"
_TIMEOUT = httpx.Timeout(20.0, connect=8.0)
_MAX_WORKERS = 4

# Set once per run after the first auth rejection, so a plan that lacks API
# access produces one warning instead of one per company.
_disabled_this_run = False


def reset() -> None:
    """Clear the per-run auth-failure latch."""
    global _disabled_this_run
    _disabled_this_run = False


def is_enabled() -> bool:
    cfg = settings()
    return bool(cfg.apollo_enabled and cfg.apollo_api_key) and not _disabled_this_run


def fetch(domain: str) -> Enrichment | None:
    """Enrich one company by domain. Returns None on any failure."""
    global _disabled_this_run
    if not is_enabled():
        return None

    try:
        response = httpx.post(
            _ENRICH_URL,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "x-api-key": settings().apollo_api_key,
            },
            json={"domain": domain},
            timeout=_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("apollo request failed for %s: %s", domain, exc)
        return None

    if response.status_code in (401, 403):
        # Expected on Basic/Professional. Latch off for the rest of the run.
        _disabled_this_run = True
        log.warning(
            "Apollo rejected the API key (HTTP %s). Raw API access requires the "
            "Organization plan; disabling enrichment for this run. Set "
            "APOLLO_ENABLED=false to silence this.",
            response.status_code,
        )
        return None

    if response.status_code != 200:
        log.debug("apollo returned HTTP %s for %s", response.status_code, domain)
        return None

    try:
        org = (response.json() or {}).get("organization")
    except ValueError:
        return None
    if not org:
        return None

    return Enrichment(
        employee_count=org.get("estimated_num_employees"),
        industry=org.get("industry"),
        hq_city=org.get("city"),
        hq_country=org.get("country"),
        linkedin_url=org.get("linkedin_url"),
        website_url=org.get("website_url"),
        technologies=[
            t for t in (org.get("technology_names") or []) if isinstance(t, str)
        ][:25],
    )


def enrich_all(candidates: list[Candidate]) -> list[Candidate]:
    """Attach enrichment where possible. Always returns every candidate."""
    if not candidates or not is_enabled():
        return candidates

    targets = [c for c in candidates if c.event.company_domain]
    if not targets:
        return candidates

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(targets))) as pool:
        results = list(pool.map(lambda c: fetch(c.event.company_domain or ""), targets))

    for candidate, enrichment in zip(targets, results):
        candidate.enrichment = enrichment

    log.info(
        "apollo enriched %d of %d candidates",
        sum(1 for r in results if r),
        len(candidates),
    )
    return candidates
