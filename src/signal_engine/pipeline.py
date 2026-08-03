"""Pipeline orchestration.

Stage order is chosen so cost falls as the funnel narrows: the cheap filters run
before the paid extraction, extraction before the slow ATS lookups, and the
expensive high-effort scoring before drafting, which only ever sees companies
that already cleared the threshold.

``dry_run`` is the default everywhere. Writing to Sheets and Slack requires an
explicit opt-in, so no accidental invocation ever posts to a real channel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from . import draft, enrich, extract, filters, llm, openings, score
from .config import settings
from .schemas import AtsProvider, Candidate, RunStats
from .sinks import slack
from .sources import rss

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    stats: RunStats
    posted: list[Candidate] = field(default_factory=list)
    all_scored: list[Candidate] = field(default_factory=list)
    filter_report: filters.FilterReport | None = None
    geo_report: filters.GeoReport | None = None
    feed_results: list[rss.FeedResult] = field(default_factory=list)
    sheet_url: str | None = None


def run(
    *,
    dry_run: bool = True,
    limit: int | None = None,
    skip_openings: bool = False,
    since_days: int | None = None,
) -> RunResult:
    """Execute one full pipeline run."""
    started = datetime.now(timezone.utc)
    run_date = started.date()
    stats = RunStats(started_at=started, dry_run=dry_run)
    llm.reset_usage()
    extract.reset_failures()
    score.reset_failures()
    enrich.reset()

    sheets_client = None
    seen_keys: dict[str, date] | None = None
    ats_cache: dict[str, tuple[AtsProvider, str]] = {}

    # State lives in the sheet. A dry run deliberately skips loading it so you
    # see the full unfiltered picture rather than a list silently thinned by
    # last week's suppressions.
    if not dry_run:
        from .sinks.sheets import SheetsClient

        sheets_client = SheetsClient()
        sheets_client.ensure_tabs()
        seen_keys = filters.normalized_seen_keys(sheets_client.read_seen())
        ats_cache = sheets_client.read_ats_cache()
        log.info(
            "loaded state: %d seen companies, %d cached ATS boards",
            len(seen_keys),
            len(ats_cache),
        )

    # ── 1. Ingest ─────────────────────────────────────────────────────────────
    articles, feed_results = rss.collect(since_days=since_days)
    stats.articles_fetched = sum(len(r.articles) for r in feed_results)
    stats.articles_after_dedupe = len(articles)
    for result in feed_results:
        if not result.ok:
            stats.errors.append(f"feed {result.feed.name}: {result.error}")

    if limit:
        articles = articles[:limit]
        log.info("limited to %d articles", len(articles))

    # ── 2. Extract ────────────────────────────────────────────────────────────
    events = extract.extract(articles)
    stats.events_extracted = len(events)

    # A lost article is not a quiet news day. Without this the two are
    # indistinguishable: a run where the endpoint rejected everything finished
    # green with an empty shortlist, and the nightly cron would have kept
    # reporting success indefinitely.
    lost = extract.failures
    if lost.articles:
        stats.errors.append(
            f"extraction lost {lost.articles}/{len(articles)} articles across "
            f"{lost.batches} batch(es): {'; '.join(lost.reasons[:3])}"
        )
        stats.extraction_failed_articles = lost.articles

    # ── 3. Filter ─────────────────────────────────────────────────────────────
    candidates, report = filters.apply(events, seen_keys=seen_keys)

    if not candidates:
        log.info("no candidates survived filtering")
        return _finish(
            stats, [], [], report, feed_results, sheets_client, run_date, dry_run
        )

    # ── 4. Enrich (optional) ──────────────────────────────────────────────────
    candidates = enrich.enrich_all(candidates)

    # ── 5. Verify openings ────────────────────────────────────────────────────
    # Runs before the geography filter, because the job board is where the
    # location data actually lives. Funding articles almost never state a
    # company's HQ — a full run measured zero out of sixty-six — so filtering
    # on the article first discarded every candidate.
    if skip_openings:
        log.info("skipping openings verification (--skip-openings)")
    else:
        results = openings.check_many(
            [(c.event.company_name, c.event.company_domain) for c in candidates],
            cache=ats_cache,
        )
        newly_discovered: dict[str, tuple[AtsProvider, str]] = {}
        for candidate, result in zip(candidates, results):
            candidate.openings = result
            domain = candidate.event.company_domain
            if domain and result.ats_provider and result.board_token:
                newly_discovered[domain] = (result.ats_provider, result.board_token)

        stats.openings_verified = sum(
            1 for c in candidates if c.openings and c.openings.status == "verified"
        )
        log.info(
            "openings: %d verified, %d no-eng-roles, %d unverified",
            stats.openings_verified,
            sum(1 for c in candidates if c.openings and c.openings.status == "none_found"),
            sum(1 for c in candidates if c.openings and c.openings.status == "unverified"),
        )
        if sheets_client and newly_discovered:
            # Caching board tokens is a latency optimisation for future runs.
            # A transient Sheets error here once killed a run that had already
            # paid for every extraction call — never worth losing the work.
            try:
                sheets_client.upsert_ats_cache(newly_discovered)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"ats_cache write: {exc}")
                log.warning("could not update the ATS cache: %s", exc)

        # Now that boards are resolved, collapse the companies that upstream
        # dedupe could not see were the same. Done before scoring so a
        # duplicate costs neither a model call nor a row in the shortlist.
        candidates, collapsed = filters.collapse_duplicate_boards(candidates)
        if collapsed:
            log.info("collapsed %d duplicate companies onto shared boards", collapsed)

    # ── 6. Geography ──────────────────────────────────────────────────────────
    candidates, geo_report = filters.apply_geo(candidates)
    stats.passed_filters = len(candidates)
    if not candidates:
        log.info("no candidates are in a target market")
        return _finish(
            stats, [], [], report, feed_results, sheets_client, run_date, dry_run,
            geo_report=geo_report,
        )

    # ── 7. Score ──────────────────────────────────────────────────────────────
    scored = score.score_all(candidates)
    # Count only what was actually scored. A candidate whose scoring call died
    # has no composite, which reads downstream as "below threshold" — the same
    # as a company judged weak — so counting it as scored would report a
    # collapsed stage as a quiet one.
    stats.scored = len(scored) - score.failures.candidates
    if score.failures.candidates:
        stats.scoring_failed_candidates = score.failures.candidates
        stats.errors.append(
            f"scoring failed for {score.failures.candidates}/{len(scored)} "
            f"candidates: {'; '.join(score.failures.reasons[:3])}"
        )
    passing = [c for c in scored if score.above_threshold(c)]

    # ── 8. Draft (threshold-passers only) ─────────────────────────────────────
    passing = draft.draft_all(passing)
    stats.posted = len(passing)

    return _finish(
        stats, passing, scored, report, feed_results, sheets_client, run_date,
        dry_run, geo_report=geo_report,
    )


def _finish(
    stats: RunStats,
    passing: list[Candidate],
    scored: list[Candidate],
    report: filters.FilterReport,
    feed_results: list[rss.FeedResult],
    sheets_client,
    run_date: date,
    dry_run: bool,
    geo_report: filters.GeoReport | None = None,
) -> RunResult:
    """Publish (unless dry) and close out run accounting."""
    stats.input_tokens = llm.usage.input_tokens
    stats.output_tokens = llm.usage.output_tokens
    stats.cache_read_tokens = llm.usage.cached_tokens
    stats.estimated_cost_usd = llm.usage.cost_usd

    sheet_url = None
    if not dry_run and sheets_client is not None:
        try:
            inserted, refreshed = sheets_client.append_shortlist(passing, run_date)
            log.info("sheet: %d new rows, %d refreshed", inserted, refreshed)
            # New rows land at the bottom regardless of score; re-sort so the
            # top of the sheet is still the part worth reading.
            if inserted:
                sheets_client.sort_shortlist()
            sheets_client.upsert_seen(passing, run_date)
            sheet_url = sheets_client.url
        except Exception as exc:  # noqa: BLE001
            stats.errors.append(f"sheets write: {exc}")
            log.exception("failed writing to Sheets")

        stats.finished_at = datetime.now(timezone.utc)
        if settings().slack_webhook_url:
            try:
                slack.send_digest(passing, run_date, sheet_url, stats)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"slack post: {exc}")
                log.exception("failed posting to Slack")
        else:
            log.info("no Slack webhook configured; skipping the digest")

        # Written last so it records the outcome of the writes above.
        try:
            sheets_client.append_run(stats)
        except Exception as exc:  # noqa: BLE001
            log.exception("failed writing run log: %s", exc)

    if stats.finished_at is None:
        stats.finished_at = datetime.now(timezone.utc)

    log.info("run complete: %s | %s", report.summary(), llm.usage.summary())
    return RunResult(
        stats=stats,
        posted=passing,
        all_scored=scored,
        filter_report=report,
        geo_report=geo_report,
        feed_results=feed_results,
        sheet_url=sheet_url,
    )
