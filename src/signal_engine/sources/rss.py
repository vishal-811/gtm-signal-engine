"""RSS ingest.

Pulls every enabled feed, normalizes wildly different feed dialects into
:class:`Article`, and collapses the duplicates that come from a dozen outlets
covering the same round.

Two properties matter more than throughput here:

* **One broken feed must not kill the run.** Feeds rot constantly — a 404, a
  redirect to an HTML error page, a malformed date. Each feed is isolated and
  its failure recorded, never raised.
* **Dedupe must be aggressive.** Google News alone syndicates the same story
  under several publisher names, and the extraction stage is the expensive
  part of the pipeline. Every duplicate removed here is money saved.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx

from ..config import Feed, feeds_config, settings
from ..schemas import Article
from ..textutil import canonical_url

log = logging.getLogger(__name__)

# Some feeds reject the default python-urllib agent.
_USER_AGENT = (
    "Mozilla/5.0 (compatible; SignalEngine/0.1; +https://github.com/hire100x/signal-engine)"
)
_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_MAX_WORKERS = 8


@dataclass
class FeedResult:
    """Per-feed outcome, so `check-feeds` can report exactly what broke."""

    feed: Feed
    articles: list[Article] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _parse_datetime(entry: Any) -> datetime | None:
    """Extract a timezone-aware publication time from a feedparser entry.

    Feeds disagree on everything here: field name, format, and whether a
    timezone is included at all. Naive datetimes are treated as UTC — being an
    hour off is harmless for a 36-hour recency window, whereas raising would
    discard an otherwise good article.
    """
    for attr in ("published_parsed", "updated_parsed"):
        struct = getattr(entry, attr, None)
        if struct:
            try:
                return datetime(*struct[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass

    for attr in ("published", "updated", "created"):
        raw = getattr(entry, attr, None)
        if not raw:
            continue
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            continue
        if parsed is None:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    return None


def _clean_summary(entry: Any, limit: int = 1200) -> str:
    """Best available body text, stripped of markup and capped.

    The cap matters: feed summaries occasionally carry a full article plus
    boilerplate, and every character reaches the extraction prompt as tokens.
    """
    raw = ""
    content = getattr(entry, "content", None)
    if content:
        raw = content[0].get("value", "")
    if not raw:
        raw = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""

    # feedparser hands back HTML for most feeds; a tag-strip is enough here
    # since we only need prose for the model to read.
    import re

    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def fetch_feed(feed: Feed) -> FeedResult:
    """Fetch and parse one feed. Never raises — failures land in ``error``."""
    try:
        response = httpx.get(
            feed.url,
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - one dead feed must not stop the run
        return FeedResult(feed=feed, error=f"{type(exc).__name__}: {exc}")

    parsed = feedparser.parse(response.content)

    # `bozo` means the XML was malformed. feedparser usually still recovers
    # entries, so this is only fatal when it yields nothing.
    if parsed.bozo and not parsed.entries:
        reason = getattr(parsed, "bozo_exception", "unparseable feed")
        return FeedResult(feed=feed, error=f"malformed feed: {reason}")

    articles: list[Article] = []
    for entry in parsed.entries:
        link = getattr(entry, "link", "") or ""
        title = (getattr(entry, "title", "") or "").strip()
        if not link or not title:
            continue
        articles.append(
            Article(
                title=title,
                url=link,
                source=feed.name,
                published_at=_parse_datetime(entry),
                summary=_clean_summary(entry),
            )
        )

    return FeedResult(feed=feed, articles=articles)


def fetch_all(feeds: list[Feed] | None = None) -> list[FeedResult]:
    """Fetch every enabled feed concurrently."""
    targets = feeds if feeds is not None else feeds_config().active
    if not targets:
        return []
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(targets))) as pool:
        return list(pool.map(fetch_feed, targets))


def filter_recent(
    articles: list[Article], max_age_hours: int | None = None
) -> list[Article]:
    """Drop anything older than the recency window.

    Articles with no parseable date are **kept**. A missing timestamp is a feed
    quirk, not evidence of staleness, and the extraction stage reads the
    announcement date out of the article body anyway.
    """
    hours = max_age_hours if max_age_hours is not None else settings().max_article_age_hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return [a for a in articles if a.published_at is None or a.published_at >= cutoff]


def dedupe(articles: list[Article]) -> list[Article]:
    """Collapse duplicate coverage, keeping the earliest-published version.

    Two passes, because the same story arrives both as a literal URL repeat and
    as near-identical headlines from different outlets:

    1. canonical URL (tracking params and ``www.`` stripped)
    2. normalized title
    """
    by_url: dict[str, Article] = {}
    for article in articles:
        key = canonical_url(article.url)
        existing = by_url.get(key)
        if existing is None or _is_earlier(article, existing):
            by_url[key] = article

    by_title: dict[str, Article] = {}
    for article in by_url.values():
        key = article.dedupe_key()
        existing = by_title.get(key)
        if existing is None or _is_earlier(article, existing):
            by_title[key] = article

    return sorted(
        by_title.values(),
        key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


def _is_earlier(candidate: Article, incumbent: Article) -> bool:
    """Prefer the earliest-published copy — usually the original reporting
    rather than an aggregator's rewrite."""
    if candidate.published_at is None:
        return False
    if incumbent.published_at is None:
        return True
    return candidate.published_at < incumbent.published_at


def collect(feeds: list[Feed] | None = None) -> tuple[list[Article], list[FeedResult]]:
    """Full ingest: fetch, filter by recency, dedupe.

    Returns the article list plus per-feed results so the caller can report
    which feeds failed.
    """
    results = fetch_all(feeds)
    raw = [a for r in results for a in r.articles]
    recent = filter_recent(raw)
    deduped = dedupe(recent)

    log.info(
        "ingest: %d feeds -> %d articles -> %d recent -> %d unique",
        len(results),
        len(raw),
        len(recent),
        len(deduped),
    )
    for result in results:
        if not result.ok:
            log.warning("feed '%s' failed: %s", result.feed.name, result.error)

    return deduped, results
