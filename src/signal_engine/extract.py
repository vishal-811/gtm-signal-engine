"""Turn articles into structured funding events using Claude.

Articles are sent in batches. The system prompt (long, static, and cached)
dominates a single-article request, so batching amortizes it across ten
articles and cuts the per-article cost substantially. Batches are also the unit
of failure isolation: one malformed article poisons at most its own batch, and
the run continues.

Runs at ``effort="low"`` — this is mechanical extraction against an explicit
field spec, not a judgment call, and it is the highest-volume stage.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from .config import prompt
from .llm import RefusalError, structured_call
from .schemas import Article, ExtractionBatch, FundingEvent

log = logging.getLogger(__name__)

BATCH_SIZE = 10
_MAX_WORKERS = 4
# Ten records of ~15 short fields. Generous enough that a verbose batch never
# truncates, which would fail the whole batch rather than one record.
_MAX_TOKENS = 8000


def _render_batch(articles: list[Article]) -> str:
    """Format a batch as a numbered list for the model.

    Kept out of the system prompt deliberately: the system prompt must stay
    byte-identical across calls to hit the prompt cache, so everything that
    varies per request lives here in the user turn.
    """
    blocks = []
    for index, article in enumerate(articles, start=1):
        published = (
            article.published_at.strftime("%Y-%m-%d") if article.published_at else "unknown"
        )
        blocks.append(
            f"<article index=\"{index}\">\n"
            f"Title: {article.title}\n"
            f"Source: {article.source}\n"
            f"Published: {published}\n"
            f"URL: {article.url}\n"
            f"Summary: {article.summary or '(no summary provided)'}\n"
            f"</article>"
        )
    return (
        f"Extract one record for each of the following {len(articles)} articles, "
        f"in order.\n\n" + "\n\n".join(blocks)
    )


def _extract_batch(articles: list[Article]) -> list[FundingEvent]:
    """Extract one batch. Returns [] on failure rather than raising."""
    try:
        result = structured_call(
            system=prompt("extract"),
            user=_render_batch(articles),
            schema=ExtractionBatch,
            effort="low",
            max_tokens=_MAX_TOKENS,
            label="extract",
        )
    except RefusalError as exc:
        log.warning("extraction batch refused, skipping %d articles: %s", len(articles), exc)
        return []
    except Exception as exc:  # noqa: BLE001 - one bad batch must not end the run
        log.error("extraction batch failed for %d articles: %s", len(articles), exc)
        return []

    events = result.events

    # The model is told to return one record per article in order. When it
    # doesn't, positional URL backfill below would attach the wrong source to
    # the wrong company, so trust only the count it got right.
    if len(events) != len(articles):
        log.warning(
            "extraction returned %d records for %d articles; "
            "source URLs will only be backfilled where they are missing",
            len(events),
            len(articles),
        )

    for index, event in enumerate(events):
        # Models occasionally paraphrase or truncate the URL. The article's own
        # URL is authoritative, and it is what the outreach draft cites.
        if index < len(articles) and (
            not event.source_url or event.source_url != articles[index].url
        ):
            if len(events) == len(articles):
                event.source_url = articles[index].url

    return events


def extract(articles: list[Article], batch_size: int = BATCH_SIZE) -> list[FundingEvent]:
    """Extract funding events from every article.

    Batches run concurrently; a failed batch contributes nothing and is logged.
    """
    if not articles:
        return []

    batches = [
        articles[i : i + batch_size] for i in range(0, len(articles), batch_size)
    ]
    log.info(
        "extracting %d articles in %d batches of up to %d",
        len(articles),
        len(batches),
        batch_size,
    )

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(batches))) as pool:
        results = list(pool.map(_extract_batch, batches))

    events = [event for batch in results for event in batch]
    funding = sum(1 for e in events if e.is_funding_announcement)
    log.info(
        "extracted %d records, %d of which are funding announcements",
        len(events),
        funding,
    )
    return events
