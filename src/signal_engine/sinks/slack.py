"""Slack sink — the morning digest.

Posts one Block Kit message per run to an incoming webhook. Design choices that
matter:

* **Zero-result days still post.** Silence is ambiguous — it could mean no good
  companies or a crashed cron. An explicit "nothing cleared the bar" line
  distinguishes them.
* **Email drafts are truncated in Slack**, with the full text in the Sheet. A
  90-word email per company across ten companies makes the digest unreadable.
* **A failed post never fails the run.** By the time this executes, the Sheet
  has already been written and the expensive work is done.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx

from ..config import settings
from ..schemas import Candidate, RunStats
from ..textutil import truncate_words

log = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(20.0, connect=8.0)
# Slack rejects payloads over 50 blocks; keep well clear and link out instead.
_MAX_COMPANY_BLOCKS = 20
_BODY_PREVIEW_WORDS = 28


def _openings_line(candidate: Candidate) -> str:
    openings = candidate.openings
    if openings is None or openings.status == "unverified":
        return ":grey_question: no public job board found"
    if openings.status == "none_found":
        return f":no_entry_sign: board found, 0 engineering roles ({openings.total_role_count} other)"
    fresh = ""
    if openings.newest_post_date:
        fresh = f", newest {openings.newest_post_date.date().isoformat()}"
    return f":white_check_mark: *{openings.eng_role_count} eng roles* open{fresh}"


def _company_block(candidate: Candidate, rank: int) -> dict[str, Any]:
    event = candidate.event
    amount = f"${event.amount_usd:,}" if event.amount_usd else "undisclosed"
    score = f"{candidate.composite:.2f}" if candidate.composite is not None else "—"

    lines = [
        f"*{rank}. <{event.source_url}|{event.company_name}>*  ·  score *{score}*",
        f"{event.round_stage} · {amount} · {event.hq_city or 'HQ unknown'} · {event.sector}",
        _openings_line(candidate),
    ]
    if candidate.score and candidate.score.key_signal:
        lines.append(f"> {candidate.score.key_signal}")
    if candidate.outreach:
        preview = truncate_words(candidate.outreach.body, _BODY_PREVIEW_WORDS)
        lines.append(f"✉️ *{candidate.outreach.subject}*\n_{preview}_")
    if candidate.score and candidate.score.risks:
        lines.append(f":warning: {'; '.join(candidate.score.risks[:2])}")

    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "\n".join(lines)},
    }


def build_payload(
    candidates: list[Candidate],
    run_date: date,
    sheet_url: str | None = None,
    stats: RunStats | None = None,
) -> dict[str, Any]:
    """Build the Block Kit payload. Pure — unit-tested without any network."""
    heading = run_date.strftime("%a %d %b %Y")

    if not candidates:
        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Signal Engine · {heading}"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "No companies cleared the fit threshold today. "
                        "The pipeline ran normally."
                    ),
                },
            },
        ]
    else:
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Signal Engine · {len(candidates)} companies · {heading}",
                },
            }
        ]
        for rank, candidate in enumerate(candidates[:_MAX_COMPANY_BLOCKS], start=1):
            blocks.append({"type": "divider"})
            blocks.append(_company_block(candidate, rank))

        if len(candidates) > _MAX_COMPANY_BLOCKS:
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"_+{len(candidates) - _MAX_COMPANY_BLOCKS} more "
                                "in the sheet._"
                            ),
                        }
                    ],
                }
            )

    context_bits: list[str] = []
    if sheet_url:
        context_bits.append(f"<{sheet_url}|Full shortlist and email drafts →>")
    if stats:
        context_bits.append(
            f"{stats.articles_fetched} articles → {stats.passed_filters} in-market "
            f"→ {stats.posted} posted · ${stats.estimated_cost_usd:.2f}"
        )
    if stats and stats.errors:
        context_bits.append(f":warning: {len(stats.errors)} error(s) this run")
    if context_bits:
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": "  ·  ".join(context_bits)}],
            }
        )

    summary = (
        f"Signal Engine: {len(candidates)} companies for {heading}"
        if candidates
        else f"Signal Engine: nothing cleared the bar for {heading}"
    )
    # `text` is the notification/fallback string shown in push notifications
    # and by clients that cannot render blocks.
    return {"text": summary, "blocks": blocks}


def post(payload: dict[str, Any], webhook_url: str | None = None) -> bool:
    """Post to the webhook. Returns success; never raises."""
    url = webhook_url or settings().slack_webhook_url
    if not url:
        log.warning("SLACK_WEBHOOK_URL is not set; skipping Slack post")
        return False
    try:
        response = httpx.post(url, json=payload, timeout=_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - the sheet is already written
        log.error("slack post failed: %s", exc)
        return False
    if response.status_code != 200:
        log.error("slack post returned HTTP %s: %s", response.status_code, response.text[:200])
        return False
    return True


def send_digest(
    candidates: list[Candidate],
    run_date: date,
    sheet_url: str | None = None,
    stats: RunStats | None = None,
) -> bool:
    return post(build_payload(candidates, run_date, sheet_url, stats))


def send_failure_alert(message: str) -> bool:
    """Tell the channel the run itself died, so silence is never ambiguous."""
    return post(
        {
            "text": f"Signal Engine run failed: {message}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":rotating_light: *Signal Engine run failed*\n```{message[:2000]}```",
                    },
                }
            ],
        }
    )
