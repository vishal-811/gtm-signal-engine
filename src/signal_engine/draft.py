"""Outreach email drafting.

**Nothing in this module sends anything.** It produces text that lands in a
Google Sheet and a Slack message for a human to read, edit, and send. There is
no SMTP client, no ESP integration, and no send function anywhere in this
codebase — by design, so a bug can never email a real founder.

Runs at ``effort="medium"``: quality writing matters, but this stage only sees
companies that already cleared the score threshold, so volume is low.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from .config import prompt, settings
from .llm import RefusalError, structured_call
from .schemas import Candidate, Outreach
from .textutil import truncate_words

log = logging.getLogger(__name__)

_MAX_WORKERS = 4
_MAX_TOKENS = 2000
BODY_WORD_LIMIT = 90


def system_prompt() -> str:
    """Static instructions plus the sender's identity.

    The sender block is constant for the whole run, so it stays in the system
    prompt where it caches rather than being repeated per company.
    """
    cfg = settings()
    signature = (
        "\n\n---\n\n# SENDER\n\n"
        "Sign the email with exactly these details. Do not alter them.\n\n"
        f"Name: {cfg.sender_name or '[SENDER_NAME not set]'}\n"
        f"Title: {cfg.sender_title or '[SENDER_TITLE not set]'}\n"
        f"Company: {cfg.sender_company}\n"
        f"Email: {cfg.sender_email or '[SENDER_EMAIL not set]'}\n"
    )
    return prompt("draft") + signature


def render_candidate(candidate: Candidate) -> str:
    """The facts the email may draw on — and nothing else."""
    event = candidate.event
    lines = [
        "Write a cold outreach email to this company.",
        "",
        "<company>",
        f"Name: {event.company_name}",
        f"What they do: {event.one_line_description}",
        f"Sector: {event.sector}",
        f"HQ: {event.hq_city or 'unknown'}",
        "</company>",
        "",
        "<funding>",
        f"Stage: {event.round_stage}",
        f"Amount: {'$' + format(event.amount_usd, ',') if event.amount_usd else 'undisclosed'}",
        f"Announced: {event.announced_date or 'recently'}",
        f"Lead investor: {event.investors[0] if event.investors else 'not named'}",
        "</funding>",
    ]

    openings = candidate.openings
    lines += ["", "<open_engineering_roles>"]
    if openings and openings.status == "verified" and openings.sample_titles:
        lines.append(
            f"{openings.eng_role_count} open engineering roles on their "
            f"{openings.ats_provider} board. Exact titles you may cite:"
        )
        lines.extend(f"  - {title}" for title in openings.sample_titles)
    else:
        lines.append(
            "No specific role titles are available. Reference their engineering "
            "hiring in general terms and do NOT invent a job title."
        )
    lines.append("</open_engineering_roles>")

    if candidate.score and candidate.score.key_signal:
        lines += [
            "",
            "<why_them>",
            candidate.score.key_signal,
            "</why_them>",
        ]

    return "\n".join(lines)


def draft_one(candidate: Candidate) -> Candidate:
    """Draft outreach for a single candidate, in place."""
    try:
        outreach = structured_call(
            system=system_prompt(),
            user=render_candidate(candidate),
            schema=Outreach,
            effort="medium",
            max_tokens=_MAX_TOKENS,
            label="draft",
        )
    except RefusalError as exc:
        log.warning("drafting refused for %s: %s", candidate.event.company_name, exc)
        return candidate
    except Exception as exc:  # noqa: BLE001
        log.error("drafting failed for %s: %s", candidate.event.company_name, exc)
        return candidate

    # The word ceiling is stated in the prompt, but enforcing it here means a
    # long draft is trimmed rather than silently shipped over-length.
    word_count = len(outreach.body.split())
    if word_count > BODY_WORD_LIMIT:
        log.info(
            "trimming %s draft from %d to %d words",
            candidate.event.company_name,
            word_count,
            BODY_WORD_LIMIT,
        )
        outreach.body = truncate_words(outreach.body, BODY_WORD_LIMIT)

    candidate.outreach = outreach
    return candidate


def draft_all(candidates: list[Candidate]) -> list[Candidate]:
    """Draft outreach for every candidate, concurrently."""
    if not candidates:
        return []
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(candidates))) as pool:
        drafted = list(pool.map(draft_one, candidates))
    log.info(
        "drafted outreach for %d of %d candidates",
        sum(1 for c in drafted if c.outreach),
        len(drafted),
    )
    return drafted
