"""Rubric scoring.

The model scores each criterion 0–5 with a justification; **Python computes the
weighted composite**. That split is deliberate: arithmetic done by a language
model is unauditable and occasionally wrong, whereas this way the composite is
a pure function of the criterion scores and the weights in ``rubric.yaml``,
unit-tested and reproducible.

Runs at ``effort="high"`` — this is the judgment call the whole pipeline exists
to make, and it runs on far fewer items than extraction.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from .config import Rubric, prompt, rubric as load_rubric
from .llm import RefusalError, structured_call
from .schemas import Candidate, ScoreResult, clamp_score

log = logging.getLogger(__name__)

_MAX_WORKERS = 4
_MAX_TOKENS = 4000


def render_rubric(rubric: Rubric) -> str:
    """Render rubric.yaml into prompt text.

    Appended to the static instructions to form the system prompt. Stable
    across every company in a run, so the whole thing caches — which is the
    single biggest cost lever in this stage.
    """
    blocks = []
    for criterion in rubric.criteria:
        anchors = "\n".join(
            f"  - {score} → {description}"
            for score, description in sorted(
                criterion.anchors.items(), key=lambda kv: kv[0], reverse=True
            )
        )
        blocks.append(
            f"## {criterion.id}  (weight {criterion.weight:.0%})\n\n"
            f"{criterion.question.strip()}\n\n"
            f"Anchors:\n{anchors}"
        )
    return "\n\n".join(blocks)


def system_prompt(rubric: Rubric | None = None) -> str:
    return f"{prompt('score')}\n\n{render_rubric(rubric or load_rubric())}"


def render_candidate(candidate: Candidate) -> str:
    """Everything the model needs about one company, in the user turn."""
    event = candidate.event
    lines = [
        "<company>",
        f"Name: {event.company_name}",
        f"Domain: {event.company_domain or 'unknown'}",
        f"What they do: {event.one_line_description}",
        f"Sector: {event.sector}",
        f"HQ: {event.hq_city or 'unknown'}, {event.hq_country or 'unknown'}",
        f"Matched target market: {candidate.market or 'none'}",
        "</company>",
        "",
        "<funding>",
        f"Stage: {event.round_stage}",
        f"Amount: {'$' + format(event.amount_usd, ',') if event.amount_usd else 'undisclosed'}",
        f"Announced: {event.announced_date or 'unknown'}",
        f"Investors: {', '.join(event.investors) if event.investors else 'not named'}",
        f"Source article: {event.source_url}",
        "</funding>",
        "",
    ]

    openings = candidate.openings
    lines.append("<engineering_openings>")
    if openings is None or openings.status == "unverified":
        lines.append(
            "Status: UNVERIFIED — no public job board could be found for this "
            "company. This is weaker evidence of hiring than a verified board."
        )
        if openings and openings.board_token:
            lines.append(f"(A previously known board '{openings.board_token}' no longer responds.)")
    elif openings.status == "none_found":
        lines.append(
            f"Status: BOARD FOUND, ZERO ENGINEERING ROLES. Their {openings.ats_provider} "
            f"board lists {openings.total_role_count} open role(s), none of which are "
            "engineering."
        )
        lines.append(f"Board: {openings.board_url}")
    else:
        lines.append(f"Status: VERIFIED via {openings.ats_provider}")
        lines.append(
            f"Open engineering roles: {openings.eng_role_count} "
            f"(of {openings.total_role_count} total open roles)"
        )
        if openings.newest_post_date:
            lines.append(
                f"Most recent engineering post: {openings.newest_post_date.date()}"
            )
        if openings.sample_titles:
            lines.append("Sample engineering titles:")
            lines.extend(f"  - {t}" for t in openings.sample_titles)
        lines.append(f"Board: {openings.board_url}")
    lines.append("</engineering_openings>")

    if candidate.enrichment:
        enrichment = candidate.enrichment
        lines += [
            "",
            "<firmographics>",
            f"Employees: {enrichment.employee_count or 'unknown'}",
            f"Industry: {enrichment.industry or 'unknown'}",
            f"Tech stack: {', '.join(enrichment.technologies[:15]) or 'unknown'}",
            "</firmographics>",
        ]

    lines += ["", "Score this company against every criterion in the rubric."]
    return "\n".join(lines)


def composite(result: ScoreResult, rubric: Rubric | None = None) -> float:
    """Weighted total, computed here rather than by the model.

    Criterion scores are clamped to 0–5 (the schema deliberately carries no
    numeric bounds, because the structured-output subset does not support them
    and a stray 7 would otherwise fail validation and lose the whole record).

    A criterion the model omitted contributes 0 rather than being dropped from
    the denominator — silently rescaling would let a model skip its way to a
    higher score.
    """
    rubric = rubric or load_rubric()
    by_id = {c.id: c for c in result.criteria}

    total = 0.0
    for criterion in rubric.criteria:
        scored = by_id.get(criterion.id)
        if scored is None:
            log.warning("model omitted criterion '%s'; counting it as 0", criterion.id)
            continue
        total += clamp_score(scored.score) * criterion.weight
    return round(total, 3)


def score_one(candidate: Candidate, rubric: Rubric | None = None) -> Candidate:
    """Score a single candidate in place and return it.

    On failure the candidate comes back with ``score`` and ``composite`` unset,
    which the pipeline treats as below-threshold rather than crashing the run.
    """
    rubric = rubric or load_rubric()
    try:
        result = structured_call(
            system=system_prompt(rubric),
            user=render_candidate(candidate),
            schema=ScoreResult,
            effort="high",
            max_tokens=_MAX_TOKENS,
            label="score",
        )
    except RefusalError as exc:
        log.warning("scoring refused for %s: %s", candidate.event.company_name, exc)
        return candidate
    except Exception as exc:  # noqa: BLE001
        log.error("scoring failed for %s: %s", candidate.event.company_name, exc)
        return candidate

    candidate.score = result
    candidate.composite = composite(result, rubric)
    return candidate


def score_all(candidates: list[Candidate]) -> list[Candidate]:
    """Score every candidate concurrently, returned ranked best-first."""
    if not candidates:
        return []
    rubric = load_rubric()

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(candidates))) as pool:
        scored = list(pool.map(lambda c: score_one(c, rubric), candidates))

    scored.sort(key=lambda c: c.composite if c.composite is not None else -1, reverse=True)
    passing = sum(1 for c in scored if above_threshold(c, rubric))
    log.info(
        "scored %d candidates, %d above threshold %.2f",
        len(scored),
        passing,
        rubric.threshold,
    )
    return scored


def above_threshold(candidate: Candidate, rubric: Rubric | None = None) -> bool:
    rubric = rubric or load_rubric()
    return candidate.composite is not None and candidate.composite > rubric.threshold
