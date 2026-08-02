"""Google Sheets sink — also the pipeline's state store.

Four tabs, created on first use:

===========  =================================================================
shortlist    The deliverable. One row per company per run.
seen         Dedupe ledger: which companies were already posted, and when.
ats_cache    domain → (provider, board token), so discovery runs once ever.
runs         Run log: volumes, errors, and cost. The health dashboard.
===========  =================================================================

Using the sheet as the state store rather than a database keeps the whole
system inspectable by a non-engineer and means GitHub Actions needs no
persistent volume and no commit-back to git.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from ..config import settings
from ..schemas import AtsProvider, Candidate, RunStats

log = logging.getLogger(__name__)

SHORTLIST = "shortlist"
SEEN = "seen"
ATS_CACHE = "ats_cache"
RUNS = "runs"

SHORTLIST_HEADERS = [
    "run_date", "rank", "company", "domain", "market", "score",
    "round_stage", "amount_usd", "announced", "investors", "hq", "sector",
    "openings_status", "eng_roles", "total_roles", "newest_post",
    "sample_titles", "board_url", "key_signal", "risks",
    "criteria_breakdown", "email_subject", "email_body",
    "personalization_hook", "source_url",
]
SEEN_HEADERS = ["key", "company", "first_seen", "last_seen", "last_score", "times_posted"]
ATS_CACHE_HEADERS = ["domain", "provider", "board_token", "resolved_at"]
RUNS_HEADERS = [
    "started_at", "finished_at", "dry_run", "articles_fetched",
    "articles_after_dedupe", "events_extracted", "passed_filters",
    "openings_verified", "scored", "posted", "input_tokens", "output_tokens",
    "cache_read_tokens", "estimated_cost_usd", "errors",
]

_TABS = {
    SHORTLIST: SHORTLIST_HEADERS,
    SEEN: SEEN_HEADERS,
    ATS_CACHE: ATS_CACHE_HEADERS,
    RUNS: RUNS_HEADERS,
}


class SheetsClient:
    """Thin wrapper over gspread. Constructed lazily so importing this module
    never requires credentials — tests and `show-config` must work without."""

    def __init__(self, sheet_id: str | None = None) -> None:
        cfg = settings()
        self.sheet_id = sheet_id or cfg.google_sheet_id
        if not self.sheet_id:
            raise RuntimeError("GOOGLE_SHEET_ID is not set")

        info = cfg.google_credentials_info()
        if not info:
            raise RuntimeError(
                "No Google service-account credentials found. Set "
                "GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON."
            )

        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        self._sheet = gspread.authorize(creds).open_by_key(self.sheet_id)
        self._tabs_ready = False

    # ── tab management ────────────────────────────────────────────────────────

    def ensure_tabs(self, force: bool = False) -> None:
        """Create any missing tab with its header row.

        Runs at most once per client. Every read and write funnels through
        :meth:`_tab`, so without the latch this would list the spreadsheet's
        worksheets on each of the half-dozen sheet operations in a run —
        needless calls against a 60-reads-per-minute quota.
        """
        if self._tabs_ready and not force:
            return

        existing = {ws.title for ws in self._sheet.worksheets()}
        for title, headers in _TABS.items():
            if title in existing:
                continue
            worksheet = self._sheet.add_worksheet(
                title=title, rows=1000, cols=max(len(headers), 10)
            )
            worksheet.append_row(headers, value_input_option="RAW")
            worksheet.freeze(rows=1)
            log.info("created sheet tab '%s'", title)

        # A brand-new spreadsheet ships with a "Sheet1" that would otherwise
        # sit there confusing whoever opens it.
        if "Sheet1" in existing and len(existing) > 1:
            try:
                self._sheet.del_worksheet(self._sheet.worksheet("Sheet1"))
            except Exception:  # noqa: BLE001 - cosmetic only
                pass

        self._tabs_ready = True

    def _tab(self, title: str):
        self.ensure_tabs()
        return self._sheet.worksheet(title)

    # ── reads (state) ─────────────────────────────────────────────────────────

    def read_seen(self) -> list[tuple[str, date]]:
        """Return (key, last_seen) pairs for the dedupe ledger."""
        rows = self._tab(SEEN).get_all_records()
        out: list[tuple[str, date]] = []
        for row in rows:
            key = str(row.get("key", "")).strip()
            raw = str(row.get("last_seen", "")).strip()
            if not key or not raw:
                continue
            parsed = _parse_date(raw)
            if parsed:
                out.append((key, parsed))
        return out

    def read_ats_cache(self) -> dict[str, tuple[AtsProvider, str]]:
        rows = self._tab(ATS_CACHE).get_all_records()
        cache: dict[str, tuple[AtsProvider, str]] = {}
        for row in rows:
            domain = str(row.get("domain", "")).strip().lower()
            provider = str(row.get("provider", "")).strip().lower()
            token = str(row.get("board_token", "")).strip()
            if domain and provider and token:
                cache[domain] = (provider, token)  # type: ignore[assignment]
        return cache

    # ── writes ────────────────────────────────────────────────────────────────

    def append_shortlist(self, candidates: list[Candidate], run_date: date) -> int:
        if not candidates:
            return 0
        rows = [
            _shortlist_row(candidate, rank, run_date)
            for rank, candidate in enumerate(candidates, start=1)
        ]
        self._tab(SHORTLIST).append_rows(rows, value_input_option="RAW")
        return len(rows)

    def upsert_seen(self, candidates: list[Candidate], run_date: date) -> None:
        """Record posted companies so they are suppressed next time.

        Reads the whole tab and rewrites changed rows individually. Volumes are
        tens of rows per day, so a read-modify-write is simpler and safer than
        maintaining an index, and it keeps the tab human-editable.
        """
        if not candidates:
            return
        worksheet = self._tab(SEEN)
        records = worksheet.get_all_records()
        index = {str(r.get("key", "")).strip(): i for i, r in enumerate(records)}

        new_rows: list[list[Any]] = []
        for candidate in candidates:
            key = candidate.key
            score = candidate.composite if candidate.composite is not None else ""
            if key in index:
                # +2: one for the header row, one for 1-based sheet rows.
                row_number = index[key] + 2
                previous = records[index[key]]
                times = _as_int(previous.get("times_posted"), default=1) + 1
                worksheet.update(
                    range_name=f"D{row_number}:F{row_number}",
                    values=[[run_date.isoformat(), score, times]],
                    value_input_option="RAW",
                )
            else:
                new_rows.append(
                    [
                        key,
                        candidate.event.company_name,
                        run_date.isoformat(),
                        run_date.isoformat(),
                        score,
                        1,
                    ]
                )
        if new_rows:
            worksheet.append_rows(new_rows, value_input_option="RAW")

    def upsert_ats_cache(self, discovered: dict[str, tuple[AtsProvider, str]]) -> None:
        """Persist newly resolved board tokens. Only appends unknown domains."""
        if not discovered:
            return
        worksheet = self._tab(ATS_CACHE)
        known = set(self.read_ats_cache())
        now = datetime.now().isoformat(timespec="seconds")
        rows = [
            [domain, provider, token, now]
            for domain, (provider, token) in discovered.items()
            if domain not in known
        ]
        if rows:
            worksheet.append_rows(rows, value_input_option="RAW")

    def append_run(self, stats: RunStats) -> None:
        self._tab(RUNS).append_rows(
            [
                [
                    stats.started_at.isoformat(timespec="seconds"),
                    stats.finished_at.isoformat(timespec="seconds")
                    if stats.finished_at
                    else "",
                    "yes" if stats.dry_run else "no",
                    stats.articles_fetched,
                    stats.articles_after_dedupe,
                    stats.events_extracted,
                    stats.passed_filters,
                    stats.openings_verified,
                    stats.scored,
                    stats.posted,
                    stats.input_tokens,
                    stats.output_tokens,
                    stats.cache_read_tokens,
                    round(stats.estimated_cost_usd, 4),
                    " | ".join(stats.errors[:10]),
                ]
            ],
            value_input_option="RAW",
        )

    @property
    def url(self) -> str:
        return f"https://docs.google.com/spreadsheets/d/{self.sheet_id}"


# ── row rendering ─────────────────────────────────────────────────────────────


def _shortlist_row(candidate: Candidate, rank: int, run_date: date) -> list[Any]:
    event = candidate.event
    openings = candidate.openings
    score = candidate.score
    outreach = candidate.outreach

    breakdown = ""
    if score:
        breakdown = " | ".join(
            f"{c.id} {c.score:.1f}: {c.reason}" for c in score.criteria
        )

    return [
        run_date.isoformat(),
        rank,
        event.company_name,
        event.company_domain or "",
        candidate.market or "",
        candidate.composite if candidate.composite is not None else "",
        event.round_stage,
        event.amount_usd or "",
        event.announced_date.isoformat() if event.announced_date else "",
        ", ".join(event.investors),
        f"{event.hq_city or ''}, {event.hq_country or ''}".strip(", "),
        event.sector,
        openings.status if openings else "",
        openings.eng_role_count if openings else "",
        openings.total_role_count if openings else "",
        openings.newest_post_date.date().isoformat()
        if openings and openings.newest_post_date
        else "",
        ", ".join(openings.sample_titles) if openings else "",
        (openings.board_url or "") if openings else "",
        score.key_signal if score else "",
        "; ".join(score.risks) if score else "",
        breakdown,
        outreach.subject if outreach else "",
        outreach.body if outreach else "",
        outreach.personalization_hook if outreach else "",
        event.source_url,
    ]


def _parse_date(raw: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
