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
import time
from datetime import date, datetime
from typing import Any

from gspread.utils import rowcol_to_a1

from ..config import settings
from ..schemas import AtsProvider, Candidate, RunStats

log = logging.getLogger(__name__)

SHORTLIST = "shortlist"
SEEN = "seen"
ATS_CACHE = "ats_cache"
RUNS = "runs"

# Ordered for reading left-to-right the way the decision is actually made:
# is it worth my time (score, market, roles), what is the pitch (signal, email),
# then the evidence, then the raw audit trail.
SHORTLIST_HEADERS = [
    # decide
    "run_date", "rank", "score", "company", "market", "eng_roles",
    "round_stage", "amount_usd", "sector",
    # act
    "key_signal", "email_subject", "email_body",
    # verify
    "openings_status", "sample_titles", "newest_post", "board_url", "risks",
    # context
    "announced", "investors", "hq", "domain", "total_roles",
    # audit
    "criteria_breakdown", "personalization_hook", "source_url",
]

# Columns holding prose, which need wrapping and a wide-but-bounded width.
_WIDE_COLUMNS = {
    "key_signal": 320, "email_body": 420, "email_subject": 200,
    "sample_titles": 300, "risks": 300, "criteria_breakdown": 400,
    "board_url": 220, "source_url": 220, "company": 150,
}
_NARROW_DEFAULT = 90
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


def _with_retry(operation, *, attempts: int = 3, label: str = "sheets"):
    """Retry a Sheets call through transient network and 5xx failures.

    Observed live: a `Connection reset by peer` mid-run aborted a pipeline that
    had already spent every extraction call. Google's own quota guidance is to
    back off and retry, and these operations are all idempotent enough to be
    safe to repeat.
    """
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            transient = any(
                marker in str(exc).lower()
                for marker in ("connection", "timeout", "reset", "503", "500", "429")
            )
            if not transient or attempt == attempts:
                raise
            log.warning(
                "%s attempt %d/%d failed (%s); retrying in %.0fs",
                label, attempt, attempts, type(exc).__name__, delay,
            )
            time.sleep(delay)
            delay *= 2


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
        # Re-read: `existing` predates the tabs just created, so testing it
        # meant the default sheet was never actually removed.
        titles = {ws.title for ws in self._sheet.worksheets()}
        if "Sheet1" in titles and len(titles) > 1:
            try:
                self._sheet.del_worksheet(self._sheet.worksheet("Sheet1"))
            except Exception:  # noqa: BLE001 - cosmetic only
                pass

        self._tabs_ready = True

        # Style the deliverable the first time it is created. Doing it here
        # rather than per-run means the styling calls cost nothing nightly.
        if SHORTLIST not in existing:
            try:
                self.format_shortlist()
            except Exception:  # noqa: BLE001 - cosmetic; never fail a run for it
                log.warning("could not apply shortlist formatting", exc_info=True)

    # ── presentation ──────────────────────────────────────────────────────────

    def format_shortlist(self) -> None:
        """Make the shortlist tab readable at a glance.

        Applied once, on demand — not on every run, since these are
        spreadsheet-level style calls and re-issuing them each night would
        waste API quota for no change.
        """
        ws = self._tab(SHORTLIST)
        sheet_id = ws.id
        last_col = len(SHORTLIST_HEADERS)

        requests: list[dict[str, Any]] = [
            # Header row and the company column stay visible while scrolling
            # through 25 columns of detail.
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "gridProperties": {"frozenRowCount": 1, "frozenColumnCount": 4},
                    },
                    "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount",
                }
            },
            {
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1},
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.17, "green": 0.24, "blue": 0.31},
                            "textFormat": {
                                "bold": True,
                                "foregroundColor": {"red": 1, "green": 1, "blue": 1},
                            },
                            "verticalAlignment": "MIDDLE",
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
                }
            },
            # Top-align every data row: with wrapped prose in some columns and
            # short values in others, middle alignment looks ragged.
            {
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": 1},
                    "cell": {
                        "userEnteredFormat": {
                            "verticalAlignment": "TOP",
                            "wrapStrategy": "CLIP",
                        }
                    },
                    "fields": "userEnteredFormat(verticalAlignment,wrapStrategy)",
                }
            },
        ]

        for index, name in enumerate(SHORTLIST_HEADERS):
            width = _WIDE_COLUMNS.get(name, _NARROW_DEFAULT)
            requests.append(
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "COLUMNS",
                            "startIndex": index,
                            "endIndex": index + 1,
                        },
                        "properties": {"pixelSize": width},
                        "fields": "pixelSize",
                    }
                }
            )
            if name in _WIDE_COLUMNS and name not in ("board_url", "source_url"):
                requests.append(
                    {
                        "repeatCell": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": 1,
                                "startColumnIndex": index,
                                "endColumnIndex": index + 1,
                            },
                            "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP"}},
                            "fields": "userEnteredFormat.wrapStrategy",
                        }
                    }
                )

        def col(name: str) -> int:
            return SHORTLIST_HEADERS.index(name)

        # Money as money, score to two places — otherwise 200000000 and
        # 3.6500000000000004 both land in the sheet verbatim.
        for name, pattern in (("amount_usd", "$#,##0"), ("score", "0.00")):
            index = col(name)
            requests.append(
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 1,
                            "startColumnIndex": index,
                            "endColumnIndex": index + 1,
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "numberFormat": {"type": "NUMBER", "pattern": pattern}
                            }
                        },
                        "fields": "userEnteredFormat.numberFormat",
                    }
                }
            )

        # Colour the score so the strong leads are findable without reading.
        score_range = {
            "sheetId": sheet_id,
            "startRowIndex": 1,
            "startColumnIndex": col("score"),
            "endColumnIndex": col("score") + 1,
        }
        requests.append(
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [score_range],
                        "gradientRule": {
                            "minpoint": {
                                "color": {"red": 0.96, "green": 0.80, "blue": 0.80},
                                "type": "NUMBER",
                                "value": "2",
                            },
                            "maxpoint": {
                                "color": {"red": 0.72, "green": 0.88, "blue": 0.75},
                                "type": "NUMBER",
                                "value": "5",
                            },
                        },
                    },
                    "index": 0,
                }
            }
        )
        # Grey out rows whose openings could not be verified — present, but
        # visibly weaker evidence than a confirmed board.
        requests.append(
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [
                            {
                                "sheetId": sheet_id,
                                "startRowIndex": 1,
                                "startColumnIndex": 0,
                                "endColumnIndex": last_col,
                            }
                        ],
                        "booleanRule": {
                            "condition": {
                                "type": "CUSTOM_FORMULA",
                                "values": [{"userEnteredValue": '=$M2="unverified"'}],
                            },
                            "format": {
                                "textFormat": {
                                    "foregroundColor": {
                                        "red": 0.5, "green": 0.5, "blue": 0.5
                                    }
                                }
                            },
                        },
                    },
                    "index": 1,
                }
            }
        )

        self._sheet.batch_update({"requests": requests})
        log.info("applied shortlist formatting")

    def migrate_shortlist_columns(self) -> int:
        """Rewrite existing rows into the current column order.

        The header order changed after rows already existed. Rewriting by
        column *name* preserves them; a blind reorder would silently shift
        every value one column left.
        """
        ws = self._tab(SHORTLIST)
        values = ws.get_all_values()
        if not values:
            return 0
        old_headers = values[0]
        if old_headers == SHORTLIST_HEADERS:
            return 0

        records = [dict(zip(old_headers, row)) for row in values[1:]]
        ws.clear()
        rows = [SHORTLIST_HEADERS] + [
            [r.get(name, "") for name in SHORTLIST_HEADERS] for r in records
        ]
        ws.update(values=rows, range_name="A1", value_input_option="RAW")
        log.info("migrated %d shortlist row(s) to the new column order", len(records))
        return len(records)

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

    def append_shortlist(
        self, candidates: list[Candidate], run_date: date
    ) -> tuple[int, int]:
        """Write the shortlist, refreshing rows already present.

        Upsert rather than append. A blind append duplicated any company two
        runs both surfaced — and re-running a backfill duplicated the whole
        sheet. The refreshed row is the useful one anyway: role counts and
        board contents move between runs, so the newer read supersedes.

        Identity is the resolved board URL, falling back to domain and then to
        the normalized company key. Board URL is preferred because outlets
        disagree about domains — Etched arrived as both ``etched.com`` and
        ``etched.ai`` — while one board is unambiguously one employer.

        Returns ``(inserted, updated)``.
        """
        if not candidates:
            return (0, 0)

        worksheet = self._tab(SHORTLIST)
        existing = _with_retry(worksheet.get_all_values, label="read_shortlist")
        header = existing[0] if existing else list(SHORTLIST_HEADERS)
        index = _identity_index(header, existing[1:] if existing else [])

        appended: list[list[Any]] = []
        updates: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            row = _shortlist_row(candidate, rank, run_date)
            hit = next(
                (index[k] for k in _identity_keys(candidate) if k in index), None
            )
            if hit is None:
                appended.append(row)
            else:
                updates.append(
                    {
                        "range": rowcol_to_a1(hit, 1)
                        + ":"
                        + rowcol_to_a1(hit, len(row)),
                        "values": [row],
                    }
                )

        if updates:
            _with_retry(
                lambda: worksheet.batch_update(updates, value_input_option="RAW"),
                label="update_shortlist",
            )
        if appended:
            _with_retry(
                lambda: worksheet.append_rows(appended, value_input_option="RAW"),
                label="append_shortlist",
            )
        log.info(
            "shortlist: %d new, %d refreshed in place", len(appended), len(updates)
        )
        return (len(appended), len(updates))

    def dedupe_shortlist(self) -> int:
        """Drop rows sharing an identity with an earlier row. Returns the count.

        Repairs sheets written before the upsert existed. Deletes one row per
        pass and re-reads in between: a loop that collects row numbers upfront
        and deletes them in sequence shifts every index below the first delete,
        which cost a real row here once.
        """
        removed = 0
        while True:
            rows = _with_retry(
                self._tab(SHORTLIST).get_all_values, label="read_shortlist"
            )
            if len(rows) < 2:
                break
            victim = _first_duplicate_row(rows[0], rows[1:])
            if victim is None:
                break
            self._tab(SHORTLIST).delete_rows(victim)
            removed += 1
        return removed

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


def _location_summary(candidate: Candidate) -> str:
    """Human-readable location for the sheet.

    Prefers the job-board locations, because those are what the geography
    filter actually matched on. Falls back to the article's stated HQ, which
    is usually blank.
    """
    openings = candidate.openings
    if openings and openings.locations:
        return ", ".join(openings.locations[:3])
    event = candidate.event
    parts = [p for p in (event.hq_city, event.hq_country) if p]
    return ", ".join(parts)


def _identity_keys(candidate: Candidate) -> list[str]:
    """Identities this candidate may already be filed under, best first.

    Ordered by trust: a board URL is one employer; a domain is what an outlet
    guessed; a name is a last resort and collides across unrelated startups.
    """
    keys: list[str] = []
    board = candidate.openings.board_url if candidate.openings else None
    if board:
        keys.append(f"board:{board.strip().lower().rstrip('/')}")
    if candidate.event.company_domain:
        keys.append(f"domain:{candidate.event.company_domain.strip().lower()}")
    if candidate.key:
        keys.append(f"key:{candidate.key}")
    return keys


def _row_identity_keys(header: list[str], row: list[str]) -> list[str]:
    """The same identities, recovered from a sheet row."""

    def cell(name: str) -> str:
        try:
            return (row[header.index(name)] or "").strip().lower()
        except (ValueError, IndexError):
            return ""

    keys = []
    if board := cell("board_url"):
        keys.append(f"board:{board.rstrip('/')}")
    if domain := cell("domain"):
        keys.append(f"domain:{domain}")
    if company := cell("company"):
        keys.append(f"key:{company}")
    return keys


def _identity_index(header: list[str], rows: list[list[str]]) -> dict[str, int]:
    """Map every identity to its 1-based sheet row (header occupies row 1)."""
    index: dict[str, int] = {}
    for offset, row in enumerate(rows, start=2):
        for key in _row_identity_keys(header, row):
            index.setdefault(key, offset)
    return index


def _first_duplicate_row(header: list[str], rows: list[list[str]]) -> int | None:
    """Row number of the first row whose identity an earlier row already holds."""
    seen: set[str] = set()
    for offset, row in enumerate(rows, start=2):
        keys = _row_identity_keys(header, row)
        if any(k in seen for k in keys):
            return offset
        seen.update(keys)
    return None


def _shortlist_row(candidate: Candidate, rank: int, run_date: date) -> list[Any]:
    """One shortlist row, keyed by header name so the order can change in one
    place without silently shifting every value."""
    event = candidate.event
    openings = candidate.openings
    score = candidate.score
    outreach = candidate.outreach

    values: dict[str, Any] = {
        "run_date": run_date.isoformat(),
        "rank": rank,
        "score": candidate.composite if candidate.composite is not None else "",
        "company": event.company_name,
        "market": candidate.market or "",
        "eng_roles": openings.eng_role_count if openings else "",
        "round_stage": event.round_stage,
        "amount_usd": event.amount_usd or "",
        "sector": event.sector,
        "key_signal": score.key_signal if score else "",
        "email_subject": outreach.subject if outreach else "",
        "email_body": outreach.body if outreach else "",
        "openings_status": openings.status if openings else "",
        "sample_titles": ", ".join(openings.sample_titles) if openings else "",
        "newest_post": (
            openings.newest_post_date.date().isoformat()
            if openings and openings.newest_post_date
            else ""
        ),
        "board_url": (openings.board_url or "") if openings else "",
        "risks": "; ".join(score.risks) if score else "",
        "announced": event.announced_date.isoformat() if event.announced_date else "",
        "investors": ", ".join(event.investors),
        # Job-board locations where we have them — that is what the market was
        # actually matched on — falling back to whatever the article stated.
        "hq": _location_summary(candidate),
        "domain": event.company_domain or "",
        "total_roles": openings.total_role_count if openings else "",
        "criteria_breakdown": (
            " | ".join(f"{c.id} {c.score:.1f}: {c.reason}" for c in score.criteria)
            if score
            else ""
        ),
        "personalization_hook": outreach.personalization_hook if outreach else "",
        "source_url": event.source_url,
    }
    return [values[name] for name in SHORTLIST_HEADERS]


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
