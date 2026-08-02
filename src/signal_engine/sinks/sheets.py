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
import re
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _Column:
    """One shortlist column.

    ``key`` is the stable internal name rows are built with; ``label`` is what
    the sheet shows. Keeping them apart means a column can be renamed for
    readability without touching row construction, and the migration can map
    an old sheet onto the new layout by key.
    """

    key: str
    label: str
    width: int
    kind: str = "text"  # text | prose | number | money | score | date
    hidden: bool = False


# Ordered the way the decision is actually made: is this worth my time, what do
# I say, what backs it up, then the audit trail. The last group is hidden — it
# is there to be checked when a score looks wrong, not read every morning.
_COLUMNS: list[_Column] = [
    # ── decide ────────────────────────────────────────────────────────────────
    _Column("company", "Company", 160),
    _Column("score", "Score", 65, "score"),
    _Column("market", "Market", 110),
    _Column("eng_roles", "Eng Roles", 80, "number"),
    _Column("round_stage", "Stage", 85),
    _Column("amount_usd", "Amount", 115, "money"),
    # The two dates sit together and say plainly which is which: one is when
    # the company raised, the other is when this pipeline last looked. A single
    # ambiguous "run_date" column read as though the backfill had only covered
    # its own run day.
    _Column("announced", "Funded On", 100, "date"),
    _Column("run_date", "Scanned On", 100, "date"),
    # ── act ───────────────────────────────────────────────────────────────────
    _Column("key_signal", "Key Signal", 340, "prose"),
    _Column("email_subject", "Email Subject", 230, "prose"),
    _Column("email_body", "Email Body", 320, "prose"),
    # ── verify ────────────────────────────────────────────────────────────────
    _Column("openings_status", "Board", 100),
    _Column("sample_titles", "Sample Roles", 280, "prose"),
    _Column("newest_post", "Latest Post", 100, "date"),
    _Column("board_url", "Board URL", 210),
    # ── context ───────────────────────────────────────────────────────────────
    _Column("sector", "Sector", 140),
    _Column("risks", "Risks", 280, "prose"),
    _Column("investors", "Investors", 210, "prose"),
    _Column("hq", "Locations", 210, "prose"),
    _Column("source_url", "Source", 210),
    # ── audit (hidden) ────────────────────────────────────────────────────────
    _Column("domain", "Domain", 150, hidden=True),
    _Column("total_roles", "All Roles", 80, "number", hidden=True),
    _Column("criteria_breakdown", "Score Breakdown", 420, "prose", hidden=True),
    _Column("personalization_hook", "Hook", 260, "prose", hidden=True),
    # Rank is per-run, so it repeats across runs and means nothing once rows
    # from several runs share a sheet. Sorting by Score is the real ranking.
    _Column("rank", "Rank", 60, "number", hidden=True),
]

SHORTLIST_HEADERS = [c.label for c in _COLUMNS]
_BY_KEY = {c.key: c for c in _COLUMNS}
_NUMERIC_KINDS = {"number", "money", "score"}
# One scannable prose column wraps; the rest clip. Wrapping every prose column
# pushed rows past 300px, so only two fit on screen.
_WRAPPED = {"key_signal"}
_ROW_HEIGHT = 60
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
        """Make the shortlist readable at a glance.

        Applied on demand, not per run — these are style calls and re-issuing
        them nightly would burn quota for no change.

        The layout goal is a table you can scan, not a wall of prose. Rows are
        pinned to a fixed height because wrapped text had stretched them past
        300px, leaving two companies visible on a screen.
        """
        ws = self._tab(SHORTLIST)
        sheet_id = ws.id
        last_col = len(_COLUMNS)
        idx = {c.key: i for i, c in enumerate(_COLUMNS)}

        def data_range(col: int | None = None) -> dict[str, Any]:
            rng: dict[str, Any] = {"sheetId": sheet_id, "startRowIndex": 1}
            if col is not None:
                rng["startColumnIndex"] = col
                rng["endColumnIndex"] = col + 1
            return rng

        requests: list[dict[str, Any]] = [
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "gridProperties": {
                            "frozenRowCount": 1,
                            "frozenColumnCount": 2,
                        },
                    },
                    "fields": (
                        "gridProperties.frozenRowCount,"
                        "gridProperties.frozenColumnCount"
                    ),
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {
                                "red": 0.13, "green": 0.19, "blue": 0.26
                            },
                            "textFormat": {
                                "bold": True,
                                "fontSize": 10,
                                "foregroundColor": {"red": 1, "green": 1, "blue": 1},
                            },
                            "verticalAlignment": "MIDDLE",
                            "horizontalAlignment": "LEFT",
                            "wrapStrategy": "CLIP",
                        }
                    },
                    "fields": (
                        "userEnteredFormat(backgroundColor,textFormat,"
                        "verticalAlignment,horizontalAlignment,wrapStrategy)"
                    ),
                }
            },
            {
                "repeatCell": {
                    "range": data_range(),
                    "cell": {
                        "userEnteredFormat": {
                            "verticalAlignment": "TOP",
                            "wrapStrategy": "CLIP",
                            "textFormat": {"fontSize": 10},
                        }
                    },
                    "fields": (
                        "userEnteredFormat(verticalAlignment,wrapStrategy,"
                        "textFormat)"
                    ),
                }
            },
            # A fixed height is what actually makes the sheet scannable. Without
            # it a single long key signal decides how tall its row is.
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "ROWS",
                        "startIndex": 1,
                    },
                    "properties": {"pixelSize": _ROW_HEIGHT},
                    "fields": "pixelSize",
                }
            },
            {"setBasicFilter": {"filter": {"range": {"sheetId": sheet_id}}}},
        ]

        for i, column in enumerate(_COLUMNS):
            requests.append(
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "COLUMNS",
                            "startIndex": i,
                            "endIndex": i + 1,
                        },
                        "properties": {
                            "pixelSize": column.width,
                            "hiddenByUser": column.hidden,
                        },
                        "fields": "pixelSize,hiddenByUser",
                    }
                }
            )
            if column.key in _WRAPPED:
                requests.append(
                    {
                        "repeatCell": {
                            "range": data_range(i),
                            "cell": {
                                "userEnteredFormat": {"wrapStrategy": "WRAP"}
                            },
                            "fields": "userEnteredFormat.wrapStrategy",
                        }
                    }
                )
            pattern = {
                "money": "$#,##0",
                "score": "0.00",
                "number": "0",
            }.get(column.kind)
            if pattern:
                requests.append(
                    {
                        "repeatCell": {
                            "range": data_range(i),
                            "cell": {
                                "userEnteredFormat": {
                                    "numberFormat": {
                                        "type": "NUMBER",
                                        "pattern": pattern,
                                    },
                                    "horizontalAlignment": "RIGHT",
                                }
                            },
                            "fields": (
                                "userEnteredFormat(numberFormat,"
                                "horizontalAlignment)"
                            ),
                        }
                    }
                )
            elif column.kind == "date":
                requests.append(
                    {
                        "repeatCell": {
                            "range": data_range(i),
                            "cell": {
                                "userEnteredFormat": {
                                    "numberFormat": {
                                        "type": "DATE",
                                        "pattern": "yyyy-mm-dd",
                                    }
                                }
                            },
                            "fields": "userEnteredFormat.numberFormat",
                        }
                    }
                )

        # Clear any rules from an earlier layout before adding these, or the
        # old ones keep colouring whatever column now sits at their index.
        for _ in range(6):
            requests.append(
                {"deleteConditionalFormatRule": {"sheetId": sheet_id, "index": 0}}
            )

        requests.append(
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [data_range(idx["score"])],
                        "gradientRule": {
                            "minpoint": {
                                "color": {
                                    "red": 0.98, "green": 0.85, "blue": 0.83
                                },
                                "type": "NUMBER",
                                "value": "2",
                            },
                            "midpoint": {
                                "color": {
                                    "red": 1.0, "green": 0.95, "blue": 0.80
                                },
                                "type": "NUMBER",
                                "value": "3.6",
                            },
                            "maxpoint": {
                                "color": {
                                    "red": 0.78, "green": 0.91, "blue": 0.79
                                },
                                "type": "NUMBER",
                                "value": "4.6",
                            },
                        },
                    },
                    "index": 0,
                }
            }
        )
        # Derive the column letter rather than hardcoding it. This formula used
        # to say $M2 and would have silently graded the wrong column the moment
        # the layout changed.
        status_a1 = re.sub(r"\d+", "", rowcol_to_a1(2, idx["openings_status"] + 1))
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
                                "values": [
                                    {
                                        "userEnteredValue": (
                                            f'=${status_a1}2="unverified"'
                                        )
                                    }
                                ],
                            },
                            "format": {
                                "textFormat": {
                                    "foregroundColor": {
                                        "red": 0.55, "green": 0.55, "blue": 0.55
                                    }
                                }
                            },
                        },
                    },
                    "index": 1,
                }
            }
        )

        # Deleting rules that do not exist is an error, so the clears above are
        # sent in their own best-effort pass.
        clears = [r for r in requests if "deleteConditionalFormatRule" in r]
        rest = [r for r in requests if "deleteConditionalFormatRule" not in r]
        for clear in clears:
            try:
                self._sheet.batch_update({"requests": [clear]})
            except Exception:  # noqa: BLE001 - nothing left to delete
                break
        self._sheet.batch_update({"requests": rest})
        self._apply_banding(sheet_id, last_col)
        log.info("applied shortlist formatting")

    def _apply_banding(self, sheet_id: int, last_col: int) -> None:
        """Alternating row shading. Re-adding over an existing band errors, so
        this is separate and best-effort rather than part of the main batch."""
        try:
            self._sheet.batch_update(
                {
                    "requests": [
                        {
                            "addBanding": {
                                "bandedRange": {
                                    "range": {
                                        "sheetId": sheet_id,
                                        "startRowIndex": 1,
                                        "startColumnIndex": 0,
                                        "endColumnIndex": last_col,
                                    },
                                    "rowProperties": {
                                        "firstBandColor": {
                                            "red": 1, "green": 1, "blue": 1
                                        },
                                        "secondBandColor": {
                                            "red": 0.97,
                                            "green": 0.975,
                                            "blue": 0.98,
                                        },
                                    },
                                }
                            }
                        }
                    ]
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("banding already present: %s", exc)

    def migrate_shortlist_columns(self) -> int:
        """Rewrite existing rows into the current column order and labels.

        Rewrites by column *identity*, not position — a blind reorder would
        shift every value one column left. Rows are matched on either the
        current label or the legacy snake_case name, so a sheet written before
        the rename still lands correctly.

        Numbers are restored on the way through. ``get_all_values`` returns
        every cell as a string, and writing those back RAW left the migrated
        rows as text while freshly appended rows stayed numeric — visible in
        the sheet as one row of left-aligned ``200000000`` among properly
        formatted currency.
        """
        ws = self._tab(SHORTLIST)
        values = _with_retry(ws.get_all_values, label="read_shortlist")
        if not values:
            return 0
        old_headers = values[0]
        if old_headers == SHORTLIST_HEADERS:
            return 0

        records = [dict(zip(old_headers, row)) for row in values[1:]]
        rows = [SHORTLIST_HEADERS]
        for record in records:
            row = []
            for column in _COLUMNS:
                raw = record.get(column.label)
                if raw in (None, ""):
                    raw = record.get(column.key, "")
                row.append(_coerce(column, raw))
            rows.append(row)

        ws.clear()
        _with_retry(
            lambda: ws.update(values=rows, range_name="A1", value_input_option="RAW"),
            label="migrate_shortlist",
        )
        log.info("migrated %d shortlist row(s) to the new column layout", len(records))
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

    def sort_shortlist(self) -> None:
        """Order the sheet best-first.

        Run after every write. Upserting refreshes rows in place and appends
        new ones at the bottom, so without this the ordering degrades into
        "whenever we first saw it" — and the top of the sheet stops being the
        part worth reading.
        """
        ws = self._tab(SHORTLIST)
        _with_retry(
            lambda: self._sheet.batch_update(
                {
                    "requests": [
                        {
                            "sortRange": {
                                "range": {
                                    "sheetId": ws.id,
                                    "startRowIndex": 1,
                                    "startColumnIndex": 0,
                                    "endColumnIndex": len(_COLUMNS),
                                },
                                "sortSpecs": [
                                    {
                                        "dimensionIndex": SHORTLIST_HEADERS.index(
                                            "Score"
                                        ),
                                        "sortOrder": "DESCENDING",
                                    }
                                ],
                            }
                        }
                    ]
                }
            ),
            label="sort_shortlist",
        )

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


def _coerce(column: _Column, raw: Any) -> Any:
    """Restore a cell's type after a round-trip through ``get_all_values``.

    Everything comes back as a string, so a currency column reads ``"$6,000,000"``
    and would be written back as text — right-alignment and number formatting
    silently lost. Anything unparseable is passed through untouched rather than
    blanked; a value we cannot read is still worth keeping.
    """
    if raw is None:
        return ""
    if column.kind not in _NUMERIC_KINDS or not isinstance(raw, str):
        return raw
    cleaned = raw.strip().replace("$", "").replace(",", "")
    if not cleaned:
        return ""
    try:
        return float(cleaned) if column.kind == "score" else int(float(cleaned))
    except ValueError:
        return raw


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

    def cell(key: str) -> str:
        """Look up by key, tolerating both the current label and the legacy
        snake_case header so identity still resolves mid-migration."""
        for name in (_BY_KEY[key].label, key):
            try:
                return (row[header.index(name)] or "").strip().lower()
            except (ValueError, IndexError):
                continue
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
    return [values[column.key] for column in _COLUMNS]


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
