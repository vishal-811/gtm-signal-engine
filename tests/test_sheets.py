"""Sheets sink: row identity, upsert, and duplicate repair."""

from datetime import date

from signal_engine.sinks import sheets




class _FakeWS:
    """Minimal worksheet: records batch_update/append_rows, applies deletes."""

    def __init__(self, values):
        self.values = [list(r) for r in values]
        self.appended: list[list] = []
        self.updates: list[dict] = []

    def get_all_values(self):
        return [list(r) for r in self.values]

    def append_rows(self, rows, value_input_option=None):
        self.appended.extend(rows)
        self.values.extend(list(r) for r in rows)

    def batch_update(self, updates, value_input_option=None):
        self.updates.extend(updates)

    def delete_rows(self, n):
        del self.values[n - 1]


def _client_with(ws, monkeypatch):
    c = sheets.SheetsClient.__new__(sheets.SheetsClient)
    monkeypatch.setattr(c, "_tab", lambda name: ws, raising=False)
    return c


def _cand(name, domain, board, score=4.0):
    from signal_engine.schemas import Candidate, FundingEvent, OpeningsResult
    ev = FundingEvent(
        is_funding_announcement=True, company_name=name, company_domain=domain,
        round_stage="seed", amount_usd=1, investors=[], sector="ai",
        one_line_description="d", source_url="https://n", extraction_confidence=0.9,
    )
    c = Candidate(event=ev)
    c.openings = OpeningsResult(status="verified", board_url=board, eng_role_count=2)
    c.composite = score
    return c


def test_shortlist_refreshes_instead_of_duplicating(monkeypatch):
    """Re-running must update the existing row, never append a second one.

    A blind append duplicated every company that two runs both surfaced, and
    re-running a 14-day backfill duplicated the entire sheet.
    """
    header = list(sheets.SHORTLIST_HEADERS)
    existing = sheets._shortlist_row(
        _cand("Etched", "etched.com", "https://jobs.ashbyhq.com/etched"),
        1, date(2026, 8, 2),
    )
    ws = _FakeWS([header, existing])
    client = _client_with(ws, monkeypatch)

    inserted, refreshed = client.append_shortlist(
        [_cand("Etched", "etched.com", "https://jobs.ashbyhq.com/etched", score=4.5)],
        date(2026, 8, 3),
    )
    assert (inserted, refreshed) == (0, 1)
    assert ws.appended == []


def test_shortlist_matches_on_board_despite_a_different_domain(monkeypatch):
    """etched.ai and etched.com are the same employer — same board."""
    header = list(sheets.SHORTLIST_HEADERS)
    ws = _FakeWS([header, sheets._shortlist_row(
        _cand("Etched", "etched.com", "https://jobs.ashbyhq.com/etched"),
        1, date(2026, 8, 2))])
    client = _client_with(ws, monkeypatch)

    inserted, refreshed = client.append_shortlist(
        [_cand("Etched", "etched.ai", "https://jobs.ashbyhq.com/etched")],
        date(2026, 8, 3),
    )
    assert (inserted, refreshed) == (0, 1), "board URL should win over domain"


def test_shortlist_appends_genuinely_new_companies(monkeypatch):
    header = list(sheets.SHORTLIST_HEADERS)
    ws = _FakeWS([header, sheets._shortlist_row(
        _cand("Etched", "etched.com", "https://b/etched"), 1, date(2026, 8, 2))])
    client = _client_with(ws, monkeypatch)

    inserted, refreshed = client.append_shortlist(
        [_cand("Meshy", "meshy.ai", "https://b/meshy")], date(2026, 8, 3)
    )
    assert (inserted, refreshed) == (1, 0)


def test_dedupe_shortlist_survives_shifting_row_numbers(monkeypatch):
    """Three copies must collapse to one without taking out a bystander.

    Regression: collecting row numbers upfront and deleting them in sequence
    shifts every index below the first delete. That deleted an innocent row
    from the live sheet.
    """
    header = list(sheets.SHORTLIST_HEADERS)
    rows = [header]
    for dom in ("etched.com", "", "etched.ai"):
        rows.append(sheets._shortlist_row(
            _cand("Etched", dom, "https://b/etched"), 1, date(2026, 8, 2)))
    rows.insert(2, sheets._shortlist_row(
        _cand("Humanoid", "humanoid.ai", "https://b/humanoid"), 2, date(2026, 8, 2)))
    ws = _FakeWS(rows)
    client = _client_with(ws, monkeypatch)

    removed = client.dedupe_shortlist()

    names = [r[header.index("Company")] for r in ws.values[1:]]
    assert removed == 2
    assert names == ["Etched", "Humanoid"], "bystander must survive"


def test_every_row_field_has_a_column_and_vice_versa():
    """_shortlist_row is keyed by internal name; a typo would KeyError live."""
    row = sheets._shortlist_row(
        _cand("X", "x.com", "https://b/x"), 1, date(2026, 8, 3)
    )
    assert len(row) == len(sheets._COLUMNS) == len(sheets.SHORTLIST_HEADERS)
    assert len({c.key for c in sheets._COLUMNS}) == len(sheets._COLUMNS)
    assert len(set(sheets.SHORTLIST_HEADERS)) == len(sheets.SHORTLIST_HEADERS)


def test_migration_restores_numbers_lost_to_string_round_trip():
    """get_all_values() stringifies everything; writing that back RAW left
    migrated rows as text beside numeric ones — visibly ragged in the sheet."""
    money = sheets._BY_KEY["amount_usd"]
    score = sheets._BY_KEY["score"]
    roles = sheets._BY_KEY["eng_roles"]

    assert sheets._coerce(money, "$6,000,000") == 6_000_000
    assert sheets._coerce(money, "200000000") == 200_000_000
    assert sheets._coerce(score, "4.40") == 4.40
    assert sheets._coerce(roles, "12") == 12
    # Blank and unparseable must survive, not become 0 or vanish.
    assert sheets._coerce(money, "") == ""
    assert sheets._coerce(money, "undisclosed") == "undisclosed"
    # Prose is never touched, even when it happens to look numeric.
    assert sheets._coerce(sheets._BY_KEY["company"], "23andMe") == "23andMe"


def test_two_distinct_date_columns_are_present_and_labelled():
    """The single ambiguous run_date read as if the backfill covered one day."""
    labels = sheets.SHORTLIST_HEADERS
    assert "Funded On" in labels and "Scanned On" in labels
    assert abs(labels.index("Funded On") - labels.index("Scanned On")) == 1
    assert sheets._BY_KEY["announced"].label == "Funded On"
    assert sheets._BY_KEY["run_date"].label == "Scanned On"


def test_audit_columns_are_hidden_and_decision_columns_are_not():
    hidden = {c.label for c in sheets._COLUMNS if c.hidden}
    assert {"Score Breakdown", "Hook", "Rank"} <= hidden
    for must_show in ("Company", "Score", "Funded On", "Key Signal", "Amount"):
        assert not sheets._BY_KEY[
            next(c.key for c in sheets._COLUMNS if c.label == must_show)
        ].hidden


def test_only_one_prose_column_wraps():
    """Wrapping every prose column stretched rows past 300px."""
    assert sheets._WRAPPED == {"key_signal"}
    assert sheets._ROW_HEIGHT <= 80


def test_migration_maps_legacy_snake_case_headers(monkeypatch):
    """A sheet written before the rename must survive the migration."""
    legacy = ["run_date", "score", "company", "amount_usd", "announced"]
    ws = _FakeWS([legacy, ["2026-08-02", "3.65", "Simile", "200000000", "2026-08-02"]])
    ws.cleared = False
    ws.clear = lambda: setattr(ws, "cleared", True)
    written = {}
    ws.update = lambda values, range_name, value_input_option: written.update(
        rows=values
    )
    client = _client_with(ws, monkeypatch)

    assert client.migrate_shortlist_columns() == 1
    rows = written["rows"]
    assert rows[0] == sheets.SHORTLIST_HEADERS
    row = dict(zip(rows[0], rows[1]))
    assert row["Company"] == "Simile"
    assert row["Amount"] == 200_000_000, "must be numeric, not text"
    assert row["Score"] == 3.65
    assert row["Funded On"] == "2026-08-02"
    assert row["Scanned On"] == "2026-08-02"


def test_sort_is_by_score_descending(monkeypatch):
    """Appends land at the bottom whatever they score; the sheet must re-sort."""
    ws = _FakeWS([list(sheets.SHORTLIST_HEADERS)])
    ws.id = 7
    client = _client_with(ws, monkeypatch)
    captured = {}
    client._sheet = type("S", (), {"batch_update": lambda _s, body: captured.update(body)})()

    client.sort_shortlist()

    spec = captured["requests"][0]["sortRange"]["sortSpecs"][0]
    assert spec["sortOrder"] == "DESCENDING"
    assert spec["dimensionIndex"] == sheets.SHORTLIST_HEADERS.index("Score")
