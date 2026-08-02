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

    names = [r[header.index("company")] for r in ws.values[1:]]
    assert removed == 2
    assert names == ["Etched", "Humanoid"], "bystander must survive"
