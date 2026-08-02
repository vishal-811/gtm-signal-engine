"""Credential preflight checks.

Only the pure ones are tested here — the Claude/Sheets/Slack checks make real
network calls by design, which is the whole point of them.
"""

from __future__ import annotations

import pytest

from signal_engine import credentials


@pytest.fixture
def sender(monkeypatch):
    cfg = credentials.settings()
    for field, value in (
        ("sender_name", "Vishal"),
        ("sender_title", "Founder"),
        ("sender_email", "vishal@example.com"),
        ("sender_company", "Hire100x"),
    ):
        monkeypatch.setattr(cfg, field, value, raising=False)
    return cfg


class TestSenderIdentity:
    def test_passes_with_a_complete_identity(self, sender):
        result = credentials.check_sender_identity()
        assert result.ok
        assert "Vishal" in result.detail
        assert "vishal@example.com" in result.detail

    @pytest.mark.parametrize(
        "field,expected",
        [
            ("sender_name", "SENDER_NAME"),
            ("sender_title", "SENDER_TITLE"),
            ("sender_email", "SENDER_EMAIL"),
        ],
    )
    def test_reports_each_missing_field_by_name(
        self, field, expected, sender, monkeypatch
    ):
        monkeypatch.setattr(sender, field, "", raising=False)
        result = credentials.check_sender_identity()

        assert not result.ok
        assert expected in result.detail
        assert result.fix

    def test_whitespace_only_counts_as_missing(self, sender, monkeypatch):
        monkeypatch.setattr(sender, "sender_email", "   ", raising=False)
        assert not credentials.check_sender_identity().ok

    @pytest.mark.parametrize(
        "bad",
        [
            "not-an-email",
            "missing-at-sign.com",
            "no-tld@localhost",
            "has space@example.com",
            "@example.com",          # no local part
            "v@",                    # no domain
            "a@b@c.com",             # two @ signs
            "v@example.",            # empty final label
            "v@.com",                # empty first label
        ],
    )
    def test_rejects_a_malformed_address(self, bad, sender, monkeypatch):
        # This address is signed onto every cold email that goes out, so a
        # typo is worth catching at setup rather than in a founder's inbox.
        monkeypatch.setattr(sender, "sender_email", bad, raising=False)
        result = credentials.check_sender_identity()

        assert not result.ok
        assert "does not look like an email" in result.detail

    @pytest.mark.parametrize(
        "good",
        [
            "vishalssharma811@gmail.com",
            "first.last@hire100x.io",
            "v+tag@sub.domain.co.uk",
        ],
    )
    def test_accepts_real_world_address_shapes(self, good, sender, monkeypatch):
        monkeypatch.setattr(sender, "sender_email", good, raising=False)
        assert credentials.check_sender_identity().ok


class TestApolloCheck:
    def test_disabled_apollo_is_skipped_not_failed(self, monkeypatch):
        # Apollo is optional; a disabled integration must never fail the
        # preflight or block a live run.
        monkeypatch.setattr(
            credentials.settings(), "apollo_enabled", False, raising=False
        )
        result = credentials.check_apollo()

        assert result.ok
        assert result.skipped

    def test_enabled_without_a_key_fails_with_a_fix_hint(self, monkeypatch):
        cfg = credentials.settings()
        monkeypatch.setattr(cfg, "apollo_enabled", True, raising=False)
        monkeypatch.setattr(cfg, "apollo_api_key", "", raising=False)
        result = credentials.check_apollo()

        assert not result.ok
        assert "Organization plan" in result.fix
