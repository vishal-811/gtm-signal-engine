"""CLI wiring and the credential preflight.

The preflight matters more than it looks: without it a missing key surfaces
only after every feed has been fetched and every article silently discarded,
which reads like a quiet news day rather than a config error.
"""

from __future__ import annotations

import pytest
import yaml

from signal_engine import cli
from signal_engine.config import PROJECT_ROOT


@pytest.fixture
def blank_settings(monkeypatch):
    """Settings with every credential cleared."""
    cfg = cli.config.settings()
    for field in (
        "anthropic_api_key",
        "google_sheet_id",
        "google_service_account_file",
        "google_service_account_json",
        "slack_webhook_url",
        "sender_name",
        "sender_title",
    ):
        monkeypatch.setattr(cfg, field, "", raising=False)
    return cfg


class TestPreflight:
    def test_dry_run_needs_only_the_claude_key(self, blank_settings, monkeypatch):
        monkeypatch.setattr(blank_settings, "anthropic_api_key", "sk-test", raising=False)
        assert cli._preflight(dry_run=True) is None

    def test_dry_run_reports_a_missing_claude_key(self, blank_settings):
        assert cli._preflight(dry_run=True) == "ANTHROPIC_API_KEY"

    def test_live_run_requires_every_publishing_credential(self, blank_settings, monkeypatch):
        monkeypatch.setattr(blank_settings, "anthropic_api_key", "sk-test", raising=False)
        missing = cli._preflight(dry_run=False)

        assert "GOOGLE_SHEET_ID" in missing
        assert "SLACK_WEBHOOK_URL" in missing
        assert "SENDER_NAME" in missing

    def test_live_run_passes_when_everything_is_set(self, blank_settings, monkeypatch):
        for field, value in (
            ("anthropic_api_key", "sk-test"),
            ("google_sheet_id", "sheet123"),
            ("google_service_account_json", '{"client_email":"a@b.com"}'),
            ("slack_webhook_url", "https://hooks.slack.com/services/x"),
            ("sender_name", "Aarif"),
            ("sender_title", "Founder"),
        ):
            monkeypatch.setattr(blank_settings, field, value, raising=False)
        assert cli._preflight(dry_run=False) is None

    def test_a_sender_name_without_a_title_is_still_incomplete(
        self, blank_settings, monkeypatch
    ):
        for field, value in (
            ("anthropic_api_key", "sk-test"),
            ("google_sheet_id", "s"),
            ("google_service_account_json", '{"client_email":"a@b.com"}'),
            ("slack_webhook_url", "https://hooks.slack.com/x"),
            ("sender_name", "Aarif"),
        ):
            monkeypatch.setattr(blank_settings, field, value, raising=False)
        assert "SENDER_NAME and SENDER_TITLE" in cli._preflight(dry_run=False)


class TestParser:
    def test_run_defaults_to_dry(self):
        args = cli.build_parser().parse_args(["run"])
        assert args.live is False

    def test_dry_run_flag_is_accepted_explicitly(self):
        # Documented in the README; must not be an unrecognized-argument error.
        args = cli.build_parser().parse_args(["run", "--dry-run"])
        assert args.live is False

    def test_live_and_dry_run_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["run", "--live", "--dry-run"])

    def test_run_accepts_its_documented_flags(self):
        args = cli.build_parser().parse_args(
            ["run", "--live", "--limit", "5", "--skip-openings", "--show-drafts"]
        )
        assert args.live and args.limit == 5
        assert args.skip_openings and args.show_drafts

    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    @pytest.mark.parametrize(
        "command",
        ["verify-credentials", "show-config", "check-feeds", "run"],
    )
    def test_every_subcommand_is_wired_to_a_handler(self, command):
        args = cli.build_parser().parse_args([command])
        assert callable(args.func)

    def test_score_one_requires_a_url(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["score-one"])

    def test_check_openings_requires_a_company(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["check-openings"])


class TestWorkflows:
    """A malformed workflow fails silently — GitHub just never runs it."""

    @pytest.mark.parametrize("name", ["daily.yml", "tests.yml"])
    def test_workflow_yaml_parses(self, name):
        path = PROJECT_ROOT / ".github" / "workflows" / name
        assert path.exists(), f"missing workflow {name}"
        assert yaml.safe_load(path.read_text())

    def test_daily_workflow_is_scheduled_and_manually_triggerable(self):
        path = PROJECT_ROOT / ".github" / "workflows" / "daily.yml"
        # PyYAML parses a bare `on:` key as the boolean True.
        triggers = yaml.safe_load(path.read_text())[True]

        assert "schedule" in triggers
        assert "workflow_dispatch" in triggers
        assert triggers["schedule"][0]["cron"] == "30 2 * * *"

    def test_daily_workflow_passes_every_required_secret(self):
        raw = (PROJECT_ROOT / ".github" / "workflows" / "daily.yml").read_text()
        for secret in (
            "ANTHROPIC_API_KEY",
            "GOOGLE_SHEET_ID",
            "GOOGLE_SERVICE_ACCOUNT_JSON",
            "SLACK_WEBHOOK_URL",
            "SENDER_NAME",
            "SENDER_TITLE",
        ):
            assert f"secrets.{secret}" in raw, f"workflow never passes {secret}"

    def test_daily_workflow_notifies_on_failure(self):
        # Otherwise a broken cron is indistinguishable from a quiet day.
        raw = (PROJECT_ROOT / ".github" / "workflows" / "daily.yml").read_text()
        assert "if: failure()" in raw

    def test_daily_workflow_serializes_runs(self):
        # Overlapping runs would double-post and race on the `seen` ledger.
        workflow = yaml.safe_load(
            (PROJECT_ROOT / ".github" / "workflows" / "daily.yml").read_text()
        )
        assert workflow["concurrency"]["group"]
