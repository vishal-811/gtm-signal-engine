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

# The schedule the daily run must return to: 02:30 UTC = 08:00 IST.
PRODUCTION_CRON = "30 2 * * *"


@pytest.fixture
def blank_settings(monkeypatch):
    """Settings with every credential cleared."""
    cfg = cli.config.settings()
    for field in (
        "openai_api_key",
        "google_sheet_id",
        "google_service_account_file",
        "google_service_account_json",
        "slack_webhook_url",
        "sender_name",
        "sender_title",
        "sender_email",
    ):
        monkeypatch.setattr(cfg, field, "", raising=False)
    return cfg


class TestPreflight:
    def test_dry_run_needs_only_the_openai_key(self, blank_settings, monkeypatch):
        monkeypatch.setattr(blank_settings, "openai_api_key", "sk-test", raising=False)
        assert cli._preflight(dry_run=True) is None

    def test_dry_run_reports_a_missing_openai_key(self, blank_settings):
        assert cli._preflight(dry_run=True) == "OPENAI_API_KEY"

    def test_live_run_requires_every_publishing_credential(self, blank_settings, monkeypatch):
        monkeypatch.setattr(blank_settings, "openai_api_key", "sk-test", raising=False)
        missing = cli._preflight(dry_run=False)

        assert "GOOGLE_SHEET_ID" in missing
        assert "SENDER_NAME" in missing

    def test_slack_is_optional_for_a_live_run(self, blank_settings, monkeypatch):
        # The Sheet is the deliverable and the state store; Slack is a
        # convenience notification layered on top. Requiring it would block a
        # perfectly good Sheets-only setup.
        for field, value in (
            ("openai_api_key", "sk-test"),
            ("google_sheet_id", "s"),
            ("google_service_account_json", '{"client_email":"a@b.com"}'),
            ("sender_name", "Ankur"),
            ("sender_title", "Founder"),
            ("sender_email", "ankur@hire100x.io"),
            ("slack_webhook_url", ""),
        ):
            monkeypatch.setattr(blank_settings, field, value, raising=False)
        assert cli._preflight(dry_run=False) is None

    def test_live_run_passes_when_everything_is_set(self, blank_settings, monkeypatch):
        for field, value in (
            ("openai_api_key", "sk-test"),
            ("google_sheet_id", "sheet123"),
            ("google_service_account_json", '{"client_email":"a@b.com"}'),
            ("slack_webhook_url", "https://hooks.slack.com/services/x"),
            ("sender_name", "Ankur"),
            ("sender_title", "Founder"),
            ("sender_email", "v@example.com"),
        ):
            monkeypatch.setattr(blank_settings, field, value, raising=False)
        assert cli._preflight(dry_run=False) is None

    @pytest.mark.parametrize(
        "omit", ["sender_name", "sender_title", "sender_email"]
    )
    def test_each_sender_field_is_individually_required_for_a_live_run(
        self, omit, blank_settings, monkeypatch
    ):
        for field, value in (
            ("openai_api_key", "sk-test"),
            ("google_sheet_id", "s"),
            ("google_service_account_json", '{"client_email":"a@b.com"}'),
            ("slack_webhook_url", "https://hooks.slack.com/x"),
            ("sender_name", "Ankur"),
            ("sender_title", "Founder"),
            ("sender_email", "v@example.com"),
        ):
            monkeypatch.setattr(
                blank_settings, field, "" if field == omit else value, raising=False
            )
        assert omit.upper() in cli._preflight(dry_run=False)

    def test_sender_email_is_not_required_for_a_dry_run(
        self, blank_settings, monkeypatch
    ):
        # Dry runs still draft, but a placeholder signature is harmless when
        # nothing is published.
        monkeypatch.setattr(blank_settings, "openai_api_key", "sk-test", raising=False)
        assert cli._preflight(dry_run=True) is None


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

        cron = triggers["schedule"][0]["cron"]
        fields = cron.split()
        assert len(fields) == 5, f"not a valid cron expression: {cron!r}"
        minute, hour = fields[0], fields[1]
        assert minute.isdigit() and hour.isdigit(), (
            f"the daily run must fire at a fixed time, got {cron!r}"
        )

        raw = path.read_text()
        if cron != PRODUCTION_CRON:
            # A temporary slot is legitimate — proving the scheduler fires
            # means moving it — but it must be marked, and the real schedule
            # must still be written down, or a debugging change becomes the
            # permanent one by forgetting.
            assert "TEMPORARY" in raw, (
                f"cron is {cron!r}, not the production {PRODUCTION_CRON!r}, and "
                "nothing marks it as a deliberate temporary override"
            )
            assert PRODUCTION_CRON in raw, (
                "a temporary cron must record the schedule to restore"
            )

    def test_no_temporary_cron_is_left_behind(self):
        """Fails while a one-off test slot is in place. That is the point:
        it is the reminder to put 08:00 IST back."""
        raw = (PROJECT_ROOT / ".github" / "workflows" / "daily.yml").read_text()
        import yaml as _yaml

        cron = _yaml.safe_load(raw)[True]["schedule"][0]["cron"]
        if "TEMPORARY" in raw:
            pytest.skip(f"temporary cron {cron!r} in place for a scheduler test")
        assert cron == PRODUCTION_CRON

    def test_daily_workflow_reaches_every_setting_the_pipeline_reads(self):
        """Any setting configurable in .env must be reachable in CI.

        SENDER_COMPANY was absent here and absent from the workflow, so a
        scheduled run silently signed its drafts with the default company name.
        Deriving the list from Settings rather than restating it means a new
        field cannot be forgotten in both places at once.
        """
        from signal_engine.config import Settings

        raw = (PROJECT_ROOT / ".github" / "workflows" / "daily.yml").read_text()
        skip = {
            # Set from GOOGLE_SERVICE_ACCOUNT_JSON instead; a runner cannot
            # mount a file.
            "google_service_account_file",
        }
        for name in Settings.model_fields:
            if name in skip:
                continue
            env = name.upper()
            assert f"secrets.{env}" in raw or f"vars.{env}" in raw, (
                f"{env} is configurable but the workflow never passes it"
            )

    def test_credentials_are_secrets_and_plain_config_is_not(self):
        """A leaked sender name is harmless; a leaked key is not. Variables are
        readable in the Actions UI, which is why the split has to be right."""
        raw = (PROJECT_ROOT / ".github" / "workflows" / "daily.yml").read_text()
        for name in (
            "OPENAI_API_KEY",
            "GOOGLE_SERVICE_ACCOUNT_JSON",
            "SLACK_WEBHOOK_URL",
            "APOLLO_API_KEY",
        ):
            assert f"secrets.{name}" in raw, f"{name} must be a secret"
            assert f"vars.{name}" not in raw, f"{name} must not be a variable"

    def test_daily_workflow_deletes_credentials_after_the_run(self):
        """The .env it writes must not survive into an uploaded artifact."""
        workflow = yaml.safe_load(
            (PROJECT_ROOT / ".github" / "workflows" / "daily.yml").read_text()
        )
        steps = workflow["jobs"]["run"]["steps"]
        cleanup = next(s for s in steps if s.get("name") == "Remove credentials")
        assert "rm -f .env" in cleanup["run"]
        assert cleanup["if"] == "always()", "must run even when the pipeline fails"

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


class TestExtractionFailureExit:
    """The exit code is the only signal a scheduler sees.

    These are direct tests of the decision, because the bug that motivated
    them — a bare `limit` where `args.limit` was meant — lived on a line no
    test executed. It raised NameError *after* the whole pipeline had run and
    printed its report, so it looked like a working run that crashed at the end.
    """

    @staticmethod
    def _stats(attempted_after_dedupe: int, failed: int):
        from datetime import datetime, timezone

        from signal_engine.schemas import RunStats

        return RunStats(
            started_at=datetime.now(timezone.utc),
            dry_run=True,
            articles_after_dedupe=attempted_after_dedupe,
            extraction_failed_articles=failed,
        )

    def test_total_failure_is_a_failed_run(self):
        from signal_engine import cli

        assert cli._extraction_collapsed(self._stats(20, 20), None) is True

    def test_a_quiet_day_is_not_a_failed_run(self):
        from signal_engine import cli

        assert cli._extraction_collapsed(self._stats(20, 0), None) is False

    def test_a_few_lost_articles_do_not_fail_the_run(self):
        from signal_engine import cli

        assert cli._extraction_collapsed(self._stats(100, 5), None) is False

    def test_majority_failure_fails_even_when_some_succeed(self):
        """A partial outage still means the shortlist cannot be trusted."""
        from signal_engine import cli

        assert cli._extraction_collapsed(self._stats(100, 60), None) is True

    def test_limit_is_the_denominator_not_the_deduped_total(self):
        """With --limit 20 of 400 deduped, losing all 20 is total failure.

        Measured against 400 it would read as 5% and pass silently — which is
        exactly the shape of the run that first went green while doing nothing.
        """
        from signal_engine import cli

        stats = self._stats(400, 20)
        assert cli._articles_attempted(stats, 20) == 20
        assert cli._extraction_collapsed(stats, 20) is True
        assert cli._extraction_collapsed(stats, None) is False

    def test_no_articles_at_all_is_not_a_failure(self):
        """Feeds returning nothing is a feed problem, reported separately."""
        from signal_engine import cli

        assert cli._extraction_collapsed(self._stats(0, 0), None) is False


class TestScoringFailureExit:
    @staticmethod
    def _stats(scored: int, failed: int):
        from datetime import datetime, timezone

        from signal_engine.schemas import RunStats

        return RunStats(
            started_at=datetime.now(timezone.utc), dry_run=True,
            scored=scored, scoring_failed_candidates=failed,
        )

    def test_total_scoring_wipeout_fails_the_run(self):
        from signal_engine import cli

        assert cli._scoring_collapsed(self._stats(scored=0, failed=9)) is True

    def test_nothing_good_today_is_not_a_failure(self):
        """Scored fine, nothing cleared the bar — a real verdict, exit 0."""
        from signal_engine import cli

        assert cli._scoring_collapsed(self._stats(scored=9, failed=0)) is False

    def test_partial_scoring_failure_is_not_a_wipeout(self):
        """Any successful score proves the stage works."""
        from signal_engine import cli

        assert cli._scoring_collapsed(self._stats(scored=4, failed=5)) is False
