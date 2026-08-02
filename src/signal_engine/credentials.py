"""Credential preflight.

Every external dependency is checked with a real, cheap round-trip rather than
a "the env var is non-empty" check — an API key that exists but lacks
permissions, or a sheet that was never shared with the service account, fails
identically to a missing key at 2:30am and is far harder to diagnose from a CI
log.

Each check returns a ``CheckResult`` with a fix hint, so `verify-credentials`
tells you what to do rather than just what broke.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import settings


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    skipped: bool = False


def check_anthropic() -> CheckResult:
    name = "Claude API"
    fix = (
        "Create a key at https://console.anthropic.com/settings/keys and set "
        "ANTHROPIC_API_KEY in .env. Note this is a billable API key, separate "
        "from a Claude.ai subscription."
    )
    if not settings().anthropic_api_key:
        return CheckResult(name, False, "ANTHROPIC_API_KEY is not set", fix)
    try:
        from .llm import ping

        model = ping()
    except Exception as exc:  # noqa: BLE001 - report any failure to the user
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}", fix)
    return CheckResult(name, True, f"reachable, responded as {model}")


def check_google_sheets() -> CheckResult:
    name = "Google Sheets"
    fix = (
        "1. https://console.cloud.google.com → create/select a project\n"
        "     2. APIs & Services → Library → enable 'Google Sheets API'\n"
        "     3. Credentials → Create credentials → Service account → "
        "Keys → Add key → JSON\n"
        "     4. Save the JSON as ./service-account.json and set "
        "GOOGLE_SERVICE_ACCOUNT_FILE\n"
        "     5. Create a Google Sheet, set GOOGLE_SHEET_ID from its URL\n"
        "     6. Share that sheet with the service account's client_email "
        "as an Editor"
    )
    cfg = settings()
    if not cfg.google_sheet_id:
        return CheckResult(name, False, "GOOGLE_SHEET_ID is not set", fix)

    try:
        info = cfg.google_credentials_info()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, False, f"credentials JSON is malformed: {exc}", fix)
    if not info:
        return CheckResult(
            name,
            False,
            "no service-account credentials found (set GOOGLE_SERVICE_ACCOUNT_FILE "
            "or GOOGLE_SERVICE_ACCOUNT_JSON)",
            fix,
        )

    email = info.get("client_email", "unknown")
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        sheet = gspread.authorize(creds).open_by_key(cfg.google_sheet_id)
        tabs = [ws.title for ws in sheet.worksheets()]
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name,
            False,
            f"{type(exc).__name__}: {exc}\n     (service account is {email} — "
            f"is the sheet shared with it as an Editor?)",
            fix,
        )
    return CheckResult(
        name, True, f"opened '{sheet.title}' as {email}; tabs: {', '.join(tabs)}"
    )


def check_slack() -> CheckResult:
    name = "Slack"
    fix = (
        "https://api.slack.com/apps → Create New App → From scratch → "
        "Incoming Webhooks → Activate → Add New Webhook to Workspace → pick a "
        "channel → copy the URL into SLACK_WEBHOOK_URL"
    )
    url = settings().slack_webhook_url
    if not url:
        return CheckResult(name, False, "SLACK_WEBHOOK_URL is not set", fix)
    if not url.startswith("https://hooks.slack.com/"):
        return CheckResult(
            name, False, "does not look like an incoming-webhook URL", fix
        )
    try:
        # Slack validates the payload before posting; an intentionally empty
        # one returns `invalid_payload` from a live webhook and `no_service`
        # from a dead one, which distinguishes the two without spamming the
        # channel.
        response = httpx.post(url, content="", timeout=10)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}", fix)

    body = response.text.strip()
    if body == "invalid_payload":
        return CheckResult(name, True, "webhook is live (empty-payload probe accepted)")
    if response.status_code == 200:
        return CheckResult(name, True, "webhook responded 200")
    return CheckResult(name, False, f"HTTP {response.status_code}: {body}", fix)


def check_apollo() -> CheckResult:
    name = "Apollo (optional)"
    fix = (
        "Raw API access requires Apollo's Organization plan (~$119/user/mo). "
        "On Basic or Professional, leave APOLLO_ENABLED=false — the pipeline "
        "does not need it. To check: Apollo → Settings → Integrations → API."
    )
    cfg = settings()
    if not cfg.apollo_enabled:
        return CheckResult(
            name, True, "disabled (pipeline runs fully without it)", skipped=True
        )
    if not cfg.apollo_api_key:
        return CheckResult(
            name, False, "APOLLO_ENABLED=true but APOLLO_API_KEY is not set", fix
        )
    try:
        response = httpx.post(
            "https://api.apollo.io/api/v1/organizations/enrich",
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "x-api-key": cfg.apollo_api_key,
            },
            json={"domain": "anthropic.com"},
            timeout=20,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}", fix)

    if response.status_code == 200:
        return CheckResult(name, True, "enrichment endpoint reachable")
    if response.status_code in (401, 403):
        return CheckResult(
            name,
            False,
            f"HTTP {response.status_code} — key rejected. This is the expected "
            "result on the Basic/Professional plans, which do not grant raw API "
            "access. Set APOLLO_ENABLED=false.",
            fix,
        )
    return CheckResult(name, False, f"HTTP {response.status_code}: {response.text[:200]}", fix)


def check_sender_identity() -> CheckResult:
    name = "Sender identity"
    cfg = settings()
    missing = [
        field
        for field, value in (
            ("SENDER_NAME", cfg.sender_name),
            ("SENDER_TITLE", cfg.sender_title),
        )
        if not value.strip()
    ]
    if missing:
        return CheckResult(
            name,
            False,
            f"missing {', '.join(missing)}",
            "Set these in .env — they sign the drafted outreach emails.",
        )
    return CheckResult(
        name, True, f"{cfg.sender_name}, {cfg.sender_title} @ {cfg.sender_company}"
    )


def run_all() -> list[CheckResult]:
    return [
        check_anthropic(),
        check_google_sheets(),
        check_slack(),
        check_sender_identity(),
        check_apollo(),
    ]
