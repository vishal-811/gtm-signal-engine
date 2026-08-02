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


def check_openai() -> CheckResult:
    name = "LLM API"
    fix = (
        "Create a key at https://platform.openai.com/api-keys and set "
        "OPENAI_API_KEY in .env. The key also needs credit — check "
        "https://platform.openai.com/settings/organization/billing"
    )
    cfg = settings()
    if not cfg.openai_api_key:
        return CheckResult(name, False, "OPENAI_API_KEY is not set", fix)

    from . import llm

    where = llm.endpoint()

    # Listed first: if OPENAI_MODEL is wrong, the ping fails with a bare 404
    # and the useful information is the set of ids this key can actually use.
    # A gateway may not implement /v1/models at all, which is not fatal.
    models: list[str] | None
    try:
        models = llm.available_models(limit=500)
    except Exception as exc:  # noqa: BLE001
        models = None
        model_note = f"model list unavailable ({type(exc).__name__})"
    else:
        model_note = f"{len(models)} models listed"

    known = {llm.normalize_model_id(m) for m in models} if models is not None else None
    if known is not None and llm.normalize_model_id(cfg.openai_model) not in known:
        preview = ", ".join(models[:12]) or "(none returned)"
        return CheckResult(
            name,
            False,
            f"OPENAI_MODEL={cfg.openai_model!r} is not available at {where}.\n"
            f"     Models you can use: {preview}",
            "Set OPENAI_MODEL in .env to one of the ids listed above.",
        )

    # The capability the whole pipeline rests on. Checked explicitly because an
    # endpoint can pass a plain chat call and still ignore response_format,
    # which would otherwise surface as parse failures on every article.
    ok, detail = llm.probe_structured_output()
    if not ok:
        return CheckResult(
            name,
            False,
            f"reached {where}, but strict structured output failed: {detail}",
            "Every stage needs strict JSON-schema output. If this endpoint does "
            "not support it, switch OPENAI_BASE_URL back to OpenAI direct "
            "(leave it blank) or pick a model on this gateway that does.",
        )

    rates = (
        "cost estimates on"
        if llm.usage.cost_is_estimated
        else "cost estimates off (set OPENAI_*_COST_PER_MTOK)"
    )
    return CheckResult(
        name,
        True,
        f"{where} · model {cfg.openai_model} · {model_note} · {detail} · {rates}",
    )


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
        "Apollo → Settings → Integrations → API → create a key. Whether your "
        "plan allows raw API calls varies and the docs disagree, so this check "
        "just tries it. If the key is rejected, set APOLLO_ENABLED=false — the "
        "pipeline does not need it."
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
            json={"domain": "stripe.com"},
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
            f"HTTP {response.status_code} — key rejected, which usually means "
            "this plan does not grant raw API access. Set APOLLO_ENABLED=false; "
            "the pipeline is fully functional without it.",
            fix,
        )
    return CheckResult(name, False, f"HTTP {response.status_code}: {response.text[:200]}", fix)


def _looks_like_email(value: str) -> bool:
    """Shape check on the sender address.

    Not RFC-5322 validation and not a check that the mailbox exists — just
    enough to catch a typo before it is signed onto every cold email that goes
    out. Requires a non-empty local part, exactly one ``@``, and a domain with
    at least one dot and no empty labels.
    """
    if not value or any(c.isspace() for c in value):
        return False
    parts = value.split("@")
    if len(parts) != 2:
        return False
    local, domain = parts
    if not local or not domain:
        return False
    labels = domain.split(".")
    return len(labels) >= 2 and all(labels)


def check_sender_identity() -> CheckResult:
    name = "Sender identity"
    cfg = settings()
    fix = "Set these in .env — they sign the drafted outreach emails."
    missing = [
        field
        for field, value in (
            ("SENDER_NAME", cfg.sender_name),
            ("SENDER_TITLE", cfg.sender_title),
            ("SENDER_EMAIL", cfg.sender_email),
        )
        if not value.strip()
    ]
    if missing:
        return CheckResult(name, False, f"missing {', '.join(missing)}", fix)

    email = cfg.sender_email.strip()
    if not _looks_like_email(email):
        return CheckResult(
            name,
            False,
            f"SENDER_EMAIL does not look like an email address: {email!r}",
            fix,
        )

    return CheckResult(
        name,
        True,
        f"{cfg.sender_name}, {cfg.sender_title} @ {cfg.sender_company} <{email}>",
    )


def run_all() -> list[CheckResult]:
    return [
        check_openai(),
        check_google_sheets(),
        check_slack(),
        check_sender_identity(),
        check_apollo(),
    ]
