"""Command-line entry point.

    signal-engine verify-credentials    check every external dependency
    signal-engine show-config           print loaded config without any network
    signal-engine check-feeds           fetch every RSS feed and report health
    signal-engine check-openings        ATS lookup for one company
    signal-engine score-one <url>       score a single article, for rubric tuning
    signal-engine run                   the full pipeline (dry by default)

`run` writes nothing unless given --live. Everything else is read-only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from . import config, credentials
from .schemas import Article, Candidate

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )


# ── verify-credentials ────────────────────────────────────────────────────────


def cmd_verify_credentials(_args: argparse.Namespace) -> int:
    console.print("\n[bold]Checking credentials…[/bold]\n")
    results = credentials.run_all()

    required_failures = 0
    for result in results:
        if result.skipped:
            mark, style = "○", "dim"
        elif result.ok:
            mark, style = "✓", "green"
        else:
            mark, style = "✗", "red"
            if not result.name.endswith("(optional)"):
                required_failures += 1

        console.print(f"[{style}]{mark}[/{style}] [bold]{result.name}[/bold]")
        console.print(f"    {result.detail}")
        if not result.ok and result.fix:
            console.print(f"    [yellow]fix:[/yellow] {result.fix}")
        console.print()

    if required_failures:
        console.print(
            f"[red bold]{required_failures} required check(s) failed.[/red bold] "
            "Fix the items above and re-run.\n"
        )
        return 1

    console.print("[green bold]All required credentials are working.[/green bold]\n")
    return 0


# ── show-config ───────────────────────────────────────────────────────────────


def cmd_show_config(_args: argparse.Namespace) -> int:
    cfg = config.settings()

    console.print("\n[bold]Settings[/bold]")
    settings_table = Table(show_header=False, box=None, padding=(0, 2))
    settings_table.add_row("Sender", f"{cfg.sender_name or '—'}, {cfg.sender_title or '—'}")
    settings_table.add_row("Sender email", cfg.sender_email or "—")
    settings_table.add_row("Company", cfg.sender_company)
    settings_table.add_row("Apollo", "enabled" if cfg.apollo_enabled else "disabled")
    settings_table.add_row("Dedupe window", f"{cfg.dedupe_window_days} days")
    settings_table.add_row("Max event age", f"{cfg.max_event_age_days} days")
    settings_table.add_row("Max article age", f"{cfg.max_article_age_hours} hours")
    settings_table.add_row(
        "Secrets present",
        ", ".join(
            filter(
                None,
                [
                    "openai" if cfg.openai_api_key else "",
                    "sheet-id" if cfg.google_sheet_id else "",
                    "google-creds" if cfg.google_credentials_info() else "",
                    "slack" if cfg.slack_webhook_url else "",
                ],
            )
        )
        or "[red]none[/red]",
    )
    console.print(settings_table)

    feeds = config.feeds_config()
    console.print(
        f"\n[bold]Feeds[/bold]  {len(feeds.active)} enabled "
        f"/ {len(feeds.feeds)} total"
    )
    for feed in feeds.feeds:
        state = "" if feed.enabled else " [dim](disabled)[/dim]"
        console.print(f"    {feed.name}{state}")

    geo = config.geo_config()
    console.print("\n[bold]Markets[/bold]")
    for market in geo.markets:
        console.print(f"    {market.label} — {len(market.aliases)} aliases")

    titles = config.eng_titles_config()
    console.print(
        f"\n[bold]Engineering titles[/bold]  {len(titles.include)} include / "
        f"{len(titles.exclude)} exclude patterns"
    )

    rubric = config.rubric()
    console.print(f"\n[bold]Rubric[/bold]  threshold {rubric.threshold} / 5.0")
    rubric_table = Table(box=None, padding=(0, 2))
    rubric_table.add_column("criterion")
    rubric_table.add_column("weight", justify="right")
    rubric_table.add_column("anchors", justify="right")
    for criterion in rubric.criteria:
        rubric_table.add_row(
            criterion.id, f"{criterion.weight:.0%}", str(len(criterion.anchors))
        )
    console.print(rubric_table)
    console.print()
    return 0


# ── check-feeds ───────────────────────────────────────────────────────────────


def cmd_check_feeds(args: argparse.Namespace) -> int:
    from .sources import rss

    feeds = config.feeds_config()
    console.print(f"\n[bold]Fetching {len(feeds.active)} feeds…[/bold]\n")
    results = rss.fetch_all()

    table = Table(box=None, padding=(0, 2))
    table.add_column("")
    table.add_column("feed")
    table.add_column("items", justify="right")
    table.add_column("newest")
    table.add_column("detail", overflow="fold")

    now = datetime.now(timezone.utc)
    failed = 0
    empty = 0
    for result in sorted(results, key=lambda r: (r.ok, len(r.articles))):
        if not result.ok:
            failed += 1
            table.add_row("[red]✗[/red]", result.feed.name, "—", "—", result.error or "")
            continue

        dated = [a.published_at for a in result.articles if a.published_at]
        if dated:
            age_hours = (now - max(dated)).total_seconds() / 3600
            newest = f"{age_hours:.0f}h ago"
        else:
            newest = "[dim]no dates[/dim]"

        if not result.articles:
            empty += 1
            table.add_row(
                "[yellow]![/yellow]",
                result.feed.name,
                "0",
                newest,
                "parsed but returned nothing — consider disabling",
            )
        else:
            table.add_row(
                "[green]✓[/green]",
                result.feed.name,
                str(len(result.articles)),
                newest,
                "",
            )

    console.print(table)

    raw = [a for r in results for a in r.articles]
    recent = rss.filter_recent(raw)
    unique = rss.dedupe(recent)
    console.print(
        f"\n[bold]{len(raw)}[/bold] items → "
        f"[bold]{len(recent)}[/bold] within "
        f"{config.settings().max_article_age_hours}h → "
        f"[bold]{len(unique)}[/bold] unique after dedupe"
    )
    if failed or empty:
        console.print(
            f"[yellow]{failed} failed, {empty} empty.[/yellow] "
            "Set `enabled: false` in config/feeds.yaml for anything permanently dead."
        )

    if args.show:
        console.print("\n[bold]Sample of deduped articles[/bold]")
        for article in unique[: args.show]:
            when = (
                article.published_at.strftime("%b %d %H:%M")
                if article.published_at
                else "undated"
            )
            console.print(f"  [dim]{when}[/dim]  {article.title}")
            console.print(f"            [dim]{article.source} · {article.url}[/dim]")

    console.print()
    return 1 if failed else 0


# ── check-endpoint ────────────────────────────────────────────────────────────


def cmd_check_endpoint(args: argparse.Namespace) -> int:
    """Try a key and base URL without saving them anywhere.

    Exists so credentials can be tested by trial and error — which is what
    bringing up a new gateway involves — without a bad one landing in .env, and
    without the key going through a shell history or a chat window.
    """
    import getpass

    from . import llm

    key = args.key or getpass.getpass("API key (hidden, then Enter): ").strip()
    if not key:
        console.print("[red]No key given.[/red]")
        return 2

    cfg = config.settings()
    saved = (cfg.openai_api_key, cfg.openai_base_url, cfg.openai_model,
             cfg.openai_user_agent)
    cfg.openai_api_key = key
    cfg.openai_base_url = args.base_url or ""
    if args.user_agent is not None:
        cfg.openai_user_agent = args.user_agent
    if args.model:
        cfg.openai_model = args.model
    llm.reset_client()
    llm.reset_usage()
    llm.reset_effort_probe()

    console.print(f"\n[bold]Endpoint[/bold]  {llm.endpoint()}")
    console.print(f"[bold]Model[/bold]     {cfg.openai_model}\n")

    exit_code = 0
    try:
        # 1. Can we reach it and enumerate models?
        try:
            models = llm.available_models(limit=1000)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]✗[/red] model list: {type(exc).__name__}: {str(exc)[:200]}")
            models = None
            exit_code = 1
        else:
            console.print(f"[green]✓[/green] model list: {len(models)} models reachable")
            if args.list_models:
                for m in models:
                    console.print(f"      {m}")
            elif models:
                console.print(f"      e.g. {', '.join(models[:8])}")

        # 2. Is the configured model actually one of them?
        if models is not None:
            known = {llm.normalize_model_id(m) for m in models}
            if llm.normalize_model_id(cfg.openai_model) in known:
                console.print(f"[green]✓[/green] model '{cfg.openai_model}' is available")
            else:
                console.print(f"[red]✗[/red] model '{cfg.openai_model}' NOT in the list")
                near = [m for m in models if cfg.openai_model.lower() in m.lower()]
                if near:
                    console.print(f"      closest matches: {', '.join(near[:8])}")
                console.print("      re-run with --list-models to see everything")
                exit_code = 1

        # 3. The capability the pipeline actually depends on.
        ok, detail = llm.probe_structured_output()
        if ok:
            console.print(f"[green]✓[/green] strict structured output: {detail}")
        else:
            console.print(f"[red]✗[/red] strict structured output: {detail}")
            console.print(
                "      Every stage needs this. An endpoint that fails here "
                "cannot run the pipeline."
            )
            exit_code = 1

        if llm.usage.calls:
            console.print(
                f"\n[dim]structured-output mode: {llm.structured_mode()} · "
                f"{llm.usage.summary()}[/dim]"
            )

        if exit_code == 0:
            console.print(
                "\n[green bold]All checks passed.[/green bold] Put these in .env:"
            )
            console.print(f"  OPENAI_BASE_URL={args.base_url or ''}")
            console.print(f"  OPENAI_MODEL={cfg.openai_model}")
            console.print("  OPENAI_API_KEY=<the key you just entered>")
            if cfg.openai_user_agent:
                console.print(f"  OPENAI_USER_AGENT={cfg.openai_user_agent}")
            if llm.structured_mode() == "prompt":
                console.print(
                    "  LLM_STRUCTURED_MODE=prompt"
                    "   [dim]# endpoint ignores response_format[/dim]"
                )
            console.print()
        else:
            console.print("\n[yellow]Nothing was written to .env.[/yellow]\n")
    finally:
        (cfg.openai_api_key, cfg.openai_base_url, cfg.openai_model,
         cfg.openai_user_agent) = saved
        llm.reset_client()

    return exit_code


# ── check-openings ────────────────────────────────────────────────────────────


def cmd_check_openings(args: argparse.Namespace) -> int:
    from . import openings

    result = openings.check(args.company, args.domain)
    console.print(f"\n[bold]{args.company}[/bold]")

    colour = {"verified": "green", "none_found": "yellow", "unverified": "red"}[
        result.status
    ]
    console.print(f"  status:    [{colour}]{result.status}[/{colour}]")
    console.print(f"  provider:  {result.ats_provider or '—'}")
    console.print(f"  board:     {result.board_url or '—'}")
    console.print(
        f"  eng roles: {result.eng_role_count} of {result.total_role_count} total"
    )
    if result.newest_post_date:
        console.print(f"  newest:    {result.newest_post_date.date()}")
    for title in result.sample_titles:
        console.print(f"    · {title}")
    console.print()
    return 0


# ── score-one ─────────────────────────────────────────────────────────────────


def cmd_score_one(args: argparse.Namespace) -> int:
    """Run one article end-to-end. The loop for tuning rubric.yaml."""
    from . import draft, extract, filters, llm, openings, score

    article = Article(
        title=args.title or args.url,
        url=args.url,
        source="manual",
        published_at=datetime.now(timezone.utc),
        summary=args.text or "",
    )
    if not args.text:
        console.print("[dim]Fetching article text…[/dim]")
        article.summary = _fetch_article_text(args.url)

    console.print("[dim]Extracting…[/dim]")
    events = extract.extract([article])
    if not events:
        console.print("[red]Extraction returned nothing.[/red]")
        return 1

    event = events[0]
    if not event.is_funding_announcement:
        console.print(
            f"[yellow]Not a funding announcement[/yellow] "
            f"(company guess: {event.company_name})"
        )
        return 0

    market = filters.match_market(event.hq_city, event.hq_country)
    candidate = Candidate(event=event, market=market)

    console.print(
        f"  {event.company_name} · {event.round_stage} · "
        f"{'$' + format(event.amount_usd, ',') if event.amount_usd else 'undisclosed'} · "
        f"{event.hq_city or '?'} → market: {market or '[red]none[/red]'}"
    )

    console.print("[dim]Checking openings…[/dim]")
    candidate.openings = openings.check(event.company_name, event.company_domain)
    console.print(
        f"  {candidate.openings.status} · {candidate.openings.eng_role_count} eng roles"
    )

    console.print("[dim]Scoring…[/dim]")
    score.score_one(candidate)
    if not candidate.score:
        console.print("[red]Scoring failed.[/red]")
        return 1

    rubric = config.rubric()
    weights = {c.id: c.weight for c in rubric.criteria}
    table = Table(box=None, padding=(0, 2))
    table.add_column("criterion")
    table.add_column("score", justify="right")
    table.add_column("weight", justify="right")
    table.add_column("contribution", justify="right")
    table.add_column("reason", overflow="fold")
    for criterion in candidate.score.criteria:
        weight = weights.get(criterion.id, 0.0)
        table.add_row(
            criterion.id,
            f"{criterion.score:.1f}",
            f"{weight:.0%}",
            f"{criterion.score * weight:.2f}",
            criterion.reason,
        )
    console.print()
    console.print(table)

    passed = candidate.composite is not None and candidate.composite > rubric.threshold
    verdict = "[green]ABOVE[/green]" if passed else "[red]below[/red]"
    console.print(
        f"\n  composite [bold]{candidate.composite:.2f}[/bold] / 5.0  "
        f"({verdict} threshold {rubric.threshold})"
    )
    console.print(f"  key signal: {candidate.score.key_signal}")
    if candidate.score.risks:
        console.print(f"  risks: {'; '.join(candidate.score.risks)}")

    if passed and args.with_draft:
        console.print("\n[dim]Drafting outreach…[/dim]")
        draft.draft_one(candidate)
        if candidate.outreach:
            console.print(f"\n  [bold]Subject:[/bold] {candidate.outreach.subject}")
            console.print(f"  [dim]hook: {candidate.outreach.personalization_hook}[/dim]\n")
            for line in candidate.outreach.body.splitlines():
                console.print(f"  {line}")

    console.print(f"\n[dim]{llm.usage.summary()}[/dim]\n")
    return 0


def _fetch_article_text(url: str, limit: int = 4000) -> str:
    import re

    import httpx

    try:
        response = httpx.get(
            url,
            timeout=25,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SignalEngine/0.1)"},
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Could not fetch the article ({exc}); "
                      "scoring on the URL alone.[/yellow]")
        return ""
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", response.text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


# ── run ───────────────────────────────────────────────────────────────────────


def _preflight(dry_run: bool) -> str | None:
    """Check the credentials this run will actually need.

    Fails before the ingest stage rather than after: without this, a missing
    key surfaces only once every feed has been fetched and every article
    discarded, which reads like a quiet news day instead of a config error.
    """
    cfg = config.settings()
    missing: list[str] = []
    if not cfg.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not dry_run:
        if not cfg.google_sheet_id:
            missing.append("GOOGLE_SHEET_ID")
        if not cfg.google_credentials_info():
            missing.append("GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON")
        # Slack is optional — the Sheet is the deliverable.
        for var, value in (
            ("SENDER_NAME", cfg.sender_name),
            ("SENDER_TITLE", cfg.sender_title),
            ("SENDER_EMAIL", cfg.sender_email),
        ):
            if not value.strip():
                missing.append(var)
    return ", ".join(missing) if missing else None


def cmd_run(args: argparse.Namespace) -> int:
    from . import pipeline
    from .sinks import slack

    dry_run = not args.live

    missing = _preflight(dry_run)
    if missing:
        console.print(
            f"\n[red]Cannot start:[/red] missing {missing}.\n"
            "Run [bold]signal-engine verify-credentials[/bold] for setup steps.\n"
        )
        return 2
    if dry_run:
        console.print(
            "\n[bold yellow]DRY RUN[/bold yellow] — nothing will be written to "
            "Sheets or Slack. Pass --live to publish.\n"
        )
    else:
        console.print("\n[bold red]LIVE RUN[/bold red] — writing to Sheets and Slack.\n")

    try:
        result = pipeline.run(
            dry_run=dry_run, limit=args.limit, skip_openings=args.skip_openings
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[red]Run failed:[/red] {exc}\n")
        if not dry_run:
            # Otherwise a crashed cron is indistinguishable from a quiet day.
            slack.send_failure_alert(f"{type(exc).__name__}: {exc}")
        raise

    stats = result.stats
    console.print("[bold]Funnel[/bold]")
    console.print(
        f"  {stats.articles_fetched} fetched → {stats.articles_after_dedupe} unique "
        f"→ {stats.events_extracted} extracted → {stats.passed_filters} in-market "
        f"→ {stats.scored} scored → [bold]{stats.posted} posted[/bold]"
    )
    if result.filter_report:
        console.print(f"  [dim]{result.filter_report.summary()}[/dim]")

    if result.all_scored:
        rubric = config.rubric()
        table = Table(box=None, padding=(0, 2))
        table.add_column("", justify="right")
        table.add_column("company")
        table.add_column("score", justify="right")
        table.add_column("round")
        table.add_column("eng", justify="right")
        table.add_column("market")
        table.add_column("key signal", overflow="fold")
        for rank, candidate in enumerate(result.all_scored, start=1):
            passed = (
                candidate.composite is not None
                and candidate.composite > rubric.threshold
            )
            style = "" if passed else "dim"
            openings = candidate.openings
            eng = str(openings.eng_role_count) if openings else "—"
            if openings and openings.status == "unverified":
                eng = "?"
            table.add_row(
                str(rank),
                f"[{style}]{candidate.event.company_name}[/{style}]" if style else candidate.event.company_name,
                f"{candidate.composite:.2f}" if candidate.composite is not None else "—",
                candidate.event.round_stage,
                eng,
                candidate.market or "—",
                candidate.score.key_signal if candidate.score else "",
            )
        console.print()
        console.print(table)

    if args.show_drafts:
        for candidate in result.posted:
            if not candidate.outreach:
                continue
            console.print(f"\n[bold]{candidate.event.company_name}[/bold]")
            console.print(f"  Subject: {candidate.outreach.subject}")
            for line in candidate.outreach.body.splitlines():
                console.print(f"  {line}")

    console.print(f"\n[dim]{stats.input_tokens:,} in / {stats.output_tokens:,} out · "
                  f"cache read {stats.cache_read_tokens:,} · "
                  f"est. ${stats.estimated_cost_usd:.3f}[/dim]")
    if stats.errors:
        console.print(f"[yellow]{len(stats.errors)} error(s):[/yellow]")
        for err in stats.errors[:10]:
            console.print(f"  · {err}")
    if result.sheet_url:
        console.print(f"[dim]{result.sheet_url}[/dim]")
    if not dry_run and not config.settings().slack_webhook_url:
        console.print(
            "[dim]Slack not configured — digest skipped, shortlist is in the "
            "Sheet.[/dim]"
        )
    console.print()
    return 0


# ── entry point ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-engine",
        description="Automated GTM intelligence pipeline for Hire100x.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify-credentials",
        help="check that OpenAI, Google Sheets, and Slack are all reachable",
    )
    verify.set_defaults(func=cmd_verify_credentials)

    show = subparsers.add_parser(
        "show-config", help="print loaded configuration (no network calls)"
    )
    show.set_defaults(func=cmd_show_config)

    check = subparsers.add_parser(
        "check-feeds", help="fetch every enabled RSS feed and report its health"
    )
    check.add_argument(
        "--show",
        type=int,
        default=0,
        metavar="N",
        help="also print the N most recent deduped article titles",
    )
    check.set_defaults(func=cmd_check_feeds)

    endpoint_cmd = subparsers.add_parser(
        "check-endpoint",
        help="try an API key / base URL without saving them (for bringing up a gateway)",
    )
    endpoint_cmd.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help="OpenAI-compatible endpoint, e.g. https://agentrouter.org/v1. "
        "Omit for OpenAI direct.",
    )
    endpoint_cmd.add_argument(
        "--model", default=None, help="model id to test (default: OPENAI_MODEL)"
    )
    endpoint_cmd.add_argument(
        "--key",
        default=None,
        help="API key. Omit to be prompted with hidden input, which keeps it "
        "out of your shell history.",
    )
    endpoint_cmd.add_argument(
        "--user-agent",
        default=None,
        metavar="UA",
        help="override the client User-Agent. Some gateways gate on it and "
        "reject unrecognised clients.",
    )
    endpoint_cmd.add_argument(
        "--list-models", action="store_true", help="print every available model id"
    )
    endpoint_cmd.set_defaults(func=cmd_check_endpoint)

    openings_cmd = subparsers.add_parser(
        "check-openings", help="run the ATS openings check for one company"
    )
    openings_cmd.add_argument("company", help="company name")
    openings_cmd.add_argument(
        "--domain", default=None, help="company domain, e.g. acme.com"
    )
    openings_cmd.set_defaults(func=cmd_check_openings)

    score_cmd = subparsers.add_parser(
        "score-one",
        help="score a single article end-to-end (the loop for tuning rubric.yaml)",
    )
    score_cmd.add_argument("url", help="article URL")
    score_cmd.add_argument("--title", default=None, help="override the article title")
    score_cmd.add_argument(
        "--text", default=None, help="paste the article text instead of fetching it"
    )
    score_cmd.add_argument(
        "--with-draft",
        action="store_true",
        help="also draft the outreach email if it clears the threshold",
    )
    score_cmd.set_defaults(func=cmd_score_one)

    run_cmd = subparsers.add_parser(
        "run", help="run the full pipeline (dry by default — use --live to publish)"
    )
    # Dry run is the default. `--dry-run` is accepted explicitly so the safe
    # invocation can be spelled out in scripts and docs rather than relying on
    # the reader knowing what the default is.
    mode = run_cmd.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help="actually write to Sheets and post to Slack",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="write nothing (the default; accepted for explicitness)",
    )
    run_cmd.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="process at most N articles (fast iteration)",
    )
    run_cmd.add_argument(
        "--skip-openings",
        action="store_true",
        help="skip ATS verification (much faster; scores will be weaker)",
    )
    run_cmd.add_argument(
        "--show-drafts", action="store_true", help="print the drafted emails"
    )
    run_cmd.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        console.print(f"\n[red]Config error:[/red] {exc}\n")
        return 2
    except ValueError as exc:
        console.print(f"\n[red]Invalid config:[/red] {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
