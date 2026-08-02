# Signal Engine

Automated GTM intelligence pipeline for Hire100x.

Every morning it pulls the last 24 hours of startup funding news, scores each
company against the fit rubric in `rubric.yaml` using the Claude API, verifies
the company has live engineering openings, and posts a ranked shortlist with
drafted cold emails to Google Sheets and Slack.

```
RSS feeds ──▶ Extract (Claude) ──▶ Filter (geo/recency/dedupe) ──▶ Enrich (Apollo, optional)
                                                                            │
                                                                            ▼
Slack + Sheets ◀── Draft (Claude) ◀── Score (Claude + rubric) ◀── Verify openings (ATS APIs)
```

**This codebase never sends email.** Drafts land in Sheets and Slack for you to
copy, edit, and send yourself. There is no SMTP client, no ESP integration, and
no send function anywhere in the repo.

---

## Setup

### 1. Install

```bash
cd signal-engine
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e .
```

Everything below assumes `.venv/bin/signal-engine`. Activate the venv
(`source .venv/bin/activate`) if you'd rather type `signal-engine`.

### 2. Configure

```bash
cp .env.example .env
```

Fill in the four required credentials below, then run
`signal-engine verify-credentials`. It does a real round-trip against each
service and tells you exactly what to fix — you never discover a broken
credential at 2:30am in a CI log.

#### Claude API key (required)

1. https://console.anthropic.com/settings/keys → **Create Key**
2. Set `ANTHROPIC_API_KEY` in `.env`

A billable API key, **not** the same as a Claude.ai subscription. Expect roughly
$1–3/day at typical volume; every run's actual cost is logged to the `runs`
sheet tab so you can tune on real numbers.

#### Google Sheets (required)

1. https://console.cloud.google.com → create or select a project
2. **APIs & Services → Library** → "Google Sheets API" → **Enable**
3. **Credentials → Create credentials → Service account** → name it → **Done**
4. Click the service account → **Keys → Add key → Create new key → JSON**
5. Save it as `service-account.json` in the repo root (gitignored) and set
   `GOOGLE_SERVICE_ACCOUNT_FILE=./service-account.json`
6. Create a Google Sheet. Copy its ID from the URL —
   `docs.google.com/spreadsheets/d/`**`THIS_PART`**`/edit` — into `GOOGLE_SHEET_ID`
7. **Share that sheet** with the `client_email` from the JSON, as an **Editor**.
   This is the step people forget; without it every write fails.

The four tabs are created automatically on first live run.

#### Slack (required)

1. https://api.slack.com/apps → **Create New App → From scratch**
2. **Incoming Webhooks** → **Activate**
3. **Add New Webhook to Workspace** → pick a channel → **Allow**
4. Copy the URL into `SLACK_WEBHOOK_URL`

#### Sender identity (required)

These sign the drafted emails:

```
SENDER_NAME=Your Name
SENDER_TITLE=Founder
SENDER_EMAIL=you@example.com
SENDER_COMPANY=Hire100x
```

`SENDER_EMAIL` is the address you will actually send from. The pipeline never
sends anything itself — it writes the drafts so they read and sign correctly
when you paste them into that account. `verify-credentials` shape-checks the
address, because a typo here would be signed onto every cold email that goes
out.

#### Apollo (optional)

Raw Apollo API access is gated to their **Organization** plan (~$119/user/mo).
On Basic or Professional you can only integrate via Zapier/Make/CRM connectors.

Check at **Apollo → Settings → Integrations → API**. If you can create a key,
set `APOLLO_ENABLED=true` and `APOLLO_API_KEY` for firmographic enrichment
(headcount, HQ, industry, tech stack) that sharpens scoring. If not, leave it
off — the pipeline is fully functional without it, and the ATS check is the
load-bearing hiring signal, not Apollo. If the key is ever rejected mid-run,
enrichment disables itself for that run and the pipeline continues.

### 3. Verify

```bash
signal-engine verify-credentials
```

Green across the board before running anything live.

---

## Usage

```bash
# Read-only, no credentials needed
signal-engine show-config                    # loaded config, no network
signal-engine check-feeds --show 10          # feed health + sample articles
signal-engine check-openings Vercel --domain vercel.com

# Needs the Claude key
signal-engine score-one <article-url> --with-draft   # tune the rubric
signal-engine run --dry-run                          # full pipeline, writes nothing
signal-engine run --dry-run --limit 20               # fast iteration

# Needs everything
signal-engine run --live                     # writes to Sheets + posts to Slack
```

`run` is **dry by default**. Publishing requires an explicit `--live`, so no
accidental invocation ever posts to a real channel.

Useful flags: `--limit N` (cap articles), `--skip-openings` (much faster, weaker
scores), `--show-drafts` (print the emails), `-v` (debug logging).

### Tuning the rubric

`score-one` is the loop. Pick an article for a company you already have an
opinion about, run it, and compare:

```bash
signal-engine score-one https://techcrunch.com/... --with-draft
```

It prints every criterion, its score, its weight, its contribution to the
composite, and the model's one-line reason — so you can see exactly which
criterion is miscalibrated, edit `rubric.yaml`, and re-run.

---

## Configuration

Everything tunable is data, not code.

| File | Controls |
|---|---|
| `rubric.yaml` | Fit criteria, weights, score anchors, posting threshold |
| `config/feeds.yaml` | RSS sources. Add, remove, or `enabled: false` |
| `config/geo.yaml` | City/region aliases for the three target markets |
| `config/eng_titles.yaml` | Which job titles count as engineering |
| `prompts/*.md` | System prompt for each Claude stage |
| `.env` | Secrets and runtime tuning |

`rubric.yaml` weights must sum to exactly 1.0 — the app refuses to start
otherwise rather than silently mis-scoring.

### Two config gotchas worth knowing

**Google News RSS sorts by relevance, not date.** Without a `when:2d` suffix on
the query, those feeds return 100 stale items each and nothing from the last
24 hours. Every Google News URL in `feeds.yaml` carries it. Don't remove it.

**`eng_titles.yaml` `include` holds role nouns, not disciplines.** Listing
"platform" or "infrastructure" directly matches things like "Account Executive,
Strategic Platform Partnerships". Every real engineering title already contains
a role noun, so the adjectives add false positives and no recall.

---

## How the sheet works

The Google Sheet is both the deliverable and the state store — no database, and
nothing is committed back to git.

| Tab | Purpose |
|---|---|
| `shortlist` | The deliverable. One row per company per run, including the full email draft and the per-criterion score breakdown. |
| `seen` | Dedupe ledger. Suppresses a company for `DEDUPE_WINDOW_DAYS` (default 30) after it's posted. |
| `ats_cache` | domain → job-board token, so board discovery runs once per company ever, not daily. |
| `runs` | Health dashboard: volumes at each funnel stage, errors, tokens, cost. |

All four tabs are human-editable. Deleting a row from `seen` un-suppresses that
company on the next run.

---

## Scheduling

`.github/workflows/daily.yml` runs at **02:30 UTC (08:00 IST)**. Change the one
cron line to re-time it.

Repository secrets required:

| Secret | Notes |
|---|---|
| `ANTHROPIC_API_KEY` | |
| `GOOGLE_SHEET_ID` | |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | The **full contents** of the JSON file — Actions can't mount a file |
| `SLACK_WEBHOOK_URL` | |
| `SENDER_NAME`, `SENDER_TITLE`, `SENDER_EMAIL` | |
| `APOLLO_API_KEY` | Only if you set the `APOLLO_ENABLED` repo variable to `true` |

Scheduled runs are always live. The **Run workflow** button defaults to a dry
run; tick `live` to publish. If the workflow fails before the pipeline can
report for itself — install failure, timeout, cancellation — a separate step
posts to Slack, so a broken cron is never mistaken for a quiet day.

---

## Development

```bash
.venv/bin/pytest              # 294 tests, no network, no API calls, no cost
```

The suite runs entirely on recorded fixtures: real RSS XML, and real Greenhouse,
Lever, and Ashby responses captured from live boards. Claude calls are stubbed —
what's tested is the logic around them (batching, the source-URL backfill,
failure isolation, score arithmetic, the word ceiling), which is where the
correctness risk actually lives.

### Design notes

**Score arithmetic is Python, not Claude.** The model scores each criterion 0–5
with a justification; the weighted composite is computed in `score.composite()`.
Arithmetic from a language model is unauditable and occasionally wrong. A
criterion the model omits counts as 0 rather than being dropped from the
denominator — otherwise a model could raise its own score by skipping the
criterion it scores worst on.

**Unverified ≠ no openings.** A company with no discoverable job board is
reported `unverified` and kept in the shortlist with a visible flag. Plenty of
seed-stage teams hire off a Notion page; a missing board is absence of evidence,
not evidence of absence. It's scored lower than a verified board, but not
dropped.

**Effort is set per stage.** Extraction runs at `low` (mechanical, high volume),
scoring at `high` (the judgment call the pipeline exists for), drafting at
`medium`. Prompt caching is asserted, not assumed — if the cache never engages,
`llm.py` logs a warning rather than letting the bill quietly multiply.

**One broken feed never stops a run.** Every feed is fetched in isolation and
its failure recorded in the `runs` tab. The same applies to ATS boards, one
extraction batch, one company's scoring, and one draft.

### Known gaps

- **Workable is not supported.** Its widget endpoint returned 404 for every
  token tried *and* for a control, so the response shape couldn't be verified.
  Shipping an unverified client would produce silent `unverified` results that
  look like "no board" rather than "broken code". Add it when a real Workable
  board is available to test against.
- **The Batch API is not used.** It would halve token cost and suits a daily
  cadence, at the price of a polling loop. Worth adding once real volume is
  known — the `runs` tab will tell you whether it's worth it.
