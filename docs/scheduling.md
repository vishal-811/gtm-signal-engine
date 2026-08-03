# Scheduling the daily run

## Why this document exists

GitHub Actions' `schedule` trigger did not fire for this repository. Two slots
were tested on 2026-08-03, one with 6 minutes of registration lead and one with
28:

| Trigger             | Fired    |
| ------------------- | -------- |
| `schedule`          | 0 of 2   |
| `workflow_dispatch` | 5 of 5, immediately |

The configuration was verified correct each time — cron on the default branch,
workflow `active`, valid expression, secrets present — and the identical
workflow succeeded on every manual trigger. GitHub documents no SLA for
scheduled workflows and deprioritises them on low-activity repositories.

The `schedule` block is still in `daily.yml`. It costs nothing to leave, and it
may well start working as the repository accumulates history. Treat it as a
bonus rather than the mechanism.

## The fix: keep the workflow, replace the trigger

`workflow_dispatch` is reliable here. Any external scheduler that can make one
authenticated HTTP request can drive it, and GitHub still supplies the compute,
the secrets and the logs.

### The request

```
POST https://api.github.com/repos/vishal-811/gtm-signal-engine/actions/workflows/daily.yml/dispatches
Accept:        application/vnd.github+json
Authorization: Bearer <TOKEN>
Content-Type:  application/json

{"ref": "main", "inputs": {"live": "true"}}
```

A `204 No Content` means accepted. Note `live` is the string `"true"`, not a
boolean — `workflow_dispatch` inputs are always strings over the API.

### The token

Create a **fine-grained** personal access token at
<https://github.com/settings/personal-access-tokens/new>:

- Repository access: **Only select repositories** → `gtm-signal-engine`
- Permissions: **Actions → Read and write** (this alone is enough)
- Expiration: set a real one and diary the renewal

Do not use a classic token. A classic `repo` token can read and write every
repository on the account, and this one is going into a third-party form.

## Option 1 — cron-job.org (simplest)

Free, no card required.

1. Sign up at <https://cron-job.org>.
2. **Create cronjob**, URL as above, method **POST**.
3. Add the three headers and the JSON body from the request above.
4. Schedule: `02:30` UTC daily — set the account timezone first, or use `08:00`
   with the timezone set to Asia/Kolkata.
5. Save, then use **Test run** and confirm a run appears under
   <https://github.com/vishal-811/gtm-signal-engine/actions>.

## Option 2 — Cloudflare Workers (token stays in infrastructure you control)

Free tier, no card. Cron Triggers are included.

`wrangler.toml`:

```toml
name = "signal-engine-cron"
main = "src/index.js"
compatibility_date = "2026-01-01"

[triggers]
crons = ["30 2 * * *"]
```

`src/index.js`:

```js
export default {
  async scheduled(event, env, ctx) {
    const res = await fetch(
      "https://api.github.com/repos/vishal-811/gtm-signal-engine/actions/workflows/daily.yml/dispatches",
      {
        method: "POST",
        headers: {
          Accept: "application/vnd.github+json",
          Authorization: `Bearer ${env.GH_TOKEN}`,
          "Content-Type": "application/json",
          // GitHub rejects requests without a User-Agent.
          "User-Agent": "signal-engine-cron",
        },
        body: JSON.stringify({ ref: "main", inputs: { live: "true" } }),
      },
    );
    // Surfaces in `wrangler tail`; a silent failure here looks exactly like a
    // quiet news day, which is the failure mode this whole document exists to
    // avoid.
    console.log(`dispatch -> ${res.status}`);
    if (!res.ok) throw new Error(`dispatch failed: ${res.status} ${await res.text()}`);
  },
};
```

Then:

```
wrangler secret put GH_TOKEN
wrangler deploy
```

## Option 3 — Google Cloud Scheduler

Three jobs free permanently and the most reliable of the three, but Cloud
Scheduler requires a billing account attached to the project even on free tier.
Only worth it if that is already true.

## Confirming it works

The run log is the health dashboard. After the first scheduled trigger:

```
gh run list -R vishal-811/gtm-signal-engine --workflow="Daily signal run" --limit 5
```

An externally dispatched run shows `workflow_dispatch`, not `schedule` — that
is expected and is the whole point.

The `runs` tab of the sheet records volumes, errors and cost per run. A day with
zero extracted events, or a spike in errors, means a feed broke or a provider
changed its response shape.
