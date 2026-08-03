#!/usr/bin/env python3
"""Create (and verify) the cron-job.org job that triggers the daily run.

GitHub's own `schedule` trigger did not fire for this repository — see
docs/scheduling.md. `workflow_dispatch` is reliable, so an external scheduler
drives it and GitHub still supplies the compute, secrets and logs.

Usage:

    CRONJOB_API_KEY=...  GH_DISPATCH_TOKEN=...  python scripts/setup_cronjob_org.py
    # add --test to also create a throwaway job a few minutes out, confirm the
    # workflow actually starts, and delete it again

Neither secret is written anywhere. The GitHub token is sent to cron-job.org
once, as the header it must replay; it is stored there, not here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

API = "https://api.cron-job.org"
REPO = "vishal-811/gtm-signal-engine"
WORKFLOW = "daily.yml"
DISPATCH_URL = (
    f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW}/dispatches"
)
# 08:00 Asia/Kolkata. Set as a real timezone rather than as 02:30 UTC so the
# run stays at 08:00 local if India ever shifts its offset.
RUN_TIMEZONE = "Asia/Kolkata"
RUN_HOUR, RUN_MINUTE = 8, 0


def call(method: str, path: str, key: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = response.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:400]
        raise SystemExit(
            f"cron-job.org {method} {path} failed: HTTP {exc.code}\n  {detail}"
        ) from exc


def job_body(gh_token: str, hours: list[int], minutes: list[int], title: str) -> dict:
    return {
        "job": {
            "url": DISPATCH_URL,
            "enabled": True,
            "title": title,
            "saveResponses": True,  # so a 401 is visible in the history
            "requestMethod": 1,  # POST
            "extendedData": {
                "headers": {
                    "Authorization": f"Bearer {gh_token}",
                    "Accept": "application/vnd.github+json",
                    "Content-Type": "application/json",
                    # GitHub rejects API requests without a User-Agent.
                    "User-Agent": "signal-engine-cron",
                },
                # `live` must be the string "true". workflow_dispatch inputs
                # are always strings over the API; a JSON boolean is rejected.
                "body": json.dumps({"ref": "main", "inputs": {"live": "true"}}),
            },
            "schedule": {
                "timezone": RUN_TIMEZONE,
                "hours": hours,
                "minutes": minutes,
                "mdays": [-1],
                "months": [-1],
                "wdays": [-1],
            },
        }
    }


def scheduled_run_count() -> int:
    out = subprocess.run(
        ["gh", "run", "list", "-R", REPO, "--workflow", "Daily signal run",
         "--limit", "30", "--json", "databaseId"],
        capture_output=True, text=True, check=True,
    )
    return len(json.loads(out.stdout))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="also prove the chain end to end with a throwaway job")
    args = parser.parse_args()

    key = os.environ.get("CRONJOB_API_KEY", "").strip()
    gh_token = os.environ.get("GH_DISPATCH_TOKEN", "").strip()
    if not key or not gh_token:
        print("Set CRONJOB_API_KEY and GH_DISPATCH_TOKEN.", file=sys.stderr)
        return 2

    print(f"target: {DISPATCH_URL}")
    print(f"schedule: {RUN_HOUR:02d}:{RUN_MINUTE:02d} {RUN_TIMEZONE}\n")

    existing = call("GET", "/jobs", key).get("jobs", [])
    mine = [j for j in existing if DISPATCH_URL in (j.get("url") or "")]
    if mine:
        print(f"already configured: job {mine[0]['jobId']} — leaving it alone")
    else:
        created = call("PUT", "/jobs", key,
                       job_body(gh_token, [RUN_HOUR], [RUN_MINUTE],
                                "Signal Engine — daily 08:00 IST"))
        print(f"created daily job: {created.get('jobId')}")

    if not args.test:
        return 0

    # cron-job.org has no manual-trigger endpoint, so proving the chain means
    # scheduling a throwaway a few minutes out and watching GitHub for the run.
    before = scheduled_run_count()
    fire = datetime.now(timezone.utc) + timedelta(minutes=4)
    local = fire.astimezone()
    print(f"\ntest job will fire at {fire:%H:%M} UTC; runs before = {before}")
    probe = call("PUT", "/jobs", key,
                 job_body(gh_token, [fire.hour], [fire.minute],
                          "Signal Engine — one-off connectivity test"))
    probe_id = probe.get("jobId")
    # UTC hour/minute against a Kolkata-scheduled job would fire 5.5h late.
    call("PATCH", f"/jobs/{probe_id}", key,
         {"job": {"schedule": {"timezone": "UTC", "hours": [fire.hour],
                               "minutes": [fire.minute], "mdays": [-1],
                               "months": [-1], "wdays": [-1]}}})
    print(f"created test job {probe_id}; waiting up to 9 minutes")

    fired = False
    try:
        for _ in range(54):
            time.sleep(10)
            if scheduled_run_count() > before:
                print(f"\nWORKFLOW STARTED at {datetime.now(timezone.utc):%H:%M:%S} UTC")
                fired = True
                break
        else:
            print("\nno new run appeared — check the job history at "
                  "https://console.cron-job.org for the HTTP status returned")
    finally:
        call("DELETE", f"/jobs/{probe_id}", key)
        print(f"deleted test job {probe_id}")

    return 0 if fired else 1


if __name__ == "__main__":
    raise SystemExit(main())
