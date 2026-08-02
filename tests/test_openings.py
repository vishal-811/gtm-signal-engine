"""Openings verification is the load-bearing hiring signal — it drives 30% of
the composite score. The title classifier in particular is where precision was
lost during development, so the misclassifications found against live boards
are pinned here as regression tests.

Fixtures are real responses recorded from Vercel (Greenhouse), Match Group
(Lever), and Ashby (Ashby), trimmed to a deliberate mix of engineering and
non-engineering roles.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from signal_engine import openings

FIXTURES = Path(__file__).parent / "fixtures"
GREENHOUSE = json.loads((FIXTURES / "greenhouse_jobs.json").read_text())
LEVER = json.loads((FIXTURES / "lever_jobs.json").read_text())
ASHBY = json.loads((FIXTURES / "ashby_jobs.json").read_text())


class TestIsEngineeringTitle:
    @pytest.mark.parametrize(
        "title",
        [
            "Software Engineer",
            "Senior Software Engineer",
            "Android Engineer III",
            "Back-end Engineer, Platform",
            "Backend Engineer (Trust & Safety), Platform",
            "Founding Full Stack Engineer, AI Incubation",
            "Full Stack Developer",
            "iOS Developer",
            "AI Engineer",
            "Machine Learning Engineer",
            "DevOps Engineer",
            "Site Reliability Engineer",
            "SRE II",
            "Software Development Engineer II",
            # Senior IC ladders must NOT be caught by the leadership exclusions.
            "Staff Engineer, Infrastructure",
            "Principal Engineer",
            # AI labs use this as their IC engineer title; it has no role noun.
            "Member of Technical Staff",
            # Research *Engineer* is an IC role; only "scientist" is excluded.
            "Research Engineer",
        ],
    )
    def test_counts_ic_engineering_roles(self, title):
        assert openings.is_engineering_title(title) is True

    @pytest.mark.parametrize(
        "title",
        [
            # Matched "platform" before include-list was narrowed to role nouns.
            "Account Executive, Strategic Platform Partnerships",
            # Leadership, not the ICs Hire100x places.
            "Engineering Manager - EU",
            "Director of Forward Deployed Engineering",
            "Head of Engineering",
            "VP Engineering",
            "CTO",
            # Escaped the "support engineer" exclusion before the -ing rule.
            "Support Engineering Manager - Americas",
            # Escaped "forward deployed engineer" before separator normalizing.
            "Forward-Deployed Engineer",
            "Forward-Deployed Engineer\xa0",
            # Revenue-side and adjacent.
            "Sales Engineer",
            "Solutions Engineer",
            "Director, Solutions Architect",
            "DevRel Engineer, Agentic Infrastructure",
            "Developer Advocate",
            # Matched "engineering" though it is a comms role.
            "Communications Lead, Infrastructure and Engineering",
            # Matched "developer" though it is a research role.
            "Data Scientist, Developer Productivity",
            "Research Scientist",
            # Internal IT rather than product engineering.
            "IT Systems Engineer",
            # Non-software engineering.
            "Mechanical Engineer",
            "Hardware Engineer",
            # Not permanent IC headcount.
            "Software Engineering Intern",
            "Anthropic Fellows Program, ML Systems & Performance",
            "Apprenticeship - Junior Brand Designer",
            # Non-engineering entirely.
            "Product Manager",
            "Technical Program Manager",
            "Technical Recruiter",
        ],
    )
    def test_rejects_non_ic_engineering_roles(self, title):
        assert openings.is_engineering_title(title) is False

    def test_is_case_insensitive(self):
        assert openings.is_engineering_title("SOFTWARE ENGINEER")
        assert not openings.is_engineering_title("SALES ENGINEER")

    def test_handles_empty_title(self):
        assert openings.is_engineering_title("") is False


class TestNormalizeTitleText:
    @pytest.mark.parametrize(
        "raw",
        ["forward-deployed", "forward deployed", "forward_deployed", "forward\xa0deployed"],
    )
    def test_separator_variants_collapse_to_one_form(self, raw):
        assert openings.normalize_title_text(raw) == "forward deployed"

    def test_collapses_repeated_whitespace_and_trims(self):
        assert openings.normalize_title_text("  Software   Engineer  ") == "software engineer"


class TestProviderClients:
    @respx.mock
    def test_greenhouse_parses_real_response(self):
        respx.get(url__startswith="https://boards-api.greenhouse.io/").mock(
            return_value=httpx.Response(200, json=GREENHOUSE)
        )
        postings = openings.fetch_greenhouse("vercel")

        assert postings is not None and len(postings) == 8
        first = postings[0]
        assert first.title
        assert first.url and first.url.startswith("http")
        assert first.posted_at is not None and first.posted_at.tzinfo is not None

    @respx.mock
    def test_lever_parses_real_response_including_epoch_millis(self):
        respx.get(url__startswith="https://api.lever.co/").mock(
            return_value=httpx.Response(200, json=LEVER)
        )
        postings = openings.fetch_lever("matchgroup")

        assert postings is not None and len(postings) == 8
        assert postings[0].title == "Android Engineer III"
        # Lever sends epoch milliseconds, not ISO-8601.
        assert postings[0].posted_at is not None
        assert postings[0].posted_at.year > 2000

    @respx.mock
    def test_ashby_parses_real_response(self):
        respx.get(url__startswith="https://api.ashbyhq.com/").mock(
            return_value=httpx.Response(200, json=ASHBY)
        )
        postings = openings.fetch_ashby("ashby")

        assert postings is not None
        assert all(p.title for p in postings)

    @respx.mock
    def test_ashby_skips_unlisted_drafts(self):
        payload = {
            "jobs": [
                {"title": "Live Engineer", "isListed": True, "jobUrl": "u", "publishedAt": None},
                {"title": "Draft Engineer", "isListed": False, "jobUrl": "u", "publishedAt": None},
            ]
        }
        respx.get(url__startswith="https://api.ashbyhq.com/").mock(
            return_value=httpx.Response(200, json=payload)
        )
        postings = openings.fetch_ashby("x")
        assert [p.title for p in postings] == ["Live Engineer"]

    @respx.mock
    def test_missing_board_returns_none_not_empty_list(self):
        # None means "no such board"; [] means "board exists, no roles". The
        # distinction drives unverified vs none_found downstream.
        respx.get(url__startswith="https://boards-api.greenhouse.io/").mock(
            return_value=httpx.Response(404)
        )
        assert openings.fetch_greenhouse("nope") is None

    @respx.mock
    def test_html_error_page_with_200_is_treated_as_missing(self):
        respx.get(url__startswith="https://api.lever.co/").mock(
            return_value=httpx.Response(200, text="<html>oops</html>")
        )
        assert openings.fetch_lever("x") is None

    @respx.mock
    def test_network_failure_is_swallowed(self):
        respx.get(url__startswith="https://api.ashbyhq.com/").mock(
            side_effect=httpx.ConnectTimeout("nope")
        )
        assert openings.fetch_ashby("x") is None


class TestParseDt:
    def test_parses_greenhouse_iso_with_offset(self):
        parsed = openings._parse_dt("2026-06-02T08:58:57-04:00")
        assert parsed is not None and parsed.tzinfo is not None

    def test_parses_ashby_iso_utc(self):
        parsed = openings._parse_dt("2024-03-04T14:29:08.532+00:00")
        assert parsed is not None and parsed.year == 2024

    def test_parses_lever_epoch_millis(self):
        parsed = openings._parse_dt(1779223091267)
        assert parsed is not None and parsed.year > 2020

    @pytest.mark.parametrize("bad", [None, "", "not a date", float("nan")])
    def test_bad_values_return_none_rather_than_raising(self, bad):
        assert openings._parse_dt(bad) is None


class TestDiscovery:
    @respx.mock
    def test_finds_a_greenhouse_board_link_on_the_careers_page(self):
        respx.get("https://acme.com/careers").mock(
            return_value=httpx.Response(
                200,
                text='<a href="https://boards.greenhouse.io/acmeco">Open roles</a>',
            )
        )
        assert openings.discover_from_careers_page("acme.com") == ("greenhouse", "acmeco")

    @respx.mock
    def test_finds_the_newer_greenhouse_host(self):
        respx.get("https://acme.com/careers").mock(
            return_value=httpx.Response(
                200, text='<iframe src="https://job-boards.greenhouse.io/acmeco">'
            )
        )
        assert openings.discover_from_careers_page("acme.com") == ("greenhouse", "acmeco")

    @respx.mock
    def test_finds_an_embedded_greenhouse_job_board(self):
        respx.get("https://acme.com/careers").mock(
            return_value=httpx.Response(
                200,
                text='<script src="https://boards.greenhouse.io/embed/job_board/js?for=acmeco">',
            )
        )
        assert openings.discover_from_careers_page("acme.com") == ("greenhouse", "acmeco")

    @respx.mock
    def test_finds_lever_and_ashby_links(self):
        respx.get("https://lev.com/careers").mock(
            return_value=httpx.Response(200, text='href="https://jobs.lever.co/levco"')
        )
        assert openings.discover_from_careers_page("lev.com") == ("lever", "levco")

        respx.get("https://ash.com/careers").mock(
            return_value=httpx.Response(200, text='href="https://jobs.ashbyhq.com/ashco"')
        )
        assert openings.discover_from_careers_page("ash.com") == ("ashby", "ashco")

    @respx.mock
    def test_falls_through_paths_until_one_has_a_board(self):
        respx.get("https://acme.com/careers").mock(return_value=httpx.Response(404))
        respx.get("https://acme.com/jobs").mock(
            return_value=httpx.Response(200, text='href="https://jobs.lever.co/acme"')
        )
        assert openings.discover_from_careers_page("acme.com") == ("lever", "acme")

    @respx.mock
    def test_returns_none_when_no_board_link_exists(self):
        respx.route(host="acme.com").mock(
            return_value=httpx.Response(200, text="<html>We are not hiring</html>")
        )
        assert openings.discover_from_careers_page("acme.com") is None

    def test_cache_short_circuits_discovery(self):
        # No respx mock installed: any HTTP call would raise, proving the cache
        # was used rather than the network.
        cache = {"acme.com": ("greenhouse", "cached-token")}
        assert openings.discover("Acme", "acme.com", cache) == ("greenhouse", "cached-token")


class TestCheck:
    @respx.mock
    def test_verified_when_engineering_roles_exist(self):
        respx.route(host="acme.com").mock(
            return_value=httpx.Response(200, text='href="https://jobs.lever.co/matchgroup"')
        )
        respx.get(url__startswith="https://api.lever.co/").mock(
            return_value=httpx.Response(200, json=LEVER)
        )
        result = openings.check("Acme", "acme.com")

        assert result.status == "verified"
        assert result.ats_provider == "lever"
        assert result.eng_role_count == 5
        assert result.total_role_count == 8
        assert result.board_url == "https://jobs.lever.co/matchgroup"
        assert len(result.sample_titles) <= 6

    @respx.mock
    def test_none_found_when_board_exists_but_has_no_engineering_roles(self):
        payload = [
            {"text": "Account Executive", "hostedUrl": "u", "createdAt": 1700000000000,
             "categories": {"location": "NYC"}},
        ]
        respx.route(host="acme.com").mock(
            return_value=httpx.Response(200, text='href="https://jobs.lever.co/acme"')
        )
        respx.get(url__startswith="https://api.lever.co/").mock(
            return_value=httpx.Response(200, json=payload)
        )
        result = openings.check("Acme", "acme.com")

        assert result.status == "none_found"
        assert result.eng_role_count == 0
        assert result.total_role_count == 1

    @respx.mock
    def test_unverified_when_no_board_can_be_found(self):
        # A company with no discoverable board is flagged, never dropped —
        # plenty of seed-stage teams hire off a Notion page.
        respx.route(host="acme.com").mock(return_value=httpx.Response(404))
        for host in ("boards-api.greenhouse.io", "api.lever.co", "api.ashbyhq.com"):
            respx.route(host=host).mock(return_value=httpx.Response(404))
        result = openings.check("Acme", "acme.com")

        assert result.status == "unverified"
        assert result.eng_role_count == 0

    @respx.mock
    def test_reports_the_newest_engineering_post_date(self):
        respx.route(host="acme.com").mock(
            return_value=httpx.Response(200, text='href="https://jobs.lever.co/matchgroup"')
        )
        respx.get(url__startswith="https://api.lever.co/").mock(
            return_value=httpx.Response(200, json=LEVER)
        )
        result = openings.check("Acme", "acme.com")
        assert result.newest_post_date is not None

    @respx.mock
    def test_stale_cache_entry_degrades_to_unverified(self):
        # A cached token whose board has since gone away must not crash.
        respx.get(url__startswith="https://boards-api.greenhouse.io/").mock(
            return_value=httpx.Response(404)
        )
        cache = {"acme.com": ("greenhouse", "gone")}
        result = openings.check("Acme", "acme.com", cache)

        assert result.status == "unverified"
        assert result.board_token == "gone"
