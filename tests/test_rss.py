"""Ingest is the stage most exposed to the outside world — feeds change format,
go 404, and syndicate the same story a dozen ways. These tests pin the two
behaviours the pipeline depends on: one bad feed never stops a run, and
duplicates are collapsed before they reach the (paid) extraction stage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import respx

from signal_engine.config import Feed
from signal_engine.schemas import Article
from signal_engine.sources import rss

FIXTURES = Path(__file__).parent / "fixtures"
FEED_XML = (FIXTURES / "sample_feed.xml").read_bytes()

NOW = datetime.now(timezone.utc)


def _article(
    title: str,
    url: str,
    *,
    hours_ago: float | None = 1.0,
    source: str = "test",
) -> Article:
    return Article(
        title=title,
        url=url,
        source=source,
        published_at=None if hours_ago is None else NOW - timedelta(hours=hours_ago),
    )


class TestFetchFeed:
    @respx.mock
    def test_parses_a_well_formed_feed(self):
        respx.get("https://example.com/feed").mock(
            return_value=httpx.Response(200, content=FEED_XML)
        )
        result = rss.fetch_feed(Feed(name="Example", url="https://example.com/feed"))

        assert result.ok
        # The item with no <link> is skipped; the other three survive.
        assert len(result.articles) == 3
        assert result.articles[0].title == "Acme raises $20M Series A led by Foundry"
        assert result.articles[0].source == "Example"

    @respx.mock
    def test_strips_html_from_summaries(self):
        respx.get("https://example.com/feed").mock(
            return_value=httpx.Response(200, content=FEED_XML)
        )
        summary = rss.fetch_feed(
            Feed(name="Example", url="https://example.com/feed")
        ).articles[0].summary

        assert "<" not in summary and ">" not in summary
        assert "San Francisco" in summary

    @respx.mock
    def test_keeps_items_with_no_publication_date(self):
        respx.get("https://example.com/feed").mock(
            return_value=httpx.Response(200, content=FEED_XML)
        )
        articles = rss.fetch_feed(
            Feed(name="Example", url="https://example.com/feed")
        ).articles

        undated = [a for a in articles if a.published_at is None]
        assert len(undated) == 1
        assert undated[0].title == "Undated item"

    @respx.mock
    def test_http_error_is_recorded_not_raised(self):
        respx.get("https://dead.example.com/feed").mock(
            return_value=httpx.Response(404)
        )
        result = rss.fetch_feed(Feed(name="Dead", url="https://dead.example.com/feed"))

        assert not result.ok
        assert "404" in result.error
        assert result.articles == []

    @respx.mock
    def test_network_error_is_recorded_not_raised(self):
        respx.get("https://timeout.example.com/feed").mock(
            side_effect=httpx.ConnectTimeout("timed out")
        )
        result = rss.fetch_feed(
            Feed(name="Timeout", url="https://timeout.example.com/feed")
        )

        assert not result.ok
        assert "ConnectTimeout" in result.error

    @respx.mock
    def test_html_error_page_yields_an_error_not_a_crash(self):
        # A feed that 200s with an HTML error page is a common failure mode.
        respx.get("https://html.example.com/feed").mock(
            return_value=httpx.Response(200, content=b"<html><body>Not a feed</body></html>")
        )
        result = rss.fetch_feed(Feed(name="HTML", url="https://html.example.com/feed"))

        assert result.articles == []

    @respx.mock
    def test_one_dead_feed_does_not_stop_the_others(self):
        respx.get("https://good.example.com/feed").mock(
            return_value=httpx.Response(200, content=FEED_XML)
        )
        respx.get("https://dead.example.com/feed").mock(return_value=httpx.Response(500))

        results = rss.fetch_all(
            [
                Feed(name="Good", url="https://good.example.com/feed"),
                Feed(name="Dead", url="https://dead.example.com/feed"),
            ]
        )

        assert sum(r.ok for r in results) == 1
        assert sum(len(r.articles) for r in results) == 3


class TestFilterRecent:
    def test_drops_articles_older_than_the_window(self):
        kept = rss.filter_recent(
            [
                _article("fresh", "https://x.com/1", hours_ago=2),
                _article("stale", "https://x.com/2", hours_ago=100),
            ],
            max_age_hours=36,
        )
        assert [a.title for a in kept] == ["fresh"]

    def test_keeps_undated_articles(self):
        # A missing timestamp is a feed quirk, not evidence of staleness — the
        # extraction stage reads the real announcement date from the body.
        kept = rss.filter_recent(
            [_article("undated", "https://x.com/1", hours_ago=None)], max_age_hours=1
        )
        assert len(kept) == 1

    def test_boundary_article_is_kept(self):
        kept = rss.filter_recent(
            [_article("edge", "https://x.com/1", hours_ago=35.9)], max_age_hours=36
        )
        assert len(kept) == 1


class TestDedupe:
    def test_collapses_urls_differing_only_by_tracking_params(self):
        result = rss.dedupe(
            [
                _article("Acme raises $20M", "https://x.com/acme?utm_source=rss"),
                _article("Acme raises $20M", "https://x.com/acme"),
            ]
        )
        assert len(result) == 1

    def test_collapses_near_identical_headlines_from_different_outlets(self):
        result = rss.dedupe(
            [
                _article("Acme raises $20M Series A", "https://tc.com/a", source="TC"),
                _article("Acme Raises $20M Series A!", "https://vb.com/b", source="VB"),
            ]
        )
        assert len(result) == 1

    def test_keeps_the_earliest_published_copy(self):
        # The earliest copy is usually the original reporting rather than an
        # aggregator's rewrite.
        result = rss.dedupe(
            [
                _article("Acme raises $20M", "https://agg.com/a", hours_ago=1, source="Aggregator"),
                _article("Acme raises $20M", "https://tc.com/a", hours_ago=5, source="TechCrunch"),
            ]
        )
        assert len(result) == 1
        assert result[0].source == "TechCrunch"

    def test_does_not_merge_genuinely_different_stories(self):
        result = rss.dedupe(
            [
                _article("Acme raises $20M", "https://x.com/a"),
                _article("Beta raises $8M", "https://x.com/b"),
            ]
        )
        assert len(result) == 2

    def test_returns_newest_first(self):
        result = rss.dedupe(
            [
                _article("Older story", "https://x.com/a", hours_ago=10),
                _article("Newer story", "https://x.com/b", hours_ago=1),
            ]
        )
        assert [a.title for a in result] == ["Newer story", "Older story"]

    def test_undated_articles_survive_dedupe(self):
        result = rss.dedupe([_article("Undated", "https://x.com/a", hours_ago=None)])
        assert len(result) == 1

    def test_empty_input(self):
        assert rss.dedupe([]) == []


class TestParseDatetime:
    @pytest.mark.parametrize(
        "raw",
        [
            "Sat, 01 Aug 2026 14:30:00 +0000",
            "Sat, 01 Aug 2026 14:30:00 GMT",
        ],
    )
    def test_parses_common_rfc822_formats(self, raw):
        entry = type("Entry", (), {"published": raw})()
        parsed = rss._parse_datetime(entry)
        assert parsed is not None and parsed.tzinfo is not None

    def test_returns_none_for_unparseable_dates(self):
        entry = type("Entry", (), {"published": "last Tuesday-ish"})()
        assert rss._parse_datetime(entry) is None

    def test_naive_datetimes_are_treated_as_utc(self):
        entry = type("Entry", (), {"published": "Sat, 01 Aug 2026 14:30:00 -0000"})()
        parsed = rss._parse_datetime(entry)
        assert parsed is not None and parsed.tzinfo is not None
