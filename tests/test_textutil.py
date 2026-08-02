"""String helpers underpin dedupe and ATS board discovery — both silently
degrade rather than crash when they get this wrong, so they get real coverage."""

from __future__ import annotations

import pytest

from signal_engine.textutil import (
    canonical_url,
    company_slug,
    domain_from_url,
    normalize_company,
    normalize_title,
    truncate_words,
)


class TestNormalizeTitle:
    def test_collapses_outlet_variations_of_the_same_story(self):
        # Four outlets covering one round must produce one dedupe key.
        variants = [
            "Acme raises $20M Series A",
            "Acme Raises $20M Series A!",
            "Acme raises $20M Series A.",
        ]
        keys = {normalize_title(v) for v in variants}
        assert len(keys) == 1

    def test_strips_leading_articles(self):
        assert normalize_title("The Acme Story") == normalize_title("Acme Story")

    def test_distinct_stories_stay_distinct(self):
        assert normalize_title("Acme raises $20M") != normalize_title("Beta raises $20M")


class TestNormalizeCompany:
    @pytest.mark.parametrize(
        "raw",
        ["Acme", "Acme, Inc.", "Acme Inc", "Acme Technologies", "ACME  Labs"],
    )
    def test_legal_suffixes_collapse_to_the_same_key(self, raw):
        assert normalize_company(raw) == "acme"

    def test_never_returns_empty_for_a_suffix_only_name(self):
        # "AI" is in the suffix list; a company literally named "AI" must not
        # normalize to "" and collide with every other empty-normalizing name.
        assert normalize_company("AI") != ""

    def test_different_companies_do_not_collide(self):
        assert normalize_company("Acme Labs") != normalize_company("Beta Labs")


class TestCompanySlug:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Acme Labs", "acmelabs"),
            ("Acme, Inc.", "acmeinc"),
            ("Foo-Bar", "foobar"),
            ("7Shifts", "7shifts"),
        ],
    )
    def test_produces_ats_style_tokens(self, name, expected):
        assert company_slug(name) == expected


class TestCanonicalUrl:
    def test_strips_utm_parameters(self):
        assert canonical_url(
            "https://techcrunch.com/post?utm_source=rss&utm_medium=feed"
        ) == canonical_url("https://techcrunch.com/post")

    def test_strips_www_and_trailing_slash_and_fragment(self):
        assert canonical_url("https://www.example.com/a/#top") == canonical_url(
            "https://example.com/a"
        )

    def test_preserves_meaningful_query_parameters(self):
        assert "id=42" in canonical_url("https://example.com/p?id=42&utm_source=x")

    def test_different_articles_stay_different(self):
        assert canonical_url("https://example.com/a") != canonical_url(
            "https://example.com/b"
        )


class TestDomainFromUrl:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://www.acme.com/careers", "acme.com"),
            ("http://acme.com", "acme.com"),
            ("acme.com", "acme.com"),
            ("https://acme.com:8443/x", "acme.com"),
            ("https://jobs.acme.co.uk/", "jobs.acme.co.uk"),
        ],
    )
    def test_extracts_bare_domain(self, raw, expected):
        assert domain_from_url(raw) == expected

    def test_returns_none_for_empty(self):
        assert domain_from_url("") is None


class TestTruncateWords:
    def test_leaves_short_text_untouched(self):
        assert truncate_words("one two three", 5) == "one two three"

    def test_cuts_at_the_word_limit(self):
        assert truncate_words("one two three four", 2) == "one two…"
