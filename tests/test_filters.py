"""The geo matcher decides which companies ever reach scoring, so a
false negative silently costs a lead and a false positive wastes an ATS lookup
and an outreach draft. It gets adversarial coverage."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from signal_engine import filters
from signal_engine.schemas import FundingEvent

TODAY = datetime.now(timezone.utc).date()


def _event(**overrides) -> FundingEvent:
    base = dict(
        is_funding_announcement=True,
        company_name="Acme",
        company_domain="acme.com",
        round_stage="series-a",
        amount_usd=20_000_000,
        announced_date=TODAY,
        investors=["Foundry"],
        hq_city="San Francisco",
        hq_country="United States",
        sector="devtools",
        one_line_description="Developer tooling.",
        source_url="https://example.com/acme",
        extraction_confidence=0.9,
    )
    base.update(overrides)
    return FundingEvent(**base)


class TestMatchMarket:
    @pytest.mark.parametrize(
        "city,expected",
        [
            ("San Francisco", "sf-bay-area"),
            ("San Francisco, CA", "sf-bay-area"),
            ("Palo Alto", "sf-bay-area"),
            ("Mountain View, California", "sf-bay-area"),
            ("Oakland", "sf-bay-area"),
            ("New York", "nyc-metro"),
            ("New York City, NY", "nyc-metro"),
            ("Brooklyn", "nyc-metro"),
            ("Jersey City", "nyc-metro"),
            ("Bengaluru", "bengaluru"),
            ("Bangalore, India", "bengaluru"),
            ("Koramangala, Bengaluru", "bengaluru"),
        ],
    )
    def test_recognizes_target_market_cities(self, city, expected):
        assert filters.match_market(city) == expected

    @pytest.mark.parametrize(
        "city,market",
        [
            ("Austin", "us-other"), ("Seattle", "us-other"), ("Boston", "us-other"),
            ("Mumbai", "india-other"), ("Delhi", "india-other"),
            ("London", "uk"), ("Toronto", "canada"), ("Berlin", "europe"),
            ("Singapore", "apac"),
        ],
    )
    def test_wider_regions_resolve_to_their_tier(self, city, market):
        assert filters.match_market(city) == market

    @pytest.mark.parametrize(
        "city", ["Lagos", "Nairobi", "Karachi", "Bogota", "Almaty"]
    )
    def test_rejects_genuinely_uncovered_cities(self, city):
        assert filters.match_market(city) is None

    @pytest.mark.parametrize(
        "location,expected",
        [
            # A hub must beat the broad region that contains it, even though
            # both alias strings are the same length.
            ("San Francisco, CA, United States", "sf-bay-area"),
            ("New York, United States", "nyc-metro"),
            ("Bengaluru, India", "bengaluru"),
            # ...and a non-hub city in the same country falls to tier 2.
            ("Austin, TX, United States", "us-other"),
            ("Pune, India", "india-other"),
        ],
    )
    def test_hub_outranks_the_region_containing_it(self, location, expected):
        assert filters.match_market(location) == expected

    def test_south_san_francisco_is_in_the_bay_area(self):
        # It is a real Bay Area biotech hub, not a false positive to guard
        # against — the longest-alias rule must resolve it correctly.
        assert filters.match_market("South San Francisco, CA") == "sf-bay-area"

    def test_word_boundaries_prevent_substring_false_positives(self):
        # "sf" must not fire inside another word, and "york" must not stand in
        # for "New York".
        assert filters.match_market("Yorkshire") is None
        assert filters.match_market("Transfer City") is None

    def test_exclusion_phrases_veto_a_match(self):
        # An entity name that contains a city name is not a location.
        assert filters.match_market("New York Times Building") is None
        assert filters.match_market("Bangalore Rural") is None

    def test_unknown_country_does_not_veto(self):
        # Feeds routinely omit the country; the city alias is signal enough.
        assert filters.match_market("Bengaluru", None) == "bengaluru"

    def test_handles_missing_city(self):
        assert filters.match_market(None) is None
        assert filters.match_market("") is None

    def test_is_case_insensitive(self):
        assert filters.match_market("SAN FRANCISCO") == "sf-bay-area"
        assert filters.match_market("bengaluru") == "bengaluru"


class TestPredicates:
    def test_non_funding_events_are_invalid(self):
        assert not filters.is_valid_funding(_event(is_funding_announcement=False))

    def test_confidence_threshold(self):
        assert filters.is_confident(_event(extraction_confidence=0.6))
        assert not filters.is_confident(_event(extraction_confidence=0.59))

    def test_recent_event_passes(self):
        assert filters.is_recent(_event(announced_date=TODAY - timedelta(days=3)))

    def test_stale_event_fails(self):
        assert not filters.is_recent(
            _event(announced_date=TODAY - timedelta(days=60)), max_age_days=14
        )

    def test_missing_announcement_date_is_kept(self):
        # The article already passed the ingest recency window; a missing field
        # is an extraction gap, not evidence the round is old.
        assert filters.is_recent(_event(announced_date=None))


class TestApply:
    def test_keeps_a_good_candidate(self):
        candidates, report = filters.apply([_event()])
        assert len(candidates) == 1
        # Market is no longer decided here — apply_geo() sets it after the
        # openings check, because the article rarely states a location.
        assert candidates[0].market is None
        assert report.kept == 1

    def test_counts_each_rejection_reason_separately(self):
        events = [
            _event(company_name="NotFunding", is_funding_announcement=False),
            _event(company_name="LowConf", extraction_confidence=0.2),
            _event(company_name="Stale", announced_date=TODAY - timedelta(days=90)),
            _event(company_name="Good"),
        ]
        candidates, report = filters.apply(events)

        assert [c.event.company_name for c in candidates] == ["Good"]
        assert report.not_funding == 1
        assert report.low_confidence == 1
        assert report.too_old == 1
        assert report.input_count == 4

    def test_collapses_the_same_company_appearing_twice_in_one_run(self):
        events = [
            _event(company_name="Acme", company_domain="acme.com"),
            _event(company_name="Acme Inc.", company_domain="acme.com"),
        ]
        candidates, report = filters.apply(events)

        assert len(candidates) == 1
        assert report.duplicate == 1

    def test_dedupes_by_normalized_name_when_domain_is_unknown(self):
        events = [
            _event(company_name="Acme Labs", company_domain=None),
            _event(company_name="Acme Labs, Inc.", company_domain=None),
        ]
        candidates, _ = filters.apply(events)
        assert len(candidates) == 1

    def test_suppresses_a_company_posted_inside_the_dedupe_window(self):
        seen = {"acme.com": TODAY - timedelta(days=5)}
        candidates, report = filters.apply(
            [_event()], seen_keys=seen, dedupe_window_days=30
        )
        assert candidates == []
        assert report.recently_seen == 1

    def test_allows_a_company_whose_suppression_window_has_expired(self):
        seen = {"acme.com": TODAY - timedelta(days=45)}
        candidates, _ = filters.apply([_event()], seen_keys=seen, dedupe_window_days=30)
        assert len(candidates) == 1

    def test_boundary_of_the_suppression_window_still_suppresses(self):
        seen = {"acme.com": TODAY - timedelta(days=30)}
        candidates, _ = filters.apply([_event()], seen_keys=seen, dedupe_window_days=30)
        assert candidates == []

    def test_passing_no_seen_map_skips_suppression_entirely(self):
        # Dry runs pass None so you can see the full unfiltered picture.
        candidates, report = filters.apply([_event()], seen_keys=None)
        assert len(candidates) == 1
        assert report.recently_seen == 0

    def test_records_examples_of_what_was_rejected(self):
        _, report = filters.apply(
            [_event(company_name="Shaky Co", extraction_confidence=0.1)]
        )
        assert any("Shaky Co" in note for note in report.rejected_examples)

    def test_empty_input(self):
        candidates, report = filters.apply([])
        assert candidates == []
        assert report.input_count == 0


class TestNormalizedSeenKeys:
    def test_domains_pass_through_unchanged(self):
        result = filters.normalized_seen_keys([("acme.com", date(2026, 7, 1))])
        assert result == {"acme.com": date(2026, 7, 1)}

    def test_bare_names_are_normalized(self):
        result = filters.normalized_seen_keys([("Acme Inc.", date(2026, 7, 1))])
        assert "acme" in result

    def test_keeps_the_most_recent_sighting(self):
        result = filters.normalized_seen_keys(
            [("acme.com", date(2026, 6, 1)), ("acme.com", date(2026, 7, 1))]
        )
        assert result["acme.com"] == date(2026, 7, 1)


class TestApplyGeo:
    """Geography now resolves from ATS job locations, because funding articles
    almost never print a company's HQ — a full run measured zero out of 66."""

    def _candidate(self, locations=None, hq_city=None):
        from signal_engine.schemas import Candidate, OpeningsResult

        c = Candidate(event=_event(hq_city=hq_city))
        if locations is not None:
            c.openings = OpeningsResult(status="verified", locations=locations)
        return c

    def test_matches_on_a_job_location(self):
        kept, report = filters.apply_geo(
            [self._candidate(locations=["Hybrid - San Francisco, New York City"])]
        )
        assert len(kept) == 1
        assert kept[0].market == "sf-bay-area"
        assert report.matched_on_jobs == 1

    def test_checks_every_location_not_just_the_first(self):
        kept, _ = filters.apply_geo(
            [self._candidate(locations=["Lagos", "Nairobi", "Bengaluru"])]
        )
        assert kept and kept[0].market == "bengaluru"

    def test_falls_back_to_the_article_hq(self):
        kept, report = filters.apply_geo(
            [self._candidate(locations=[], hq_city="New York")]
        )
        assert len(kept) == 1
        assert kept[0].market == "nyc-metro"
        assert report.matched_on_article == 1

    def test_job_locations_win_over_the_article(self):
        # The board is the more reliable source, and where the roles are is
        # what matters for placing engineers.
        kept, report = filters.apply_geo(
            [self._candidate(locations=["Bengaluru"], hq_city="New York")]
        )
        assert kept[0].market == "bengaluru"
        assert report.matched_on_jobs == 1

    def test_out_of_market_is_dropped(self):
        kept, report = filters.apply_geo([self._candidate(locations=["Lagos"])])
        assert kept == []
        assert report.out_of_market == 1
        assert report.location_unknown == 0

    def test_unknown_location_is_counted_separately_from_out_of_market(self):
        # A pile of "unknown" means the openings stage is failing to find
        # boards, which is a different problem from genuinely foreign companies.
        kept, report = filters.apply_geo([self._candidate(locations=[])])
        assert kept == []
        assert report.location_unknown == 1
        assert report.out_of_market == 0
        assert report.unknown_examples

    def test_candidate_with_no_openings_at_all(self):
        from signal_engine.schemas import Candidate

        kept, report = filters.apply_geo([Candidate(event=_event(hq_city=None))])
        assert kept == [] and report.location_unknown == 1

    def test_empty_input(self):
        kept, report = filters.apply_geo([])
        assert kept == [] and report.input_count == 0


class TestMarketFromLocations:
    def test_returns_none_for_empty(self):
        assert filters.market_from_locations([]) is None

    def test_handles_remote_prefixes(self):
        assert filters.market_from_locations(["Remote - San Francisco"]) == "sf-bay-area"

    def test_ignores_unmatched_locations(self):
        assert filters.market_from_locations(["Lagos", "Nairobi"]) is None

    def test_remote_emea_resolves_to_europe(self):
        assert filters.market_from_locations(["Remote - EMEA"]) == "europe"


def _cand(name, domain, board, conf=0.9, amount=None):
    from signal_engine.schemas import Candidate, FundingEvent, OpeningsResult
    ev = FundingEvent(
        is_funding_announcement=True, company_name=name, company_domain=domain,
        round_stage="seed", amount_usd=amount, investors=[], sector="ai",
        one_line_description="x", source_url=f"https://n/{name}",
        extraction_confidence=conf,
    )
    c = Candidate(event=ev)
    c.openings = OpeningsResult(status="verified", board_url=board, eng_role_count=3)
    return c


def test_collapse_duplicate_boards_merges_conflicting_domains():
    """Same board = same employer, whatever domain the article claimed.

    Regression: a real run shortlisted Etched three times — etched.com,
    etched.ai, and no domain — because dedupe keys on domain and the outlets
    disagreed. All three pointed at one Ashby board.
    """
    cands = [
        _cand("Etched", "etched.com", "https://jobs.ashbyhq.com/etched"),
        _cand("Etched", "", "https://jobs.ashbyhq.com/etched"),
        _cand("Etched", "etched.ai", "https://jobs.ashbyhq.com/etched"),
        _cand("Meshy", "meshy.ai", "https://jobs.ashbyhq.com/meshy"),
    ]
    kept, collapsed = filters.collapse_duplicate_boards(cands)

    assert collapsed == 2
    assert [c.event.company_name for c in kept] == ["Etched", "Meshy"]
    # Must keep a row with a real domain: the `seen` ledger keys on it, so
    # keeping the blank one would let Etched back in tomorrow.
    assert kept[0].event.company_domain


def test_collapse_prefers_the_richer_row():
    cands = [
        _cand("Acme", "acme.com", "https://b/acme", conf=0.6, amount=None),
        _cand("Acme", "acme.com", "https://b/acme", conf=0.95, amount=5_000_000),
    ]
    kept, _ = filters.collapse_duplicate_boards(cands)
    assert kept[0].event.amount_usd == 5_000_000


def test_collapse_never_merges_boardless_companies_by_name():
    """Two unrelated startups share a name often enough that name is unsafe."""
    cands = [_cand("Nova", "nova-ai.com", None), _cand("Nova", "novabank.io", None)]
    kept, collapsed = filters.collapse_duplicate_boards(cands)
    assert collapsed == 0 and len(kept) == 2


def test_collapse_preserves_input_order():
    cands = [
        _cand("Zed", "zed.dev", "https://b/zed"),
        _cand("Alpha", "alpha.com", "https://b/alpha"),
        _cand("Zed", "zed.com", "https://b/zed"),
    ]
    kept, _ = filters.collapse_duplicate_boards(cands)
    assert [c.event.company_name for c in kept] == ["Zed", "Alpha"]
