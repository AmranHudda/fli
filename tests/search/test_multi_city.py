"""Offline tests for multi-city search through the browser transport.

The browser itself is never started here: ``fli.search._browser.fetch_shopping_body``
is replaced by a fake that serves canned ``GetShoppingResults`` bodies, chosen
by which first-leg flight (if any) the requested ``tfs`` pins. The live
counterpart is ``test_multi_city_live.py``.
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest

from fli.models import Airport, FlightSearchFilters, FlightSegment, PassengerInfo, SortBy
from fli.models.google_flights.base import TripType
from fli.search import _browser
from fli.search._tfs import build_tfs
from fli.search.exceptions import (
    BrowserUnavailableError,
    SearchRejectedError,
    SearchUnsupportedError,
)
from fli.search.flights import (
    MAX_MULTI_CITY_PAGES,
    SearchFlights,
    combo_price,
    multi_city_page_count,
)
from tests.search._pages import as_search_page
from tests.search.test_parse_flights_data import _leg, _row

TRIP_CODE_FIELD = b"\x98\x01"  # field 19, varint


def _future(days: int) -> str:
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")


def _filters(legs, **overrides) -> FlightSearchFilters:
    overrides = {"passenger_info": PassengerInfo(adults=1), **overrides}
    return FlightSearchFilters(
        trip_type=TripType.MULTI_CITY,
        flight_segments=[
            FlightSegment(
                departure_airport=[[origin, 0]],
                arrival_airport=[[dest, 0]],
                travel_date=date,
            )
            for origin, dest, date in legs
        ],
        **overrides,
    )


def _open_jaw(**overrides) -> FlightSearchFilters:
    return _filters(
        [(Airport.BOS, Airport.FCO, _future(30)), (Airport.FCO, Airport.ORD, _future(44))],
        **overrides,
    )


def _flight(airline: str, number: str, dep: str, arr: str, price: float):
    return _row(
        legs=[
            _leg(
                dep_iata=dep,
                arr_iata=arr,
                airline_code=airline,
                flight_number=number,
                airline_name=airline,
            )
        ],
        price=price,
        primary_airline_code=airline,
        primary_airline_name=airline,
    )


def _rpc_body(rows: list) -> str:
    """Wrap rows the way ``GetShoppingResults`` frames them."""
    inner = [[None, None, None, None, "FAKE_SESSION"], None, [rows], None]
    return ")]}'\n\n" + json.dumps([["wrb.fr", None, json.dumps(inner)]])


def _tfs_bytes(url: str) -> bytes:
    tfs = parse_qs(urlparse(url).query)["tfs"][0]
    return base64.urlsafe_b64decode(tfs + "=" * (-len(tfs) % 4))


# First board: two first-leg options, each quoting the cheapest total it can
# be part of. Second boards (one per pinned first leg) quote exact totals.
FIRST_BOARD = [
    _flight("UA", "1111", "BOS", "FCO", 2133),
    _flight("AC", "2222", "BOS", "FCO", 2334),
]
SECOND_BOARDS = {
    b"1111": [
        _flight("LH", "3333", "FCO", "ORD", 2133),
        _flight("AC", "4444", "FCO", "ORD", 2418),
    ],
    b"2222": [
        _flight("AC", "5555", "FCO", "ORD", 2334),
        _flight("CX", "6666", "FCO", "ORD", 8685),
    ],
}


@pytest.fixture
def fake_browser(monkeypatch):
    """Serve canned multi-city boards and record every URL requested."""
    urls: list[str] = []

    def _fetch(url: str) -> str:
        urls.append(url)
        raw = _tfs_bytes(url)
        for number, rows in SECOND_BOARDS.items():
            if number in raw:
                return _rpc_body(rows)
        return _rpc_body(FIRST_BOARD)

    monkeypatch.setattr(_browser, "fetch_shopping_body", _fetch)
    return urls


@pytest.fixture
def no_http(monkeypatch):
    """Fail if a multi-city search falls back to the plain-HTTP page transport."""

    def _refuse(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError("multi-city search must not use the HTTP page transport")

    monkeypatch.setattr("fli.search.client.Client.get", _refuse, raising=False)


class TestTfsEncoding:
    def test_multi_city_encodes_trip_code_3_when_allowed(self):
        tfs = build_tfs(_open_jaw(), allow_multi_city=True)
        raw = base64.urlsafe_b64decode(tfs + "=" * (-len(tfs) % 4))
        assert raw.endswith(TRIP_CODE_FIELD + b"\x03")
        for code in (b"BOS", b"FCO", b"ORD"):
            assert code in raw

    def test_http_path_still_refuses_multi_city(self):
        """The page inlines no multi-city rows, so plain HTTP must not pretend."""
        with pytest.raises(SearchUnsupportedError, match="Multi-city"):
            build_tfs(_open_jaw())


class TestMultiCitySearch:
    def test_routes_through_browser_and_prices_each_combination(self, fake_browser, no_http):
        combos = SearchFlights().search(_open_jaw(sort_by=SortBy.CHEAPEST), top_n=2)

        # One first board plus one second board per expanded first leg.
        assert len(fake_browser) == 3
        assert all(_tfs_bytes(u).endswith(TRIP_CODE_FIELD + b"\x03") for u in fake_browser)

        got = [
            (combo[0].legs[0].flight_number, combo[1].legs[0].flight_number, combo_price(combo))
            for combo in combos
        ]
        # Sorted by the trip's real total, which the second board quotes.
        assert got == [
            ("1111", "3333", 2133),
            ("2222", "5555", 2334),
            ("1111", "4444", 2418),
            ("2222", "6666", 8685),
        ]

    def test_best_keeps_googles_order(self, fake_browser, no_http):
        """BEST is Google's own ranking, which fli cannot reproduce, so order is kept."""
        combos = SearchFlights().search(_open_jaw(sort_by=SortBy.BEST), top_n=2)
        assert [(c[0].legs[0].flight_number, c[1].legs[0].flight_number) for c in combos] == [
            ("1111", "3333"),
            ("1111", "4444"),
            ("2222", "5555"),
            ("2222", "6666"),
        ]

    def test_top_n_limits_expansion(self, fake_browser, no_http):
        combos = SearchFlights().search(_open_jaw(), top_n=1)
        assert len(fake_browser) == 2
        assert {combo[0].legs[0].flight_number for combo in combos} == {"1111"}

    def test_second_board_pins_the_chosen_first_flight(self, fake_browser, no_http):
        SearchFlights().search(_open_jaw(), top_n=2)
        pinned = fake_browser[1:]
        assert any(b"1111" in _tfs_bytes(u) for u in pinned)
        assert any(b"2222" in _tfs_bytes(u) for u in pinned)
        assert b"1111" not in _tfs_bytes(fake_browser[0])

    def test_rejection_from_google_surfaces(self, monkeypatch):
        body = ')]}\'\n\n[["wrb.fr",null,null,null,null,[13]]]'
        monkeypatch.setattr(_browser, "fetch_shopping_body", lambda url: body)
        with pytest.raises(SearchRejectedError):
            SearchFlights().search(_open_jaw())

    def test_too_many_pages_is_refused_before_any_fetch(self, fake_browser):
        five_legs = _filters(
            [
                (Airport.JFK, Airport.LHR, _future(30)),
                (Airport.LHR, Airport.CDG, _future(33)),
                (Airport.CDG, Airport.FCO, _future(36)),
                (Airport.FCO, Airport.ATH, _future(39)),
                (Airport.ATH, Airport.JFK, _future(42)),
            ]
        )
        assert multi_city_page_count(5, 3) > MAX_MULTI_CITY_PAGES
        with pytest.raises(ValueError, match="top_n"):
            SearchFlights().search(five_legs, top_n=3)
        assert fake_browser == []

    @pytest.mark.parametrize(
        ("legs", "top_n", "pages"),
        [(2, 3, 4), (3, 3, 13), (2, 10, 11), (4, 2, 15)],
    )
    def test_page_count(self, legs, top_n, pages):
        assert multi_city_page_count(legs, top_n) == pages

    def test_sort_by_duration_orders_combinations(self, fake_browser):
        combos = SearchFlights().search(_open_jaw(sort_by=SortBy.DURATION), top_n=2)
        totals = [sum(segment.duration for segment in combo) for combo in combos]
        assert totals == sorted(totals)


class TestBrowserUnavailable:
    def test_missing_playwright_raises_with_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "playwright.async_api", None)
        fetcher = _browser.BrowserFetcher(timeout=5)
        try:
            with pytest.raises(BrowserUnavailableError, match=r"flights\[browser\]"):
                fetcher.fetch("https://www.google.com/travel/flights?tfs=x")
        finally:
            fetcher.close()

    def test_close_without_start_is_a_no_op(self):
        fetcher = _browser.BrowserFetcher()
        fetcher.close()
        fetcher.close()


class TestComboPricing:
    def test_round_trip_combinations_are_sorted_by_real_total(self, monkeypatch):
        """The outbound board's price is only the cheapest total it can be part of."""
        outbound_page = as_search_page(
            [[None, None, None, None, "S"], None, [[_flight("AA", "100", "JFK", "LAX", 458)]], None]
        )
        return_page = as_search_page(
            [
                [None, None, None, None, "S"],
                None,
                [
                    [
                        _flight("AA", "200", "LAX", "JFK", 935),
                        _flight("AA", "300", "LAX", "JFK", 458),
                        _flight("AA", "400", "LAX", "JFK", 532),
                    ]
                ],
                None,
            ]
        )
        sf = SearchFlights()
        calls: list[str] = []

        def _get(url, **kwargs):  # noqa: ANN001
            body = outbound_page if not calls else return_page
            calls.append(url)
            return type("R", (), {"text": body, "raise_for_status": lambda self: None})()

        monkeypatch.setattr(sf.client, "get", _get)
        filters = FlightSearchFilters(
            trip_type=TripType.ROUND_TRIP,
            sort_by=SortBy.CHEAPEST,
            passenger_info=PassengerInfo(adults=1),
            flight_segments=[
                FlightSegment(
                    departure_airport=[[Airport.JFK, 0]],
                    arrival_airport=[[Airport.LAX, 0]],
                    travel_date=_future(30),
                ),
                FlightSegment(
                    departure_airport=[[Airport.LAX, 0]],
                    arrival_airport=[[Airport.JFK, 0]],
                    travel_date=_future(37),
                ),
            ],
        )
        combos = sf.search(filters, top_n=1)
        assert [combo_price(c) for c in combos] == [458, 532, 935]

    def test_combo_price_falls_back_to_first_segment(self):
        first = SearchFlights._parse_flights_data(_flight("BA", "1", "JFK", "LHR", 900))
        last = SearchFlights._parse_flights_data(_flight("BA", "2", "LHR", "JFK", 1))
        last.price = None
        assert combo_price((first, last)) == 900


class TestBookingLinkTripType:
    @staticmethod
    def _trip_code(url: str) -> int:
        raw = _tfs_bytes(url)
        index = raw.rindex(TRIP_CODE_FIELD)
        return raw[index + len(TRIP_CODE_FIELD)]

    def _pair(self, out, back):
        return (
            SearchFlights._parse_flights_data(_flight("UA", "1", *out, 500)),
            SearchFlights._parse_flights_data(_flight("UA", "2", *back, 500)),
        )

    def test_open_jaw_is_inferred_as_multi_city(self):
        url = SearchFlights().build_flight_booking_url(self._pair(("BOS", "FCO"), ("MXP", "ORD")))
        assert self._trip_code(url) == 3

    def test_retracing_pair_is_inferred_as_round_trip(self):
        url = SearchFlights().build_flight_booking_url(self._pair(("JFK", "LAX"), ("LAX", "JFK")))
        assert self._trip_code(url) == 1

    def test_explicit_trip_type_wins(self):
        url = SearchFlights().build_flight_booking_url(
            self._pair(("JFK", "LAX"), ("LAX", "JFK")), trip_type=TripType.MULTI_CITY
        )
        assert self._trip_code(url) == 3


REFUSED = ')]}\'\n\n[["wrb.fr",null,null,null,null,[13]]]'
INVALID = ')]}\'\n\n[["wrb.fr",null,null,null,null,[3]]]'


class TestIntermittentRefusals:
    """Google refuses a share of multi-city boards with error 13, in clusters."""

    @pytest.fixture
    def no_sleep(self, monkeypatch):
        delays: list[float] = []
        monkeypatch.setattr("fli.search.flights._sleep", delays.append)
        return delays

    @staticmethod
    def _serve(monkeypatch, refuse):
        """Serve the canned boards, refusing a URL while ``refuse(url, n)`` is true.

        ``n`` counts earlier requests for the same URL.
        """
        counts: dict[str, int] = {}

        def _fetch(url: str) -> str:
            n = counts.get(url, 0)
            counts[url] = n + 1
            if refuse(url, n):
                return REFUSED
            raw = _tfs_bytes(url)
            for number, rows in SECOND_BOARDS.items():
                if number in raw:
                    return _rpc_body(rows)
            return _rpc_body(FIRST_BOARD)

        monkeypatch.setattr(_browser, "fetch_shopping_body", _fetch)
        return counts

    def test_refused_board_is_retried(self, monkeypatch, no_sleep):
        self._serve(monkeypatch, lambda url, n: n < 2)
        combos = SearchFlights().search(_open_jaw(sort_by=SortBy.CHEAPEST), top_n=2)
        assert [combo_price(c) for c in combos] == [2133, 2334, 2418, 8685]
        assert no_sleep, "expected backoff between attempts"

    def test_persistently_refused_option_is_skipped_with_a_warning(
        self, monkeypatch, no_sleep, caplog
    ):
        self._serve(monkeypatch, lambda url, n: b"2222" in _tfs_bytes(url))
        with caplog.at_level("WARNING", logger="fli.search.flights"):
            combos = SearchFlights().search(_open_jaw(sort_by=SortBy.CHEAPEST), top_n=2)

        assert {c[0].legs[0].flight_number for c in combos} == {"1111"}
        assert any("Skipped 1 of 2" in r.getMessage() for r in caplog.records)

    def test_every_option_refused_raises_retryable(self, monkeypatch, no_sleep):
        self._serve(
            monkeypatch, lambda url, n: b"1111" in _tfs_bytes(url) or b"2222" in _tfs_bytes(url)
        )
        with pytest.raises(SearchRejectedError) as info:
            SearchFlights().search(_open_jaw(), top_n=2)
        assert info.value.retryable is True

    def test_first_board_refused_on_every_attempt_raises_retryable(self, monkeypatch, no_sleep):
        from fli.search.flights import MULTI_CITY_ATTEMPTS

        counts = self._serve(monkeypatch, lambda url, n: True)
        with pytest.raises(SearchRejectedError) as info:
            SearchFlights().search(_open_jaw())
        assert info.value.retryable is True
        assert sum(counts.values()) == MULTI_CITY_ATTEMPTS

    def test_other_rejections_are_not_retried(self, monkeypatch, no_sleep):
        calls: list[str] = []

        def _fetch(url: str) -> str:
            calls.append(url)
            return INVALID

        monkeypatch.setattr(_browser, "fetch_shopping_body", _fetch)
        with pytest.raises(SearchRejectedError) as info:
            SearchFlights().search(_open_jaw())
        assert info.value.code == 3
        assert len(calls) == 1
        assert no_sleep == []
