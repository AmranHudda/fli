"""Tests for pricing a multi-city trip as separate one-way tickets."""

from __future__ import annotations

import pytest

from fli.models import Airline, Airport, MaxStops, PassengerInfo, SeatType, SortBy
from fli.models.google_flights.base import TripType
from fli.search.flights import SearchFlights
from fli.search.separate import (
    SEPARATE_TICKETS_NOTE,
    SeparateTickets,
    compare_with_one_ticket,
    leg_filters,
    search_separate_tickets,
)
from tests.search.test_multi_city import _filters, _flight, _future


def _result(airline: str, number: str, dep: str, arr: str, price: float | None):
    result = SearchFlights._parse_flights_data(_flight(airline, number, dep, arr, price or 1))
    result.price = price
    return result


def _trip(**overrides):
    return _filters(
        [(Airport.BOS, Airport.FCO, _future(30)), (Airport.FCO, Airport.ORD, _future(44))],
        **overrides,
    )


class _FakeSearch:
    """Serve canned one-way results keyed by each leg's origin airport."""

    boards: dict[Airport, list] = {}
    seen: list = []

    def search(self, filters, **kwargs):  # noqa: ANN001
        type(self).seen.append(filters)
        origin = filters.flight_segments[0].departure_airport[0][0]
        return self.boards.get(origin)


@pytest.fixture
def fake_search():
    _FakeSearch.seen = []
    _FakeSearch.boards = {
        Airport.BOS: [
            _result("UA", "1796", "BOS", "FCO", 1236),
            _result("AC", "8837", "BOS", "FCO", 1360),
        ],
        Airport.FCO: [
            _result("CX", "694", "FCO", "ORD", 900),
            _result("LH", "763", "FCO", "ORD", 628),
            _result("AI", "173", "FCO", "ORD", None),
        ],
    }
    return _FakeSearch


class TestLegFilters:
    def test_each_leg_becomes_a_cheapest_first_one_way(self):
        trip = _trip(
            seat_type=SeatType.BUSINESS,
            stops=MaxStops.ONE_STOP_OR_FEWER,
            airlines=[Airline.UA],
            passenger_info=PassengerInfo(adults=2),
            sort_by=SortBy.DURATION,
        )
        second = leg_filters(trip, 1)

        assert second.trip_type == TripType.ONE_WAY
        assert second.sort_by == SortBy.CHEAPEST
        assert len(second.flight_segments) == 1
        assert second.flight_segments[0].departure_airport[0][0] == Airport.FCO
        assert second.flight_segments[0].arrival_airport[0][0] == Airport.ORD
        # Everything that defines the kind of trip carries over.
        assert second.seat_type == SeatType.BUSINESS
        assert second.stops == MaxStops.ONE_STOP_OR_FEWER
        assert second.airlines == [Airline.UA]
        assert second.passenger_info.adults == 2

    def test_does_not_mutate_the_original(self):
        trip = _trip()
        leg_filters(trip, 0)
        assert trip.trip_type == TripType.MULTI_CITY
        assert len(trip.flight_segments) == 2

    def test_clears_a_pinned_flight(self):
        trip = _trip()
        trip.flight_segments[0].selected_flight = _result("UA", "1", "BOS", "FCO", 1)
        assert leg_filters(trip, 0).flight_segments[0].selected_flight is None


class TestSearchSeparateTickets:
    def test_keeps_each_legs_cheapest_priced_fare(self, fake_search):
        separate = search_separate_tickets(_trip(), currency="USD", search_factory=fake_search)

        assert [leg.legs[0].flight_number for leg in separate.legs] == ["1796", "763"]
        assert separate.complete
        assert separate.total == 1864
        assert len(fake_search.seen) == 2
        assert all(f.trip_type == TripType.ONE_WAY for f in fake_search.seen)

    def test_a_leg_with_no_fare_leaves_the_total_unknown(self, fake_search):
        fake_search.boards[Airport.FCO] = None
        separate = search_separate_tickets(_trip(), search_factory=fake_search)

        assert separate.legs[1] is None
        assert not separate.complete
        assert separate.total is None

    def test_a_failing_leg_raises(self):
        class _Boom:
            def search(self, filters, **kwargs):  # noqa: ANN001
                raise RuntimeError("network down")

        with pytest.raises(RuntimeError, match="network down"):
            search_separate_tickets(_trip(), search_factory=_Boom)


class TestCompareWithOneTicket:
    @staticmethod
    def _separate(*prices):
        return SeparateTickets(
            legs=tuple(_result("UA", str(i), "BOS", "FCO", p) for i, p in enumerate(prices))
        )

    @staticmethod
    def _combo(price):
        return (
            _result("UA", "1", "BOS", "FCO", price + 500),
            _result("LH", "2", "FCO", "ORD", price),
        )

    def test_separate_cheaper(self):
        summary = compare_with_one_ticket(
            self._separate(1236, 628), [self._combo(2418), self._combo(2133)]
        )
        assert summary["separate_total"] == 1864
        assert summary["one_ticket_total"] == 2133
        assert summary["difference"] == -269
        assert summary["cheaper"] == "separate"
        assert summary["note"] == SEPARATE_TICKETS_NOTE

    def test_one_ticket_cheaper(self):
        summary = compare_with_one_ticket(self._separate(1500, 900), [self._combo(2133)])
        assert summary["difference"] == 267
        assert summary["cheaper"] == "one_ticket"

    def test_same_price(self):
        summary = compare_with_one_ticket(self._separate(1000, 1133), [self._combo(2133)])
        assert summary["cheaper"] == "same"

    def test_no_one_ticket_results(self):
        summary = compare_with_one_ticket(self._separate(1236, 628), None)
        assert summary["separate_total"] == 1864
        assert summary["one_ticket_total"] is None
        assert summary["difference"] is None
        assert summary["cheaper"] is None
