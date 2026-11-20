"""Live multi-city search through a real headless browser.

Skipped unless ``--live`` is given, and skipped (not failed) when the
optional ``browser`` extra or a browser binary is missing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fli.models import Airport, FlightSearchFilters, FlightSegment, PassengerInfo, SortBy
from fli.models.google_flights.base import TripType
from fli.search.exceptions import BrowserUnavailableError
from fli.search.flights import SearchFlights, combo_price

pytestmark = pytest.mark.live


def _future(days: int) -> str:
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")


def test_open_jaw_returns_priced_combinations():
    pytest.importorskip("playwright")
    filters = FlightSearchFilters(
        trip_type=TripType.MULTI_CITY,
        sort_by=SortBy.CHEAPEST,
        passenger_info=PassengerInfo(adults=1),
        flight_segments=[
            FlightSegment(
                departure_airport=[[Airport.JFK, 0]],
                arrival_airport=[[Airport.LHR, 0]],
                travel_date=_future(45),
            ),
            FlightSegment(
                departure_airport=[[Airport.CDG, 0]],
                arrival_airport=[[Airport.JFK, 0]],
                travel_date=_future(52),
            ),
        ],
    )
    try:
        combos = SearchFlights().search(filters, top_n=2)
    except BrowserUnavailableError as exc:
        pytest.skip(str(exc))

    assert combos, "expected at least one JFK-LHR / CDG-JFK combination"
    for out, back in combos:
        assert out.legs[0].departure_airport == Airport.JFK
        assert out.legs[-1].arrival_airport == Airport.LHR
        assert back.legs[0].departure_airport == Airport.CDG
        assert back.legs[-1].arrival_airport == Airport.JFK
    prices = [combo_price(c) for c in combos]
    assert all(p and p > 0 for p in prices)
    assert prices == sorted(prices)
