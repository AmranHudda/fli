"""Price a multi-city trip as separate one-way tickets, for comparison.

Google prices a multi-city trip as one ticket. Buying each leg as its own
one-way is sometimes cheaper and sometimes much dearer, and the only way to
know is to price both. This module runs one one-way search per leg (over the
fast plain-HTTP page transport, in parallel) and keeps each leg's cheapest
fare, so a caller can put the sum next to the one-ticket price.

Separate tickets trade away protection: each leg is its own booking, so a
delay that makes you miss a later leg is your problem, and checked bags are
not transferred between tickets. :data:`SEPARATE_TICKETS_NOTE` says so for
every surface that shows the comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fli.models import FlightResult, FlightSearchFilters
from fli.models.google_flights.base import SortBy, TripType
from fli.search._concurrency import parallel_map
from fli.search.flights import SearchFlights, combo_price

SEPARATE_TICKETS_NOTE = (
    "Separate tickets are separate bookings: if a delay makes you miss a later "
    "leg, that airline owes you nothing, and checked bags are not transferred "
    "between tickets. Usually fine when the legs are days apart."
)


@dataclass(frozen=True)
class SeparateTickets:
    """The cheapest one-way fare found for each leg, in leg order.

    ``legs[i]`` is ``None`` when the one-way search for leg ``i`` found no
    priced flight, in which case :attr:`total` is ``None`` too: a partial
    sum would understate the trip.
    """

    legs: tuple[FlightResult | None, ...]

    @property
    def complete(self) -> bool:
        """Whether every leg has a priced flight."""
        return all(leg is not None and leg.price is not None for leg in self.legs)

    @property
    def total(self) -> float | None:
        """Sum of the per-leg fares, or ``None`` if any leg is missing."""
        if not self.complete:
            return None
        return sum(leg.price for leg in self.legs)  # type: ignore[union-attr, misc]

    @property
    def currency(self) -> str | None:
        """Currency of the first priced leg."""
        return next((leg.currency for leg in self.legs if leg and leg.currency), None)


def leg_filters(filters: FlightSearchFilters, index: int) -> FlightSearchFilters:
    """Return one-way, cheapest-first filters for leg ``index`` of ``filters``.

    Every other filter (cabin, stops, airlines, alliances, layovers,
    passengers, the leg's own time window) carries over, so the two prices
    being compared are for the same kind of trip.
    """
    segment = filters.flight_segments[index].model_copy(deep=True)
    segment.selected_flight = None
    return filters.model_copy(
        update={
            "trip_type": TripType.ONE_WAY,
            "flight_segments": [segment],
            "sort_by": SortBy.CHEAPEST,
        },
        deep=True,
    )


def search_separate_tickets(
    filters: FlightSearchFilters,
    *,
    currency: str | None = None,
    language: str | None = None,
    country: str | None = None,
    search_factory: Callable[[], Any] = SearchFlights,
) -> SeparateTickets:
    """Search every leg of ``filters`` as a one-way and keep each leg's cheapest fare.

    Legs are searched in parallel, each with its own ``SearchFlights`` (the
    class is not thread-safe). A leg whose search raises propagates the
    error; a leg with no priced flight comes back as ``None``.

    Args:
        filters: A multi-city (or any multi-segment) search.
        currency: ISO 4217 code for prices; use the same as the one-ticket
            search so the two totals are comparable.
        language: Optional ``hl`` locale.
        country: Optional ``gl`` locale.
        search_factory: Builds the per-leg searcher. Tests inject fakes here.

    """

    def cheapest(index: int) -> FlightResult | None:
        results = search_factory().search(
            leg_filters(filters, index), currency=currency, language=language, country=country
        )
        priced = [r for r in results or [] if isinstance(r, FlightResult) and r.price is not None]
        return min(priced, key=lambda r: r.price) if priced else None

    return SeparateTickets(legs=tuple(parallel_map(cheapest, range(len(filters.flight_segments)))))


def compare_with_one_ticket(
    separate: SeparateTickets,
    combos: list[tuple[FlightResult, ...]] | None,
) -> dict[str, Any]:
    """Summarise separate tickets against the cheapest one-ticket combination.

    Returns plain numbers so the CLI and MCP can share them:
    ``separate_total``, ``one_ticket_total`` (``None`` when the multi-city
    search found nothing), ``difference`` (separate minus one ticket, so
    negative means separate tickets are cheaper), ``cheaper``
    (``"separate"``, ``"one_ticket"``, ``"same"`` or ``None`` when either side
    is missing), ``currency`` and ``note``.
    """
    one_ticket_prices = [p for p in (combo_price(c) for c in combos or []) if p is not None]
    one_ticket = min(one_ticket_prices) if one_ticket_prices else None
    separate_total = separate.total

    difference = cheaper = None
    if one_ticket is not None and separate_total is not None:
        difference = round(separate_total - one_ticket, 2)
        cheaper = "separate" if difference < 0 else "one_ticket" if difference > 0 else "same"

    return {
        "separate_total": separate_total,
        "one_ticket_total": one_ticket,
        "difference": difference,
        "cheaper": cheaper,
        "currency": separate.currency,
        "note": SEPARATE_TICKETS_NOTE,
    }
