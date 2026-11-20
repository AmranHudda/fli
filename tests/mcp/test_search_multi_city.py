"""Tests for the ``search_multi_city`` MCP tool (browser faked; see test_multi_city.py)."""

from __future__ import annotations

import base64
from urllib.parse import parse_qs, urlparse

from fli.mcp.server import MultiCityLeg, MultiCitySearchParams, _execute_multi_city_search
from fli.search import _browser
from fli.search._browser import INSTALL_HINT
from fli.search.exceptions import BrowserUnavailableError
from tests.search.test_multi_city import FIRST_BOARD, SECOND_BOARDS, _future, _rpc_body


def _params(**overrides) -> MultiCitySearchParams:
    return MultiCitySearchParams(
        legs=[
            MultiCityLeg(origin="BOS", destination="FCO", departure_date=_future(30)),
            MultiCityLeg(origin="FCO", destination="ORD", departure_date=_future(44)),
        ],
        sort_by="CHEAPEST",
        **{"compare_separate_tickets": False, **overrides},
    )


def _trip_code(url: str) -> int:
    tfs = parse_qs(urlparse(url).query)["tfs"][0]
    raw = base64.urlsafe_b64decode(tfs + "=" * (-len(tfs) % 4))
    return raw[raw.rindex(b"\x98\x01") + 2]


def _serve_boards(monkeypatch) -> list[str]:
    urls: list[str] = []

    def _fetch(url: str) -> str:
        urls.append(url)
        tfs = parse_qs(urlparse(url).query)["tfs"][0]
        raw = base64.urlsafe_b64decode(tfs + "=" * (-len(tfs) % 4))
        for number, rows in SECOND_BOARDS.items():
            if number in raw:
                return _rpc_body(rows)
        return _rpc_body(FIRST_BOARD)

    monkeypatch.setattr(_browser, "fetch_shopping_body", _fetch)
    return urls


def test_returns_whole_trips_priced_by_their_real_total(monkeypatch):
    urls = _serve_boards(monkeypatch)
    response = _execute_multi_city_search(_params(top_n=2))

    assert response["success"] is True
    assert response["trip_type"] == "MULTI_CITY"
    assert response["pages_loaded"] == len(urls) == 3
    assert [f["price"] for f in response["flights"]] == [2133, 2334, 2418, 8685]

    first = response["flights"][0]
    assert [leg["flight_number"] for leg in first["legs"]] == ["1111", "3333"]
    assert _trip_code(first["booking_url"]) == 3
    assert _trip_code(response["search_url"]) == 3


def test_missing_browser_is_a_typed_unsupported_error(monkeypatch):
    def _missing(url: str) -> str:
        raise BrowserUnavailableError(INSTALL_HINT)

    monkeypatch.setattr(_browser, "fetch_shopping_body", _missing)
    response = _execute_multi_city_search(_params())

    assert response["success"] is False
    assert response["error_type"] == "unsupported_error"
    assert response["retryable"] is False
    assert "flights[browser]" in response["error"]


def test_too_many_pages_is_a_validation_error(monkeypatch):
    urls = _serve_boards(monkeypatch)
    legs = [
        MultiCityLeg(origin=o, destination=d, departure_date=_future(30 + 3 * i))
        for i, (o, d) in enumerate(
            [("JFK", "LHR"), ("LHR", "CDG"), ("CDG", "FCO"), ("FCO", "ATH"), ("ATH", "JFK")]
        )
    ]
    response = _execute_multi_city_search(MultiCitySearchParams(legs=legs, top_n=3))

    assert response["success"] is False
    assert response["error_type"] == "validation_error"
    assert "top_n" in response["error"]
    assert urls == []


def test_unknown_airport_is_a_validation_error(monkeypatch):
    _serve_boards(monkeypatch)
    response = _execute_multi_city_search(
        MultiCitySearchParams(
            legs=[
                MultiCityLeg(origin="ZZZ", destination="FCO", departure_date=_future(30)),
                MultiCityLeg(origin="FCO", destination="ORD", departure_date=_future(44)),
            ]
        )
    )
    assert response["success"] is False
    assert response["error_type"] == "validation_error"


def test_separate_tickets_are_compared_by_default(monkeypatch):
    from fli.search.separate import SeparateTickets
    from tests.search.test_separate_tickets import _result

    _serve_boards(monkeypatch)
    captured = {}

    def _separate(filters, **kwargs):  # noqa: ANN001
        captured["legs"] = len(filters.flight_segments)
        return SeparateTickets(
            legs=(
                _result("UA", "1796", "BOS", "FCO", 1236),
                _result("LH", "763", "FCO", "ORD", 628),
            )
        )

    monkeypatch.setattr("fli.mcp.server.search_separate_tickets", _separate)
    response = _execute_multi_city_search(_params(top_n=2, compare_separate_tickets=True))

    summary = response["separate_tickets"]
    assert captured["legs"] == 2
    assert summary["separate_total"] == 1864
    assert summary["one_ticket_total"] == 2133
    assert summary["difference"] == -269
    assert summary["cheaper"] == "separate"
    assert [leg["legs"][0]["flight_number"] for leg in summary["legs"]] == ["1796", "763"]
    assert _trip_code(summary["legs"][0]["booking_url"]) == 2  # one-way deep link
    assert "note" in summary


def test_separate_ticket_failure_does_not_fail_the_search(monkeypatch):
    _serve_boards(monkeypatch)

    def _boom(filters, **kwargs):  # noqa: ANN001
        raise RuntimeError("one-way search exploded")

    monkeypatch.setattr("fli.mcp.server.search_separate_tickets", _boom)
    response = _execute_multi_city_search(_params(top_n=2, compare_separate_tickets=True))

    assert response["success"] is True
    assert response["count"] == 4
    assert "one-way search exploded" in response["separate_tickets"]["error"]
