"""Multi-city flight search CLI command."""

import re
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from fli.cli.enums import OutputFormat
from fli.cli.errors import json_error_payload, report_cli_error
from fli.cli.utils import (
    build_json_error_response,
    build_json_success_response,
    display_flight_results,
    emit_json,
    serialize_flight_result,
    validate_currency,
    validate_time_range,
)
from fli.core import (
    build_multi_city_segments,
    classify_error,
    format_validation_error,
    normalize_date,
    parse_airlines,
    parse_cabin_class,
    parse_max_stops,
    parse_sort_by,
    resolve_airport,
)
from fli.core.parsers import ParseError
from fli.models import (
    FlightSearchFilters,
    PassengerInfo,
    TimeRestrictions,
)
from fli.search import SearchClientError, SearchFlights
from fli.search._tfs import build_tfs, page_url

LEG_PATTERN = re.compile(r"^([A-Za-z]{3}),([A-Za-z]{3}),(\d{4}-\d{1,2}-\d{1,2})$")


def _parse_leg(value: str) -> tuple[str, str, str]:
    """Parse a leg string in ORIGIN,DEST,DATE format.

    Returns:
        Tuple of (origin, destination, date).

    Raises:
        typer.BadParameter: If the format is invalid.

    """
    match = LEG_PATTERN.match(value)
    if not match:
        raise typer.BadParameter(
            f"Invalid leg format: '{value}'. Expected ORIGIN,DEST,DATE (e.g., SEA,HKG,2026-12-26)"
        )
    return match.group(1).upper(), match.group(2).upper(), match.group(3)


def multi(
    legs: Annotated[
        list[str],
        typer.Option(
            "--leg",
            "-l",
            help="Flight leg in ORIGIN,DEST,DATE format (repeatable, minimum 2)",
        ),
    ],
    departure_window: Annotated[
        str | None,
        typer.Option(
            "--time",
            "-t",
            help="Departure time window in 24h format (e.g., 6-20)",
            callback=validate_time_range,
        ),
    ] = None,
    airlines: Annotated[
        list[str] | None,
        typer.Option(
            "--airlines",
            "-a",
            help="Airline IATA codes (e.g., BA,KL or repeated --airlines BA --airlines KL)",
        ),
    ] = None,
    cabin_class: Annotated[
        str,
        typer.Option(
            "--class",
            "-c",
            help="Cabin class (ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST)",
        ),
    ] = "ECONOMY",
    max_stops: Annotated[
        str,
        typer.Option(
            "--stops",
            "-s",
            help="Maximum stops (ANY, 0 for non-stop, 1 for one stop, 2+ for two stops)",
        ),
    ] = "ANY",
    sort_by: Annotated[
        str,
        typer.Option(
            "--sort",
            "-o",
            help="Sort results by (CHEAPEST, DURATION, DEPARTURE_TIME, ARRIVAL_TIME)",
        ),
    ] = "CHEAPEST",
    passengers: Annotated[
        int,
        typer.Option(
            "--passengers",
            "-p",
            help="Number of adult passengers",
            min=1,
        ),
    ] = 1,
    children: Annotated[
        int,
        typer.Option(
            "--children",
            help="Number of children",
            min=0,
        ),
    ] = 0,
    infants_in_seat: Annotated[
        int,
        typer.Option(
            "--infants-in-seat",
            help="Number of infants in seat",
            min=0,
        ),
    ] = 0,
    infants_on_lap: Annotated[
        int,
        typer.Option(
            "--infants-on-lap",
            help="Number of infants on lap",
            min=0,
        ),
    ] = 0,
    top_n: Annotated[
        int,
        typer.Option(
            "--top-n",
            "-n",
            help=(
                "Options per leg to expand into the next leg. A trip with k legs loads "
                "1 + n + ... + n^(k-1) pages in a browser, so keep it small for 3+ legs."
            ),
            min=1,
            max=10,
        ),
    ] = 3,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            help="Output format: text or json",
            case_sensitive=False,
        ),
    ] = OutputFormat.TEXT,
    currency: Annotated[
        str,
        typer.Option(
            "--currency",
            callback=validate_currency,
            help="Currency code (USD, EUR, GBP, JPY...). Passed to Google as `curr=`.",
        ),
    ] = "USD",
):
    """Search for multi-city flights with multiple legs.

    Each leg specifies an origin, destination, and date. At least two legs are required.
    Prices are for the whole trip on one ticket, as Google Flights quotes multi-city.

    Multi-city results load in a headless browser, so this command needs the
    optional browser extra: pip install 'flights[browser]'.

    Example:
        fli multi --leg SEA,HKG,2026-12-26 --leg PEK,SEA,2027-01-02
        fli multi -l SEA,NRT,2026-12-26 -l NRT,HKG,2026-12-30 -l HKG,SEA,2027-01-05 -c BUSINESS
        fli multi -l SEA,NRT,2026-12-26 -l HKG,SEA,2027-01-05 --passengers 2
        fli multi -l SEA,NRT,2026-12-26 -l HKG,SEA,2027-01-05 --passengers 2 --children 1

    """
    query: dict[str, Any] = {
        "legs": legs,
        "cabin_class": cabin_class.upper(),
        "max_stops": max_stops.upper(),
        "sort_by": sort_by.upper(),
        "top_n": top_n,
        "currency": currency,
    }
    json_mode = output_format == OutputFormat.JSON

    def fail(exc: Exception, message: str) -> typer.Exit:
        """Report an input or search error in the requested format."""
        if json_mode:
            emit_json(
                build_json_error_response(
                    search_type="flights",
                    message=message,
                    query=query,
                    **classify_error(exc).as_fields(),
                )
            )
        else:
            typer.echo(f"Error: {message}")
        return typer.Exit(1)

    try:
        if len(legs) < 2:
            typer.echo("Error: multi-city search requires at least 2 legs")
            raise typer.Exit(1)

        # Parse and validate each leg
        parsed_legs = []
        for leg_str in legs:
            origin, destination, date = _parse_leg(leg_str)
            date = normalize_date(date)
            origin_airport = resolve_airport(origin)
            destination_airport = resolve_airport(destination)
            parsed_legs.append((origin_airport, destination_airport, date))
        query["legs"] = [
            {"origin": o.name.lstrip("_"), "destination": d.name.lstrip("_"), "departure_date": dt}
            for o, d, dt in parsed_legs
        ]

        # Parse shared filter parameters
        seat_type = parse_cabin_class(cabin_class)
        stops = parse_max_stops(max_stops)
        parsed_airlines = parse_airlines(airlines)
        sort = parse_sort_by(sort_by)

        # Build time restrictions
        time_restrictions = None
        if departure_window:
            time_restrictions = TimeRestrictions(
                earliest_departure=departure_window[0],
                latest_departure=departure_window[1],
            )

        # Build multi-city segments using shared builder
        segments, trip_type = build_multi_city_segments(
            legs=parsed_legs,
            time_restrictions=time_restrictions,
        )

        # Create search filters
        filters = FlightSearchFilters(
            trip_type=trip_type,
            passenger_info=PassengerInfo(
                adults=passengers,
                children=children,
                infants_in_seat=infants_in_seat,
                infants_on_lap=infants_on_lap,
            ),
            flight_segments=segments,
            stops=stops,
            seat_type=seat_type,
            airlines=parsed_airlines,
            sort_by=sort,
        )

        # The same multi-city board on Google Flights, for opening in a browser.
        search_url = page_url(build_tfs(filters, allow_multi_city=True), currency)

        search_client = SearchFlights()
        results = search_client.search(filters, top_n=top_n, currency=currency)

        if not results:
            if json_mode:
                emit_json(
                    build_json_success_response(
                        search_type="flights",
                        trip_type=trip_type,
                        query=query,
                        results_key="flights",
                        results=[],
                        booking_url=search_url,
                    )
                )
                return
            typer.echo("No flights found.")
            raise typer.Exit(1)

        booking_urls = [
            search_client.build_flight_booking_url(
                result,
                currency=currency,
                seat_type=seat_type,
                passenger_info=filters.passenger_info,
                trip_type=trip_type,
            )
            for result in results
        ]

        if json_mode:
            emit_json(
                build_json_success_response(
                    search_type="flights",
                    trip_type=trip_type,
                    query=query,
                    results_key="flights",
                    results=[
                        serialize_flight_result(result, default_currency=currency, booking_url=url)
                        for result, url in zip(results, booking_urls, strict=False)
                    ],
                    booking_url=search_url,
                )
            )
            return

        display_flight_results(
            results,
            trip_type=trip_type,
            default_currency=currency,
            booking_url=search_url,
            booking_urls=booking_urls,
        )

    except ParseError as e:
        raise fail(e, str(e)) from e
    except ValidationError as e:
        raise fail(e, format_validation_error(e)) from e
    except (AttributeError, ValueError) as e:
        raise fail(e, str(e)) from e
    except SearchClientError as e:
        if json_mode:
            info = json_error_payload(e, command="multi")
            payload = build_json_error_response(
                search_type="flights",
                message=info.message,
                error_type=info.error_type,
                retryable=info.retryable,
                http_status=info.http_status,
                query=query,
            )
            payload["error"]["log_path"] = str(info.log_path)
            emit_json(payload)
            raise typer.Exit(1) from e
        raise report_cli_error(e, command="multi") from e
    except (typer.Exit, typer.Abort):
        # click.exceptions.Exit/Abort are RuntimeError subclasses, so without
        # this clause the broad except below would catch either of the
        # deliberate `raise typer.Exit(1)` calls above (too few legs, empty
        # results) and report it as a crash: bogus "Unexpected error" text
        # plus a traceback log file for a perfectly normal outcome.
        raise
    except Exception as e:  # noqa: BLE001 — fall back to clean reporting
        raise report_cli_error(e, command="multi") from e
