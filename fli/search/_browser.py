"""Browser transport for the searches the plain-HTTP page transport cannot serve.

Since 2026-08 Google's ``GetShoppingResults`` RPC only answers requests that
carry an ``x-goog-batchexecute-bgr`` header, and that header is produced by
the search page's own JavaScript for each exact request body (see
:mod:`fli.search._tfs`). One-way and round-trip searches sidestep it by
reading the rows Google inlines into the page. Multi-city boards are not
inlined: the page fetches them client-side through the gated RPC.

This module loads that page in a real Chrome (via Playwright), lets Google's
own script issue and sign the RPC, and hands back the response body, which
decodes with the same :mod:`fli.search._wire` and
:mod:`fli.search._decoders` code the RPC transport always used. Nothing is
forged or replayed: the browser makes exactly the request a person opening
the same URL would make.

Design:

* One headless browser per process, started lazily on the first multi-city
  search and closed at exit. Launching takes a few seconds; each page after
  that costs roughly 2-6 seconds.
* Playwright's async API runs on a private event loop in a daemon thread.
  :func:`fetch_shopping_body` is a plain blocking call that is safe to make
  from any thread, so the round-trip/multi-city expansion workers in
  :func:`fli.search._concurrency.parallel_map` can share the one browser.
* At most ``FLI_BROWSER_PAGES`` pages (default 4) load at once.

Configuration (environment variables):

* ``FLI_BROWSER_CHANNEL``: Playwright channel to launch. Defaults to
  ``chrome`` (the installed Google Chrome), falling back to Playwright's
  bundled Chromium when Chrome is not installed. Set it to ``chromium`` to
  use the bundled build directly.
* ``FLI_BROWSER_HEADLESS``: ``0`` to show the browser window (debugging).
* ``FLI_BROWSER_PAGES``: maximum concurrent page loads.
* ``FLI_BROWSER_TIMEOUT``: seconds to wait for the results RPC per page
  (default 20; it normally arrives within 1-3 seconds).

Requires the optional ``browser`` extra: ``pip install 'flights[browser]'``.
If Chrome is not installed, also run ``playwright install chromium``.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import threading
from typing import Any

from fli.search.client import SOCS_COOKIE
from fli.search.exceptions import BrowserUnavailableError, SearchTimeoutError

logger = logging.getLogger(__name__)

SHOPPING_RPC = "GetShoppingResults"

INSTALL_HINT = (
    "Multi-city search loads Google Flights in a headless browser and needs the "
    "optional 'browser' extra: pip install 'flights[browser]' (or "
    "uv tool install 'flights[browser]'). If Google Chrome is not installed, "
    "also run: playwright install chromium"
)

# Resource types that never affect the results RPC. Skipping them makes each
# page load noticeably faster and lighter.
_BLOCKED_RESOURCES = frozenset({"image", "media", "font"})


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", name, raw)
        return default


class BrowserFetcher:
    """A shared headless browser that returns ``GetShoppingResults`` bodies.

    Use :func:`get_browser_fetcher` rather than constructing one directly, so
    the whole process shares one browser.
    """

    def __init__(
        self,
        *,
        channel: str | None = None,
        headless: bool | None = None,
        max_pages: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """Configure the fetcher. Nothing is launched until the first fetch."""
        self.channel = channel or os.environ.get("FLI_BROWSER_CHANNEL", "chrome")
        if headless is None:
            headless = os.environ.get("FLI_BROWSER_HEADLESS", "1") not in ("0", "false", "no")
        self.headless = headless
        self.max_pages = max_pages or _env_int("FLI_BROWSER_PAGES", 4)
        self.timeout = timeout or float(_env_int("FLI_BROWSER_TIMEOUT", 20))

        self._thread_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # Created on the loop thread (asyncio primitives bind to their loop).
        self._start_lock: asyncio.Lock | None = None
        self._pages: asyncio.Semaphore | None = None
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None

    # -- lifecycle -----------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._thread_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(target=loop.run_forever, name="fli-browser", daemon=True)
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    async def _ensure_browser(self) -> Any:
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
            self._pages = asyncio.Semaphore(self.max_pages)
        async with self._start_lock:
            if self._context is not None:
                return self._context
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise BrowserUnavailableError(INSTALL_HINT) from exc

            self._playwright = await async_playwright().start()
            self._browser = await self._launch(self._playwright)
            context = await self._browser.new_context(locale="en-US")
            if SOCS_COOKIE:
                await context.add_cookies(
                    [
                        {
                            "name": "SOCS",
                            "value": SOCS_COOKIE,
                            "domain": ".google.com",
                            "path": "/",
                        }
                    ]
                )
            await context.route("**/*", _skip_heavy_resources)
            self._context = context
            return context

    async def _launch(self, playwright: Any) -> Any:
        """Launch the configured channel, falling back to bundled Chromium."""
        chromium = playwright.chromium
        channel = None if self.channel in ("", "chromium") else self.channel
        try:
            return await chromium.launch(channel=channel, headless=self.headless)
        except Exception as first:  # noqa: BLE001 — Playwright raises its own Error type
            if channel is None:
                raise BrowserUnavailableError(f"{INSTALL_HINT} ({first})") from first
            logger.info("Could not launch %s (%s); trying bundled Chromium", channel, first)
            try:
                return await chromium.launch(headless=self.headless)
            except Exception as second:  # noqa: BLE001
                raise BrowserUnavailableError(f"{INSTALL_HINT} ({second})") from second

    async def _aclose(self) -> None:
        for closer in (self._context, self._browser):
            if closer is not None:
                try:
                    await closer.close()
                except Exception:  # noqa: BLE001 — best effort at shutdown
                    logger.debug("browser close failed", exc_info=True)
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # noqa: BLE001
                logger.debug("playwright stop failed", exc_info=True)
        self._context = self._browser = self._playwright = None

    def close(self) -> None:
        """Close the browser and stop the loop thread. Safe to call twice."""
        with self._thread_lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None:
            return
        if loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._aclose(), loop).result(timeout=10)
            except Exception:  # noqa: BLE001
                logger.debug("browser shutdown did not finish cleanly", exc_info=True)
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        self._start_lock = self._pages = None

    # -- fetching ------------------------------------------------------------

    async def _fetch(self, url: str, rpc: str) -> str:
        context = await self._ensure_browser()
        assert self._pages is not None
        async with self._pages:
            page = await context.new_page()
            try:
                async with page.expect_response(
                    lambda response: rpc in response.url, timeout=self.timeout * 1000
                ) as response_info:
                    await page.goto(url, wait_until="commit")
                response = await response_info.value
                return await response.text()
            finally:
                await page.close()

    def fetch(self, url: str, rpc: str = SHOPPING_RPC, attempts: int = 2) -> str:
        """Load ``url`` and return the body of the first ``rpc`` response.

        Blocking and thread-safe. Retries a page that never issued the RPC
        (a slow or odd page variant) once before giving up.

        Raises:
            BrowserUnavailableError: Playwright or a browser is not installed.
            SearchTimeoutError: The page never made the results request.

        """
        loop = self._ensure_loop()
        last: Exception | None = None
        for attempt in range(attempts):
            future = asyncio.run_coroutine_threadsafe(self._fetch(url, rpc), loop)
            try:
                return future.result(timeout=self.timeout + 30)
            except BrowserUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 — Playwright TimeoutError and friends
                last = exc
                logger.debug("browser fetch attempt %d/%d failed: %s", attempt + 1, attempts, exc)
        raise SearchTimeoutError(
            f"Google Flights did not return results in the browser after {attempts} "
            f"attempts ({type(last).__name__}: {last}). Page: {url}"
        ) from last


async def _skip_heavy_resources(route: Any) -> None:
    if route.request.resource_type in _BLOCKED_RESOURCES:
        await route.abort()
    else:
        await route.continue_()


_shared: BrowserFetcher | None = None
_shared_lock = threading.Lock()


def get_browser_fetcher() -> BrowserFetcher:
    """Return the process-wide :class:`BrowserFetcher`, creating it on first use."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = BrowserFetcher()
            atexit.register(_shared.close)
        return _shared


def fetch_shopping_body(url: str) -> str:
    """Load a search-page URL in the shared browser; return the results RPC body."""
    return get_browser_fetcher().fetch(url)
