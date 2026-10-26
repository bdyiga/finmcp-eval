"""Small SEC EDGAR client: ticker -> CIK lookup and Company Facts, with a local JSON cache.

Safety properties:
- Only https requests to data.sec.gov and www.sec.gov; redirects are not followed.
- Cache paths are built from validated integers (CIK), never from raw input.
- Requests are spaced out (SEC allows at most 10 requests per second).
- Retries are bounded and only for timeouts, 429, and 5xx responses.
- Cache files are written atomically (temp file + os.replace).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from .schemas import normalize_ticker

log = logging.getLogger(__name__)

ALLOWED_HOSTS = frozenset({"data.sec.gov", "www.sec.gov"})
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

MAX_RESPONSE_BYTES = 150 * 1024 * 1024  # The largest Company Facts files are tens of MB.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_WAIT_SECONDS = 60.0


class SecError(RuntimeError):
    """A user-facing error. Messages contain no secrets."""


class SecClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        *,
        cache_ttl_days: float = 30.0,
        min_interval: float = 0.15,
        max_retries: int = 2,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_ttl_seconds = cache_ttl_days * 86400
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._last_request = float("-inf")
        self._lock = threading.Lock()
        self._ticker_map: dict[str, tuple[int, str]] | None = None
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=httpx.Timeout(timeout, connect=10.0),
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    # -- public API ----------------------------------------------------------

    def lookup_cik(self, ticker: str) -> tuple[int, str]:
        """Return (cik, company title) for a ticker, or raise SecError."""
        ticker = normalize_ticker(ticker)
        if self._ticker_map is None:
            raw = self._cached_json(self.cache_dir / "company_tickers.json", TICKER_MAP_URL)
            self._ticker_map = _parse_ticker_map(raw)
        try:
            return self._ticker_map[ticker]
        except KeyError:
            raise SecError(f"ticker {ticker} was not found in SEC's company ticker list") from None

    def company_facts(self, cik: int) -> dict[str, Any]:
        if not isinstance(cik, int) or isinstance(cik, bool) or not 0 < cik < 10**10:
            raise SecError("invalid CIK")
        path = self.cache_dir / "companyfacts" / f"CIK{cik:010d}.json"
        doc = self._cached_json(path, COMPANY_FACTS_URL.format(cik=cik))
        if not isinstance(doc, dict) or "facts" not in doc:
            raise SecError("unexpected Company Facts response format")
        return doc

    # -- cache ----------------------------------------------------------------

    def _cached_json(self, path: Path, url: str) -> Any:
        path = path.resolve()
        if not path.is_relative_to(self.cache_dir):
            raise SecError("refusing to use a cache path outside the cache directory")
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if self.cache_ttl_seconds <= 0 or age <= self.cache_ttl_seconds:
                try:
                    with path.open("r", encoding="utf-8") as fh:
                        log.info("cache hit: %s", path.name)
                        return json.load(fh)
                except (OSError, json.JSONDecodeError):
                    log.warning("cache file %s is unreadable; refetching", path.name)
        data = self._get_json(url)
        write_json_atomic(path, data)
        return data

    # -- HTTP -----------------------------------------------------------------

    def _get_json(self, url: str) -> Any:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise SecError("refusing to contact a host other than the SEC API")

        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                body, status, headers = self._fetch(url)
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    self._backoff(attempt, None)
                    continue
                raise SecError("SEC request timed out") from None
            except httpx.HTTPError as exc:
                raise SecError(f"SEC request failed: {type(exc).__name__}") from None

            if status == 200:
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    raise SecError("SEC returned invalid JSON") from None
            if status in RETRY_STATUSES and attempt < self.max_retries:
                self._backoff(attempt, headers.get("Retry-After"))
                continue
            raise SecError(_status_message(status))
        raise SecError("SEC request failed after retries")  # pragma: no cover

    def _fetch(self, url: str) -> tuple[bytes, int, httpx.Headers]:
        with self._http.stream("GET", url) as resp:
            if resp.status_code != 200:
                return b"", resp.status_code, resp.headers
            chunks, size = [], 0
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise SecError("SEC response is unexpectedly large")
                chunks.append(chunk)
            return b"".join(chunks), 200, resp.headers

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_request + self.min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._last_request = self._clock()

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = 2.0 * (2**attempt)
        if retry_after and retry_after.strip().isdigit():
            delay = float(retry_after.strip())
        delay = min(delay, MAX_RETRY_WAIT_SECONDS)
        log.warning("SEC request throttled or failed; retrying in %.1fs", delay)
        self._sleep(delay)


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON to a temp file in the same directory, then atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _parse_ticker_map(raw: Any) -> dict[str, tuple[int, str]]:
    """SEC's file looks like {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}."""
    if not isinstance(raw, dict):
        raise SecError("unexpected company ticker list format")
    out: dict[str, tuple[int, str]] = {}
    for row in raw.values():
        if not isinstance(row, dict):
            continue
        ticker, cik, title = row.get("ticker"), row.get("cik_str"), row.get("title", "")
        if isinstance(ticker, str) and isinstance(cik, int) and not isinstance(cik, bool):
            out.setdefault(ticker.upper().replace(".", "-"), (cik, str(title)))
    return out


def _status_message(status: int) -> str:
    if status == 403:
        return (
            "SEC returned 403 Forbidden. Check that SEC_CONTACT_NAME and SEC_CONTACT_EMAIL are set "
            "and that requests are not too frequent."
        )
    if status == 404:
        return "SEC has no data at this address (404); the company may not file XBRL financial data"
    if status == 429:
        return "SEC rate limit reached (429); wait a few minutes and try again"
    if 300 <= status < 400:
        return f"SEC returned an unexpected redirect ({status})"
    return f"SEC request failed with HTTP {status}"
