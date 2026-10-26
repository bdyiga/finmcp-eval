"""Shared test fixtures. All SEC data here is synthetic and exists only for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from finmcp_eval.sec import SecClient

TEST_CIK = 1234567
TICKER_MAP = {"0": {"cik_str": TEST_CIK, "ticker": "TEST", "title": "Test Corp"}}


def fact(val, end, *, start=None, fy, form="10-K", fp="FY", filed=None, accn=None, frame=None) -> dict[str, Any]:
    f = {
        "end": end,
        "val": val,
        "accn": accn or f"0001234567-{fy % 100:02d}-000010",
        "fy": fy,
        "fp": fp,
        "form": form,
        "filed": filed or f"{fy}-02-15",
    }
    if start:
        f["start"] = start
    if frame:
        f["frame"] = frame
    return f


def facts_doc(gaap: dict[str, dict[str, list[dict[str, Any]]]], cik: int = TEST_CIK) -> dict[str, Any]:
    """Build a Company Facts document from {tag: {unit: [facts]}}."""
    return {
        "cik": cik,
        "entityName": "Test Corp",
        "facts": {"us-gaap": {tag: {"label": tag, "units": units} for tag, units in gaap.items()}},
    }


@pytest.fixture
def calendar_doc() -> dict[str, Any]:
    """A calendar-year filer with FY2023 and FY2024 10-Ks (FY2024 restates FY2023 revenue)."""
    a23, a24 = "0001234567-24-000010", "0001234567-25-000010"
    return facts_doc(
        {
            "Revenues": {
                "USD": [
                    fact(900, "2022-12-31", start="2022-01-01", fy=2023, accn=a23, filed="2024-02-15"),
                    fact(1000, "2023-12-31", start="2023-01-01", fy=2023, accn=a23, filed="2024-02-15"),
                    fact(250, "2023-12-31", start="2023-10-01", fy=2023, accn=a23, filed="2024-02-15"),
                    fact(1010, "2023-12-31", start="2023-01-01", fy=2024, accn=a24, filed="2025-02-15"),
                    fact(1200, "2024-12-31", start="2024-01-01", fy=2024, accn=a24, filed="2025-02-15"),
                    fact(300, "2024-03-31", start="2024-01-01", fy=2024, form="10-Q", fp="Q1"),
                ]
            },
            "Assets": {
                "USD": [
                    fact(5000, "2022-12-31", fy=2023, accn=a23, filed="2024-02-15"),
                    fact(5500, "2023-12-31", fy=2023, accn=a23, filed="2024-02-15"),
                    fact(5500, "2023-12-31", fy=2024, accn=a24, filed="2025-02-15"),
                    fact(6100, "2024-12-31", fy=2024, accn=a24, filed="2025-02-15"),
                ]
            },
        }
    )


@pytest.fixture
def make_sec_client(tmp_path: Path) -> Callable[..., tuple[SecClient, list[str]]]:
    """Return a factory for SecClient backed by a mock transport; also returns the list of requested URLs."""

    def factory(doc: dict[str, Any] | None = None, handler=None, **kwargs) -> tuple[SecClient, list[str]]:
        requests: list[str] = []

        def default_handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.endswith("company_tickers.json"):
                return httpx.Response(200, json=TICKER_MAP)
            if url.endswith(f"CIK{TEST_CIK:010d}.json") and doc is not None:
                return httpx.Response(200, content=json.dumps(doc).encode())
            return httpx.Response(404)

        def recording(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            return (handler or default_handler)(request)

        kwargs.setdefault("sleep", lambda s: None)
        client = SecClient(
            "Test User test@example.com",
            kwargs.pop("cache_dir", tmp_path / "cache"),
            transport=httpx.MockTransport(recording),
            **kwargs,
        )
        return client, requests

    return factory
