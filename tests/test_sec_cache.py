"""SEC client: caching, validation, host restriction, rate limiting, and retries."""

import json
import os
import time

import httpx
import pytest

from finmcp_eval.sec import SecClient, SecError, write_json_atomic

from conftest import TEST_CIK, TICKER_MAP


def test_fetches_once_then_reuses_cache(make_sec_client, calendar_doc, tmp_path):
    client, requests = make_sec_client(calendar_doc)
    cik, title = client.lookup_cik("test")
    assert (cik, title) == (TEST_CIK, "Test Corp")
    assert client.company_facts(cik)["entityName"] == "Test Corp"
    assert len(requests) == 2

    cached = tmp_path / "cache" / "companyfacts" / f"CIK{TEST_CIK:010d}.json"
    assert json.loads(cached.read_text(encoding="utf-8")) == calendar_doc

    # A new client (e.g. a new server process) reads from disk without any request.
    client2, requests2 = make_sec_client(calendar_doc)
    client2.lookup_cik("TEST")
    client2.company_facts(TEST_CIK)
    assert requests2 == []


def test_expired_cache_is_refetched(make_sec_client, calendar_doc, tmp_path):
    client, _ = make_sec_client(calendar_doc, cache_ttl_days=1)
    client.company_facts(TEST_CIK)
    path = tmp_path / "cache" / "companyfacts" / f"CIK{TEST_CIK:010d}.json"
    old = time.time() - 3 * 86400
    os.utime(path, (old, old))

    client2, requests = make_sec_client(calendar_doc, cache_ttl_days=1)
    client2.company_facts(TEST_CIK)
    assert len(requests) == 1


def test_corrupt_cache_is_refetched(make_sec_client, calendar_doc, tmp_path):
    path = tmp_path / "cache" / "companyfacts" / f"CIK{TEST_CIK:010d}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    client, requests = make_sec_client(calendar_doc)
    assert client.company_facts(TEST_CIK)["cik"] == TEST_CIK
    assert len(requests) == 1


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "x" / "data.json"
    write_json_atomic(target, {"a": 1})
    write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert [p.name for p in target.parent.iterdir()] == ["data.json"]


def test_atomic_write_failure_keeps_old_file(tmp_path):
    target = tmp_path / "data.json"
    write_json_atomic(target, {"ok": True})
    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})
    assert json.loads(target.read_text()) == {"ok": True}
    assert [p.name for p in tmp_path.iterdir()] == ["data.json"]


@pytest.mark.parametrize("bad", ["../etc", "AAPL;rm", "", "TOOLONGTICKER", "a/b", "AA PL", "AAPL\x00", 123])
def test_rejects_invalid_tickers(make_sec_client, bad):
    client, requests = make_sec_client()
    with pytest.raises(ValueError):
        client.lookup_cik(bad)
    assert requests == []


def test_unknown_ticker(make_sec_client):
    client, _ = make_sec_client()
    with pytest.raises(SecError, match="not found"):
        client.lookup_cik("ZZZZ")


@pytest.mark.parametrize("cik", [0, -1, 10**10, True, "123"])
def test_rejects_invalid_cik(make_sec_client, cik):
    client, requests = make_sec_client()
    with pytest.raises(SecError):
        client.company_facts(cik)
    assert requests == []


@pytest.mark.parametrize(
    "url", ["https://example.com/x.json", "http://data.sec.gov/x.json", "https://data.sec.gov.evil.com/x"]
)
def test_refuses_non_sec_hosts(make_sec_client, url):
    client, requests = make_sec_client()
    with pytest.raises(SecError, match="host"):
        client._get_json(url)
    assert requests == []


def test_sends_contact_user_agent(tmp_path):
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["User-Agent"]
        return httpx.Response(200, json=TICKER_MAP)

    client = SecClient("Jane Doe jane@example.com", tmp_path, transport=httpx.MockTransport(handler))
    client.lookup_cik("TEST")
    assert seen["ua"] == "Jane Doe jane@example.com"


def test_retries_429_with_retry_after_then_succeeds(make_sec_client):
    calls = {"n": 0}
    sleeps = []

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "5"})
        return httpx.Response(200, json=TICKER_MAP)

    client, _ = make_sec_client(handler=handler, sleep=sleeps.append, min_interval=0)
    assert client.lookup_cik("TEST")[0] == TEST_CIK
    assert 5.0 in sleeps


def test_retries_are_bounded(make_sec_client):
    sleeps = []
    client, requests = make_sec_client(
        handler=lambda r: httpx.Response(503, headers={"Retry-After": "9999"}),
        sleep=sleeps.append,
        max_retries=2,
        min_interval=0,
    )
    with pytest.raises(SecError, match="503"):
        client.lookup_cik("TEST")
    assert len(requests) == 3
    assert max(sleeps) <= 60.0  # Retry-After is capped.


def test_timeouts_are_retried_then_reported(make_sec_client):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client, requests = make_sec_client(handler=handler, max_retries=1, min_interval=0)
    with pytest.raises(SecError, match="timed out"):
        client.lookup_cik("TEST")
    assert len(requests) == 2


def test_403_explains_user_agent(make_sec_client):
    client, requests = make_sec_client(handler=lambda r: httpx.Response(403))
    with pytest.raises(SecError, match="SEC_CONTACT_NAME"):
        client.lookup_cik("TEST")
    assert len(requests) == 1  # 403 is not retried.


def test_redirects_are_not_followed(make_sec_client):
    client, requests = make_sec_client(
        handler=lambda r: httpx.Response(302, headers={"Location": "https://example.com/"})
    )
    with pytest.raises(SecError, match="redirect"):
        client.lookup_cik("TEST")
    assert len(requests) == 1


def test_throttle_spaces_requests(tmp_path):
    now = {"t": 100.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now["t"] += s

    client = SecClient(
        "T t@example.com",
        tmp_path,
        min_interval=0.2,
        sleep=sleep,
        clock=lambda: now["t"],
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    client._get_json("https://data.sec.gov/a.json")
    client._get_json("https://data.sec.gov/b.json")
    assert sleeps == [pytest.approx(0.2)]
