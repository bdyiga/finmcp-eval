"""Select one annual fact from an SEC Company Facts document.

This module is pure (no network, no files) so it is easy to test.

How a fact is selected
----------------------
1. Find the company's 10-K for the requested fiscal year: filings whose `form` is
   10-K or 10-K/A, `fp` is "FY", and `fy` equals the requested year. The SEC sets
   `fy` from the filing's own fiscal-year label, so non-calendar years work
   (e.g. Walmart's FY2024 ends 2024-01-31, Apple's FY2023 ends 2023-09-30).
2. That filing also contains prior-year comparatives with the same `fy`, so the
   fiscal year's end date is the latest annual-period end date in the filing.
3. For each candidate US-GAAP tag (priority order from schemas.METRICS), keep facts
   whose period ends on that date (and, for flow metrics such as revenue, whose
   period is about one year long). Values from the fiscal year's own 10-K win, even
   over a higher-priority tag that only appears in later filings; a later filing
   reporting a different value for the same period is listed as a possible restatement.
4. If those facts disagree with each other, return "ambiguous" with the
   candidates instead of picking one.
5. If no filing is labeled with the fiscal year, fall back to annual facts whose
   period ends in that calendar year, and mark the result as lower confidence.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .schemas import METRICS, MetricSpec

ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})
ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 340, 380  # Covers 52/53-week fiscal years.
TAXONOMY = "us-gaap"


def select_fact(doc: dict[str, Any], metric: str, fiscal_year: int, ticker: str = "") -> dict[str, Any]:
    spec = METRICS[metric]
    cik = int(doc.get("cik", 0))
    base: dict[str, Any] = {
        "ticker": ticker,
        "company_name": doc.get("entityName"),
        "cik": cik,
        "metric": metric,
        "metric_description": spec.description,
        "fiscal_year": fiscal_year,
        "source_url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
    }
    gaap: dict[str, Any] = doc.get("facts", {}).get(TAXONOMY, {})

    period_end = fiscal_year_end(gaap, fiscal_year)
    if period_end is not None:
        method = "fiscal_year_label"
        explanation = (
            f"Used the 10-K labeled fiscal year {fiscal_year}; its annual period ends {period_end}."
        )
        confidence = "high"
    else:
        method = "period_end_calendar_year"
        explanation = (
            f"No 10-K is labeled fiscal year {fiscal_year}; used annual facts whose period ends in "
            f"calendar {fiscal_year}. Verify this matches the company's fiscal-year naming."
        )
        confidence = "medium"

    matches: list[tuple[str, list[dict[str, Any]]]] = []
    for tag in spec.tags:
        facts = _candidate_facts(gaap, tag, spec, fiscal_year, period_end)
        if facts:
            matches.append((tag, facts))

    if not matches:
        return {
            **base,
            "status": "not_found",
            "value": None,
            "message": (
                f"No {spec.unit} 10-K fact found for {metric} in fiscal year {fiscal_year}. "
                f"Tags tried: {', '.join(spec.tags)}."
            ),
            "available_fiscal_years": available_fiscal_years(gaap, spec),
        }

    # Prefer the fiscal year's own 10-K over tag priority: companies change tags over time
    # (e.g. SalesRevenueNet before ASC 606), so a higher-priority tag may only appear in a
    # later filing's comparatives, which can hold restated values.
    tag, facts = next(
        ((t, fs) for t, fs in matches if _split_primary(fs, fiscal_year, method)[0]), matches[0]
    )
    primary, later = _split_primary(facts, fiscal_year, method)
    pool = primary or later
    if not primary:
        confidence = "medium"
        explanation += " The value comes from a later filing's comparative figures."

    distinct_periods = {(f.get("start"), f["end"]) for f in pool}
    distinct_values = {f["val"] for f in pool}
    if len(distinct_values) > 1 or len(distinct_periods) > 1:
        return {
            **base,
            "status": "ambiguous",
            "value": None,
            "taxonomy": TAXONOMY,
            "tag": tag,
            "unit": spec.unit,
            "message": (
                "Several different values match this request; none was chosen. "
                "Inspect the candidates and filings."
            ),
            "candidates": [_describe(f, cik) for f in _sorted(pool)][:10],
            "selection": {"method": method, "explanation": explanation, "confidence": "low"},
        }

    chosen = _sorted(pool)[0]  # Earliest-filed of identical values: the original report.
    notes: list[str] = []
    if tag != spec.tags[0]:
        notes.append(f"Higher-priority tags had no matching fact in this fiscal year's 10-K; used {tag}.")
    restated = sorted({f["val"] for f in later if f["val"] != chosen["val"]}) if primary else []
    if restated:
        notes.append(
            f"A later filing reports a different value for the same period ({restated}); "
            "possibly a restatement. Returned the value from the fiscal year's own 10-K."
        )
    alternatives = []
    for alt_tag, alt_facts in matches:
        if alt_tag == tag:
            continue
        alt_vals = sorted({f["val"] for f in alt_facts})
        if alt_vals != [chosen["val"]]:
            alternatives.append({"tag": alt_tag, "values": alt_vals[:5]})
    if alternatives:
        notes.append("Lower-priority tags report different values; see alternatives.")

    return {
        **base,
        "status": "ok",
        "value": _num(chosen["val"]),
        "unit": spec.unit,
        "taxonomy": TAXONOMY,
        "tag": tag,
        **_describe(chosen, cik),
        "selection": {
            "method": method,
            "explanation": " ".join([explanation, *notes]),
            "confidence": confidence,
            "tags_tried": list(spec.tags),
        },
        "alternatives": alternatives,
    }


def fiscal_year_end(gaap: dict[str, Any], fiscal_year: int) -> str | None:
    """Return the end date (YYYY-MM-DD) of the fiscal year, using the 10-K labeled with that year.

    If several filings carry the label (a 10-K and a 10-K/A), the earliest-filed one is used.
    """
    by_accn: dict[str, tuple[str, str]] = {}  # accn -> (filed, max annual end)
    for tag_data in gaap.values():
        for unit_facts in tag_data.get("units", {}).values():
            for f in unit_facts:
                if (
                    f.get("fy") == fiscal_year
                    and f.get("fp") == "FY"
                    and f.get("form") in ANNUAL_FORMS
                    and _is_annual(f)
                ):
                    accn = f["accn"]
                    filed, end = by_accn.get(accn, (f.get("filed", ""), ""))
                    by_accn[accn] = (filed, max(end, f["end"]))
    if not by_accn:
        return None
    _, end = min(by_accn.values())
    return end


def available_fiscal_years(gaap: dict[str, Any], spec: MetricSpec) -> list[int]:
    years: set[int] = set()
    for tag in spec.tags:
        for f in gaap.get(tag, {}).get("units", {}).get(spec.unit, []):
            if f.get("form") in ANNUAL_FORMS and f.get("fp") == "FY" and isinstance(f.get("fy"), int):
                years.add(f["fy"])
    return sorted(years)


def _candidate_facts(
    gaap: dict[str, Any], tag: str, spec: MetricSpec, fiscal_year: int, period_end: str | None
) -> list[dict[str, Any]]:
    out = []
    for f in gaap.get(tag, {}).get("units", {}).get(spec.unit, []):
        if f.get("form") not in ANNUAL_FORMS or "end" not in f or "val" not in f:
            continue
        if spec.period_type == "duration":
            if not _is_annual(f):
                continue
        elif f.get("start"):
            continue  # Instant metrics must not have a start date.
        if period_end is not None:
            if f["end"] != period_end:
                continue
        elif not (f["end"].startswith(f"{fiscal_year}-") and f.get("fp") == "FY"):
            continue
        out.append(f)
    return out


def _split_primary(
    facts: list[dict[str, Any]], fiscal_year: int, method: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate facts from the fiscal year's own 10-K from those in later filings."""
    if method != "fiscal_year_label":
        return facts, []
    primary = [f for f in facts if f.get("fy") == fiscal_year]
    later = [f for f in facts if f.get("fy") != fiscal_year]
    return primary, later


def _describe(f: dict[str, Any], cik: int) -> dict[str, Any]:
    accn = f.get("accn", "")
    return {
        "value": _num(f["val"]),
        "period_start": f.get("start"),
        "period_end": f["end"],
        "fiscal_period": f.get("fp"),
        "filing_fiscal_year": f.get("fy"),
        "form": f.get("form"),
        "filed": f.get("filed"),
        "accession_number": accn,
        "frame": f.get("frame"),
        "filing_index_url": (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/" if accn else None
        ),
    }


def _sorted(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(facts, key=lambda f: (f.get("filed", ""), f.get("accn", "")))


def _is_annual(f: dict[str, Any]) -> bool:
    start, end = f.get("start"), f.get("end")
    if not start or not end:
        return False
    try:
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError:
        return False
    return ANNUAL_MIN_DAYS <= days <= ANNUAL_MAX_DAYS


def _num(v: Any) -> int | float:
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v
