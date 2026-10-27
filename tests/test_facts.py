"""SEC fact selection and fiscal-year handling."""

from finmcp_eval.facts import fiscal_year_end, select_fact

from conftest import facts_doc, fact


def test_selects_annual_value_from_fiscal_years_own_10k(calendar_doc):
    r = select_fact(calendar_doc, "revenue", 2023, ticker="TEST")
    assert r["status"] == "ok"
    assert r["value"] == 1000  # Not the Q4 value (250) nor the later restated value (1010).
    assert r["period_start"] == "2023-01-01" and r["period_end"] == "2023-12-31"
    assert r["form"] == "10-K" and r["fiscal_period"] == "FY"
    assert r["tag"] == "Revenues" and r["unit"] == "USD"
    assert r["accession_number"] == "0001234567-24-000010"
    assert r["selection"]["confidence"] == "high"
    assert "restatement" in r["selection"]["explanation"]
    assert r["source_url"].startswith("https://data.sec.gov/api/xbrl/companyfacts/CIK0001234567")


def test_prior_year_comparatives_do_not_define_the_fiscal_year(calendar_doc):
    # The FY2023 10-K also contains FY2022 numbers; the FY end must be the latest one.
    gaap = calendar_doc["facts"]["us-gaap"]
    assert fiscal_year_end(gaap, 2023) == "2023-12-31"
    assert fiscal_year_end(gaap, 2024) == "2024-12-31"


def test_instant_metric_uses_balance_at_fiscal_year_end(calendar_doc):
    r = select_fact(calendar_doc, "total_assets", 2024)
    assert r["status"] == "ok" and r["value"] == 6100 and r["period_end"] == "2024-12-31"
    assert r["period_start"] is None


def test_non_calendar_fiscal_year_uses_company_label():
    # FY2024 ends 2024-01-31 (retailer-style). Calendar 2023 is mostly in FY2024.
    acc = "0001234567-24-000020"
    doc = facts_doc(
        {
            "Revenues": {
                "USD": [
                    fact(800, "2023-01-31", start="2022-02-01", fy=2024, accn=acc),
                    fact(850, "2024-01-31", start="2023-02-01", fy=2024, accn=acc),
                ]
            }
        }
    )
    r = select_fact(doc, "revenue", 2024)
    assert r["status"] == "ok" and r["value"] == 850
    assert r["period_end"] == "2024-01-31"
    assert r["selection"]["method"] == "fiscal_year_label"


def test_53_week_year_counts_as_annual():
    doc = facts_doc(
        {"Revenues": {"USD": [fact(500, "2023-09-30", start="2022-09-25", fy=2023)]}}  # 370 days
    )
    assert select_fact(doc, "revenue", 2023)["value"] == 500


def test_falls_back_to_next_tag_variation_and_says_so():
    doc = facts_doc(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "USD": [fact(700, "2023-12-31", start="2023-01-01", fy=2023)]
            }
        }
    )
    r = select_fact(doc, "revenue", 2023)
    assert r["status"] == "ok" and r["value"] == 700
    assert r["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert "Higher-priority tags had no matching fact" in r["selection"]["explanation"]


def test_own_10k_beats_higher_priority_tag_from_later_filing():
    # Mirrors Amazon FY2016 (found in the first live run): the FY2016 10-K used SalesRevenueNet;
    # RevenueFromContractWithCustomer... only appears as a comparative in a later 10-K.
    doc = facts_doc(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "USD": [fact(990, "2016-12-31", start="2016-01-01", fy=2018, accn="LATER", filed="2019-02-01")]
            },
            "SalesRevenueNet": {
                "USD": [fact(1000, "2016-12-31", start="2016-01-01", fy=2016, accn="OWN", filed="2017-02-10")]
            },
        }
    )
    r = select_fact(doc, "revenue", 2016)
    assert r["status"] == "ok" and r["value"] == 1000
    assert r["tag"] == "SalesRevenueNet" and r["accession_number"] == "OWN"
    assert r["selection"]["confidence"] == "high"
    assert r["alternatives"] == [{"tag": "RevenueFromContractWithCustomerExcludingAssessedTax", "values": [990]}]


def test_lower_priority_tag_disagreement_is_reported():
    doc = facts_doc(
        {
            "Revenues": {"USD": [fact(1000, "2023-12-31", start="2023-01-01", fy=2023)]},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "USD": [fact(950, "2023-12-31", start="2023-01-01", fy=2023)]
            },
        }
    )
    r = select_fact(doc, "revenue", 2023)
    assert r["value"] == 1000
    assert r["alternatives"] == [{"tag": "RevenueFromContractWithCustomerExcludingAssessedTax", "values": [950]}]


def test_conflicting_values_are_ambiguous_not_guessed():
    doc = facts_doc(
        {
            "NetIncomeLoss": {
                "USD": [
                    fact(100, "2023-12-31", start="2023-01-01", fy=2023, accn="A-1", filed="2024-02-01"),
                    fact(90, "2023-12-31", start="2023-01-01", fy=2023, form="10-K/A", accn="A-2", filed="2024-05-01"),
                ]
            }
        }
    )
    r = select_fact(doc, "net_income", 2023)
    assert r["status"] == "ambiguous"
    assert r["value"] is None
    assert {c["value"] for c in r["candidates"]} == {100, 90}
    assert r["selection"]["confidence"] == "low"


def test_identical_values_in_amendment_are_not_ambiguous():
    doc = facts_doc(
        {
            "NetIncomeLoss": {
                "USD": [
                    fact(100, "2023-12-31", start="2023-01-01", fy=2023, accn="A-1", filed="2024-02-01"),
                    fact(100, "2023-12-31", start="2023-01-01", fy=2023, form="10-K/A", accn="A-2", filed="2024-05-01"),
                ]
            }
        }
    )
    r = select_fact(doc, "net_income", 2023)
    assert r["status"] == "ok" and r["value"] == 100 and r["accession_number"] == "A-1"


def test_missing_metric_is_not_found_with_available_years(calendar_doc):
    r = select_fact(calendar_doc, "capital_expenditures", 2023)
    assert r["status"] == "not_found" and r["value"] is None
    assert "PaymentsToAcquirePropertyPlantAndEquipment" in r["message"]

    r = select_fact(calendar_doc, "revenue", 2019)
    assert r["status"] == "not_found"
    assert r["available_fiscal_years"] == [2023, 2024]


def test_without_fiscal_year_label_falls_back_with_medium_confidence():
    # Facts only appear as comparatives in a later 10-K, so no filing is labeled fy=2021.
    doc = facts_doc({"Revenues": {"USD": [fact(400, "2021-12-31", start="2021-01-01", fy=2023)]}})
    r = select_fact(doc, "revenue", 2021)
    assert r["status"] == "ok" and r["value"] == 400
    assert r["selection"]["method"] == "period_end_calendar_year"
    assert r["selection"]["confidence"] == "medium"


def test_eps_uses_per_share_unit():
    doc = facts_doc(
        {"EarningsPerShareDiluted": {"USD/shares": [fact(2.5, "2023-12-31", start="2023-01-01", fy=2023)]}}
    )
    r = select_fact(doc, "eps_diluted", 2023)
    assert r["value"] == 2.5 and r["unit"] == "USD/shares"
