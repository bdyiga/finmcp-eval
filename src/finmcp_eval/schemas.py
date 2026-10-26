"""Shared definitions: supported metrics, tool argument schemas, and the tool allowlist.

The MCP server and the model adapter both import from here, so the arguments the
model may send and the arguments the server accepts are defined in one place.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """How to find one metric in SEC Company Facts.

    `tags` are US-GAAP XBRL tags in priority order: the first tag that has a
    matching fact wins, and any lower-priority tag that disagrees is reported.
    """

    description: str
    period_type: Literal["duration", "instant"]
    unit: str
    tags: tuple[str, ...]


METRICS: dict[str, MetricSpec] = {
    "revenue": MetricSpec(
        "Total revenue / net sales",
        "duration",
        "USD",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "RevenuesNetOfInterestExpense",
        ),
    ),
    "cost_of_revenue": MetricSpec(
        "Cost of revenue / cost of goods sold",
        "duration",
        "USD",
        ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"),
    ),
    "gross_profit": MetricSpec("Gross profit", "duration", "USD", ("GrossProfit",)),
    "operating_income": MetricSpec("Operating income (loss)", "duration", "USD", ("OperatingIncomeLoss",)),
    "net_income": MetricSpec(
        "Net income (loss) attributable to the parent company",
        "duration",
        "USD",
        ("NetIncomeLoss", "ProfitLoss"),
    ),
    "eps_diluted": MetricSpec(
        "Diluted earnings per share", "duration", "USD/shares", ("EarningsPerShareDiluted",)
    ),
    "operating_cash_flow": MetricSpec(
        "Net cash provided by (used in) operating activities",
        "duration",
        "USD",
        (
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ),
    "capital_expenditures": MetricSpec(
        "Capital expenditures (payments to acquire property, plant and equipment)",
        "duration",
        "USD",
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
            "PaymentsForCapitalImprovements",
        ),
    ),
    "depreciation_amortization": MetricSpec(
        "Depreciation and amortization",
        "duration",
        "USD",
        (
            "DepreciationDepletionAndAmortization",
            "DepreciationAndAmortization",
            "DepreciationAmortizationAndAccretionNet",
        ),
    ),
    "dividends_paid": MetricSpec(
        "Cash dividends paid",
        "duration",
        "USD",
        ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    ),
    "total_assets": MetricSpec("Total assets", "instant", "USD", ("Assets",)),
    "current_assets": MetricSpec("Total current assets", "instant", "USD", ("AssetsCurrent",)),
    "total_liabilities": MetricSpec("Total liabilities", "instant", "USD", ("Liabilities",)),
    "current_liabilities": MetricSpec("Total current liabilities", "instant", "USD", ("LiabilitiesCurrent",)),
    "total_equity": MetricSpec(
        "Total stockholders' equity attributable to the parent",
        "instant",
        "USD",
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ),
    "cash": MetricSpec(
        "Cash and cash equivalents",
        "instant",
        "USD",
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "Cash",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
    ),
    "inventory": MetricSpec("Inventory, net", "instant", "USD", ("InventoryNet",)),
    "accounts_receivable": MetricSpec(
        "Accounts receivable, net (current)", "instant", "USD", ("AccountsReceivableNetCurrent",)
    ),
    "property_plant_equipment_net": MetricSpec(
        "Property, plant and equipment, net",
        "instant",
        "USD",
        (
            "PropertyPlantAndEquipmentNet",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
        ),
    ),
}

Metric = Literal[
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "eps_diluted",
    "operating_cash_flow",
    "capital_expenditures",
    "depreciation_amortization",
    "dividends_paid",
    "total_assets",
    "current_assets",
    "total_liabilities",
    "current_liabilities",
    "total_equity",
    "cash",
    "inventory",
    "accounts_receivable",
    "property_plant_equipment_net",
]
assert set(get_args(Metric)) == set(METRICS), "Metric literal and METRICS must match"

# ---------------------------------------------------------------------------
# Validators shared by the server, the adapter, and the question loader
# ---------------------------------------------------------------------------

# Uppercase letters/digits, optionally one class suffix such as BRK.B or BRK-B.
TICKER_RE = re.compile(r"^[A-Z]{1,6}([.\-][A-Z]{1,2})?$")
MIN_FISCAL_YEAR = 2009  # XBRL financial data starts around 2009.
MAX_FISCAL_YEAR = 2100


def normalize_ticker(value: Any) -> str:
    """Return the ticker in SEC's format (BRK.B -> BRK-B) or raise ValueError."""
    if not isinstance(value, str):
        raise ValueError("ticker must be a string")
    ticker = value.strip().upper()
    if not TICKER_RE.fullmatch(ticker):
        raise ValueError("ticker must be 1-6 letters, optionally followed by a class suffix like BRK.B")
    return ticker.replace(".", "-")


def validate_fiscal_year(value: Any) -> int:
    """Accept an int (or an integral float, which some models send) in the supported range."""
    if isinstance(value, bool):
        raise ValueError("fiscal_year must be an integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("fiscal_year must be an integer")
        value = int(value)
    if not isinstance(value, int):
        raise ValueError("fiscal_year must be an integer")
    if not MIN_FISCAL_YEAR <= value <= MAX_FISCAL_YEAR:
        raise ValueError(f"fiscal_year must be between {MIN_FISCAL_YEAR} and {MAX_FISCAL_YEAR}")
    return value


# ---------------------------------------------------------------------------
# Tool argument models
# ---------------------------------------------------------------------------


class FactArgs(BaseModel):
    """Arguments for get_financial_fact."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Stock ticker, e.g. AAPL or BRK.B")
    metric: Metric = Field(description="Which standard reported metric to fetch")
    fiscal_year: int = Field(
        description="The company's fiscal year as labeled in its 10-K (e.g. 2023)",
        ge=MIN_FISCAL_YEAR,
        le=MAX_FISCAL_YEAR,
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, v: Any) -> str:
        return normalize_ticker(v)

    @field_validator("fiscal_year", mode="before")
    @classmethod
    def _year(cls, v: Any) -> int:
        return validate_fiscal_year(v)


Operation = Literal["add", "subtract", "multiply", "divide", "percent_change", "percent_of", "average"]

# Number of operands each operation accepts: (min, max).
OPERAND_COUNTS: dict[str, tuple[int, int]] = {
    "add": (2, 20),
    "subtract": (2, 2),
    "multiply": (2, 20),
    "divide": (2, 2),
    "percent_change": (2, 2),
    "percent_of": (2, 2),
    "average": (1, 20),
}
assert set(get_args(Operation)) == set(OPERAND_COUNTS)

MAX_ABS_OPERAND = 1e15  # Larger than any line item in a 10-K, in raw dollars.


class CalculateArgs(BaseModel):
    """Arguments for calculate."""

    model_config = ConfigDict(extra="forbid")

    operation: Operation = Field(
        description=(
            "add: sum all operands. subtract: a - b. multiply: product of operands. divide: a / b. "
            "percent_change: (b - a) / |a| * 100 where a is the old value and b the new value. "
            "percent_of: a / b * 100. average: arithmetic mean of operands."
        )
    )
    operands: list[float] = Field(description="Numbers to operate on, in order", min_length=1, max_length=20)

    @field_validator("operands", mode="before")
    @classmethod
    def _operands(cls, v: Any) -> list[float]:
        if not isinstance(v, list):
            raise ValueError("operands must be a list of numbers")
        out: list[float] = []
        for item in v:
            # Reject bools and strings explicitly: pydantic would otherwise coerce them.
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("every operand must be a number")
            f = float(item)
            if not math.isfinite(f):
                raise ValueError("operands must be finite numbers")
            if abs(f) > MAX_ABS_OPERAND:
                raise ValueError(f"operands must have absolute value <= {MAX_ABS_OPERAND:g}")
            out.append(f)
        return out


# ---------------------------------------------------------------------------
# Tool allowlist and descriptions (single source of truth)
# ---------------------------------------------------------------------------

FACT_TOOL_DESCRIPTION = (
    "Look up one annual, US-GAAP figure that a US public company reported to the SEC in its 10-K, "
    "using the SEC Company Facts API. Values are in raw units (e.g. dollars, not millions). "
    "Returns the value, unit, period, filing form, accession number, and how the fact was selected. "
    "status is 'ok', 'ambiguous' (several conflicting values; none chosen), or 'not_found'."
)
CALCULATE_TOOL_DESCRIPTION = (
    "Deterministic arithmetic on numbers you provide. Supported operations: add, subtract, multiply, "
    "divide, percent_change, percent_of, average. Do not pass expressions or code."
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]

    def json_schema(self) -> dict[str, Any]:
        """A plain JSON Schema for the arguments, suitable for model function declarations."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        schema.pop("description", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema


TOOL_SPECS: dict[str, ToolSpec] = {
    "get_financial_fact": ToolSpec("get_financial_fact", FACT_TOOL_DESCRIPTION, FactArgs),
    "calculate": ToolSpec("calculate", CALCULATE_TOOL_DESCRIPTION, CalculateArgs),
}
ALLOWED_TOOL_NAMES = frozenset(TOOL_SPECS)
