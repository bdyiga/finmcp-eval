"""Local MCP server (stdio only) exposing exactly two tools: get_financial_fact and calculate.

Run it directly with `uv run finmcp-server`, or let the evaluation launch it.
Logs go to stderr because stdout carries the MCP protocol.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field, ValidationError

from . import config
from .calculator import CalculationError, calculate as run_calculation
from .facts import select_fact
from .schemas import (
    CALCULATE_TOOL_DESCRIPTION,
    FACT_TOOL_DESCRIPTION,
    MAX_FISCAL_YEAR,
    MIN_FISCAL_YEAR,
    CalculateArgs,
    FactArgs,
    Metric,
    Operation,
)
from .sec import SecClient, SecError

log = logging.getLogger("finmcp_eval.server")

mcp = MCPServer("finmcp-eval")

_sec_client: SecClient | None = None


def get_sec_client() -> SecClient:
    """Create the SEC client on first use so the server can start (and list tools) without SEC settings."""
    global _sec_client
    if _sec_client is None:
        try:
            _sec_client = SecClient(
                config.sec_user_agent(), config.cache_dir(), cache_ttl_days=config.cache_ttl_days()
            )
        except config.ConfigError as exc:
            raise ToolError(str(exc)) from None
    return _sec_client


def set_sec_client(client: SecClient | None) -> None:
    """Used by tests to inject a client backed by a mock transport."""
    global _sec_client
    _sec_client = client


@mcp.tool(description=FACT_TOOL_DESCRIPTION)
def get_financial_fact(
    ticker: Annotated[str, Field(description="Stock ticker, e.g. AAPL or BRK.B")],
    metric: Annotated[Metric, Field(description="Which standard reported metric to fetch")],
    fiscal_year: Annotated[
        int,
        Field(description="Fiscal year as labeled in the 10-K", ge=MIN_FISCAL_YEAR, le=MAX_FISCAL_YEAR),
    ],
) -> dict[str, Any]:
    try:
        args = FactArgs.model_validate({"ticker": ticker, "metric": metric, "fiscal_year": fiscal_year})
    except ValidationError as exc:
        raise ToolError(_validation_message(exc)) from None
    client = get_sec_client()
    try:
        cik, _title = client.lookup_cik(args.ticker)
        doc = client.company_facts(cik)
    except SecError as exc:
        raise ToolError(str(exc)) from None
    return select_fact(doc, args.metric, args.fiscal_year, ticker=args.ticker)


@mcp.tool(description=CALCULATE_TOOL_DESCRIPTION)
def calculate(
    operation: Annotated[Operation, Field(description=CalculateArgs.model_fields["operation"].description)],
    operands: Annotated[list[float], Field(description="Numbers to operate on, in order", min_length=1, max_length=20)],
) -> dict[str, Any]:
    try:
        args = CalculateArgs.model_validate({"operation": operation, "operands": operands})
        return run_calculation(args)
    except ValidationError as exc:
        raise ToolError(_validation_message(exc)) from None
    except CalculationError as exc:
        raise ToolError(str(exc)) from None


def _validation_message(exc: ValidationError) -> str:
    parts = [f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()[:5]]
    return "invalid arguments: " + "; ".join(parts)


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
