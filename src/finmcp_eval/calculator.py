"""Deterministic, allowlisted arithmetic. No eval, no exec, no expressions."""

from __future__ import annotations

import math
from typing import Any

from .schemas import OPERAND_COUNTS, CalculateArgs


class CalculationError(ValueError):
    """The calculation was well-formed but cannot be computed (e.g. division by zero)."""


def calculate(args: CalculateArgs) -> dict[str, Any]:
    """Run one validated operation and return the result with a readable expression."""
    op, xs = args.operation, args.operands
    lo, hi = OPERAND_COUNTS[op]
    if not lo <= len(xs) <= hi:
        expected = f"exactly {lo}" if lo == hi else f"{lo} to {hi}"
        raise CalculationError(f"'{op}' needs {expected} operands, got {len(xs)}")

    if op == "add":
        result, expr = math.fsum(xs), " + ".join(map(_fmt, xs))
    elif op == "subtract":
        result, expr = xs[0] - xs[1], f"{_fmt(xs[0])} - {_fmt(xs[1])}"
    elif op == "multiply":
        result, expr = math.prod(xs), " * ".join(map(_fmt, xs))
    elif op == "divide":
        _nonzero(xs[1], "divisor")
        result, expr = xs[0] / xs[1], f"{_fmt(xs[0])} / {_fmt(xs[1])}"
    elif op == "percent_change":
        _nonzero(xs[0], "old value (first operand)")
        result = (xs[1] - xs[0]) / abs(xs[0]) * 100
        expr = f"({_fmt(xs[1])} - {_fmt(xs[0])}) / |{_fmt(xs[0])}| * 100"
    elif op == "percent_of":
        _nonzero(xs[1], "whole (second operand)")
        result, expr = xs[0] / xs[1] * 100, f"{_fmt(xs[0])} / {_fmt(xs[1])} * 100"
    elif op == "average":
        result, expr = math.fsum(xs) / len(xs), f"mean({', '.join(map(_fmt, xs))})"
    else:  # pragma: no cover - the Literal type makes this unreachable
        raise CalculationError(f"unsupported operation: {op}")

    if not math.isfinite(result):
        raise CalculationError("result is not a finite number")
    return {"operation": op, "operands": xs, "result": result, "expression": expr}


def _nonzero(x: float, what: str) -> None:
    if x == 0:
        raise CalculationError(f"division by zero: {what} is 0")


def _fmt(x: float) -> str:
    return str(int(x)) if x.is_integer() else repr(x)
