"""Calculator: allowlisted operations and strict input validation."""

import pytest
from pydantic import ValidationError

from finmcp_eval.calculator import CalculationError, calculate
from finmcp_eval.schemas import CalculateArgs


def calc(op, *xs):
    return calculate(CalculateArgs(operation=op, operands=list(xs)))["result"]


def test_operations():
    assert calc("add", 1, 2, 3.5) == 6.5
    assert calc("subtract", 10, 4) == 6
    assert calc("multiply", 2, 3, 4) == 24
    assert calc("divide", 1, 4) == 0.25
    assert calc("percent_change", 200, 250) == 25.0
    assert calc("percent_change", -100, -50) == 50.0  # Uses |old| as the base.
    assert calc("percent_of", 25, 200) == 12.5
    assert calc("average", 2, 4, 9) == 5


def test_result_includes_expression():
    out = calculate(CalculateArgs(operation="divide", operands=[3, 2]))
    assert out["expression"] == "3 / 2"


@pytest.mark.parametrize(
    "op,xs", [("divide", [1, 0]), ("percent_change", [0, 5]), ("percent_of", [5, 0])]
)
def test_division_by_zero(op, xs):
    with pytest.raises(CalculationError, match="division by zero"):
        calculate(CalculateArgs(operation=op, operands=xs))


@pytest.mark.parametrize("op,xs", [("subtract", [1]), ("divide", [1, 2, 3]), ("add", [1])])
def test_wrong_operand_count(op, xs):
    with pytest.raises(CalculationError, match="operands"):
        calculate(CalculateArgs(operation=op, operands=xs))


@pytest.mark.parametrize(
    "payload",
    [
        {"operation": "pow", "operands": [2, 3]},
        {"operation": "eval", "operands": [1]},
        {"operation": "add", "operands": ["1", 2]},
        {"operation": "add", "operands": [True, 2]},
        {"operation": "add", "operands": [float("nan"), 2]},
        {"operation": "add", "operands": [float("inf"), 2]},
        {"operation": "add", "operands": [1e16, 2]},
        {"operation": "add", "operands": list(range(21))},
        {"operation": "add", "operands": []},
        {"operation": "add", "operands": "1+2"},
        {"operation": "add", "operands": [1, 2], "expression": "__import__('os')"},
    ],
)
def test_rejects_invalid_input(payload):
    with pytest.raises(ValidationError):
        CalculateArgs.model_validate(payload)
