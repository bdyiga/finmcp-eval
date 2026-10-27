"""The model -> adapter -> MCP tool-call loop, with mocked model responses and mocked SEC access."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from google.genai import errors as genai_errors
from google.genai import types
from mcp import Client

from finmcp_eval import server
from finmcp_eval.agent import SYSTEM_INSTRUCTION, McpToolExecutor, answer_question
from finmcp_eval.backends import BackendError, GeminiBackend, ModelTurn, ToolCall
from finmcp_eval.config import GenerationSettings

SETTINGS = GenerationSettings(temperature=0.0, max_output_tokens=512, seed=7)


@pytest.fixture
def connect(make_sec_client, calendar_doc):
    """Return an async context manager yielding an executor bound to the real server, in-process.

    (A context manager entered inside each test, because anyio task groups must be
    entered and exited in the same task.)
    """

    @asynccontextmanager
    async def _connect():
        sec, _ = make_sec_client(calendar_doc)
        server.set_sec_client(sec)
        try:
            async with Client(server.mcp) as client:
                ex = McpToolExecutor(client)
                await ex.verify_tools()
                yield ex
        finally:
            server.set_sec_client(None)

    return _connect


# ---------------------------------------------------------------------------
# A scripted backend: exercises the loop independent of any model SDK.
# ---------------------------------------------------------------------------


class ScriptedBackend:
    name, model = "scripted", "scripted-model"

    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.started_with_tools: list[str] | None = None
        self.system_instruction = ""
        self.tool_results: list[list[tuple[ToolCall, dict[str, Any]]]] = []

    def start(self, system_instruction, question, tools):
        self.system_instruction = system_instruction
        self.started_with_tools = [t.name for t in tools] if tools else None
        return {"question": question}

    async def generate(self, conv):
        return self.turns.pop(0)

    def add_tool_results(self, conv, results):
        self.tool_results.append(results)


async def test_lookup_then_calculate_then_answer(connect):
    backend = ScriptedBackend(
        [
            ModelTurn("", [ToolCall("get_financial_fact", {"ticker": "test", "metric": "revenue", "fiscal_year": 2024.0})]),
            ModelTurn("", [ToolCall("get_financial_fact", {"ticker": "TEST", "metric": "revenue", "fiscal_year": 2023})]),
            ModelTurn("", [ToolCall("calculate", {"operation": "percent_change", "operands": [1000, 1200]})]),
            ModelTurn("20.0% growth", []),
        ]
    )
    async with connect() as executor:
        rec = await answer_question(backend, "What was revenue growth?", executor)
    assert rec.error is None
    assert rec.answer == "20.0% growth"
    assert backend.started_with_tools == ["get_financial_fact", "calculate"]
    assert [t.ok for t in rec.tool_calls] == [True, True, True]
    assert rec.tool_calls[0].args == {"ticker": "TEST", "metric": "revenue", "fiscal_year": 2024}
    assert rec.tool_calls[0].result["value"] == 1200
    assert rec.tool_calls[1].result["value"] == 1000
    assert rec.tool_calls[2].result["result"] == 20.0
    # The model received each result.
    assert backend.tool_results[2][0][1] == {"result": rec.tool_calls[2].result}


async def test_no_tools_mode_offers_no_tools_and_same_instruction():
    backend = ScriptedBackend([ModelTurn("USD 1,000", [])])
    rec = await answer_question(backend, "What was revenue?", None)
    assert backend.started_with_tools is None
    assert backend.system_instruction == SYSTEM_INSTRUCTION
    assert rec.answer == "USD 1,000" and rec.tool_calls == []


async def test_unknown_tool_is_refused_and_reported_to_model(connect):
    backend = ScriptedBackend(
        [ModelTurn("", [ToolCall("run_python", {"code": "import os"})]), ModelTurn("Unable to determine", [])]
    )
    async with connect() as executor:
        rec = await answer_question(backend, "q", executor)
    assert rec.tool_calls[0].ok is False
    assert "unknown tool" in rec.tool_calls[0].error
    assert backend.tool_results[0][0][1] == {"error": rec.tool_calls[0].error}


@pytest.mark.parametrize(
    "args,needle",
    [
        ({"ticker": "../../etc", "metric": "revenue", "fiscal_year": 2023}, "ticker"),
        ({"ticker": "TEST", "metric": "adjusted_ebitda", "fiscal_year": 2023}, "metric"),
        ({"ticker": "TEST", "metric": "revenue", "fiscal_year": 1800}, "fiscal_year"),
        ({"ticker": "TEST", "metric": "revenue", "fiscal_year": 2023, "url": "http://x"}, "url"),
        ("not an object", "JSON object"),
    ],
)
async def test_invalid_arguments_are_rejected_before_calling_mcp(connect, args, needle):
    backend = ScriptedBackend([ModelTurn("", [ToolCall("get_financial_fact", args)]), ModelTurn("done", [])])
    async with connect() as executor:
        rec = await answer_question(backend, "q", executor)
    assert rec.tool_calls[0].ok is False
    assert needle in rec.tool_calls[0].error


async def test_tool_errors_are_passed_back(connect):
    backend = ScriptedBackend(
        [ModelTurn("", [ToolCall("calculate", {"operation": "divide", "operands": [1, 0]})]), ModelTurn("n/a", [])]
    )
    async with connect() as executor:
        rec = await answer_question(backend, "q", executor)
    assert rec.tool_calls[0].ok is False
    assert "division by zero" in rec.tool_calls[0].error


async def test_tool_round_limit(connect):
    call = ToolCall("calculate", {"operation": "add", "operands": [1, 1]})
    backend = ScriptedBackend([ModelTurn("", [call]) for _ in range(20)])
    async with connect() as executor:
        rec = await answer_question(backend, "q", executor)
    assert rec.error == "tool_round_limit"


async def test_verify_tools_fails_closed_on_unexpected_tools():
    from mcp.server import MCPServer

    rogue = MCPServer("rogue")

    @rogue.tool()
    def shell(cmd: str) -> str:  # pragma: no cover - never called
        return cmd

    async with Client(rogue) as client:
        with pytest.raises(RuntimeError, match="unexpected tools"):
            await McpToolExecutor(client).verify_tools()


# ---------------------------------------------------------------------------
# The real Gemini adapter with a fake google-genai client.
# ---------------------------------------------------------------------------


class FakeGenai:
    """Mimics client.aio.models.generate_content and records each request."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.aio = self
        self.models = self

    async def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def model_response(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


async def test_gemini_adapter_round_trip(connect):
    fake = FakeGenai(
        [
            model_response(
                types.Part(
                    function_call=types.FunctionCall(
                        id="call-1", name="get_financial_fact", args={"ticker": "TEST", "metric": "total_assets", "fiscal_year": 2024}
                    )
                )
            ),
            model_response(types.Part(text="thinking...", thought=True), types.Part(text="USD 6,100 total assets.")),
        ]
    )
    backend = GeminiBackend("gemini-test", SETTINGS, client=fake)
    async with connect() as executor:
        rec = await answer_question(backend, "What were total assets in FY2024?", executor)

    assert rec.error is None
    assert rec.answer == "USD 6,100 total assets."  # Thought parts are excluded.
    assert rec.tool_calls[0].result["value"] == 6100

    config = fake.requests[0]["config"]
    assert config.automatic_function_calling.disable is True
    assert config.temperature == 0.0 and config.seed == 7
    assert config.system_instruction == SYSTEM_INSTRUCTION
    names = [d.name for d in config.tools[0].function_declarations]
    assert names == ["get_financial_fact", "calculate"]
    # Gemini rejects allowed_function_names unless mode is ANY (found in the first live run).
    fcc = config.tool_config.function_calling_config
    assert fcc.mode == "AUTO" and fcc.allowed_function_names is None
    assert "$ref" not in str(config.tools[0].function_declarations[0].parameters_json_schema)

    # Second request: question, model's call, then our function response with the same id.
    contents = fake.requests[1]["contents"]
    assert len(contents) == 3
    fr = contents[2].parts[0].function_response
    assert fr.id == "call-1" and fr.name == "get_financial_fact"
    assert fr.response["result"]["value"] == 6100


async def test_gemini_no_tools_config_has_no_tools():
    fake = FakeGenai([model_response(types.Part(text="Unable to determine"))])
    backend = GeminiBackend("gemini-test", SETTINGS, client=fake)
    rec = await answer_question(backend, "q", None)
    assert rec.answer == "Unable to determine"
    assert fake.requests[0]["config"].tools is None


async def test_gemini_api_errors_become_row_errors(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("finmcp_eval.backends.asyncio.sleep", no_sleep)
    err = genai_errors.APIError(400, {"error": {"code": 400, "message": "bad model", "status": "INVALID_ARGUMENT"}})
    rate = genai_errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    fake = FakeGenai([rate, model_response(types.Part(text="ok"))])
    rec = await answer_question(GeminiBackend("m", SETTINGS, client=fake), "q", None)
    assert rec.answer == "ok" and len(fake.requests) == 2  # 429 retried once.

    fake = FakeGenai([err])
    rec = await answer_question(GeminiBackend("m", SETTINGS, client=fake), "q", None)
    assert rec.error == "backend_error" and "400" in rec.error_message


def test_backend_error_is_runtime_error():
    assert issubclass(BackendError, RuntimeError)
