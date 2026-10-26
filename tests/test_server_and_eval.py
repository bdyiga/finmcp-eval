"""MCP server surface, question CSV validation, and results CSV output."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from mcp import Client

from finmcp_eval import config, server
from finmcp_eval.agent import server_parameters
from finmcp_eval.backends import ModelTurn
from finmcp_eval.evaluate import RESULT_COLUMNS, csv_safe, run_eval, summarize
from finmcp_eval.questions import QuestionFileError, REQUIRED_COLUMNS, load_questions

ROOT = Path(__file__).resolve().parents[1]


async def test_server_exposes_exactly_two_tools():
    async with Client(server.mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"get_financial_fact", "calculate"}
    assert set(tools["get_financial_fact"].input_schema["required"]) == {"ticker", "metric", "fiscal_year"}


async def test_fact_tool_end_to_end_over_mcp(make_sec_client, calendar_doc):
    sec, requests = make_sec_client(calendar_doc)
    server.set_sec_client(sec)
    try:
        async with Client(server.mcp) as client:
            r = await client.call_tool("get_financial_fact", {"ticker": "TEST", "metric": "revenue", "fiscal_year": 2024})
            again = await client.call_tool("get_financial_fact", {"ticker": "TEST", "metric": "revenue", "fiscal_year": 2023})
            missing = await client.call_tool("get_financial_fact", {"ticker": "NOPE", "metric": "revenue", "fiscal_year": 2023})
    finally:
        server.set_sec_client(None)
    assert not r.is_error and r.structured_content["value"] == 1200
    assert again.structured_content["value"] == 1000
    assert len(requests) == 2  # Ticker map + facts, then served from memory/disk.
    assert missing.is_error and "not found" in missing.content[0].text


async def test_missing_sec_contact_is_a_clear_tool_error(monkeypatch):
    monkeypatch.delenv("SEC_CONTACT_NAME", raising=False)
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    server.set_sec_client(None)
    async with Client(server.mcp) as client:
        r = await client.call_tool("get_financial_fact", {"ticker": "TEST", "metric": "revenue", "fiscal_year": 2023})
    assert r.is_error and "SEC_CONTACT_NAME" in r.content[0].text


def test_server_subprocess_env_excludes_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")
    monkeypatch.setenv("SEC_CONTACT_NAME", "Jane")
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "jane@example.com")
    params = server_parameters()
    assert "GEMINI_API_KEY" not in params.env
    assert params.env["SEC_CONTACT_EMAIL"] == "jane@example.com"
    assert params.args == ["-m", "finmcp_eval.server"]


@pytest.mark.parametrize(
    "name,email,ok",
    [("Jane Doe", "jane@example.com", True), ("", "jane@example.com", False), ("Jane", "nope", False), ("Ja\r\nne", "j@e.com", False)],
)
def test_sec_user_agent_validation(monkeypatch, name, email, ok):
    monkeypatch.setenv("SEC_CONTACT_NAME", name)
    monkeypatch.setenv("SEC_CONTACT_EMAIL", email)
    if ok:
        assert config.sec_user_agent() == f"{name} {email}"
    else:
        with pytest.raises(config.ConfigError):
            config.sec_user_agent()


@pytest.mark.parametrize("host", ["http://evil.com:11434", "https://127.0.0.1:11434", "http://127.0.0.1:11434/x"])
def test_ollama_host_must_be_local(monkeypatch, host):
    monkeypatch.setenv("OLLAMA_HOST", host)
    with pytest.raises(config.ConfigError):
        config.ollama_host()


# --- Question CSV --------------------------------------------------------------

HEADER = ",".join(REQUIRED_COLUMNS)


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "q.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_template_has_required_columns():
    _, warnings = load_questions(ROOT / "data" / "questions_template.csv")
    assert any("found 0" in w for w in warnings)


def test_loads_valid_rows(tmp_path):
    p = write(
        tmp_path,
        HEADER + "\nq1,What is X?,USD 1 million,brk.b,revenue;net_income,2022;2023,calculation\n",
    )
    qs, _ = load_questions(p)
    assert qs[0].ticker == "BRK-B" and qs[0].metrics == ("revenue", "net_income") and qs[0].fiscal_years == (2022, 2023)


def test_reports_all_problems(tmp_path):
    p = write(
        tmp_path,
        HEADER + "\nq1,,x,TEST,revenue,2023,lookup\nq1,Q?,,../x,revenue,20x3,reasoning\n",
    )
    with pytest.raises(QuestionFileError) as e:
        load_questions(p)
    msg = str(e.value)
    for needle in ["question must be", "duplicate", "correct_answer", "ticker", "fiscal_year", "question_type"]:
        assert needle in msg


def test_missing_columns(tmp_path):
    with pytest.raises(QuestionFileError, match="missing required columns"):
        load_questions(write(tmp_path, "question,answer\nx,y\n"))


# --- Results CSV -----------------------------------------------------------------


class EchoBackend:
    name, model = "scripted", "scripted-model"

    def start(self, system_instruction, question, tools):
        return question

    async def generate(self, conv):
        return ModelTurn("=HYPERLINK(\"http://x\")" if "inject" in conv else "USD 5 million", [])

    def add_tool_results(self, conv, results):  # pragma: no cover
        raise AssertionError


async def test_run_eval_writes_blank_grades(tmp_path):
    p = write(tmp_path, HEADER + "\nq1,Plain?,USD 5 million,TEST,revenue,2023,lookup\nq2,inject?,x,TEST,revenue,2023,lookup\n")
    qs, _ = load_questions(p)
    out = tmp_path / "results.csv"
    errors = await run_eval(qs, ["no_tools"], EchoBackend(), out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert errors == 0
    assert tuple(rows[0]) == RESULT_COLUMNS
    assert [r["mode"] for r in rows] == ["no_tools", "no_tools"]
    assert rows[0]["model_answer"] == "USD 5 million"
    assert rows[1]["model_answer"].startswith("'=")  # Formula injection neutralized.
    assert all(r["manual_grade"] == "" for r in rows)


def test_summarize_requires_grades(tmp_path, capsys):
    out = tmp_path / "r.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        w.writerow({"mode": "no_tools", "question_type": "lookup", "manual_grade": "correct"})
        w.writerow({"mode": "with_tools", "question_type": "lookup", "manual_grade": ""})
    assert summarize(out) == 1

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    rows[1]["manual_grade"] = "incorrect"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert summarize(out) == 0
    table = capsys.readouterr().out
    assert "| no_tools | all | 1 | 1 | 100.0% |" in table
    assert "| with_tools | all | 0 | 1 | 0.0% |" in table


@pytest.mark.parametrize("value,expected", [("-5%", "-5%"), ("-$5", "'-$5"), ("@SUM(A1)", "'@SUM(A1)"), ("ok", "ok"), ("", "")])
def test_csv_safe(value, expected):
    assert csv_safe(value) == expected
