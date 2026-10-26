"""Command-line entry point.

    finmcp-eval run --questions data/questions.csv           # both modes, every question once
    finmcp-eval summarize results/results_<timestamp>.csv    # after hand grading
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp import Client

from . import config
from .agent import AnswerRecord, McpToolExecutor, answer_question, server_parameters
from .backends import ChatBackend, GeminiBackend, OllamaBackend
from .questions import Question, QuestionFileError, load_questions

log = logging.getLogger("finmcp_eval")

MODES = ("no_tools", "with_tools")
GRADES = ("correct", "incorrect")
RESULT_COLUMNS = (
    "question_id",
    "question_type",
    "ticker",
    "metric",
    "fiscal_year",
    "question",
    "correct_answer",
    "mode",
    "backend",
    "model",
    "temperature",
    "seed",
    "model_answer",
    "tool_calls_json",
    "num_tool_calls",
    "error",
    "error_message",
    "latency_seconds",
    "run_timestamp",
    "manual_grade",
    "grader_notes",
)


def make_backend(name: str) -> ChatBackend:
    settings = config.generation_settings()
    if name == "gemini":
        return GeminiBackend(
            config.gemini_model(),
            settings,
            api_key=config.gemini_api_key(),
            min_interval=config.gemini_min_interval(),
        )
    return OllamaBackend(config.ollama_model(), settings, host=config.ollama_host())


async def run_eval(
    questions: list[Question], modes: list[str], backend: ChatBackend, out_path: Path
) -> int:
    """Run each question once per mode, writing one CSV row per (question, mode) as it goes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    settings = config.generation_settings()
    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    errors = 0

    async def run_mode(mode: str, executor: McpToolExecutor | None, writer: csv.DictWriter, fh: Any) -> None:
        nonlocal errors
        for n, q in enumerate(questions, start=1):
            log.info("[%s] %d/%d %s", mode, n, len(questions), q.question_id)
            started = time.monotonic()
            rec = await answer_question(backend, q.question, executor)
            errors += rec.error is not None
            writer.writerow(_row(q, mode, backend, settings, rec, time.monotonic() - started, run_ts))
            fh.flush()

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        if "no_tools" in modes:
            await run_mode("no_tools", None, writer, fh)
        if "with_tools" in modes:
            async with Client(server_parameters(), read_timeout_seconds=180) as client:
                executor = McpToolExecutor(client)
                await executor.verify_tools()
                await run_mode("with_tools", executor, writer, fh)
    if isinstance(backend, OllamaBackend):
        await backend.aclose()
    return errors


def _row(
    q: Question,
    mode: str,
    backend: ChatBackend,
    settings: config.GenerationSettings,
    rec: AnswerRecord,
    latency: float,
    run_ts: str,
) -> dict[str, Any]:
    row = {
        "question_id": q.question_id,
        "question_type": q.question_type,
        "ticker": q.ticker,
        "metric": ";".join(q.metrics),
        "fiscal_year": ";".join(map(str, q.fiscal_years)),
        "question": q.question,
        "correct_answer": q.correct_answer,
        "mode": mode,
        "backend": backend.name,
        "model": backend.model,
        "temperature": settings.temperature,
        "seed": settings.seed,
        "model_answer": rec.answer,
        "tool_calls_json": json.dumps([t.as_dict() for t in rec.tool_calls], default=str) if mode == "with_tools" else "",
        "num_tool_calls": len(rec.tool_calls) if mode == "with_tools" else "",
        "error": rec.error or "",
        "error_message": rec.error_message or "",
        "latency_seconds": f"{latency:.2f}",
        "run_timestamp": run_ts,
        "manual_grade": "",
        "grader_notes": "",
    }
    return {k: csv_safe(v) for k, v in row.items()}


def csv_safe(value: Any) -> Any:
    """Stop spreadsheet apps from treating untrusted text (model output) as a formula."""
    if not isinstance(value, str) or not value:
        return value
    if value[0] in "=+@\t\r" or (value[0] == "-" and not value[1:2].replace(".", "").isdigit()):
        return "'" + value
    return value


def summarize(path: Path) -> int:
    import pandas as pd

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    grades = df["manual_grade"].str.strip().str.lower()
    bad = df[~grades.isin(GRADES)]
    if len(bad):
        print(
            f"{len(bad)} of {len(df)} rows have no valid manual_grade (expected one of {GRADES}). "
            "Finish hand grading before summarizing.",
            file=sys.stderr,
        )
        return 1
    df["is_correct"] = grades.eq("correct")
    table = (
        df.groupby(["mode", "question_type"])["is_correct"].agg(correct="sum", total="count").reset_index()
    )
    overall = df.groupby("mode")["is_correct"].agg(correct="sum", total="count").reset_index()
    overall["question_type"] = "all"
    table = pd.concat([table, overall], ignore_index=True)
    table["accuracy"] = (table["correct"] / table["total"]).map("{:.1%}".format)
    print("| Mode | Question type | Correct | Total | Accuracy |")
    print("|---|---|---|---|---|")
    for r in table.sort_values(["mode", "question_type"]).itertuples():
        print(f"| {r.mode} | {r.question_type} | {r.correct} | {r.total} | {r.accuracy} |")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finmcp-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run every question once in each mode")
    run.add_argument("--questions", type=Path, required=True, help="question CSV (see README)")
    run.add_argument("--out", type=Path, help="results CSV (default: results/results_<UTC timestamp>.csv)")
    run.add_argument("--modes", default=",".join(MODES), help="comma-separated subset of: no_tools,with_tools")
    run.add_argument("--backend", choices=("gemini", "ollama"), default="gemini")
    run.add_argument("--limit", type=int, help="only the first N questions (for a quick trial)")

    summ = sub.add_parser("summarize", help="tabulate hand grades in a results CSV")
    summ.add_argument("results", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    # The HTTP libraries log full request URLs at INFO; keep them quiet.
    for noisy in ("httpx", "httpcore", "google_genai", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "summarize":
        return summarize(args.results)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes or any(m not in MODES for m in modes):
        parser.error(f"--modes must be a comma-separated subset of {','.join(MODES)}")
    try:
        questions, warnings = load_questions(args.questions)
        for w in warnings:
            log.warning(w)
        if args.limit:
            questions = questions[: args.limit]
        if "with_tools" in modes:
            config.sec_user_agent()  # Fail early if SEC contact details are missing.
        backend = make_backend(args.backend)
    except (QuestionFileError, config.ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = args.out or Path("results") / f"results_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    if out.exists():
        print(f"error: {out} already exists; choose another --out", file=sys.stderr)
        return 2
    log.info("backend=%s model=%s questions=%d modes=%s", backend.name, backend.model, len(questions), modes)
    errors = asyncio.run(run_eval(questions, modes, backend, out))
    log.info("wrote %s (%d rows with errors)", out, errors)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
