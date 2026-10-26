"""Load and validate the question CSV. See README "Input CSV schema"."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schemas import METRICS, normalize_ticker, validate_fiscal_year

REQUIRED_COLUMNS = ("question_id", "question", "correct_answer", "ticker", "metric", "fiscal_year", "question_type")
OPTIONAL_COLUMNS = ("financebench_id", "company_name", "answer_unit", "notes")
QUESTION_TYPES = frozenset({"lookup", "calculation"})
EXPECTED_QUESTION_COUNT = 30
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class QuestionFileError(ValueError):
    pass


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    correct_answer: str
    ticker: str
    metrics: tuple[str, ...]
    fiscal_years: tuple[int, ...]
    question_type: str
    financebench_id: str = ""
    company_name: str = ""
    answer_unit: str = ""
    notes: str = ""


def load_questions(path: Path) -> tuple[list[Question], list[str]]:
    """Return (questions, warnings). Raises QuestionFileError listing every problem found."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise QuestionFileError(f"missing required columns: {', '.join(missing)}")
    unknown = [c for c in df.columns if c not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    warnings = [f"ignoring unknown columns: {', '.join(unknown)}"] if unknown else []

    errors: list[str] = []
    questions: list[Question] = []
    seen: set[str] = set()
    for i, row in enumerate(df.to_dict("records"), start=2):  # Row 1 is the header.
        row = {k: str(v).strip() for k, v in row.items()}
        where = f"row {i}"
        qid = row["question_id"]
        if not _ID_RE.fullmatch(qid):
            errors.append(f"{where}: question_id must be 1-64 letters, digits, '_' or '-'")
        elif qid in seen:
            errors.append(f"{where}: duplicate question_id {qid}")
        seen.add(qid)
        if not row["question"] or len(row["question"]) > 2000:
            errors.append(f"{where}: question must be 1-2000 characters")
        if not row["correct_answer"]:
            errors.append(f"{where}: correct_answer is empty")
        if row["question_type"] not in QUESTION_TYPES:
            errors.append(f"{where}: question_type must be one of {sorted(QUESTION_TYPES)}")
        try:
            ticker = normalize_ticker(row["ticker"])
        except ValueError as exc:
            errors.append(f"{where}: {exc}")
            ticker = ""
        metrics = tuple(m.strip() for m in row["metric"].split(";") if m.strip())
        if not metrics:
            errors.append(f"{where}: metric is empty")
        for m in metrics:
            if m not in METRICS:
                warnings.append(f"{where}: metric '{m}' is not supported by get_financial_fact")
        years: list[int] = []
        for y in (y.strip() for y in row["fiscal_year"].split(";") if y.strip()):
            try:
                years.append(validate_fiscal_year(int(y)))
            except ValueError as exc:
                errors.append(f"{where}: fiscal_year '{y}': {exc}")
        if not years:
            errors.append(f"{where}: fiscal_year is empty")
        questions.append(
            Question(
                question_id=qid,
                question=row["question"],
                correct_answer=row["correct_answer"],
                ticker=ticker,
                metrics=metrics,
                fiscal_years=tuple(years),
                question_type=row["question_type"],
                **{c: row.get(c, "") for c in OPTIONAL_COLUMNS},
            )
        )
    if errors:
        raise QuestionFileError("invalid question file:\n  " + "\n  ".join(errors))
    if len(questions) != EXPECTED_QUESTION_COUNT:
        warnings.append(f"expected {EXPECTED_QUESTION_COUNT} questions, found {len(questions)}")
    return questions, warnings
