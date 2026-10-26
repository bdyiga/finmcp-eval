# finmcp-eval

A small experiment that asks one language model the same 30 FinanceBench lookup and calculation
questions in two ways:

| Mode | What the model gets |
|---|---|
| `no_tools` | The question only. |
| `with_tools` | The question, plus two tools served by a local MCP server: `get_financial_fact` (SEC Company Facts API) and `calculate` (deterministic arithmetic). |

Everything else stays the same in both modes: the questions, the model, the system instruction and the generation
settings (temperature, seed, max tokens). Answers are **graded by hand**. There is no LLM judge, and
no claim about which mode is better should be made until grading is done.

## Project layout

```
finmcp-eval/
├── pyproject.toml            # dependencies (mcp pinned to 2.3.0) and the two CLI commands
├── uv.lock                   # exact resolved versions
├── .env.example              # placeholder settings; copy to .env
├── .gitignore
├── data/
│   └── questions_template.csv  # header only; put your questions.csv here
├── results/                  # results CSVs are written here (git-ignored)
├── cache/                    # SEC JSON cache (git-ignored, created on first use)
├── src/finmcp_eval/
│   ├── schemas.py            # metrics, US-GAAP tag lists, strict tool argument models, tool allowlist
│   ├── facts.py              # picks one fact from a Company Facts document (pure logic)
│   ├── sec.py                # SEC HTTP client: host allowlist, rate limit, retries, atomic cache
│   ├── calculator.py         # allowlisted arithmetic, no eval
│   ├── server.py             # MCP server (stdio) exposing exactly the two tools
│   ├── backends.py           # Gemini and Ollama chat backends
│   ├── agent.py              # model <-> MCP adapter and per-question tool loop
│   ├── questions.py          # question CSV schema and validation
│   ├── config.py             # environment variable settings
│   └── evaluate.py           # `finmcp-eval run` and `finmcp-eval summarize`
└── tests/                    # pytest; no network, credentials, or real model calls
```

## Assumptions

- You supply `data/questions.csv` with 30 public FinanceBench questions about **US-GAAP 10-K figures**.
  Adjusted and non-GAAP metrics, reasoning questions, and anything that needs the filing text (PDF or
  section retrieval) are out of scope. No questions, answers, or results ship with this repo.
- "Fiscal year" means the company's own label (Walmart's FY2024 ends on 2024-01-31). The tool relies on
  the `fy` field that SEC stores for each 10-K.
- The tool returns values in raw units, such as dollars. The model must convert them to the scale the question asks for,
  for example millions.
- The model only ever sees the `question` column. The `ticker`, `metric`, and `fiscal_year` columns are
  metadata for you and the graders. They are never sent to the model in either mode, so neither mode gets
  extra hints.
- The default model is `gemini-3.8-flash`, which is on the Gemini free tier as of October 2026. Set `GEMINI_MODEL` to
  any other model that supports function calling.
- The MCP Python SDK is pinned to **2.3.0**, where `FastMCP` was renamed `MCPServer`. The code uses only
  the 2.x APIs (`mcp.server.MCPServer`, `mcp.Client`, `mcp.StdioServerParameters`).

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.11+ for you
  if needed.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
- **Node.js 22.19 or later**, only to run MCP Inspector during development. Everything else is Python.

## Setup

```bash
git clone <your-repo-url> finmcp-eval
cd finmcp-eval
uv sync                       # creates .venv from uv.lock
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`, `SEC_CONTACT_NAME`, and `SEC_CONTACT_EMAIL`. SEC requires a real
name and email in the User-Agent of every request
([SEC guidance](https://www.sec.gov/os/accessing-edgar-data)).

All commands below use `uv run --env-file .env ...`, so the settings are read from `.env` without any
extra library.

## Commands

```bash
# Run the tests (no network, no credentials needed)
uv run pytest

# Start the MCP server on stdio. It waits for an MCP client on stdin; you normally don't run
# this by hand, because the evaluation launches it. Press Ctrl+C to stop.
uv run --env-file .env finmcp-server

# Inspect the tools with MCP Inspector (web UI). Inspector launches the server itself.
npx @modelcontextprotocol/inspector@2.12.0 uv run --env-file .env finmcp-server

# ...or with Inspector's CLI mode:
npx @modelcontextprotocol/inspector@2.12.0 --cli uv run --env-file .env finmcp-server --method tools/list
npx @modelcontextprotocol/inspector@2.12.0 --cli uv run --env-file .env finmcp-server \
  --method tools/call --tool-name get_financial_fact \
  --tool-args-json '{"ticker":"AAPL","metric":"revenue","fiscal_year":2023}'

# Quick trial: the first 2 questions, both modes
uv run --env-file .env finmcp-eval run --questions data/questions.csv --limit 2

# Full run: every question once in each mode (no_tools first, then with_tools)
uv run --env-file .env finmcp-eval run --questions data/questions.csv

# Optional: one mode only, or a custom output path
uv run --env-file .env finmcp-eval run --questions data/questions.csv --modes with_tools --out results/tools_only.csv

# After hand grading: tabulate grades by mode and question type
uv run finmcp-eval summarize results/results_<timestamp>.csv
```

In PowerShell, replace the trailing `\` line continuations with a backtick (`` ` ``), or put the
command on one line. Inside `--tool-args-json`, wrap the JSON in single quotes as shown.

A full run makes 60 or more model requests: one per question per mode, plus one for each tool round.
At the default pacing of 6 seconds between requests, that takes about 10 to 20 minutes. Each row is written as soon as it is
finished, so a partial file is still usable if the run stops.

## Input CSV schema

UTF-8 CSV with a header row. `data/questions_template.csv` contains the header.

| Column | Required | Format | Purpose |
|---|---|---|---|
| `question_id` | yes | 1–64 chars: letters, digits, `_`, `-`; unique | Your row ID |
| `question` | yes | 1–2000 chars | The only text sent to the model |
| `correct_answer` | yes | free text (e.g. `$1,577.00 million`) | FinanceBench reference answer, for graders |
| `ticker` | yes | e.g. `MMM`, `BRK.B` | Company (metadata) |
| `metric` | yes | one or more supported metric names separated by `;` (e.g. `revenue;total_assets`) | What the question needs (metadata) |
| `fiscal_year` | yes | one or more years separated by `;` (e.g. `2022;2023`) | Fiscal year(s) involved (metadata) |
| `question_type` | yes | `lookup` or `calculation` | For grouping results |
| `financebench_id` | no | free text | Original FinanceBench ID |
| `company_name` | no | free text | |
| `answer_unit` | no | free text (e.g. `USD millions`, `%`) | Grader hint |
| `notes` | no | free text | |

Optional columns may be blank. Metric names that `get_financial_fact` does not support only produce
a warning. The loader reports every problem in the file at once, and warns if there are not exactly 30
questions.

**Supported metrics:** `revenue`, `cost_of_revenue`, `gross_profit`, `operating_income`, `net_income`,
`eps_diluted`, `operating_cash_flow`, `capital_expenditures`, `depreciation_amortization`,
`dividends_paid`, `total_assets`, `current_assets`, `total_liabilities`, `current_liabilities`,
`total_equity`, `cash`, `inventory`, `accounts_receivable`, `property_plant_equipment_net`.
Each metric's US-GAAP tag variations, in priority order, are listed in `src/finmcp_eval/schemas.py`.

## Tools

### `get_financial_fact(ticker, metric, fiscal_year)`

1. Looks up the ticker in SEC's `company_tickers.json` to get the CIK.
2. Fetches `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`, or reads it from the cache.
3. Finds the 10-K (or 10-K/A) whose fiscal-year label is `fiscal_year`. The latest annual period end in
   that filing is the fiscal year end; its prior-year comparatives are ignored.
4. Tries each US-GAAP tag for the metric in priority order. Flow metrics such as revenue must cover about one
   year (340–380 days, which allows 52/53-week years). Balance-sheet metrics must be instants at the fiscal year end.
5. Returns `status: "ok"` with the value, unit, period, form, filing date, accession number, a link to the
   filing index, and a `selection` block (`method`, `explanation`, `confidence`). A later restatement or
   a lower-priority tag that disagrees is reported in the explanation and `alternatives`.
6. Returns `status: "ambiguous"` and **no value** when the matching facts disagree (for example, a 10-K
   and a 10-K/A with different numbers). It lists the candidates instead.
7. Returns `status: "not_found"` with the tags tried and the fiscal years that do exist.
8. If no 10-K carries the fiscal-year label, it falls back to annual facts whose period ends in that
   calendar year, with `confidence: "medium"` and an explanation.

### `calculate(operation, operands)`

`add`, `subtract` (a − b), `multiply`, `divide` (a / b), `percent_change` ((b − a) / |a| × 100),
`percent_of` (a / b × 100), and `average`. Operands must be 1–20 finite numbers with absolute value of
at most 1e15. Strings, booleans, extra fields and unknown operations are rejected, and division by zero is an error.

## Results CSV

One row per question per mode:

`question_id, question_type, ticker, metric, fiscal_year, question, correct_answer, mode, backend, model,
temperature, seed, model_answer, tool_calls_json, num_tool_calls, error, error_message,
latency_seconds, run_timestamp, manual_grade, grader_notes`

- `tool_calls_json` lists each call's name, validated arguments, `ok` flag, and result or error
  (`with_tools` rows only).
- `error` is blank on success, or one of `backend_error`, `empty_answer`, `tool_round_limit`,
  `unexpected_tool_call`, `invalid_question`.
- `manual_grade` and `grader_notes` are left blank for you.
- Text that starts with `=`, `+`, `@`, or a non-numeric `-` gets a leading `'`, so spreadsheet apps
  do not run it as a formula.

## Hand-grading instructions

1. Open the results CSV in a spreadsheet. Sort or filter by `question_id` so both modes of the same
   question sit next to each other.
2. If possible, hide the `mode`, `tool_calls_json`, and `num_tool_calls` columns while grading, so
   you grade the answer and not the mode.
3. For each row, compare `model_answer` with `correct_answer` and set `manual_grade` to exactly
   `correct` or `incorrect`. Decide your rules before you start and apply them to both modes alike:
   - **Correct** means the final number matches the reference after rounding to the precision the
     reference uses (e.g. `$1,577 million` matches `$1,577.00 million`), with the right unit and scale.
     Count a different scale as correct if it is the same value (`$1.577 billion`).
   - **Incorrect** covers a wrong number, the wrong year or metric, a missing unit that makes the answer
     ambiguous, "Unable to determine", or a row with an `error`.
4. Write anything noteworthy in `grader_notes`, for example "right value, wrong fiscal year label".
5. Run `uv run finmcp-eval summarize <file>`. It refuses to summarize until every row has a valid grade.
   Copy its table into the Results section below.

## Results

Not run yet. Fill this in from `finmcp-eval summarize` after hand grading.

| Mode | Question type | Correct | Total | Accuracy |
|---|---|---|---|---|
| no_tools | lookup | | | |
| no_tools | calculation | | | |
| no_tools | all | | | |
| with_tools | lookup | | | |
| with_tools | calculation | | | |
| with_tools | all | | | |

Model: ____ · Backend: ____ · Run date: ____ · Grader(s): ____

## Security notes

- **Secrets:** API keys and SEC contact details come only from environment variables (`.env`, which
  is git-ignored). `.env.example` holds placeholders only. Secrets are never logged, and HTTP client
  logging is turned down so request URLs and headers are not printed.
- **Least privilege for the server:** the MCP server runs as a local subprocess over **stdio only**,
  with no network listener. It is launched with a fixed command (this venv's Python,
  `-m finmcp_eval.server`) and no shell. The MCP SDK passes only a small safe set of environment
  variables, and this project adds just the SEC and cache settings. **`GEMINI_API_KEY` never reaches
  the server.**
- **Tool allowlist:** the model is offered exactly two function declarations. Before any call, the
  adapter checks the tool name against the allowlist and validates the arguments against strict
  pydantic models: no extra fields, a ticker regex, a fiscal-year range, an enum of metrics and
  operations, and bounded finite numbers. Unknown tools and invalid arguments are refused and the
  error goes back to the model. At startup, the evaluation checks that the server lists exactly the
  expected tools and stops if it doesn't. Automatic function calling in the Gemini SDK is turned off,
  so the model can never invoke Python functions directly.
- **Untrusted input:** question text and model output are treated as data. The system instruction tells
  the model to ignore instructions embedded in questions. Tool results sent back to the model are
  size-capped, and model output written to CSV is protected against formula injection.
- **SEC requests:** only `https://data.sec.gov` and `https://www.sec.gov` are allowed, and redirects
  are not followed. Tickers are validated before use, and cache file names are built from the numeric
  CIK, never from raw input. Requests are spaced at least 0.15 s apart (SEC's limit is 10 per second),
  responses are size-capped, timeouts are set, and retries are bounded: at most 2, only for timeouts,
  429 and 5xx, honoring `Retry-After` up to 60 s.
- **Cache:** written atomically (temp file in the same directory, then `os.replace`). A corrupt or expired
  file is re-fetched.
- **Calculator:** no `eval`, `exec`, shell or generated code. Only fixed operations on validated
  numbers.
- **Data sent to Gemini:** only public benchmark questions and public SEC data. Do not put
  confidential or private information in the question file.
- **Gemini free tier:** which models are free, the rate limits, and the data-use terms are set by Google
  and **can change at any time**. Under the free-tier terms, Google may use your prompts and responses to
  improve its products. Check the current
  [pricing](https://ai.google.dev/gemini-api/docs/pricing),
  [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) and
  [terms](https://ai.google.dev/gemini-api/terms) before running.

## Optional local fallback: Ollama

If free-tier limits prevent a full run, you can run both modes against a local model through
[Ollama](https://ollama.com). It runs on your own machine, so there is no per-request API charge.
Speed and feasibility depend on your hardware (RAM/VRAM).

```bash
ollama pull qwen3:8b          # or any model listed with tool-calling support on ollama.com
# in .env: OLLAMA_MODEL=qwen3:8b  (OLLAMA_HOST must be a local http://127.0.0.1 or localhost URL)
uv run --env-file .env finmcp-eval run --questions data/questions.csv --backend ollama
```

Use the same backend and model for **both** modes; a single `run` command already does this. Don't
compare a Gemini `no_tools` run with an Ollama `with_tools` run.

## Limitations

- Company Facts holds only XBRL-tagged numbers. Figures that appear only in the filing text, or that
  use company-specific (non US-GAAP) tags, return `not_found`.
- The tag lists cover common variations, not all of them. For example, banks may report revenue
  differently. Check `selection.explanation` and `alternatives` in `tool_calls_json` when grading
  surprising answers.
- Each question runs once per mode at temperature 0. Gemini does not guarantee deterministic output,
  even with a fixed seed.
