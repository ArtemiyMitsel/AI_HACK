# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This is a submission for a hackathon (risk-hackathon.ru): an LLM-powered agent that reads an arbitrary tabular binary-classification dataset from `data/`, generates at most **5** features, and writes `output/train.csv` (ID + target + 5 features) and `output/test.csv` (ID + 5 features). The submission is evaluated by training a default CatBoost on those features against hidden test labels. The hard constraints — exactly 5 features, matching column order between train/test, ≤600s runtime — are enforced by [src/utils/check_submission.py](src/utils/check_submission.py) and must never be broken.

Full task description is in [README.md](README.md) (in Russian).

## Commands

```bash
uv venv && uv sync          # set up env from pyproject.toml
python run.py               # generate output/{train,test}.csv
python src/utils/check_submission.py   # full local dry-run: runs run.py in a subprocess, enforces runtime limit, validates output schema
```

`.env` must contain `GIGACHAT_CREDENTIALS` and `GIGACHAT_SCOPE` — `run.py` will `RuntimeError` on startup without them, and the submission validator (and competition platform) require the `.env` file to be present. There is no test suite.

## Architecture

Entrypoint → pipeline:

1. [run.py](run.py) builds a `GigaChat` client (`langchain_gigachat`, default model `GigaChat-2-Max`) and calls `make_agent_submission(gigachat)`.
2. [src/utils/baseline.py](src/utils/baseline.py) is the entire agent. Despite the filename, this is the real solution, not a baseline — `make_baseline_submission` is just an alias for `make_agent_submission`. The pipeline in `make_agent_submission` ([src/utils/baseline.py:856](src/utils/baseline.py#L856)):
   - Loads every `*.csv` in `data/` into a dict of DataFrames and reads `data/readme.txt`.
   - Infers the **target column** ([infer_target_column](src/utils/baseline.py#L97)) by parsing the readme (`"<col> - целевая переменная"`) and scoring binary/low-cardinality columns. Infers the **id column** ([infer_id_column](src/utils/baseline.py#L131)) by uniqueness + name heuristics.
   - Collects feature-generation **attempts** from two sources, then materializes and scores them:
     - **LLM attempts** ([llm_generate_attempts](src/utils/baseline.py#L211)): sends table profiles + readme to GigaChat, asks for up to `MAX_LLM_ATTEMPTS` (2) attempts of 5 features each, returned as JSON in a constrained DSL.
     - **Heuristic attempts** ([build_related_table_attempts](src/utils/baseline.py#L534)): detects joinable related tables by key-name + value-overlap scoring, emits count/mean/sum/nunique/first aggregations grouped into 3 candidate attempts.
   - Each attempt is executed by [build_features_for_attempt](src/utils/baseline.py#L470), scored by 3-fold stratified CV CatBoost in [evaluate_attempt](src/utils/baseline.py#L674) (sampled to `EVAL_SAMPLE_ROWS=20_000` for speed), and the 5 most important features per attempt are kept.
   - A **fallback attempt** `base_row_statistics` ([build_fallback_row_features](src/utils/baseline.py#L507)) computes null-count and numeric row stats directly from `train.csv` / `test.csv` — this guarantees 5 valid features even when no related tables are present.
   - The attempt with the highest CV AUC wins. Output is written via [format_output_frames](src/utils/baseline.py#L784) and optionally scored on any leaked holdout target in `test.csv` via [evaluate_public_test](src/utils/baseline.py#L811).

### Feature DSL (LLM output contract)

`build_features_for_attempt` consumes a strict JSON schema — if you change the DSL in the LLM prompt at [baseline.py:223](src/utils/baseline.py#L223), you must keep the three `kind` branches in sync:

- `direct` — pass through a base-table column.
- `aggregate` — group a related table by `group_key`, join to base via `base_key`, apply one of `SUPPORTED_AGGS` to `value_column`, with optional `filters`. Only **single-key joins to the base** are supported (no multi-hop joins).
- `binary_op` — `divide` / `subtract` / `add` on two previously-built features (resolved through `name_map` so raw LLM names still work).

Feature names are sanitized + deduped via [safe_feature_name](src/utils/baseline.py#L71).

### Hard invariants

- Every attempt **must** yield exactly `MAX_FEATURES = 5` columns after materialization or it's dropped ([baseline.py:743](src/utils/baseline.py#L743)). `output/train.csv` and `output/test.csv` must have the same 5 feature columns in the same order — `check_submission.py` asserts this.
- `output/train.csv` must contain `[id, target, *features]`; `output/test.csv` must contain `[id, *features]` (target absent).
- Total wall-clock runtime ≤ 600s. The CV sampling, `thread_count=1`, and 200-iteration CatBoost in `MODEL_PARAMS` are sized for this budget — be careful when increasing them.
- All I/O is relative to CWD: reads only from `data/`, writes only to `output/`. Don't hardcode absolute paths — the solution is executed in a clean environment on the grading platform.

## Notes on [src/utils/scoring.py](src/utils/scoring.py)

This file is a **reference** of how the organizers score submissions (CatBoost on the 5 features against hidden labels). It imports from `app.*` modules that don't exist in this repo and will not run locally — don't try to execute or fix it. Use it only to understand the target metric (ROC AUC / Gini) and scoring params.
