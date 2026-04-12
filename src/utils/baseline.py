from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


DATA_DIR = Path("data4")
OUTPUT_DIR = Path("output")
MAX_FEATURES = 5
MAX_ATTEMPTS = 10
MAX_LLM_ATTEMPTS = 3
MAX_LLM_FEATURES_PER_ATTEMPT = 10
CV_FOLDS = 3
RANDOM_STATE = 42
EVAL_SAMPLE_ROWS = 20_000
POOL_EVAL_SAMPLE_ROWS = 15_000
POOL_MAX_FEATURES = 40
COMPOSITE_KEY_OVERLAP_MIN = 0.5
TABLE_PROFILE_MAX_COLUMNS = 30
RELATED_JOIN_CANDIDATES_PER_TABLE = 2
ATTEMPT_SELECTION_SHORTLIST = 7
POOL_SELECTION_SHORTLIST = 9
TOP_ATTEMPTS_TO_REFINE = 2
HOLDOUT_FRACTION = 0.2
LEAKAGE_CORR_THRESHOLD = 0.995
NEAR_CONSTANT_SHARE = 0.995
MIN_POOL_COLS_FOR_STACK = 10
STACKED_MODEL_ITERATIONS = 200
TIME_BUDGET_SEC = 600
TIME_BUDGET_LLM_REFINE_SKIP = 480
TIME_BUDGET_STACK_SKIP = 530
TIME_BUDGET_REFINE_SKIP = 550
ROW_INDEX_CORR_THRESHOLD = 0.999
ENRICH_MAX_COLUMNS = 40
ENRICH_TOTAL_MAX = 60
ENRICH_OVERLAP_MIN = 0.95

MODEL_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "thread_count": 2,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}
SCREEN_MODEL_PARAMS = {
    **MODEL_PARAMS,
    "iterations": 180,
    "learning_rate": 0.07,
}
SELECTION_MODEL_PARAMS = {
    **MODEL_PARAMS,
    "iterations": 140,
    "learning_rate": 0.08,
}

SUPPORTED_AGGS = {
    "count", "nunique", "sum", "mean", "min", "max", "std", "median", "first",
    "last", "q25", "q75", "skew", "range",
}
SUPPORTED_BINARY_OPS = {"divide", "subtract", "add", "multiply"}


def _is_string_or_object_dtype(series: pd.Series) -> bool:
    """Check if a Series has string-like dtype (works on both pandas 2.x and 3.x)."""
    return pd.api.types.is_string_dtype(series) and not pd.api.types.is_numeric_dtype(series)


@dataclass
class AttemptResult:
    name: str
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    cv_auc: float
    selected_features: list[str]
    importances: dict[str, float]
    train_pool: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_pool: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdout_auc: float | None = None


@dataclass
class EvalCache:
    """Precomputed stratified sample indices + CV folds for a single `train_cv`.

    Indices are positional into a 0..n-1 reset frame, so they align with any
    `train_out` built via `attach_feature_frame` on the same `train_cv`.
    """
    sample_indices_eval: np.ndarray
    splits_eval: list[tuple[np.ndarray, np.ndarray]]
    sample_indices_pool: np.ndarray
    splits_pool: list[tuple[np.ndarray, np.ndarray]]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def candidate_key_names(name: str) -> set[str]:
    normalized = normalize_name(name)
    variants = {normalized}
    if normalized.endswith("id"):
        variants.add(normalized[:-2])
    else:
        variants.add(f"{normalized}id")
    return {item for item in variants if item}


def safe_feature_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_") or "feature"
    candidate = base[:60]
    idx = 2
    while candidate in used:
        suffix = f"_{idx}"
        candidate = f"{base[: max(1, 60 - len(suffix))]}{suffix}"
        idx += 1
    used.add(candidate)
    return candidate


def parse_readme(data_dir: Path) -> str:
    readme_path = data_dir / "readme.txt"
    if not readme_path.exists():
        return ""
    return readme_path.read_text(encoding="utf-8", errors="ignore")


_ROW_INDEX_NAME_RE = re.compile(r"^unnamed(?:\:\s*|_)\d+$", flags=re.IGNORECASE)


def _is_row_index_name(name: str) -> bool:
    """Name-based check: pandas saved row-index columns like ``Unnamed: 0``."""
    return isinstance(name, str) and bool(_ROW_INDEX_NAME_RE.match(name.strip()))


def _is_row_index_series(series: pd.Series, sample_rows: int = 5000) -> bool:
    """Content-based check: monotonic integer sequence highly correlated with row order.

    Returns True only when the column is numeric, has ``nunique == len`` with no
    NaNs, and ``|pearson(values, np.arange(n))| > ROW_INDEX_CORR_THRESHOLD``.
    Used in the pool filter where ``id_column`` is already protected.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    n = len(series)
    if n < 20:
        return False
    try:
        if int(series.nunique(dropna=True)) != n:
            return False
    except Exception:
        return False
    try:
        s_num = pd.to_numeric(series, errors="coerce")
        if s_num.isna().any():
            return False
        if n > sample_rows:
            idx = np.linspace(0, n - 1, num=sample_rows, dtype=int)
            vals = s_num.to_numpy()[idx]
            positions = idx.astype(float)
        else:
            vals = s_num.to_numpy()
            positions = np.arange(n, dtype=float)
        if np.std(vals) == 0:
            return False
        corr = float(np.corrcoef(vals, positions)[0, 1])
        return abs(corr) > ROW_INDEX_CORR_THRESHOLD
    except Exception:
        return False


def load_data_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*.csv")):
        df = read_table(path)
        dropped = [col for col in df.columns if _is_row_index_name(col)]
        if dropped:
            df = df.drop(columns=dropped)
            logger.info("Dropped row-index cols from {}: {}", path.name, dropped)
        tables[path.name] = df
    return tables


def _related_table_is_one_to_one(series: pd.Series) -> bool:
    """True when every value in *series* is unique (or no NaN-inflated dup)."""
    try:
        if series.isna().any():
            return False
        return bool(series.is_unique)
    except Exception:
        return False


def _pick_1to1_join_key(
    related_df: pd.DataFrame,
    base_frame: pd.DataFrame,
) -> tuple[str, str, float] | None:
    """Return (base_key, related_key, overlap_ratio) for the best 1:1 join, or None."""
    base_columns = [col for col in base_frame.columns if col in base_frame]
    base_samples: dict[str, set[str]] = {}
    for col in base_columns:
        try:
            base_samples[col] = set(
                base_frame[col].dropna().astype(str).head(5000)
            )
        except Exception:
            continue

    best: tuple[str, str, float] | None = None
    for rcol in related_df.columns:
        try:
            rseries = related_df[rcol]
        except Exception:
            continue
        if not _related_table_is_one_to_one(rseries):
            continue
        try:
            rvalues = set(rseries.astype(str).head(20000))
        except Exception:
            continue
        if not rvalues:
            continue

        for bcol in base_columns:
            bvalues = base_samples.get(bcol)
            if not bvalues:
                continue
            # Name match bonus: matching key names get priority.
            name_match = bool(candidate_key_names(bcol) & candidate_key_names(rcol))
            # For id_column specifically we require full name match or overlap.
            hits = len(bvalues & rvalues)
            overlap = hits / max(1, len(bvalues))
            if overlap < ENRICH_OVERLAP_MIN and not (name_match and overlap >= 0.5):
                continue
            score = overlap + (0.1 if name_match else 0.0)
            if best is None or score > best[2]:
                best = (bcol, rcol, overlap)
    return best


def _column_is_safe_to_enrich(
    related_df: pd.DataFrame,
    col: str,
    target_column: str,
    join_key: str,
) -> bool:
    if col in {target_column, join_key}:
        return False
    if _is_row_index_name(col):
        return False
    series = related_df[col]
    # Row-index content check (correlation with row order).
    try:
        if _is_row_index_series(series.reset_index(drop=True)):
            return False
    except Exception:
        pass
    # High-cardinality text IDs: nunique == len with string dtype → likely a row ID.
    if _is_string_or_object_dtype(series):
        try:
            nn = int(series.nunique(dropna=True))
            if nn >= max(20, int(len(series) * 0.95)):
                return False
        except Exception:
            pass
    return True


def enrich_base_with_lookups(
    tables: dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Left-merge 1:1 lookup columns from related tables into the base train/test.

    Dataset-agnostic: only fires when a related table has a truly unique join
    key covering ≥ ``ENRICH_OVERLAP_MIN`` of base rows. Transaction-grain tables
    (where the key repeats) are skipped. Returns the enriched train/test and a
    manifest describing what was added, for downstream LLM prompting.
    """
    manifest: list[dict[str, Any]] = []
    base_concat = pd.concat([train_df, test_df], ignore_index=True)
    existing_base_columns = set(train_df.columns) | set(test_df.columns)
    total_added = 0

    for table_name, related_df in list(tables.items()):
        if table_name in {"train.csv", "test.csv"}:
            continue
        if table_name.startswith("__prejoin__"):
            continue
        if total_added >= ENRICH_TOTAL_MAX:
            logger.info(
                "Enrichment total cap reached ({}); skipping remaining tables", ENRICH_TOTAL_MAX
            )
            break
        if related_df is None or related_df.empty:
            continue

        picked = _pick_1to1_join_key(related_df, base_concat)
        if picked is None:
            continue
        base_key, related_key, overlap = picked
        logger.info(
            "Enrichment candidate {}: base_key={} related_key={} overlap={:.3f}",
            table_name,
            base_key,
            related_key,
            overlap,
        )

        # Pick columns to merge.
        eligible_cols: list[str] = []
        for col in related_df.columns:
            if col == related_key:
                continue
            if not _column_is_safe_to_enrich(
                related_df, col, target_column, related_key
            ):
                continue
            eligible_cols.append(col)

        if not eligible_cols:
            continue

        # Rank by variance (numeric) / cardinality (string) so the caps pick signal.
        def _col_signal(c: str) -> float:
            s = related_df[c]
            try:
                if pd.api.types.is_numeric_dtype(s):
                    v = float(pd.to_numeric(s, errors="coerce").var(skipna=True))
                    return 0.0 if (v != v) else v  # NaN → 0
                return float(s.nunique(dropna=True))
            except Exception:
                return 0.0

        eligible_cols.sort(key=_col_signal, reverse=True)
        eligible_cols = eligible_cols[:ENRICH_MAX_COLUMNS]
        budget = max(0, ENRICH_TOTAL_MAX - total_added)
        eligible_cols = eligible_cols[:budget]
        if not eligible_cols:
            continue

        table_stem = table_name.replace(".csv", "")
        prefix = f"{table_stem}__"
        # Build merge frame keyed by related_key (deduped, first-row wins).
        merge_df = related_df[[related_key, *eligible_cols]].drop_duplicates(
            subset=[related_key], keep="first"
        )

        # Resolve name collisions with existing base columns.
        rename_map: dict[str, str] = {}
        added_cols: list[str] = []
        used_prefix = False
        for col in eligible_cols:
            target_name = col
            if target_name in existing_base_columns or target_name == id_column:
                target_name = f"{prefix}{col}"
                used_prefix = True
            rename_map[col] = target_name
            added_cols.append(target_name)

        merge_df = merge_df.rename(columns=rename_map)

        # Align merge key dtype with base_key in train/test to prevent key-mismatch.
        for frame_name, frame in (("train", train_df), ("test", test_df)):
            try:
                base_dtype = frame[base_key].dtype
                merge_df[related_key] = merge_df[related_key].astype(base_dtype, errors="ignore")
            except Exception:
                pass

        train_merged = train_df.merge(
            merge_df, how="left", left_on=base_key, right_on=related_key
        )
        test_merged = test_df.merge(
            merge_df, how="left", left_on=base_key, right_on=related_key
        )
        if related_key != base_key and related_key in train_merged.columns:
            train_merged = train_merged.drop(columns=[related_key])
        if related_key != base_key and related_key in test_merged.columns:
            test_merged = test_merged.drop(columns=[related_key])

        train_df = train_merged.reset_index(drop=True)
        test_df = test_merged.reset_index(drop=True)
        existing_base_columns = set(train_df.columns) | set(test_df.columns)
        total_added += len(added_cols)

        manifest.append(
            {
                "table": table_name,
                "join_key": related_key,
                "base_key": base_key,
                "added_cols": added_cols,
                "prefix": prefix if used_prefix else "",
                "overlap": round(float(overlap), 4),
            }
        )
        logger.info(
            "Enriched base from {} via {}: added {} cols (prefix={})",
            table_name,
            related_key,
            len(added_cols),
            prefix if used_prefix else "",
        )

    return train_df, test_df, manifest


def _sampled_nunique(series: pd.Series, sample_size: int = 2000) -> int:
    sample = series.dropna().head(sample_size)
    if sample.empty:
        return 0
    return int(sample.nunique())


def _is_binary_candidate(series: pd.Series) -> bool:
    return _sampled_nunique(series) == 2


def infer_target_column(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    readme_text: str = "",
) -> str:
    readme_match = re.search(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*-\s*целевая переменная", readme_text, flags=re.M | re.I)
    if readme_match:
        candidate = readme_match.group(1)
        if candidate in train_df.columns:
            return candidate

    scored: list[tuple[float, str]] = []
    for col in train_df.columns:
        series = train_df[col]
        nunique = _sampled_nunique(series)
        unique_ratio = series.nunique(dropna=False) / max(1, len(train_df))
        missing_ratio = float(series.isna().mean())
        normalized = normalize_name(col)
        score = 0.0
        if _is_binary_candidate(series):
            score += 14.0
        if normalized in {"target", "label", "y", "flag", "default", "badflag", "isdefault"}:
            score += 10.0
        if normalized.startswith("target") or normalized.endswith("target"):
            score += 4.0
        if nunique <= 5:
            score += 2.0
        if col not in test_df.columns:
            score += 6.0
        if unique_ratio >= 0.95:
            score -= 12.0
        if nunique <= 1:
            score -= 20.0
        if "id" in normalized or normalized.endswith("key"):
            score -= 8.0
        if missing_ratio > 0.5:
            score -= 2.0
        scored.append((score, col))
    if scored:
        best_score, best_col = max(scored)
        if best_score > 0:
            return best_col
    raise ValueError("Не удалось определить target column")


def infer_id_column(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> str:
    common_cols = [col for col in train_df.columns if col in test_df.columns and col != target_column]
    scored: list[tuple[float, str]] = []
    for col in common_cols:
        score = 0.0
        train_unique = train_df[col].nunique(dropna=False) >= max(1, int(len(train_df) * 0.95))
        test_unique = test_df[col].nunique(dropna=False) >= max(1, int(len(test_df) * 0.95))
        normalized = normalize_name(col)
        sampled_cardinality = max(_sampled_nunique(train_df[col]), _sampled_nunique(test_df[col]))
        if train_unique:
            score += 6.0
        if test_unique:
            score += 6.0
        if train_unique and test_unique:
            score += 4.0
        if normalized in {"id", "clientid", "customerid", "userid", "applicationid", "requestid"}:
            score += 6.0
        if normalized.endswith("id") or normalized.startswith("id"):
            score += 4.0
        if any(token in normalized for token in {"target", "label", "flag", "default"}):
            score -= 10.0
        if any(token in normalized for token in {"date", "time", "timestamp"}):
            score -= 4.0
        if sampled_cardinality <= 10 or _is_binary_candidate(train_df[col]) or _is_binary_candidate(test_df[col]):
            score -= 6.0
        if float(train_df[col].isna().mean()) > 0.1 or float(test_df[col].isna().mean()) > 0.1:
            score -= 3.0
        scored.append((score, col))
    if scored:
        best_score, best_col = max(scored)
        if best_score > 0:
            return best_col
    if common_cols:
        return common_cols[0]
    raise ValueError("Не удалось определить id column")


def select_profile_columns(
    df: pd.DataFrame,
    id_column: str,
    target_column: str,
    base_keys: list[str] | None = None,
    max_columns: int = TABLE_PROFILE_MAX_COLUMNS,
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    priority_norm = {normalize_name(id_column), normalize_name(target_column)}
    priority_norm.update(normalize_name(col) for col in (base_keys or []))

    def add(col: str) -> None:
        if col in df.columns and col not in seen and len(selected) < max_columns:
            selected.append(col)
            seen.add(col)

    for col in df.columns:
        normalized = normalize_name(col)
        if normalized in priority_norm or normalized.endswith("id") or normalized.endswith("key"):
            add(col)

    numeric_cols = [col for col in df.columns if col not in seen and pd.api.types.is_numeric_dtype(df[col])]
    numeric_cols.sort(
        key=lambda col: (
            float(df[col].notna().mean()),
            min(_sampled_nunique(df[col]), 500),
        ),
        reverse=True,
    )
    for col in numeric_cols[:10]:
        add(col)

    def _categorical_score(col: str) -> tuple[float, int]:
        series = df[col]
        sampled_nunique = _sampled_nunique(series)
        medium_cardinality_bonus = 1 if 2 <= sampled_nunique <= 200 else 0
        return float(series.notna().mean()) + medium_cardinality_bonus, min(sampled_nunique, 500)

    categorical_cols = [col for col in df.columns if col not in seen and _is_string_or_object_dtype(df[col])]
    categorical_cols.sort(key=_categorical_score, reverse=True)
    for col in categorical_cols[:10]:
        add(col)

    for col in df.columns:
        add(col)
        if len(selected) >= max_columns:
            break
    return selected


def build_table_profiles(
    tables: dict[str, pd.DataFrame],
    base_ids: pd.Series,
    id_column: str,
    target_column: str,
    base_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    base_id_sample = set(base_ids.dropna().astype(str).head(1000))
    base_keys = base_keys or []
    for name, df in tables.items():
        columns: list[dict[str, Any]] = []
        for col in select_profile_columns(
            df,
            id_column=id_column,
            target_column=target_column,
            base_keys=base_keys,
        ):
            series = df[col]
            col_profile: dict[str, Any] = {
                "name": col,
                "dtype": str(series.dtype),
                "nunique": int(series.nunique(dropna=True)),
                "missing": int(series.isna().sum()),
            }
            if _is_string_or_object_dtype(series):
                values = series.dropna().astype(str).head(3).tolist()
                if values:
                    col_profile["examples"] = values
            columns.append(col_profile)
        possible_keys = []
        for col in df.columns:
            normalized = normalize_name(col)
            if normalized == normalize_name(id_column):
                possible_keys.append(col)
                continue
            overlap = len(base_id_sample & set(df[col].dropna().astype(str).head(1000)))
            if overlap >= 20:
                possible_keys.append(col)
        is_prejoin = name.startswith("__prejoin__")
        composite_grain: list[str] = []
        if is_prejoin and base_keys:
            composite_grain = [bk for bk in base_keys if bk in df.columns]
        profiles.append(
            {
                "table": name,
                "rows": int(len(df)),
                "columns": columns,
                "possible_join_keys": possible_keys[:5],
                "is_base": name in {"train.csv", "test.csv"},
                "target_column": target_column if name == "train.csv" else None,
                "kind": "prejoin" if is_prejoin else "regular",
                "composite_grain": composite_grain,
            }
        )
    return profiles


def extract_json_block(text: str) -> dict[str, Any]:
    if not text:
        return {}
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    payload = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    if not payload.strip():
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", payload)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}


def _format_enrichment_hint(enrichment_manifest: list[dict[str, Any]] | None) -> str:
    if not enrichment_manifest:
        return ""
    lines: list[str] = ["\nВ базовую таблицу уже добавлены 1:1 lookup-колонки из таблиц:"]
    for entry in enrichment_manifest:
        cols = entry.get("added_cols", [])
        shown = ", ".join(map(str, cols[:20]))
        if len(cols) > 20:
            shown += f", ... (+{len(cols) - 20})"
        prefix = entry.get("prefix", "")
        prefix_note = f" (prefix={prefix})" if prefix else ""
        lines.append(
            f" - {entry.get('table')} via {entry.get('join_key')}{prefix_note}: {shown}"
        )
    lines.append(
        "Эти колонки уже доступны через `direct` на базовой таблице — не нужно агрегировать."
    )
    return "\n".join(lines) + "\n"


def llm_generate_attempts(
    gigachat: Any,
    readme_text: str,
    tables: dict[str, pd.DataFrame],
    base_ids: pd.Series,
    id_column: str,
    target_column: str,
    base_keys: list[str] | None = None,
    enrichment_manifest: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if gigachat is None:
        return []

    profiles = build_table_profiles(tables, base_ids, id_column, target_column, base_keys=base_keys)
    composite_hint = ""
    if base_keys and len(base_keys) >= 2:
        composite_hint = (
            f"\nВ базовой таблице обнаружены составные ключи связи: {base_keys}. "
            f"Если в профилях есть таблицы `__prejoin__*` с `composite_grain`, содержащим эти ключи, "
            f"стройте признаки на уровне пары этих ключей через `group_keys`/`base_keys` — "
            f"такие признаки обычно сильнее, чем признаки уровня одного ключа.\n"
        )
    enrichment_hint = _format_enrichment_hint(enrichment_manifest)
    prompt = f"""
Ты проектируешь только кандидаты табличных признаков для бинарной классификации.

Контекст задачи:
- Есть train/test и дополнительные связанные таблицы.
- Главный ключ базовой сущности: {id_column}
- Целевая переменная: {target_column}
- Нужно предложить максимум {MAX_LLM_ATTEMPTS} попытки, максимум {MAX_LLM_FEATURES_PER_ATTEMPT} кандидатов признаков в каждой попытке.
- Используй только таблицы и колонки, перечисленные ниже.
- Если таблица не нужна, игнорируй ее.
- Отдавай только JSON без пояснений.
{composite_hint}{enrichment_hint}
Поддерживаемый DSL:
Каждая попытка должна содержать от {MAX_FEATURES} до {MAX_LLM_FEATURES_PER_ATTEMPT} кандидатов, отсортированных от самых сильных к самым слабым.
Исполнитель попытается материализовать кандидаты по порядку, отберет до {MAX_FEATURES} лучших валидных признаков и при необходимости автоматически дополнит попытку.
LLM сам решает, сколько из них:
- взять как уже существующие поля базовой таблицы;
- построить агрегациями;
- получить как комбинации ранее созданных признаков.
- `direct` используй только для колонок, которые уже есть в базовой таблице train/test.
- Если признак берется из любой другой таблицы (`users.csv`, `products.csv` и т.д.), описывай его как `aggregate` c `agg="first"` и корректными `group_key`/`base_key`.
- Избегай кандидатов, которые зависят от колонок, отсутствующих в перечисленных таблицах.

1. direct:
{{
  "name": "base_user_id",
  "kind": "direct",
  "column": "user_id"
}}

2. aggregate по одному ключу:
{{
  "name": "payments_mean_amount",
  "kind": "aggregate",
  "table": "payments.csv",
  "group_key": "client_id",
  "base_key": "client_id",
  "value_column": "amount",
  "agg": "mean",
  "filters": [{{"column": "status", "op": "==", "value": "approved"}}]
}}

3. binary_op:
{{
  "name": "avg_amount_per_payment",
  "kind": "binary_op",
  "op": "divide",
  "left": "payments_sum_amount",
  "right": "payments_count"
}}

4. aggregate с составной группировкой (для пары ключей в pre-joined таблицах):
{{
  "name": "pair_reorder_ratio",
  "kind": "aggregate",
  "table": "__prejoin__orders__order_items",
  "group_keys": ["user_id", "product_id"],
  "base_keys": ["user_id", "product_id"],
  "value_column": "reordered",
  "agg": "mean",
  "filters": []
}}

Разрешенные агрегации: {sorted(SUPPORTED_AGGS)}
Разрешенные операции: {sorted(SUPPORTED_BINARY_OPS)}
Используй `group_keys`/`base_keys` только для таблиц `__prejoin__*` с заполненным `composite_grain`.
В остальных случаях используй одноключевые агрегации `group_key`/`base_key`.
Если полезно, можешь не создавать новые признаки, а выбрать часть уже существующих полей базовой таблицы.

Описание данных:
{readme_text[:7000]}

Профили таблиц:
{json.dumps(profiles, ensure_ascii=False, indent=2)[:20000]}

Верни JSON формата:
{{
  "attempts": [
    {{
      "name": "attempt_name",
      "features": [ ... ]
    }}
  ]
}}
"""
    try:
        response = gigachat.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        parsed = extract_json_block(content)
        attempts = parsed.get("attempts", []) if isinstance(parsed, dict) else []
        if isinstance(attempts, list):
            return attempts[:MAX_LLM_ATTEMPTS]
    except Exception as exc:  # pragma: no cover - внешняя сеть/LLM
        logger.warning("Не удалось получить кандидаты от GigaChat: {}", exc)
    return []


def _validate_refine_specs(
    attempts: list[dict[str, Any]],
    available_feature_names: set[str],
) -> list[dict[str, Any]]:
    """Drop ``binary_op`` specs whose ``left``/``right`` reference unknown names.

    Walks each attempt in order, accumulating ``name`` values as they are
    declared so that a ``binary_op`` can reference an earlier spec in the same
    attempt. Any ``binary_op`` whose references cannot be resolved within that
    scope or against ``available_feature_names`` is silently dropped, with a
    per-attempt count logged.
    """
    validated: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        features = attempt.get("features")
        if not isinstance(features, list):
            continue
        kept: list[dict[str, Any]] = []
        known: set[str] = set(available_feature_names)
        dropped_count = 0
        for spec in features:
            if not isinstance(spec, dict):
                continue
            kind = spec.get("kind")
            name = spec.get("name")
            if kind == "binary_op":
                left = str(spec.get("left", ""))
                right = str(spec.get("right", ""))
                if left not in known or right not in known:
                    dropped_count += 1
                    continue
            kept.append(spec)
            if isinstance(name, str):
                known.add(name)
        if dropped_count:
            logger.info(
                "LLM refinement: {} binary_op specs dropped from {} due to unresolved refs",
                dropped_count,
                attempt.get("name", "?"),
            )
        if kept:
            validated.append({**attempt, "features": kept})
    return validated


def llm_refine_attempts(
    gigachat: Any,
    readme_text: str,
    tables: dict[str, pd.DataFrame],
    base_ids: pd.Series,
    id_column: str,
    target_column: str,
    base_keys: list[str] | None,
    evaluated: list[AttemptResult],
    enrichment_manifest: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Second-round LLM call: show top-3 scored attempts + importances, ask for 1-2 refinements."""
    if gigachat is None or not evaluated:
        return []

    top_k = sorted(evaluated, key=lambda r: r.cv_auc, reverse=True)[:3]
    if not top_k or top_k[0].cv_auc < 0.55:
        logger.info(
            "Skipping LLM refinement: best first-round CV AUC {:.4f} below noise floor",
            top_k[0].cv_auc if top_k else 0.0,
        )
        return []

    profiles = build_table_profiles(tables, base_ids, id_column, target_column, base_keys=base_keys)
    top_summary = []
    available_feature_names: set[str] = set()
    for r in top_k:
        top_imps = sorted(r.importances.items(), key=lambda item: item[1], reverse=True)[:MAX_FEATURES]
        top_summary.append(
            {
                "name": r.name,
                "cv_auc": round(r.cv_auc, 5),
                "features": r.selected_features,
                "top_importances": [
                    {"feature": name, "importance": round(float(imp), 4)}
                    for name, imp in top_imps
                ],
            }
        )
        available_feature_names.update(r.selected_features)
    # Include base columns (post-enrichment) so binary_op can combine them.
    if "train.csv" in tables:
        for col in tables["train.csv"].columns:
            if col not in {id_column, target_column}:
                available_feature_names.add(str(col))

    composite_hint = ""
    if base_keys and len(base_keys) >= 2:
        composite_hint = (
            f"\nВ базовой таблице обнаружены составные ключи связи: {base_keys}. "
            f"Pair-level агрегации на `__prejoin__*` таблицах обычно сильнее, "
            f"чем одиночные.\n"
        )
    enrichment_hint = _format_enrichment_hint(enrichment_manifest)
    refs_hint = (
        "\nЕсли используешь `binary_op`, ссылайся ТОЛЬКО на `name` признаков, "
        "определенных в этой же попытке ВЫШЕ по списку, или на те, что перечислены "
        "в разделе \"Доступные ранее созданные признаки\" (названия копируй точно).\n"
    )

    prompt = f"""
Ты уточняешь табличные признаки для бинарной классификации. Уже известен результат первого раунда.

Контекст:
- Главный ключ: {id_column}
- Целевая переменная: {target_column}
- Ниже — топ-3 лучших попытки из первого раунда с их CV AUC и важностями признаков.
- Предложи 1-2 НОВЫЕ попытки (не копируй существующие) по той же схеме DSL:
  (a) ratio/product/difference комбинации сильных признаков через `binary_op`;
  (b) новые агрегации на найденных join-ключах с `last`/`q75`/`skew`/`range`/`median`;
  (c) признаки для редких, но потенциально важных сегментов (через filters).
- Используй только таблицы и колонки, перечисленные ниже.
- Каждая попытка содержит от {MAX_FEATURES} до {MAX_LLM_FEATURES_PER_ATTEMPT} кандидатов, отсортированных от сильных к слабым.
- Отдавай только JSON без пояснений.
{composite_hint}{enrichment_hint}{refs_hint}
Разрешенные агрегации: {sorted(SUPPORTED_AGGS)}
Разрешенные операции: {sorted(SUPPORTED_BINARY_OPS)}

Доступные ранее созданные признаки:
{json.dumps(sorted(available_feature_names), ensure_ascii=False)[:3000]}

Топ-3 из первого раунда:
{json.dumps(top_summary, ensure_ascii=False, indent=2)[:6000]}

Описание данных:
{readme_text[:5000]}

Профили таблиц:
{json.dumps(profiles, ensure_ascii=False, indent=2)[:15000]}

Верни JSON формата:
{{
  "attempts": [
    {{
      "name": "refined_name",
      "features": [ ... ]
    }}
  ]
}}
"""
    try:
        response = gigachat.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        parsed = extract_json_block(content)
        attempts = parsed.get("attempts", []) if isinstance(parsed, dict) else []
        if isinstance(attempts, list):
            validated = _validate_refine_specs(attempts[:2], available_feature_names)
            logger.info(
                "LLM refinement produced {} attempts: {}",
                len(validated),
                [a.get("name", "?") for a in validated],
            )
            return validated
    except Exception as exc:  # pragma: no cover - external network/LLM
        logger.warning("LLM refinement call failed: {}", exc)
    return []


def normalize_group_key(hinted_key: Any, table_df: pd.DataFrame) -> str | None:
    if isinstance(hinted_key, str):
        return hinted_key if hinted_key in table_df.columns else None
    if isinstance(hinted_key, list):
        for item in hinted_key:
            if isinstance(item, str) and item in table_df.columns:
                return item
    return None


def choose_table_join_key(base_ids: pd.Series, table_df: pd.DataFrame, hinted_key: Any, id_column: str) -> str | None:
    normalized_hint = normalize_group_key(hinted_key, table_df)
    if normalized_hint:
        return normalized_hint
    if id_column in table_df.columns:
        return id_column

    base_sample = set(base_ids.dropna().astype(str).head(1000))
    best_key = None
    best_score = 0
    for col in table_df.columns:
        normalized = normalize_name(col)
        score = 0
        if normalized == normalize_name(id_column):
            score += 5
        overlap = len(base_sample & set(table_df[col].dropna().astype(str).head(1000)))
        score += overlap
        if score > best_score:
            best_score = score
            best_key = col
    return best_key if best_score >= 10 else None


def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    filtered = df
    for item in filters:
        col = item.get("column")
        op = item.get("op")
        value = item.get("value")
        if col not in filtered.columns:
            continue
        if op == "==":
            filtered = filtered[filtered[col] == value]
        elif op == "!=":
            filtered = filtered[filtered[col] != value]
        elif op == ">":
            filtered = filtered[pd.to_numeric(filtered[col], errors="coerce") > value]
        elif op == ">=":
            filtered = filtered[pd.to_numeric(filtered[col], errors="coerce") >= value]
        elif op == "<":
            filtered = filtered[pd.to_numeric(filtered[col], errors="coerce") < value]
        elif op == "<=":
            filtered = filtered[pd.to_numeric(filtered[col], errors="coerce") <= value]
    return filtered


def choose_base_join_key(base_frame: pd.DataFrame, group_key: str, hinted_base_key: Any, id_column: str) -> str | None:
    normalized_hint = normalize_group_key(hinted_base_key, base_frame)
    if normalized_hint:
        return normalized_hint
    if group_key in base_frame.columns:
        return group_key

    group_variants = candidate_key_names(group_key)
    for col in base_frame.columns:
        if candidate_key_names(col) & group_variants:
            return col
    if id_column in base_frame.columns:
        return id_column
    return None


def choose_join_columns(
    base_frame: pd.DataFrame,
    table_df: pd.DataFrame,
    hinted_table_key: Any,
    hinted_base_key: Any,
    id_column: str,
) -> tuple[str | None, str | None]:
    table_key = normalize_group_key(hinted_table_key, table_df)
    base_key = normalize_group_key(hinted_base_key, base_frame)
    if table_key and base_key:
        return table_key, base_key
    if table_key:
        return table_key, choose_base_join_key(base_frame, table_key, hinted_base_key, id_column)

    best_pair: tuple[str, str] | None = None
    best_score = 0
    base_samples = {
        col: set(base_frame[col].dropna().astype(str).head(2000))
        for col in base_frame.columns
    }
    table_samples = {
        col: set(table_df[col].dropna().astype(str).head(2000))
        for col in table_df.columns
    }
    for base_col in base_frame.columns:
        base_variants = candidate_key_names(base_col)
        for table_col in table_df.columns:
            score = 0
            if candidate_key_names(table_col) & base_variants:
                score += 10
            score += min(len(base_samples[base_col] & table_samples[table_col]), 20)
            if score > best_score:
                best_score = score
                best_pair = (table_col, base_col)
    return best_pair if best_pair and best_score > 0 else (None, None)


def _as_key_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, (str, int))]
    if isinstance(value, str):
        return [value]
    return []


def resolve_group_keys(
    spec: dict[str, Any],
    table_df: pd.DataFrame,
    base_frame: pd.DataFrame,
    base_ids: pd.Series,
    id_column: str,
) -> tuple[list[str], list[str]] | None:
    """Normalize legacy (group_key/base_key) and new (group_keys/base_keys) spec forms."""
    group_keys_spec = _as_key_list(spec.get("group_keys"))
    base_keys_spec = _as_key_list(spec.get("base_keys"))

    if group_keys_spec:
        resolved_group: list[str] = []
        for hint in group_keys_spec:
            resolved = choose_table_join_key(base_ids, table_df, hint, id_column)
            if not resolved or resolved not in table_df.columns:
                return None
            resolved_group.append(resolved)

        if base_keys_spec and len(base_keys_spec) == len(resolved_group):
            resolved_base: list[str] = []
            for hint, gkey in zip(base_keys_spec, resolved_group, strict=False):
                resolved = choose_base_join_key(base_frame, gkey, hint, id_column)
                if not resolved or resolved not in base_frame.columns:
                    return None
                resolved_base.append(resolved)
        else:
            resolved_base = []
            for gkey in resolved_group:
                resolved = choose_base_join_key(base_frame, gkey, None, id_column)
                if not resolved or resolved not in base_frame.columns:
                    return None
                resolved_base.append(resolved)

        if len(set(resolved_group)) != len(resolved_group) or len(set(resolved_base)) != len(resolved_base):
            return None
        return resolved_group, resolved_base

    group_key = choose_table_join_key(base_ids, table_df, spec.get("group_key"), id_column)
    if not group_key or group_key not in table_df.columns:
        return None
    base_key = choose_base_join_key(base_frame, group_key, spec.get("base_key"), id_column)
    if not base_key or base_key not in base_frame.columns:
        return None
    return [group_key], [base_key]


def build_aggregate_feature(
    base_frame: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    base_ids: pd.Series,
    id_column: str,
    spec: dict[str, Any],
) -> pd.Series | None:
    table_name = spec.get("table")
    if not isinstance(table_name, str):
        logger.warning("Пропускаем aggregate feature без корректного table: {}", spec)
        return None
    if table_name not in tables:
        logger.warning("Пропускаем aggregate feature, таблица не найдена: {}", spec)
        return None

    table_df = tables[table_name].copy()
    resolved = resolve_group_keys(spec, table_df, base_frame, base_ids, id_column)
    if resolved is None:
        logger.warning("Пропускаем aggregate feature, не найдены ключи: {}", spec)
        return None
    group_keys, base_keys = resolved

    filters = spec.get("filters", [])
    if isinstance(filters, list) and filters:
        table_df = apply_filters(table_df, filters)

    agg = str(spec.get("agg", "")).lower()
    if agg not in SUPPORTED_AGGS:
        logger.warning("Пропускаем aggregate feature, неподдерживаемая агрегация: {}", spec)
        return None

    value_column = spec.get("value_column")
    single_key = len(group_keys) == 1

    if agg == "count":
        if single_key:
            grouped = table_df.groupby(group_keys[0], dropna=False).size()
            return base_frame[base_keys[0]].map(grouped).reset_index(drop=True)
        out_col = "__agg_value__"
        grouped_df = (
            table_df.groupby(group_keys, dropna=False)
            .size()
            .reset_index(name=out_col)
        )
    else:
        if not isinstance(value_column, str) or value_column not in table_df.columns:
            logger.warning("Пропускаем aggregate feature, value_column не найден: {}", spec)
            return None
        series = table_df[value_column]
        if agg in {"sum", "mean", "min", "max", "std", "median", "q25", "q75", "skew", "range"}:
            series = pd.to_numeric(series, errors="coerce")
        keys_df = table_df[group_keys]
        grouped_source = pd.concat([keys_df.reset_index(drop=True), series.reset_index(drop=True).rename(value_column)], axis=1)
        gb = grouped_source.groupby(group_keys, dropna=False)[value_column]
        if agg == "nunique":
            grouped_obj = gb.nunique()
        elif agg == "first":
            grouped_obj = gb.first()
        elif agg == "last":
            grouped_obj = gb.last()
        elif agg == "sum":
            grouped_obj = gb.sum()
        elif agg == "mean":
            grouped_obj = gb.mean()
        elif agg == "min":
            grouped_obj = gb.min()
        elif agg == "max":
            grouped_obj = gb.max()
        elif agg == "std":
            grouped_obj = gb.std()
        elif agg == "median":
            grouped_obj = gb.median()
        elif agg == "q25":
            grouped_obj = gb.quantile(0.25)
        elif agg == "q75":
            grouped_obj = gb.quantile(0.75)
        elif agg == "skew":
            grouped_obj = gb.skew()
        elif agg == "range":
            grouped_obj = gb.max() - gb.min()
        else:
            grouped_obj = gb.median()

        if single_key:
            return base_frame[base_keys[0]].map(grouped_obj).reset_index(drop=True)
        out_col = "__agg_value__"
        grouped_df = grouped_obj.reset_index().rename(columns={value_column: out_col})

    rename_map = dict(zip(group_keys, base_keys, strict=False))
    merge_right = grouped_df.rename(columns=rename_map)
    merged = base_frame[base_keys].reset_index(drop=True).merge(
        merge_right, how="left", on=base_keys
    )
    return merged[out_col].reset_index(drop=True)


def build_binary_feature(feature_frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series | None:
    left = spec.get("left")
    right = spec.get("right")
    op = spec.get("op")
    if left not in feature_frame.columns or right not in feature_frame.columns:
        logger.warning("Пропускаем binary_op feature, отсутствуют входные колонки: {}", spec)
        return None
    left_series = pd.to_numeric(feature_frame[left], errors="coerce")
    right_series = pd.to_numeric(feature_frame[right], errors="coerce")
    if op == "divide":
        denominator = right_series.replace(0, np.nan)
        return left_series / denominator
    if op == "subtract":
        return left_series - right_series
    if op == "add":
        return left_series + right_series
    if op == "multiply":
        return left_series * right_series
    logger.warning("Пропускаем binary_op feature, неподдерживаемая операция: {}", spec)
    return None


def build_direct_feature(base_frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series | None:
    column = spec.get("column")
    if not isinstance(column, str):
        logger.warning("Пропускаем direct feature, колонка не найдена: {}", spec)
        return None
    if column in base_frame.columns:
        return base_frame[column].reset_index(drop=True)
    logger.warning("Пропускаем direct feature, колонка не найдена в базовой таблице: {}", spec)
    return None


def build_lookup_direct_feature(
    base_frame: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    id_column: str,
    spec: dict[str, Any],
) -> pd.Series | None:
    table_name = spec.get("table")
    column = spec.get("column")
    if not isinstance(table_name, str) or table_name not in tables or not isinstance(column, str):
        return None

    table_df = tables[table_name].copy()
    if column not in table_df.columns:
        logger.warning("Пропускаем lookup direct feature, колонка не найдена: {}", spec)
        return None

    table_key, base_key = choose_join_columns(
        base_frame=base_frame,
        table_df=table_df,
        hinted_table_key=spec.get("group_key"),
        hinted_base_key=spec.get("base_key"),
        id_column=id_column,
    )
    if not table_key or not base_key:
        logger.warning("Пропускаем lookup direct feature, не удалось подобрать ключи: {}", spec)
        return None

    lookup = (
        table_df[[table_key, column]]
        .dropna(subset=[table_key])
        .drop_duplicates(subset=[table_key], keep="first")
        .set_index(table_key)[column]
    )
    return base_frame[base_key].map(lookup)


def _resolve_direct_fallback(
    base_frame: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    id_column: str,
    spec: dict[str, Any],
) -> dict[str, Any] | None:
    """When a `direct` LLM spec lacks a known column, try to locate the column
    in a related table with a good join key. Returns a synthesized
    ``lookup_direct``-style spec (a `direct` spec augmented with ``table``,
    ``group_key``, and ``base_key`` hints) or None if no plausible match exists.
    """
    column = spec.get("column")
    if not isinstance(column, str):
        return None
    target_norm = normalize_name(column)

    best: tuple[float, str, str, str, str] | None = None
    for tname, tdf in tables.items():
        if tname in {"train.csv", "test.csv"}:
            continue
        if tname.startswith("__prejoin__"):
            continue
        # Exact or sanitized column match.
        candidate_col = None
        for col in tdf.columns:
            if col == column or normalize_name(col) == target_norm:
                candidate_col = col
                break
        if candidate_col is None:
            continue

        # Find join key: prefer a unique key with high overlap.
        picked = _pick_1to1_join_key(tdf, base_frame)
        if picked is None:
            continue
        base_key, related_key, overlap = picked
        if overlap < 0.9:
            continue
        score = overlap + (0.1 if related_key == id_column else 0.0)
        if best is None or score > best[0]:
            best = (score, tname, candidate_col, related_key, base_key)

    if best is None:
        return None
    _, tname, col, related_key, base_key = best
    return {
        **spec,
        "table": tname,
        "column": col,
        "group_key": related_key,
        "base_key": base_key,
    }


def build_completion_feature_specs(
    base_frame: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    id_column: str,
    target_column: str,
    related_attempts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    base_direct_cols = [col for col in base_frame.columns if col not in {id_column, target_column}]
    for col in base_direct_cols:
        specs.append({"name": f"base_{col}", "kind": "direct", "column": col})

    for attempt in related_attempts or build_related_table_attempts(tables, base_frame, id_column, target_column):
        specs.extend(attempt.get("features", []))
    return specs


def build_features_for_attempt(
    attempt: dict[str, Any],
    base_frame: pd.DataFrame,
    id_column: str,
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    feature_frame = pd.DataFrame({id_column: base_frame[id_column].reset_index(drop=True)})
    used_names = set(feature_frame.columns)
    name_map: dict[str, str] = {}

    for feature_spec in attempt.get("features", []):
        if not isinstance(feature_spec, dict):
            logger.warning("Пропускаем невалидный feature spec: {}", feature_spec)
            continue
        raw_name = str(feature_spec.get("name", "feature"))
        feature_name = safe_feature_name(raw_name, used_names)
        name_map[raw_name] = feature_name
        kind = feature_spec.get("kind")
        series: pd.Series | None = None
        if kind == "direct":
            series = build_direct_feature(base_frame, feature_spec)
            if series is None:
                series = build_lookup_direct_feature(base_frame, tables, id_column, feature_spec)
            if series is None:
                fallback_spec = _resolve_direct_fallback(base_frame, tables, id_column, feature_spec)
                if fallback_spec is not None:
                    logger.info(
                        "Direct-fallback: resolving {} -> table={} col={}",
                        feature_spec.get("column"),
                        fallback_spec.get("table"),
                        fallback_spec.get("column"),
                    )
                    series = build_lookup_direct_feature(base_frame, tables, id_column, fallback_spec)
        elif kind == "aggregate":
            series = build_aggregate_feature(base_frame, tables, base_frame[id_column], id_column, feature_spec)
        elif kind == "binary_op":
            binary_spec = dict(feature_spec)
            binary_spec["left"] = name_map.get(str(feature_spec.get("left")), feature_spec.get("left"))
            binary_spec["right"] = name_map.get(str(feature_spec.get("right")), feature_spec.get("right"))
            series = build_binary_feature(feature_frame, binary_spec)
        else:
            logger.warning("Пропускаем feature с неподдерживаемым kind: {}", feature_spec)
            continue
        if series is None:
            continue
        feature_frame[feature_name] = series
    return feature_frame


def build_fallback_row_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame({id_column: df[id_column]})
        feature_cols = [c for c in df.columns if c not in {id_column, target_column}]
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        result["ft_null_count"] = df[feature_cols].isna().sum(axis=1)
        if numeric_cols:
            num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            result["ft_numeric_mean"] = num_df.mean(axis=1)
            result["ft_numeric_std"] = num_df.std(axis=1)
            result["ft_numeric_max"] = num_df.max(axis=1)
            result["ft_numeric_min"] = num_df.min(axis=1)
        else:
            result["ft_numeric_mean"] = 0.0
            result["ft_numeric_std"] = 0.0
            result["ft_numeric_max"] = 0.0
            result["ft_numeric_min"] = 0.0
        return result

    return enrich(train_df), enrich(test_df)


def detect_base_join_keys(
    base_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    id_column: str,
    target_column: str,
) -> list[str]:
    """Detect which non-id columns in the base table look like join keys into related tables."""
    candidate_scores: dict[str, float] = {}
    base_cols = [c for c in base_df.columns if c not in {id_column, target_column}]
    for col in base_cols:
        base_vals = set(base_df[col].dropna().astype(str).head(5000))
        if len(base_vals) < 20:
            continue
        base_variants = candidate_key_names(col)
        best_ratio = 0.0
        for tname, tdf in tables.items():
            if tname in {"train.csv", "test.csv"}:
                continue
            for tcol in tdf.columns:
                t_vals = set(tdf[tcol].dropna().astype(str).head(5000))
                if not t_vals:
                    continue
                name_match = bool(candidate_key_names(tcol) & base_variants)
                overlap = len(base_vals & t_vals) / max(1, len(base_vals))
                if overlap < COMPOSITE_KEY_OVERLAP_MIN:
                    continue
                if not name_match and len(t_vals) < 100:
                    continue
                if overlap > best_ratio:
                    best_ratio = overlap
        if best_ratio >= COMPOSITE_KEY_OVERLAP_MIN:
            candidate_scores[col] = best_ratio
    ordered = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [col for col, _ in ordered[:4]]


def detect_prejoin_pairs(
    tables: dict[str, pd.DataFrame],
    base_keys: list[str],
) -> list[dict[str, Any]]:
    """Find pairs of related tables (A, B) whose inner join exposes all base_keys via a bridge key."""
    if len(base_keys) < 2:
        return []
    base_key_set = set(base_keys)
    candidates = [
        (name, df)
        for name, df in tables.items()
        if name not in {"train.csv", "test.csv"} and not name.startswith("__prejoin__")
    ]
    specs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for (a_name, a_df), (b_name, b_df) in combinations(candidates, 2):
        a_cols = set(a_df.columns)
        b_cols = set(b_df.columns)

        shared = set()
        for col in a_cols:
            for bcol in b_cols:
                if col == bcol or (candidate_key_names(col) & candidate_key_names(bcol)):
                    if col in a_df.columns and bcol in b_df.columns:
                        shared.add((col, bcol))

        for a_bridge, b_bridge in shared:
            if a_bridge in base_key_set or b_bridge in base_key_set:
                continue
            try:
                a_unique = a_df[a_bridge].dropna().astype(str).unique()
                b_unique = b_df[b_bridge].dropna().astype(str).unique()
                if len(a_unique) > 30000:
                    a_sample = set(pd.Series(a_unique).sample(n=30000, random_state=0))
                else:
                    a_sample = set(a_unique)
                b_full = set(b_unique)
            except Exception:
                continue
            if not a_sample or not b_full:
                continue
            overlap_ratio = len(a_sample & b_full) / max(1, len(a_sample))
            if overlap_ratio < 0.5:
                continue
            # Check that the combined schema covers all base keys
            combined_cols = a_cols | b_cols
            if not base_key_set.issubset(combined_cols):
                continue
            a_stem = a_name.replace(".csv", "")
            b_stem = b_name.replace(".csv", "")
            synth_name = f"__prejoin__{a_stem}__{b_stem}"
            if synth_name in seen_names:
                continue
            seen_names.add(synth_name)
            specs.append(
                {
                    "name": synth_name,
                    "left": a_name,
                    "right": b_name,
                    "left_on": a_bridge,
                    "right_on": b_bridge,
                    "base_keys": list(base_keys),
                }
            )
            break  # one bridge per pair
    return specs


def _downcast_for_merge(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_integer_dtype(series):
            try:
                out[col] = pd.to_numeric(series, downcast="integer")
            except Exception:
                pass
        elif pd.api.types.is_float_dtype(series):
            try:
                out[col] = pd.to_numeric(series, downcast="float")
            except Exception:
                pass
    return out


def materialize_prejoins(
    tables: dict[str, pd.DataFrame],
    specs: list[dict[str, Any]],
) -> None:
    """Mutate `tables` in place, adding the materialized prejoin tables."""
    for spec in specs:
        name = spec["name"]
        if name in tables:
            continue
        left_name = spec["left"]
        right_name = spec["right"]
        if left_name not in tables or right_name not in tables:
            continue
        left_df = tables[left_name]
        right_df = tables[right_name]
        left_on = spec["left_on"]
        right_on = spec["right_on"]
        base_keys = spec.get("base_keys", [])

        left_keep = [left_on] + [c for c in left_df.columns if c != left_on and (c in base_keys or pd.api.types.is_numeric_dtype(left_df[c]))]
        right_keep = [right_on] + [c for c in right_df.columns if c != right_on and (c in base_keys or pd.api.types.is_numeric_dtype(right_df[c]))]
        # Dedupe while preserving order
        seen_l: set[str] = set()
        left_cols = [c for c in left_keep if not (c in seen_l or seen_l.add(c))]
        seen_r: set[str] = set()
        right_cols = [c for c in right_keep if not (c in seen_r or seen_r.add(c))]

        left_proj = _downcast_for_merge(left_df[left_cols])
        right_proj = _downcast_for_merge(right_df[right_cols])

        try:
            if left_on == right_on:
                merged = left_proj.merge(right_proj, how="inner", on=left_on)
            else:
                merged = left_proj.merge(right_proj, how="inner", left_on=left_on, right_on=right_on)
                if right_on in merged.columns and right_on != left_on:
                    merged = merged.drop(columns=[right_on])
        except Exception as exc:
            logger.warning("Не удалось выполнить prejoin {}: {}", name, exc)
            continue

        # Verify all base keys survived the merge
        if not all(bk in merged.columns for bk in base_keys):
            logger.warning("Prejoin {} не содержит все base_keys {}", name, base_keys)
            continue

        tables[name] = merged
        logger.info(
            "Registered prejoin {}: {} rows, keys={}",
            name,
            len(merged),
            base_keys,
        )


def build_related_table_attempts(
    tables: dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    one_to_one_specs: list[tuple[int, dict[str, Any]]] = []
    aggregate_specs: list[tuple[int, dict[str, Any]]] = []
    diversity_specs: list[tuple[int, dict[str, Any]]] = []
    base_columns = [col for col in base_df.columns if col != target_column]
    base_samples = {
        col: set(base_df[col].dropna().astype(str).head(4000))
        for col in base_columns[:50]
    }

    for table_name, df in tables.items():
        if table_name in {"train.csv", "test.csv"}:
            continue
        join_candidates: list[tuple[int, str, str]] = []
        for base_key in base_columns:
            base_variant = candidate_key_names(base_key)
            for table_key in df.columns:
                score = 0
                if candidate_key_names(table_key) & base_variant:
                    score += 10
                sample_overlap = len(base_samples.get(base_key, set()) & set(df[table_key].dropna().astype(str).head(4000)))
                score += min(sample_overlap, 20)
                if score > 0:
                    join_candidates.append((score, base_key, table_key))

        if not join_candidates:
            continue

        selected_join_candidates: list[tuple[int, str, str]] = []
        seen_join_pairs: set[tuple[str, str]] = set()
        for score, base_key, join_key in sorted(join_candidates, reverse=True):
            join_pair = (base_key, join_key)
            if join_pair in seen_join_pairs:
                continue
            seen_join_pairs.add(join_pair)
            selected_join_candidates.append((score, base_key, join_key))
            if len(selected_join_candidates) >= RELATED_JOIN_CANDIDATES_PER_TABLE:
                break

        for score, base_key, join_key in selected_join_candidates:
            table_stem = table_name.replace(".csv", "")
            if str(base_key) == str(join_key):
                feature_prefix = f"{table_stem}_{base_key}"
            else:
                feature_prefix = f"{table_stem}_{base_key}_{join_key}"
            is_one_to_one = df[join_key].nunique(dropna=True) >= max(1, int(len(df) * 0.95))

            if is_one_to_one:
                candidate_cols = [col for col in df.columns if col != join_key][:10]
                for col in candidate_cols:
                    one_to_one_specs.append(
                        (
                            score,
                            {
                                "name": f"{feature_prefix}_{col}_first",
                                "kind": "aggregate",
                                "table": table_name,
                                "group_key": join_key,
                                "base_key": base_key,
                                "value_column": col,
                                "agg": "first",
                                "filters": [],
                            },
                        )
                    )

            aggregate_specs.append(
                (
                    score,
                    {
                        "name": f"{feature_prefix}_count",
                        "kind": "aggregate",
                        "table": table_name,
                        "group_key": join_key,
                        "base_key": base_key,
                        "value_column": None,
                        "agg": "count",
                        "filters": [],
                    },
                )
            )

            numeric_cols = [col for col in df.columns if col != join_key and pd.api.types.is_numeric_dtype(df[col])][:8]
            categorical_cols = [col for col in df.columns if col != join_key and _is_string_or_object_dtype(df[col])][:5]

            for col in numeric_cols:
                aggregate_specs.append(
                    (
                        score,
                        {
                            "name": f"{feature_prefix}_{col}_mean",
                            "kind": "aggregate",
                            "table": table_name,
                            "group_key": join_key,
                            "base_key": base_key,
                            "value_column": col,
                            "agg": "mean",
                            "filters": [],
                        },
                    )
                )
                diversity_specs.append(
                    (
                        score,
                        {
                            "name": f"{feature_prefix}_{col}_sum",
                            "kind": "aggregate",
                            "table": table_name,
                            "group_key": join_key,
                            "base_key": base_key,
                            "value_column": col,
                            "agg": "sum",
                            "filters": [],
                        },
                    )
                )
                diversity_specs.append(
                    (
                        score,
                        {
                            "name": f"{feature_prefix}_{col}_q75",
                            "kind": "aggregate",
                            "table": table_name,
                            "group_key": join_key,
                            "base_key": base_key,
                            "value_column": col,
                            "agg": "q75",
                            "filters": [],
                        },
                    )
                )
                diversity_specs.append(
                    (
                        score,
                        {
                            "name": f"{feature_prefix}_{col}_range",
                            "kind": "aggregate",
                            "table": table_name,
                            "group_key": join_key,
                            "base_key": base_key,
                            "value_column": col,
                            "agg": "range",
                            "filters": [],
                        },
                    )
                )
            for col in categorical_cols:
                diversity_specs.append(
                    (
                        score,
                        {
                            "name": f"{feature_prefix}_{col}_nunique",
                            "kind": "aggregate",
                            "table": table_name,
                            "group_key": join_key,
                            "base_key": base_key,
                            "value_column": col,
                            "agg": "nunique",
                            "filters": [],
                        },
                    )
                )

    if one_to_one_specs:
        ranked_specs = [spec for _, spec in sorted(one_to_one_specs, key=lambda item: item[0], reverse=True)]
        attempts.append({"name": "shared_key_firsts", "features": ranked_specs[:POOL_MAX_FEATURES]})
    if aggregate_specs:
        ranked_specs = [spec for _, spec in sorted(aggregate_specs, key=lambda item: item[0], reverse=True)]
        attempts.append({"name": "shared_key_counts_means", "features": ranked_specs[:POOL_MAX_FEATURES]})
    if diversity_specs:
        ranked_specs = [spec for _, spec in sorted(diversity_specs, key=lambda item: item[0], reverse=True)]
        attempts.append({"name": "shared_key_sums_nunique", "features": ranked_specs[:POOL_MAX_FEATURES]})
    return attempts


def _is_binary_like(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    vals = set(pd.unique(series.dropna()))
    if not vals:
        return False
    return vals.issubset({0, 1, 0.0, 1.0, True, False})


def _is_order_like(series: pd.Series) -> bool:
    if not pd.api.types.is_integer_dtype(series):
        return False
    try:
        clean = series.dropna()
        if clean.empty:
            return False
        if clean.min() < 0:
            return False
        if clean.max() >= 1e5:
            return False
    except Exception:
        return False
    return True


def build_pair_feature_attempts(
    tables: dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
    id_column: str,
    target_column: str,
    base_keys: list[str],
) -> list[dict[str, Any]]:
    """Emit composite-groupby pair-feature attempts on any registered prejoin tables."""
    if len(base_keys) < 2:
        return []

    attempts: list[dict[str, Any]] = []
    for tname, tdf in tables.items():
        if not tname.startswith("__prejoin__"):
            continue
        if not all(bk in tdf.columns for bk in base_keys):
            continue

        binary_cols: list[str] = []
        order_cols: list[str] = []
        generic_numeric_cols: list[str] = []
        for col in tdf.columns:
            if col in base_keys:
                continue
            series = tdf[col]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            if _is_binary_like(series):
                binary_cols.append(col)
            elif _is_order_like(series):
                order_cols.append(col)
            else:
                generic_numeric_cols.append(col)

        # Rank generic numeric by non-null fraction (stable proxy for usefulness).
        def _non_null_frac(c: str) -> float:
            return float(tdf[c].notna().mean())

        binary_cols.sort(key=_non_null_frac, reverse=True)
        order_cols.sort(key=_non_null_frac, reverse=True)
        generic_numeric_cols.sort(key=_non_null_frac, reverse=True)

        primary_base = base_keys[0]

        def pair_count_spec(name: str = "pair_count") -> dict[str, Any]:
            return {
                "name": name,
                "kind": "aggregate",
                "table": tname,
                "group_keys": list(base_keys),
                "base_keys": list(base_keys),
                "agg": "count",
                "value_column": None,
                "filters": [],
            }

        def scope_single_count_spec(name: str = "scope_user_count") -> dict[str, Any]:
            return {
                "name": name,
                "kind": "aggregate",
                "table": tname,
                "group_keys": [primary_base],
                "base_keys": [primary_base],
                "agg": "count",
                "value_column": None,
                "filters": [],
            }

        def pair_value_spec(col: str, agg: str, out: str) -> dict[str, Any]:
            return {
                "name": out,
                "kind": "aggregate",
                "table": tname,
                "group_keys": list(base_keys),
                "base_keys": list(base_keys),
                "agg": agg,
                "value_column": col,
                "filters": [],
            }

        def scope_value_spec(col: str, agg: str, out: str) -> dict[str, Any]:
            return {
                "name": out,
                "kind": "aggregate",
                "table": tname,
                "group_keys": [primary_base],
                "base_keys": [primary_base],
                "agg": agg,
                "value_column": col,
                "filters": [],
            }

        def binary_op_spec(op: str, out: str, left: str, right: str) -> dict[str, Any]:
            return {
                "name": out,
                "kind": "binary_op",
                "op": op,
                "left": left,
                "right": right,
            }

        # pair_core: canonical five
        core_features: list[dict[str, Any]] = [pair_count_spec("pair_count")]
        if binary_cols:
            core_features.append(pair_value_spec(binary_cols[0], "mean", f"pair_{binary_cols[0]}_mean"))
        if order_cols:
            oc = order_cols[0]
            core_features.append(pair_value_spec(oc, "max", f"pair_{oc}_max"))
            core_features.append(scope_value_spec(oc, "max", f"scope_{primary_base}_{oc}_max"))
            core_features.append(
                binary_op_spec("subtract", f"pair_{oc}_since_last", f"scope_{primary_base}_{oc}_max", f"pair_{oc}_max")
            )
        if len(core_features) < 5:
            core_features.append(scope_single_count_spec(f"scope_{primary_base}_count"))
            core_features.append(binary_op_spec("divide", "pair_frequency", "pair_count", f"scope_{primary_base}_count"))

        # pair_position: emphasize generic numerics / cart-like means
        position_features: list[dict[str, Any]] = [pair_count_spec("pair_count_p")]
        for col in generic_numeric_cols[:3]:
            position_features.append(pair_value_spec(col, "mean", f"pair_{col}_mean"))
        if binary_cols:
            position_features.append(pair_value_spec(binary_cols[0], "sum", f"pair_{binary_cols[0]}_sum"))
        if len(position_features) < 5 and order_cols:
            position_features.append(pair_value_spec(order_cols[0], "min", f"pair_{order_cols[0]}_min"))

        # pair_recency: emphasize recency / frequency signals
        recency_features: list[dict[str, Any]] = [pair_count_spec("pair_count_r")]
        if order_cols:
            oc = order_cols[0]
            recency_features.append(pair_value_spec(oc, "max", f"pair_{oc}_max_r"))
            recency_features.append(scope_value_spec(oc, "max", f"scope_{primary_base}_{oc}_max_r"))
            recency_features.append(
                binary_op_spec(
                    "subtract",
                    f"pair_{oc}_since_last_r",
                    f"scope_{primary_base}_{oc}_max_r",
                    f"pair_{oc}_max_r",
                )
            )
            recency_features.append(pair_value_spec(oc, "last", f"pair_{oc}_last_r"))
        else:
            recency_features.append(scope_single_count_spec(f"scope_{primary_base}_count_r"))
            recency_features.append(
                binary_op_spec("divide", "pair_frequency_r", "pair_count_r", f"scope_{primary_base}_count_r")
            )

        for name, feats in (
            ("pair_core", core_features),
            ("pair_position", position_features),
            ("pair_recency", recency_features),
        ):
            if len(feats) >= 2:
                attempts.append({"name": name, "features": feats[:POOL_MAX_FEATURES]})

    return attempts


def prepare_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    frame = df.copy()
    cat_features: list[int] = []
    for idx, col in enumerate(frame.columns):
        if _is_string_or_object_dtype(frame[col]):
            frame[col] = frame[col].fillna("__nan__").astype(str)
            cat_features.append(idx)
        else:
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)
    return frame, cat_features


def maybe_sample_for_eval(df: pd.DataFrame, target_column: str, sample_rows: int = EVAL_SAMPLE_ROWS) -> pd.DataFrame:
    if len(df) <= sample_rows:
        return df
    # Use index-based stratified sampling to avoid pandas 3.x groupby().apply()
    # dropping the grouping column from results.
    indices: list[int] = []
    per_class = sample_rows // max(1, df[target_column].nunique())
    for _, group in df.groupby(target_column, group_keys=False):
        n = min(len(group), max(1, per_class))
        indices.extend(group.sample(n=n, random_state=RANDOM_STATE).index.tolist())
    if len(indices) < sample_rows:
        extra_pool = df.drop(index=indices)
        extra_n = min(len(extra_pool), sample_rows - len(indices))
        if extra_n > 0:
            indices.extend(extra_pool.sample(n=extra_n, random_state=RANDOM_STATE).index.tolist())
    return df.loc[indices].reset_index(drop=True)


def build_cv_splits(y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]] | None:
    class_counts = y.value_counts(dropna=False)
    if len(class_counts) < 2:
        return None
    min_class_count = int(class_counts.min())
    n_splits = min(CV_FOLDS, min_class_count)
    if n_splits < 2:
        return None
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(np.zeros(len(y)), y))


def _stratified_sample_positions(
    train_df: pd.DataFrame,
    target_column: str,
    sample_rows: int,
    eligible_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Return positional indices (0..n-1) of a stratified sample from train_df.

    The caller is expected to have reset train_df's index so positional indices
    are meaningful. Matches `maybe_sample_for_eval`'s distribution logic.
    If `eligible_positions` is provided, the sample is restricted to that subset
    (so holdout rows are excluded from CV sampling).
    """
    reset = train_df.reset_index(drop=True)
    if eligible_positions is not None:
        pool = reset.iloc[eligible_positions]
    else:
        pool = reset
    n = len(pool)
    if n <= sample_rows:
        return np.asarray(sorted(pool.index.tolist()), dtype=np.int64)
    positions: list[int] = []
    per_class = sample_rows // max(1, int(pool[target_column].nunique()))
    for _, group in pool.groupby(target_column, group_keys=False):
        take = min(len(group), max(1, per_class))
        positions.extend(group.sample(n=take, random_state=RANDOM_STATE).index.tolist())
    if len(positions) < sample_rows:
        used = set(positions)
        extra_pool = pool.loc[~pool.index.isin(used)]
        take = min(len(extra_pool), sample_rows - len(positions))
        if take > 0:
            positions.extend(extra_pool.sample(n=take, random_state=RANDOM_STATE).index.tolist())
    return np.asarray(sorted(positions), dtype=np.int64)


def held_out_split_indices(
    train_df: pd.DataFrame,
    target_column: str,
    fraction: float = HOLDOUT_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train_cv / holdout split over positional indices of train_df.

    Returns `(train_cv_positions, holdout_positions)`.
    """
    n = len(train_df)
    if n == 0 or fraction <= 0 or fraction >= 1:
        return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
    reset = train_df.reset_index(drop=True)
    if target_column not in reset.columns:
        return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
    holdout: list[int] = []
    for _, group in reset.groupby(target_column, group_keys=False):
        take = max(1, int(round(len(group) * fraction)))
        take = min(take, max(0, len(group) - 1))
        if take <= 0:
            continue
        holdout.extend(group.sample(n=take, random_state=RANDOM_STATE).index.tolist())
    if not holdout:
        return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
    holdout_arr = np.asarray(sorted(set(holdout)), dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    mask[holdout_arr] = False
    train_cv_arr = np.where(mask)[0].astype(np.int64)
    return train_cv_arr, holdout_arr


def build_eval_cache(
    train_cv: pd.DataFrame,
    target_column: str,
    eligible_positions: np.ndarray | None = None,
) -> EvalCache | None:
    """Precompute shared eval/pool sample indices + CV folds for a train frame.

    If `eligible_positions` is set, sampling is constrained to that subset of
    train_cv rows (used to keep holdout rows out of the CV sample).
    """
    if target_column not in train_cv.columns or len(train_cv) == 0:
        return None
    reset = train_cv.reset_index(drop=True)

    sample_eval = _stratified_sample_positions(
        reset, target_column, EVAL_SAMPLE_ROWS, eligible_positions=eligible_positions
    )
    y_eval = reset.iloc[sample_eval][target_column].reset_index(drop=True)
    splits_eval = build_cv_splits(y_eval)
    if splits_eval is None:
        return None

    sample_pool = _stratified_sample_positions(
        reset, target_column, POOL_EVAL_SAMPLE_ROWS, eligible_positions=eligible_positions
    )
    y_pool = reset.iloc[sample_pool][target_column].reset_index(drop=True)
    splits_pool = build_cv_splits(y_pool)
    if splits_pool is None:
        splits_pool = splits_eval
        sample_pool = sample_eval

    return EvalCache(
        sample_indices_eval=sample_eval,
        splits_eval=splits_eval,
        sample_indices_pool=sample_pool,
        splits_pool=splits_pool,
    )


def compute_base_only_holdout_auc(
    train_df: pd.DataFrame,
    train_cv_positions: np.ndarray,
    holdout_positions: np.ndarray,
    id_column: str,
    target_column: str,
) -> tuple[float | None, list[str]]:
    """Fit a CatBoost on all base (post-enrichment) columns and score on holdout.

    Returns ``(auc, feature_cols)`` or ``(None, [])`` on any failure. This is
    the last-mile sanity line for the wide-lookup regression case: if the
    pipeline's winner can't beat this simple baseline, we fall back to a
    top-5 forward selection on these same columns.
    """
    if len(holdout_positions) < 20 or len(train_cv_positions) < 20:
        return None, []
    if target_column not in train_df.columns:
        return None, []

    feature_cols = [
        c for c in train_df.columns
        if c not in {id_column, target_column}
        and not _is_row_index_name(c)
    ]
    if not feature_cols:
        return None, []

    y_full = pd.to_numeric(train_df[target_column], errors="coerce").reset_index(drop=True)
    base_full = train_df[feature_cols].reset_index(drop=True)

    try:
        y_tr = y_full.iloc[train_cv_positions]
        y_ho = y_full.iloc[holdout_positions]
    except Exception:
        return None, []

    valid_tr = y_tr.notna()
    valid_ho = y_ho.notna()
    if valid_tr.sum() < 20 or valid_ho.sum() < 20:
        return None, []
    if y_ho[valid_ho].nunique() < 2:
        return None, []

    X_tr = base_full.iloc[train_cv_positions].loc[valid_tr.values]
    X_ho = base_full.iloc[holdout_positions].loc[valid_ho.values]
    y_tr_use = y_tr[valid_tr]
    y_ho_use = y_ho[valid_ho]

    try:
        X_tr_prep, cat_features = prepare_model_frame(X_tr)
        X_ho_prep, _ = prepare_model_frame(X_ho)
        for col in X_tr_prep.columns:
            if col in X_ho_prep.columns and X_tr_prep[col].dtype != X_ho_prep[col].dtype:
                X_ho_prep[col] = X_ho_prep[col].astype(X_tr_prep[col].dtype)
        params = {**MODEL_PARAMS, "iterations": 200}
        model = CatBoostClassifier(**params)
        model.fit(X_tr_prep, y_tr_use, cat_features=cat_features or None)
        probs = model.predict_proba(X_ho_prep)[:, 1]
        auc = float(roc_auc_score(y_ho_use, probs))
        return auc, feature_cols
    except Exception as exc:
        logger.warning("Base-only sanity fit failed: {}", exc)
        return None, []


def build_base_only_fallback_attempt(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_cv_positions: np.ndarray,
    holdout_positions: np.ndarray,
    id_column: str,
    target_column: str,
    feature_pool: list[str],
    eval_cache: EvalCache | None,
) -> AttemptResult | None:
    """Pick the top-MAX_FEATURES columns from *feature_pool* by single-feature
    holdout AUC, then build and evaluate that as an ``AttemptResult``. Used as
    the last-resort winner when every candidate regresses against the base.
    """
    if not feature_pool:
        return None
    if target_column not in train_df.columns:
        return None

    y_full = pd.to_numeric(train_df[target_column], errors="coerce").reset_index(drop=True)
    try:
        y_tr = y_full.iloc[train_cv_positions]
        y_ho = y_full.iloc[holdout_positions]
    except Exception:
        return None
    valid_tr = y_tr.notna()
    valid_ho = y_ho.notna()
    if valid_tr.sum() < 20 or valid_ho.sum() < 20:
        return None

    scores: list[tuple[float, str]] = []
    base_frame = train_df.reset_index(drop=True)
    for col in feature_pool:
        try:
            X_col = base_frame[[col]].iloc[train_cv_positions].loc[valid_tr.values]
            X_ho = base_frame[[col]].iloc[holdout_positions].loc[valid_ho.values]
            Xt, cat = prepare_model_frame(X_col)
            Xh, _ = prepare_model_frame(X_ho)
            for c in Xt.columns:
                if c in Xh.columns and Xt[c].dtype != Xh[c].dtype:
                    Xh[c] = Xh[c].astype(Xt[c].dtype)
            model = CatBoostClassifier(**{**MODEL_PARAMS, "iterations": 120})
            model.fit(Xt, y_tr[valid_tr], cat_features=cat or None)
            probs = model.predict_proba(Xh)[:, 1]
            auc = float(roc_auc_score(y_ho[valid_ho], probs))
            scores.append((auc, col))
        except Exception:
            continue

    scores.sort(reverse=True)
    top_cols = [col for _, col in scores[:MAX_FEATURES]]
    if len(top_cols) < MAX_FEATURES:
        remaining = [c for c in feature_pool if c not in top_cols]
        top_cols.extend(remaining[: MAX_FEATURES - len(top_cols)])
    top_cols = top_cols[:MAX_FEATURES]
    if not top_cols:
        return None

    used: set[str] = {id_column}
    sanitized_map: dict[str, str] = {}
    train_feat = pd.DataFrame({id_column: train_df[id_column].reset_index(drop=True)})
    test_feat = pd.DataFrame({id_column: test_df[id_column].reset_index(drop=True)})
    for col in top_cols:
        safe = safe_feature_name(col, used)
        sanitized_map[col] = safe
        train_feat[safe] = train_df[col].reset_index(drop=True)
        test_feat[safe] = test_df[col].reset_index(drop=True)
    feature_cols = [sanitized_map[c] for c in top_cols]

    result = evaluate_attempt(
        attempt_name="base_only_top5",
        train_out=attach_feature_frame(train_df, train_feat, id_column),
        test_out=attach_feature_frame(test_df, test_feat, id_column),
        feature_cols=feature_cols,
        target_column=target_column,
        id_column=id_column,
        eval_cache=eval_cache,
    )
    return result


def score_on_holdout(
    attempt: AttemptResult,
    train_df: pd.DataFrame,
    train_cv_positions: np.ndarray,
    holdout_positions: np.ndarray,
    target_column: str,
) -> float | None:
    """Train on train_cv rows, score on holdout rows, using the attempt's 5 features."""
    if len(holdout_positions) < 20 or len(train_cv_positions) < 20:
        return None
    if target_column not in train_df.columns:
        return None

    feature_cols = [c for c in attempt.selected_features if c in attempt.train_features.columns]
    if not feature_cols:
        return None

    y_full = pd.to_numeric(train_df[target_column], errors="coerce").reset_index(drop=True)
    features_full = attempt.train_features[feature_cols].reset_index(drop=True)
    if len(features_full) != len(y_full):
        return None

    try:
        y_tr = y_full.iloc[train_cv_positions]
        y_ho = y_full.iloc[holdout_positions]
    except Exception:
        return None

    valid_tr = y_tr.notna()
    valid_ho = y_ho.notna()
    if valid_tr.sum() < 20 or valid_ho.sum() < 20:
        return None
    if y_ho[valid_ho].nunique() < 2:
        return None

    X_tr = features_full.iloc[train_cv_positions].loc[valid_tr.values]
    X_ho = features_full.iloc[holdout_positions].loc[valid_ho.values]
    y_tr_use = y_tr[valid_tr]
    y_ho_use = y_ho[valid_ho]

    try:
        X_tr_prep, cat_features = prepare_model_frame(X_tr)
        X_ho_prep, _ = prepare_model_frame(X_ho)
        for col in X_tr_prep.columns:
            if col in X_ho_prep.columns and X_tr_prep[col].dtype != X_ho_prep[col].dtype:
                X_ho_prep[col] = X_ho_prep[col].astype(X_tr_prep[col].dtype)
        model = CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X_tr_prep, y_tr_use, cat_features=cat_features or None)
        probs = model.predict_proba(X_ho_prep)[:, 1]
        return float(roc_auc_score(y_ho_use, probs))
    except Exception as e:
        logger.warning("Holdout scoring failed for {}: {}", attempt.name, e)
        return None


def mean_cv_auc_for_features(
    X_prepared: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    categorical_cols: set[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_params: dict[str, Any],
) -> float | None:
    if not feature_cols:
        return None
    subset_cat_features = [idx for idx, col in enumerate(feature_cols) if col in categorical_cols]
    fold_scores: list[float] = []
    for train_idx, valid_idx in splits:
        X_train = X_prepared.iloc[train_idx][feature_cols]
        X_valid = X_prepared.iloc[valid_idx][feature_cols]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        model = CatBoostClassifier(**model_params)
        model.fit(X_train, y_train, cat_features=subset_cat_features or None)
        probs = model.predict_proba(X_valid)[:, 1]
        fold_scores.append(float(roc_auc_score(y_valid, probs)))
    if not fold_scores:
        return None
    return float(np.mean(fold_scores))


def choose_feature_subset_by_auc(
    attempt_name: str,
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    target_column: str,
    *,
    sample_rows: int,
    pre_rank: dict[str, float] | None = None,
    shortlist_size: int = ATTEMPT_SELECTION_SHORTLIST,
    model_params: dict[str, Any] | None = None,
    eval_cache: EvalCache | None = None,
) -> list[str]:
    usable_cols = [col for col in feature_cols if col in train_frame.columns]
    if len(usable_cols) <= MAX_FEATURES:
        return usable_cols

    ranked_cols = sorted(
        usable_cols,
        key=lambda col: ((pre_rank or {}).get(col, 0.0), col),
        reverse=True,
    )
    shortlist = ranked_cols[: max(MAX_FEATURES, shortlist_size)]

    cached_indices: np.ndarray | None = None
    cached_splits: list[tuple[np.ndarray, np.ndarray]] | None = None
    if eval_cache is not None:
        if sample_rows == POOL_EVAL_SAMPLE_ROWS:
            cached_indices = eval_cache.sample_indices_pool
            cached_splits = eval_cache.splits_pool
        else:
            cached_indices = eval_cache.sample_indices_eval
            cached_splits = eval_cache.splits_eval
        if cached_indices is not None and int(cached_indices.max(initial=-1)) >= len(train_frame):
            cached_indices = None
            cached_splits = None

    if cached_indices is not None:
        eval_train = train_frame.iloc[cached_indices].reset_index(drop=True)
    else:
        eval_train = maybe_sample_for_eval(train_frame, target_column, sample_rows=sample_rows)
    X = eval_train[shortlist].copy()
    y = eval_train[target_column]
    X_prepared, cat_features = prepare_model_frame(X)
    splits = cached_splits if cached_splits is not None else build_cv_splits(y)
    if splits is None:
        return shortlist[:MAX_FEATURES]

    categorical_cols = {X_prepared.columns[idx] for idx in cat_features}
    selection_model_params = model_params or SELECTION_MODEL_PARAMS
    selected: list[str] = []
    remaining = shortlist.copy()

    while remaining and len(selected) < MAX_FEATURES:
        best_col: str | None = None
        best_score = float("-inf")
        for col in remaining:
            candidate_cols = [*selected, col]
            candidate_score = mean_cv_auc_for_features(
                X_prepared,
                y,
                candidate_cols,
                categorical_cols,
                splits,
                selection_model_params,
            )
            if candidate_score is None:
                continue
            candidate_rank = (pre_rank or {}).get(col, 0.0)
            best_rank = (pre_rank or {}).get(best_col, float("-inf")) if best_col else float("-inf")
            if candidate_score > best_score + 1e-9 or (
                abs(candidate_score - best_score) <= 1e-9 and candidate_rank > best_rank
            ):
                best_col = col
                best_score = candidate_score
        if best_col is None:
            break
        selected.append(best_col)
        remaining.remove(best_col)

    for col in shortlist:
        if col not in selected:
            selected.append(col)
        if len(selected) >= MAX_FEATURES:
            break

    logger.info("AUC-driven feature selection for {} -> {}", attempt_name, selected[:MAX_FEATURES])
    return selected[:MAX_FEATURES]


def evaluate_attempt(
    attempt_name: str,
    train_out: pd.DataFrame,
    test_out: pd.DataFrame,
    feature_cols: list[str],
    target_column: str,
    id_column: str,
    *,
    sample_rows: int = EVAL_SAMPLE_ROWS,
    model_params: dict[str, Any] | None = None,
    eval_cache: EvalCache | None = None,
) -> AttemptResult | None:
    usable_cols = [col for col in feature_cols if col in train_out.columns and col in test_out.columns]
    if not usable_cols:
        return None

    train_frame = train_out.copy()
    test_frame = test_out.copy()

    cached_indices: np.ndarray | None = None
    cached_splits: list[tuple[np.ndarray, np.ndarray]] | None = None
    if eval_cache is not None:
        if sample_rows == POOL_EVAL_SAMPLE_ROWS:
            cached_indices = eval_cache.sample_indices_pool
            cached_splits = eval_cache.splits_pool
        else:
            cached_indices = eval_cache.sample_indices_eval
            cached_splits = eval_cache.splits_eval
        if cached_indices is not None and int(cached_indices.max(initial=-1)) >= len(train_frame):
            cached_indices = None
            cached_splits = None

    if cached_indices is not None:
        eval_train = train_frame.iloc[cached_indices].reset_index(drop=True)
    else:
        eval_train = maybe_sample_for_eval(train_frame, target_column, sample_rows=sample_rows)

    X = eval_train[usable_cols].copy()
    y = eval_train[target_column]
    X_prepared, cat_features = prepare_model_frame(X)

    if cached_splits is not None:
        splits = cached_splits
    else:
        splits = build_cv_splits(y)
    if splits is None:
        logger.warning("Attempt {} skipped: not enough class diversity for CV", attempt_name)
        return None

    current_model_params = model_params or SCREEN_MODEL_PARAMS
    fold_scores: list[float] = []
    importances = {col: 0.0 for col in usable_cols}

    try:
        for train_idx, valid_idx in splits:
            X_train = X_prepared.iloc[train_idx]
            X_valid = X_prepared.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            model = CatBoostClassifier(**current_model_params)
            model.fit(X_train, y_train, cat_features=cat_features or None)
            probs = model.predict_proba(X_valid)[:, 1]
            fold_scores.append(float(roc_auc_score(y_valid, probs)))

            fold_importance = dict(zip(X_prepared.columns, model.get_feature_importance().tolist(), strict=False))
            for col in usable_cols:
                importances[col] += float(fold_importance.get(col, 0.0))
    except Exception as exc:
        logger.warning("Attempt {} failed during CV: {}", attempt_name, exc)
        return None
    if not fold_scores:
        return None

    mean_auc = float(np.mean(fold_scores))
    importances = {k: v / len(fold_scores) for k, v in importances.items()}
    if len(usable_cols) <= MAX_FEATURES:
        selected = usable_cols
    else:
        selected = [name for name, _ in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:MAX_FEATURES]]

    return AttemptResult(
        name=attempt_name,
        train_features=train_frame[[id_column, *selected]].reset_index(drop=True),
        test_features=test_frame[[id_column, *selected]].reset_index(drop=True),
        cv_auc=mean_auc,
        selected_features=selected,
        importances=importances,
        train_pool=train_frame[[id_column, *usable_cols]].reset_index(drop=True),
        test_pool=test_frame[[id_column, *usable_cols]].reset_index(drop=True),
    )


def materialize_attempts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    attempts: list[dict[str, Any]],
    id_column: str,
    target_column: str,
    completion_specs: list[dict[str, Any]] | None = None,
    eval_cache: EvalCache | None = None,
) -> list[AttemptResult]:
    base_frame = pd.concat([train_df, test_df], ignore_index=True)
    train_count = len(train_df)
    results: list[AttemptResult] = []
    if completion_specs is None:
        completion_specs = build_completion_feature_specs(base_frame, tables, id_column, target_column)

    for attempt in attempts[:MAX_ATTEMPTS]:
        feature_frame = build_features_for_attempt(attempt, base_frame, id_column, tables)
        feature_cols = [col for col in feature_frame.columns if col != id_column]
        if len(feature_cols) < MAX_FEATURES:
            supplemented = build_features_for_attempt(
                {"name": f"{attempt.get('name', 'attempt')}_completion", "features": completion_specs},
                base_frame,
                id_column,
                tables,
            )
            added_cols: list[str] = []
            for col in supplemented.columns:
                if col == id_column or col in feature_frame.columns:
                    continue
                feature_frame[col] = supplemented[col]
                added_cols.append(col)
                if len([name for name in feature_frame.columns if name != id_column]) >= MAX_FEATURES:
                    break
            feature_cols = [col for col in feature_frame.columns if col != id_column]
            if added_cols:
                logger.info(
                    "Попытка {} была дополнена признаками {}",
                    attempt.get("name", "attempt"),
                    added_cols,
                )
        if not feature_cols:
            logger.warning(
                "Пропускаем попытку {}, потому что после материализации не осталось валидных признаков",
                attempt.get("name", "attempt"),
            )
            continue
        if len(feature_cols) < MAX_FEATURES:
            logger.warning(
                "Попытка {} после дополнения содержит только {} признаков из {} целевых",
                attempt.get("name", "attempt"),
                len(feature_cols),
                MAX_FEATURES,
            )

        train_features = feature_frame.iloc[:train_count].reset_index(drop=True)
        test_features = feature_frame.iloc[train_count:].reset_index(drop=True)

        # Early-drop degenerate attempts: if every feature is constant on train,
        # CatBoost will crash ("all features constant"). Skip cleanly.
        try:
            max_nunique = int(train_features[feature_cols].nunique(dropna=False).max())
        except Exception:
            max_nunique = 2
        if max_nunique < 2:
            logger.warning(
                "Пропускаем попытку {}: все признаки константны на train",
                attempt.get("name", "attempt"),
            )
            continue

        train_out = attach_feature_frame(train_df, train_features, id_column)
        test_out = attach_feature_frame(test_df, test_features, id_column)

        result = evaluate_attempt(
            attempt_name=attempt.get("name", "attempt"),
            train_out=train_out,
            test_out=test_out,
            feature_cols=feature_cols,
            target_column=target_column,
            id_column=id_column,
            eval_cache=eval_cache,
        )
        if result is not None:
            results.append(result)
            logger.info(
                "Attempt {} got CV AUC {:.5f} with features {}",
                result.name,
                result.cv_auc,
                result.selected_features,
            )

    return results


def refine_attempt_result(
    attempt: AttemptResult,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str,
    target_column: str,
    *,
    sample_rows: int = EVAL_SAMPLE_ROWS,
    shortlist_size: int = ATTEMPT_SELECTION_SHORTLIST,
    eval_cache: EvalCache | None = None,
) -> AttemptResult | None:
    pool_cols = [col for col in attempt.train_pool.columns if col != id_column and col in attempt.test_pool.columns]
    if not pool_cols:
        return None

    train_out = attach_feature_frame(train_df, attempt.train_pool, id_column)
    test_out = attach_feature_frame(test_df, attempt.test_pool, id_column)
    selected_cols = choose_feature_subset_by_auc(
        attempt.name,
        train_out,
        pool_cols,
        target_column,
        sample_rows=sample_rows,
        pre_rank=attempt.importances,
        shortlist_size=shortlist_size,
        eval_cache=eval_cache,
    )
    return evaluate_attempt(
        attempt_name=attempt.name,
        train_out=train_out,
        test_out=test_out,
        feature_cols=selected_cols,
        target_column=target_column,
        id_column=id_column,
        sample_rows=sample_rows,
        model_params=MODEL_PARAMS,
        eval_cache=eval_cache,
    )


def attach_feature_frame(base_df: pd.DataFrame, feature_df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    return pd.concat(
        [base_df.reset_index(drop=True), feature_df.drop(columns=[id_column]).reset_index(drop=True)],
        axis=1,
    )


def _column_hash(series: pd.Series) -> str:
    try:
        values = pd.util.hash_pandas_object(series.reset_index(drop=True), index=False).values
        return hashlib.md5(values.tobytes()).hexdigest()
    except Exception:
        return hashlib.md5(str(series.tolist()).encode("utf-8", errors="ignore")).hexdigest()


def _is_degenerate_col(
    train_series: pd.Series,
    train_target: pd.Series,
    sample_rows: int = 5000,
) -> tuple[bool, str]:
    """Return (is_degenerate, reason). Catches constants, near-constants, and leakers."""
    n = len(train_series)
    if n == 0:
        return True, "empty"
    try:
        unique_count = int(train_series.nunique(dropna=True))
    except Exception:
        unique_count = 2
    if unique_count <= 1:
        return True, "constant"
    try:
        vc = train_series.value_counts(dropna=False)
        if len(vc) > 0:
            top_share = float(vc.iloc[0]) / float(n)
            if top_share >= NEAR_CONSTANT_SHARE:
                return True, f"near_constant({top_share:.4f})"
    except Exception:
        pass
    if pd.api.types.is_numeric_dtype(train_series):
        try:
            if n > sample_rows:
                idx = np.random.RandomState(RANDOM_STATE).choice(n, size=sample_rows, replace=False)
                s = train_series.iloc[idx]
                t = train_target.iloc[idx]
            else:
                s = train_series
                t = train_target
            s_num = pd.to_numeric(s, errors="coerce")
            t_num = pd.to_numeric(t, errors="coerce")
            mask = s_num.notna() & t_num.notna()
            if mask.sum() >= 20:
                s_vals = s_num[mask].to_numpy()
                t_vals = t_num[mask].to_numpy()
                if np.std(s_vals) > 0 and np.std(t_vals) > 0:
                    corr = float(np.corrcoef(s_vals, t_vals)[0, 1])
                    if abs(corr) > LEAKAGE_CORR_THRESHOLD:
                        return True, f"target_leak(corr={corr:.4f})"
        except Exception:
            pass
    return False, ""


def pool_and_select_global_features(
    evaluated: list[AttemptResult],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str,
    target_column: str,
    eval_cache: EvalCache | None = None,
) -> AttemptResult | None:
    """Pool all materialized features from every attempt, CV-select a global top-5."""
    if not evaluated:
        return None

    train_pool = pd.DataFrame({id_column: train_df[id_column].reset_index(drop=True)})
    test_pool = pd.DataFrame({id_column: test_df[id_column].reset_index(drop=True)})
    train_target_series = train_df[target_column].reset_index(drop=True) if target_column in train_df.columns else None

    pre_rank: dict[str, float] = {}
    seen_hashes: set[str] = set()
    dropped_reasons: list[str] = []

    for attempt in evaluated:
        if attempt.train_pool.empty or attempt.test_pool.empty:
            continue
        if id_column not in attempt.train_pool.columns or id_column not in attempt.test_pool.columns:
            continue
        for col in attempt.train_pool.columns:
            if col == id_column:
                continue
            if col not in attempt.test_pool.columns:
                continue
            pooled_name = f"{attempt.name}__{col}"
            if pooled_name in train_pool.columns:
                continue
            series = attempt.train_pool[col]
            if pd.api.types.is_numeric_dtype(series):
                if series.dropna().empty:
                    continue
            if train_target_series is not None:
                series_for_check = series.reset_index(drop=True)
                if len(series_for_check) == len(train_target_series):
                    is_degen, reason = _is_degenerate_col(series_for_check, train_target_series)
                    if is_degen:
                        dropped_reasons.append(f"{pooled_name}:{reason}")
                        continue
                    if _is_row_index_name(col) or _is_row_index_series(series_for_check):
                        dropped_reasons.append(f"{pooled_name}:row_index")
                        continue
            else:
                if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) <= 1:
                    continue
                if _is_row_index_name(col) or _is_row_index_series(series.reset_index(drop=True)):
                    dropped_reasons.append(f"{pooled_name}:row_index")
                    continue
            col_hash = _column_hash(series)
            if col_hash in seen_hashes:
                continue
            seen_hashes.add(col_hash)
            train_pool[pooled_name] = series.reset_index(drop=True)
            test_pool[pooled_name] = attempt.test_pool[col].reset_index(drop=True)
            pre_rank[pooled_name] = float(attempt.importances.get(col, 0.0))

    if dropped_reasons:
        logger.info("Pool dropped {} degenerate/leaky cols: {}", len(dropped_reasons), dropped_reasons[:10])

    pooled_cols = [c for c in train_pool.columns if c != id_column]
    if len(pooled_cols) < MAX_FEATURES:
        return None

    if len(pooled_cols) > POOL_MAX_FEATURES:
        ranked = sorted(pooled_cols, key=lambda c: pre_rank.get(c, 0.0), reverse=True)
        keep = set(ranked[:POOL_MAX_FEATURES])
        drop_cols = [c for c in pooled_cols if c not in keep]
        if drop_cols:
            train_pool = train_pool.drop(columns=drop_cols)
            test_pool = test_pool.drop(columns=drop_cols)
        pooled_cols = [c for c in train_pool.columns if c != id_column]

    train_out = attach_feature_frame(train_df, train_pool, id_column)
    test_out = attach_feature_frame(test_df, test_pool, id_column)

    selection_pass = evaluate_attempt(
        attempt_name="__pool_select__",
        train_out=train_out,
        test_out=test_out,
        feature_cols=pooled_cols,
        target_column=target_column,
        id_column=id_column,
        sample_rows=POOL_EVAL_SAMPLE_ROWS,
        model_params=SCREEN_MODEL_PARAMS,
        eval_cache=eval_cache,
    )
    if selection_pass is None:
        return None

    final_cols = choose_feature_subset_by_auc(
        "__global_pool__",
        train_out,
        pooled_cols,
        target_column,
        sample_rows=POOL_EVAL_SAMPLE_ROWS,
        pre_rank=selection_pass.importances,
        shortlist_size=POOL_SELECTION_SHORTLIST,
        eval_cache=eval_cache,
    )
    if len(final_cols) < MAX_FEATURES:
        return None
    final_pass = evaluate_attempt(
        attempt_name="__global_pool__",
        train_out=train_out,
        test_out=test_out,
        feature_cols=final_cols,
        target_column=target_column,
        id_column=id_column,
        sample_rows=EVAL_SAMPLE_ROWS,
        model_params=MODEL_PARAMS,
        eval_cache=eval_cache,
    )
    if final_pass is None:
        return None
    # Stash the full pool on the result so downstream stackers / refiners can
    # reuse it without re-building.
    final_pass.train_pool = train_pool.copy()
    final_pass.test_pool = test_pool.copy()
    # Also stash the pre-rank so the stacker knows which raw cols were strongest.
    final_pass.importances = {**selection_pass.importances, **final_pass.importances}
    logger.info(
        "Global pool selection: {} pooled cols -> top {} = {}; final CV AUC {:.5f}",
        len(pooled_cols),
        MAX_FEATURES,
        final_cols,
        final_pass.cv_auc,
    )
    return final_pass


def build_stacked_meta_attempt(
    pool_result: AttemptResult | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str,
    target_column: str,
    train_cv_positions: np.ndarray,
    eval_cache: EvalCache | None = None,
) -> AttemptResult | None:
    """Stack a CatBoost on the full pool → emit OOF train pred + test pred as one feature.

    Combine the OOF meta with the top-4 raw pool cols (by importance) and run a
    normal evaluate_attempt on the 5 to get a CV AUC and importances. Guard:
    require pool ≥ MIN_POOL_COLS_FOR_STACK to be worth running.
    """
    if pool_result is None:
        return None
    train_pool = pool_result.train_pool
    test_pool = pool_result.test_pool
    if train_pool.empty or test_pool.empty:
        return None
    if id_column not in train_pool.columns or id_column not in test_pool.columns:
        return None

    pool_cols = [c for c in train_pool.columns if c != id_column and c in test_pool.columns]
    if len(pool_cols) < MIN_POOL_COLS_FOR_STACK:
        logger.info(
            "Skipping stacked meta: pool has {} cols < {}",
            len(pool_cols),
            MIN_POOL_COLS_FOR_STACK,
        )
        return None

    y_full = pd.to_numeric(train_df[target_column], errors="coerce").reset_index(drop=True)
    if y_full.isna().any():
        logger.warning("Skipping stacked meta: target has NaN values")
        return None

    train_cv_set = set(int(p) for p in train_cv_positions) if len(train_cv_positions) > 0 else None
    if train_cv_set is not None and len(train_cv_set) < 100:
        return None

    X_full_raw = train_pool[pool_cols].reset_index(drop=True)
    X_full_prep, cat_features = prepare_model_frame(X_full_raw)
    X_test_raw = test_pool[pool_cols].reset_index(drop=True)
    X_test_prep, _ = prepare_model_frame(X_test_raw)
    for col in X_full_prep.columns:
        if col in X_test_prep.columns and X_full_prep[col].dtype != X_test_prep[col].dtype:
            X_test_prep[col] = X_test_prep[col].astype(X_full_prep[col].dtype)

    if train_cv_set is not None:
        cv_mask = np.zeros(len(X_full_prep), dtype=bool)
        cv_mask[list(train_cv_set)] = True
    else:
        cv_mask = np.ones(len(X_full_prep), dtype=bool)

    # Downsample train_cv rows for stacker training to keep runtime in budget.
    # Reuse the eval_cache pool sample if it exists and sits inside cv_mask.
    cv_positions_arr = np.where(cv_mask)[0]
    sample_positions: np.ndarray
    if eval_cache is not None and len(eval_cache.sample_indices_pool) > 0:
        pool_sample = eval_cache.sample_indices_pool
        if int(pool_sample.max(initial=-1)) < len(X_full_prep):
            inside_cv = pool_sample[cv_mask[pool_sample]]
            if len(inside_cv) >= 100:
                sample_positions = inside_cv
            else:
                sample_positions = cv_positions_arr
        else:
            sample_positions = cv_positions_arr
    else:
        sample_positions = cv_positions_arr

    if len(sample_positions) > POOL_EVAL_SAMPLE_ROWS:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_positions = np.asarray(
            sorted(rng.choice(sample_positions, size=POOL_EVAL_SAMPLE_ROWS, replace=False)),
            dtype=np.int64,
        )

    X_cv = X_full_prep.iloc[sample_positions].reset_index(drop=True)
    y_cv = y_full.iloc[sample_positions].reset_index(drop=True)
    if y_cv.nunique() < 2 or len(y_cv) < 50:
        return None

    n_splits = min(5, int(y_cv.value_counts().min()))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_cv = np.full(len(X_cv), np.nan, dtype=np.float64)
    stack_params = {**MODEL_PARAMS, "iterations": STACKED_MODEL_ITERATIONS}
    try:
        for fold_train_idx, fold_valid_idx in skf.split(X_cv, y_cv):
            model = CatBoostClassifier(**stack_params)
            model.fit(
                X_cv.iloc[fold_train_idx],
                y_cv.iloc[fold_train_idx],
                cat_features=cat_features or None,
            )
            oof_cv[fold_valid_idx] = model.predict_proba(X_cv.iloc[fold_valid_idx])[:, 1]
    except Exception as e:
        logger.warning("Stacked meta OOF training failed: {}", e)
        return None

    # Single full fit on the sampled train_cv → used for both (a) filling
    # stacked preds for non-sampled train rows and (b) test predictions.
    try:
        full_model = CatBoostClassifier(**stack_params)
        full_model.fit(X_cv, y_cv, cat_features=cat_features or None)
    except Exception as e:
        logger.warning("Stacked meta full-fit failed: {}", e)
        return None

    stacked_oof_train = np.zeros(len(X_full_prep), dtype=np.float64)
    # Rows that were in the OOF sample → use their OOF preds.
    sample_mask = np.zeros(len(X_full_prep), dtype=bool)
    sample_mask[sample_positions] = True
    stacked_oof_train[sample_mask] = oof_cv
    # Everyone else (non-sampled train rows + holdout rows) → full-model pred.
    if (~sample_mask).any():
        try:
            stacked_oof_train[~sample_mask] = full_model.predict_proba(
                X_full_prep.loc[~sample_mask]
            )[:, 1]
        except Exception as e:
            logger.warning("Stacked meta fill failed: {}", e)
            return None

    try:
        stacked_oof_test = full_model.predict_proba(X_test_prep)[:, 1]
    except Exception as e:
        logger.warning("Stacked meta test prediction failed: {}", e)
        return None

    # Pick top-4 raw cols by pool importance, skipping any that duplicate the meta.
    importances = pool_result.importances or {}
    ranked_raw = sorted(pool_cols, key=lambda c: importances.get(c, 0.0), reverse=True)
    top_raw = ranked_raw[: MAX_FEATURES - 1]
    if len(top_raw) < MAX_FEATURES - 1:
        return None

    meta_name = "stacked_oof"
    train_features = pd.DataFrame({id_column: train_df[id_column].reset_index(drop=True)})
    train_features[meta_name] = stacked_oof_train
    for col in top_raw:
        train_features[col] = train_pool[col].reset_index(drop=True)

    test_features = pd.DataFrame({id_column: test_df[id_column].reset_index(drop=True)})
    test_features[meta_name] = stacked_oof_test
    for col in top_raw:
        test_features[col] = test_pool[col].reset_index(drop=True)

    feature_cols = [meta_name, *top_raw]
    train_out = attach_feature_frame(train_df, train_features, id_column)
    test_out = attach_feature_frame(test_df, test_features, id_column)
    result = evaluate_attempt(
        attempt_name="stacked_oof_meta",
        train_out=train_out,
        test_out=test_out,
        feature_cols=feature_cols,
        target_column=target_column,
        id_column=id_column,
        sample_rows=EVAL_SAMPLE_ROWS,
        model_params=MODEL_PARAMS,
        eval_cache=eval_cache,
    )
    if result is not None:
        logger.info(
            "Stacked OOF meta attempt: CV AUC {:.5f} features={}",
            result.cv_auc,
            result.selected_features,
        )
    return result


def format_output_frames(
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
    feature_train: pd.DataFrame,
    feature_test: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [col for col in feature_train.columns if col != id_column]
    train_output = pd.concat(
        [
            train_source[[id_column, target_column]].reset_index(drop=True),
            feature_train[feature_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    test_base_cols = [id_column]
    test_output = pd.concat(
        [
            test_source[test_base_cols].reset_index(drop=True),
            feature_test[feature_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    return train_output, test_output


def evaluate_public_test(
    train_df: pd.DataFrame,
    test_features_df: pd.DataFrame,
    source_test_df: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> float | None:
    if target_column not in source_test_df.columns:
        return None

    y_test = pd.to_numeric(source_test_df[target_column], errors="coerce")
    valid_mask = y_test.notna()
    if not valid_mask.any():
        return None

    test_eval = test_features_df.loc[valid_mask].copy()
    y_test = y_test.loc[valid_mask]
    if y_test.nunique() < 2:
        logger.warning("Пропускаем holdout test AUC: в test target только один класс")
        return None

    reserved = {target_column, id_column}
    feature_cols = [col for col in train_df.columns if col in test_eval.columns and col not in reserved]
    if not feature_cols:
        logger.warning("Пропускаем holdout test AUC: нет общих признаков для train/test")
        return None

    X_train = train_df[feature_cols].copy()
    y_train = pd.to_numeric(train_df[target_column], errors="coerce")
    train_mask = y_train.notna()
    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask]
    X_test = test_eval[feature_cols].copy()

    X_train_prepared, cat_features = prepare_model_frame(X_train)
    X_test_prepared, _ = prepare_model_frame(X_test)

    model = CatBoostClassifier(**MODEL_PARAMS)
    model.fit(X_train_prepared, y_train, cat_features=cat_features or None)
    test_proba = model.predict_proba(X_test_prepared)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_proba))
    logger.info("Holdout test AUC on input data: {:.5f}", test_auc)
    return test_auc


def make_agent_submission(gigachat: Any | None = None) -> None:
    t0 = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - t0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readme_text = parse_readme(DATA_DIR)
    tables = load_data_tables(DATA_DIR)
    if "train.csv" not in tables or "test.csv" not in tables:
        raise FileNotFoundError("В data/ должны быть train.csv и test.csv")

    train_df = tables["train.csv"].copy().reset_index(drop=True)
    test_df = tables["test.csv"].copy().reset_index(drop=True)
    target_column = infer_target_column(train_df, test_df, readme_text=readme_text)
    id_column = infer_id_column(train_df, test_df, target_column)
    logger.info("Определены ключевые колонки: id={}, target={}", id_column, target_column)

    train_df, test_df, enrichment_manifest = enrich_base_with_lookups(
        tables=tables,
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        target_column=target_column,
    )
    if enrichment_manifest:
        tables["train.csv"] = train_df
        tables["test.csv"] = test_df
        logger.info(
            "Base enriched with {} column(s) from {} lookup table(s)",
            sum(len(m["added_cols"]) for m in enrichment_manifest),
            len(enrichment_manifest),
        )

    train_cv_positions, holdout_positions = held_out_split_indices(
        train_df, target_column, fraction=HOLDOUT_FRACTION
    )
    logger.info(
        "Train holdout split: train_cv={}, train_holdout={}",
        len(train_cv_positions),
        len(holdout_positions),
    )
    eligible = train_cv_positions if len(holdout_positions) > 0 else None
    eval_cache = build_eval_cache(train_df, target_column, eligible_positions=eligible)
    if eval_cache is not None:
        logger.info(
            "Built shared eval cache: eval_sample={}, pool_sample={}, folds={}",
            len(eval_cache.sample_indices_eval),
            len(eval_cache.sample_indices_pool),
            len(eval_cache.splits_eval),
        )

    base_only_auc, base_only_feature_pool = compute_base_only_holdout_auc(
        train_df=train_df,
        train_cv_positions=train_cv_positions,
        holdout_positions=holdout_positions,
        id_column=id_column,
        target_column=target_column,
    )
    if base_only_auc is not None:
        logger.info(
            "Base-only holdout AUC: {:.5f} (on enriched base with {} cols)",
            base_only_auc,
            len(base_only_feature_pool),
        )

    base_concat = pd.concat([train_df, test_df], ignore_index=True)
    base_keys = detect_base_join_keys(base_concat, tables, id_column, target_column)
    if base_keys:
        logger.info("Detected composite base keys: {}", base_keys)
        prejoin_specs = detect_prejoin_pairs(tables, base_keys)
        if prejoin_specs:
            materialize_prejoins(tables, prejoin_specs)

    related_attempts = build_related_table_attempts(
        tables=tables,
        base_df=base_concat,
        id_column=id_column,
        target_column=target_column,
    )
    completion_specs = build_completion_feature_specs(
        base_frame=base_concat,
        tables=tables,
        id_column=id_column,
        target_column=target_column,
        related_attempts=related_attempts,
    )

    all_attempts: list[dict[str, Any]] = []
    all_attempts.extend(
        llm_generate_attempts(
            gigachat=gigachat,
            readme_text=readme_text,
            tables=tables,
            base_ids=pd.concat([train_df[id_column], test_df[id_column]], ignore_index=True),
            id_column=id_column,
            target_column=target_column,
            base_keys=base_keys,
            enrichment_manifest=enrichment_manifest,
        )
    )
    all_attempts.extend(related_attempts)
    all_attempts.extend(
        build_pair_feature_attempts(
            tables=tables,
            base_df=base_concat,
            id_column=id_column,
            target_column=target_column,
            base_keys=base_keys,
        )
    )

    evaluated = materialize_attempts(
        train_df=train_df,
        test_df=test_df,
        tables=tables,
        attempts=all_attempts,
        id_column=id_column,
        target_column=target_column,
        completion_specs=completion_specs,
        eval_cache=eval_cache,
    )

    # LLM refinement round: show the model the first-round winners + importances
    # and let it propose 1-2 informed follow-up attempts.
    if evaluated and elapsed() <= TIME_BUDGET_LLM_REFINE_SKIP:
        refine_attempts_specs = llm_refine_attempts(
            gigachat=gigachat,
            readme_text=readme_text,
            tables=tables,
            base_ids=pd.concat([train_df[id_column], test_df[id_column]], ignore_index=True),
            id_column=id_column,
            target_column=target_column,
            base_keys=base_keys,
            evaluated=evaluated,
            enrichment_manifest=enrichment_manifest,
        )
        if refine_attempts_specs:
            refined_evaluated = materialize_attempts(
                train_df=train_df,
                test_df=test_df,
                tables=tables,
                attempts=refine_attempts_specs,
                id_column=id_column,
                target_column=target_column,
                completion_specs=completion_specs,
                eval_cache=eval_cache,
            )
            evaluated.extend(refined_evaluated)
    elif evaluated:
        logger.warning(
            "Skipping LLM refinement (elapsed {:.1f}s > {:.0f}s budget)",
            elapsed(),
            float(TIME_BUDGET_LLM_REFINE_SKIP),
        )

    fallback_train, fallback_test = build_fallback_row_features(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        target_column=target_column,
    )
    fallback_cols = [col for col in fallback_train.columns if col != id_column][:MAX_FEATURES]
    try:
        fallback_max_nunique = int(fallback_train[fallback_cols].nunique(dropna=False).max())
    except Exception:
        fallback_max_nunique = 2
    if fallback_max_nunique < 2:
        logger.warning("Пропускаем base_row_statistics: все fallback-признаки константны")
        fallback_result = None
    else:
        fallback_result = evaluate_attempt(
            attempt_name="base_row_statistics",
            train_out=attach_feature_frame(train_df, fallback_train[[id_column, *fallback_cols]], id_column),
            test_out=attach_feature_frame(test_df, fallback_test[[id_column, *fallback_cols]], id_column),
            feature_cols=fallback_cols,
            target_column=target_column,
            id_column=id_column,
            eval_cache=eval_cache,
        )
    if fallback_result is not None:
        evaluated.append(fallback_result)
        logger.info(
            "Attempt {} got CV AUC {:.5f} with features {}",
            fallback_result.name,
            fallback_result.cv_auc,
            fallback_result.selected_features,
        )

    pooled_result = pool_and_select_global_features(
        evaluated=evaluated,
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        target_column=target_column,
        eval_cache=eval_cache,
    )

    stacked_result: AttemptResult | None = None
    if pooled_result is not None and elapsed() <= TIME_BUDGET_STACK_SKIP:
        stacked_result = build_stacked_meta_attempt(
            pool_result=pooled_result,
            train_df=train_df,
            test_df=test_df,
            id_column=id_column,
            target_column=target_column,
            train_cv_positions=train_cv_positions,
            eval_cache=eval_cache,
        )
    elif pooled_result is not None:
        logger.warning(
            "Skipping stacked meta (elapsed {:.1f}s > {:.0f}s budget)",
            elapsed(),
            float(TIME_BUDGET_STACK_SKIP),
        )

    final_evaluated: list[AttemptResult] = []
    if elapsed() > TIME_BUDGET_REFINE_SKIP:
        logger.warning(
            "Skipping refinement (elapsed {:.1f}s > {:.0f}s budget)",
            elapsed(),
            float(TIME_BUDGET_REFINE_SKIP),
        )
    else:
        for attempt in sorted(evaluated, key=lambda item: item.cv_auc, reverse=True)[:TOP_ATTEMPTS_TO_REFINE]:
            if elapsed() > TIME_BUDGET_REFINE_SKIP:
                logger.warning("Refinement loop budget exhausted; stopping early")
                break
            refined = refine_attempt_result(
                attempt,
                train_df=train_df,
                test_df=test_df,
                id_column=id_column,
                target_column=target_column,
                eval_cache=eval_cache,
            )
            final_evaluated.append(refined or attempt)

    candidates = [r for r in [pooled_result, stacked_result, *final_evaluated] if r is not None]
    if not candidates:
        candidates = [r for r in [pooled_result, stacked_result, *evaluated] if r is not None]
    if candidates:
        best_single = max((r.cv_auc for r in final_evaluated), default=0.0)
        pool_auc_str = f"{pooled_result.cv_auc:.5f}" if pooled_result else "n/a"

        run_holdout = (
            len(holdout_positions) >= 20
            and elapsed() <= TIME_BUDGET_REFINE_SKIP
        )
        best_attempt = max(candidates, key=lambda item: item.cv_auc)
        if run_holdout:
            top_k = sorted(candidates, key=lambda item: item.cv_auc, reverse=True)[:3]
            for cand in top_k:
                if elapsed() > TIME_BUDGET_REFINE_SKIP:
                    logger.warning("Holdout scoring budget exhausted; stopping early")
                    break
                ho = score_on_holdout(
                    cand,
                    train_df=train_df,
                    train_cv_positions=train_cv_positions,
                    holdout_positions=holdout_positions,
                    target_column=target_column,
                )
                cand.holdout_auc = ho
                logger.info(
                    "Holdout score: {} CV AUC {:.5f} holdout AUC {}",
                    cand.name,
                    cand.cv_auc,
                    f"{ho:.5f}" if ho is not None else "n/a",
                )
            scored_top = [c for c in top_k if c.holdout_auc is not None]
            if scored_top:
                best_attempt = max(
                    scored_top,
                    key=lambda item: (item.holdout_auc or float("-inf"), item.cv_auc),
                )

        # Regression guard: if every candidate regressed vs a base-only fit,
        # fall back to a top-5 single-feature selection on the enriched base.
        if (
            base_only_auc is not None
            and base_only_feature_pool
            and best_attempt.holdout_auc is not None
            and best_attempt.holdout_auc < base_only_auc - 0.01
            and elapsed() <= TIME_BUDGET_REFINE_SKIP
        ):
            logger.warning(
                "Winner {} degraded vs base-only ({:.5f} < {:.5f}-0.01); "
                "falling back to base-only top-5",
                best_attempt.name,
                best_attempt.holdout_auc,
                base_only_auc,
            )
            fallback_attempt = build_base_only_fallback_attempt(
                train_df=train_df,
                test_df=test_df,
                train_cv_positions=train_cv_positions,
                holdout_positions=holdout_positions,
                id_column=id_column,
                target_column=target_column,
                feature_pool=base_only_feature_pool,
                eval_cache=eval_cache,
            )
            if fallback_attempt is not None:
                fallback_holdout = score_on_holdout(
                    fallback_attempt,
                    train_df=train_df,
                    train_cv_positions=train_cv_positions,
                    holdout_positions=holdout_positions,
                    target_column=target_column,
                )
                fallback_attempt.holdout_auc = fallback_holdout
                logger.info(
                    "Base-only fallback {}: CV AUC {:.5f} holdout AUC {}",
                    fallback_attempt.name,
                    fallback_attempt.cv_auc,
                    f"{fallback_holdout:.5f}" if fallback_holdout is not None else "n/a",
                )
                if (
                    fallback_holdout is not None
                    and fallback_holdout > (best_attempt.holdout_auc or float("-inf"))
                ):
                    best_attempt = fallback_attempt

        holdout_auc_str = (
            f"{best_attempt.holdout_auc:.5f}" if best_attempt.holdout_auc is not None else "n/a"
        )
        base_only_str = (
            f"{base_only_auc:.5f}" if base_only_auc is not None else "n/a"
        )
        logger.info(
            "Winner: {} (CV AUC {:.5f}, holdout AUC {}); pool={} best_single={:.5f} base_only={}",
            best_attempt.name,
            best_attempt.cv_auc,
            holdout_auc_str,
            pool_auc_str,
            best_single,
            base_only_str,
        )
        final_train, final_test = format_output_frames(
            train_source=train_df,
            test_source=test_df,
            feature_train=best_attempt.train_features,
            feature_test=best_attempt.test_features,
            id_column=id_column,
            target_column=target_column,
        )
    else:
        logger.warning("Не удалось построить табличные агрегации, используем fallback-признаки по строкам")
        final_train, final_test = format_output_frames(
            train_source=train_df,
            test_source=test_df,
            feature_train=fallback_train[[id_column, *fallback_cols]],
            feature_test=fallback_test[[id_column, *fallback_cols]],
            id_column=id_column,
            target_column=target_column,
        )

    final_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    final_test.to_csv(OUTPUT_DIR / "test.csv", index=False)
    logger.info("Сохранены output/train.csv и output/test.csv")
    evaluate_public_test(
        train_df=final_train,
        test_features_df=final_test,
        source_test_df=test_df,
        target_column=target_column,
        id_column=id_column,
    )


def make_baseline_submission(*args: Any, **kwargs: Any) -> None:
    make_agent_submission(*args, **kwargs)
