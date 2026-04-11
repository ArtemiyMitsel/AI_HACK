from __future__ import annotations

import hashlib
import json
import re
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


DATA_DIR = Path("data")
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

MODEL_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "thread_count": 1,
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

SUPPORTED_AGGS = {"count", "nunique", "sum", "mean", "min", "max", "std", "median", "first"}
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


def load_data_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*.csv")):
        tables[path.name] = read_table(path)
    return tables


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


def llm_generate_attempts(
    gigachat: Any,
    readme_text: str,
    tables: dict[str, pd.DataFrame],
    base_ids: pd.Series,
    id_column: str,
    target_column: str,
    base_keys: list[str] | None = None,
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
{composite_hint}
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
        if agg in {"sum", "mean", "min", "max", "std", "median"}:
            series = pd.to_numeric(series, errors="coerce")
        keys_df = table_df[group_keys]
        grouped_source = pd.concat([keys_df.reset_index(drop=True), series.reset_index(drop=True).rename(value_column)], axis=1)
        if agg == "nunique":
            grouped_obj = grouped_source.groupby(group_keys, dropna=False)[value_column].nunique()
        elif agg == "first":
            grouped_obj = grouped_source.groupby(group_keys, dropna=False)[value_column].first()
        elif agg == "sum":
            grouped_obj = grouped_source.groupby(group_keys, dropna=False)[value_column].sum()
        elif agg == "mean":
            grouped_obj = grouped_source.groupby(group_keys, dropna=False)[value_column].mean()
        elif agg == "min":
            grouped_obj = grouped_source.groupby(group_keys, dropna=False)[value_column].min()
        elif agg == "max":
            grouped_obj = grouped_source.groupby(group_keys, dropna=False)[value_column].max()
        elif agg == "std":
            grouped_obj = grouped_source.groupby(group_keys, dropna=False)[value_column].std()
        else:
            grouped_obj = grouped_source.groupby(group_keys, dropna=False)[value_column].median()

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
            feature_prefix = f"{table_name.replace('.csv', '')}_{base_key}_{join_key}"
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
            recency_features.append(pair_value_spec(oc, "min", f"pair_{oc}_min_r"))
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
    eval_train = maybe_sample_for_eval(train_frame, target_column, sample_rows=sample_rows)
    X = eval_train[shortlist].copy()
    y = eval_train[target_column]
    X_prepared, cat_features = prepare_model_frame(X)
    splits = build_cv_splits(y)
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
) -> AttemptResult | None:
    usable_cols = [col for col in feature_cols if col in train_out.columns and col in test_out.columns]
    if not usable_cols:
        return None

    train_frame = train_out.copy()
    test_frame = test_out.copy()
    eval_train = maybe_sample_for_eval(train_frame, target_column, sample_rows=sample_rows)

    X = eval_train[usable_cols].copy()
    y = eval_train[target_column]
    X_prepared, cat_features = prepare_model_frame(X)

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
        train_out = attach_feature_frame(train_df, train_features, id_column)
        test_out = attach_feature_frame(test_df, test_features, id_column)

        result = evaluate_attempt(
            attempt_name=attempt.get("name", "attempt"),
            train_out=train_out,
            test_out=test_out,
            feature_cols=feature_cols,
            target_column=target_column,
            id_column=id_column,
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


def pool_and_select_global_features(
    evaluated: list[AttemptResult],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> AttemptResult | None:
    """Pool all materialized features from every attempt, CV-select a global top-5."""
    if not evaluated:
        return None

    train_pool = pd.DataFrame({id_column: train_df[id_column].reset_index(drop=True)})
    test_pool = pd.DataFrame({id_column: test_df[id_column].reset_index(drop=True)})

    pre_rank: dict[str, float] = {}
    seen_hashes: set[str] = set()

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
                if series.nunique(dropna=True) <= 1:
                    continue
            col_hash = _column_hash(series)
            if col_hash in seen_hashes:
                continue
            seen_hashes.add(col_hash)
            train_pool[pooled_name] = series.reset_index(drop=True)
            test_pool[pooled_name] = attempt.test_pool[col].reset_index(drop=True)
            pre_rank[pooled_name] = float(attempt.importances.get(col, 0.0))

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
    )
    if final_pass is None:
        return None
    logger.info(
        "Global pool selection: {} pooled cols -> top {} = {}; final CV AUC {:.5f}",
        len(pooled_cols),
        MAX_FEATURES,
        final_cols,
        final_pass.cv_auc,
    )
    return final_pass


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readme_text = parse_readme(DATA_DIR)
    tables = load_data_tables(DATA_DIR)
    if "train.csv" not in tables or "test.csv" not in tables:
        raise FileNotFoundError("В data/ должны быть train.csv и test.csv")

    train_df = tables["train.csv"].copy()
    test_df = tables["test.csv"].copy()
    target_column = infer_target_column(train_df, test_df, readme_text=readme_text)
    id_column = infer_id_column(train_df, test_df, target_column)
    logger.info("Определены ключевые колонки: id={}, target={}", id_column, target_column)

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
    )

    fallback_train, fallback_test = build_fallback_row_features(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        target_column=target_column,
    )
    fallback_cols = [col for col in fallback_train.columns if col != id_column][:MAX_FEATURES]
    fallback_result = evaluate_attempt(
        attempt_name="base_row_statistics",
        train_out=attach_feature_frame(train_df, fallback_train[[id_column, *fallback_cols]], id_column),
        test_out=attach_feature_frame(test_df, fallback_test[[id_column, *fallback_cols]], id_column),
        feature_cols=fallback_cols,
        target_column=target_column,
        id_column=id_column,
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
    )

    final_evaluated: list[AttemptResult] = []
    for attempt in sorted(evaluated, key=lambda item: item.cv_auc, reverse=True)[:TOP_ATTEMPTS_TO_REFINE]:
        refined = refine_attempt_result(
            attempt,
            train_df=train_df,
            test_df=test_df,
            id_column=id_column,
            target_column=target_column,
        )
        final_evaluated.append(refined or attempt)

    candidates = [r for r in [pooled_result, *final_evaluated] if r is not None]
    if not candidates:
        candidates = [r for r in [pooled_result, *evaluated] if r is not None]
    if candidates:
        best_attempt = max(candidates, key=lambda item: item.cv_auc)
        best_single = max((r.cv_auc for r in final_evaluated), default=0.0)
        pool_auc_str = f"{pooled_result.cv_auc:.5f}" if pooled_result else "n/a"
        logger.info(
            "Winner: {} (CV AUC {:.5f}); pool={} best_single={:.5f}",
            best_attempt.name,
            best_attempt.cv_auc,
            pool_auc_str,
            best_single,
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
