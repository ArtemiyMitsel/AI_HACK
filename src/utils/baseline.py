from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
MAX_ATTEMPTS = 4
MAX_LLM_ATTEMPTS = 2
MAX_LLM_FEATURES_PER_ATTEMPT = 5
CV_FOLDS = 3
RANDOM_STATE = 42
EVAL_SAMPLE_ROWS = 20_000

MODEL_PARAMS = {
    "iterations": 200,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "thread_count": 1,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}

SUPPORTED_AGGS = {"count", "nunique", "sum", "mean", "min", "max", "std", "median", "first"}
SUPPORTED_BINARY_OPS = {"divide", "subtract", "add"}


@dataclass
class AttemptResult:
    name: str
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    cv_auc: float
    selected_features: list[str]
    importances: dict[str, float]


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


def infer_target_column(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    readme_text: str = "",
) -> str:
    readme_match = re.search(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*-\s*целевая переменная", readme_text, flags=re.M)
    if readme_match:
        candidate = readme_match.group(1)
        if candidate in train_df.columns:
            return candidate

    scored: list[tuple[float, str]] = []
    for col in train_df.columns:
        series = train_df[col].dropna()
        nunique = series.nunique()
        score = 0.0
        if nunique == 2:
            score += 10.0
        if normalize_name(col) in {"target", "label", "y", "flag", "default"}:
            score += 8.0
        if nunique <= 5:
            score += 2.0
        if col not in test_df.columns:
            score += 3.0
        if train_df[col].nunique(dropna=False) >= max(1, int(len(train_df) * 0.95)):
            score -= 6.0
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
        if train_unique:
            score += 5.0
        if test_unique:
            score += 5.0
        if normalize_name(col) in {"id", "clientid", "customerid", "userid", "applicationid", "requestid"}:
            score += 5.0
        scored.append((score, col))
    if scored:
        return max(scored)[1]
    if common_cols:
        return common_cols[0]
    raise ValueError("Не удалось определить id column")


def build_table_profiles(
    tables: dict[str, pd.DataFrame], base_ids: pd.Series, id_column: str, target_column: str
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    base_id_sample = set(base_ids.dropna().astype(str).head(1000))
    for name, df in tables.items():
        columns: list[dict[str, Any]] = []
        for col in df.columns[:30]:
            series = df[col]
            col_profile: dict[str, Any] = {
                "name": col,
                "dtype": str(series.dtype),
                "nunique": int(series.nunique(dropna=True)),
                "missing": int(series.isna().sum()),
            }
            if series.dtype == "object":
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
        profiles.append(
            {
                "table": name,
                "rows": int(len(df)),
                "columns": columns,
                "possible_join_keys": possible_keys[:5],
                "is_base": name in {"train.csv", "test.csv"},
                "target_column": target_column if name == "train.csv" else None,
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
) -> list[dict[str, Any]]:
    if gigachat is None:
        return []

    profiles = build_table_profiles(tables, base_ids, id_column, target_column)
    prompt = f"""
Ты проектируешь только кандидаты табличных признаков для бинарной классификации.

Контекст задачи:
- Есть train/test и дополнительные связанные таблицы.
- Главный ключ базовой сущности: {id_column}
- Целевая переменная: {target_column}
- Нужно предложить максимум {MAX_LLM_ATTEMPTS} попытки, максимум {MAX_LLM_FEATURES_PER_ATTEMPT} признаков в каждой попытке.
- Используй только таблицы и колонки, перечисленные ниже.
- Если таблица не нужна, игнорируй ее.
- Отдавай только JSON без пояснений.

Поддерживаемый DSL:
Каждая попытка должна содержать ровно {MAX_FEATURES} итоговых признаков.
LLM сам решает, сколько из них:
- взять как уже существующие поля базовой таблицы;
- построить агрегациями;
- получить как комбинации ранее созданных признаков.

1. direct:
{{
  "name": "base_user_id",
  "kind": "direct",
  "column": "user_id"
}}

2. aggregate:
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

Разрешенные агрегации: {sorted(SUPPORTED_AGGS)}
Разрешенные операции: {sorted(SUPPORTED_BINARY_OPS)}
Не придумывай сложные join-цепочки, только агрегации по одному ключу до базовой таблицы.
Если полезно, можешь не создавать новые признаки, а выбрать часть уже существующих полей базовой таблицы.

Описание данных:
{readme_text[:7000]}

Профили таблиц:
{json.dumps(profiles, ensure_ascii=False, indent=2)[:15000]}

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
    group_key = choose_table_join_key(base_ids, table_df, spec.get("group_key"), id_column)
    if not group_key or group_key not in table_df.columns:
        logger.warning("Пропускаем aggregate feature, не найден group_key: {}", spec)
        return None
    base_key = choose_base_join_key(base_frame, group_key, spec.get("base_key"), id_column)
    if not base_key or base_key not in base_frame.columns:
        logger.warning("Пропускаем aggregate feature, не найден base_key: {}", spec)
        return None

    filters = spec.get("filters", [])
    if isinstance(filters, list) and filters:
        table_df = apply_filters(table_df, filters)

    agg = str(spec.get("agg", "")).lower()
    if agg not in SUPPORTED_AGGS:
        logger.warning("Пропускаем aggregate feature, неподдерживаемая агрегация: {}", spec)
        return None

    value_column = spec.get("value_column")
    grouped: pd.Series
    if agg == "count":
        grouped = table_df.groupby(group_key, dropna=False).size()
    else:
        if value_column not in table_df.columns:
            logger.warning("Пропускаем aggregate feature, value_column не найден: {}", spec)
            return None
        series = table_df[value_column]
        if agg in {"sum", "mean", "min", "max", "std", "median"}:
            series = pd.to_numeric(series, errors="coerce")
        grouped_source = pd.DataFrame({group_key: table_df[group_key], value_column: series})
        if agg == "nunique":
            grouped = grouped_source.groupby(group_key, dropna=False)[value_column].nunique()
        elif agg == "first":
            grouped = grouped_source.groupby(group_key, dropna=False)[value_column].first()
        elif agg == "sum":
            grouped = grouped_source.groupby(group_key, dropna=False)[value_column].sum()
        elif agg == "mean":
            grouped = grouped_source.groupby(group_key, dropna=False)[value_column].mean()
        elif agg == "min":
            grouped = grouped_source.groupby(group_key, dropna=False)[value_column].min()
        elif agg == "max":
            grouped = grouped_source.groupby(group_key, dropna=False)[value_column].max()
        elif agg == "std":
            grouped = grouped_source.groupby(group_key, dropna=False)[value_column].std()
        else:
            grouped = grouped_source.groupby(group_key, dropna=False)[value_column].median()

    result = base_frame[base_key].map(grouped)
    return result


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
    logger.warning("Пропускаем binary_op feature, неподдерживаемая операция: {}", spec)
    return None


def build_direct_feature(base_frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series | None:
    column = spec.get("column")
    if not isinstance(column, str) or column not in base_frame.columns:
        logger.warning("Пропускаем direct feature, колонка не найдена: {}", spec)
        return None
    return base_frame[column].reset_index(drop=True)


def build_features_for_attempt(
    attempt: dict[str, Any],
    base_frame: pd.DataFrame,
    id_column: str,
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    feature_frame = pd.DataFrame({id_column: base_frame[id_column].reset_index(drop=True)})
    used_names = set(feature_frame.columns)
    name_map: dict[str, str] = {}

    for feature_spec in attempt.get("features", [])[:MAX_FEATURES]:
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


def build_related_table_attempts(
    tables: dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    one_to_one_specs: list[dict[str, Any]] = []
    aggregate_specs: list[dict[str, Any]] = []
    diversity_specs: list[dict[str, Any]] = []
    base_columns = [col for col in base_df.columns if col != target_column]
    base_samples = {
        col: set(base_df[col].dropna().astype(str).head(2000))
        for col in base_columns[:20]
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
                sample_overlap = len(base_samples.get(base_key, set()) & set(df[table_key].dropna().astype(str).head(2000)))
                score += min(sample_overlap, 20)
                if score > 0:
                    join_candidates.append((score, base_key, table_key))

        if not join_candidates:
            continue

        _, base_key, join_key = max(join_candidates)
        is_one_to_one = df[join_key].nunique(dropna=True) >= max(1, int(len(df) * 0.95))

        if is_one_to_one:
            candidate_cols = [col for col in df.columns if col != join_key][:4]
            for col in candidate_cols:
                one_to_one_specs.append(
                    {
                        "name": f"{table_name.replace('.csv', '')}_{col}_first",
                        "kind": "aggregate",
                        "table": table_name,
                        "group_key": join_key,
                        "base_key": base_key,
                        "value_column": col,
                        "agg": "first",
                        "filters": [],
                    }
                )

        aggregate_specs.append(
            {
                "name": f"{table_name.replace('.csv', '')}_{join_key}_count",
                "kind": "aggregate",
                "table": table_name,
                "group_key": join_key,
                "base_key": base_key,
                "value_column": None,
                "agg": "count",
                "filters": [],
            }
        )

        numeric_cols = [col for col in df.columns if col != join_key and pd.api.types.is_numeric_dtype(df[col])][:3]
        categorical_cols = [col for col in df.columns if col != join_key and df[col].dtype == "object"][:2]

        for col in numeric_cols:
            aggregate_specs.append(
                {
                    "name": f"{table_name.replace('.csv', '')}_{col}_mean",
                    "kind": "aggregate",
                    "table": table_name,
                    "group_key": join_key,
                    "base_key": base_key,
                    "value_column": col,
                    "agg": "mean",
                    "filters": [],
                }
            )
            diversity_specs.append(
                {
                    "name": f"{table_name.replace('.csv', '')}_{col}_sum",
                    "kind": "aggregate",
                    "table": table_name,
                    "group_key": join_key,
                    "base_key": base_key,
                    "value_column": col,
                    "agg": "sum",
                    "filters": [],
                }
            )
        for col in categorical_cols:
            diversity_specs.append(
                {
                    "name": f"{table_name.replace('.csv', '')}_{col}_nunique",
                    "kind": "aggregate",
                    "table": table_name,
                    "group_key": join_key,
                    "base_key": base_key,
                    "value_column": col,
                    "agg": "nunique",
                    "filters": [],
                }
            )

    if one_to_one_specs:
        attempts.append({"name": "shared_key_firsts", "features": one_to_one_specs[:MAX_FEATURES]})
    if aggregate_specs:
        attempts.append({"name": "shared_key_counts_means", "features": aggregate_specs[:MAX_FEATURES]})
    if diversity_specs:
        attempts.append({"name": "shared_key_sums_nunique", "features": diversity_specs[:MAX_FEATURES]})
    return attempts


def prepare_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    frame = df.copy()
    cat_features: list[int] = []
    for idx, col in enumerate(frame.columns):
        if frame[col].dtype == "object":
            frame[col] = frame[col].fillna("__nan__").astype(str)
            cat_features.append(idx)
        else:
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)
    return frame, cat_features


def maybe_sample_for_eval(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if len(df) <= EVAL_SAMPLE_ROWS:
        return df
    sampled = (
        df.groupby(target_column, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), EVAL_SAMPLE_ROWS // 2), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    return sampled


def evaluate_attempt(
    attempt_name: str,
    train_out: pd.DataFrame,
    test_out: pd.DataFrame,
    feature_cols: list[str],
    target_column: str,
    id_column: str,
) -> AttemptResult | None:
    usable_cols = [col for col in feature_cols if col in train_out.columns and col in test_out.columns]
    if not usable_cols:
        return None

    train_frame = train_out.copy()
    test_frame = test_out.copy()
    reserved = {target_column, id_column}
    eval_train = maybe_sample_for_eval(train_frame, target_column)

    X = eval_train[usable_cols].copy()
    y = eval_train[target_column]
    X_prepared, cat_features = prepare_model_frame(X)

    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores: list[float] = []
    importances = {col: 0.0 for col in usable_cols}

    for train_idx, valid_idx in splitter.split(X_prepared, y):
        X_train = X_prepared.iloc[train_idx]
        X_valid = X_prepared.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train, cat_features=cat_features or None)
        probs = model.predict_proba(X_valid)[:, 1]
        fold_scores.append(float(roc_auc_score(y_valid, probs)))

        fold_importance = dict(zip(X_prepared.columns, model.get_feature_importance().tolist(), strict=False))
        for col in usable_cols:
            importances[col] += float(fold_importance.get(col, 0.0))

    mean_auc = float(np.mean(fold_scores))
    importances = {k: v / CV_FOLDS for k, v in importances.items()}
    selected = [name for name, _ in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:MAX_FEATURES]]

    return AttemptResult(
        name=attempt_name,
        train_features=train_frame[[id_column, *selected]],
        test_features=test_frame[[id_column, *selected]],
        cv_auc=mean_auc,
        selected_features=selected,
        importances=importances,
    )


def materialize_attempts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    attempts: list[dict[str, Any]],
    id_column: str,
    target_column: str,
) -> list[AttemptResult]:
    base_frame = pd.concat([train_df, test_df], ignore_index=True)
    train_count = len(train_df)
    results: list[AttemptResult] = []

    for attempt in attempts[:MAX_ATTEMPTS]:
        feature_frame = build_features_for_attempt(attempt, base_frame, id_column, tables)
        feature_cols = [col for col in feature_frame.columns if col != id_column]
        if len(feature_cols) != MAX_FEATURES:
            logger.warning(
                "Пропускаем попытку {}, потому что после материализации получилось {} признаков вместо {}",
                attempt.get("name", "attempt"),
                len(feature_cols),
                MAX_FEATURES,
            )
            continue

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


def attach_feature_frame(base_df: pd.DataFrame, feature_df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    return pd.concat(
        [base_df.reset_index(drop=True), feature_df.drop(columns=[id_column]).reset_index(drop=True)],
        axis=1,
    )


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

    all_attempts: list[dict[str, Any]] = []
    all_attempts.extend(
        llm_generate_attempts(
            gigachat=gigachat,
            readme_text=readme_text,
            tables=tables,
            base_ids=pd.concat([train_df[id_column], test_df[id_column]], ignore_index=True),
            id_column=id_column,
            target_column=target_column,
        )
    )
    all_attempts.extend(
        build_related_table_attempts(
            tables=tables,
            base_df=pd.concat([train_df, test_df], ignore_index=True),
            id_column=id_column,
            target_column=target_column,
        )
    )

    evaluated = materialize_attempts(
        train_df=train_df,
        test_df=test_df,
        tables=tables,
        attempts=all_attempts,
        id_column=id_column,
        target_column=target_column,
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

    if evaluated:
        best_attempt = max(evaluated, key=lambda item: item.cv_auc)
        logger.info(
            "Выбрана попытка {} с CV AUC {:.5f}",
            best_attempt.name,
            best_attempt.cv_auc,
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
