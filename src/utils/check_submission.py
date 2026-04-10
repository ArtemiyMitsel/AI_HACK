from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
PYPROJECT_PATH = ROOT / "pyproject.toml"
RUN_PATH = ROOT / "run.py"
ENV_PATH = ROOT / ".env"

MAX_RUNTIME_SEC = 600
MAX_FEATURES = 5


def read_table(path: Path) -> pd.DataFrame:
    """
    Читает csv с автоопределением разделителя.
    """
    assert path.exists(), f"Файл не найден: {path}"
    return pd.read_csv(path, sep=None, engine="python")


def load_pyproject() -> dict:
    assert PYPROJECT_PATH.exists(), "Отсутствует pyproject.toml"
    with PYPROJECT_PATH.open("rb") as f:
        return tomllib.load(f)


def get_project_dependencies(pyproject: dict) -> list[str]:
    project = pyproject.get("project", {})
    deps = project.get("dependencies", [])
    assert isinstance(deps, list), "project.dependencies в pyproject.toml должен быть списком"
    return [str(x).lower() for x in deps]


def ensure_env_file() -> None:
    assert ENV_PATH.exists(), "Отсутствует .env"

    content = ENV_PATH.read_text(encoding="utf-8").split('\n')
    assert max(env_var.startswith("GIGACHAT_CREDENTIALS") for env_var in content), "В .env отсутствует GIGACHAT_CREDENTIALS"
    assert max(env_var.startswith("GIGACHAT_SCOPE") for env_var in content), "В .env отсутствует GIGACHAT_SCOPE"

def ensure_required_files() -> None:
    assert RUN_PATH.exists(), "Отсутствует run.py"
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), "Отсутствует папка data/"
    assert (DATA_DIR / "train.csv").exists(), "Отсутствует data/train.csv (нужно лишь для запуска локальной проверки)"
    assert (DATA_DIR / "test.csv").exists(), "Отсутствует data/test.csv (нужно лишь для запуска локальной проверки)"
    assert (DATA_DIR / "readme.txt").exists(), "Отсутствует data/readme.txt (нужно лишь для запуска локальной проверки)"


def ensure_dependencies() -> None:
    pyproject = load_pyproject()
    deps = get_project_dependencies(pyproject)

    required_markers = ["catboost", "pandas", "numpy",
                        "langchain-gigachat", "python-dotenv"]
    missing = [dep for dep in required_markers if not any(dep in x for x in deps)]

    assert not missing, (
        "В pyproject.toml отсутствуют обязательные зависимости "
        f"(или их нельзя однозначно распознать): {missing}"
    )


def clean_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_solution() -> tuple[int, float, str, str]:
    env = os.environ.copy()

    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "run.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=MAX_RUNTIME_SEC,
        env=env,
    )
    elapsed = time.perf_counter() - start

    return proc.returncode, elapsed, proc.stdout, proc.stderr


def assert_output_files_exist() -> tuple[Path, Path]:
    train_out = OUTPUT_DIR / "train.csv"
    test_out = OUTPUT_DIR / "test.csv"

    assert train_out.exists(), "После запуска не найден output/train.csv"
    assert test_out.exists(), "После запуска не найден output/test.csv"

    assert train_out.stat().st_size > 0, "output/train.csv пустой"
    assert test_out.stat().st_size > 0, "output/test.csv пустой"

    return train_out, test_out


def assert_output_structure(
    input_train: pd.DataFrame,
    input_test: pd.DataFrame,
    output_train: pd.DataFrame,
    output_test: pd.DataFrame,
) -> None:

    if "target" in input_train.columns:
        target_column = "target"
    else:
        train_only = [col for col in input_train.columns if col not in input_test.columns]
        assert len(train_only) == 1, (
            "Ожидалась одна колонка, которая есть только в train и отсутствует в test. "
            f"Найдено: {train_only}"
        )
        target_column = train_only[0]

    common_cols = [col for col in input_train.columns if col in input_test.columns and col != target_column]
    likely_id_candidates = [
        col
        for col in common_cols
        if input_train[col].nunique(dropna=False) >= max(1, int(len(input_train) * 0.95))
        and input_test[col].nunique(dropna=False) >= max(1, int(len(input_test) * 0.95))
    ]
    assert likely_id_candidates, "Не удалось определить ID-колонку по входным данным"
    id_column = likely_id_candidates[0]

    # 1. Проверка обязательных колонок
    assert id_column in output_train.columns, f"В output/train.csv отсутствует ID-колонка: {id_column}"
    assert target_column in output_train.columns, f"В output/train.csv отсутствует target-колонка: {target_column}"
    assert id_column in output_test.columns, f"В output/test.csv отсутствует ID-колонка: {id_column}"
    assert target_column not in output_test.columns, f"В output/test.csv не должно быть target-колонки: {target_column}"

    # 2. Проверка фичей
    reserved_train = {id_column, target_column}
    reserved_test = {id_column}

    feature_cols_train = [c for c in output_train.columns if c not in reserved_train]
    feature_cols_test = [c for c in output_test.columns if c not in reserved_test]

    assert feature_cols_train == feature_cols_test, (
        "Набор признаков в output/train.csv и output/test.csv должен совпадать по именам и порядку.\n"
        f"train features: {feature_cols_train}\n"
        f"test features: {feature_cols_test}"
    )

    assert len(feature_cols_train) == MAX_FEATURES, (
        f"Количество признаков должно быть ровно {MAX_FEATURES}, "
        f"получено: {len(feature_cols_train)}"
    )

    # 3. Проверка, что признаки содержат данные
    for col in feature_cols_train:
        assert not output_train[col].isna().all(), f"Признак {col} в train полностью NaN"
        assert not output_test[col].isna().all(), f"Признак {col} в test полностью NaN"

    # 4. Проверка на дубли имен колонок
    assert output_train.columns.is_unique, "В output/train.csv есть дублирующиеся имена колонок"
    assert output_test.columns.is_unique, "В output/test.csv есть дублирующиеся имена колонок"


def main() -> None:
    ensure_required_files()
    ensure_env_file()
    ensure_dependencies()

    input_train = read_table(DATA_DIR / "train.csv")
    input_test = read_table(DATA_DIR / "test.csv")

    clean_output_dir()

    try:
        returncode, elapsed, stdout, stderr = run_solution()
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"Решение превысило лимит времени {MAX_RUNTIME_SEC} секунд"
        ) from e

    assert returncode == 0, (
        "run.py завершился с ошибкой.\n"
        f"Return code: {returncode}\n\n"
        f"STDOUT:\n{stdout[-5000:]}\n\n"
        f"STDERR:\n{stderr[-5000:]}"
    )

    assert elapsed <= MAX_RUNTIME_SEC, (
        f"Решение работало слишком долго: {elapsed:.2f} сек. "
        f"Лимит: {MAX_RUNTIME_SEC} сек."
    )

    train_out_path, test_out_path = assert_output_files_exist()

    output_train = read_table(train_out_path)
    output_test = read_table(test_out_path)

    assert_output_structure(
        input_train=input_train,
        input_test=input_test,
        output_train=output_train,
        output_test=output_test,
    )

    print("OK: submit passed basic checks")
    print(f"Runtime: {elapsed:.2f} sec")
    print(f"Generated features: {len(output_test.columns) - len(input_test.columns)}")
    print(f"Output files: {train_out_path}, {test_out_path}")


if __name__ == "__main__":
    main()