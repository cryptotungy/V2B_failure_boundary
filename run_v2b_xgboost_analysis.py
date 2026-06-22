#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run XGBoost analysis for V2B incremental saving data.

The script reads v2b_incremental_saving_dataset.csv, excludes leakage columns,
fits an overall intervention model and scenario-specific models, produces
model diagnostics, feature importance, binned marginal effects, interaction
analysis, and a Traditional Chinese Markdown report.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


try:
    mpl_cache = Path.cwd() / ".matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
except Exception:
    pass
os.environ.setdefault("MPLBACKEND", "Agg")


def require_packages() -> None:
    missing: list[str] = []
    required = [
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "xgboost",
    ]
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(
            "ERROR: Missing required Python packages: "
            + ", ".join(missing)
            + "\nInstall them first, for example:\n"
            + "  pip install numpy pandas matplotlib scikit-learn xgboost",
            file=sys.stderr,
        )
        raise SystemExit(1)


require_packages()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


RANDOM_STATE = 42
BASELINE_SCENARIO = "S0_PV_only_unmanaged_EV"
ANALYSIS_SCENARIOS = [
    "S1_PV_battery",
    "S2_PV_smart_EV_charging",
    "S3_PV_V2B",
    "S4_PV_battery_V2B",
]
ALL_SCENARIOS = [BASELINE_SCENARIO] + ANALYSIS_SCENARIOS


FIELD_GROUPS: dict[str, dict[str, Any]] = {
    "基本情境欄位": {
        "description": "識別同一組外生條件、調度策略類型與天氣型態。",
        "columns": ["base_case_id", "scenario", "weather_type"],
    },
    "PV 特徵": {
        "description": "描述 PV 裝置規模、發電量、尖峰功率，以及在儲能介入前的 PV 餘電程度。",
        "columns": [
            "PV_install_rate",
            "pv_total_kWh",
            "pv_peak_kW",
            "pv_midday_kWh",
            "pv_to_load_ratio",
            "pv_surplus_before_storage_kWh",
            "pv_surplus_ratio",
        ],
    },
    "電價特徵": {
        "description": "描述最低/最高電價、價差，以及實際可被調度利用的電價 spread。",
        "columns": [
            "price_gap",
            "min_price",
            "max_price",
            "actual_price_spread",
            "price_gap_ratio",
        ],
    },
    "EV 特徵": {
        "description": "描述 EV 數量、容量、到離場時間、SOC 需求與可調度彈性。",
        "columns": [
            "ev_count",
            "ev_total_capacity_kWh",
            "avg_arrival_time",
            "avg_departure_time",
            "avg_parking_hours",
            "avg_initial_soc",
            "avg_target_soc",
            "ev_to_load_ratio",
            "parking_flexibility",
            "ev_peak_available_hours",
        ],
    },
    "固定電池特徵": {
        "description": "描述固定電池數量、容量、功率，以及相對於負載與 PV 的配置比例。",
        "columns": [
            "bat_count",
            "battery_total_capacity_kWh",
            "battery_power_kW",
            "battery_to_load_ratio",
            "storage_to_pv_ratio",
        ],
    },
    "負載特徵": {
        "description": "描述建築用電總量、尖峰負載與尖峰發生時間。",
        "columns": ["load_total_kWh", "load_peak_kW", "load_peak_time"],
    },
    "成本與目標欄位": {
        "description": "描述 PV-only baseline 成本、各 scenario 成本，以及相對 baseline 的額外 saving。",
        "columns": [
            "pv_only_baseline_cost",
            "scenario_cost",
            "incremental_saving",
            "incremental_saving_rate",
        ],
    },
    "調度結果欄位": {
        "description": "描述調度後的電網購電、PV 自用/棄電、電池與 EV 充放電結果；這些是模型不可使用的事後結果。",
        "columns": [
            "grid_import_total_kWh",
            "grid_import_peak_kW",
            "pv_self_consumption_kWh",
            "pv_self_consumption_rate",
            "pv_curtailment_kWh",
            "battery_charge_kWh",
            "battery_discharge_kWh",
            "battery_cycles",
            "ev_charge_kWh",
            "ev_discharge_kWh",
            "ev_unmet_kWh",
            "ev_departure_violation_count",
        ],
    },
    "case-level 比較欄位": {
        "description": "以同一 base case 內不同策略互相比較而來的 saving 指標；這些是目標衍生欄位，不能作為模型特徵。",
        "columns": [
            "battery_extra_saving",
            "smart_ev_extra_saving",
            "v2b_total_extra_saving",
            "v2b_extra_over_smart_charging",
            "integrated_extra_saving",
            "battery_v2b_interaction_saving",
            "S4_vs_best_single_saving",
        ],
    },
}


SCENARIO_SWITCH_COLUMNS = [
    "battery_enabled",
    "smart_ev_charging_enabled",
    "ev_v2b_enabled",
]


LEAKAGE_COLUMNS = {
    "base_case_id",
    "scenario",
    "incremental_saving",
    "incremental_saving_rate",
    "incremental_saving_per_EV",
    "incremental_saving_per_battery",
    "pv_only_baseline_cost",
    "scenario_cost",
    "grid_import_total_kWh",
    "grid_import_peak_kW",
    "pv_self_consumption_kWh",
    "pv_self_consumption_rate",
    "pv_curtailment_kWh",
    "battery_charge_kWh",
    "battery_discharge_kWh",
    "battery_cycles",
    "ev_charge_kWh",
    "ev_discharge_kWh",
    "ev_unmet_kWh",
    "ev_departure_violation_count",
    "battery_extra_saving",
    "smart_ev_extra_saving",
    "v2b_total_extra_saving",
    "v2b_extra_over_smart_charging",
    "integrated_extra_saving",
    "battery_v2b_interaction_saving",
    "S4_vs_best_single_saving",
}


GROUP_IMPORTANCE_COLUMNS = {
    "PV group": [
        "PV_install_rate",
        "pv_total_kWh",
        "pv_peak_kW",
        "pv_midday_kWh",
        "pv_to_load_ratio",
        "pv_surplus_before_storage_kWh",
        "pv_surplus_ratio",
    ],
    "Price group": [
        "price_gap",
        "min_price",
        "max_price",
        "actual_price_spread",
        "price_gap_ratio",
    ],
    "EV group": [
        "ev_count",
        "ev_total_capacity_kWh",
        "avg_arrival_time",
        "avg_departure_time",
        "avg_parking_hours",
        "avg_initial_soc",
        "avg_target_soc",
        "ev_to_load_ratio",
        "parking_flexibility",
        "ev_peak_available_hours",
    ],
    "Battery group": [
        "bat_count",
        "battery_total_capacity_kWh",
        "battery_power_kW",
        "battery_to_load_ratio",
        "storage_to_pv_ratio",
    ],
    "Load group": ["load_total_kWh", "load_peak_kW", "load_peak_time"],
    "Scenario group": SCENARIO_SWITCH_COLUMNS,
}


BINNED_FEATURES = [
    "price_gap",
    "actual_price_spread",
    "pv_surplus_before_storage_kWh",
    "pv_surplus_ratio",
    "ev_count",
    "bat_count",
    "parking_flexibility",
    "ev_peak_available_hours",
    "storage_to_pv_ratio",
]


BINNED_FIGURES = {
    "price_gap": "fig_binned_price_gap.png",
    "pv_surplus_before_storage_kWh": "fig_binned_pv_surplus.png",
    "ev_count": "fig_binned_ev_count.png",
    "bat_count": "fig_binned_bat_count.png",
    "parking_flexibility": "fig_binned_parking_flexibility.png",
    "ev_peak_available_hours": "fig_binned_ev_peak_available_hours.png",
}


INTERACTION_METRICS = [
    "battery_v2b_interaction_saving",
    "S4_vs_best_single_saving",
    "v2b_extra_over_smart_charging",
]


S4_CONDITION_FEATURES = {
    "price_gap": "price_gap",
    "PV surplus": "pv_surplus_before_storage_kWh",
    "parking_flexibility": "parking_flexibility",
    "ev_peak_available_hours": "ev_peak_available_hours",
    "storage_to_pv_ratio": "storage_to_pv_ratio",
}


@dataclass
class ModelResult:
    name: str
    model: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    groups_train: pd.Series | None
    groups_test: pd.Series | None
    feature_columns: list[str]
    transformed_feature_names: list[str]
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze V2B incremental saving with XGBoost."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="v2b_incremental_saving_dataset.csv",
        help="Path to v2b_incremental_saving_dataset.csv. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="v2b_xgboost_outputs",
        help="Directory for all CSV, PNG, and Markdown outputs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for splits, models, and permutation tests.",
    )
    parser.add_argument(
        "--shap-sample",
        type=int,
        default=1000,
        help="Maximum number of test rows used for SHAP if shap is installed.",
    )
    return parser.parse_args()


def resolve_input_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path.resolve()

    script_dir_candidate = Path(__file__).resolve().parent / path_text
    if script_dir_candidate.exists():
        return script_dir_candidate.resolve()

    raise FileNotFoundError(
        f"Input CSV not found: {path_text}. Put v2b_incremental_saving_dataset.csv "
        "in the current directory or pass --input /path/to/file.csv."
    )


def ensure_output_dir(path_text: str) -> Path:
    output_dir = Path(path_text).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def configure_plots(output_dir: Path) -> None:
    mpl_dir = output_dir / "_matplotlib_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")


def validate_columns(df: pd.DataFrame) -> None:
    expected = set()
    for spec in FIELD_GROUPS.values():
        expected.update(spec["columns"])
    expected.update(SCENARIO_SWITCH_COLUMNS)
    expected.update(["incremental_saving_per_EV", "incremental_saving_per_battery"])

    missing = sorted(expected - set(df.columns))
    if missing:
        print("WARNING: Missing expected columns:", file=sys.stderr)
        for col in missing:
            print(f"  - {col}", file=sys.stderr)

    fatal = ["base_case_id", "scenario", "incremental_saving"]
    fatal_missing = [col for col in fatal if col not in df.columns]
    if fatal_missing:
        raise ValueError(
            "Required core columns are missing: " + ", ".join(fatal_missing)
        )


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows).copy()
    if df.empty:
        return "_No rows._"

    def cell_to_text(value: Any) -> str:
        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass
        if isinstance(value, (float, np.floating)):
            text = f"{float(value):.4f}".rstrip("0").rstrip(".")
        else:
            text = str(value)
        return text.replace("\n", " ").replace("|", "\\|")

    columns = [str(col).replace("|", "\\|") for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell_to_text(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def scenario_sort_key(name: str) -> int:
    if name in ALL_SCENARIOS:
        return ALL_SCENARIOS.index(name)
    return len(ALL_SCENARIOS)


def ordered_scenario_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        scenario_order=df["scenario"].map(lambda x: scenario_sort_key(str(x)))
    ).sort_values(["scenario_order", "scenario"]).drop(columns=["scenario_order"])


def generate_csv_overview(
    df: pd.DataFrame, input_path: Path, output_dir: Path
) -> tuple[pd.DataFrame, Path, Path]:
    scenario_summary = (
        df.groupby("scenario", dropna=False)["incremental_saving"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )
    scenario_summary = ordered_scenario_frame(scenario_summary)

    summary_path = output_dir / "01_scenario_summary.csv"
    save_csv(scenario_summary, summary_path)

    lines: list[str] = []
    lines.append("# 01 CSV Overview")
    lines.append("")
    lines.append(f"- Source file: `{input_path}`")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Columns: {df.shape[1]:,}")
    lines.append(f"- Unique `base_case_id`: {df['base_case_id'].nunique():,}")
    lines.append(
        f"- Scenario types: {df['scenario'].nunique():,} "
        + ", ".join(map(str, ordered_scenario_frame(df[["scenario"]].drop_duplicates())["scenario"]))
    )
    lines.append("")
    lines.append("## Scenario Counts")
    lines.append("")
    scenario_counts = (
        df["scenario"]
        .value_counts(dropna=False)
        .rename_axis("scenario")
        .reset_index(name="count")
    )
    scenario_counts = ordered_scenario_frame(scenario_counts)
    lines.append(df_to_markdown(scenario_counts))
    lines.append("")
    lines.append("## Incremental Saving Summary by Scenario")
    lines.append("")
    lines.append(df_to_markdown(scenario_summary.round(4)))
    lines.append("")
    lines.append("## Column Groups and Meanings")
    lines.append("")
    lines.append(
        "`incremental_saving = pv_only_baseline_cost - scenario_cost`，"
        "代表在建築已經有 PV 的 baseline 之上，特定調度策略額外創造的省錢金額。"
    )
    lines.append("")
    for group_name, spec in FIELD_GROUPS.items():
        cols = [col for col in spec["columns"] if col in df.columns]
        missing = [col for col in spec["columns"] if col not in df.columns]
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append(spec["description"])
        lines.append("")
        lines.append("- Columns: " + ", ".join(f"`{col}`" for col in cols))
        if missing:
            lines.append("- Missing in this CSV: " + ", ".join(f"`{col}`" for col in missing))
        lines.append("")

    overview_path = output_dir / "01_csv_overview.md"
    overview_path.write_text("\n".join(lines), encoding="utf-8")
    return scenario_summary, overview_path, summary_path


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_feature_frame(
    df: pd.DataFrame, scenario_specific: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    exclude = set(LEAKAGE_COLUMNS)
    if scenario_specific:
        exclude.update(SCENARIO_SWITCH_COLUMNS)
    feature_cols = [col for col in df.columns if col not in exclude]
    X = df[feature_cols].copy()

    if scenario_specific:
        nonconstant_cols = []
        for col in X.columns:
            if X[col].nunique(dropna=False) > 1:
                nonconstant_cols.append(col)
        X = X[nonconstant_cols].copy()
        feature_cols = nonconstant_cols

    return X, feature_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    try:
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    except TypeError:
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
        )


def make_xgb_pipeline(X: pd.DataFrame, random_state: int) -> Pipeline:
    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X)),
            ("model", model),
        ]
    )


def get_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names: list[str] = []
        for transformer_name, _, cols in preprocessor.transformers_:
            if transformer_name == "remainder":
                continue
            if isinstance(cols, slice):
                continue
            names.extend([str(col) for col in cols])
        return names


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series | None, pd.Series | None]:
    if groups is not None and groups.nunique(dropna=True) >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        return (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
            groups.iloc[train_idx],
            groups.iloc[test_idx],
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, None, None


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
    }


def cross_validation_metrics(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    random_state: int,
    n_splits: int = 5,
) -> dict[str, float]:
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }
    if groups is not None and groups.nunique(dropna=True) >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        try:
            cv_result = cross_validate(
                clone(pipeline),
                X,
                y,
                groups=groups,
                cv=cv,
                scoring=scoring,
                n_jobs=None,
                error_score="raise",
            )
        except TypeError:
            cv_result = cross_validate(
                clone(pipeline),
                X,
                y,
                params={"groups": groups},
                cv=cv,
                scoring=scoring,
                n_jobs=None,
                error_score="raise",
            )
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_result = cross_validate(
            clone(pipeline),
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=None,
            error_score="raise",
        )

    rmse_values = np.sqrt(np.maximum(-cv_result["test_mse"], 0))
    return {
        "cv_r2_mean": float(np.mean(cv_result["test_r2"])),
        "cv_r2_std": float(np.std(cv_result["test_r2"])),
        "cv_mae_mean": float(np.mean(-cv_result["test_mae"])),
        "cv_mae_std": float(np.std(-cv_result["test_mae"])),
        "cv_rmse_mean": float(np.mean(rmse_values)),
        "cv_rmse_std": float(np.std(rmse_values)),
    }


def fit_and_evaluate_model(
    name: str,
    df_model: pd.DataFrame,
    random_state: int,
    scenario_specific: bool,
) -> ModelResult:
    X, feature_cols = build_feature_frame(df_model, scenario_specific=scenario_specific)
    y = df_model["incremental_saving"].copy()
    groups = df_model["base_case_id"].copy() if "base_case_id" in df_model.columns else None

    X_train, X_test, y_train, y_test, groups_train, groups_test = split_train_test(
        X, y, groups, random_state
    )
    pipeline = make_xgb_pipeline(X_train, random_state=random_state)
    cv_metrics = cross_validation_metrics(
        pipeline, X, y, groups=groups, random_state=random_state
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_metrics = regression_metrics(y_test, y_pred)
    train_pred = pipeline.predict(X_train)
    train_metrics = regression_metrics(y_train, train_pred)
    transformed_feature_names = get_feature_names(pipeline)

    metrics = {
        "model": name,
        "n_total": int(len(df_model)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "raw_feature_count": int(len(feature_cols)),
        "transformed_feature_count": int(len(transformed_feature_names)),
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        **cv_metrics,
    }

    return ModelResult(
        name=name,
        model=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        groups_train=groups_train,
        groups_test=groups_test,
        feature_columns=feature_cols,
        transformed_feature_names=transformed_feature_names,
        metrics=metrics,
    )


def save_model_performance(results: list[ModelResult], output_dir: Path) -> pd.DataFrame:
    performance = pd.DataFrame([result.metrics for result in results])
    columns = [
        "model",
        "n_total",
        "n_train",
        "n_test",
        "raw_feature_count",
        "transformed_feature_count",
        "train_r2",
        "test_r2",
        "test_mae",
        "test_rmse",
        "cv_r2_mean",
        "cv_r2_std",
        "cv_mae_mean",
        "cv_mae_std",
        "cv_rmse_mean",
        "cv_rmse_std",
    ]
    performance = performance[columns]
    save_csv(performance, output_dir / "02_model_performance.csv")
    return performance


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(
    result: ModelResult, analysis_df: pd.DataFrame, output_dir: Path
) -> None:
    plot_df = pd.DataFrame(
        {
            "actual": result.y_test,
            "predicted": result.y_pred,
            "scenario": analysis_df.loc[result.y_test.index, "scenario"],
        }
    )
    plt.figure(figsize=(7.2, 5.4))
    for scenario in sorted(plot_df["scenario"].unique(), key=scenario_sort_key):
        sub = plot_df[plot_df["scenario"] == scenario]
        plt.scatter(sub["actual"], sub["predicted"], s=24, alpha=0.75, label=scenario)
    min_value = float(min(plot_df["actual"].min(), plot_df["predicted"].min()))
    max_value = float(max(plot_df["actual"].max(), plot_df["predicted"].max()))
    plt.plot([min_value, max_value], [min_value, max_value], "k--", linewidth=1)
    plt.xlabel("Actual incremental saving")
    plt.ylabel("Predicted incremental saving")
    plt.title("Actual vs Predicted Incremental Saving")
    plt.legend(loc="best")
    save_figure(output_dir / "fig_actual_vs_predicted.png")


def plot_residuals(
    result: ModelResult, analysis_df: pd.DataFrame, output_dir: Path
) -> None:
    residuals = result.y_test.to_numpy() - result.y_pred
    plot_df = pd.DataFrame(
        {
            "predicted": result.y_pred,
            "residual": residuals,
            "scenario": analysis_df.loc[result.y_test.index, "scenario"],
        }
    )
    plt.figure(figsize=(7.2, 5.4))
    for scenario in sorted(plot_df["scenario"].unique(), key=scenario_sort_key):
        sub = plot_df[plot_df["scenario"] == scenario]
        plt.scatter(sub["predicted"], sub["residual"], s=24, alpha=0.75, label=scenario)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted incremental saving")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residual Plot")
    plt.legend(loc="best")
    save_figure(output_dir / "fig_residual_plot.png")


def plot_scenario_average_and_boxplot(analysis_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    scenario_stats = (
        analysis_df.groupby("scenario")["incremental_saving"]
        .agg(mean="mean", median="median", count="count")
        .reset_index()
    )
    scenario_stats = ordered_scenario_frame(scenario_stats)

    plt.figure(figsize=(8, 5))
    plt.bar(
        scenario_stats["scenario"],
        scenario_stats["mean"],
        color=["#4C78A8", "#59A14F", "#F28E2B", "#E15759"][: len(scenario_stats)],
    )
    plt.xlabel("Scenario")
    plt.ylabel("Average incremental saving")
    plt.title("Average Incremental Saving by Scenario")
    plt.xticks(rotation=18, ha="right")
    save_figure(output_dir / "fig_scenario_average_saving.png")

    ordered = sorted(analysis_df["scenario"].unique(), key=scenario_sort_key)
    box_data = [
        analysis_df.loc[analysis_df["scenario"] == scenario, "incremental_saving"]
        for scenario in ordered
    ]
    plt.figure(figsize=(8.5, 5.2))
    try:
        plt.boxplot(box_data, tick_labels=ordered, patch_artist=True)
    except TypeError:
        plt.boxplot(box_data, labels=ordered, patch_artist=True)
    plt.xlabel("Scenario")
    plt.ylabel("Incremental saving")
    plt.title("Incremental Saving Distribution by Scenario")
    plt.xticks(rotation=18, ha="right")
    save_figure(output_dir / "fig_scenario_boxplot.png")

    return scenario_stats


def plot_barh(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    title: str,
    xlabel: str,
    path: Path,
    color: str = "#4C78A8",
) -> None:
    plot_df = df.sort_values(value_col, ascending=True).copy()
    plt.figure(figsize=(8, max(4.8, 0.28 * len(plot_df) + 1.5)))
    plt.barh(plot_df[label_col], plot_df[value_col], color=color)
    plt.xlabel(xlabel)
    plt.title(title)
    save_figure(path)


def native_feature_importance(result: ModelResult, output_dir: Path) -> pd.DataFrame:
    xgb_model = result.model.named_steps["model"]
    importances = np.asarray(xgb_model.feature_importances_, dtype=float)
    names = result.transformed_feature_names
    if len(importances) != len(names):
        names = [f"feature_{idx}" for idx in range(len(importances))]

    native_df = pd.DataFrame(
        {
            "feature": names,
            "importance": importances,
        }
    )
    native_df = native_df.sort_values("importance", ascending=False).reset_index(drop=True)
    native_df.insert(0, "rank", np.arange(1, len(native_df) + 1))
    save_csv(native_df, output_dir / "03_feature_importance_native.csv")
    plot_barh(
        native_df.head(20),
        value_col="importance",
        label_col="feature",
        title="XGBoost Native Feature Importance (Top 20)",
        xlabel="Native importance",
        path=output_dir / "fig_native_importance.png",
    )
    return native_df


def permutation_feature_importance(
    result: ModelResult, output_dir: Path, random_state: int
) -> pd.DataFrame:
    perm = permutation_importance(
        result.model,
        result.X_test,
        result.y_test,
        scoring="r2",
        n_repeats=20,
        random_state=random_state,
        # Keep this single-process so the script also runs in restricted
        # environments where joblib/loky cannot create semaphores.
        n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": result.feature_columns,
            "importance_mean_r2_drop": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean_r2_drop", ascending=False)
    perm_df = perm_df.reset_index(drop=True)
    perm_df.insert(0, "rank", np.arange(1, len(perm_df) + 1))
    save_csv(perm_df, output_dir / "04_feature_importance_permutation.csv")
    plot_barh(
        perm_df.head(20),
        value_col="importance_mean_r2_drop",
        label_col="feature",
        title="Permutation Importance on Test Set (Top 20)",
        xlabel="R2 drop after permutation",
        path=output_dir / "fig_permutation_importance.png",
        color="#59A14F",
    )
    return perm_df


def group_permutation_importance(
    result: ModelResult,
    groups: dict[str, list[str]],
    output_dir: Path,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    baseline_r2 = float(r2_score(result.y_test, result.model.predict(result.X_test)))
    rows: list[dict[str, Any]] = []

    for group_name, columns in groups.items():
        available = [col for col in columns if col in result.X_test.columns]
        if not available:
            rows.append(
                {
                    "group": group_name,
                    "n_features": 0,
                    "features": "",
                    "baseline_r2": baseline_r2,
                    "mean_r2_drop": np.nan,
                    "std_r2_drop": np.nan,
                    "min_r2_drop": np.nan,
                    "max_r2_drop": np.nan,
                }
            )
            continue

        drops = []
        for _ in range(30):
            X_perm = result.X_test.copy()
            permutation = rng.permutation(len(X_perm))
            for col in available:
                X_perm[col] = X_perm[col].to_numpy()[permutation]
            perm_r2 = float(r2_score(result.y_test, result.model.predict(X_perm)))
            drops.append(baseline_r2 - perm_r2)

        rows.append(
            {
                "group": group_name,
                "n_features": len(available),
                "features": ", ".join(available),
                "baseline_r2": baseline_r2,
                "mean_r2_drop": float(np.mean(drops)),
                "std_r2_drop": float(np.std(drops)),
                "min_r2_drop": float(np.min(drops)),
                "max_r2_drop": float(np.max(drops)),
            }
        )

    group_df = pd.DataFrame(rows).sort_values("mean_r2_drop", ascending=False)
    group_df = group_df.reset_index(drop=True)
    group_df.insert(0, "rank", np.arange(1, len(group_df) + 1))
    save_csv(group_df, output_dir / "05_group_permutation_importance.csv")
    plot_barh(
        group_df.dropna(subset=["mean_r2_drop"]),
        value_col="mean_r2_drop",
        label_col="group",
        title="Group Permutation Importance",
        xlabel="R2 drop after group permutation",
        path=output_dir / "fig_group_importance.png",
        color="#F28E2B",
    )
    return group_df


def shap_analysis(
    result: ModelResult,
    output_dir: Path,
    random_state: int,
    max_sample: int,
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        import shap  # type: ignore
    except ImportError:
        message = "shap is not installed; skipping TreeSHAP analysis."
        warnings.warn(message)
        shap_df = pd.DataFrame(
            [
                {
                    "rank": np.nan,
                    "feature": "SHAP skipped",
                    "mean_abs_shap": np.nan,
                    "direction": "",
                    "corr_feature_value_vs_shap": np.nan,
                    "note": message,
                }
            ]
        )
        save_csv(shap_df, output_dir / "06_shap_importance.csv")
        return shap_df, message

    sample_size = min(max_sample, len(result.X_test))
    X_sample = result.X_test.sample(n=sample_size, random_state=random_state)
    preprocessor = result.model.named_steps["preprocess"]
    xgb_model = result.model.named_steps["model"]
    X_transformed = preprocessor.transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed)
    feature_names = result.transformed_feature_names

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.title("SHAP Summary Plot")
    plt.savefig(output_dir / "fig_shap_summary.png", bbox_inches="tight", dpi=220)
    plt.close()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    rows = []
    for idx, feature in enumerate(feature_names):
        x_col = X_transformed[:, idx]
        shap_col = shap_values[:, idx]
        if np.std(x_col) > 0 and np.std(shap_col) > 0:
            corr = float(np.corrcoef(x_col, shap_col)[0, 1])
        else:
            corr = np.nan
        if pd.isna(corr):
            direction = "flat/unclear"
        elif corr > 0.05:
            direction = "higher feature values tend to increase predicted saving"
        elif corr < -0.05:
            direction = "higher feature values tend to decrease predicted saving"
        else:
            direction = "nonlinear or weak monotonic direction"
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(mean_abs[idx]),
                "direction": direction,
                "corr_feature_value_vs_shap": corr,
                "note": "",
            }
        )
    shap_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    shap_df = shap_df.reset_index(drop=True)
    shap_df.insert(0, "rank", np.arange(1, len(shap_df) + 1))
    save_csv(shap_df, output_dir / "06_shap_importance.csv")
    return shap_df, None


def binned_marginal_effects(
    analysis_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in BINNED_FEATURES:
        if feature not in analysis_df.columns:
            continue
        feature_df = analysis_df[[feature, "incremental_saving"]].dropna().copy()
        if feature_df[feature].nunique() < 2:
            continue
        q = min(5, int(feature_df[feature].nunique()))
        try:
            bins = pd.qcut(feature_df[feature], q=q, duplicates="drop")
        except ValueError:
            continue
        feature_df["bin"] = bins
        grouped = feature_df.groupby("bin", observed=True)
        for idx, (bin_label, sub) in enumerate(grouped, start=1):
            rows.append(
                {
                    "feature": feature,
                    "bin_number": idx,
                    "bin_label": str(bin_label),
                    "count": int(len(sub)),
                    "feature_min": float(sub[feature].min()),
                    "feature_max": float(sub[feature].max()),
                    "mean_incremental_saving": float(sub["incremental_saving"].mean()),
                    "median_incremental_saving": float(sub["incremental_saving"].median()),
                }
            )

    binned_df = pd.DataFrame(rows)
    save_csv(binned_df, output_dir / "07_binned_marginal_effects.csv")

    for feature, filename in BINNED_FIGURES.items():
        plot_feature = binned_df[binned_df["feature"] == feature].copy()
        if plot_feature.empty:
            continue
        plt.figure(figsize=(7.2, 4.8))
        plt.plot(
            plot_feature["bin_number"],
            plot_feature["mean_incremental_saving"],
            marker="o",
            linewidth=2,
            color="#4C78A8",
        )
        plt.xticks(plot_feature["bin_number"], plot_feature["bin_label"], rotation=28, ha="right")
        plt.xlabel(f"{feature} quantile bin")
        plt.ylabel("Average incremental saving")
        plt.title(f"Binned Marginal Effect: {feature}")
        save_figure(output_dir / filename)

    return binned_df


def first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def interaction_analysis(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    available_metrics = [col for col in INTERACTION_METRICS if col in df.columns]
    case_agg = {"scenario": "first"}
    for col in available_metrics:
        case_agg[col] = first_non_null
    case_df = df.groupby("base_case_id", as_index=False).agg(case_agg)

    rows: list[dict[str, Any]] = []
    for metric in available_metrics:
        values = case_df[metric].dropna()
        rows.append(
            {
                "analysis_type": "case_metric_summary",
                "variable": metric,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "positive_ratio": float((values > 0).mean()),
                "threshold": np.nan,
                "low_count": np.nan,
                "high_count": np.nan,
                "low_mean_s4_incremental_saving": np.nan,
                "high_mean_s4_incremental_saving": np.nan,
                "high_minus_low_mean": np.nan,
                "low_median_s4_incremental_saving": np.nan,
                "high_median_s4_incremental_saving": np.nan,
            }
        )

    s4_df = df[df["scenario"] == "S4_PV_battery_V2B"].copy()
    for label, feature in S4_CONDITION_FEATURES.items():
        if feature not in s4_df.columns:
            continue
        valid = s4_df[[feature, "incremental_saving"]].dropna().copy()
        if valid[feature].nunique() < 2:
            continue
        threshold = float(valid[feature].median())
        low = valid[valid[feature] <= threshold]
        high = valid[valid[feature] > threshold]
        rows.append(
            {
                "analysis_type": "S4_condition_high_low",
                "variable": label,
                "count": int(len(valid)),
                "mean": np.nan,
                "median": np.nan,
                "positive_ratio": np.nan,
                "threshold": threshold,
                "low_count": int(len(low)),
                "high_count": int(len(high)),
                "low_mean_s4_incremental_saving": float(low["incremental_saving"].mean()),
                "high_mean_s4_incremental_saving": float(high["incremental_saving"].mean()),
                "high_minus_low_mean": float(
                    high["incremental_saving"].mean() - low["incremental_saving"].mean()
                ),
                "low_median_s4_incremental_saving": float(low["incremental_saving"].median()),
                "high_median_s4_incremental_saving": float(high["incremental_saving"].median()),
            }
        )

    interaction_df = pd.DataFrame(rows)
    save_csv(interaction_df, output_dir / "08_interaction_analysis.csv")

    if "battery_v2b_interaction_saving" in case_df.columns:
        values = case_df["battery_v2b_interaction_saving"].dropna()
        plt.figure(figsize=(7.2, 4.8))
        plt.hist(values, bins=30, color="#4C78A8", alpha=0.85)
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Battery-V2B interaction saving")
        plt.ylabel("Number of base cases")
        plt.title("Distribution of Battery-V2B Interaction Saving")
        save_figure(output_dir / "fig_interaction_battery_v2b.png")

    if "S4_vs_best_single_saving" in case_df.columns:
        values = case_df["S4_vs_best_single_saving"].dropna()
        plt.figure(figsize=(7.2, 4.8))
        plt.hist(values, bins=30, color="#59A14F", alpha=0.85)
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("S4 vs best single strategy saving")
        plt.ylabel("Number of base cases")
        plt.title("Distribution of S4 Advantage over Best Single Strategy")
        save_figure(output_dir / "fig_s4_vs_best_single.png")

    return interaction_df


def top_feature_text(perm_df: pd.DataFrame, n: int = 10) -> str:
    if perm_df.empty:
        return "無可用 permutation importance 結果。"
    lines = []
    for _, row in perm_df.head(n).iterrows():
        lines.append(
            f"- `{row['feature']}`：R² drop = {format_number(row['importance_mean_r2_drop'], 4)}"
        )
    return "\n".join(lines)


def group_importance_text(group_df: pd.DataFrame) -> str:
    if group_df.empty:
        return "無可用 group permutation importance 結果。"
    lines = []
    for _, row in group_df.iterrows():
        lines.append(
            f"- {row['group']}：平均 R² drop = {format_number(row['mean_r2_drop'], 4)}"
        )
    return "\n".join(lines)


def get_metric_row(performance_df: pd.DataFrame, model_name: str) -> pd.Series | None:
    sub = performance_df[performance_df["model"] == model_name]
    if sub.empty:
        return None
    return sub.iloc[0]


def get_interaction_row(interaction_df: pd.DataFrame, variable: str) -> pd.Series | None:
    sub = interaction_df[
        (interaction_df["analysis_type"] == "case_metric_summary")
        & (interaction_df["variable"] == variable)
    ]
    if sub.empty:
        return None
    return sub.iloc[0]


def generate_report(
    df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    scenario_stats: pd.DataFrame,
    performance_df: pd.DataFrame,
    native_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    group_df: pd.DataFrame,
    binned_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    shap_df: pd.DataFrame | None,
    shap_message: str | None,
    output_dir: Path,
) -> Path:
    overall = get_metric_row(performance_df, "overall_intervention_S1_S4")
    s4_row = scenario_stats[scenario_stats["scenario"] == "S4_PV_battery_V2B"]
    best_scenario_row = scenario_stats.sort_values("mean", ascending=False).head(1)

    report: list[str] = []
    report.append("# V2B XGBoost Incremental Saving Analysis Report")
    report.append("")
    report.append("## 1. 研究目的")
    report.append("")
    report.append(
        "本研究不再分析「裝設 PV 是否省錢」，而是以 `S0_PV_only_unmanaged_EV` "
        "作為已有 PV 且 EV 到站即充的 baseline，進一步分析在已有 PV 的建築中，"
        "固定電池、智慧 EV 充電、V2B 與固定電池加 V2B 整合調度是否能額外提升 saving。"
    )
    report.append("")
    report.append("## 2. 資料說明")
    report.append("")
    report.append(
        f"本資料共有 {len(df):,} 筆、{df.shape[1]:,} 欄，包含 "
        f"{df['base_case_id'].nunique():,} 個 `base_case_id`。每個 base case 代表同一組外生條件，"
        "並在多個 scenario 下重複模擬。"
    )
    report.append("")
    report.append("Scenario 定義如下：")
    report.append("")
    report.append("- `S0_PV_only_unmanaged_EV`：已有 PV，沒有固定電池，EV 到站即充，不智慧調度、不放電。")
    report.append("- `S1_PV_battery`：已有 PV，加上固定電池調度。")
    report.append("- `S2_PV_smart_EV_charging`：已有 PV，EV 可智慧充電，但不可放電。")
    report.append("- `S3_PV_V2B`：已有 PV，EV 可智慧充電並可 V2B 放電。")
    report.append("- `S4_PV_battery_V2B`：已有 PV，固定電池與 EV V2B 整合調度。")
    report.append("")
    report.append(
        "目標變數為 `incremental_saving = pv_only_baseline_cost - scenario_cost`，"
        "代表相對於 PV-only baseline 的額外省錢金額。"
    )
    report.append("")
    report.append("### Scenario summary")
    report.append("")
    report.append(df_to_markdown(scenario_summary.round(4)))
    report.append("")
    report.append("主要欄位可分為基本情境、PV、電價、EV、固定電池、負載、成本目標、調度結果，以及 case-level 比較欄位。")
    report.append("其中成本/目標、調度結果與 case-level 比較欄位屬於事後結果或目標衍生資訊，因此不放入 XGBoost 特徵。")
    report.append("")
    report.append("## 3. 模型設定")
    report.append("")
    report.append(
        "模型使用 `XGBRegressor`，主要參數為 `n_estimators=500`、`max_depth=4`、"
        "`learning_rate=0.03`、`subsample=0.8`、`colsample_bytree=0.8`、"
        "`objective='reg:squarederror'`、`random_state=42`。"
    )
    report.append("")
    report.append(
        "建模時排除 `S0_PV_only_unmanaged_EV`，主要模型只使用 S1-S4。"
        "整體 intervention model 使用 `battery_enabled`、`smart_ev_charging_enabled`、"
        "`ev_v2b_enabled` 作為 scenario switch，而不是把 `scenario` 本身放入特徵。"
        "scenario-specific model 則分別針對 S1、S2、S3、S4 建模，並移除固定不變的 scenario switch。"
    )
    report.append("")
    report.append(
        "train/test split 採 0.2 test size；若 `base_case_id` 可用，則採 group-aware split，"
        "避免同一 base case 的不同 scenario 同時出現在 train 與 test。5-fold cross validation 也使用 "
        "`GroupKFold` 進行相同控制。"
    )
    report.append("")
    report.append("## 4. 模型表現")
    report.append("")
    report.append(df_to_markdown(performance_df.round(4)))
    report.append("")
    if overall is not None:
        report.append(
            f"整體 intervention model 的 test R² 為 {format_number(overall['test_r2'], 4)}，"
            f"MAE 為 {format_number(overall['test_mae'], 3)}，"
            f"RMSE 為 {format_number(overall['test_rmse'], 3)}；"
            f"5-fold CV 平均 R² 為 {format_number(overall['cv_r2_mean'], 4)}。"
        )
        report.append("")
        if float(overall["test_r2"]) >= 0.8:
            report.append("整體而言，模型對 `incremental_saving` 的可預測性高，代表資料中的設計條件與調度策略足以解釋大部分 saving 變異。")
        elif float(overall["test_r2"]) >= 0.5:
            report.append("整體而言，模型具備中等預測能力，但仍可能有部分調度細節或未觀測條件未被特徵捕捉。")
        else:
            report.append("整體而言，模型預測能力有限，應謹慎解讀特徵重要度，並檢查是否仍缺少關鍵情境變數。")
        report.append("")
    report.append("相關診斷圖已輸出：`fig_actual_vs_predicted.png`、`fig_residual_plot.png`。")
    report.append("")
    report.append("## 5. 整體特徵重要度")
    report.append("")
    report.append("Permutation importance 前 10 名如下，正式結論優先採用此方法：")
    report.append("")
    report.append(top_feature_text(perm_df, 10))
    report.append("")
    report.append("Group permutation importance 結果如下：")
    report.append("")
    report.append(group_importance_text(group_df))
    report.append("")
    report.append("解讀重點：")
    report.append("")
    for group_name, keywords, label in [
        ("PV group", ["pv_surplus_before_storage_kWh", "pv_surplus_ratio", "pv_total_kWh"], "PV surplus/PV 規模"),
        ("Price group", ["price_gap", "actual_price_spread", "price_gap_ratio"], "price gap/電價 spread"),
        ("EV group", ["parking_flexibility", "ev_peak_available_hours", "avg_parking_hours"], "EV availability/parking flexibility"),
        ("Battery group", ["battery_total_capacity_kWh", "battery_power_kW", "storage_to_pv_ratio"], "battery capacity/固定電池配置"),
        ("Scenario group", SCENARIO_SWITCH_COLUMNS, "scenario switch"),
    ]:
        group_sub = group_df[group_df["group"] == group_name]
        group_drop = float(group_sub["mean_r2_drop"].iloc[0]) if not group_sub.empty else np.nan
        top_hit = perm_df[perm_df["feature"].isin(keywords)].head(1)
        if not top_hit.empty:
            hit_text = f"，其中 `{top_hit.iloc[0]['feature']}` 在單一特徵 permutation 中也相對重要"
        else:
            hit_text = ""
        report.append(f"- {label}：group R² drop = {format_number(group_drop, 4)}{hit_text}。")
    report.append("")
    report.append("Native feature importance 與 permutation importance 圖已分別輸出為 `fig_native_importance.png` 與 `fig_permutation_importance.png`；group importance 圖已輸出為 `fig_group_importance.png`。")
    report.append("")
    report.append("## 6. 各 Scenario 結果")
    report.append("")
    report.append(df_to_markdown(scenario_stats.round(4)))
    report.append("")
    if not best_scenario_row.empty:
        best = best_scenario_row.iloc[0]
        report.append(
            f"平均 `incremental_saving` 最高的策略為 `{best['scenario']}`，"
            f"平均 saving 為 {format_number(best['mean'], 3)}。"
        )
        report.append("")
    report.append(
        "S1 代表固定電池單獨效益；S2 代表只有智慧 EV 充電的效益；"
        "S3 代表在智慧充電之外加入 V2B 的效益；S4 則代表固定電池與 V2B 整合調度。"
        "比較 S1-S4 的平均值與箱型圖，可以判斷單一策略與整合策略在不同情境下的 saving 分布。"
    )
    report.append("")
    report.append("相關圖已輸出：`fig_scenario_average_saving.png`、`fig_scenario_boxplot.png`。")
    report.append("")
    report.append("## 7. 交互作用分析")
    report.append("")
    for metric, interpretation in [
        ("battery_v2b_interaction_saving", "固定電池與 V2B 的互補/替代效果"),
        ("S4_vs_best_single_saving", "S4 相對於最佳單一策略的優勢"),
        ("v2b_extra_over_smart_charging", "V2B 相對於只做智慧充電的額外價值"),
    ]:
        row = get_interaction_row(interaction_df, metric)
        if row is None:
            continue
        report.append(
            f"- `{metric}`（{interpretation}）：平均值 {format_number(row['mean'], 3)}，"
            f"中位數 {format_number(row['median'], 3)}，正值比例 {format_number(row['positive_ratio'] * 100, 2)}%。"
        )
    report.append("")
    interaction_row = get_interaction_row(interaction_df, "battery_v2b_interaction_saving")
    if interaction_row is not None:
        if float(interaction_row["mean"]) > 0 and float(interaction_row["positive_ratio"]) > 0.5:
            report.append("就平均值與正值比例而言，固定電池與 V2B 呈現互補效果。")
        elif float(interaction_row["mean"]) < 0:
            report.append("就平均值而言，固定電池與 V2B 在本資料中更接近替代關係；兩者同時使用時可能競爭同一段 PV surplus 或價差套利空間。")
        else:
            report.append("固定電池與 V2B 的互補效果不強，需進一步依條件分群判讀。")
    report.append("")
    condition_rows = interaction_df[interaction_df["analysis_type"] == "S4_condition_high_low"]
    if not condition_rows.empty:
        report.append("S4 高低條件比較如下，`high_minus_low_mean` 為高條件平均 saving 減低條件平均 saving：")
        report.append("")
        condition_table = condition_rows[
            [
                "variable",
                "threshold",
                "low_mean_s4_incremental_saving",
                "high_mean_s4_incremental_saving",
                "high_minus_low_mean",
            ]
        ].round(4)
        report.append(df_to_markdown(condition_table))
        report.append("")
    report.append("相關圖已輸出：`fig_interaction_battery_v2b.png`、`fig_s4_vs_best_single.png`。")
    report.append("")
    report.append("## 8. 決策建議")
    report.append("")
    report.append("- 若建築已經有 PV，應優先確認中午是否有足夠 PV surplus；沒有可吸收或可移轉的餘電時，儲能與 V2B 的邊際價值會下降。")
    report.append("- 若 price gap 或 actual price spread 太小，固定電池與 V2B 的時間電價套利效益會受到限制。")
    report.append("- 若 EV 停留時間短、無法涵蓋尖峰電價時段，V2B 即使技術上可行，也不一定能創造顯著 saving。")
    report.append("- 固定電池適合作為穩定且可預期的儲能資源；EV 則更適合作為具停車行為約束的彈性儲能。")
    report.append("- S4 是否值得採用，不應只看它是否高於 baseline，而應檢查它是否顯著高於 S1 或 S3 這類單一策略。")
    report.append("")
    report.append("## 9. 限制與後續工作")
    report.append("")
    report.append("- 本資料為 Monte Carlo 情境模擬資料，不等同真實場域因果推論；模型學到的是模擬資料中的統計關係。")
    report.append("- 若未納入固定電池 degradation cost 或 EV battery degradation cost，`incremental_saving` 可能高估。")
    report.append("- 若未納入 demand charge，不能過度宣稱削峰效益；目前 saving 主要反映 energy charge 或調度成本差異。")
    report.append("- 後續可加入 PDP、ICE、真實停車資料、需量電費、電池退化成本與更多 tariff 結構，提升工程判讀力。")
    report.append("")
    report.append("## 10. 報告用結論段落")
    report.append("")
    if s4_row.empty:
        s4_text = "S4 整合策略的平均 saving 已於輸出表中提供"
    else:
        s4_text = f"S4 整合策略的平均 incremental saving 為 {format_number(s4_row.iloc[0]['mean'], 3)}"
    report.append(
        "在已有 PV 的建築情境下，本研究以 PV-only 且 EV 到站即充作為 baseline，"
        "並以 XGBoost 分析固定電池、智慧 EV 充電、V2B 與整合調度對額外 saving 的影響。"
        "結果顯示，額外 saving 並非單純由是否導入單一設備決定，而是受到 PV surplus、電價 spread、"
        "EV 停車可用性、固定電池容量配置與策略開關共同影響。"
        f"{s4_text}，但其工程價值應進一步與最佳單一策略比較；"
        "`S4_vs_best_single_saving` 與 `battery_v2b_interaction_saving` 的分布可用來判斷固定電池與 V2B "
        "是互補還是替代。整體而言，若場域具備充足 PV 餘電、明顯尖離峰價差，以及可涵蓋尖峰時段的 EV 停留條件，"
        "則 V2B 與固定電池整合調度較有機會產生額外經濟效益；反之，若上述條件不足，應優先採取較簡單且穩定的單一策略，"
        "並在納入電池退化成本與需量電費後再評估整合策略的投資合理性。"
    )
    report.append("")
    report.append("## Appendix: SHAP")
    report.append("")
    if shap_message:
        report.append(f"TreeSHAP 分析已跳過：{shap_message}")
    elif shap_df is not None and not shap_df.empty:
        report.append("SHAP 平均絕對值前 20 名如下：")
        report.append("")
        report.append(df_to_markdown(shap_df.head(20).round(5)))
    else:
        report.append("未產生 SHAP 結果。")
    report.append("")

    report_path = output_dir / "V2B_XGBoost_incremental_saving_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    output_dir = ensure_output_dir(args.output_dir)
    configure_plots(output_dir)

    print(f"Reading CSV: {input_path}")
    df = pd.read_csv(input_path)
    validate_columns(df)

    scenario_summary, overview_path, summary_path = generate_csv_overview(
        df, input_path, output_dir
    )

    analysis_df = df[df["scenario"].isin(ANALYSIS_SCENARIOS)].copy()
    if analysis_df.empty:
        raise ValueError("No S1-S4 records found after excluding S0 baseline.")

    print("Fitting overall intervention model (S1-S4)...")
    overall_result = fit_and_evaluate_model(
        "overall_intervention_S1_S4",
        analysis_df,
        random_state=args.random_state,
        scenario_specific=False,
    )

    results = [overall_result]
    for scenario in ANALYSIS_SCENARIOS:
        sub = analysis_df[analysis_df["scenario"] == scenario].copy()
        if sub.empty:
            print(f"WARNING: No rows for {scenario}; skipping scenario-specific model.")
            continue
        print(f"Fitting scenario-specific model: {scenario}")
        results.append(
            fit_and_evaluate_model(
                f"scenario_specific_{scenario}",
                sub,
                random_state=args.random_state,
                scenario_specific=True,
            )
        )

    performance_df = save_model_performance(results, output_dir)
    plot_actual_vs_predicted(overall_result, analysis_df, output_dir)
    plot_residuals(overall_result, analysis_df, output_dir)
    scenario_stats = plot_scenario_average_and_boxplot(analysis_df, output_dir)

    print("Computing feature importance...")
    native_df = native_feature_importance(overall_result, output_dir)
    perm_df = permutation_feature_importance(
        overall_result, output_dir, random_state=args.random_state
    )
    group_df = group_permutation_importance(
        overall_result,
        GROUP_IMPORTANCE_COLUMNS,
        output_dir,
        random_state=args.random_state,
    )
    shap_df, shap_message = shap_analysis(
        overall_result,
        output_dir,
        random_state=args.random_state,
        max_sample=args.shap_sample,
    )

    print("Computing binned marginal effects and interaction analysis...")
    binned_df = binned_marginal_effects(analysis_df, output_dir)
    interaction_df = interaction_analysis(df, output_dir)

    report_path = generate_report(
        df=df,
        analysis_df=analysis_df,
        scenario_summary=scenario_summary,
        scenario_stats=scenario_stats,
        performance_df=performance_df,
        native_df=native_df,
        perm_df=perm_df,
        group_df=group_df,
        binned_df=binned_df,
        interaction_df=interaction_df,
        shap_df=shap_df,
        shap_message=shap_message,
        output_dir=output_dir,
    )

    print("\nGenerated outputs:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != ".DS_Store":
            print(path)
    print(f"\nMain report: {report_path}")
    print(f"Overview: {overview_path}")
    print(f"Scenario summary: {summary_path}")


if __name__ == "__main__":
    main()
