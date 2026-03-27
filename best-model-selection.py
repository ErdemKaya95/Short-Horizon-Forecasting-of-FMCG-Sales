# -*- coding: utf-8 -*-
"""Best model selection (Feb-2026 target) across KNN, LR, Prophet, SARIMAX.

What it does
- Reads model output Excels:
  - KNN / LR: from ./ml-preds/*.xlsx (no accuracy column)
  - Prophet / SARIMAX: from ./output/*_ALL_YYYYMM.xlsx (already has accuracy + date)
- Computes row-level metrics for KNN/LR:
  - abs_error = |y_actual - y_pred|
  - mape in [0, 1] as abs_error / max(|y_actual|, 1), with special handling for y_actual==0
  - accuracy = 1 - mape, clipped to [0, 1]
- For each unq_key, computes average accuracy per model over training window:
  - 2025-09 .. 2026-01 (inclusive)
- Selects the best model per unq_key (highest avg accuracy; deterministic tie-break)
- Uses the selected model to pick Feb-2026 predictions (2026-02) and writes Excel outputs.

Outputs (under ./output/)
- model_accuracy_means_202509_202601.xlsx
- best_model_feb_202602.xlsx

Formatting
- Output Excels round numeric columns to 2 decimals for readability.

Notes on MAPE when y_actual == 0
- If y_actual==0 and y_pred==0 -> mape=0, accuracy=1
- If y_actual==0 and y_pred!=0 -> mape=1, accuracy=0

Usage
  python best_model_selection_feb.py

Optional env vars
- KRVTS_ML_PREDS_DIR: defaults to ./ml-preds
- KRVTS_TS_OUTPUT_DIR: defaults to ./output
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


try:
    import openpyxl  # noqa: F401
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'openpyxl'. Install with: pip install openpyxl\n"
        f"Original error: {e}"
    )


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
DEFAULT_ML_PREDS_DIR = os.getenv("KRVTS_ML_PREDS_DIR", os.path.join(_REPO_ROOT, "ml-preds"))
DEFAULT_TS_OUTPUT_DIR = os.getenv("KRVTS_TS_OUTPUT_DIR", os.path.join(_REPO_ROOT, "output"))

TRAIN_START = pd.Timestamp("2025-09-01")
TRAIN_END = pd.Timestamp("2026-01-01")
TARGET_MONTH = pd.Timestamp("2026-02-01")

EPSILON_ACTUAL_ZERO = 1e-6

MODEL_PRIORITY = ["prophet", "sarimax", "lr", "knn"]  # used only for tie-breaks


_MONTH_TOKEN_MAP = {
    "sept": 9,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    "jan": 1,
    "feb": 2,
}


def _ensure_datetime_month_start(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def parse_month_from_knn_lr_filename(filename: str) -> pd.Timestamp:
    """Parse month token like 'Sept25', 'Jan26' from filenames.

    Examples
    - knn_predictSept25_12Mar26.xlsx -> 2025-09-01
    - lr_predictFeb26_12Mar26.xlsx -> 2026-02-01
    """
    base = os.path.basename(filename)
    m = re.search(r"predict([A-Za-z]+)(\d{2})_", base)
    if not m:
        raise ValueError(f"Could not parse month token from filename: {base}")

    mon_str = m.group(1).strip().lower()
    year_2 = int(m.group(2))

    # Normalize Sept/September variants
    if mon_str.startswith("sept"):
        mon_key = "sept"
    else:
        mon_key = mon_str[:3]

    if mon_key not in _MONTH_TOKEN_MAP:
        raise ValueError(f"Unknown month token '{mon_str}' in filename: {base}")

    month = _MONTH_TOKEN_MAP[mon_key]
    year = 2000 + year_2
    return pd.Timestamp(year=year, month=month, day=1)


def add_mape_accuracy_columns(
    df: pd.DataFrame,
    y_actual_col: str = "y_actual",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Compute per-row abs_error, mape (0-1), accuracy (0-1).

    mape is defined as:
      - if y_actual == 0 and y_pred == 0 -> 0
      - if y_actual == 0 and y_pred != 0 -> 1
      - else abs(y_actual - y_pred) / max(abs(y_actual), 1)

    This keeps values bounded and matches the scale seen in Prophet/SARIMAX outputs.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    y_a = pd.to_numeric(out[y_actual_col], errors="coerce")
    y_p = pd.to_numeric(out[y_pred_col], errors="coerce")

    abs_err = (y_a - y_p).abs()

    # Handle y_actual == 0 separately
    is_zero = y_a.fillna(np.nan).eq(0)
    mape = pd.Series(np.nan, index=out.index, dtype="float64")

    # Non-zero actual: bounded denom
    denom = y_a.abs().where(y_a.abs() >= 1.0, 1.0)
    mape_nonzero = (abs_err / (denom + 1e-9)).clip(lower=0.0, upper=1.0)
    mape = mape.where(is_zero, mape_nonzero)

    # Zero actual
    both_zero = is_zero & abs_err.fillna(np.nan).eq(0)
    zero_actual_pred_nonzero = is_zero & ~both_zero
    mape = mape.mask(both_zero, 0.0)
    mape = mape.mask(zero_actual_pred_nonzero, 1.0)

    acc = (1.0 - mape).clip(lower=0.0, upper=1.0)

    out["abs_error"] = abs_err
    out["mape"] = mape.round(6)
    out["accuracy"] = acc.round(6)
    return out


def zero_epsilon_actuals(df: pd.DataFrame, col: str = "y_actual", eps: float = EPSILON_ACTUAL_ZERO) -> pd.DataFrame:
    """Replace tiny epsilon values in y_actual with 0.0 for stability/readability."""
    if df is None or df.empty or col not in df.columns:
        return df
    out = df.copy()
    y = pd.to_numeric(out[col], errors="coerce")
    out[col] = y.where(y.abs() >= eps, 0.0)
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _select_first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def load_knn_or_lr_excel(path: str, model: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path, engine="openpyxl")
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    df = _normalize_columns(df)

    unq_col = _select_first_existing_col(df, ["unq_key", "UNQ_KEY", "key"]) or "unq_key"
    y_pred_col = _select_first_existing_col(df, ["y_pred", "Y_PRED", "pred", "predict"]) or "y_pred"
    y_actual_col = _select_first_existing_col(df, ["y_actual", "Y_ACTUAL", "actual"]) or "y_actual"

    needed = [unq_col, y_pred_col, y_actual_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}. Columns={list(df.columns)}")

    out = df[needed].rename(columns={unq_col: "unq_key", y_pred_col: "y_pred", y_actual_col: "y_actual"}).copy()
    out["date"] = parse_month_from_knn_lr_filename(path)
    out["model"] = model

    out["unq_key"] = out["unq_key"].astype(str)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    out["y_actual"] = pd.to_numeric(out["y_actual"], errors="coerce")

    out = zero_epsilon_actuals(out, col="y_actual")

    out = add_mape_accuracy_columns(out)
    return out


def load_prophet_or_sarimax_excel(path: str, model: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path, engine="openpyxl")
    preferred = "final_predictions"
    sheet = preferred if preferred in [s.lower() for s in xls.sheet_names] else xls.sheet_names[0]
    if sheet != xls.sheet_names[0]:
        # Find actual-cased sheet name
        for s in xls.sheet_names:
            if s.lower() == preferred:
                sheet = s
                break
    df = pd.read_excel(xls, sheet_name=sheet)
    df = _normalize_columns(df)

    unq_col = _select_first_existing_col(df, ["unq_key", "UNQ_KEY", "key"]) or "unq_key"
    date_col = _select_first_existing_col(df, ["date", "ds", "Date"]) or "date"
    y_pred_col = _select_first_existing_col(df, ["y_pred", "yhat", "Y_PRED"]) or "y_pred"
    y_actual_col = _select_first_existing_col(df, ["y_actual", "actual", "y", "Y_ACTUAL"]) or "y_actual"

    needed = [unq_col, date_col, y_pred_col, y_actual_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}. Columns={list(df.columns)}")

    out = df[needed].rename(
        columns={unq_col: "unq_key", date_col: "date", y_pred_col: "y_pred", y_actual_col: "y_actual"}
    ).copy()

    out["date"] = _ensure_datetime_month_start(out["date"])
    out["model"] = model
    out["unq_key"] = out["unq_key"].astype(str)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    out["y_actual"] = pd.to_numeric(out["y_actual"], errors="coerce")

    out = zero_epsilon_actuals(out, col="y_actual")

    # Prefer provided accuracy/mape if present; otherwise compute.
    if "accuracy" in df.columns and "mape" in df.columns:
        out["mape"] = pd.to_numeric(df["mape"], errors="coerce").clip(lower=0.0, upper=1.0).round(6)
        out["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce").clip(lower=0.0, upper=1.0).round(6)
        if "abs_error" in df.columns:
            out["abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce")
        else:
            out = add_mape_accuracy_columns(out)
    else:
        out = add_mape_accuracy_columns(out)

    return out


def list_model_files(ml_preds_dir: str, ts_output_dir: str) -> Dict[str, List[str]]:
    def _glob_xlsx(folder: str) -> List[str]:
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".xlsx") and not f.startswith("~$")
        ]

    ml_files = _glob_xlsx(ml_preds_dir)
    ts_files = _glob_xlsx(ts_output_dir)

    out: Dict[str, List[str]] = {"knn": [], "lr": [], "prophet": [], "sarimax": []}

    for p in ml_files:
        b = os.path.basename(p).lower()
        if b.startswith("knn_"):
            out["knn"].append(p)
        elif b.startswith("lr_"):
            out["lr"].append(p)

    for p in ts_files:
        b = os.path.basename(p).lower()
        if b.startswith("prophet_all_"):
            out["prophet"].append(p)
        elif b.startswith("sarimax_all_"):
            out["sarimax"].append(p)

    for k in out:
        out[k] = sorted(out[k])

    return out


def consolidate_all(ml_preds_dir: str, ts_output_dir: str) -> pd.DataFrame:
    files = list_model_files(ml_preds_dir, ts_output_dir)

    frames: List[pd.DataFrame] = []

    for path in files["knn"]:
        frames.append(load_knn_or_lr_excel(path, model="knn"))
    for path in files["lr"]:
        frames.append(load_knn_or_lr_excel(path, model="lr"))
    for path in files["prophet"]:
        frames.append(load_prophet_or_sarimax_excel(path, model="prophet"))
    for path in files["sarimax"]:
        frames.append(load_prophet_or_sarimax_excel(path, model="sarimax"))

    if not frames:
        raise RuntimeError("No model files found.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["unq_key", "date"]).copy()
    all_df["date"] = _ensure_datetime_month_start(all_df["date"])

    all_df = zero_epsilon_actuals(all_df, col="y_actual")

    # Keep only needed columns in a consistent order
    cols = ["unq_key", "date", "model", "y_actual", "y_pred", "abs_error", "mape", "accuracy"]
    for c in cols:
        if c not in all_df.columns:
            all_df[c] = np.nan
    all_df = all_df[cols]

    return all_df


def compute_mean_accuracies(all_df: pd.DataFrame) -> pd.DataFrame:
    train_df = all_df[(all_df["date"] >= TRAIN_START) & (all_df["date"] <= TRAIN_END)].copy()
    means = (
        train_df.groupby(["unq_key", "model"], as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "avg_accuracy"})
    )
    means["avg_accuracy"] = means["avg_accuracy"].round(6)
    return means


def compute_monthly_accuracies(all_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly mean accuracy per (unq_key, model, date) in the train window."""
    train_df = all_df[(all_df["date"] >= TRAIN_START) & (all_df["date"] <= TRAIN_END)].copy()
    monthly = (
        train_df.groupby(["unq_key", "model", "date"], as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "monthly_accuracy"})
    )
    monthly["monthly_accuracy"] = monthly["monthly_accuracy"].round(6)
    return monthly


def _priority_rank(model: str) -> int:
    try:
        return MODEL_PRIORITY.index(model)
    except ValueError:
        return 999


def select_best_model_per_key(mean_df: pd.DataFrame) -> pd.DataFrame:
    df = mean_df.copy()
    df["priority"] = df["model"].map(_priority_rank)

    # Highest avg_accuracy wins; tie-breaker by priority list (lower is better)
    df = df.sort_values(["unq_key", "avg_accuracy", "priority"], ascending=[True, False, True])

    best = df.drop_duplicates(subset=["unq_key"], keep="first").rename(
        columns={"model": "best_model", "avg_accuracy": "best_avg_accuracy"}
    )
    return best[["unq_key", "best_model", "best_avg_accuracy"]]


def build_feb_output(all_df: pd.DataFrame, best_models: pd.DataFrame) -> pd.DataFrame:
    feb_df = all_df[all_df["date"] == TARGET_MONTH].copy()

    # In case a key is missing the selected model in Feb, fallback to next best available.
    # Precompute ranks of models per key, using avg_accuracy then priority
    mean_all = compute_mean_accuracies(all_df)
    mean_all["priority"] = mean_all["model"].map(_priority_rank)
    mean_all = mean_all.sort_values(["unq_key", "avg_accuracy", "priority"], ascending=[True, False, True])

    # For faster lookup, group lists
    model_rankings = (
        mean_all.groupby("unq_key")["model"].apply(list).to_dict()
    )

    out_rows: List[pd.Series] = []
    feb_by_key = {k: g for k, g in feb_df.groupby("unq_key")}

    for unq_key, ranking in model_rankings.items():
        g = feb_by_key.get(unq_key)
        if g is None or g.empty:
            continue

        available_models = set(g["model"].dropna().astype(str).tolist())
        chosen_model = None
        for m in ranking:
            if m in available_models:
                chosen_model = m
                break
        if chosen_model is None:
            continue

        row = g[g["model"] == chosen_model].sort_values("accuracy", ascending=False).head(1)
        if row.empty:
            continue

        s = row.iloc[0].copy()
        s["selected_model"] = chosen_model
        out_rows.append(s)

    if not out_rows:
        raise RuntimeError("Could not build Feb output; no matching rows found.")

    out_df = pd.DataFrame(out_rows)

    # Merge best model accuracy for reporting
    out_df = out_df.merge(best_models, on="unq_key", how="left")

    # Keep tidy columns
    keep = [
        "unq_key",
        "date",
        "selected_model",
        "best_model",
        "best_avg_accuracy",
        "y_actual",
        "y_pred",
        "abs_error",
        "mape",
        "accuracy",
    ]
    for c in keep:
        if c not in out_df.columns:
            out_df[c] = np.nan
    out_df = out_df[keep].sort_values(["unq_key"]).reset_index(drop=True)

    return out_df


def write_outputs(all_df: pd.DataFrame, mean_df: pd.DataFrame, best_df: pd.DataFrame, feb_df: pd.DataFrame) -> Tuple[str, str]:
    os.makedirs(DEFAULT_TS_OUTPUT_DIR, exist_ok=True)

    # 1) Mean accuracies workbook
    mean_path = os.path.join(DEFAULT_TS_OUTPUT_DIR, "model_accuracy_means_202509_202601.xlsx")
    mean_df_out = mean_df.copy()
    if "avg_accuracy" in mean_df_out.columns:
        mean_df_out["avg_accuracy"] = pd.to_numeric(mean_df_out["avg_accuracy"], errors="coerce").round(2)

    monthly_long = compute_monthly_accuracies(all_df)
    monthly_long_out = monthly_long.copy()
    monthly_long_out["monthly_accuracy"] = pd.to_numeric(
        monthly_long_out["monthly_accuracy"], errors="coerce"
    ).round(2)
    monthly_long_out["month"] = monthly_long_out["date"].dt.strftime("%Y-%m")

    monthly_wide = (
        monthly_long_out.pivot_table(
            index=["unq_key", "model"],
            columns="month",
            values="monthly_accuracy",
            aggfunc="mean",
        )
        .reset_index()
    )

    wide = mean_df_out.pivot(index="unq_key", columns="model", values="avg_accuracy").reset_index()
    model_cols = [c for c in MODEL_PRIORITY if c in wide.columns]
    if model_cols:
        wide["best_model"] = wide[model_cols].idxmax(axis=1)
    else:
        wide["best_model"] = np.nan

    best_df_out = best_df.copy()
    if "best_avg_accuracy" in best_df_out.columns:
        best_df_out["best_avg_accuracy"] = pd.to_numeric(best_df_out["best_avg_accuracy"], errors="coerce").round(2)

    with pd.ExcelWriter(mean_path, engine="openpyxl") as writer:
        mean_df_out.sort_values(["unq_key", "model"]).to_excel(writer, index=False, sheet_name="means_long")
        wide.to_excel(writer, index=False, sheet_name="means_wide")
        best_df_out.sort_values(["unq_key"]).to_excel(writer, index=False, sheet_name="best_model")
        monthly_long_out.sort_values(["unq_key", "model", "date"]).to_excel(
            writer, index=False, sheet_name="monthly_long"
        )
        monthly_wide.sort_values(["unq_key", "model"]).to_excel(
            writer, index=False, sheet_name="monthly_wide"
        )

    # 2) Feb selection workbook
    feb_path = os.path.join(DEFAULT_TS_OUTPUT_DIR, "best_model_feb_202602.xlsx")
    with pd.ExcelWriter(feb_path, engine="openpyxl") as writer:
        feb_df_out = feb_df.copy()
        feb_df_out = zero_epsilon_actuals(feb_df_out, col="y_actual")
        for c in ["best_avg_accuracy", "y_actual", "y_pred", "abs_error", "mape", "accuracy"]:
            if c in feb_df_out.columns:
                feb_df_out[c] = pd.to_numeric(feb_df_out[c], errors="coerce").round(2)
        feb_df_out.to_excel(writer, index=False, sheet_name="feb_selected")
        # Optional: store Feb rows of all models for audit
        all_feb = all_df[all_df["date"] == TARGET_MONTH].sort_values(["unq_key", "model"]).reset_index(drop=True)
        all_feb = zero_epsilon_actuals(all_feb, col="y_actual")
        for c in ["y_actual", "y_pred", "abs_error", "mape", "accuracy"]:
            if c in all_feb.columns:
                all_feb[c] = pd.to_numeric(all_feb[c], errors="coerce").round(2)
        all_feb.to_excel(writer, index=False, sheet_name="feb_all_models")

    return mean_path, feb_path


def main() -> None:
    ml_preds_dir = DEFAULT_ML_PREDS_DIR
    ts_output_dir = DEFAULT_TS_OUTPUT_DIR

    all_df = consolidate_all(ml_preds_dir=ml_preds_dir, ts_output_dir=ts_output_dir)

    mean_df = compute_mean_accuracies(all_df)
    best_df = select_best_model_per_key(mean_df)

    feb_selected = build_feb_output(all_df, best_df)

    mean_path, feb_path = write_outputs(all_df, mean_df, best_df, feb_selected)

    # Minimal console summary
    print("Rows consolidated:", len(all_df))
    print("Unique unq_key:", all_df["unq_key"].nunique())
    print("Mean window:", str(TRAIN_START.date()), "..", str(TRAIN_END.date()))
    print("Target month:", str(TARGET_MONTH.date()))
    print("Wrote:", mean_path)
    print("Wrote:", feb_path)


if __name__ == "__main__":
    main()
