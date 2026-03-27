# -*- coding: utf-8 -*-
"""SARIMAX(exog) monthly forecasting runner.

For each `unq_key`, this script fits a SARIMAX model with exogenous regressors and
produces a 1-step-ahead forecast for each target month in the configured range.

Workflow per target month:
- Uses (target - 2 months) as validation (`val`) to select the best SARIMAX spec
- Uses (target - 1 month) as test (`test`) to report rolling performance
- Forecasts the target month (`final_predictions`)

Extras:
- `LIMIT_KEYS`: limit the number of processed `unq_key` values
- `LIMIT_STRATEGY`: ('head' | 'random' | 'top_volume_recent')
- `DISPLAY_HEAD_N`: number of example rows printed to console

Excel outputs (per month):
- `val_predictions`
- `test_predictions`
- `final_predictions`

Notes:
- This file is a GitHub-ready copy of the original script. Only comments,
  docstrings, and user-facing strings were translated/cleaned. Modeling logic is
  unchanged.
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from joblib import Parallel, delayed
from datetime import datetime


# === Paths ===
# Defaults point to files within this repository/workspace.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Generic filenames/variables for GitHub.
FILE_PATH = os.getenv("MODEL_INPUT_PATH", os.path.join(_HERE, "model_input.csv"))

EXPORT_DIR = os.getenv("OUTPUT_DIR", os.path.join(_HERE, "output"))
EXPORT_PATH = os.path.join(EXPORT_DIR, "sarimax.xlsx")  # overridden per target month

# === User parameters (limit/preview) ===
LIMIT_KEYS = None              # None or an integer (e.g., 200)
LIMIT_STRATEGY = "random"  # 'head' | 'random' | 'top_volume_recent'
RECENT_WINDOW_MONTHS = 36      # last X months for 'top_volume_recent'
RANDOM_SEED = 42               # for 'random'
DISPLAY_HEAD_N = 10            # number of rows to print in console
SARIMAX_FLAG = "TEST"    # "TEST" → last 1 month validation, "BEST_MODEL" → last 2 months validation
SARIMAX_VAL_MIN = 1            # minimum validation months (fallback)
PRED_MONTH = 2                 # target forecast month (2 = February)
EXOG_FILL_METHOD = "ffill"     # "ffill" (recommended, no leakage) | "ffill_bfill" (leaky)

# SARIMAX tuning (small grid; can be expanded for paper experiments)
# Note: trend='c' (intercept) is usually more stable across many series.
SARIMAX_MAXITER = 50
SARIMAX_TRIED_SPECS = [
    ((0, 0, 0), (0, 0, 0, 0), "c"),
    ((1, 0, 0), (0, 0, 0, 0), "c"),
    ((0, 0, 1), (0, 0, 0, 0), "c"),
    ((1, 0, 1), (0, 0, 0, 0), "c"),
    ((0, 0, 0), (0, 1, 1, 12), "c"),
    ((1, 0, 1), (0, 1, 1, 12), "c"),
]

# === Warning filters ===
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="statsmodels")


# === Helpers ===
def to_numeric_safely(s):
    """Cleans Turkish decimal comma / spaces and casts to float; returns NaN on failure."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return s
    if isinstance(s, str):
        z = s.strip().replace("\u00A0", "").replace(" ", "")
        is_pct = False
        if z.endswith('%'):
            is_pct = True
            z = z[:-1]
        z = z.replace(",", ".")
        try:
            val = float(z)
            return val / 100.0 if is_pct else val
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan


def select_exog_columns(df):
    """Selects numeric columns to be used as exogenous regressors.

    Excludes target/ID columns and columns with obvious leakage risk.
    """
    exclude_cols = {
        "alt_kategori", "urun_adi", "urun_kodu", "unq_key", "satis_hacmi",
        "musteri_id", "sap_kodu", "mobis_kodu", "mobis_kodu2",
        "bolge", "date", "actual"
    }
    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols and not c.startswith("Unnamed") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols


# === Prediction normalization ===
def normalize_predict(y_true, y_pred):
    """Normalize predictions.

    - If 0 <= pred < 1 → round down to 0
    - If pred >= 1 → round to nearest integer with 0.5 threshold
    """
    if pd.isna(y_pred):
        return np.nan
    if y_pred < 0:
        return 0
    if 0 <= y_pred < 1:
        return 0
    frac, base = np.modf(y_pred)
    base = int(base)
    return base + 1 if frac >= 0.5 else base


# === Confusion-matrix style classification ===
def classify_confusion(row):
    if row["actual"] == 0:
        if 0 <= row["predict"] < 1:
            return "TP"
        else:
            return "FP"
    else:
        if row["predict"] < 1:
            return "FN"
        else:
            return "TN"


def compute_metrics(y_true, y_pred):
    """Returns MAE, RMSE, MAPE(%).

    - Filters valid (finite) pairs because sklearn doesn't accept NaN/inf.
    - Excludes y_true == 0 observations for MAPE.
    """
    y_true = pd.Series(y_true, dtype="float64")
    y_pred = pd.Series(y_pred, dtype="float64")

    # Valid pairs: non-NaN and finite on both sides
    valid = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid].to_numpy()
    y_pred = y_pred[valid].to_numpy()

    if y_true.size == 0:
        return np.nan, np.nan, np.nan

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    nonzero = y_true != 0
    mape = (np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
            if np.any(nonzero) else np.nan)

    return mae, rmse, mape


def add_mape_accuracy_columns(df, y_actual_col="y_actual", y_pred_col="y_pred"):
    """Adds row-level `mape` and `accuracy` columns.

    - `mape`: 0–1 (clamped)
    - `accuracy`: 0–1 (clamped), accuracy = 1 - mape

    Notes:
    - Some inputs may have epsilon-level actuals (e.g., 1e-8); treat those as 0.
    - To avoid division by zero, denominator is floored at 1.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    y_a = pd.to_numeric(out.get(y_actual_col), errors="coerce")
    y_p = pd.to_numeric(out.get(y_pred_col), errors="coerce")

    eps = 1e-6
    y_a_clean = y_a.where(y_a.abs() >= eps, 0.0)
    abs_err = (y_a_clean - y_p).abs()

    denom = y_a_clean.abs().where(y_a_clean.abs() >= 1.0, 1.0)
    mape = (abs_err / (denom + 1e-9)).clip(lower=0.0, upper=1.0)
    acc = (1.0 - mape).clip(lower=0.0, upper=1.0)

    out["abs_error"] = abs_err
    out["mape"] = mape
    out["accuracy"] = acc

    # Reduce scientific notation in Excel
    out["mape"] = out["mape"].round(6)
    out["accuracy"] = out["accuracy"].round(6)
    return out


def build_history_columns(df, targets_df, window=12):
    """Builds lagged actual history columns for each target row.

    `targets_df` must contain at least ['unq_key','target_date'] (from pred_df).
    Returns the last `window` months of actuals BEFORE the target month.

    Column names: `gecmis_satis_01` .. `gecmis_satis_{window}`
    - 01 = oldest, window = most recent (t-1)
    """
    targets_df = targets_df.copy()
    targets_df["target_date"] = pd.to_datetime(targets_df["target_date"])

    cols = [f"gecmis_satis_{i:02d}" for i in range(1, window+1)]
    out = pd.DataFrame(index=targets_df.index, columns=cols, dtype=float)

    df = df.sort_values(["unq_key", "date"])
    for idx, row in targets_df.iterrows():
        key = row["unq_key"]
        t = row["target_date"]
        past = (
            df[(df["unq_key"] == key) & (df["date"] < t)]
              .sort_values("date")["actual"]
              .tail(window)
              .tolist()
        )
        # Pad on the left with NaN (fixed number of columns)
        if len(past) < window:
            past = [np.nan] * (window - len(past)) + past
        out.loc[idx, cols] = past
    return out


def count_nonzero_last_k_months(df, targets_df, k=12):
    """Counts non-zero sales in the last k months before each target (exclusive)."""
    targets_df = targets_df.copy()
    targets_df["target_date"] = pd.to_datetime(targets_df["target_date"])
    counts = pd.Series(index=targets_df.index, dtype=float)

    df = df.sort_values(["unq_key", "date"])
    for idx, row in targets_df.iterrows():
        key = row["unq_key"]
        t = row["target_date"]
        start = t - pd.DateOffset(months=k)
        sub = df[(df["unq_key"] == key) & (df["date"] < t) & (df["date"] >= start)]
        counts.loc[idx] = (sub["actual"] > 0).sum()
    return counts.astype(int)


def tune_and_forecast_one_step(y_train, exog_train, exog_forecast_row, flag="BEST_MODEL"):
    """Selects a SARIMAX spec using the last 1–2 months as validation, then forecasts 1 step.

    flag:
        TEST → last 1 month validation
        BEST_MODEL → last 2 months validation (if possible)
    """
    # Candidate specifications (light set; can be expanded for paper experiments)
    tried_specs = SARIMAX_TRIED_SPECS if isinstance(SARIMAX_TRIED_SPECS, list) and len(SARIMAX_TRIED_SPECS) > 0 else [
        ((0, 0, 0), (0, 0, 0, 0), "c"),
    ]

    # Frequency settings
    y_train = y_train.asfreq('MS')
    exog_train = exog_train.asfreq('MS')

    # Validation window length
    val_points = 1 if flag == "TEST" else 2
    if len(y_train) < (val_points + 4):  # very short series → fallback (at least 4 + val)
        return float(y_train.iloc[-1]), {"success": False, "fallback": "short_series", "order": None, "seasonal_order": None}
    if val_points > len(y_train) - 3:
        val_points = SARIMAX_VAL_MIN

    core_y = y_train.iloc[:-val_points]
    val_y = y_train.iloc[-val_points:]
    core_exog = exog_train.iloc[:-val_points] if exog_train.shape[1] > 0 else exog_train
    val_exog = exog_train.iloc[-val_points:] if exog_train.shape[1] > 0 else exog_train

    results = []
    for order, seasonal_order, trend in tried_specs:
        try:
            model = sm.tsa.statespace.SARIMAX(
                endog=core_y,
                exog=core_exog if core_exog.shape[1] > 0 else None,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False, maxiter=SARIMAX_MAXITER)
            preds_val = res.forecast(steps=val_points, exog=val_exog if core_exog.shape[1] > 0 else None)
            # Compute MAPE
            actual_v = val_y.values
            pred_v = preds_val.values
            mask = actual_v != 0
            if mask.any():
                mape_v = np.mean(np.abs((actual_v[mask] - pred_v[mask]) / actual_v[mask]))
            else:
                mape_v = np.nan
            results.append({
                'order': order,
                'seasonal_order': seasonal_order,
                'trend': trend,
                'mape_val': mape_v,
                'val_date': val_y.index[-1] if len(val_y.index) > 0 else None,
                'val_pred': float(preds_val.iloc[-1]) if len(preds_val) > 0 else np.nan,
                'val_true': float(val_y.iloc[-1]) if len(val_y) > 0 else np.nan,
                'success': True
            })
        except Exception as e:
            results.append({
                'order': order,
                'seasonal_order': seasonal_order,
                'trend': trend,
                'mape_val': np.inf,
                'val_date': val_y.index[-1] if len(val_y.index) > 0 else None,
                'val_pred': np.nan,
                'val_true': float(val_y.iloc[-1]) if len(val_y) > 0 else np.nan,
                'success': False,
                'error': str(e)
            })

    # Select best specification
    valid_results = [r for r in results if r['success']]
    if not valid_results:
        # No successful fit → naive
        return float(y_train.iloc[-1]), {"success": False, "fallback": "no_valid_spec", "order": None, "seasonal_order": None}

    def _mape_key(r):
        v = r.get('mape_val')
        return np.inf if (v is None or pd.isna(v)) else float(v)

    best = min(valid_results, key=_mape_key)

    # Fit on full training data and forecast target
    try:
        final_model = sm.tsa.statespace.SARIMAX(
            endog=y_train,
            exog=exog_train if exog_train.shape[1] > 0 else None,
            order=best['order'],
            seasonal_order=best['seasonal_order'],
            trend=best.get('trend'),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        final_res = final_model.fit(disp=False, maxiter=SARIMAX_MAXITER)
        final_pred = final_res.forecast(steps=1, exog=exog_forecast_row if exog_train.shape[1] > 0 else None)
        yhat = float(final_pred.iloc[0])
        if np.isfinite(yhat):
            yhat = max(0.0, yhat)
        return yhat, {
            'order': best['order'],
            'seasonal_order': best['seasonal_order'],
            'trend': best.get('trend'),
            'success': True,
            'val_mape': best['mape_val'],
            'val_date': best.get('val_date'),
            'val_pred': best.get('val_pred'),
            'val_true': best.get('val_true'),
            'val_points': val_points,
            'mode_flag': flag
        }
    except Exception as e:
        return float(y_train.iloc[-1]), {"success": False, "fallback": "final_fit_fail", "error": str(e), 'order': best['order'], 'seasonal_order': best['seasonal_order'], 'trend': best.get('trend')}


def limit_keys(df, all_keys, target_aug_date, limit_n=None, strategy="head",
               recent_window_months=12, seed=42):
    """Limits `unq_key` list using a selection strategy.

    Strategies:
    - head: first N keys
    - random: random N keys
    - top_volume_recent: top N keys by sum of actuals in the last `recent_window_months`

    The target month is not a hard requirement anymore — short series can be included.
    """

    if (limit_n is None) or (limit_n <= 0) or (limit_n >= len(all_keys)):
        return all_keys  # no limit

    # Include all keys; target-month condition removed
    keys = all_keys

    if len(keys) <= limit_n:
        return keys

    if strategy == "head":
        return keys[:limit_n]

    elif strategy == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(keys), size=limit_n, replace=False)
        return [keys[i] for i in idx]

    elif strategy == "top_volume_recent":
        # Date range for last `recent_window_months`
        start_date = target_aug_date - pd.DateOffset(months=recent_window_months)
        recent = df[(df['date'] > start_date) & (df['date'] <= target_aug_date)]
        tmp = recent[['unq_key', 'actual']].copy()
        tmp['actual'] = pd.to_numeric(tmp['actual'], errors='coerce')
        agg = tmp.groupby('unq_key', as_index=False)['actual'].sum().rename(columns={'actual': 'sum_recent_actual'})
        agg = agg[agg['unq_key'].isin(keys)].sort_values('sum_recent_actual', ascending=False)
        return agg['unq_key'].head(limit_n).tolist()

    else:
        return keys[:limit_n]


# === Read and preprocess data ===
df = pd.read_csv(FILE_PATH)

for required_col in ['date', 'actual', 'unq_key']:
    if required_col not in df.columns:
        raise ValueError(f"Missing required column '{required_col}' in input data.")

# Date

df['date'] = pd.to_datetime(df['date'])

# Convert string-like numeric columns
for col in df.columns:
    if col == 'date':
        continue
    if df[col].dtype == 'object':
        tmp = df[col].map(to_numeric_safely)
        if tmp.isna().mean() < 0.8:
            df[col] = tmp

# Sort

df = df.sort_values(['unq_key', 'date']).reset_index(drop=True)

# === Target months (run month by month) ===
# Desired range: from 2025-09 to 2026-02 (inclusive) by default.
TARGET_START_DATE = os.getenv("TARGET_START", "2025-09-01")
TARGET_END_DATE = os.getenv("TARGET_END", "2026-02-01")

try:
    _t_start = pd.to_datetime(TARGET_START_DATE).to_period("M").to_timestamp()
    _t_end = pd.to_datetime(TARGET_END_DATE).to_period("M").to_timestamp()
except Exception as e:
    raise ValueError(f"Failed to parse TARGET_START/END: start={TARGET_START_DATE}, end={TARGET_END_DATE}. Error: {e}")

if _t_start > _t_end:
    _t_start, _t_end = _t_end, _t_start

target_months = pd.period_range(_t_start.to_period("M"), _t_end.to_period("M"), freq="M").to_timestamp()

# Exogenous columns (once)
exog_cols = select_exog_columns(df)
print(f"Selected exogenous columns ({len(exog_cols)}): {exog_cols[:15]}")
if len(exog_cols) == 0:
    print("No numeric exogenous columns found. Feature importance may be empty.")

# Fill missing exog (group-wise ffill/bfill → remaining NaNs to 0)
if len(exog_cols) > 0:
    if EXOG_FILL_METHOD == "ffill_bfill":
        df[exog_cols] = df.groupby('unq_key')[exog_cols].ffill().bfill()
    else:
        df[exog_cols] = df.groupby('unq_key')[exog_cols].ffill()
    df[exog_cols] = df[exog_cols].fillna(0)

# Cast target to numeric

df['actual'] = pd.to_numeric(df['actual'], errors='coerce')
neg_count = (df['actual'] < 0).sum()
if neg_count > 0:
    print(f"Dropping {neg_count} negative actual values.")
    df = df[df['actual'] >= 0].copy()

os.makedirs(EXPORT_DIR, exist_ok=True)
all_keys = df['unq_key'].dropna().unique().tolist()

# Avoid repeated df[df['unq_key']==key] scanning (speed)
_df_by_key = df.groupby('unq_key', sort=False)

# Available month set after negative filtering
available_months = set(df['date'].dt.to_period('M').dt.to_timestamp())

for target_aug_date in target_months:
    if target_aug_date not in available_months:
        print(f"Target month not found in data, skipping: {target_aug_date.date()}")
        continue

    _yyyymm = target_aug_date.strftime("%Y%m")
    EXPORT_PATH = os.path.join(EXPORT_DIR, f"sarimax_ALL_{_yyyymm}.xlsx")

    val_date = (target_aug_date - pd.DateOffset(months=2)).to_period('M').to_timestamp()
    test_date = (target_aug_date - pd.DateOffset(months=1)).to_period('M').to_timestamp()

    print(f"\nRunning month: {target_aug_date.date()} | test: {test_date.date()} | val: {val_date.date()} | export: {os.path.basename(EXPORT_PATH)}")

    proc_keys = limit_keys(
        df, all_keys, target_aug_date,
        limit_n=LIMIT_KEYS,
        strategy=LIMIT_STRATEGY,
        recent_window_months=RECENT_WINDOW_MONTHS,
        seed=RANDOM_SEED
    )
    print(f"Total keys: {len(all_keys)} | Processing keys: {len(proc_keys)} | Strategy: {LIMIT_STRATEGY}")

    # Output containers and running stats (reset per month)
    val_rows = []
    test_rows = []
    forecast_rows = []
    running_true, running_pred = [], []
    n_done = n_success = n_naive = 0

    # === Main loop (tqdm + dynamic postfix) ===
    pbar = tqdm(proc_keys, desc=f"SARIMAX {target_aug_date.strftime('%Y%m')}", ncols=120)
    for key in pbar:
        try:
            sub = _df_by_key.get_group(key).copy().sort_values('date')
        except KeyError:
            continue

        # Required rows: val/test/target
        if (val_date not in set(sub['date'])) or (test_date not in set(sub['date'])) or (target_aug_date not in set(sub['date'])):
            continue

        val_row = sub[sub['date'] == val_date].copy()
        test_row = sub[sub['date'] == test_date].copy()
        target_row = sub[sub['date'] == target_aug_date].copy()

        # Tuning train: include validation month (last point = val)
        train_for_tune = sub[sub['date'] <= val_date].copy()
        train_for_final = sub[sub['date'] < target_aug_date].copy()  # exclude target (include test)

        if train_for_tune.empty or train_for_final.empty:
            continue

        train_for_tune = train_for_tune.set_index('date')
        train_for_final = train_for_final.set_index('date')
        val_row = val_row.set_index('date')
        test_row = test_row.set_index('date')
        target_row = target_row.set_index('date')

        # --- TEST forecast (t-1): pick spec using val (t-2), forecast test ---
        y_train_tune = train_for_tune['actual'].dropna()
        if y_train_tune.empty:
            continue
        X_train_tune = train_for_tune[exog_cols] if len(exog_cols) > 0 else pd.DataFrame(index=y_train_tune.index)
        X_test = test_row[exog_cols] if len(exog_cols) > 0 else pd.DataFrame(index=test_row.index)
        if isinstance(X_test, pd.Series):
            X_test = X_test.to_frame().T

        try:
            yhat_test, info = tune_and_forecast_one_step(y_train_tune, X_train_tune, X_test, flag="TEST")
            y_true_test = float(test_row['actual'].iloc[0]) if not np.isnan(test_row['actual'].iloc[0]) else np.nan
        except Exception as e:
            yhat_test = float(y_train_tune.iloc[-1])
            y_true_test = float(test_row['actual'].iloc[0]) if not np.isnan(test_row['actual'].iloc[0]) else np.nan
            info = {'success': False, 'fallback': 'naive-except', 'error': str(e)}

        if not np.isfinite(yhat_test):
            yhat_test = float(y_train_tune.iloc[-1])
            if isinstance(info, dict):
                info['fallback'] = (info.get('fallback') or 'nonfinite_forecast')
                info['success'] = False

        # --- VALIDATION output (val_date) ---
        y_true_val = float(val_row['actual'].iloc[0]) if ('actual' in val_row.columns and pd.notna(val_row['actual'].iloc[0])) else np.nan
        y_pred_val = info.get('val_pred') if isinstance(info, dict) else np.nan
        if (y_pred_val is None) or (not np.isfinite(y_pred_val)):
            core_for_val = train_for_tune.loc[train_for_tune.index < val_date, 'actual'].dropna()
            if len(core_for_val) > 0:
                y_pred_val = float(core_for_val.iloc[-1])
            else:
                y_pred_val = y_true_val
        val_rows.append({
            'unq_key': key,
            'target_date': val_date.date(),
            'y_true': y_true_val,
            'y_pred': float(y_pred_val) if y_pred_val is not None else np.nan,
            'model_success': info.get('success') if isinstance(info, dict) else None,
            'val_mape': info.get('val_mape') if isinstance(info, dict) else None,
            'best_spec': str({
                'order': info.get('order') if isinstance(info, dict) else None,
                'seasonal_order': info.get('seasonal_order') if isinstance(info, dict) else None,
                'trend': info.get('trend') if isinstance(info, dict) else None,
            }),
        })

        test_rows.append({
            'unq_key': key,
            'target_date': test_date.date(),
            'y_true': y_true_test,
            'y_pred': yhat_test,
            'abs_error': (abs(y_true_test - yhat_test) if pd.notna(y_true_test) and pd.notna(yhat_test) else np.nan),
            'model_success': info.get('success'),
            'order': info.get('order'),
            'seasonal_order': info.get('seasonal_order'),
            'trend': info.get('trend'),
            'val_points': info.get('val_points'),
            'val_mape': info.get('val_mape'),
        })

        # --- TARGET forecast: forecast target month with selected spec ---
        y_train_final = train_for_final['actual'].dropna()
        if y_train_final.empty:
            continue
        X_train_final = train_for_final[exog_cols] if len(exog_cols) > 0 else pd.DataFrame(index=y_train_final.index)
        X_target = target_row[exog_cols] if len(exog_cols) > 0 else pd.DataFrame(index=target_row.index)
        if isinstance(X_target, pd.Series):
            X_target = X_target.to_frame().T

        def _forecast_with_spec(y_series, X_df, X_row, spec_info):
            if not isinstance(spec_info, dict) or not spec_info.get('order'):
                return float(y_series.iloc[-1]), {"success": False, "fallback": "no_spec"}
            try:
                exog_full = None
                exog_row = None
                if X_df is not None and X_df.shape[1] > 0:
                    exog_full = X_df.asfreq('MS').fillna(0)
                    exog_row = X_row.copy().fillna(0)
                m = sm.tsa.statespace.SARIMAX(
                    endog=y_series.asfreq('MS'),
                    exog=exog_full,
                    order=tuple(spec_info.get('order')),
                    seasonal_order=tuple(spec_info.get('seasonal_order')),
                    trend=spec_info.get('trend'),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                r = m.fit(disp=False, maxiter=SARIMAX_MAXITER)
                pred = r.forecast(steps=1, exog=exog_row)
                return float(pred.iloc[0]), {"success": True}
            except Exception as e:
                return float(y_series.iloc[-1]), {"success": False, "fallback": "final_fit_fail", "error": str(e)}

        yhat_target, info_target = _forecast_with_spec(y_train_final, X_train_final, X_target, info)
        if not np.isfinite(yhat_target):
            yhat_target = float(y_train_final.iloc[-1])
            if isinstance(info_target, dict):
                info_target['success'] = False
                info_target['fallback'] = (info_target.get('fallback') or 'nonfinite_forecast')
        else:
            yhat_target = max(0.0, float(yhat_target))

        forecast_rows.append({
            'unq_key': key,
            'target_date': target_aug_date.date(),
            'y_true': float(target_row['actual'].iloc[0]) if not np.isnan(target_row['actual'].iloc[0]) else np.nan,
            'y_pred': yhat_target,
            'model_success': info_target.get('success') if isinstance(info_target, dict) else None,
            'order': info.get('order') if isinstance(info, dict) else None,
            'seasonal_order': info.get('seasonal_order') if isinstance(info, dict) else None,
            'trend': info.get('trend') if isinstance(info, dict) else None,
        })

        # Running stats update (based on test month)
        n_done += 1
        if info.get('success'):
            n_success += 1
        if info.get('fallback') is not None:
            n_naive += 1

        if pd.notna(y_true_test) and pd.notna(yhat_test):
            running_true.append(y_true_test)
            running_pred.append(yhat_test)

        if len(running_true) > 0:
            r_mae, r_rmse, r_mape = compute_metrics(running_true, running_pred)
        else:
            r_mae = r_rmse = r_mape = np.nan

        success_rate = (n_success / n_done) * 100 if n_done > 0 else 0.0
        naive_rate = (n_naive / n_done) * 100 if n_done > 0 else 0.0

        pbar.set_postfix({
            "done": f"{n_done}/{len(proc_keys)}",
            "success_%": f"{success_rate:.1f}",
            "naive_%": f"{naive_rate:.1f}",
            "MAE": f"{r_mae:.2f}" if pd.notna(r_mae) else "NA",
            "RMSE": f"{r_rmse:.2f}" if pd.notna(r_rmse) else "NA",
            "MAPE_%": f"{r_mape:.1f}" if pd.notna(r_mape) else "NA"
        })

    pbar.close()

    # === Prediction tables ===
    val_df = pd.DataFrame(val_rows).sort_values(['unq_key'])
    pred_df = pd.DataFrame(test_rows).sort_values(['unq_key'])
    forecast_df = pd.DataFrame(forecast_rows).sort_values(['unq_key'])

    # === Minimal output mode: only val/test/final sheets ===
    simple_val = val_df[[c for c in ["unq_key", "target_date", "y_true", "y_pred"] if c in val_df.columns]].copy()
    if "target_date" in simple_val.columns:
        simple_val = simple_val.rename(columns={"target_date": "date"})
    if "y_true" in simple_val.columns:
        simple_val = simple_val.rename(columns={"y_true": "y_actual"})
    simple_val = add_mape_accuracy_columns(simple_val, y_actual_col="y_actual", y_pred_col="y_pred")

    simple_target = forecast_df[[c for c in ["unq_key", "target_date", "y_true", "y_pred"] if c in forecast_df.columns]].copy()
    if "target_date" in simple_target.columns:
        simple_target = simple_target.rename(columns={"target_date": "date"})
    if "y_true" in simple_target.columns:
        simple_target = simple_target.rename(columns={"y_true": "y_actual"})
    simple_target = add_mape_accuracy_columns(simple_target, y_actual_col="y_actual", y_pred_col="y_pred")

    simple_test = pred_df[[c for c in ["unq_key", "target_date", "y_true", "y_pred"] if c in pred_df.columns]].copy()
    if "target_date" in simple_test.columns:
        simple_test = simple_test.rename(columns={"target_date": "date"})
    if "y_true" in simple_test.columns:
        simple_test = simple_test.rename(columns={"y_true": "y_actual"})
    simple_test = add_mape_accuracy_columns(simple_test, y_actual_col="y_actual", y_pred_col="y_pred")

    with pd.ExcelWriter(EXPORT_PATH, engine="openpyxl") as writer:
        if simple_val is not None and not simple_val.empty:
            simple_val.to_excel(writer, sheet_name="val_predictions", index=False)
        simple_target.to_excel(writer, sheet_name="final_predictions", index=False)
        if simple_test is not None and not simple_test.empty:
            simple_test.to_excel(writer, sheet_name="test_predictions", index=False)

    print(f"Wrote minimal SARIMAX output: {EXPORT_PATH}")

print("\nSARIMAX monthly runs completed.")
