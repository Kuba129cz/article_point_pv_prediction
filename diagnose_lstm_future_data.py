#!/usr/bin/env python3
"""Diagnose train/validation/test distribution differences in NWP features.

Run from the repository root:
    PYTHONPATH=. python diagnose_lstm_future_data.py

This does not train or modify a model. It writes CSV summaries under
    diagnostics/lstm_future/
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATASET_PATH = "data/pv_dwd_icon_eu_previous_day.csv"
OUTPUT_DIR = "diagnostics/lstm_future"
TARGET_COL = "PV_energy"
FEATURE_COLS = [
    "Forecast_temperature",
    "Forecast_humidity",
    "Forecast cloud_cover",
    "Forecast_GHI",
    "Forecast_DHI",
    "wind_u",
    "wind_v",
]


def main() -> None:
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if "timestamp" not in df.columns:
        raise ValueError("Dataset CSV must contain a 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    if df.index.has_duplicates:
        raise ValueError("Dataset contains duplicate timestamps.")
    df = df.asfreq("h")

    train_end = pd.Timestamp("2025-05-01 00:00:00", tz="UTC")
    val_end = pd.Timestamp("2025-10-01 00:00:00", tz="UTC")
    splits = {
        "train": df.loc[df.index < train_end].copy(),
        "validation": df.loc[(df.index >= train_end) & (df.index < val_end)].copy(),
        "test": df.loc[df.index >= val_end].copy(),
    }

    selected = [c for c in [TARGET_COL, *FEATURE_COLS] if c in df.columns]
    missing_cols = [c for c in [TARGET_COL, *FEATURE_COLS] if c not in df.columns]
    if missing_cols:
        print("WARNING: Some expected columns are absent:", ", ".join(missing_cols))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    print("Rows per split:")
    for split_name, part in splits.items():
        print(f"  {split_name:10s}: {len(part):6d} rows; {part.index.min()} .. {part.index.max()}")
        for col in selected:
            x = pd.to_numeric(part[col], errors="coerce")
            finite = x[np.isfinite(x)]
            rows.append({
                "split": split_name,
                "column": col,
                "n_rows": int(len(part)),
                "n_missing_or_nonfinite": int(len(part) - len(finite)),
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "std": float(finite.std()) if len(finite) > 1 else np.nan,
                "min": float(finite.min()) if len(finite) else np.nan,
                "p01": float(finite.quantile(0.01)) if len(finite) else np.nan,
                "p10": float(finite.quantile(0.10)) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "p90": float(finite.quantile(0.90)) if len(finite) else np.nan,
                "p99": float(finite.quantile(0.99)) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
            })
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(OUTPUT_DIR, "feature_distribution_by_split.csv"), index=False)

    # Month-by-month profiles are more informative than split-wide averages,
    # because the test period contains different seasons from the training period.
    monthly_rows = []
    for split_name, part in splits.items():
        for month, group in part.groupby(part.index.to_period("M"), sort=True):
            row = {"split": split_name, "year_month": str(month), "N": len(group)}
            for col in selected:
                values = pd.to_numeric(group[col], errors="coerce")
                values = values[np.isfinite(values)]
                row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{col}_median"] = float(values.median()) if len(values) else np.nan
                row[f"{col}_p95"] = float(values.quantile(0.95)) if len(values) else np.nan
                row[f"{col}_max"] = float(values.max()) if len(values) else np.nan
            if TARGET_COL in group.columns and "Forecast_GHI" in group.columns:
                pair = group[[TARGET_COL, "Forecast_GHI"]].apply(pd.to_numeric, errors="coerce").dropna()
                row["corr_PV_energy_Forecast_GHI"] = (
                    float(pair.corr().iloc[0, 1]) if len(pair) >= 2 else np.nan
                )
                # Correlation when forecast GHI is positive is sometimes more useful
                # for solar production than the all-hours correlation.
                daytime = pair.loc[pair["Forecast_GHI"] > 20]
                row["corr_PV_energy_Forecast_GHI_GHI_gt20"] = (
                    float(daytime.corr().iloc[0, 1]) if len(daytime) >= 2 else np.nan
                )
            monthly_rows.append(row)

    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(os.path.join(OUTPUT_DIR, "feature_profile_by_month.csv"), index=False)

    print("\nSummary written to:")
    print(f"  {OUTPUT_DIR}/feature_distribution_by_split.csv")
    print(f"  {OUTPUT_DIR}/feature_profile_by_month.csv")

    print("\nKey columns by split:")
    key_cols = [c for c in [TARGET_COL, "Forecast_GHI", "Forecast cloud_cover"] if c in selected]
    if key_cols:
        print(stats.loc[stats["column"].isin(key_cols), [
            "split", "column", "n_missing_or_nonfinite", "mean", "std", "median", "p90", "p99", "max"
        ]].to_string(index=False))

    print("\nInterpretation hints:")
    print("- Look for many missing/non-finite values or large train/validation/test changes in Forecast_GHI.")
    print("- Compare the same calendar months across years in feature_profile_by_month.csv.")
    print("- A large drop in Forecast_GHI while PV_energy remains high can explain systematic underprediction.")
    print("- This script diagnoses covariate/target distributions only; it does not prove the cause of the model error.")


if __name__ == "__main__":
    main()
