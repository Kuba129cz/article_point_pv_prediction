#!/usr/bin/env python3
"""Generate 24-hour day-ahead ensemble forecasts with saved CNN-LSTM members.

Default target dates (UTC): 2025-12-03, 2026-04-30, and 2026-06-08.
Run from the project root, for example:
    PYTHONPATH=. python -u trains/predict_cnnlstm_ensemble_days.py
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.ensemble_utils import get_device, load_ensemble, load_raw_dataset, transform_frame
from src.metrics import calculate_metrics_unscaled

DEFAULT_DATES = ["2025-12-03", "2026-04-30", "2026-06-08"]


def predict_one_day(
    data: pd.DataFrame,
    target_date: str,
    members: list[Any],
    device: torch.device,
) -> pd.DataFrame:
    """Predict the 24 target-day hours using each saved CNN-LSTM ensemble member."""
    start = pd.Timestamp(target_date, tz="UTC")
    lookback = int(members[0].config["lookback"])
    horizon = int(members[0].config["horizon"])
    if horizon != 24:
        raise ValueError(
            f"This script expects a 24-hour forecast horizon; model horizon={horizon}."
        )

    target_times = pd.date_range(start=start, periods=horizon, freq="h")
    history_times = pd.date_range(
        end=start - pd.Timedelta(hours=1), periods=lookback, freq="h"
    )
    required_times = history_times.append(target_times)
    chunk = data.reindex(required_times)
    if not chunk.index.equals(required_times):
        raise ValueError(f"Timestamp alignment failed for target day {target_date}.")

    all_member_predictions: list[np.ndarray] = []
    member_columns: dict[str, np.ndarray] = {}

    for member in members:
        cfg = member.config
        if member.model.__class__.__module__ != "src.models.cnnlstm":
            raise TypeError(
                f"Run {member.run_id!r} loaded architecture "
                f"{member.model.__class__.__module__!r}; expected 'src.models.cnnlstm'."
            )

        lookback_cols = list(cfg["lookback_cols"])
        horizon_cols = list(cfg.get("horizon_cols", []))
        target_col = str(cfg["target_col"])
        needed_cols = list(dict.fromkeys(lookback_cols + horizon_cols + [target_col]))
        missing_cols = [col for col in needed_cols if col not in chunk.columns]
        if missing_cols:
            raise ValueError(f"Missing input/target columns for {member.run_id}: {missing_cols}")

        history_raw = chunk.iloc[:lookback]
        future_raw = chunk.iloc[lookback:lookback + horizon]

        # The target-day PV measurements are never included in model inputs. They are
        # only copied to the output for retrospective scoring, if present in the CSV.
        for col in lookback_cols:
            values = history_raw[col].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                bad_times = history_raw.index[~np.isfinite(values)].astype(str).tolist()[:5]
                raise ValueError(
                    f"Missing/non-finite historical feature {col!r} for {member.run_id} "
                    f"at {bad_times}."
                )
        for col in horizon_cols:
            values = future_raw[col].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                bad_times = future_raw.index[~np.isfinite(values)].astype(str).tolist()[:5]
                raise ValueError(
                    f"Missing/non-finite future feature {col!r} for {member.run_id} "
                    f"at {bad_times}."
                )

        # Apply this member's saved scalers; never fit scalers on the forecast day.
        scaled_chunk = transform_frame(chunk, member.preprocessor)
        x_past_np = scaled_chunk.iloc[:lookback][lookback_cols].to_numpy(dtype=np.float32)
        if horizon_cols:
            x_future_np = scaled_chunk.iloc[lookback:lookback + horizon][horizon_cols].to_numpy(dtype=np.float32)
        else:
            x_future_np = np.empty((horizon, 0), dtype=np.float32)

        if not np.isfinite(x_past_np).all() or not np.isfinite(x_future_np).all():
            raise ValueError(
                f"Non-finite scaled inputs for run {member.run_id}, target date {target_date}."
            )

        x_past = torch.from_numpy(np.ascontiguousarray(x_past_np)).unsqueeze(0).to(device)
        x_future = torch.from_numpy(np.ascontiguousarray(x_future_np)).unsqueeze(0).to(device)
        member.model.eval()
        with torch.no_grad():
            output = member.model(x_past, x_future).squeeze(-1)

        pred_scaled = output.detach().cpu().numpy()
        pred_original = np.asarray(
            member.preprocessor.inverse_transform_target(pred_scaled), dtype=float
        ).reshape(-1)
        if pred_original.size != horizon:
            raise ValueError(
                f"Model {member.run_id} returned {pred_original.size} predictions; expected {horizon}."
            )

        all_member_predictions.append(pred_original)
        member_columns[f"prediction_{member.run_id}"] = pred_original

    predictions_stack = np.stack(all_member_predictions, axis=0)
    ensemble_prediction = predictions_stack.mean(axis=0)
    ensemble_std = predictions_stack.std(axis=0, ddof=0)

    target_col = str(members[0].config["target_col"])
    actual = data[target_col].reindex(target_times).to_numpy(dtype=float)
    result = pd.DataFrame(
        {
            "timestamp_utc": target_times,
            "target_date_utc": target_date,
            "actual": actual,
            "ensemble_prediction": ensemble_prediction,
            "ensemble_member_std": ensemble_std,
        }
    )
    for name, values in member_columns.items():
        result[name] = values
    result["error_prediction_minus_actual"] = result["ensemble_prediction"] - result["actual"]
    result["absolute_error"] = result["error_prediction_minus_actual"].abs()

    cloud_col = str(members[0].config.get("cloud_col", "Forecast cloud_cover"))
    if cloud_col in data.columns:
        result["cloud_cover"] = data[cloud_col].reindex(target_times).to_numpy(dtype=float)

    if np.isfinite(actual).all():
        nominal_power = float(members[0].config.get("nominal_output", 1293.0))
        metrics = calculate_metrics_unscaled(actual, ensemble_prediction, nominal_power=nominal_power)
        print(
            f"{target_date}: ensemble MAE={metrics['mae']:.3f}, "
            f"RMSE={metrics['rmse']:.3f}, "
            f"Active MAE={metrics['active_mae']:.3f}"
        )
    else:
        missing_count = int((~np.isfinite(actual)).sum())
        print(
            f"{target_date}: predictions generated, but {missing_count} actual target values "
            "are missing; daily metrics skipped."
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="saved_models/cnnlstm")
    parser.add_argument("--preprocessor_dir", default="saved_preprocessors/cnnlstm")
    parser.add_argument(
        "--model_prefix",
        default="cnnlstm_ensemble",
        help="Prefix of saved CNN-LSTM run IDs.",
    )
    parser.add_argument(
        "--expected_models",
        default=5,
        type=int,
        help="Expected number of members; use 0 to accept any number.",
    )
    parser.add_argument("--dataset_path", default="data/pv_dwd_icon_eu_previous_day.csv")
    parser.add_argument(
        "--output_dir",
        default="saved_models/cnnlstm/ensemble_day_ahead_predictions",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=DEFAULT_DATES,
        help="Target dates in UTC, YYYY-MM-DD.",
    )
    args = parser.parse_args()

    # Validate date strings before loading model checkpoints.
    for target_date in args.dates:
        parsed = pd.Timestamp(target_date)
        if parsed.tz is not None or parsed != parsed.normalize():
            raise ValueError(
                f"Pass dates as YYYY-MM-DD without a time or timezone; got {target_date!r}."
            )

    device = get_device()
    print(f"Device: {device}")
    members = load_ensemble(
        args.model_dir,
        args.preprocessor_dir,
        args.model_prefix,
        args.expected_models,
        device,
    )
    print("Loaded CNN-LSTM ensemble members:")
    for member in members:
        print(f"  - {member.run_id} ({member.model.__class__.__module__})")

    data = load_raw_dataset(args.dataset_path)
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = []

    for target_date in args.dates:
        day = predict_one_day(data, target_date, members, device)
        day_path = os.path.join(args.output_dir, f"cnnlstm_ensemble_forecast_{target_date}.csv")
        day.to_csv(day_path, index=False)
        print(f"Saved {target_date} forecast to: {day_path}")
        outputs.append(day)

    if outputs:
        combined = pd.concat(outputs, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "cnnlstm_ensemble_forecasts_all_days.csv")
        combined.to_csv(combined_path, index=False)
        print(f"Saved combined predictions to: {combined_path}")


if __name__ == "__main__":
    main()
