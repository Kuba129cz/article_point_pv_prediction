# predict_lstm_ensemble_days.py
"""Generate 24-hour, day-ahead ensemble forecasts for specific UTC target dates."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.ensemble_utils import (
    get_device,
    load_ensemble,
    load_raw_dataset,
    transform_frame,
)
from src.metrics import calculate_metrics_unscaled

DEFAULT_DATES = ["2025-12-03", "2026-04-30", "2026-06-08"]


def predict_one_day(data: pd.DataFrame, target_date: str, members, device: torch.device) -> pd.DataFrame:
    start = pd.Timestamp(target_date, tz="UTC")
    lookback = int(members[0].config["lookback"])
    horizon = int(members[0].config["horizon"])
    if horizon != 24:
        raise ValueError(f"This script is designed for a 24-hour day-ahead forecast, but model horizon={horizon}.")

    target_times = pd.date_range(start=start, periods=horizon, freq="h", tz="UTC")
    history_times = pd.date_range(end=start - pd.Timedelta(hours=1), periods=lookback, freq="h", tz="UTC")
    required_times = history_times.append(target_times)
    chunk = data.reindex(required_times)
    if len(chunk) != lookback + horizon:
        raise ValueError(f"Could not create a {lookback + horizon}-row context for {target_date}.")
    if not chunk.index.equals(required_times):
        raise ValueError(f"Timestamp alignment failed for target day {target_date}.")

    all_member_predictions = []
    member_columns = {}
    for member in members:
        cfg = member.config
        lookback_cols = list(cfg["lookback_cols"])
        horizon_cols = list(cfg.get("horizon_cols", []))
        target_col = cfg["target_col"]

        required_input_cols = list(dict.fromkeys(lookback_cols + horizon_cols))
        missing_cols = [col for col in required_input_cols if col not in chunk.columns]
        if missing_cols:
            raise ValueError(f"Missing input columns for {member.run_id}: {missing_cols}")

        # Only model input rows/columns are checked. Actual target values for the forecast day
        # are not fed into the model; they are copied to output only for retrospective test scoring.
        history_raw = chunk.iloc[:lookback]
        future_raw = chunk.iloc[lookback:lookback + horizon]
        for col in lookback_cols:
            vals = history_raw[col].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                bad_times = history_raw.index[~np.isfinite(vals)].astype(str).tolist()[:5]
                raise ValueError(f"Missing/nonfinite historical input {col!r} for {member.run_id} at {bad_times}.")
        for col in horizon_cols:
            vals = future_raw[col].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                bad_times = future_raw.index[~np.isfinite(vals)].astype(str).tolist()[:5]
                raise ValueError(f"Missing/nonfinite future input {col!r} for {member.run_id} at {bad_times}.")

        scaled_chunk = transform_frame(chunk, member.preprocessor)
        x_past_np = scaled_chunk.iloc[:lookback][lookback_cols].to_numpy(dtype=np.float32)
        if horizon_cols:
            x_future_np = scaled_chunk.iloc[lookback:lookback + horizon][horizon_cols].to_numpy(dtype=np.float32)
        else:
            x_future_np = np.empty((horizon, 0), dtype=np.float32)

        if not np.isfinite(x_past_np).all() or not np.isfinite(x_future_np).all():
            raise ValueError(f"Non-finite scaled model inputs for run {member.run_id}, target date {target_date}.")

        x_past = torch.from_numpy(np.ascontiguousarray(x_past_np)).unsqueeze(0).to(device)
        x_future = torch.from_numpy(np.ascontiguousarray(x_future_np)).unsqueeze(0).to(device)
        member.model.eval()
        with torch.no_grad():
            output = member.model(x_past, x_future).squeeze(-1)
        pred_scaled = output.detach().cpu().numpy()
        pred_original = np.asarray(member.preprocessor.inverse_transform_target(pred_scaled), dtype=float).reshape(-1)
        if pred_original.size != horizon:
            raise ValueError(
                f"Model {member.run_id} returned {pred_original.size} predictions, expected {horizon}."
            )
        all_member_predictions.append(pred_original)
        member_columns[f"prediction_{member.run_id}"] = pred_original

    predictions_stack = np.stack(all_member_predictions, axis=0)
    ensemble_prediction = predictions_stack.mean(axis=0)
    ensemble_std = predictions_stack.std(axis=0, ddof=0)

    target_col = members[0].config["target_col"]
    actual = data[target_col].reindex(target_times).to_numpy(dtype=float)
    result = pd.DataFrame({
        "timestamp_utc": target_times,
        "target_date_utc": target_date,
        "actual": actual,
        "ensemble_prediction": ensemble_prediction,
        "ensemble_member_std": ensemble_std,
    })
    for name, values in member_columns.items():
        result[name] = values
    result["error_prediction_minus_actual"] = result["ensemble_prediction"] - result["actual"]
    result["absolute_error"] = result["error_prediction_minus_actual"].abs()
    if "Forecast cloud_cover" in data.columns:
        result["cloud_cover"] = data["Forecast cloud_cover"].reindex(target_times).to_numpy(dtype=float)

    finite_actual = np.isfinite(actual)
    if finite_actual.all():
        nominal_power = float(members[0].config.get("nominal_output", 1293.0))
        metrics = calculate_metrics_unscaled(actual, ensemble_prediction, nominal_power=nominal_power)
        print(
            f"{target_date}: ensemble MAE={metrics['mae']:.3f}, "
            f"RMSE={metrics['rmse']:.3f}, Active MAE={metrics['active_mae']:.3f}"
        )
    else:
        print(f"{target_date}: predictions generated; some actual target values are missing, so daily metrics skipped.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="saved_models/lstm")
    parser.add_argument("--preprocessor_dir", default="saved_preprocessors/lstm")
    parser.add_argument("--model_prefix", default="lstm_ensemble")
    parser.add_argument("--expected_models", default=5, type=int, help="Use 0 to accept any number of matching models")
    parser.add_argument("--dataset_path", default="data/pv_dwd_icon_eu_previous_day.csv")
    parser.add_argument("--output_dir", default="saved_models/lstm/ensemble_day_ahead_predictions")
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES, help="Target dates in UTC, YYYY-MM-DD")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    members = load_ensemble(
        args.model_dir, args.preprocessor_dir, args.model_prefix, args.expected_models, device,
    )
    print("Ensemble members:")
    for member in members:
        print(f"  - {member.run_id}")

    data = load_raw_dataset(args.dataset_path)
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = []
    for target_date in args.dates:
        # Strictly require a date, then forecast target-day 00:00..23:00 UTC from the
        # preceding `lookback` hourly records (24 hours for the trained configuration).
        parsed = pd.Timestamp(target_date)
        if parsed.tz is not None or parsed != parsed.normalize():
            raise ValueError(f"Pass dates as YYYY-MM-DD (without a time/timezone), got {target_date!r}.")
        day = predict_one_day(data, target_date, members, device)
        day_path = os.path.join(args.output_dir, f"ensemble_forecast_{target_date}.csv")
        day.to_csv(day_path, index=False)
        print(f"Saved {target_date} forecast to: {day_path}")
        outputs.append(day)

    if outputs:
        combined = pd.concat(outputs, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "ensemble_forecasts_all_days.csv")
        combined.to_csv(combined_path, index=False)
        print(f"Saved combined predictions to: {combined_path}")


if __name__ == "__main__":
    main()
