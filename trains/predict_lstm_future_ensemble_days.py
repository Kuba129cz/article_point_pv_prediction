"""Generate 24-hour day-ahead ensemble forecasts using saved lstm_future models only.

The script loads models named like:
    saved_models/lstm_future/lstm_future_ensemble_seed42.pth
    saved_models/lstm_future/lstm_future_ensemble_seed42_config.json

and the corresponding scaler files/metadata from saved_preprocessors/lstm_future.

Run from the project root:
    PYTHONPATH=. python -u trains/predict_lstm_future_ensemble_days.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models.lstm_future import Model as FutureLSTMModel
from src.preprocessing import Preprocessor
from src.metrics import calculate_metrics_unscaled

DEFAULT_DATES = ["2025-12-03", "2026-04-30", "2026-06-08"]


@dataclass
class EnsembleMember:
    run_id: str
    config: dict[str, Any]
    model: torch.nn.Module
    preprocessor: Preprocessor


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_raw_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the hourly data with UTC timestamps, matching the training script."""
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    if "timestamp" not in data.columns:
        raise ValueError("Dataset CSV must contain a 'timestamp' column.")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.set_index("timestamp").sort_index()
    if data.index.has_duplicates:
        raise ValueError("Dataset contains duplicate timestamps.")
    return data.asfreq("h")


def discover_run_ids(model_dir: str, model_prefix: str, expected_models: int) -> list[str]:
    """Find matching saved model/config pairs."""
    pattern = os.path.join(model_dir, f"{model_prefix}*_config.json")
    config_paths = sorted(glob.glob(pattern))
    run_ids: list[str] = []

    for config_path in config_paths:
        suffix = "_config.json"
        basename = os.path.basename(config_path)
        if not basename.endswith(suffix):
            continue
        run_id = basename[:-len(suffix)]
        weights_path = os.path.join(model_dir, f"{run_id}.pth")
        if os.path.isfile(weights_path):
            run_ids.append(run_id)

    if not run_ids:
        raise FileNotFoundError(
            f"No lstm_future models found in {model_dir!r} with prefix {model_prefix!r}. "
            "Expected matching <run_id>.pth and <run_id>_config.json files."
        )
    if expected_models > 0 and len(run_ids) != expected_models:
        raise ValueError(
            f"Expected {expected_models} models with prefix {model_prefix!r}, "
            f"but found {len(run_ids)}: {run_ids}. Use --expected_models 0 to accept any count."
        )
    return run_ids


def load_member(
    model_dir: str,
    preprocessor_dir: str,
    run_id: str,
    device: torch.device,
) -> EnsembleMember:
    """Load one lstm_future model and its own fitted scalers."""
    config_path = os.path.join(model_dir, f"{run_id}_config.json")
    weights_path = os.path.join(model_dir, f"{run_id}.pth")
    metadata_path = os.path.join(preprocessor_dir, f"{run_id}_preprocessor_config.json")

    for path in (config_path, weights_path, metadata_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file for model {run_id!r} not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata_run_id = metadata.get("run_id", run_id)
    if metadata_run_id != run_id:
        raise ValueError(
            f"Scaler metadata run_id={metadata_run_id!r} does not match model run_id={run_id!r}."
        )

    # Explicitly instantiate lstm_future; this script never imports lstm_history.
    model = FutureLSTMModel(args=Namespace(**config)).to(device)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preprocessor = Preprocessor(
        lookback_cols=metadata["lookback_cols"],
        horizon_cols=metadata["horizon_cols"],
        target_col=metadata["target_col"],
    )
    preprocessor.fold_idx = metadata.get("fold_idx", run_id)
    preprocessor.load_scalers(save_dir=preprocessor_dir)

    if list(config["lookback_cols"]) != list(preprocessor.lookback_cols):
        raise ValueError(f"Lookback columns mismatch between model and scalers for {run_id}.")
    if list(config.get("horizon_cols", [])) != list(preprocessor.horizon_cols):
        raise ValueError(f"Horizon columns mismatch between model and scalers for {run_id}.")
    if str(config["target_col"]) != str(preprocessor.target_col):
        raise ValueError(f"Target column mismatch between model and scalers for {run_id}.")

    expected_input_size = len(config["lookback_cols"]) + len(config.get("horizon_cols", []))
    if int(config.get("input_size", -1)) != expected_input_size:
        raise ValueError(
            f"Model {run_id} has input_size={config.get('input_size')}, but its feature lists require "
            f"{expected_input_size} ({len(config['lookback_cols'])} historical + "
            f"{len(config.get('horizon_cols', []))} future features)."
        )

    print(
        f"Loaded {run_id}: architecture={model.__class__.__module__}, "
        f"input_size={config['input_size']}, "
        f"lookback_features={len(config['lookback_cols'])}, "
        f"horizon_features={len(config.get('horizon_cols', []))}"
    )
    return EnsembleMember(run_id, config, model, preprocessor)


def validate_ensemble(members: list[EnsembleMember]) -> None:
    if not members:
        raise ValueError("No ensemble members were loaded.")

    keys = (
        "lookback", "horizon", "target_col", "lookback_cols", "horizon_cols",
        "input_size", "output_size", "hidden_size", "dropout", "is_bidirectional", "num_layers",
    )
    reference = members[0]
    for member in members[1:]:
        mismatches = [
            f"{key}: {reference.config.get(key)!r} vs {member.config.get(key)!r}"
            for key in keys
            if reference.config.get(key) != member.config.get(key)
        ]
        if member.model.__class__.__module__ != "src.models.lstm_future":
            mismatches.append(f"wrong architecture: {member.model.__class__.__module__}")
        if mismatches:
            raise ValueError(
                f"Ensemble member {member.run_id!r} is incompatible with {reference.run_id!r}: "
                + "; ".join(mismatches)
            )


def transform_frame(data: pd.DataFrame, preprocessor: Preprocessor) -> pd.DataFrame:
    """Apply a member's saved scalers without refitting, matching training preprocessing."""
    required = set(
        list(preprocessor.lookback_cols)
        + list(preprocessor.horizon_cols)
        + [preprocessor.target_col]
    )
    missing = sorted(col for col in required if col not in data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns required by saved preprocessor: {missing}")

    scaled = data.copy()
    raw = data  # Always transform ORIGINAL values, not values already scaled by another scaler.

    if preprocessor.lookback_cols_to_scale:
        cols = preprocessor.lookback_cols_to_scale
        scaled[cols] = preprocessor.lookback_scaler.transform(raw[cols])

    if preprocessor.horizon_cols_to_scale:
        cols = preprocessor.horizon_cols_to_scale
        scaled[cols] = preprocessor.horizon_scaler.transform(raw[cols])

    # The target scaler is applied last because training preprocessing does the same;
    # PV_energy may also be part of lookback_cols.
    scaled[[preprocessor.target_col]] = preprocessor.target_scaler.transform(raw[[preprocessor.target_col]])
    return scaled


def predict_one_day(
    data: pd.DataFrame,
    target_date: str,
    members: list[EnsembleMember],
    device: torch.device,
) -> pd.DataFrame:
    """Generate one 24-value forecast for target_date from the previous lookback hours."""
    start = pd.Timestamp(target_date, tz="UTC")
    lookback = int(members[0].config["lookback"])
    horizon = int(members[0].config["horizon"])
    if horizon != 24:
        raise ValueError(f"Expected a 24-hour horizon, but model horizon={horizon}.")

    target_times = pd.date_range(start=start, periods=horizon, freq="h")
    history_times = pd.date_range(
        end=start - pd.Timedelta(hours=1), periods=lookback, freq="h"
    )
    required_times = history_times.append(target_times)
    chunk = data.reindex(required_times)

    if not chunk.index.equals(required_times):
        raise ValueError(f"Timestamp alignment failed for target date {target_date}.")
    if len(chunk) != lookback + horizon:
        raise ValueError(f"Could not build {lookback + horizon} hourly rows for {target_date}.")

    all_predictions: list[np.ndarray] = []
    member_predictions: dict[str, np.ndarray] = {}

    for member in members:
        cfg = member.config
        lookback_cols = list(cfg["lookback_cols"])
        horizon_cols = list(cfg.get("horizon_cols", []))

        history_raw = chunk.iloc[:lookback]
        future_raw = chunk.iloc[lookback:lookback + horizon]

        # Historical inputs include prior PV_energy; future inputs are weather/time features only.
        for col in lookback_cols:
            vals = history_raw[col].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                bad = history_raw.index[~np.isfinite(vals)].astype(str).tolist()[:5]
                raise ValueError(
                    f"Missing/nonfinite historical input {col!r} for {member.run_id} "
                    f"on {target_date}: {bad}"
                )
        for col in horizon_cols:
            vals = future_raw[col].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                bad = future_raw.index[~np.isfinite(vals)].astype(str).tolist()[:5]
                raise ValueError(
                    f"Missing/nonfinite future input {col!r} for {member.run_id} "
                    f"on {target_date}: {bad}"
                )

        scaled_chunk = transform_frame(chunk, member.preprocessor)
        x_past_np = scaled_chunk.iloc[:lookback][lookback_cols].to_numpy(dtype=np.float32)
        x_future_np = (
            scaled_chunk.iloc[lookback:lookback + horizon][horizon_cols].to_numpy(dtype=np.float32)
            if horizon_cols else np.empty((horizon, 0), dtype=np.float32)
        )

        if not np.isfinite(x_past_np).all() or not np.isfinite(x_future_np).all():
            raise ValueError(
                f"Nonfinite scaled inputs for model {member.run_id} on {target_date}."
            )

        x_past = torch.from_numpy(np.ascontiguousarray(x_past_np)).unsqueeze(0).to(device)
        x_future = torch.from_numpy(np.ascontiguousarray(x_future_np)).unsqueeze(0).to(device)
        member.model.eval()
        with torch.no_grad():
            output = member.model(x_past, x_future)

        pred_scaled = output.detach().cpu().numpy().reshape(-1)
        pred_original = np.asarray(
            member.preprocessor.inverse_transform_target(pred_scaled), dtype=float
        ).reshape(-1)
        if pred_original.size != horizon:
            raise ValueError(
                f"Model {member.run_id} returned {pred_original.size} values; expected {horizon}."
            )

        all_predictions.append(pred_original)
        member_predictions[f"prediction_{member.run_id}"] = pred_original

    pred_stack = np.stack(all_predictions, axis=0)
    ensemble_prediction = pred_stack.mean(axis=0)
    ensemble_std = pred_stack.std(axis=0, ddof=0)

    target_col = str(members[0].config["target_col"])
    actual = data[target_col].reindex(target_times).to_numpy(dtype=float)
    result = pd.DataFrame({
        "timestamp_utc": target_times,
        "target_date_utc": target_date,
        "actual": actual,
        "ensemble_prediction": ensemble_prediction,
        "ensemble_member_std": ensemble_std,
    })
    for column, values in member_predictions.items():
        result[column] = values
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
            f"RMSE={metrics['rmse']:.3f}, Active MAE={metrics['active_mae']:.3f}"
        )
    else:
        print(
            f"{target_date}: predictions generated, but actual PV values are missing; "
            "daily metrics were skipped."
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="saved_models/lstm_future")
    parser.add_argument("--preprocessor_dir", default="saved_preprocessors/lstm_future")
    parser.add_argument("--model_prefix", default="lstm_future_ensemble")
    parser.add_argument(
        "--expected_models", default=5, type=int,
        help="Expected number of models; use 0 to accept any number of matches.",
    )
    parser.add_argument("--dataset_path", default="data/pv_dwd_icon_eu_previous_day.csv")
    parser.add_argument(
        "--output_dir",
        default="saved_models/lstm_future/ensemble_day_ahead_predictions",
    )
    parser.add_argument(
        "--dates", nargs="+", default=DEFAULT_DATES,
        help="Target dates in UTC, YYYY-MM-DD.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    run_ids = discover_run_ids(args.model_dir, args.model_prefix, args.expected_models)
    members = [load_member(args.model_dir, args.preprocessor_dir, run_id, device) for run_id in run_ids]
    validate_ensemble(members)
    print(f"Loaded {len(members)} compatible lstm_future ensemble members.")

    data = load_raw_dataset(args.dataset_path)
    os.makedirs(args.output_dir, exist_ok=True)
    outputs: list[pd.DataFrame] = []

    for target_date in args.dates:
        # Require literal YYYY-MM-DD without time/timezone.
        try:
            parsed = pd.to_datetime(target_date, format="%Y-%m-%d", errors="raise")
        except ValueError as exc:
            raise ValueError(
                f"Dates must use YYYY-MM-DD without a time or timezone, got {target_date!r}."
            ) from exc
        if parsed.strftime("%Y-%m-%d") != target_date:
            raise ValueError(f"Dates must use YYYY-MM-DD without a time or timezone, got {target_date!r}.")

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
