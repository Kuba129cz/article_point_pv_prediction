"""Evaluate a saved multi-seed LSTM ensemble on validation and test splits."""
# evaluate_lstm_ensemble.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.ensemble_utils import (
    actual_targets_for_times,
    get_device,
    get_split_contexts,
    load_ensemble,
    load_raw_dataset,
    make_dataset_and_times,
    predict_scaled_dataset,
)
from src.metrics import calculate_all_metrics, calculate_metrics_by_horizon


def save_metrics(output_dir: str, split_name: str, detailed: dict[str, Any], by_horizon: dict[str, pd.DataFrame]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_data = {key: value for key, value in detailed.items() if isinstance(value, dict)}
    with open(os.path.join(output_dir, f"ensemble_{split_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, allow_nan=True)
    for key, value in detailed.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(output_dir, f"ensemble_{split_name}_{key}.csv"), index=True)
    for key, value in by_horizon.items():
        value.to_csv(os.path.join(output_dir, f"ensemble_{split_name}_{key}.csv"), index=True)


def evaluate_split(
    split_name: str,
    context_raw: pd.DataFrame,
    split_raw: pd.DataFrame,
    full_raw: pd.DataFrame,
    members,
    args,
    device: torch.device,
    output_dir: str,
) -> None:
    all_predictions = []
    target_times_reference = None
    window_count_reference = None

    for member in members:
        dataset, target_times = make_dataset_and_times(context_raw, split_raw, member)
        if target_times_reference is None:
            target_times_reference = target_times
            window_count_reference = len(dataset)
        else:
            if len(dataset) != window_count_reference or not np.array_equal(target_times, target_times_reference):
                raise ValueError(
                    f"Member {member.run_id} does not produce the same windows/timestamps as the other ensemble members."
                )

        preds_scaled, _ = predict_scaled_dataset(member.model, dataset, args.batch_size, device)
        preds_original = np.asarray(member.preprocessor.inverse_transform_target(preds_scaled), dtype=float)
        if preds_original.shape != target_times.shape:
            raise ValueError(
                f"Member {member.run_id}: prediction shape {preds_original.shape} does not match "
                f"timestamp shape {target_times.shape}."
            )
        all_predictions.append(preds_original)
        print(f"  {split_name}: predicted with {member.run_id} ({len(dataset)} windows)")

    predictions_stack = np.stack(all_predictions, axis=0)  # (members, windows, horizon)
    ensemble_predictions = predictions_stack.mean(axis=0)
    ensemble_std = predictions_stack.std(axis=0, ddof=0)
    assert target_times_reference is not None

    config = members[0].config
    targets = actual_targets_for_times(full_raw, config["target_col"], target_times_reference)
    raw_features = split_raw
    common_kwargs = dict(
        X_unscaled=raw_features,
        y_targets=targets,
        y_preds=ensemble_predictions,
        target_times=target_times_reference,
        nominal_power=args.nominal_output,
        cloud_col=args.cloud_col,
    )
    detailed = calculate_all_metrics(**common_kwargs)
    by_horizon = calculate_metrics_by_horizon(**common_kwargs)

    print(f"\n{'=' * 24} ENSEMBLE {split_name.upper()} {'=' * 24}")
    print(f"Models: {len(members)} | windows: {len(targets)} | forecast horizon: {targets.shape[1]}")
    for name in ("overall", "active", "clear_sky", "variable_sky", "cloudy_sky"):
        print(f"\n{split_name.title()} — {name}:")
        print(pd.Series(detailed[name]).to_string())
    for name in ("seasonal_all", "seasonal_active"):
        print(f"\n{split_name.title()} — {name}:")
        print(detailed[name].to_string())
    print(f"\n{split_name.title()} — MAE by forecast horizon:")
    print(by_horizon["by_horizon_overall"][["N", "mae", "rmse", "nmae", "nrmse", "r2", "mbe"]].to_string())
    print(f"\n{split_name.title()} — Active MAE by forecast horizon:")
    print(by_horizon["by_horizon_active"][
        ["N", "active_mae", "active_rmse", "active_nmae", "active_nrmse", "active_r2", "active_mbe"]
    ].to_string())

    save_metrics(output_dir, split_name, detailed, by_horizon)

    # Save the ensemble's window-level predictions too, with forecast timestamps and model spread.
    rows = []
    for i in range(target_times_reference.shape[0]):
        for step in range(target_times_reference.shape[1]):
            row = {
                "window": i,
                "horizon_hour": step + 1,
                "target_timestamp_utc": pd.to_datetime(target_times_reference[i, step], utc=True).isoformat(),
                "actual": float(targets[i, step]),
                "ensemble_prediction": float(ensemble_predictions[i, step]),
                "ensemble_member_std": float(ensemble_std[i, step]),
            }
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, f"ensemble_{split_name}_predictions.csv"), index=False)
    print(f"Saved detailed ensemble {split_name} results to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="saved_models/cnnlstm")
    parser.add_argument("--preprocessor_dir", default="saved_preprocessors/cnnlstm")
    parser.add_argument("--model_prefix", default="cnnlstm_ensemble", help="Only load model run IDs starting with this prefix")
    parser.add_argument("--expected_models", default=5, type=int, help="Expected matching model count; use 0 to accept any count")
    parser.add_argument("--dataset_path", default="data/pv_dwd_icon_eu_previous_day.csv")
    parser.add_argument("--output_dir", default="saved_models/lstm/ensemble_evaluation")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--nominal_output", default=None, type=float, help="Defaults to the value saved in the model config")
    parser.add_argument("--cloud_col", default=None, help="Defaults to the value saved in the model config")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    members = load_ensemble(
        args.model_dir, args.preprocessor_dir, args.model_prefix, args.expected_models, device,
    )
    print("Loaded ensemble members:")
    for member in members:
        print(f"  - {member.run_id}")
    args.nominal_output = float(args.nominal_output if args.nominal_output is not None else members[0].config.get("nominal_output", 1293.0))
    args.cloud_col = args.cloud_col or members[0].config.get("cloud_col", "Forecast cloud_cover")

    full_raw = load_raw_dataset(args.dataset_path)
    contexts = get_split_contexts(full_raw, int(members[0].config["lookback"]))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "ensemble_members.json"), "w", encoding="utf-8") as f:
        json.dump([m.run_id for m in members], f, indent=2)

    evaluate_split("validation", *contexts["validation"], full_raw, members, args, device, args.output_dir)
    evaluate_split("test", *contexts["test"], full_raw, members, args, device, args.output_dir)
    print(f"\nEnsemble evaluation complete. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
