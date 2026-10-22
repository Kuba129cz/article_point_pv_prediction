# ensemble_utils.py
"""Shared loading, preprocessing, windowing, and prediction utilities for LSTM ensembles."""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.dataset import Dataset
from src.preprocessing import Preprocessor
from src.models import cnnlstm

TRAIN_END = pd.Timestamp("2025-05-01 00:00:00", tz="UTC")
VAL_END = pd.Timestamp("2025-10-01 00:00:00", tz="UTC")


@dataclass
class EnsembleMember:
    run_id: str
    config: dict[str, Any]
    model: torch.nn.Module
    preprocessor: Preprocessor


def load_raw_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the hourly time series exactly as used by the training script."""
    data = pd.read_csv(dataset_path)
    if "timestamp" not in data.columns:
        raise ValueError("Dataset CSV must contain a 'timestamp' column.")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.set_index("timestamp").sort_index()
    if data.index.has_duplicates:
        raise ValueError("Dataset contains duplicate timestamps.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Dataset index must be a DatetimeIndex.")
    return data.asfreq("h")


def split_raw_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = data.loc[data.index < TRAIN_END].copy()
    val = data.loc[(data.index >= TRAIN_END) & (data.index < VAL_END)].copy()
    test = data.loc[data.index >= VAL_END].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Expected nonempty train/validation/test splits, got {len(train)}/{len(val)}/{len(test)} rows."
        )
    return train, val, test


def discover_run_ids(model_dir: str, model_prefix: str, expected_models: int = 5) -> list[str]:
    """Find model run IDs using '<run_id>_config.json' and matching '<run_id>.pth'."""
    pattern = os.path.join(model_dir, f"{model_prefix}*_config.json")
    configs = sorted(glob.glob(pattern))
    run_ids = []
    for config_path in configs:
        basename = os.path.basename(config_path)
        if not basename.endswith("_config.json"):
            continue
        run_id = basename[:-len("_config.json")]
        weights_path = os.path.join(model_dir, f"{run_id}.pth")
        if os.path.isfile(weights_path):
            run_ids.append(run_id)

    if not run_ids:
        raise FileNotFoundError(
            f"No models found in {model_dir!r} with prefix {model_prefix!r}. Expected pairs "
            f"'<run_id>.pth' and '<run_id>_config.json'."
        )
    if expected_models > 0 and len(run_ids) != expected_models:
        raise ValueError(
            f"Expected {expected_models} models matching prefix {model_prefix!r}, found {len(run_ids)}: {run_ids}. "
            "Check the prefix or pass --expected_models 0 to accept any count."
        )
    return run_ids


def load_member(model_dir: str, preprocessor_dir: str, run_id: str, device: torch.device) -> EnsembleMember:
    config_path = os.path.join(model_dir, f"{run_id}_config.json")
    weights_path = os.path.join(model_dir, f"{run_id}.pth")
    if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing model weights/config for run ID {run_id!r} in {model_dir!r}.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    model = cnnlstm.Model(args=args).to(device)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:  # Older PyTorch versions do not support weights_only.
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    metadata_path = os.path.join(preprocessor_dir, f"{run_id}_preprocessor_config.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Preprocessor metadata missing for {run_id!r}: {metadata_path}. "
            "Each model must have its own saved scalers."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    preprocessor = Preprocessor(
        lookback_cols=metadata["lookback_cols"],
        horizon_cols=metadata["horizon_cols"],
        target_col=metadata["target_col"],
    )
    preprocessor.fold_idx = metadata.get("fold_idx", run_id)
    preprocessor.load_scalers(save_dir=preprocessor_dir)

    # Use metadata to ensure the scaler is paired with the correct run/model.
    if metadata.get("run_id", run_id) != run_id:
        raise ValueError(
            f"Scaler metadata run_id={metadata.get('run_id')!r} does not match model run_id={run_id!r}."
        )

    return EnsembleMember(run_id=run_id, config=config, model=model, preprocessor=preprocessor)


def load_ensemble(
    model_dir: str,
    preprocessor_dir: str,
    model_prefix: str,
    expected_models: int,
    device: torch.device,
) -> list[EnsembleMember]:
    run_ids = discover_run_ids(model_dir, model_prefix, expected_models)
    members = [load_member(model_dir, preprocessor_dir, run_id, device) for run_id in run_ids]
    validate_compatible_members(members)
    return members


def validate_compatible_members(members: list[EnsembleMember]) -> None:
    if not members:
        raise ValueError("Ensemble contains no members.")
    # All models must consume the same windows and have compatible architectures.
    compatibility_keys = (
        "lookback", "horizon", "target_col", "lookback_cols", "horizon_cols",
        "input_size", "output_size", "hidden_size", "dropout", "is_bidirectional",
        "num_layers", "cnn_channels", "cnn_kernel_size",
    )
    reference = members[0]
    for member in members[1:]:
        mismatches = []
        for key in compatibility_keys:
            left = reference.config.get(key)
            right = member.config.get(key)
            if left != right:
                mismatches.append(f"{key}: {left!r} vs {right!r}")
        for key in ("lookback_cols", "horizon_cols", "target_col"):
            left = getattr(reference.preprocessor, key)
            right = getattr(member.preprocessor, key)
            if left != right:
                mismatches.append(f"preprocessor.{key}: {left!r} vs {right!r}")
        if mismatches:
            raise ValueError(
                f"Model {member.run_id!r} is incompatible with {reference.run_id!r}: " + "; ".join(mismatches)
            )


def transform_frame(data: pd.DataFrame, preprocessor: Preprocessor) -> pd.DataFrame:
    """Apply a loaded Preprocessor's fitted scalers without refitting them."""
    scaled = data.copy()
    lookback_cols = list(preprocessor.lookback_cols_to_scale)
    horizon_cols = list(preprocessor.horizon_cols_to_scale)

    missing = [c for c in set(lookback_cols + horizon_cols + [preprocessor.target_col]) if c not in scaled.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns needed by saved preprocessor: {sorted(missing)}")

    if lookback_cols:
        scaled[lookback_cols] = preprocessor.lookback_scaler.transform(data[lookback_cols])
    if horizon_cols:
        scaled[horizon_cols] = preprocessor.horizon_scaler.transform(data[horizon_cols])
    # Match Preprocessor.fit_and_transform: target scaling is applied last and
    # overwrites target_col even when it is also listed in lookback_cols.
    scaled[[preprocessor.target_col]] = preprocessor.target_scaler.transform(data[[preprocessor.target_col]])
    return scaled


def get_split_contexts(
    data: pd.DataFrame, lookback: int,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Return (raw context+split data, raw target-only features) for validation/test."""
    train, val, test = split_raw_dataset(data)
    val_context = pd.concat([train.tail(lookback), val])
    test_context = pd.concat([val.tail(lookback), test])
    return {
        "validation": (val_context, val),
        "test": (test_context, test),
    }


def build_target_times(
    data: pd.DataFrame,
    n_windows: int,
    lookback: int,
    horizon: int,
    lookback_cols: list[str],
    horizon_cols: list[str],
    target_col: str,
) -> np.ndarray:
    """Mirror Dataset.valid_indices and return target timestamps for each sliding window."""
    index = pd.DatetimeIndex(pd.to_datetime(data.index, utc=True))
    used_cols = list(set([*(lookback_cols or []), *(horizon_cols or []), target_col]))
    missing_cols = [c for c in used_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Cannot align forecast timestamps; missing columns: {missing_cols}")

    total_window = lookback + horizon
    n_candidates = len(data) - total_window + 1
    if n_candidates < 1:
        raise ValueError(
            f"Not enough rows for windows: rows={len(data)}, lookback={lookback}, horizon={horizon}."
        )
    has_nan = data[used_cols].isna().any(axis=1)
    valid_mask = has_nan.rolling(window=total_window).sum() == 0
    valid_starts = np.where(valid_mask.to_numpy())[0] - total_window + 1
    valid_starts = valid_starts[valid_starts >= 0]
    if len(valid_starts) != n_windows:
        raise ValueError(
            "Could not align target timestamps with Dataset windows: "
            f"Dataset={n_windows} windows, timestamp helper={len(valid_starts)} windows. "
            "Check that the Dataset's NaN filtering has not changed."
        )
    index_naive_utc = index.tz_convert("UTC").tz_localize(None)
    return np.stack(
        [index_naive_utc[s + lookback:s + lookback + horizon].to_numpy() for s in valid_starts],
        axis=0,
    )


def make_dataset_and_times(
    context_raw: pd.DataFrame,
    split_raw: pd.DataFrame,
    member: EnsembleMember,
) -> tuple[Dataset, np.ndarray]:
    cfg = member.config
    scaled_context = transform_frame(context_raw, member.preprocessor)
    dataset = Dataset(
        data=scaled_context,
        lookback=int(cfg["lookback"]),
        horizon=int(cfg["horizon"]),
        lookback_cols=list(cfg["lookback_cols"]),
        horizon_cols=list(cfg.get("horizon_cols", [])),
        target_col=str(cfg["target_col"]),
    )
    target_times = build_target_times(
        data=context_raw,
        n_windows=len(dataset),
        lookback=int(cfg["lookback"]),
        horizon=int(cfg["horizon"]),
        lookback_cols=list(cfg["lookback_cols"]),
        horizon_cols=list(cfg.get("horizon_cols", [])),
        target_col=str(cfg["target_col"]),
    )
    # Ensure the context's target portion is the requested split (timestamps in metrics are indexed there).
    if not split_raw.index.is_unique:
        raise ValueError("Split timestamp index must be unique.")
    return dataset, target_times


def predict_scaled_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for x_past, x_future, y in loader:
            x_past = x_past.to(device)
            x_future = x_future.to(device)
            output = model(x_past, x_future).squeeze(-1)
            if output.shape != y.shape:
                raise ValueError(f"Model prediction shape {tuple(output.shape)} != target shape {tuple(y.shape)}")
            preds.append(output.cpu().numpy())
            targets.append(y.numpy())
    if not preds:
        raise ValueError("No valid forecast windows were produced by Dataset.")
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def actual_targets_for_times(data: pd.DataFrame, target_col: str, target_times: np.ndarray) -> np.ndarray:
    """Look up original-unit target values from each target timestamp."""
    flat_times = pd.DatetimeIndex(pd.to_datetime(np.asarray(target_times).reshape(-1), utc=True))
    values = data[target_col].reindex(flat_times)
    if values.isna().any():
        count = int(values.isna().sum())
        raise ValueError(f"Found {count} missing actual target values at predicted timestamps.")
    return values.to_numpy(dtype=float).reshape(np.asarray(target_times).shape)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
