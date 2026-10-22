# metrics.py
"""Evaluation metrics for PV forecasting models.

This module contains:
- calculate_metrics: metrics for scaled model outputs, converted back through Preprocessor.
- calculate_metrics_unscaled: metrics for values already in original units.
- calculate_all_metrics: detailed metrics by activity, cloud cover, and season for multi-step forecasts.

For calculate_all_metrics, y_targets, y_preds and target_times can have shape
(n_windows, horizon), e.g. (number_of_windows, 24). Each forecast step is evaluated
against its actual target timestamp. The same timestamp may occur multiple times
when overlapping rolling windows predict the same hour with different lead times.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def calculate_metrics(all_preds: np.ndarray, all_targets: np.ndarray, preprocessor: Any, nominal_output: float) -> dict[str, float]:
    """Calculate overall and active-hour metrics from scaled arrays.

    Predictions and targets are inverse-transformed through ``preprocessor``
    before metrics are calculated. This function is useful during training,
    e.g. for selecting the best epoch using ``active_mae``.
    """
    if nominal_output <= 0:
        raise ValueError("nominal_output has to be higher than 0.")

    real_preds = np.asarray(preprocessor.inverse_transform_target(all_preds), dtype=float)
    real_targets = np.asarray(preprocessor.inverse_transform_target(all_targets), dtype=float)

    preds = real_preds.reshape(-1)
    targets = real_targets.reshape(-1)
    if preds.shape != targets.shape:
        raise ValueError(
            f"Predictions and targets must have the same number of values; "
            f"got {preds.size} and {targets.size}."
        )
    if preds.size == 0:
        raise ValueError("Cannot calculate metrics for empty predictions/targets.")
    if not np.isfinite(preds).all() or not np.isfinite(targets).all():
        raise ValueError("Predictions or targets contain NaN/inf values.")

    errors = preds - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    r2 = float(r2_score(targets, preds)) if len(targets) >= 2 else float("nan")
    mbe = float(np.mean(errors))

    nmae = 100.0 * mae / nominal_output
    nrmse = 100.0 * rmse / nominal_output

    # Threshold kept consistent with the training script's original function.
    active_mask = targets > 0.1
    if np.any(active_mask):
        active_errors = errors[active_mask]
        active_targets = targets[active_mask]
        active_preds = preds[active_mask]
        active_mae = float(np.mean(np.abs(active_errors)))
        active_rmse = float(np.sqrt(np.mean(active_errors**2)))
        active_r2 = float(r2_score(active_targets, active_preds)) if len(active_targets) >= 2 else float("nan")
        active_mbe = float(np.mean(active_errors))
        active_nmae = 100.0 * active_mae / nominal_output
        active_nrmse = 100.0 * active_rmse / nominal_output
    else:
        active_mae = active_rmse = active_r2 = active_mbe = float("nan")
        active_nmae = active_nrmse = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "nmae": float(nmae),
        "nrmse": float(nrmse),
        "r2": r2,
        "mbe": mbe,
        "active_mae": active_mae,
        "active_rmse": active_rmse,
        "active_nmae": float(active_nmae),
        "active_nrmse": float(active_nrmse),
        "active_r2": active_r2,
        "active_mbe": active_mbe,
    }


def calculate_metrics_unscaled(y_targets: np.ndarray | pd.Series, y_preds: np.ndarray | pd.Series, nominal_power: float = 1293.0) -> dict[str, float | int]:
    """Calculate metrics for targets and predictions in original units."""
    targets = np.asarray(y_targets, dtype=float).reshape(-1)
    preds = np.asarray(y_preds, dtype=float).reshape(-1)

    if targets.shape != preds.shape:
        raise ValueError(f"Targets and predictions must have the same shape, got {targets.shape} and {preds.shape}.")
    if nominal_power <= 0:
        raise ValueError("nominal_power must be greater than zero.")
    if not np.isfinite(targets).all() or not np.isfinite(preds).all():
        raise ValueError("Targets or predictions contain NaN/inf values.")

    n = len(targets)
    if n == 0:
        return {
            "N": 0,
            "mae": np.nan, "rmse": np.nan, "nmae": np.nan,
            "nrmse": np.nan, "r2": np.nan, "mbe": np.nan,
            "active_mae": np.nan, "active_rmse": np.nan,
            "active_nmae": np.nan, "active_nrmse": np.nan,
            "active_r2": np.nan, "active_mbe": np.nan,
        }

    errors = preds - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    r2 = float(r2_score(targets, preds)) if n >= 2 else float("nan")
    mbe = float(np.mean(errors))

    # Active hours use target > 0, as in detailed breakdowns.
    active_mask = targets > 0
    if np.any(active_mask):
        active_targets = targets[active_mask]
        active_preds = preds[active_mask]
        active_errors = active_preds - active_targets
        active_mae = float(np.mean(np.abs(active_errors)))
        active_rmse = float(np.sqrt(np.mean(active_errors**2)))
        active_r2 = float(r2_score(active_targets, active_preds)) if len(active_targets) >= 2 else float("nan")
        active_mbe = float(np.mean(active_errors))
        active_nmae = float(100.0 * active_mae / nominal_power)
        active_nrmse = float(100.0 * active_rmse / nominal_power)
    else:
        active_mae = active_rmse = active_r2 = active_mbe = float("nan")
        active_nmae = active_nrmse = float("nan")

    return {
        "N": int(n),
        "mae": mae,
        "rmse": rmse,
        "nmae": float(100.0 * mae / nominal_power),
        "nrmse": float(100.0 * rmse / nominal_power),
        "r2": r2,
        "mbe": mbe,
        "active_mae": active_mae,
        "active_rmse": active_rmse,
        "active_nmae": active_nmae,
        "active_nrmse": active_nrmse,
        "active_r2": active_r2,
        "active_mbe": active_mbe,
    }


def calculate_all_metrics(X_unscaled: pd.DataFrame, y_targets: np.ndarray, y_preds: np.ndarray, target_times: np.ndarray, nominal_power: float = 1293.0, cloud_col: str = "Forecast cloud_cover") -> dict[str, Any]:
    """Calculate detailed metrics overall, by cloud cover, and by season.

    Args:
        X_unscaled: Original (not scaled) validation/test DataFrame indexed by timestamp.
        y_targets: Target values in original units, shape (n_windows, horizon).
        y_preds: Predictions in original units, same shape as ``y_targets``.
        target_times: Target timestamps matching every value, same shape as targets.
        nominal_power: Nominal PV power used for normalized metrics.
        cloud_col: Name of the cloud-cover column in ``X_unscaled``.

    Overlapping forecast windows may produce repeated target timestamps; these are
    deliberately retained so every issued forecast step contributes to the metrics.
    """
    if nominal_power <= 0:
        raise ValueError("nominal_power must be greater than zero.")

    features = X_unscaled.copy()
    features.index = pd.to_datetime(features.index, utc=True)
    if not features.index.is_unique:
        raise ValueError("Feature datetime index must be unique.")
    if cloud_col not in features.columns:
        raise ValueError(f"Missing column: {cloud_col}")

    targets = np.asarray(y_targets, dtype=float)
    preds = np.asarray(y_preds, dtype=float)
    times = np.asarray(target_times)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if times.ndim == 1:
        times = times.reshape(-1, 1)

    if targets.shape != preds.shape:
        raise ValueError(f"Target/prediction shapes differ: {targets.shape} vs {preds.shape}")
    if targets.shape != times.shape:
        raise ValueError(f"Target/time shapes differ: {targets.shape} vs {times.shape}")
    if targets.size == 0:
        raise ValueError("Cannot calculate detailed metrics for empty targets.")
    if not np.isfinite(targets).all() or not np.isfinite(preds).all():
        raise ValueError("Targets or predictions contain NaN/inf values.")

    flat_times = pd.DatetimeIndex(pd.to_datetime(times.reshape(-1), utc=True))
    data = pd.DataFrame(
        {"target": targets.reshape(-1), "predict": preds.reshape(-1)},
        index=flat_times,
    )
    data[cloud_col] = features[cloud_col].reindex(data.index).to_numpy()

    # Keep rows with missing cloud cover out of cloud-category metrics, but not overall/seasonal.
    overall_metrics = calculate_metrics_unscaled(data["target"], data["predict"], nominal_power)
    active_time = data.loc[data["target"] > 0]
    active_metrics = calculate_metrics_unscaled(active_time["target"], active_time["predict"], nominal_power)

    clear_sky_data = data.loc[data[cloud_col] < 20]
    variable_sky_data = data.loc[(data[cloud_col] >= 20) & (data[cloud_col] < 80)]
    cloudy_sky_data = data.loc[data[cloud_col] >= 80]
    clear_sky_metrics = calculate_metrics_unscaled(clear_sky_data["target"], clear_sky_data["predict"], nominal_power)
    variable_sky_metrics = calculate_metrics_unscaled(variable_sky_data["target"], variable_sky_data["predict"], nominal_power)
    cloudy_sky_metrics = calculate_metrics_unscaled(cloudy_sky_data["target"], cloudy_sky_data["predict"], nominal_power)

    month = data.index.month
    data["season"] = np.select(
        [
            np.isin(month, [12, 1, 2]),
            np.isin(month, [3, 4, 5]),
            np.isin(month, [6, 7, 8]),
        ],
        ["Winter", "Spring", "Summer"],
        default="Autumn",
    )

    seasonal_all: dict[str, dict] = {}
    seasonal_active: dict[str, dict] = {}
    for season, group in data.groupby("season", sort=False):
        seasonal_all[season] = calculate_metrics_unscaled(group["target"], group["predict"], nominal_power)
        active_group = group.loc[group["target"] > 0]
        if not active_group.empty:
            seasonal_active[season] = calculate_metrics_unscaled(
                active_group["target"], active_group["predict"], nominal_power
            )

    return {
        "overall": overall_metrics,
        "active": active_metrics,
        "clear_sky": clear_sky_metrics,
        "variable_sky": variable_sky_metrics,
        "cloudy_sky": cloudy_sky_metrics,
        "seasonal_all": pd.DataFrame.from_dict(seasonal_all, orient="index"),
        "seasonal_active": pd.DataFrame.from_dict(seasonal_active, orient="index"),
    }

def calculate_metrics_by_horizon(X_unscaled: pd.DataFrame, y_targets: np.ndarray, y_preds: np.ndarray, target_times: np.ndarray, nominal_power: float = 1293.0, cloud_col: str = "Forecast cloud_cover") -> dict[str, pd.DataFrame]:
    """Calculate detailed metrics separately for each forecast lead time.

    The input arrays should have shape ``(n_windows, horizon)``.
    Column 0 means the first predicted hour, column 1 the second, and so on.
    For each lead time, this function computes the same overall, active-hour,
    cloud-cover, and seasonal breakdowns as :func:`calculate_all_metrics`.

    Returns DataFrames indexed by ``horizon_hour`` (1..horizon), or by
    ``(horizon_hour, season)`` for seasonal metrics.
    """
    targets = np.asarray(y_targets, dtype=float)
    preds = np.asarray(y_preds, dtype=float)
    times = np.asarray(target_times)

    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if times.ndim == 1:
        times = times.reshape(-1, 1)

    if targets.ndim != 2 or preds.ndim != 2 or times.ndim != 2:
        raise ValueError("Targets, predictions, and target_times must be 2D arrays.")
    if targets.shape != preds.shape or targets.shape != times.shape:
        raise ValueError(
            f"Targets, predictions, and timestamps must have the same shape; "
            f"got {targets.shape}, {preds.shape}, and {times.shape}."
        )
    if targets.shape[1] == 0:
        raise ValueError("Forecast horizon cannot be empty.")

    scalar_groups = (
        "overall", "active", "clear_sky", "variable_sky", "cloudy_sky"
    )
    scalar_rows: dict[str, list[dict[str, float | int]]] = {
        name: [] for name in scalar_groups
    }
    seasonal_all_rows: list[pd.DataFrame] = []
    seasonal_active_rows: list[pd.DataFrame] = []

    for step in range(targets.shape[1]):
        one_step = calculate_all_metrics(
            X_unscaled=X_unscaled,
            y_targets=targets[:, step],
            y_preds=preds[:, step],
            target_times=times[:, step],
            nominal_power=nominal_power,
            cloud_col=cloud_col,
        )
        horizon_hour = step + 1

        for group_name in scalar_groups:
            row = {"horizon_hour": horizon_hour}
            row.update(one_step[group_name])
            scalar_rows[group_name].append(row)

        for source_name, destination in (
            ("seasonal_all", seasonal_all_rows),
            ("seasonal_active", seasonal_active_rows),
        ):
            seasonal = one_step[source_name].copy()
            if not seasonal.empty:
                seasonal["season"] = seasonal.index
                seasonal["horizon_hour"] = horizon_hour
                destination.append(seasonal.reset_index(drop=True))

    result: dict[str, pd.DataFrame] = {}
    for group_name, rows in scalar_rows.items():
        result[f"by_horizon_{group_name}"] = pd.DataFrame(rows).set_index("horizon_hour")

    for key, rows in (
        ("by_horizon_seasonal_all", seasonal_all_rows),
        ("by_horizon_seasonal_active", seasonal_active_rows),
    ):
        if rows:
            frame = pd.concat(rows, ignore_index=True)
            result[key] = frame.set_index(["horizon_hour", "season"]).sort_index()
        else:
            result[key] = pd.DataFrame()

    return result
