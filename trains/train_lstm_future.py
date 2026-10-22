"""Train an LSTM forecaster, evaluate detailed metrics, and save model/scalers separately.

Models and metric files are written under --path_dir. Fitted scaler files and the
metadata required to reload them are written under --preprocessor_dir.
"""

import argparse
import copy
import json
import os
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from src.preprocessing import Preprocessor
from src.dataset import Dataset
from src.models import lstm_future
from src.metrics import (
    calculate_metrics,
    calculate_all_metrics,
    calculate_metrics_by_horizon,
)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--eta_min", default=1e-6, type=float, help="Minimum learning rate for Cosine Annealing")
parser.add_argument("--print_freq", default=1, type=int, help="Print progress every N epochs")
parser.add_argument("--exp_name", default="atten", type=str, help="Experiment/model family name")
parser.add_argument("--run_id", default=None, type=str, help="Optional unique identifier for this model. Defaults to '<exp_name>_seed<seed>'.",)
parser.add_argument("--nominal_output", default=1293.0, type=float, help="Nominal PV power")
parser.add_argument("--cloud_col", default="Forecast cloud_cover", type=str, help="Cloud-cover column used for detailed metrics")
parser.add_argument("--horizon", default=24, type=int, help="Forecast horizon in hours")
parser.add_argument("--target_col", default="PV_energy", type=str, help="Target column")
parser.add_argument("--lookback", default=24, type=int, help="Number of historical time steps")
parser.add_argument("--dataset_path", default="data/pv_dwd_icon_eu_previous_day.csv", type=str)
parser.add_argument("--path_dir", default="saved_models/lstm_future", type=str, help="Common directory for model weights, model config, and metric files")
parser.add_argument("--preprocessor_dir", default="saved_preprocessors/lstm_future", type=str, help="Separate common directory for fitted scaler files and preprocessor metadata",)
parser.add_argument(
    "--lookback_cols", nargs="+",
    default=[
        "Forecast_temperature", "Forecast_humidity", "Forecast cloud_cover",
        "Forecast_GHI", "Forecast_DHI", "PV_energy", "sin_hour", "cos_hour",
        "sin_day_of_year", "cos_day_of_year", "wind_u", "wind_v",
    ],
    help="Features used in the historical input window",
)

parser.add_argument(
    "--horizon_cols", nargs="+",
    default=[
        "Forecast_temperature", "Forecast_humidity", "Forecast cloud_cover",
        "Forecast_GHI", "Forecast_DHI", "sin_hour", "cos_hour",
        "sin_day_of_year", "cos_day_of_year", "wind_u", "wind_v",
    ],
    help="Features used in the historical input window",
)
parser.add_argument("--epochs", default=15, type=int)
parser.add_argument("--patience", default=7, type=int)
parser.add_argument("--learning_rate", default=0.0005456724926154012, type=float)
parser.add_argument("--weight_decay", default=0.000634110719440628, type=float)
parser.add_argument("--hidden_size", default=64, type=int)
parser.add_argument("--dropout", default=0.02611872327247608, type=float)
parser.add_argument("--is_bidirectional", action="store_true")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_run_id(args: argparse.Namespace) -> str:
    raw_id = args.run_id or f"{args.exp_name}_seed{args.seed}"
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id).strip("._-")
    if not run_id:
        raise ValueError("run_id/exp_name must contain at least one letter or number")
    return run_id


def load_dataset(dataset_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(dataset_path)
    if "timestamp" not in dataset.columns:
        raise ValueError("Dataset CSV must contain a 'timestamp' column.")

    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
    dataset = dataset.set_index("timestamp").sort_index()

    if dataset.index.has_duplicates:
        raise ValueError("Dataset contains duplicate timestamps.")

    dataset = dataset.asfreq("h")

    train_end = pd.Timestamp("2025-05-01 00:00:00", tz="UTC")
    val_end = pd.Timestamp("2025-10-01 00:00:00", tz="UTC")

    train = dataset.loc[dataset.index < train_end].copy()
    val = dataset.loc[(dataset.index >= train_end) & (dataset.index < val_end)].copy()
    test = dataset.loc[dataset.index >= val_end].copy()
    return train, val, test


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in dataloader:
        x_past, x_future, targets = [b.to(device) for b in batch]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x_past, x_future).squeeze(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_past.size(0)
        all_preds.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    if not all_preds:
        raise ValueError("Training DataLoader produced no batches.")
    return total_loss / len(dataloader.dataset), np.concatenate(all_preds), np.concatenate(all_targets)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            x_past, x_future, targets = [b.to(device) for b in batch]
            outputs = model(x_past, x_future).squeeze(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * x_past.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if not all_preds:
        raise ValueError("Evaluation DataLoader produced no batches.")
    return total_loss / len(dataloader.dataset), np.concatenate(all_preds), np.concatenate(all_targets)


def save_model(model: torch.nn.Module, args: argparse.Namespace, model_dir: str, run_id: str) -> None:
    """Save model weights and config into the shared model directory, with unique names."""
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f"{run_id}.pth"))
    config_path = os.path.join(model_dir, f"{run_id}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


def save_preprocessor(preprocessor: Preprocessor, args: argparse.Namespace, run_id: str) -> None:
    """Save fitted sklearn scalers to a separate directory using Preprocessor.save_scalers()."""
    os.makedirs(args.preprocessor_dir, exist_ok=True)

    # Preprocessor.save_scalers/load_scalers refer to self.fold_idx, although the
    # supplied Preprocessor.__init__ does not define it. Assign a stable ID here.
    preprocessor.fold_idx = run_id
    preprocessor.save_scalers(save_dir=args.preprocessor_dir)

    metadata = {
        "fold_idx": run_id,
        "lookback_cols": args.lookback_cols,
        "horizon_cols": args.horizon_cols,
        "target_col": args.target_col,
        "run_id": run_id,
    }
    metadata_path = os.path.join(args.preprocessor_dir, f"{run_id}_preprocessor_config.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_model(model_dir: str, run_id: str, device: torch.device) -> torch.nn.Module:
    """Reload a model saved by save_model()."""
    config_path = os.path.join(model_dir, f"{run_id}_config.json")
    model_path = os.path.join(model_dir, f"{run_id}.pth")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    args = argparse.Namespace(**config_dict)
    model = lstm_future.Model(args=args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_preprocessor(preprocessor_dir: str, run_id: str) -> Preprocessor:
    """Restore fitted scalers for one model run from the separate scaler directory."""
    metadata_path = os.path.join(preprocessor_dir, f"{run_id}_preprocessor_config.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Preprocessor metadata not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    preprocessor = Preprocessor(
        lookback_cols=metadata["lookback_cols"],
        horizon_cols=metadata["horizon_cols"],
        target_col=metadata["target_col"],
    )
    preprocessor.fold_idx = metadata["fold_idx"]
    preprocessor.load_scalers(save_dir=preprocessor_dir)
    return preprocessor


def build_target_times(
    data: pd.DataFrame,
    n_windows: int,
    lookback: int,
    horizon: int,
    lookback_cols: list[str],
    horizon_cols: list[str],
    target_col: str,
) -> np.ndarray:
    """Mirror Dataset.valid_indices exactly and produce target timestamps per window.

    The provided Dataset checks NaNs across the union of lookback_cols,
    horizon_cols and target_col for each row in a whole (lookback + horizon)
    window. This helper mirrors that filtering so timestamps remain aligned.
    """
    index = pd.DatetimeIndex(pd.to_datetime(data.index, utc=True))
    used_cols = list(set([*(lookback_cols or []), *(horizon_cols or []), target_col]))
    missing_cols = [c for c in used_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Cannot align forecast times; missing columns: {missing_cols}")

    total_window = lookback + horizon
    n_candidates = len(data) - total_window + 1
    if n_candidates < 1:
        raise ValueError(
            f"Not enough rows to create windows: rows={len(data)}, lookback={lookback}, horizon={horizon}."
        )

    has_nan = data[used_cols].isna().any(axis=1)
    valid_mask = has_nan.rolling(window=total_window).sum() == 0
    valid_starts = np.where(valid_mask.to_numpy())[0] - total_window + 1
    valid_starts = valid_starts[valid_starts >= 0]

    if len(valid_starts) != n_windows:
        raise ValueError(
            "Unable to align target timestamps with Dataset windows. "
            f"Dataset returned {n_windows} windows, but matching Dataset's NaN filtering found "
            f"{len(valid_starts)} valid windows (from {n_candidates} candidates)."
        )

    index_utc_naive = index.tz_convert("UTC").tz_localize(None)
    return np.stack(
        [index_utc_naive[start + lookback:start + lookback + horizon].to_numpy() for start in valid_starts],
        axis=0,
    )


def save_detailed_results(
    output_dir: str,
    run_id: str,
    split_name: str,
    detailed: dict[str, Any],
    by_horizon: dict[str, pd.DataFrame],
) -> None:
    """Save detailed metrics with run-specific prefixes so ensemble runs don't overwrite each other."""
    os.makedirs(output_dir, exist_ok=True)
    scalar_result = {key: value for key, value in detailed.items() if isinstance(value, dict)}
    with open(os.path.join(output_dir, f"{run_id}_{split_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(scalar_result, f, indent=2, allow_nan=True)

    for key, value in detailed.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(output_dir, f"{run_id}_{split_name}_{key}.csv"), index=True)
    for key, value in by_horizon.items():
        value.to_csv(os.path.join(output_dir, f"{run_id}_{split_name}_{key}.csv"), index=True)


def print_detailed_results(split_name: str, detailed: dict[str, Any], by_horizon: dict[str, pd.DataFrame]) -> None:
    title = split_name.replace("_", " ").title()
    for key in ("overall", "active", "clear_sky", "variable_sky", "cloudy_sky"):
        print(f"\n{title} — {key}:")
        print(pd.Series(detailed[key]).to_string())

    for key in ("seasonal_all", "seasonal_active"):
        print(f"\n{title} — {key}:")
        print(detailed[key].to_string())

    print(f"\n{title} — metrics by forecast horizon (overall):")
    print(by_horizon["by_horizon_overall"][["N", "mae", "rmse", "nmae", "nrmse", "r2", "mbe"]].to_string())
    print(f"\n{title} — metrics by forecast horizon (active hours):")
    print(by_horizon["by_horizon_active"][
        ["N", "active_mae", "active_rmse", "active_nmae", "active_nrmse", "active_r2", "active_mbe"]
    ].to_string())


def run_detailed_evaluation(
    *,
    run_id: str,
    split_name: str,
    raw_features: pd.DataFrame,
    data_with_context: pd.DataFrame,
    dataset: Dataset,
    dataloader,
    model: torch.nn.Module,
    preprocessor: Preprocessor,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    loss, preds_scaled, targets_scaled = evaluate_model(
        model=model,
        dataloader=dataloader,
        criterion=torch.nn.L1Loss(),
        device=device,
    )
    preds = preprocessor.inverse_transform_target(preds_scaled)
    targets = preprocessor.inverse_transform_target(targets_scaled)
    target_times = build_target_times(
        data=data_with_context,
        n_windows=len(dataset),
        lookback=args.lookback,
        horizon=args.horizon,
        lookback_cols=args.lookback_cols,
        horizon_cols=args.horizon_cols,
        target_col=args.target_col,
    )

    detailed = calculate_all_metrics(
        X_unscaled=raw_features,
        y_targets=targets,
        y_preds=preds,
        target_times=target_times,
        nominal_power=args.nominal_output,
        cloud_col=args.cloud_col,
    )
    by_horizon = calculate_metrics_by_horizon(
        X_unscaled=raw_features,
        y_targets=targets,
        y_preds=preds,
        target_times=target_times,
        nominal_power=args.nominal_output,
        cloud_col=args.cloud_col,
    )

    print(f"\n{split_name.title()} loss (scaled): {loss:.6f}")
    print_detailed_results(split_name, detailed, by_horizon)
    save_detailed_results(args.path_dir, run_id, split_name, detailed, by_horizon)


def main(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.patience <= 0 or args.print_freq <= 0:
        raise ValueError("epochs, patience, and print_freq must all be positive integers.")
    if args.lookback <= 0 or args.horizon <= 0:
        raise ValueError("lookback and horizon must be positive integers.")

    run_id = safe_run_id(args)
    args.run_id = run_id
    args.preprocessor_id = run_id
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting on the device: {device}")
    print(f"Run ID: {run_id}")
    print(f"Model/metrics directory: {args.path_dir}")
    print(f"Preprocessor/scalers directory: {args.preprocessor_dir}")

    train_df, dev_df, test_df = load_dataset(args.dataset_path)
    if len(train_df) == 0 or len(dev_df) == 0 or len(test_df) == 0:
        raise ValueError("Train, validation, and test splits must all be non-empty.")

    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join("runs", f"{run_id}_{timestamp}"))

    try:
        preprocessor = Preprocessor(
            lookback_cols=args.lookback_cols,
            horizon_cols=args.horizon_cols,
            target_col=args.target_col,
        )
        train_scaled, val_scaled, test_scaled = preprocessor.fit_and_transform(
            train_df=train_df,
            val_df=dev_df,
            test_df=test_df,
        )

        check_cols = [
            "Forecast_humidity",
            "Forecast_temperature",
            "Forecast cloud_cover",
            "Forecast_GHI",
            "Forecast_DHI",
        ]

        splits = [
            ("train", train_df, train_scaled),
            ("validation", dev_df, val_scaled),
            ("test", test_df, test_scaled),
        ]


        # Dataset only sees the DataFrame passed to it; it does not automatically
        # fetch rows from a preceding split. Add prior-period history explicitly so
        # the first validation/test target hours have their real historical context.
        val_data = pd.concat([train_scaled.tail(args.lookback), val_scaled])
        test_data = pd.concat([val_scaled.tail(args.lookback), test_scaled])
        val_raw_with_context = pd.concat([train_df.tail(args.lookback), dev_df])
        test_raw_with_context = pd.concat([dev_df.tail(args.lookback), test_df])

        train_dataset = Dataset(
            data=train_scaled, lookback=args.lookback, horizon=args.horizon,
            lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols,
            target_col=args.target_col,
        )
        val_dataset = Dataset(
            data=val_data, lookback=args.lookback, horizon=args.horizon,
            lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols,
            target_col=args.target_col,
        )
        test_dataset = Dataset(
            data=test_data, lookback=args.lookback, horizon=args.horizon,
            lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols,
            target_col=args.target_col,
        )

        if len(train_dataset) == 0:
            raise ValueError("Training dataset contains no valid windows.")
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset contains no valid windows.")
        if len(test_dataset) == 0:
            raise ValueError("Test dataset contains no valid windows.")

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
        )

        args.input_size = len(args.lookback_cols) + len(args.horizon_cols)
        args.output_size = args.horizon
        model = lstm_future.Model(args=args).to(device)
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=args.epochs, eta_min=args.eta_min,
        )

        best_val_mae = float("inf")
        best_model_state = None
        best_val_metrics = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss, t_preds, t_targets = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
            )
            val_loss, v_preds, v_targets = evaluate_model(
                model, val_loader, criterion, device,
            )

            train_metrics = calculate_metrics(t_preds, t_targets, preprocessor, args.nominal_output)
            val_metrics = calculate_metrics(v_preds, v_targets, preprocessor, args.nominal_output)

            if val_metrics["active_mae"] < best_val_mae:
                best_val_mae = val_metrics["active_mae"]
                best_model_state = copy.deepcopy(model.state_dict())
                best_val_metrics = dict(val_metrics)
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            writer.add_scalar("Loss/Train_scaled", train_loss, epoch)
            writer.add_scalar("Loss/Validation_scaled", val_loss, epoch)
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)
            writer.add_scalars(
                "MAE", {"Train": train_metrics["mae"], "Validation": val_metrics["mae"]}, epoch,
            )
            writer.add_scalars(
                "Active_MAE",
                {"Train": train_metrics["active_mae"], "Validation": val_metrics["active_mae"]},
                epoch,
            )
            writer.add_scalar("Train/R2", train_metrics["r2"], epoch)
            writer.add_scalar("Validation/R2", val_metrics["r2"], epoch)
            writer.add_scalar("Train/MBE", train_metrics["mbe"], epoch)
            writer.add_scalar("Validation/MBE", val_metrics["mbe"], epoch)

            if (epoch + 1) % args.print_freq == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Train loss (scaled): {train_loss:.5f} | "
                    f"Val loss (scaled): {val_loss:.5f} | "
                    f"Train MAE: {train_metrics['mae']:.2f} | "
                    f"Train Active MAE: {train_metrics['active_mae']:.2f} | "
                    f"Val MAE: {val_metrics['mae']:.2f} ({val_metrics['nmae']:.2f}%) | "
                    f"Val Active MAE: {val_metrics['active_mae']:.2f} "
                    f"({val_metrics['active_nmae']:.2f}%) | "
                    f"Val R²: {val_metrics['r2']:.4f} | Val MBE: {val_metrics['mbe']:.2f}"
                )

            scheduler.step()
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}.")
                break

        if best_model_state is None or best_val_metrics is None:
            raise RuntimeError("No valid model checkpoint was obtained.")
        model.load_state_dict(best_model_state)

        # Model files and preprocessor scaler files are saved to separate roots.
        save_model(model, args, args.path_dir, run_id)
        save_preprocessor(preprocessor, args, run_id)

        best_epoch_path = os.path.join(args.path_dir, f"{run_id}_best_epoch_metrics.json")
        with open(best_epoch_path, "w", encoding="utf-8") as f:
            json.dump(
                {"best_epoch": best_epoch, "epochs_trained": epoch + 1, **best_val_metrics},
                f, indent=2, allow_nan=True,
            )

        print(f"\nBest epoch: {best_epoch}; best validation Active MAE: {best_val_mae:.4f}")
        print(f"Saved model to: {os.path.join(args.path_dir, f'{run_id}.pth')}")
        print(f"Saved fitted scalers to: {args.preprocessor_dir} (ID: {run_id})")

        run_detailed_evaluation(
            run_id=run_id,
            split_name="validation",
            raw_features=dev_df,
            data_with_context=val_raw_with_context,
            dataset=val_dataset,
            dataloader=val_loader,
            model=model,
            preprocessor=preprocessor,
            args=args,
            device=device,
        )
        run_detailed_evaluation(
            run_id=run_id,
            split_name="test",
            raw_features=test_df,
            data_with_context=test_raw_with_context,
            dataset=test_dataset,
            dataloader=test_loader,
            model=model,
            preprocessor=preprocessor,
            args=args,
            device=device,
        )

        print(f"\nSaved model/metrics in: {args.path_dir}")
        print(f"Saved preprocessor scalers in: {args.preprocessor_dir}")
    finally:
        writer.close()


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
