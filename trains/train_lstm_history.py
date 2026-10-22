import argparse
import copy
import json
import os
import random
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from src.preprocessing import Preprocessor
from src.dataset import Dataset
from src.models import lstm_history
from src.metrics import (
    calculate_metrics,
    calculate_all_metrics,
    calculate_metrics_by_horizon,
)

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="random seed for reproducibility")
parser.add_argument("--batch_size", default=64, type=int, help="size of batch")
parser.add_argument("--eta_min", default=1e-6, type=float, help="Minimum learning rate for Cosine Annealing scheduler")
parser.add_argument('--print_freq', type=int, default=1, help='Frequency of printing training progress')
parser.add_argument("--exp_name", default="atten", type=str)
parser.add_argument("--nominal_output", default=1293, type=int, help="Nominal output power of fve.")

parser.add_argument("--horizon", default=24, type=int, help="")
parser.add_argument("--target_col", default="PV_energy", type=str, help="predicted variable")

parser.add_argument("--lookback", default=24, type=int, help="number of past rows (hours) to look back for historical weather and energy data")
parser.add_argument("--dataset_path", default="data/pv_dwd_icon_eu_previous_day.csv", type=str, help="")
parser.add_argument("--path_dir", default="saved_models/lstm", type=str, help="where model will be saved.")

parser.add_argument("--lookback_cols", nargs="+", 
    default=[
        'Forecast_temperature', 'Forecast_humidity', 'Forecast cloud_cover', 'Forecast_GHI',
        'Forecast_DHI', 'PV_energy', 'sin_hour', 'cos_hour', 'sin_day_of_year', 'cos_day_of_year', 'wind_u', 'wind_v'
    ], 
    help="List of features for history (lookback)"
)
parser.add_argument("--horizon_cols", nargs="+", default=[], help="List of features for future horizon")


parser.add_argument("--epochs", default=20, type=int, help="")
parser.add_argument("--patience", default=10, type=int, help="")
parser.add_argument("--learning_rate", default=0.0002701450549289397, type=float, help="")
parser.add_argument("--weight_decay", default=0.005865378697389813, type=float, help="")
parser.add_argument("--hidden_size", default=128, type=int, help="")
parser.add_argument("--dropout", default=0.00758287426415688, type=float, help="")
parser.add_argument("--is_bidirectional", action="store_true", help="")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_dataset(dataset_path: str,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(dataset_path)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
    dataset = dataset.set_index("timestamp").sort_index()

    if dataset.index.has_duplicates:
        raise ValueError("Dataset contains duplicate timestamps.")

    dataset = dataset.asfreq("h")

    train_end = pd.Timestamp("2025-05-01 00:00:00", tz="UTC")
    val_end = pd.Timestamp("2025-10-01 00:00:00", tz="UTC")

    train = dataset.loc[dataset.index < train_end].copy()
    val = dataset.loc[(dataset.index >= train_end) &(dataset.index < val_end)].copy()
    test = dataset.loc[dataset.index >= val_end].copy()

    return train, val, test

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    for batch in dataloader:
        x_past, x_future, targets = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(x_past, x_future).squeeze(-1)   
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_past.size(0)
        all_preds.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.cpu().numpy())
                
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
                
    return total_loss / len(dataloader.dataset), np.concatenate(all_preds), np.concatenate(all_targets)


def save_model(model: torch.nn.Module, args: argparse.Namespace, path_dir: str)-> None:
    os.makedirs(path_dir, exist_ok=True) # Zabezpečení, že složka existuje
    torch.save(model.state_dict(), f"{path_dir}/model.pth")
    with open(f"{path_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

def load_model(path_dir: str, device: torch.device) -> torch.nn.Module:
    config_path = os.path.join(path_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    args = argparse.Namespace(**config_dict)
    model = model_attention.Model(args=args).to(device) # TODO 
    model_path = os.path.join(path_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def build_target_times(
    data: pd.DataFrame,
    n_windows: int,
    lookback: int,
    horizon: int,
    lookback_cols: list[str],
    horizon_cols: list[str],
    target_col: str,
) -> np.ndarray:
    """Create target timestamps for valid stride-1 rolling windows.

    Candidate window i uses rows [i:i+lookback] as historical input and
    [i+lookback:i+lookback+horizon] as its target horizon. The helper removes
    candidates with non-finite values in the columns used by Dataset. This
    supports Dataset implementations that skip invalid windows. If its filtering
    policy differs, the count check below raises instead of silently misaligning
    timestamps and predictions.
    """
    index = pd.DatetimeIndex(pd.to_datetime(data.index, utc=True))
    missing_cols = [c for c in [*lookback_cols, *horizon_cols, target_col] if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Cannot align forecast times; missing dataset columns: {missing_cols}")

    n_candidates = len(data) - lookback - horizon + 1
    if n_candidates < 1:
        raise ValueError(
            f"Not enough rows to create windows: rows={len(data)}, lookback={lookback}, horizon={horizon}."
        )

    # Prepare arrays once. These checks mirror a typical Dataset that skips any
    # window containing NaN/inf in model inputs or target values.
    lookback_values = data[lookback_cols].to_numpy(dtype=float) if lookback_cols else np.empty((len(data), 0))
    horizon_values = data[horizon_cols].to_numpy(dtype=float) if horizon_cols else np.empty((len(data), 0))
    target_values = data[target_col].to_numpy(dtype=float)

    valid_starts: list[int] = []
    for start in range(n_candidates):
        past_end = start + lookback
        target_end = past_end + horizon

        past_ok = np.isfinite(lookback_values[start:past_end]).all()
        future_ok = np.isfinite(horizon_values[past_end:target_end]).all()
        target_ok = np.isfinite(target_values[past_end:target_end]).all()
        if past_ok and future_ok and target_ok:
            valid_starts.append(start)

    if len(valid_starts) != n_windows:
        raise ValueError(
            "Unable to align target timestamps with Dataset windows. "
            f"Dataset returned {n_windows} windows, but checking finite values in the configured model "
            f"columns found {len(valid_starts)} valid windows (from {n_candidates} candidates). "
            "The filtering rules in src.dataset.Dataset may differ from this helper; align those rules "
            "before interpreting cloud-cover, seasonal, or horizon-specific metrics."
        )

    # Metrics parses these datetime64 timestamps as UTC.
    index_utc_naive = index.tz_convert("UTC").tz_localize(None)
    return np.stack(
        [index_utc_naive[start + lookback : start + lookback + horizon].to_numpy()
         for start in valid_starts],
        axis=0,
    )


def save_detailed_results(
    output_dir: str,
    split_name: str,
    detailed: dict[str, Any],
    by_horizon: dict[str, pd.DataFrame],
) -> None:
    """Save scalar metric groups to JSON and seasonal / horizon tables to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    scalar_result = {
        key: value
        for key, value in detailed.items()
        if isinstance(value, dict)
    }
    with open(os.path.join(output_dir, f"{split_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(scalar_result, f, indent=2, allow_nan=True)

    for key, value in detailed.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(output_dir, f"{split_name}_{key}.csv"), index=True)

    for key, value in by_horizon.items():
        value.to_csv(os.path.join(output_dir, f"{split_name}_{key}.csv"), index=True)

def print_detailed_results(split_name: str, detailed: dict[str, Any], by_horizon: dict[str, pd.DataFrame]) -> None:
    title = split_name.replace("_", " ").title()
    for key in ("overall", "active", "clear_sky", "variable_sky", "cloudy_sky"):
        print(f"\n{title} — {key}:")
        print(pd.Series(detailed[key]).to_string())

    for key in ("seasonal_all", "seasonal_active"):
        print(f"\n{title} — {key}:")
        print(detailed[key].to_string())

    print(f"\n{title} — MAE by forecast horizon:")
    print(by_horizon["by_horizon_overall"][["N", "mae", "rmse", "nmae", "nrmse", "r2", "mbe"]].to_string())
    print(f"\n{title} — Active MAE by forecast horizon:")
    print(by_horizon["by_horizon_active"][["N", "active_mae", "active_rmse", "active_nmae", "active_nrmse", "active_r2", "active_mbe"]].to_string())

def run_detailed_evaluation(
    *,
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
    save_detailed_results(args.path_dir, split_name, detailed, by_horizon)

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting on the device: {device}")

    train_df, dev_df, test_df = load_dataset(args.dataset_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}_{timestamp}")
    try:
        preprocessor = Preprocessor(lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col)

        train_scaled, val_scaled, test_scaled = preprocessor.fit_and_transform(train_df=train_df, val_df=dev_df, test_df=test_df)

        train_dataset = Dataset(data=train_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col)
        val_dataset = Dataset(data=val_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col)
        test_dataset = Dataset(data=test_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col)

        if len(train_dataset) == 0:
            raise ValueError("Training dataset contains no valid windows.")

        if len(val_dataset) == 0:
            raise ValueError("Validation dataset contains no valid windows.")

        if len(test_dataset) == 0:
            raise ValueError("Validation dataset contains no valid windows.")

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        args.input_size = len(args.lookback_cols) + len(args.horizon_cols)
        model = lstm_history.Model(args=args).to(device)

        criterion = torch.nn.L1Loss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.eta_min)

        best_val_mae = float("inf")
        best_model_state = None
        best_val_metrics = None
        best_epoch = 0

        patience = getattr(args, "patience", 5)
        patience_counter = 0
        print_freq = getattr(args, "print_freq", 1)

        for epoch in range(args.epochs):
            train_loss, t_preds, t_targets = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, v_preds, v_targets = evaluate_model(model, val_loader, criterion, device)

            # Quick per-epoch metrics from src.metrics; arrays are inverse-transformed there.
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

            if writer is not None:
                writer.add_scalar("Loss/Train_scaled", train_loss, epoch)
                writer.add_scalar("Loss/Validation_scaled", val_loss, epoch)
                writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)
                writer.add_scalars("MAE", {"Train": train_metrics["mae"], "Validation": val_metrics["mae"]}, epoch)
                writer.add_scalar("Validation/R2", val_metrics["r2"], epoch)
                writer.add_scalar("Validation/MBE", val_metrics["mbe"], epoch)
                writer.add_scalar("Validation/Active_MAE", val_metrics["active_mae"], epoch)

            if (epoch + 1) % args.print_freq == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Val MAE: {val_metrics['mae']:.2f} ({val_metrics['nmae']:.2f}%) | "
                    f"Active MAE: {val_metrics['active_mae']:.2f} "
                    f"({val_metrics['active_nmae']:.2f}%) | "
                    f"R²: {val_metrics['r2']:.4f} | MBE: {val_metrics['mbe']:.2f}"
                )

            scheduler.step()
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}.")
                break

        if best_model_state is None or best_val_metrics is None:
            raise RuntimeError("No valid model checkpoint was obtained.")
        model.load_state_dict(best_model_state)

        # Save best weights/config plus the fitted preprocessor used for inverse transformation/inference.
        save_model(model, args, args.path_dir, preprocessor)
        with open(os.path.join(args.path_dir, "best_epoch_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"best_epoch": best_epoch, "epochs_trained": epoch + 1, **best_val_metrics},
                f,
                indent=2,
                allow_nan=True,
            )

        print(f"\nBest epoch: {best_epoch}; best validation Active MAE: {best_val_mae:.4f}")
        print(f"Saved best model and preprocessor to: {args.path_dir}")

        # Detailed, post-training metrics use the restored best checkpoint, original units,
        # original feature DataFrames, and the target timestamp for every predicted hour.
        run_detailed_evaluation(
            split_name="validation",
            raw_features=dev_df,
            data_with_context=val_data,
            dataset=val_dataset,
            dataloader=val_loader,
            model=model,
            preprocessor=preprocessor,
            args=args,
            device=device,
        )
        run_detailed_evaluation(
            split_name="test",
            raw_features=test_df,
            data_with_context=test_data,
            dataset=test_dataset,
            dataloader=test_loader,
            model=model,
            preprocessor=preprocessor,
            args=args,
            device=device,
        )

        print(f"\nSaved model and detailed validation/test metrics to: {args.path_dir}")

    finally:
        writer.close()




if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)