import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import argparse
import random
import os
import copy
from datetime import datetime
import json
import optuna
from sklearn.metrics import r2_score

from src.preprocessing import Preprocessor
from src.dataset import Dataset
from src.models import cnnlstm

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="random seed for reproducibility")
parser.add_argument("--batch_size", default=64, type=int, help="size of batch")
parser.add_argument("--eta_min", default=1e-6, type=float, help="Minimum learning rate for Cosine Annealing scheduler")
parser.add_argument('--print_freq', type=int, default=1, help='Frequency of printing training progress')
parser.add_argument("--exp_name", default="atten", type=str)
parser.add_argument("--nominal_output", default=1293, type=int, help="Nominal output power of fve.")

parser.add_argument("--horizon", default=24, type=int, help="")
parser.add_argument("--target_col", default="PV_energy", type=str, help="predicted variable")
parser.add_argument("--cloud_col", default="Forecast cloud_cover", type=str, help="Cloud cover column used for detailed metrics")

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

def calculate_metrics(all_preds: np.ndarray, all_targets: np.ndarray, preprocessor: Preprocessor, nominal_output: float) -> dict[str, float]:
    if nominal_output <= 0:
        raise ValueError("nominal_output has to be higher than 0.")

    real_preds = preprocessor.inverse_transform_target(all_preds)
    real_targets = preprocessor.inverse_transform_target(all_targets)

    preds = real_preds.reshape(-1)
    targets = real_targets.reshape(-1)

    errors = preds - targets
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    r2 = r2_score(targets, preds)
    mbe = np.mean(errors)

    nmae = (mae / nominal_output) * 100
    nrmse = (rmse / nominal_output) * 100

    mask = targets > 0.1

    if np.any(mask):
        active_errors = errors[mask]

        active_mae = np.mean(np.abs(active_errors))
        active_rmse = np.sqrt(np.mean(active_errors ** 2))

        active_nmae = 100 * active_mae / nominal_output
        active_nrmse = 100 * active_rmse / nominal_output

        active_r2 = r2_score(targets[mask], preds[mask])
        active_mbe = np.mean(active_errors)
    else:
        active_mae = np.nan
        active_rmse = np.nan
        active_nmae = np.nan
        active_nrmse = np.nan
        active_r2 = np.nan
        active_mbe = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "nmae": float(nmae),
        "nrmse": float(nrmse),
        "r2": float(r2),
        "mbe": float(mbe),

        "active_mae": float(active_mae),
        "active_rmse": float(active_rmse),
        "active_nmae": float(active_nmae),
        "active_nrmse": float(active_nrmse),
        "active_r2": float(active_r2),
        "active_mbe": float(active_mbe),
    }


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



def calculate_metrics_unscaled(
    y_targets: np.ndarray,
    y_preds: np.ndarray,
    nominal_power: float = 1293.0,
) -> dict[str, float]:
    """Metrics for targets/predictions already converted to original units."""
    targets = np.asarray(y_targets, dtype=float).reshape(-1)
    preds = np.asarray(y_preds, dtype=float).reshape(-1)

    if targets.shape != preds.shape:
        raise ValueError("Targets and predictions must have the same shape.")
    if nominal_power <= 0:
        raise ValueError("nominal_power must be greater than zero.")

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
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r2 = float(r2_score(targets, preds)) if n >= 2 else np.nan
    mbe = float(np.mean(errors))

    active_mask = targets > 0
    if np.any(active_mask):
        active_targets = targets[active_mask]
        active_preds = preds[active_mask]
        active_errors = active_preds - active_targets
        active_mae = float(np.mean(np.abs(active_errors)))
        active_rmse = float(np.sqrt(np.mean(active_errors ** 2)))
        active_r2 = float(r2_score(active_targets, active_preds)) if len(active_targets) >= 2 else np.nan
        active_mbe = float(np.mean(active_errors))
    else:
        active_mae = active_rmse = active_r2 = active_mbe = np.nan

    return {
        "N": int(n),
        "mae": mae,
        "rmse": rmse,
        "nmae": float(100 * mae / nominal_power),
        "nrmse": float(100 * rmse / nominal_power),
        "r2": r2,
        "mbe": mbe,
        "active_mae": active_mae,
        "active_rmse": active_rmse,
        "active_nmae": float(100 * active_mae / nominal_power) if np.isfinite(active_mae) else np.nan,
        "active_nrmse": float(100 * active_rmse / nominal_power) if np.isfinite(active_rmse) else np.nan,
        "active_r2": active_r2,
        "active_mbe": active_mbe,
    }


def calculate_all_metrics(
    X_unscaled: pd.DataFrame,
    y_targets: np.ndarray,
    y_preds: np.ndarray,
    target_times: np.ndarray,
    nominal_power: float = 1293.0,
    cloud_col: str = "Forecast cloud_cover",
) -> dict:
    """Detailed metrics for multi-step predictions.

    y_targets, y_preds and target_times are matrices shaped (n_windows, horizon).
    Each prediction is evaluated at its actual target timestamp. Duplicate target
    timestamps are allowed because the same hour can be predicted from multiple
    forecast windows/leads.
    """
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
    if not np.isfinite(targets).all() or not np.isfinite(preds).all():
        raise ValueError("Targets or predictions contain NaN/inf values.")

    flat_times = pd.DatetimeIndex(pd.to_datetime(times.reshape(-1), utc=True))
    data = pd.DataFrame(
        {
            "target": targets.reshape(-1),
            "predict": preds.reshape(-1),
        },
        index=flat_times,
    )
    # Index on data may repeat because several rolling forecast windows can
    # predict the same valid hour. The feature index itself must stay unique.
    data[cloud_col] = features[cloud_col].reindex(data.index).to_numpy()

    overall_metrics = calculate_metrics_unscaled(data["target"], data["predict"], nominal_power)

    active_time = data.loc[data["target"] > 0]
    active_metrics = calculate_metrics_unscaled(active_time["target"], active_time["predict"], nominal_power)

    clear_sky_data = data.loc[data[cloud_col] < 20]
    clear_sky_metrics = calculate_metrics_unscaled(clear_sky_data["target"], clear_sky_data["predict"], nominal_power)

    variable_sky_data = data.loc[(data[cloud_col] >= 20) & (data[cloud_col] < 80)]
    variable_sky_metrics = calculate_metrics_unscaled(variable_sky_data["target"], variable_sky_data["predict"], nominal_power)

    cloudy_sky_data = data.loc[data[cloud_col] >= 80]
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

    seasonal_all = {}
    seasonal_active = {}
    for season, group in data.groupby("season", sort=False):
        seasonal_all[season] = calculate_metrics_unscaled(
            group["target"], group["predict"], nominal_power
        )
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


def transform_with_preprocessor(raw_data: pd.DataFrame, preprocessor: Preprocessor, args) -> pd.DataFrame:
    """Apply the already-fitted scalers without refitting them."""
    transformed = raw_data.copy()
    if preprocessor.lookback_cols_to_scale:
        cols = preprocessor.lookback_cols_to_scale
        transformed[cols] = preprocessor.lookback_scaler.transform(raw_data[cols])
    if preprocessor.horizon_cols_to_scale:
        cols = preprocessor.horizon_cols_to_scale
        transformed[cols] = preprocessor.horizon_scaler.transform(raw_data[cols])
    transformed[[args.target_col]] = preprocessor.target_scaler.transform(raw_data[[args.target_col]])
    return transformed


def evaluate_split_detailed(
    args,
    model: torch.nn.Module,
    history_df: pd.DataFrame,
    split_df: pd.DataFrame,
    preprocessor: Preprocessor,
    device: torch.device,
    split_name: str,
) -> dict:
    """Evaluate every prediction step and group results by cloud cover/season."""
    context = history_df.tail(args.lookback) if args.lookback > 0 else history_df.iloc[0:0]
    raw_data = pd.concat([context, split_df]).sort_index()
    scaled_data = transform_with_preprocessor(raw_data, preprocessor, args)

    dataset = Dataset(
        data=scaled_data,
        lookback=args.lookback,
        horizon=args.horizon,
        lookback_cols=args.lookback_cols,
        horizon_cols=args.horizon_cols,
        target_col=args.target_col,
    )
    if len(dataset) == 0:
        raise ValueError(f"{split_name} dataset contains no valid windows.")

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    loss, preds_scaled, targets_scaled = evaluate_model(
        model=model,
        dataloader=loader,
        criterion=torch.nn.L1Loss(),
        device=device,
    )

    preds_scaled = np.asarray(preds_scaled)
    targets_scaled = np.asarray(targets_scaled)
    if preds_scaled.ndim == 1:
        preds_scaled = preds_scaled.reshape(-1, 1)
    if targets_scaled.ndim == 1:
        targets_scaled = targets_scaled.reshape(-1, 1)

    # Inverse-transform as one target feature, then restore (windows, horizon).
    preds_real = preprocessor.inverse_transform_target(preds_scaled).reshape(preds_scaled.shape)
    targets_real = preprocessor.inverse_transform_target(targets_scaled).reshape(targets_scaled.shape)

    n_windows, n_steps = preds_real.shape
    if n_steps != args.horizon:
        raise ValueError(
            f"Model returned {n_steps} steps, but args.horizon={args.horizon}."
        )

    # Assumes Dataset creates sliding windows with stride=1: each sample's target
    # starts immediately after its lookback context and spans args.horizon rows.
    index = raw_data.index
    target_times = np.stack(
        [index[i + args.lookback : i + args.lookback + args.horizon].to_numpy()
         for i in range(n_windows)],
        axis=0,
    )
    if target_times.shape != targets_real.shape:
        raise ValueError(
            f"Could not align target timestamps {target_times.shape} with targets {targets_real.shape}. "
            "Check the windowing logic in src.dataset.Dataset."
        )

    metrics = calculate_all_metrics(
        X_unscaled=split_df,
        y_targets=targets_real,
        y_preds=preds_real,
        target_times=target_times,
        nominal_power=args.nominal_output,
        cloud_col=getattr(args, "cloud_col", "Forecast cloud_cover"),
    )

    print(f"\n{split_name} loss (scaled): {loss:.6f}")
    for group_name in ["overall", "active", "clear_sky", "variable_sky", "cloudy_sky"]:
        print(f"\n{split_name} — {group_name}:")
        print(pd.Series(metrics[group_name]).to_string())
    for group_name in ["seasonal_all", "seasonal_active"]:
        print(f"\n{split_name} — {group_name}:")
        print(metrics[group_name].to_string())

    return metrics


def save_detailed_metrics(metrics: dict, output_dir: str, split_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    scalar_metrics = {
        key: value for key, value in metrics.items()
        if not isinstance(value, pd.DataFrame)
    }
    with open(os.path.join(output_dir, f"{split_name.lower()}_metrics.json"), "w") as f:
        json.dump(scalar_metrics, f, indent=2, allow_nan=True)
    for key in ("seasonal_all", "seasonal_active"):
        metrics[key].to_csv(os.path.join(output_dir, f"{split_name.lower()}_{key}.csv"))


def run_train(args, train_df: pd.DataFrame, dev_df: pd.DataFrame, writer, device):
    preprocessor = Preprocessor(lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col)

    train_scaled, val_scaled = preprocessor.fit_and_transform(train_df=train_df, val_df=dev_df)

    if args.lookback > 0:
        val_data = pd.concat([train_scaled.tail(args.lookback), val_scaled])
    else:
        val_data = val_scaled.copy()

    train_dataset = Dataset(
        data=train_scaled,
        lookback=args.lookback,
        horizon=args.horizon,
        lookback_cols=args.lookback_cols,
        horizon_cols=args.horizon_cols,
        target_col=args.target_col,
    )

    val_dataset = Dataset(
        data=val_data,
        lookback=args.lookback,
        horizon=args.horizon,
        lookback_cols=args.lookback_cols,
        horizon_cols=args.horizon_cols,
        target_col=args.target_col,
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset contains no valid windows.")

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset contains no valid windows.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    args.input_size = len(args.lookback_cols) + len(args.horizon_cols)
    args.output_size = args.horizon

    model = cnnlstm.Model(args=args).to(device)

    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min,
    )

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

        # TensorBoard
        if writer is not None:
            writer.add_scalar(
                "Learning_Rate",
                scheduler.get_last_lr()[0],
                epoch,
            )

            writer.add_scalars(
                "MAE",
                {
                    "Train": train_metrics["mae"],
                    "Validation": val_metrics["mae"],
                },
                epoch,
            )

            writer.add_scalar(
                "Validation/R2",
                val_metrics["r2"],
                epoch,
            )

            writer.add_scalar(
                "Validation/MBE",
                val_metrics["mbe"],
                epoch,
            )

        if (epoch + 1) % print_freq == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Val MAE: {val_metrics['mae']:.2f} "
                f"({val_metrics['nmae']:.2f}%) | "
                f"Active MAE: {val_metrics['active_mae']:.2f} "
                f"({val_metrics['active_nmae']:.2f}%) | "
                f"R²: {val_metrics['r2']:.4f} | "
                f"MBE: {val_metrics['mbe']:.2f}"
            )

        scheduler.step()

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch}."
            )
            break

    if best_model_state is None or best_val_metrics is None:
        raise RuntimeError("No valid model checkpoint was obtained.")

    model.load_state_dict(best_model_state)

    result = {
        "model": model,
        "best_model_state": best_model_state,
        "preprocessor": preprocessor,
        "best_epoch": best_epoch,
        "epochs_trained": epoch + 1,
    }

    result.update({
        f"val_{name}": value
        for name, value in best_val_metrics.items()
    })

    return model, preprocessor


def evaluate_test_set(args, model, history_df: pd.DataFrame, test_df: pd.DataFrame, preprocessor: Preprocessor, device):
    context = history_df.tail(args.lookback)
    raw_data = pd.concat([context, test_df])

    test_scaled = raw_data.copy()

    if preprocessor.lookback_cols_to_scale:
        cols = preprocessor.lookback_cols_to_scale
        test_scaled[cols] = preprocessor.lookback_scaler.transform(raw_data[cols])

    if preprocessor.horizon_cols_to_scale:
        cols = preprocessor.horizon_cols_to_scale
        test_scaled[cols] = preprocessor.horizon_scaler.transform(raw_data[cols])

    test_scaled[[args.target_col]] = (
        preprocessor.target_scaler.transform(raw_data[[args.target_col]])
    )

    test_dataset = Dataset(
        data=test_scaled,
        lookback=args.lookback,
        horizon=args.horizon,
        lookback_cols=args.lookback_cols,
        horizon_cols=args.horizon_cols,
        target_col=args.target_col,
    )

    if len(test_dataset) == 0:
        raise ValueError("Test dataset contains no valid windows.")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,)

    test_loss, test_preds, test_targets = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=torch.nn.L1Loss(),
        device=device,
    )

    metrics = calculate_metrics(
        all_preds=test_preds,
        all_targets=test_targets,
        preprocessor=preprocessor,
        nominal_output=args.nominal_output,
    )

    print(f"Test loss (scaled): {test_loss:.4f}")

    for name, value in metrics.items():
        print(f"Test {name}: {value:.4f}")

    return metrics


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
    model = model_attention.Model(args=args).to(device)
    model_path = os.path.join(path_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting on the device: {device}")

    train_df, dev_df, test_df = load_dataset(args.dataset_path)
    if args.cloud_col not in dev_df.columns or args.cloud_col not in test_df.columns:
        raise ValueError(f"Cloud-cover column '{args.cloud_col}' must exist in validation and test data.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}_{timestamp}")
    try:
        model, preprocessor = run_train(
            args=args, train_df=train_df, dev_df=dev_df, writer=writer, device=device
        )
    finally:
        writer.close()

    print("Saving the model..")
    save_model(model, args, args.path_dir)

    # Validation evaluation reuses the fitted preprocessor and does not refit it.
    val_metrics = evaluate_split_detailed(
        args=args,
        model=model,
        history_df=train_df,
        split_df=dev_df,
        preprocessor=preprocessor,
        device=device,
        split_name="Validation",
    )
    save_detailed_metrics(val_metrics, args.path_dir, "validation")

    # At test time the validation period is also available as historical context.
    test_history_df = pd.concat([train_df, dev_df]).sort_index()
    test_metrics = evaluate_split_detailed(
        args=args,
        model=model,
        history_df=test_history_df,
        split_df=test_df,
        preprocessor=preprocessor,
        device=device,
        split_name="Test",
    )
    save_detailed_metrics(test_metrics, args.path_dir, "test")
    print(f"\nSaved model and detailed validation/test metrics to: {args.path_dir}")

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)