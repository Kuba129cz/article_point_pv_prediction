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
parser.add_argument("--target_col", default="PV_energy", type=str, help="predicted variable")
parser.add_argument("--test_ratio", default=0.15, type=float, help="relative size of test set")
parser.add_argument("--strategy", default="simple", type=str, choices=["simple", "expanding", "rolling"], help="strategy: 'simple', 'expanding', or 'rolling'")
parser.add_argument("--initial_train_size", default=10000, type=int, help="initial rows (hours) for training window")
parser.add_argument("--step_size", default=5000, type=int, help="step size for expanding/rolling strategy")

parser.add_argument("--lookback", default=24, type=int, help="number of past rows (hours) to look back for historical weather and energy data")
parser.add_argument("--horizon", default=24, type=int, help="number of future rows (hours) to look ahead for weather forecast and target predictions")

parser.add_argument("--dataset_path", default="data/pv_dwd_icon_eu_previous_day.csv", type=str, help="")

parser.add_argument("--lookback_cols", nargs="+", 
    default=[
        'Forecast_temperature', 'Forecast_humidity', 'Forecast cloud_cover', 'Forecast_GHI',
        'Forecast_DHI', 'PV_energy', 'sin_hour', 'cos_hour', 'sin_day_of_year', 'cos_day_of_year', 'wind_u', 'wind_v'
    ], 
    help="List of features for history (lookback)"
)
parser.add_argument("--horizon_cols", nargs="+", 
    default=[
        'Forecast_temperature', 'Forecast_humidity', 'Forecast cloud_cover', 'Forecast_GHI',
        'Forecast_DHI', 'sin_hour', 'cos_hour', 'sin_day_of_year', 'cos_day_of_year', 'wind_u', 'wind_v'
    ], 
    help="List of features for future horizon"
)


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

def load_dataset(test_ratio: float, dataset_path: str,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def run_validation(args, train_df: pd.DataFrame, dev_df: pd.DataFrame, writer, device):
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

    return result


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


# --- OPTUNA CALLBACK ---
def print_best_callback(study, trial):
    if study.best_trial.number == trial.number:
        print("\n" + "★" * 60)
        print(f"NEW BEST TRIAL: {trial.number}")
        print(f"Validation active MAE: {trial.value:.4f}")
        print("Hyperparameters:")

        for key, value in trial.params.items():
            print(f"  {key}: {value}")

        print("★" * 60 + "\n")


# --- OPTUNA OBJECTIVE FUNCTION ---
def objective(trial: optuna.Trial, base_args: argparse.Namespace, train_df: pd.DataFrame, dev_df: pd.DataFrame, device: torch.device,) -> float:
    args = copy.deepcopy(base_args)

    # same seed in each trial for better comparasion
    set_seed(args.seed)

    args.epochs = trial.suggest_int("epochs", 10, 30)
    args.patience = trial.suggest_int("patience", 5, 10)
    args.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    args.hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256])
    args.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    args.is_bidirectional = trial.suggest_categorical("is_bidirectional", [True, False])

    args.cnn_channels = trial.suggest_categorical("cnn_channels", [8, 16, 32, 64, 128])
    args.cnn_kernel_size = trial.suggest_categorical("cnn_kernel_size", [3, 5, 7])

    result = run_validation(args=args, train_df=train_df, dev_df=dev_df, writer=None, device=device)

    print(f"\nTrial {trial.number} – Validation's metrics:")

    for name, value in result.items():
        if name.startswith("val_") and isinstance(value, (int, float, np.integer, np.floating)):
            print(f"{name}: {value:.4f}")
            trial.set_user_attr(name, float(value))

    return float(result["val_active_mae"])


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Optuna optimization on the device: {device}")

    train_df, dev_df, test_df = load_dataset(args.test_ratio, args.dataset_path)
    
    study = optuna.create_study(
        study_name="cnnlstm_future_model",    
        storage="sqlite:///optuna_study.db", 
        load_if_exists=True,              
        direction="minimize"
    )
    
    study.optimize(lambda trial: objective(trial, args, train_df, dev_df, device), n_trials=100, callbacks=[print_best_callback])
    
    print("\n" + "="*60)
    print("OPTUNA OPTIMIZATION COMPLETED!")
    print(f"Best Trial number: {study.best_trial.number}")
    print(f"Best achieved Validation active MAE: {study.best_value:.2f}")
    print("\nThe winning combination of hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*60)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)