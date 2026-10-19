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

from src.data_splitting import Splitter
from src.preprocessing import Preprocessor
from src.dataset import Dataset
from src.models import model_attention

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="random seed for reproducibility")
parser.add_argument("--batch_size", default=128, type=int, help="size of batch")
parser.add_argument("--eta_min", default=1e-6, type=float, help="Minimum learning rate for Cosine Annealing scheduler")
parser.add_argument('--print_freq', type=int, default=1, help='Frequency of printing training progress')
parser.add_argument("--exp_name", default="atten", type=str)
parser.add_argument("--nominal_output", default=1293, type=int, help="Nominal output power of fve.")
parser.add_argument("--target_col", default="energy", type=str, help="predicted variable")
parser.add_argument("--test_ratio", default=0.15, type=float, help="relative size of test set")
parser.add_argument("--strategy", default="simple", type=str, choices=["simple", "expanding", "rolling"], help="strategy: 'simple', 'expanding', or 'rolling'")
parser.add_argument("--initial_train_size", default=10000, type=int, help="initial rows (hours) for training window")
parser.add_argument("--step_size", default=5000, type=int, help="step size for expanding/rolling strategy")

parser.add_argument("--lookback_cols", nargs="+", 
    default=[
        "cloud_cover.total", "pressure", "irradiance", "ozone", "humidity", 
        "openmeteo_pm10", "tmp_module", "wind_u", "wind_v", "solar_elevation", 
        "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year", "energy",
    ], 
    help="List of features for history (lookback)"
)
parser.add_argument("--horizon_cols", nargs="+", 
    default=[
        "cloud_cover.total", "pressure", "irradiance", "ozone", "humidity", 
        "openmeteo_pm10", "temperature", "wind_u", "wind_v", "solar_elevation", 
        "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year",
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

def load_dataset(test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv("data/fve_aba_dataset.csv", index_col="timestamp", parse_dates=True)
    dataset = dataset.sort_index().asfreq('1h')
    
    test_split_idx = int(len(dataset) * (1 - test_ratio))
    dev_df = dataset.iloc[:test_split_idx].copy()
    test_df = dataset.iloc[test_split_idx:].copy()
    return dev_df, test_df

def calculate_metrics(scaled_loss: float, all_preds: np.ndarray, all_targets: np.ndarray, target_mean: float, target_std: float, nominal_output: float, scaled_zero: float) -> dict:
    """Spočítá detailní metriky na reálné (ne-normalizované) škále."""
    real_preds = all_preds * target_std + target_mean
    real_targets = all_targets * target_std + target_mean
    
    real_mae = np.mean(np.abs(real_preds - real_targets))
    
    mask = real_targets > 0.1 
    if np.sum(mask) > 0:
        real_active_mae = np.mean(np.abs(real_preds[mask] - real_targets[mask]))
        active_mae_pct = (real_active_mae / nominal_output) * 100
    else:
        real_active_mae = 0.0
        active_mae_pct = 0.0
        
    r2 = r2_score(real_targets, real_preds)
    mbe = np.mean(real_preds - real_targets)
    
    return {
        "real_mae": real_mae,
        "real_active_mae": real_active_mae,
        "mae_pct": (real_mae / nominal_output) * 100,
        "active_mae_pct": active_mae_pct,
        "r2": r2,
        "mbe": mbe
    }


def train_one_epoch(model, dataloader, optimizer, criterion, device, curr_epoch):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    for batch in dataloader:
        x_past, x_future, targets = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(x_past, x_future, targets=targets, curr_epoch=curr_epoch).squeeze(-1)   
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


def run_cross_validation(args, df_splits, writer, device):
    criterion = torch.nn.L1Loss()
    global_epoch = 0
    best_val_loss = float('inf') 
    best_model_state = None
    last_preprocessor = None
    
    feature_cols = list(set(args.lookback_cols + args.horizon_cols))
    patience = getattr(args, 'patience', 5) 
    
    fold_best_maes = []
    fold_best_metrics = []
    
    for fold_idx, (train_df, val_df) in enumerate(df_splits):
        model = model_attention.Model(args=args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.eta_min)
        
        last_preprocessor = Preprocessor(feature_cols=feature_cols, target_col=args.target_col, fold_idx=fold_idx)
        train_scaled, val_scaled = last_preprocessor.process_fold(train_df=train_df, val_df=val_df)
        
        target_mean = last_preprocessor.target_scaler.mean_[0]
        target_std = last_preprocessor.target_scaler.scale_[0]
        scaled_zero = (0.0 - target_mean) / target_std
        
        train_loader = torch.utils.data.DataLoader(
            Dataset(data=train_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            Dataset(data=val_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col),
            batch_size=args.batch_size, shuffle=False
        )
        
        fold_best_val_mae = float('inf')
        fold_best_m_dict = None
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss, t_preds, t_targets = train_one_epoch(model, train_loader, optimizer, criterion, device, curr_epoch=epoch)
            val_loss, v_preds, v_targets = evaluate_model(model, val_loader, criterion, device)
            
            t_metrics = calculate_metrics(train_loss, t_preds, t_targets, target_mean, target_std, args.nominal_output, scaled_zero)
            v_metrics = calculate_metrics(val_loss, v_preds, v_targets, target_mean, target_std, args.nominal_output, scaled_zero)
            
            if v_metrics["real_mae"] < best_val_loss:
                best_val_loss = v_metrics["real_mae"]
                best_model_state = copy.deepcopy(model.state_dict())

            if v_metrics["real_mae"] < fold_best_val_mae:
                fold_best_val_mae = v_metrics["real_mae"]
                fold_best_m_dict = v_metrics
                patience_counter = 0 
            else:
                patience_counter += 1 

            if writer:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar(f'Fold_{fold_idx}/Learning_Rate', current_lr, epoch)
                writer.add_scalars(f'Fold_{fold_idx}/Real_MAE', {'Train': t_metrics["real_mae"], 'Val': v_metrics["real_mae"]}, epoch)
                writer.add_scalar(f'Fold_{fold_idx}/Val_R2', v_metrics["r2"], epoch)

            scheduler.step()
            
            # --- ZDE PŘIDÁNO: Výpis do konzole ---
            if (epoch + 1) % args.print_freq == 0 or epoch == 0:
                print(f"  Fold {fold_idx + 1} | Ep {epoch + 1}/{args.epochs} -> "
                      f"Val MAE: {v_metrics['real_mae']:.2f} ({v_metrics['mae_pct']:.2f}%) | "
                      f"Active MAE: {v_metrics['real_active_mae']:.2f} ({v_metrics['active_mae_pct']:.2f}%) | "
                      f"R²: {v_metrics['r2']:.4f} | MBE: {v_metrics['mbe']:.2f}")
            
            global_epoch += 1
            if patience_counter >= patience:
                print(f"  [Early Stopping] Fold {fold_idx + 1} ukončen předčasně v epoše {epoch + 1}.")
                break
                
        fold_best_maes.append(fold_best_val_mae)
        fold_best_metrics.append(fold_best_m_dict)

    avg_val_mae = np.mean(fold_best_maes)
    avg_val_r2 = np.mean([m["r2"] for m in fold_best_metrics])
    avg_val_mbe = np.mean([m["mbe"] for m in fold_best_metrics])
    
    avg_val_active_mae = np.mean([m["real_active_mae"] for m in fold_best_metrics])
    avg_val_active_mae_pct = np.mean([m["active_mae_pct"] for m in fold_best_metrics])
    
    return best_model_state, last_preprocessor, global_epoch, avg_val_mae, avg_val_r2, avg_val_mbe, avg_val_active_mae, avg_val_active_mae_pct


def evaluate_test_set(args, model, test_df, preprocessor, device):
    test_scaled = test_df.copy()
    if preprocessor.cols_to_scale:
        test_scaled[preprocessor.cols_to_scale] = preprocessor.feature_scaler.transform(test_df[preprocessor.cols_to_scale])
    test_scaled[[args.target_col]] = preprocessor.target_scaler.transform(test_df[[args.target_col]])

    test_loader = torch.utils.data.DataLoader(
        Dataset(data=test_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col),
        batch_size=args.batch_size, shuffle=False
    )

    target_mean = preprocessor.target_scaler.mean_[0]
    target_std = preprocessor.target_scaler.scale_[0]
    scaled_zero = (0.0 - target_mean) / target_std

    test_loss, test_preds, test_targets = evaluate_model(model, test_loader, torch.nn.L1Loss(), device)
    metrics = calculate_metrics(test_loss, test_preds, test_targets, target_mean, target_std, args.nominal_output, scaled_zero)
    return metrics


# --- OPTUNA CALLBACK ---
def print_best_callback(study, trial):
    """It clearly lists the best Trial if the current trial has broken the record."""
    if study.best_trial.number == trial.number:
        print("\n" + "★"*60)
        print(f"★★★ NEW BEST TRIAL: {trial.number} ★★★")
        print(f"Validační MAE: {trial.value:.4f}")
        print("Test MAE: {:.4f}".format(trial.user_attrs.get('test_mae', float('nan'))))
        print("Hyperparameters:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
        print("★"*60 + "\n")


# --- OPTUNA OBJECTIVE FUNCTION ---
def objective(trial: optuna.Trial, base_args: argparse.Namespace, dev_df: pd.DataFrame, test_df: pd.DataFrame, device: torch.device) -> float:
    args = copy.deepcopy(base_args)
    
    args.epochs = trial.suggest_int("epochs", 10, 30)
    args.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    
    args.past_hidden_size = trial.suggest_categorical("past_hidden_size", [16, 32, 64, 128])
    args.past_cnn_filters = trial.suggest_categorical("past_cnn_filters", [16, 32, 64, 128])
    args.past_num_lstm = trial.suggest_int("past_num_lstm", 1, 3)
    args.past_dropout = trial.suggest_float("past_dropout", 0.0, 0.5)
    
    past_num_cnn = trial.suggest_int("past_num_cnn", 1, 3)
    for i in range(past_num_cnn):
        kernel_size = trial.suggest_categorical(f"past_kernel_L{i}", [3, 5, 7])
        setattr(args, f"past_kernel_L{i}", kernel_size)
        
    args.future_hidden_size = trial.suggest_categorical("future_hidden_size", [16, 32, 64, 128])
    args.future_cnn_filters = trial.suggest_categorical("future_cnn_filters", [16, 32, 64, 128])
    args.future_num_lstm = trial.suggest_int("future_num_lstm", 1, 3)
    args.future_dropout = trial.suggest_float("future_dropout", 0.0, 0.5)
    
    future_num_cnn = trial.suggest_int("future_num_cnn", 1, 3)
    for j in range(future_num_cnn):
        kernel_size = trial.suggest_categorical(f"future_kernel_L{j}", [3, 5, 7])
        setattr(args, f"future_kernel_L{j}", kernel_size)

    args.attention_dim = trial.suggest_categorical("attention_dim", [32, 64, 128])
    args.decoder_dropout = trial.suggest_float("decoder_dropout", 0.0, 0.5)
    
    args.lookback = trial.suggest_categorical("lookback", [24, 48, 72])
    args.horizon = trial.suggest_categorical("horizon", [24, 36, 48])
    
    print(f"\n{'-'*40}\n[Trial {trial.number}] Spouštím konfiguraci:\n{json.dumps(trial.params, indent=2)}")
    
    splitter = Splitter(data=dev_df)
    df_splits = splitter.get_splits(strategy=args.strategy, initial_train_size=args.initial_train_size, step_size=args.step_size, window_size=args.step_size)
    
    best_state, final_preprocessor, _, avg_val_mae, avg_val_r2, avg_val_mbe, avg_val_active_mae, avg_val_active_mae_pct = run_cross_validation(
        args, df_splits, writer=None, device=device
    )
    
    test_mae, test_r2, test_mbe, test_active_mae, test_active_mae_pct = float('inf'), 0.0, 0.0, float('inf'), 0.0
    
    if best_state is not None:
        test_model = model_attention.Model(args=args).to(device)
        test_model.load_state_dict(best_state)
        test_metrics = evaluate_test_set(args, test_model, test_df, final_preprocessor, device)
        
        test_mae = test_metrics["real_mae"]
        test_r2 = test_metrics["r2"]
        test_mbe = test_metrics["mbe"]
        test_active_mae = test_metrics["real_active_mae"]
        test_active_mae_pct = test_metrics["active_mae_pct"]
        
    print(f"\n>> [Trial {trial.number} VÝSLEDKY]:")
    print(f"   VALIDATION -> Average MAE: {avg_val_mae:.2f} | Active MAE: {avg_val_active_mae:.2f} ({avg_val_active_mae_pct:.2f}%) | R²: {avg_val_r2:.4f} | MBE: {avg_val_mbe:.2f}")
    print(f"   TEST       ->   Final MAE: {test_mae:.2f} | Active MAE: {test_active_mae:.2f} ({test_active_mae_pct:.2f}%) | R²: {test_r2:.4f} | MBE: {test_mbe:.2f}")
    
    trial.set_user_attr("val_active_mae", avg_val_active_mae)
    trial.set_user_attr("val_active_mae_pct", avg_val_active_mae_pct)
    trial.set_user_attr("val_r2", avg_val_r2)
    trial.set_user_attr("val_mbe", avg_val_mbe)
    trial.set_user_attr("test_mae", test_mae)
    trial.set_user_attr("test_active_mae", test_active_mae)
    trial.set_user_attr("test_active_mae_pct", test_active_mae_pct)
    trial.set_user_attr("test_r2", test_r2)
    trial.set_user_attr("test_mbe", test_mbe)
    
    return avg_val_mae


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Optuna optimization on the device: {device}")

    dev_df, test_df = load_dataset(args.test_ratio)
    
    study = optuna.create_study(
        study_name="fve_attention_study",    
        storage="sqlite:///optuna_study.db", 
        load_if_exists=True,              
        direction="minimize"
    )
    
    study.optimize(lambda trial: objective(trial, args, dev_df, test_df, device), n_trials=100, callbacks=[print_best_callback])
    
    print("\n" + "="*60)
    print("OPTUNA OPTIMIZATION COMPLETED!")
    print(f"Best Trial number: {study.best_trial.number}")
    print(f"Best achieved Validation MAE: {study.best_value:.2f}")
    print("\nThe winning combination of hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*60)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)