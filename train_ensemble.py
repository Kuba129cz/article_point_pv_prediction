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

from src.data_splitting import Splitter
from src.preprocessing import Preprocessor
from src.dataset import Dataset
from src.models import model_attention

parser = argparse.ArgumentParser()

SEEDS = [42, 123, 777, 1024, 2026]
SAVE_DIR = "ensemble_models"

parser.add_argument("--batch_size", default=128, type=int, help="size of batch")
parser.add_argument("--epochs", default=28, type=int, help="number of training epochs")
parser.add_argument("--learning_rate", default=0.0006347523354663052, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=0.0008413987058716558, type=float, help="weight_decay")
parser.add_argument("--eta_min", default=1e-6, type=float, help="Minimum learning rate for Cosine Annealing scheduler")
parser.add_argument('--print_freq', type=int, default=1, help='Frequency of printing training progress')
parser.add_argument("--exp_name", default="final", type=str)

# --- HistoryEncoder (Past) ---
parser.add_argument("--past_hidden_size", default=16, type=int, help="hidden states in HistoryEncoder LSTM")
parser.add_argument("--past_cnn_filters", default=16, type=int, help="number of filters in cnn in HistoryEncoder")
parser.add_argument("--past_dropout", default=0.4402635671482907, type=float, help="Dropout rate in HistoryEncoder")
parser.add_argument("--past_kernel", default=7, type=int, help="first cnn's kernel")
parser.add_argument("--past_num_cnn", default=1, type=int, help="number of past cnns")
parser.add_argument("--past_num_lstm", default=1, type=int, help="number of past lstms")

# --- FutureEncoder (Future) ---
parser.add_argument("--future_hidden_size", default=32, type=int, help="hidden states in FutureEncoder LSTM")
parser.add_argument("--future_cnn_filters", default=32, type=int, help="number of filters in cnn in FutureEncoder")
parser.add_argument("--future_kernel_L0", default=5, type=int, help="first cnn's kernel")
parser.add_argument("--future_kernel_L1", default=3, type=int, help="second cnn's kernel")
parser.add_argument("--future_num_cnn", default=2, type=int, help="number of future cnns")
parser.add_argument("--future_num_lstm", default=1, type=int, help="number of future lstms")
parser.add_argument("--future_dropout", default=0.08649601155033557, type=float, help="Dropout rate in FutureEncoder")

# --- Decoder & Attention ---
parser.add_argument("--attention_dim", default=128, type=int, help="Dimensionality of the attention projection (internal attention space)")
parser.add_argument("--decoder_dropout", default=0.07088107433700343, type=float, help="Dropout rate inside the Decoder")

parser.add_argument("--nominal_output", default=1293, type=int, help="Nominal output power of fve.")
parser.add_argument("--target_col", default="energy", type=str, help="predicted variable")
parser.add_argument("--lookback", default=24, type=int, help="number of past rows (hours) to look back for historical weather and energy data")
parser.add_argument("--horizon", default=24, type=int, help="number of future rows (hours) to look ahead for weather forecast and target predictions")
parser.add_argument("--test_ratio", default=0.15, type=float, help="relative size of test set")

# --- Training strategy ---
parser.add_argument("--strategy", default="simple", type=str, choices=["simple", "expanding", "rolling"], help="strategy: 'simple', 'expanding', or 'rolling'")
parser.add_argument("--initial_train_size", default=10000, type=int, help="initial rows (hours) for training window")
parser.add_argument("--step_size", default=5000, type=int, help="step size for expanding/rolling strategy")

# --- Features ---
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

def calculate_metrics(scaled_loss: float, scaled_active_loss: float, target_std: float, nominal_output: float) -> dict:
    """Converts normalized loss to real MAE and percentage."""
    real_mae = scaled_loss * target_std
    real_active_mae = scaled_active_loss * target_std
    
    return {
        "real_mae": real_mae,
        "real_active_mae": real_active_mae,
        "mae_pct": (real_mae / nominal_output) * 100,
        "active_mae_pct": (real_active_mae / nominal_output) * 100
    }

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaled_zero, curr_epoch):
    model.train()
    total_loss, active_err_sum, active_count = 0.0, 0.0, 0
    
    for batch in dataloader:
        x_past, x_future, targets = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(x_past, x_future, targets=targets, curr_epoch=curr_epoch).squeeze(-1)   
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_past.size(0)
        
        with torch.no_grad():
            mask = targets > scaled_zero
            if mask.sum() > 0:
                active_err_sum += torch.abs(outputs[mask] - targets[mask]).sum().item()
                active_count += mask.sum().item()
                
    return total_loss / len(dataloader.dataset), (active_err_sum / active_count if active_count > 0 else 0.0)

def evaluate_model(model, dataloader, criterion, device, scaled_zero):
    model.eval()
    total_loss, active_err_sum, active_count = 0.0, 0.0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            x_past, x_future, targets = [b.to(device) for b in batch]
            
            outputs = model(x_past, x_future).squeeze(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * x_past.size(0)
            
            mask = targets > scaled_zero
            if mask.sum() > 0:
                active_err_sum += torch.abs(outputs[mask] - targets[mask]).sum().item()
                active_count += mask.sum().item()
                
    return total_loss / len(dataloader.dataset), (active_err_sum / active_count if active_count > 0 else 0.0)

def run_cross_validation(args, model, df_splits, writer, device):
    criterion = torch.nn.L1Loss()
    
    global_epoch = 0
    best_val_loss = float('inf') 
    best_model_state = None
    last_preprocessor = None
    
    feature_cols = list(set(args.lookback_cols + args.horizon_cols))
    patience = getattr(args, 'patience', 5) 
    
    for fold_idx, (train_df, val_df) in enumerate(df_splits):
        print(f"\n--- Starting FOLD {fold_idx + 1} ---")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.eta_min)
        
        last_preprocessor = Preprocessor(feature_cols=feature_cols, target_col=args.target_col, fold_idx=fold_idx)
        train_scaled, val_scaled = last_preprocessor.process_fold(train_df=train_df, val_df=val_df)
        
        target_std = last_preprocessor.target_scaler.scale_[0]
        scaled_zero = (0.0 - last_preprocessor.target_scaler.mean_[0]) / target_std
        
        train_loader = torch.utils.data.DataLoader(
            Dataset(data=train_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            Dataset(data=val_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col),
            batch_size=args.batch_size, shuffle=False
        )
        
        fold_best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss, train_act_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaled_zero, curr_epoch=epoch)
            val_loss, val_act_loss = evaluate_model(model, val_loader, criterion, device, scaled_zero)
            
            t_metrics = calculate_metrics(train_loss, train_act_loss, target_std, args.nominal_output)
            v_metrics = calculate_metrics(val_loss, val_act_loss, target_std, args.nominal_output)
            
            if v_metrics["real_mae"] < best_val_loss:
                best_val_loss = v_metrics["real_mae"]
                best_model_state = copy.deepcopy(model.state_dict())

            if v_metrics["real_mae"] < fold_best_val_loss:
                fold_best_val_loss = v_metrics["real_mae"]
                patience_counter = 0 
            else:
                patience_counter += 1 

            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Global/Learning_Rate', current_lr, global_epoch)
            writer.add_scalars('Global/Scaled_Loss', {'Train': train_loss, 'Val': val_loss}, global_epoch)
            writer.add_scalars('Global/Real_Error_All', {'Train': t_metrics["real_mae"], 'Val': v_metrics["real_mae"]}, global_epoch)
            writer.add_scalars('Global/Real_Error_Active', {'Train': t_metrics["real_active_mae"], 'Val': v_metrics["real_active_mae"]}, global_epoch)

            scheduler.step()
            
            if (epoch + 1) % args.print_freq == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1}/{args.epochs} | Train MAE: {t_metrics['real_mae']:.2f} | Val MAE: {v_metrics['real_mae']:.2f} | Val Active: {v_metrics['real_active_mae']:.2f}")
                if patience_counter > 0:
                    print(f"    [Early Stopping: {patience_counter}/{patience} bez zlepšení]")
            
            global_epoch += 1
            
            if patience_counter >= patience:
                print(f"    [!] Zastavuji trénování aktuálního foldu. Validační chyba se nezlepšila.")
                break 
            
        print(f"-> Fold {fold_idx + 1} completed. Final Val Real Error: {v_metrics['real_mae']:.2f}")

    return best_model_state, last_preprocessor, global_epoch

def evaluate_test_set(args, model, test_df, preprocessor, writer, device, global_epoch, seed):
    print(f"\n--- Testing on hold-out Test Set (Seed {seed}) ---")
    
    test_scaled = test_df.copy()
    if preprocessor.cols_to_scale:
        test_scaled[preprocessor.cols_to_scale] = preprocessor.feature_scaler.transform(test_df[preprocessor.cols_to_scale])
    test_scaled[[args.target_col]] = preprocessor.target_scaler.transform(test_df[[args.target_col]])

    test_loader = torch.utils.data.DataLoader(
        Dataset(data=test_scaled, lookback=args.lookback, horizon=args.horizon, lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col),
        batch_size=args.batch_size, shuffle=False
    )

    target_std = preprocessor.target_scaler.scale_[0]
    scaled_zero = (0.0 - preprocessor.target_scaler.mean_[0]) / target_std

    test_loss, test_act_loss = evaluate_model(model, test_loader, torch.nn.L1Loss(), device, scaled_zero)
    metrics = calculate_metrics(test_loss, test_act_loss, target_std, args.nominal_output)
    
    print(f"Test Error (All) pro Seed {seed}: {metrics['real_mae']:.2f} ({metrics['mae_pct']:.2f}%)")

    writer.add_scalar('Test/Scaled_Loss', test_loss, global_epoch)
    writer.add_scalar('Test/Real_Error_All', metrics['real_mae'], global_epoch)
    writer.add_scalar('Test/Real_Error_Active', metrics['real_active_mae'], global_epoch)
    
    hparams = vars(args).copy()
    hparams = {k: v for k, v in hparams.items() if isinstance(v, (int, float, str, bool))}
    writer.add_hparams(hparams, {
        "hparams/Test_Real_MAE_All": metrics['real_mae'],
        "hparams/Test_MAE_All_Pct": metrics['mae_pct']
    })

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on the device: {device}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    dev_df, test_df = load_dataset(args.test_ratio)
    splitter = Splitter(data=dev_df)
    
    df_splits_list = list(splitter.get_splits(strategy=args.strategy, initial_train_size=args.initial_train_size, step_size=args.step_size, window_size=args.step_size))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"{SAVE_DIR}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    for i, current_seed in enumerate(SEEDS):
        print(f"\n{'='*50}")
        print(f"TRAIN A MODEL {i + 1}/{len(SEEDS)} [SEED: {current_seed}]")
        print(f"{'='*50}")
        
        args.seed = current_seed
        set_seed(current_seed)
        
        model = model_attention.Model(args=args).to(device)
        
        writer = SummaryWriter(log_dir=f"runs/{args.exp_name}_{args.strategy}_{timestamp}_seed_{current_seed}")

        best_state, final_preprocessor, global_epoch = run_cross_validation(args, model, df_splits_list, writer, device)

        if best_state is not None:
            model.load_state_dict(best_state)
            
            model_save_path = os.path.join(SAVE_DIR, f"model_seed_{current_seed}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"\n-> Successfully saved: {model_save_path}")

        evaluate_test_set(args, model, test_df, final_preprocessor, writer, device, global_epoch, current_seed)
        writer.close()
        
    print("\n All 5 models have been successfully trained and saved to the folder:", SAVE_DIR)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)