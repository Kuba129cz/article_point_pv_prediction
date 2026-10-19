import os
import json
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import r2_score

from src.preprocessing import Preprocessor
from src.dataset import Dataset
from src.models import model_attention

def load_ensemble_models(model_dir, device):
    """Loads the configuration and ALL trained models from the folder."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    args = argparse.Namespace(**config_dict)
    
    # Finds all .pth files in a directory
    model_paths = glob.glob(os.path.join(model_dir, "*.pth"))
    if not model_paths:
        raise ValueError(f"Ve složce {model_dir} nebyly nalezeny žádné modely (.pth)!")
        
    print(f"Loading {len(model_paths)} models into ensemble...")
    models = []
    for path in model_paths:
        model = model_attention.Model(args=args).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
        
    return models, args

def apply_physics_mask(preds, nominal_output):
    """It prevents the model from hallucinating below zero and above the power plant's maximum."""
    return np.clip(preds, a_min=0.0, a_max=nominal_output * 1.05)

def get_ensemble_predictions(models, dataloader, device, target_std, target_mean, nominal_output):
    """It performs inference across all models in the ensemble and returns averaged predictions and real targets."""
    all_targets = []
    ensemble_preds_list = [[] for _ in models]
    
    with torch.no_grad():
        for batch in dataloader:
            x_past, x_future, targets = [b.to(device) for b in batch]
            for i, model in enumerate(models):
                preds = model(x_past, x_future).squeeze(-1)
                ensemble_preds_list[i].append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_targets = np.concatenate(all_targets) * target_std + target_mean
    for i in range(len(models)):
        ensemble_preds_list[i] = np.concatenate(ensemble_preds_list[i]) * target_std + target_mean
        
    raw_ensemble = np.mean(ensemble_preds_list, axis=0)
    masked_ensemble = apply_physics_mask(raw_ensemble, nominal_output)
    
    return all_targets.flatten(), masked_ensemble.flatten()

def print_dataset_metrics(targets, preds, nominal_output, dataset_name):
    """Calculates and lists metrics for a given data set."""
    mbe = np.mean(preds - targets)
    mae_all = np.mean(np.abs(preds - targets))
    r2 = r2_score(targets, preds)
    
    active_mask = targets > 0.1
    mae_active = np.mean(np.abs(preds[active_mask] - targets[active_mask])) if np.sum(active_mask) > 0 else 0.0
    
    print(f"\n{'='*40}")
    print(f"ENSEMBLE METRICS ON SET: {dataset_name.upper()}")
    print(f"{'='*40}")
    print(f"Error (All) : {mae_all:.2f} kW ({(mae_all/nominal_output)*100:.2f} %)")
    print(f"Error (P>0) : {mae_active:.2f} kW ({(mae_active/nominal_output)*100:.2f} %)")
    print(f"MBE (Bias)  : {mbe:.2f} kW")
    print(f"R^2 Score   : {r2:.4f}")
    
    # Metrics AFTER calibration (Bias Correction)
    corrected_preds = apply_physics_mask(preds - mbe, nominal_output)
    mae_corr_all = np.mean(np.abs(corrected_preds - targets))
    mae_corr_active = np.mean(np.abs(corrected_preds[active_mask] - targets[active_mask])) if np.sum(active_mask) > 0 else 0.0
    
    print(f"\n BIAS-REMOVED METRICS (MBE Correction) - {dataset_name.upper()}")
    print(f"Corrected Error (All) : {mae_corr_all:.2f} kW ({(mae_corr_all/nominal_output)*100:.2f} %)")
    print(f"Corrected Error (P>0) : {mae_corr_active:.2f} kW ({(mae_corr_active/nominal_output)*100:.2f} %)")
    
    return mbe

def evaluate_and_plot_ensemble(model_dir, data_path, num_plots=5, save_dir="ensemble_plots"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The analysis runs on the device: {device}")

    models, args = load_ensemble_models(model_dir, device)
    nominal_output = getattr(args, 'nominal_output', 1293)
    
    dataset = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    dataset = dataset.sort_index().asfreq('1h')
    
    test_split_idx = int(len(dataset) * (1 - args.test_ratio))
    dev_df = dataset.iloc[:test_split_idx].copy()
    test_df = dataset.iloc[test_split_idx:].copy()
    
    val_split_idx = int(len(dev_df) * (1 - args.test_ratio))
    train_df = dev_df.iloc[:val_split_idx].copy()
    val_df = dev_df.iloc[val_split_idx:].copy()
    
    feature_cols = list(set(args.lookback_cols + args.horizon_cols))
    preprocessor = Preprocessor(feature_cols=feature_cols, target_col=args.target_col, fold_idx=0)
    
    _, val_scaled = preprocessor.process_fold(train_df=train_df, val_df=val_df)
    
    test_scaled = test_df.copy()
    if preprocessor.cols_to_scale:
        test_scaled[preprocessor.cols_to_scale] = preprocessor.feature_scaler.transform(test_df[preprocessor.cols_to_scale])
    test_scaled[[args.target_col]] = preprocessor.target_scaler.transform(test_df[[args.target_col]])
    
    target_mean = preprocessor.target_scaler.mean_[0]
    target_std = preprocessor.target_scaler.scale_[0]

    val_dataset = Dataset(data=val_scaled, lookback=args.lookback, horizon=args.horizon, 
                          lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col)
    test_dataset = Dataset(data=test_scaled, lookback=args.lookback, horizon=args.horizon, 
                           lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, target_col=args.target_col)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\nRunning inference on the VALIDATION set...")
    val_targets, val_preds = get_ensemble_predictions(models, val_loader, device, target_std, target_mean, nominal_output)
    print_dataset_metrics(val_targets, val_preds, nominal_output, dataset_name="Validation")
    
    print("\nRunning inference on the TEST set...")
    test_targets, test_preds = get_ensemble_predictions(models, test_loader, device, target_std, target_mean, nominal_output)
    test_mbe = print_dataset_metrics(test_targets, test_preds, nominal_output, dataset_name="Test")
    
    np.random.seed(42)
    indices_to_plot = np.random.choice(len(test_dataset), size=num_plots, replace=False)
    print(f"\nGeneruji {num_plots} comparison charts from the TEST set to the folder '{save_dir}'...")
    
    for idx in indices_to_plot:
        y_true = test_targets[idx * args.horizon : (idx + 1) * args.horizon] 
        y_true = test_targets[idx] if test_targets.ndim == 2 else test_targets[idx*args.horizon:(idx+1)*args.horizon]
        

    with torch.no_grad():
        for i, idx in enumerate(indices_to_plot):
            x_past, x_future, targets = test_dataset[idx]
            x_past = x_past.unsqueeze(0).to(device)
            x_future = x_future.unsqueeze(0).to(device)
            
            y_true = targets.numpy() * target_std + target_mean
            
            preds_list = []
            for model in models:
                p = model(x_past, x_future).squeeze(-1)
                preds_list.append(p[0].cpu().numpy() * target_std + target_mean)
                
            y_pred_raw = apply_physics_mask(np.mean(preds_list, axis=0), nominal_output)
            y_pred_corr = apply_physics_mask(y_pred_raw - test_mbe, nominal_output)
            
            x_axis = np.arange(1, args.horizon + 1)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            ax1.plot(x_axis, y_true, label='Real production', color='#1f77b4', linewidth=2.5, marker='o')
            ax1.plot(x_axis, y_pred_raw, label=f'Ensemble ({len(models)} modelů)', color='#ff7f0e', linewidth=2.5, linestyle='--', marker='s')
            ax1.set_title(f"Original Ensemble (Test sample #{idx}) | Global MBE: {test_mbe:.2f} kW", fontsize=13, fontweight='bold')
            ax1.set_ylabel("PV power [kW]", fontsize=11)
            ax1.grid(True, linestyle=':', alpha=0.7)
            ax1.legend(fontsize=11)
            
            ax2.plot(x_axis, y_true, label='Real production', color='#1f77b4', linewidth=2.5, marker='o')
            ax2.plot(x_axis, y_pred_corr, label='Ensemble (Bias Corrected)', color='#2ca02c', linewidth=2.5, linestyle='--', marker='^')
            ax2.set_title("Calibrated Ensemble (MBE Subtracted)", fontsize=13, fontweight='bold')
            ax2.set_xlabel("Future Horizon (Clock)", fontsize=11)
            ax2.set_ylabel("PV power [kW]", fontsize=11)
            ax2.set_xticks(x_axis)
            ax2.grid(True, linestyle=':', alpha=0.7)
            ax2.legend(fontsize=11)
            
            plt.tight_layout()
            filename = os.path.join(save_dir, f"ensemble_comparison_{idx}.png")
            plt.savefig(filename, dpi=300)
            plt.close()
        
    print("The graphs were successfully generated!")

if __name__ == "__main__":
    MODEL_DIRECTORY = "./ensemble_models" 
    DATASET_PATH = "data/fve_aba_dataset.csv"
    
    evaluate_and_plot_ensemble(model_dir=MODEL_DIRECTORY, data_path=DATASET_PATH, num_plots=5)