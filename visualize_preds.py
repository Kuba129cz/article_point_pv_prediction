import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from src.preprocessing import Preprocessor
from src.dataset import Dataset
from src.models import model_attention

def load_trained_model(model_dir, device):
    """Loads the configuration and weights of the trained model from a folder."""
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "model.pth")
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    args = argparse.Namespace(**config_dict)
    
    model = model_attention.Model(args=args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, args

def plot_validation_sequences(model_dir, data_path, num_plots=5, save_dir="plots"):
    """
    Loads validation data, runs it through the model, and saves the selected number of
    random (or specific) 24h windows as PNG graphs.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The graphs will be generated on the device: {device}")

    model, args = load_trained_model(model_dir, device)

    dataset = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    dataset = dataset.sort_index().asfreq('1h')
    
    test_split_idx = int(len(dataset) * (1 - args.test_ratio))
    dev_df = dataset.iloc[:test_split_idx].copy()
    
    train_df = dev_df.iloc[:-2000]
    val_df = dev_df.iloc[-2000:]
    
    feature_cols = list(set(args.lookback_cols + args.horizon_cols))
    preprocessor = Preprocessor(feature_cols=feature_cols, target_col=args.target_col, fold_idx=0)
    _, val_scaled = preprocessor.process_fold(train_df=train_df, val_df=val_df)
    
    target_mean = preprocessor.target_scaler.mean_[0]
    target_std = preprocessor.target_scaler.scale_[0]

    val_dataset = Dataset(data=val_scaled, lookback=args.lookback, horizon=args.horizon, 
                          lookback_cols=args.lookback_cols, horizon_cols=args.horizon_cols, 
                          target_col=args.target_col)
    
    np.random.seed(42)
    indices_to_plot = np.random.choice(len(val_dataset), size=num_plots, replace=False)
    
    print(f"Generate {num_plots} graphs to folder '{save_dir}'...")
    
    with torch.no_grad():
        for i, idx in enumerate(indices_to_plot):
            x_past, x_future, targets = val_dataset[idx]
    
            x_past = x_past.unsqueeze(0).to(device)
            x_future = x_future.unsqueeze(0).to(device)
            
            outputs = model(x_past, x_future).squeeze(-1) # shape: (1, 24)
            
            preds_np = outputs[0].cpu().numpy() * target_std + target_mean
            targets_np = targets.numpy() * target_std + target_mean
            
            preds_np = np.clip(preds_np, a_min=0.0, a_max=None)
            
            plt.figure(figsize=(12, 6))
            
            x_axis = np.arange(1, args.horizon + 1)
            
            plt.plot(x_axis, targets_np, label='Real production (Ground Truth)', color='#1f77b4', linewidth=2.5, marker='o')
            plt.plot(x_axis, preds_np, label='Predictions (Attention)', color='#ff7f0e', linewidth=2.5, linestyle='--', marker='s')
            plt.title(f"Day-Ahead Photovoltaic Prediction (Validation Sample) #{idx})", fontsize=14, fontweight='bold')
            plt.xlabel("Future Horizon (Clock)", fontsize=12)
            plt.ylabel("PV power [kW]", fontsize=12)
            plt.xticks(x_axis) 
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend(fontsize=12, loc='upper left')
            
            filename = os.path.join(save_dir, f"prediction_sample_{idx}.png")
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close()
            
            print(f" Uloženo: {filename}")

if __name__ == "__main__":
    MODEL_DIRECTORY = "./final_best_model" 
    DATASET_PATH = "data/fve_aba_dataset.csv"
    
    plot_validation_sequences(model_dir=MODEL_DIRECTORY, data_path=DATASET_PATH, num_plots=300)
    print("Done!")