# src/preprocessing.py
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """Handles feature and target scaling for a single validation fold to prevent data leakage."""
    
    def __init__(self, feature_cols: list[str], target_col: str, fold_idx: int = 0):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.fold_idx = fold_idx
        
        self.cols_to_scale = [
            col for col in self.feature_cols if not (col.startswith("sin_") or col.startswith("cos_"))
        ]
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def process_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scales training and validation data without data leakage."""
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()

        if self.cols_to_scale:
            train_scaled[self.cols_to_scale] = self.feature_scaler.fit_transform(train_df[self.cols_to_scale])
            val_scaled[self.cols_to_scale] = self.feature_scaler.transform(val_df[self.cols_to_scale])

        train_scaled[[self.target_col]] = self.target_scaler.fit_transform(train_df[[self.target_col]])
        val_scaled[[self.target_col]] = self.target_scaler.transform(val_df[[self.target_col]])

        return train_scaled, val_scaled

    def save_scalers(self, save_dir: str = "checkpoints/scalers"):
        """Saves the fitted scalers and metadata to disk for future inference or evaluation."""
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.feature_scaler, f"{save_dir}/feature_scaler_fold_{self.fold_idx}.pkl")
        joblib.dump(self.target_scaler, f"{save_dir}/target_scaler_fold_{self.fold_idx}.pkl")
        
        joblib.dump(self.cols_to_scale, f"{save_dir}/scaled_features_list_fold_{self.fold_idx}.pkl")