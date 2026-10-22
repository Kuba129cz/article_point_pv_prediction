# src/preprocessing.py
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocessor:
    """
    Handles feature and target scaling for a single validation fold to prevent data leakage.
    """
    def __init__(self, lookback_cols: list[str], horizon_cols: list[str], target_col: str):
        """
        Initializes the Preprocessor with columns to be scaled and separate scalers.

        Args:
            lookback_cols: List of column names used in the lookback window.
            horizon_cols: List of column names used in the horizon window.
            target_col: Name of the target variable column.
            fold_idx: Index of the current validation fold for file naming.
        """
        self.lookback_cols = lookback_cols
        self.horizon_cols = horizon_cols
        self.target_col = target_col
        
        self.lookback_cols_to_scale = [col for col in self.lookback_cols if not (col.startswith("sin_") or col.startswith("cos_"))]
        self.horizon_cols_to_scale = [col for col in self.horizon_cols if not (col.startswith("sin_") or col.startswith("cos_"))]
        
        self.lookback_scaler = StandardScaler()
        self.horizon_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    @property
    def target_std(self):
        """
        Returns the scaling factor (standard deviation) for the target.
        """
        if not hasattr(self.target_scaler, "scale_"):
            raise ValueError("Scaler has not been fitted yet. Run process_fold first.")
        return self.target_scaler.scale_[0]

    @property
    def scaled_zero(self):
        """
        Returns the value of 0.0 in the target variable space mapped to the scaled space.
        """
        if not hasattr(self.target_scaler, "mean_"):
            raise ValueError("Scaler has not been fitted yet. Run process_fold first.")
        # Z-score: (x - mean) / std
        return (0.0 - self.target_scaler.mean_[0]) / self.target_scaler.scale_[0]
    
    def fit_and_transform(self,train_df: pd.DataFrame,val_df: pd.DataFrame,test_df: pd.DataFrame = None,) -> tuple[pd.DataFrame, ...]:
        """
        Fits scalers exclusively on training data and transforms the
        training, validation, and optionally test DataFrames.

        Returns:
            If test_df is provided:
                (train_scaled, val_scaled, test_scaled)
            Otherwise:
                (train_scaled, val_scaled)
        """
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy() if test_df is not None else None

        # 1. Scale historical input features.
        if self.lookback_cols_to_scale:
            train_scaled[self.lookback_cols_to_scale] = (
                self.lookback_scaler.fit_transform(
                    train_df[self.lookback_cols_to_scale]
                )
            )

            val_scaled[self.lookback_cols_to_scale] = (
                self.lookback_scaler.transform(
                    val_df[self.lookback_cols_to_scale]
                )
            )

            if test_scaled is not None:
                test_scaled[self.lookback_cols_to_scale] = (
                    self.lookback_scaler.transform(
                        test_df[self.lookback_cols_to_scale]
                    )
                )

        # 2. Scale known future features.
        # Always transform the ORIGINAL DataFrames, not already-scaled values.
        if self.horizon_cols_to_scale:
            train_scaled[self.horizon_cols_to_scale] = (
                self.horizon_scaler.fit_transform(
                    train_df[self.horizon_cols_to_scale]
                )
            )

            val_scaled[self.horizon_cols_to_scale] = (
                self.horizon_scaler.transform(
                    val_df[self.horizon_cols_to_scale]
                )
            )

            if test_scaled is not None:
                test_scaled[self.horizon_cols_to_scale] = (
                    self.horizon_scaler.transform(
                        test_df[self.horizon_cols_to_scale]
                    )
                )

        # 3. Scale the target using training target statistics only.
        train_scaled[[self.target_col]] = (
            self.target_scaler.fit_transform(
                train_df[[self.target_col]]
            )
        )

        val_scaled[[self.target_col]] = (
            self.target_scaler.transform(
                val_df[[self.target_col]]
            )
        )

        if test_scaled is not None:
            test_scaled[[self.target_col]] = (
                self.target_scaler.transform(
                    test_df[[self.target_col]]
                )
            )

        # 4. Return two or three DataFrames depending on the input.
        if test_scaled is not None:
            return train_scaled, val_scaled, test_scaled

        return train_scaled, val_scaled

    def save_scalers(self, save_dir: str = "checkpoints/scalers"):
        """
        Saves fitted scalers and metadata to disk using joblib.

        Args:
            save_dir: Directory where scaler files will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.lookback_scaler, f"{save_dir}/lookback_scaler_fold_{self.fold_idx}.pkl")
        joblib.dump(self.horizon_scaler, f"{save_dir}/horizon_scaler_fold_{self.fold_idx}.pkl")
        joblib.dump(self.target_scaler, f"{save_dir}/target_scaler_fold_{self.fold_idx}.pkl")
        
        joblib.dump(self.lookback_cols_to_scale, f"{save_dir}/scaled_lookback_cols_fold_{self.fold_idx}.pkl")
        joblib.dump(self.horizon_cols_to_scale, f"{save_dir}/scaled_horizon_cols_fold_{self.fold_idx}.pkl")

        print("Scalers were successfully saved!")
    
    def load_scalers(self, save_dir: str = "checkpoints/scalers"):
        """
        Loads fitted scalers and metadata from disk to restore state.

        Args:
            save_dir: Directory where scaler files are stored.

        Raises:
            FileNotFoundError: If any of the required scaler or metadata files are missing.
        """
        lookback_scaler_path = f"{save_dir}/lookback_scaler_fold_{self.fold_idx}.pkl"
        horizon_scaler_path = f"{save_dir}/horizon_scaler_fold_{self.fold_idx}.pkl"
        target_scaler_path = f"{save_dir}/target_scaler_fold_{self.fold_idx}.pkl"

        lookback_cols_to_scale_path = f"{save_dir}/scaled_lookback_cols_fold_{self.fold_idx}.pkl"
        horizon_cols_to_scale_path = f"{save_dir}/scaled_horizon_cols_fold_{self.fold_idx}.pkl"

        if not (os.path.exists(lookback_scaler_path) and os.path.exists(horizon_scaler_path) and os.path.exists(target_scaler_path)):
            raise FileNotFoundError(f"No scaler files found for fold {self.fold_idx} in the folder: {save_dir}!")
        
        if not (os.path.exists(lookback_cols_to_scale_path) and os.path.exists(horizon_cols_to_scale_path)):
            raise FileNotFoundError(f"No lookback or horizon cols files found for fold {self.fold_idx} in the folder: {save_dir}!")

        self.lookback_scaler = joblib.load(lookback_scaler_path)
        self.horizon_scaler = joblib.load(horizon_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)

        self.lookback_cols_to_scale = joblib.load(lookback_cols_to_scale_path)
        self.horizon_cols_to_scale = joblib.load(horizon_cols_to_scale_path)

        print(f"Scalers for fold {self.fold_idx} were successfully loaded from disk.")
    
    def transform(self, lookback_df: pd.DataFrame, horizon_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transforms input DataFrames using pre-fitted scalers for inference.

        Args:
            lookback_df: Raw lookback DataFrame.
            horizon_df: Raw horizon DataFrame.

        Returns:
            A tuple of (lookback_df_scaled, horizon_df_scaled).
        """
        lookback_df_scaled = lookback_df.copy()
        horizon_df_scaled = horizon_df.copy()

        if self.lookback_cols_to_scale:
            lookback_df_scaled[self.lookback_cols_to_scale] = self.lookback_scaler.transform(lookback_df[self.lookback_cols_to_scale])
        
        if self.horizon_cols_to_scale:
            horizon_df_scaled[self.horizon_cols_to_scale] = self.horizon_scaler.transform(horizon_df[self.horizon_cols_to_scale])
            
        return lookback_df_scaled, horizon_df_scaled
    
    def inverse_transform_target(self, y_scaled: np.array):
        """
        Inverse transforms scaled target predictions back to original physical units.

        Args:
            y_scaled: Predictions as numpy array with shape (batch, seq, target).

        Returns:
            Inverse transformed data in the original input shape.
        """
        orig_shape = y_scaled.shape
        
        y_flat = y_scaled.reshape(-1, 1)
        y_inv_flat = self.target_scaler.inverse_transform(y_flat)
        y_inverse = y_inv_flat.reshape(orig_shape)

        return y_inverse