# src/dataset.py
import pandas as pd
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for Solar Power Plant (FVE) forecasting using a sliding window.
    
    This dataset splits time-series data into past context (lookback) and future 
    context (horizon) to support multi-modal or encoder-decoder forecasting models.
    """
    def __init__(self, data: pd.DataFrame, lookback: int, horizon: int, lookback_cols: list[str], horizon_cols: list[str], target_col: str, expected_freq: str = '1h'):
        """Initializes the FVEDataset with rigorous data validation and sliding window preprocessing."""
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(
                "The provided DataFrame must have a pd.DatetimeIndex (e.g., set it using data.set_index('timestamp')). "
                "Time-series continuity cannot be guaranteed without a proper DatetimeIndex."
            )
        
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        
        if expected_freq in ['h', 'H']:
            expected_freq = '1h'
        expected_delta = pd.to_timedelta(expected_freq)
        
        deltas = data.index.to_series().diff().dropna()
        if deltas.empty:
            raise ValueError("The provided DataFrame does not contain enough rows to determine frequency.")
            
        detected_delta = deltas.mode()[0]

        if detected_delta != expected_delta:
            raise ValueError(
                f"Expected data frequency '{expected_delta}', but detected '{detected_delta}'. "
                "Please resample your data to the correct frequency."
            )
        
        if not (deltas == expected_delta).all():
            raise ValueError(
                "Data are not in regular intervals or missing time steps were detected. "
                "Please use data.asfreq() to ensure a consistent frequency before passing the data to the dataset."
            )
        
        super().__init__()
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        self.lookback_cols = lookback_cols
        self.horizon_cols = horizon_cols
        self.target_col = target_col

        used_cols = list(set(lookback_cols + horizon_cols + [target_col]))
        has_nan = data[used_cols].isna().any(axis=1)

        total_window = lookback + horizon

        valid_mask = (has_nan.rolling(window=total_window).sum() == 0)
        self.valid_indices = np.where(valid_mask)[0] - total_window + 1

    def __len__(self):
        """Calculates the total number of valid (NaN-free) samples in the dataset.

        Returns:
            int: Total number of available valid samples.
        """

        return len(self.valid_indices)

    def __getitem__(self, index):
        """Generates one valid sample from the dataset at the given PyTorch index.

        Args:
            index (int): The index of the valid sample (provided by DataLoader).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - x_past (torch.Tensor): Historical features of shape (lookback, len(lookback_cols)).
                - x_future (torch.Tensor): Known future features of shape (horizon, len(horizon_cols)).
                - y_target (torch.Tensor): Target values to predict of shape (horizon,).
        """
        start_idx = self.valid_indices[index]

        past_window = self.data.iloc[start_idx : start_idx + self.lookback]
        future_window = self.data.iloc[start_idx + self.lookback : start_idx + self.lookback + self.horizon]

        x_past = past_window[self.lookback_cols].values.astype(np.float32)
        x_future = future_window[self.horizon_cols].values.astype(np.float32)
        y_target = future_window[self.target_col].values.astype(np.float32)

        return (
            torch.from_numpy(x_past),
            torch.from_numpy(x_future),
            torch.from_numpy(y_target)
        )