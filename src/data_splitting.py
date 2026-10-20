import pandas as pd

class Splitter:
    """
    Manages time-seriesand returns raw pandas DataFrame splits (train_df, val_df).
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_splits(self, train_ratio: float = 0.8) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates pairs of (train_df, val_df) based on the chosen strategy.
        
        Args:
            - train_ratio (float) for 'simple' (default: 0.8)
        """
        splits = []
        total_len = len(self.data)
        split_idx = int(total_len * train_ratio)
        
        train_df = self.data.iloc[:split_idx]
        val_df = self.data.iloc[split_idx:]
        splits.append((train_df, val_df))

        return splits