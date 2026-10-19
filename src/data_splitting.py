import pandas as pd

class Splitter:
    """Manages time-series cross-validation strategies (Simple, Rolling, Expanding)
    and returns raw pandas DataFrame splits (train_df, val_df).
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_splits(self, strategy: str, **kwargs) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generates pairs of (train_df, val_df) based on the chosen strategy.
        
        Args:
            strategy (str): 'simple', 'expanding', or 'rolling'
            **kwargs: Configuration parameters for the specific strategy:
                      - train_ratio (float) for 'simple' (default: 0.8)
                      - initial_train_size (int) and step_size (int) for 'expanding'
                      - window_size (int) and step_size (int) for 'rolling'
        """
        strategy = strategy.lower().strip()
        splits = []
        total_len = len(self.data)

        if strategy == "simple":
            train_ratio = kwargs.get("train_ratio", 0.8)
            split_idx = int(total_len * train_ratio)
            
            train_df = self.data.iloc[:split_idx]
            val_df = self.data.iloc[split_idx:]
            splits.append((train_df, val_df))

        elif strategy == "expanding":
            init_size = kwargs["initial_train_size"]
            step_size = kwargs["step_size"]
            current_end = init_size

            print(f"jsem uvnitr expanding init_size={init_size} step_size={step_size} current_end={current_end} total_len={total_len}")
            
            while current_end + step_size <= total_len:
                train_df = self.data.iloc[:current_end]
                val_df = self.data.iloc[current_end : current_end + step_size]
                splits.append((train_df, val_df))
                current_end += step_size

        elif strategy == "rolling":
            window_size = kwargs["window_size"]
            step_size = kwargs["step_size"]
            current_start = 0
            
            while current_start + window_size + step_size <= total_len:
                train_end = current_start + window_size
                val_end = train_end + step_size
                
                train_df = self.data.iloc[current_start:train_end]
                val_df = self.data.iloc[train_end:val_end]
                splits.append((train_df, val_df))
                current_start += step_size
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from 'simple', 'expanding', 'rolling'.")

        return splits