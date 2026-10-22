
import pandas as pd
import numpy as np

from trains.train_lstm_future import load_dataset
from src.preprocessing import Preprocessor
from src.dataset import Dataset
from trains.train_lstm_future import parser

args = parser.parse_args([
    "--exp_name", "diagnostic",
])

train_df, val_df, test_df = load_dataset(args.dataset_path)

preprocessor = Preprocessor(
    lookback_cols=args.lookback_cols,
    horizon_cols=args.horizon_cols,
    target_col=args.target_col,
)

train_scaled, val_scaled, test_scaled = preprocessor.fit_and_transform(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
)

splits = {
    "train": (train_scaled, train_df),
    "validation": (
        pd.concat([train_scaled.tail(args.lookback), val_scaled]),
        val_df,
    ),
    "test": (
        pd.concat([val_scaled.tail(args.lookback), test_scaled]),
        test_df,
    ),
}

used_cols = list(set(
    args.lookback_cols + args.horizon_cols + [args.target_col]
))
window_size = args.lookback + args.horizon

for name, (data, raw_split) in splits.items():
    candidate_windows = max(0, len(data) - window_size + 1)

    dataset = Dataset(
        data=data,
        lookback=args.lookback,
        horizon=args.horizon,
        lookback_cols=args.lookback_cols,
        horizon_cols=args.horizon_cols,
        target_col=args.target_col,
    )

    print(f"\n{'=' * 60}")
    print(f"{name.upper()}")
    print(f"Rows including context: {len(data)}")
    print(f"Candidate windows:      {candidate_windows}")
    print(f"Valid windows:          {len(dataset)}")
    print(
        f"Rejected windows:       "
        f"{candidate_windows - len(dataset)} "
        f"({100 * (1 - len(dataset) / max(candidate_windows, 1)):.1f} %)"
    )

    print("\nMissing values by model column:")
    print(
        raw_split[used_cols]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nMean and standard deviation:")
    print(
        raw_split[used_cols]
        .agg(["mean", "std"])
        .T
        .to_string()
    )
