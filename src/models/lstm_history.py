import torch
import argparse

class Model(torch.nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.is_bidirectional = args.is_bidirectional
        self.num_layers = getattr(args, "num_layers", 1)

        self.lstm = torch.nn.LSTM(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.is_bidirectional,
            dropout=(args.dropout if self.num_layers > 1 else 0.0),
        )

        self.dropout = torch.nn.Dropout(args.dropout)

        self.linear = torch.nn.LazyLinear(args.output_size)

    def forward(self, x_past, *args, **kwargs):
        output, _ = self.lstm(x_past)

        context = output.flatten(start_dim=1)

        context = self.dropout(context)

        return self.linear(context)

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# NEW BEST TRIAL: 19
# Validation active MAE: 119.7748
# Hyperparameters:
#   epochs: 20
#   patience: 10
#   learning_rate: 0.0002701450549289397
#   weight_decay: 0.005865378697389813
#   hidden_size: 128
#   dropout: 0.00758287426415688
#   is_bidirectional: True
#   lookback: 24
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★