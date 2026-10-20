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

    def forward(self, x_past, x_future, *args, **kwargs):
        x = torch.cat([x_past, x_future], axis=-1)
        output, _ = self.lstm(x)

        context = output.flatten(start_dim=1)

        context = self.dropout(context)

        return self.linear(context)

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# NEW BEST TRIAL: 19
# Validation active MAE: 74.4433
# Hyperparameters:
#   epochs: 10
#   patience: 7
#   learning_rate: 0.0005456724926154012
#   weight_decay: 0.000634110719440628
#   hidden_size: 64
#   dropout: 0.02611872327247608
#   is_bidirectional: True
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★