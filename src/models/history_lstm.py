# src/models/lstm_history.py

import argparse
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.is_bidirectional = args.is_bidirectional
        self.num_layers = getattr(args, "num_layers", 1)

        num_directions = 2 if self.is_bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.is_bidirectional,
            dropout=(
                args.dropout if self.num_layers > 1 else 0.0
            ),
        )

        self.dropout = nn.Dropout(args.dropout)

        self.linear = nn.Linear(
            args.hidden_size * num_directions,
            args.output_size,
        )

    def forward(self, x_past, *args, **kwargs):
        _, (h_n, _) = self.lstm(x_past)

        if self.is_bidirectional:
            forward_hidden = h_n[-2]
            backward_hidden = h_n[-1]

            context = torch.cat([forward_hidden, backward_hidden], dim=-1)
        else:
            context = h_n[-1]

        context = self.dropout(context)
        return self.linear(context)