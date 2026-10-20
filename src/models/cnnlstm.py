
import torch
import argparse

class Model(torch.nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.is_bidirectional = args.is_bidirectional
        self.num_layers = getattr(args, "num_layers", 1)

        cnn_channels = getattr(args, "cnn_channels", args.hidden_size)
        kernel_size = getattr(args, "cnn_kernel_size", 3)

        if kernel_size % 2 == 0:
            raise ValueError("cnn_kernel_size musí být liché číslo.")

        self.cnn = torch.nn.Conv1d(
            in_channels=args.input_size,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.cnn_activation = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(
            input_size=cnn_channels,
            hidden_size=args.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.is_bidirectional,
            dropout=(
                args.dropout if self.num_layers > 1 else 0.0
            ),
        )

        self.dropout = torch.nn.Dropout(args.dropout)
        self.linear = torch.nn.LazyLinear(args.output_size)

    def forward(self, x_past, x_future, *args, **kwargs):
        x = torch.cat([x_past, x_future], dim=-1)

        x = x.transpose(1, 2)

        x = self.cnn(x)
        x = self.cnn_activation(x)

        x = x.transpose(1, 2)

        output, _ = self.lstm(x)
        context = output.flatten(start_dim=1)
        context = self.dropout(context)

        return self.linear(context)
