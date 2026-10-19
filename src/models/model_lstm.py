# src/models/model_lstm.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers:int = 1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            batch_first=True,
            num_layers=num_layers,
            bidirectional=True
        )
        
        self.relu = nn.ReLU()
        # self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.linear = nn.LazyLinear(out_features=output_size)
    
    def forward(self, x_past, x_predictions):
        x = torch.cat([x_past, x_predictions], dim=-1)
        
        _, (h_n, _) = self.lstm(x)
        # x = h_n[0]

        # h, _ = self.lstm(x)

        # x = h.flatten(start_dim=1)

        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        x = torch.cat([forward_hidden, backward_hidden], dim=1)

        out = self.linear(x)
        
        return out