import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size_past: int, input_size_future: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        self.total_input_channels = input_size_past + input_size_future
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.total_input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=hidden_size, 
            batch_first=True,
            num_layers=num_layers,
            bidirectional=True
        )
        
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x_past, x_predictions):
        # x_past: (batch, seq_len, 12)
        # x_predictions: (batch, seq_len, 10)
        
        x = torch.cat([x_past, x_predictions], dim=-1)
        
        x = x.transpose(1, 2)
    
        x = self.conv(x)
        
        x = x.transpose(1, 2)
        
        _, (h_n, _) = self.lstm(x)
        
        # Zpracování bidirectional výstupu
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        x = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return self.linear(x)