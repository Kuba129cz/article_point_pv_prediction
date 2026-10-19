# model_attention.py
import torch
import argparse

class Attention(torch.nn.Module):
    "A class adding Bahdanau Attention to a given RNN cell"
    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell
        self._project_encoder_layer = torch.nn.Linear(cell.hidden_size,  attention_dim)
        self._project_decoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)
        self._output_layer = torch.nn.Linear(attention_dim, 1)
    
    def setup_memory(self, encoded):
        self._encoded = encoded
        self._encoded_projected = self._project_encoder_layer(encoded)
    
    def forward(self, inputs, states):
        h, c = states
        projected_decoder = self._project_decoder_layer(h)
        sum_projection = self._encoded_projected + projected_decoder.unsqueeze(dim=1)
        logits = self._output_layer(torch.tanh(sum_projection)).squeeze(dim=-1) # shape = (batch_size, seq_len, 1) => (batch_size, seq_len)
        alphas = torch.nn.functional.softmax(logits, dim=-1) # (batch_size, seq_len)
        attention = torch.bmm(alphas.unsqueeze(dim=1), self._encoded).squeeze(dim=1) # contex vector
        inputs = torch.cat([inputs, attention], dim=-1)

        return self._cell(inputs, (h, c))
    
class HistoryEncoder(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, cnn_filters: int, cnn_kernel: int, dropout: float):
        super().__init__()
        self._conv = torch.nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=cnn_kernel, padding="same", bias=False)
        self._bnm = torch.nn.BatchNorm1d(num_features=cnn_filters)

        self._relu = torch.nn.ReLU()

        self._lstm = torch.nn.LSTM(input_size=cnn_filters, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self._dropout = torch.nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = x.transpose(1, 2)

        x = self._conv(x)
        x = self._bnm(x)
        x = self._relu(x)

        x = x.transpose(1, 2)

        h, _ = self._lstm(x)
        h = self._dropout(h)
        
        return h

class FutureEncoder(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, cnn_filters: int, cnn_kernels: list[int], dropout: float):
        super().__init__()

        self._relu = torch.nn.ReLU()
        self._cnns = torch.nn.ModuleList()
        self._bnms = torch.nn.ModuleList()
        
        current_in = input_size
        for kernel in cnn_kernels:
            self._cnns.append(
                torch.nn.Conv1d(
                    in_channels=current_in, 
                    out_channels=cnn_filters, 
                    kernel_size=kernel, 
                    padding="same", 
                    bias=False
                )
            )
            self._bnms.append(torch.nn.BatchNorm1d(num_features=cnn_filters))
            current_in = cnn_filters


        self._lstm = torch.nn.LSTM(input_size=cnn_filters, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self._dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x_future: torch.Tensor) -> torch.Tensor:
        x = x_future.transpose(1, 2)

        for i, (cnn, bn) in enumerate(zip(self._cnns, self._bnms)):
            if i == 0:
                x = self._relu(bn(cnn(x)))
            else:
                identity = x
                
                x = cnn(x)
                x = bn(x)
                x = self._relu(x)
                
                x = x + identity

        x = x.transpose(1, 2)
        
        encoded_future, _ = self._lstm(x)
        encoded_future = self._dropout(encoded_future)
        
        return encoded_future
    
class Decoder(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, attention_dim: int, meteo_features: int, dropout: float, target_features: int = 1):
        super().__init__()
        
        self._target_rnn_cell = Attention(
            cell=torch.nn.LSTMCell(
                input_size=input_size + hidden_size,
                hidden_size=hidden_size
            ), 
            attention_dim=attention_dim
        )
        self._dropout = torch.nn.Dropout(p=dropout)
        self._target_output_layer = torch.nn.Linear(in_features=hidden_size + meteo_features, out_features=target_features)
    
    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor, encoded_future: torch.Tensor, previous_power: torch.Tensor, curr_epoch: int, total_epochs: int) -> torch.Tensor:
        self._target_rnn_cell.setup_memory(encoded)

        h_0 = encoded.mean(dim=1)
        c_0 = torch.zeros_like(h_0)
        states = (h_0, c_0)

        meteo = encoded_future[:, 0, :]
        input = torch.cat([previous_power, meteo], dim=-1)

        total_epochs = total_epochs if total_epochs > 0 else 1
        epsilon = max(0.0, 1.0 - (curr_epoch / total_epochs))

        all_predictions = []
        horizon = targets.shape[1]

        for i in range(horizon):
            states = self._target_rnn_cell(input, states)
            h_state = states[0]
            
            combined = torch.cat([h_state, meteo], dim=-1)
            combined = self._dropout(combined)
            prediction = self._target_output_layer(combined)
            all_predictions.append(prediction)

            if i < horizon - 1:
                if torch.rand(1).item() < epsilon:
                    real_fve = targets[:, i : i + 1]
                else:
                    real_fve = prediction.detach() 
                  
                meteo = encoded_future[:, i + 1, :] 
                input = torch.cat([real_fve, meteo], dim=-1)

        outputs = torch.stack(all_predictions, dim=1)
        return outputs

    def decoder_prediction(self, encoded: torch.Tensor, encoded_future: torch.Tensor, previous_power: torch.Tensor) -> torch.Tensor:
        self._target_rnn_cell.setup_memory(encoded)
        
        h_0 = encoded.mean(dim=1)
        c_0 = torch.zeros_like(h_0)
        states = (h_0, c_0)

        meteo_now = encoded_future[:, 0, :]
        input = torch.cat([previous_power, meteo_now], dim=-1)

        all_predictions = []
        horizon = encoded_future.shape[1]

        for i in range(horizon):
            states = self._target_rnn_cell(input, states)
            h_state = states[0]
            
            combined = torch.cat([h_state, meteo_now], dim=-1)
            combined = self._dropout(combined)
            predictions = self._target_output_layer(combined)
            all_predictions.append(predictions)

            if i < horizon - 1:
                own_predicted_fve = predictions 
                meteo_now = encoded_future[:, i + 1, :] # Update obohacených meteo dat
                input = torch.cat([own_predicted_fve, meteo_now], dim=-1)

        outputs = torch.stack(all_predictions, dim=1)
        return outputs
    
class Model(torch.nn.Module): 
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        past_features_count = len(args.lookback_cols)
        future_features_count = len(args.horizon_cols)

        self.encoder = HistoryEncoder(
            input_size=past_features_count,
            hidden_size=args.past_hidden_size,
            cnn_filters=args.past_cnn_filters,
            cnn_kernel=args.past_kernel,
            dropout=args.past_dropout
        )
        
        future_kernels = []
        j = 0
        while hasattr(args, f"future_kernel_L{j}"):
            future_kernels.append(getattr(args, f"future_kernel_L{j}"))
            j += 1

        self.future_encoder = FutureEncoder(
            input_size=future_features_count,
            hidden_size=args.future_hidden_size,
            cnn_filters=args.future_cnn_filters, 
            cnn_kernels=future_kernels,     
            dropout=args.future_dropout
        )

        encoded_future_size = args.future_hidden_size * 2
        decoder_input_size = 1 + encoded_future_size

        self.decoder = Decoder(
            input_size=decoder_input_size,
            hidden_size=args.past_hidden_size * 2, # It must match the output from historyEncoder
            attention_dim=args.attention_dim,
            meteo_features=encoded_future_size,
            dropout=args.decoder_dropout,
            target_features=1
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor, targets: torch.Tensor | None = None, curr_epoch: int = 0) -> torch.Tensor:
        energy_idx = self._args.lookback_cols.index("energy")
        
        encoded = self.encoder(x_past)
        previous_power = x_past[:, -1, energy_idx : energy_idx + 1]
        
        encoded_future = self.future_encoder(x_future)
        
        if targets is not None:
            return self.decoder.decoder_training(
                encoded, targets, encoded_future, previous_power, 
                total_epochs=self._args.epochs, curr_epoch=curr_epoch
            )
        else:
            return self.decoder.decoder_prediction(encoded, encoded_future, previous_power)