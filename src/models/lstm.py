import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, **kwargs):
        super(LSTMModel, self).__init__()
        self.pre_layer_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            device='cuda'  # LSTM handles dropout across layers
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.pre_layer_norm(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take output of last timestep
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.mlp(out)
        return out.squeeze()