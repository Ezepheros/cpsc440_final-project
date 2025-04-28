import torch.nn as nn
import torch

import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_layers = 2,  dropout=0.5):
        super(MLPModel, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        layers = []
        prev_dim = seq_len * input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten (seq_len * input_dim)
        out = self.mlp(x)
        return out.squeeze()
