import torch
import torch.nn as nn


class TemporalFusionTransformer(nn.Module):

    def __init__(self,
                 input_dim=7,
                 static_dim=4,
                 hidden_dim=64,
                 num_heads=4,
                 output_dim=1):

        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.static_projection = nn.Linear(static_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, time_series_data, static_metadata):

        x = self.input_projection(time_series_data)

        static_emb = self.static_projection(static_metadata)
        static_emb = static_emb.unsqueeze(1)

        x = x + static_emb

        x, _ = self.lstm(x)

        attn_output, _ = self.attention(x, x, x)

        out = self.gate(attn_output[:, -1, :])
        return out


def get_model(device):
    model = TemporalFusionTransformer().to(device)
    print(f"[DEVICE] Model loaded on: {device}")
    return model
