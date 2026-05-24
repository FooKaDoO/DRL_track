import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, state_size):
        super(DQN, self).__init__()

        if state_size != 74:
            raise ValueError(f"Group-aware DQN expects 74 input features, got {state_size}")

        self.column_encoder = nn.Sequential(
            nn.Linear(40, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
        )

        self.column_conv_encoder = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(16 * 10, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
        )

        self.row_encoder = nn.Sequential(
            nn.Linear(20, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Linear(16, 12),
            nn.SiLU(),
        )

        self.summary_encoder = nn.Sequential(
            nn.Linear(7, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Linear(16, 12),
            nn.SiLU(),
        )

        self.next_piece_encoder = nn.Sequential(
            nn.Linear(7, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.SiLU(),
        )

        # 16 + 16 + 12 + 12 + 8 = 64
        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):

        was_single = x.dim() == 1
        if was_single:
            x = x.unsqueeze(0)

        column_features = torch.cat([
            x[:, 0:10],    # fill heights
            x[:, 14:24],   # holeyness per column
            x[:, 24:34],   # vertical hole depths
            x[:, 34:44],   # vertical hole clusteredness
        ], dim=1)
        column_sequence = torch.stack([
            x[:, 0:10],    # fill heights
            x[:, 14:24],   # holeyness per column
            x[:, 24:34],   # vertical hole depths
            x[:, 34:44],   # vertical hole clusteredness
        ], dim=1)
        row_features = x[:, 44:64]
        summary_features = torch.cat([
            x[:, 10:14],
            x[:, 64:67],
        ], dim=1)
        next_piece_features = x[:, 67:74]

        x = torch.cat([
            self.column_encoder(column_features),
            self.column_conv_encoder(column_sequence),
            self.row_encoder(row_features),
            self.summary_encoder(summary_features),
            self.next_piece_encoder(next_piece_features),
        ], dim=1)

        x = self.head(x)

        if was_single:
            return x.squeeze(0)

        return x
