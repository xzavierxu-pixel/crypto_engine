from __future__ import annotations

import torch
from torch import nn


class MultiScaleResidualConvBlock(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        kernel_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty.")
        branch_dim = max(1, hidden_dim // len(kernel_sizes))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, branch_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(branch_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in kernel_sizes
            ]
        )
        fused_dim = branch_dim * len(kernel_sizes)
        self.fuse = nn.Sequential(
            nn.Conv1d(fused_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, feature_dim]
        channels_first = x.transpose(1, 2)
        branches = [branch(channels_first) for branch in self.branches]
        min_length = min(branch.shape[-1] for branch in branches)
        branches = [branch[..., :min_length] for branch in branches]
        fused = self.fuse(torch.cat(branches, dim=1))
        residual = self.residual(channels_first)[..., : fused.shape[-1]]
        return self.activation(fused + residual).transpose(1, 2)


class MRCLSTMClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        cnn_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.2,
        bidirectional: bool = False,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cnn = MultiScaleResidualConvBlock(
            input_dim=input_dim,
            hidden_dim=cnn_hidden_dim,
            kernel_sizes=kernel_sizes or [3, 5, 10, 30, 60],
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            input_size=cnn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.embedding = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor, *, return_embedding: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.cnn(x)
        _, (hidden, _) = self.lstm(encoded)
        if self.lstm.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]
        embedding = self.embedding(final_hidden)
        logit = self.classifier(embedding).squeeze(-1)
        if return_embedding:
            return logit, embedding
        return logit
