from typing import Any

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # attention per each page
        )

    def forward(self, embeddings: Any, mask: Any) -> Any:
        """
        Args:
            embeddings:  (B, P, H)
            mask: (B, P)
        Returns: (B, H)
        """
        scores = self.score_layer(embeddings).squeeze(-1)  # (B, P)
        scores = scores.masked_fill(~mask, -1e9)  # mask out padded pages
        attn_weights = torch.softmax(scores, dim=1)  # (B, P)
        output = torch.sum(embeddings * attn_weights.unsqueeze(-1), dim=1)  # (B, H)
        return output
