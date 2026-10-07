"""
Baseline temporal head: GRU -> BN -> FC1 -> ReLU -> (optional Dropout) -> FC2.
Architecture from the original CricShotNet paper
(EfficientNetV2-S + GRU-128 + Dense-1024), reproduced here for the
Day-2 baseline experiment.

AUTHOR MIMIC NOTE:
The original Keras notebook has NO Dropout anywhere in this head:
    GRU(128) -> BatchNorm -> Dense(1024, relu) -> Dense(15, softmax)
The `dropout` param defaults to 0.0 when mimic_author=True (via build_model),
and 0.5 for all improved variants (adds regularization).
"""
import torch.nn as nn


class GRUHead(nn.Module):
    """
    Input:  (B, T, in_dim)  — sequence of frame features from FrameEncoder
    Output: (B, num_classes)

    Args:
        dropout: 0.0 = no dropout (matches author exactly)
                 0.5 = regularized version for improved models
    """

    def __init__(
        self,
        in_dim:      int   = 1280,
        hidden:      int   = 128,
        num_classes: int   = 15,
        dropout:     float = 0.5,   # set 0.0 for exact author mimic
    ):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, batch_first=True)
        self.bn   = nn.BatchNorm1d(hidden)
        self.fc1  = nn.Linear(hidden, 1024)
        self.relu = nn.ReLU()
        # nn.Identity() when dropout=0.0 so forward code stays identical
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc2  = nn.Linear(1024, num_classes)

    def forward(self, x):
        _, h = self.gru(x)   # h: (1, B, hidden)
        h = h.squeeze(0)     # (B, hidden)
        h = self.bn(h)
        h = self.relu(self.fc1(h))
        h = self.drop(h)
        return self.fc2(h)   # (B, num_classes)
