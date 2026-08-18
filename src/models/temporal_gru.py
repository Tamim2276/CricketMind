"""
Baseline temporal head: GRU → BN → FC1 → ReLU → FC2.
Architecture from the original CricShotNet paper
(EfficientNetV2-S + GRU-128 + Dense-1024), reproduced here for the
Day-2 baseline experiment.
"""
import torch.nn as nn


class GRUHead(nn.Module):
    """
    Input:  (B, T, in_dim)  — sequence of frame features from FrameEncoder
    Output: (B, num_classes)
    """

    def __init__(self, in_dim: int = 1280, hidden: int = 128, num_classes: int = 15):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, batch_first=True)
        self.bn   = nn.BatchNorm1d(hidden)
        self.fc1  = nn.Linear(hidden, 1024)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.fc2  = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x: (B, T, in_dim)
        _, h = self.gru(x)   # h: (1, B, hidden)
        h = h.squeeze(0)     # (B, hidden)
        h = self.bn(h)
        h = self.relu(self.fc1(h))
        h = self.drop(h)
        return self.fc2(h)   # (B, num_classes)
