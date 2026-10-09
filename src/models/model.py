"""Encoder and head, wired together.

Deliberately thin. The two halves stay separate modules because Phase 3 swaps
one or the other, never both, and `ShotModel` only has to hold them and pass
the tensor along:

    (B, 15, 3, 224, 224) -> encoder -> (B, 15, D) -> head -> (B, num_classes)
"""
import torch.nn as nn

from src.models.encoder import DEFAULT_BACKBONE, DEFAULT_SIZE, FrameEncoder
from src.models.head import DENSE, HIDDEN, NUM_CLASSES, GRUHead

__all__ = ["ShotModel", "build_model"]


class ShotModel(nn.Module):
    """A frame encoder followed by a temporal head."""

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        return self.head(self.encoder(x))


def build_model(num_classes: int = NUM_CLASSES,
                backbone: str = DEFAULT_BACKBONE, pretrained: bool = True,
                freeze: bool = False, hidden: int = HIDDEN,
                dense: int = DENSE, dropout: float = 0.0,
                pool: str = "avg", size: int = DEFAULT_SIZE) -> ShotModel:
    """The default stack. The head is sized from the encoder, never hardcoded.

    `pool="flatten"` reproduces the authors: the GRU sees the whole 7x7 grid
    instead of a pooled summary, which is 62720 features rather than 1280.
    """
    encoder = FrameEncoder(backbone, pretrained, freeze, pool, size)
    head = GRUHead(encoder.out_dim, num_classes, hidden, dense, dropout)
    return ShotModel(encoder, head)
