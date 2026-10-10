"""Fifteen feature vectors in, one prediction out.

    (B, 15, 1280)  what the encoder produced, in time order
    (B, 128)       the GRU's running summary after the last frame
    (B, 15)        one raw score per shot class

The GRU walks the frames in order, keeping a running summary and learning what
to keep and what to forget. Only the **last** hidden state is used, because by
then it has seen the whole shot -- a Pull and a Hook end up in similar
positions and differ in how they got there.

Shape of the stack is the authors': GRU(128) -> BatchNorm -> Dense(1024, ReLU)
-> Dense(num_classes). **No dropout**, because they have none and this is the
reproduction before it is an improvement.

The GRU is initialised the way Keras does it, not the way PyTorch does --
see `_init_gru_like_keras`. With a 62720-wide input the difference is nine
times the weight scale and the difference between a GRU that learns and one
whose gates are saturated from the first step.

Returns logits, not probabilities. `CrossEntropyLoss` wants logits: it does
softmax and log together in one numerically stable step, where doing them
separately loses precision on confident predictions.
"""
import torch
import torch.nn as nn

__all__ = ["GRUHead", "NUM_CLASSES", "HIDDEN", "DENSE"]


def _init_gru_like_keras(gru: nn.GRU) -> None:
    """Glorot on the input weights, orthogonal on the recurrent, zero bias.

    PyTorch initialises every GRU weight as uniform(+-1/sqrt(hidden)),
    **ignoring how wide the input is**. That is fine at 1280 inputs and badly
    wrong at 62720: measured at init, the gate pre-activations come out with
    std 12.78 and **64% of the gates are saturated**, where the sigmoid
    gradient is about zero. Two thirds of the GRU starts dead.

    Keras uses glorot_uniform, which divides by sqrt(fan_in + fan_out), giving
    +-0.0098 instead of +-0.0884 -- nine times smaller at this width. Since
    the authors trained in Keras, this is both the faithful choice and the
    correct one.
    """
    for name, param in gru.named_parameters():
        if name.startswith("weight_ih"):
            nn.init.xavier_uniform_(param)
        elif name.startswith("weight_hh"):
            # (3 * hidden, hidden): one orthogonal block per gate
            hidden = gru.hidden_size
            for i in range(0, param.shape[0], hidden):
                nn.init.orthogonal_(param[i:i + hidden])
        elif name.startswith("bias"):
            nn.init.zeros_(param)

NUM_CLASSES = 15
HIDDEN = 128
DENSE = 1024


class GRUHead(nn.Module):
    """(B, T, in_dim) -> (B, num_classes) logits."""

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES,
                 hidden: int = HIDDEN, dense: int = DENSE,
                 dropout: float = 0.0, keras_init: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden = hidden

        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        if keras_init:
            _init_gru_like_keras(self.gru)
        self.norm = nn.BatchNorm1d(hidden)
        self.fc1 = nn.Linear(hidden, dense)
        self.relu = nn.ReLU()
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()
        self.fc2 = nn.Linear(dense, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
        if x.shape[2] != self.in_dim:
            raise ValueError(
                f"head was built for {self.in_dim} features, got {x.shape[2]}")

        # h is (layers, B, hidden); one layer, so take it and drop that axis
        _, h = self.gru(x)
        summary = h[-1]

        out = self.norm(summary)
        out = self.relu(self.fc1(out))
        out = self.drop(out)
        return self.fc2(out)
