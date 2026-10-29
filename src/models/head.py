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

**BatchNorm uses their epsilon and momentum, not torch's.** With 62720
non-centred inputs the GRU saturates whatever the backbone does -- measured at
41% of units pinned by epoch 1 even with the backbone frozen -- and a pinned
unit has zero variance, so BatchNorm's epsilon is the only thing standing
between it and a division by nothing. Theirs is 1e-3, torch's is 1e-5: 31.6x
amplification instead of 316x. On the same epoch-5 weights, that one number
took val loss from 3.1031 to 1.4819 and top-1 from 55.11% to 56.50%.

Returns logits, not probabilities. `CrossEntropyLoss` wants logits: it does
softmax and log together in one numerically stable step, where doing them
separately loses precision on confident predictions.
"""
import torch
import torch.nn as nn

__all__ = ["GRUHead", "LSTMHead", "LinearHead", "MLPHead", "PoolHead",
           "TransformerHead", "TCNHead", "AttnPoolHead", "NetVLADHead",
           "STGCNHead", "FrameNorm", "pool_frames", "POOLS", "WIDTH", "FRAMES",
           "CLUSTERS", "NUM_CLASSES", "HIDDEN", "DENSE"]


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


def _init_lstm_like_keras(lstm: nn.LSTM) -> None:
    """The GRU's init, plus Keras's forget-gate bias of 1.

    Torch's LSTM gates are stacked (input, forget, cell, output), so the
    forget gate is the second block of `hidden` rows. Keras starts it at 1
    so the cell remembers by default; torch has two biases, so one gets
    the 1 and the other stays 0.
    """
    _init_gru_like_keras(lstm)          # works for any gate count
    hidden = lstm.hidden_size
    for name, param in lstm.named_parameters():
        if name.startswith("bias_ih"):
            with torch.no_grad():
                param[hidden:2 * hidden] = 1.0

NUM_CLASSES = 15
HIDDEN = 128
DENSE = 1024

# Both read out of the authors' .keras config, and both differ from PyTorch's
# defaults. Keras and torch define momentum inversely: Keras keeps `momentum`
# of the old value, torch takes `momentum` of the new one, so their 0.99 is
# 0.01 here. See `_authors_batchnorm` in the docstring.
BN_EPS = 1e-3            # theirs: epsilon 0.001. Torch defaults to 1e-5.
BN_MOMENTUM = 0.01       # theirs: momentum 0.99. Torch defaults to 0.1.


class GRUHead(nn.Module):
    """(B, T, in_dim) -> (B, num_classes) logits."""

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES,
                 hidden: int = HIDDEN, dense: int = DENSE,
                 dropout: float = 0.0, keras_init: bool = True,
                 bn_eps: float = BN_EPS, bn_momentum: float = BN_MOMENTUM,
                 bidirectional: bool = False, keras_dense: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden = hidden
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1

        # one direction builds exactly what it always did, in the same
        # order, so A03 still draws the same random weights as cached_gru
        self.gru = nn.GRU(in_dim, hidden, batch_first=True,
                          bidirectional=bidirectional)
        if keras_init:
            _init_gru_like_keras(self.gru)
        width = hidden * self.directions
        self.norm = nn.BatchNorm1d(width, eps=bn_eps, momentum=bn_momentum)
        self.fc1 = nn.Linear(width, dense)
        self.relu = nn.ReLU()
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()
        self.fc2 = nn.Linear(dense, num_classes)
        if keras_dense:
            # Keras' Dense: glorot uniform, zero bias. Opt-in, so every
            # earlier config still draws the weights it always drew
            for layer in (self.fc1, self.fc2):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def pinned_fraction(self):
        """How much of the GRU is stuck at tanh's rail. Free to read.

        A hidden unit that saturates on every clip has a running mean of
        +-1 and a running variance of 0, and BatchNorm then divides by
        sqrt(0 + eps) = 316x at eval time -- which is how epoch 5 of the
        first good run scored 55% top-1 with a loss of 3.10. The stats are
        already there, so this costs nothing and catches it by epoch 2.

        Returns None when there is nothing to read it off.
        """
        mean = getattr(self.norm, "running_mean", None)
        if mean is None:
            return None
        return (mean.abs() > 0.99).float().mean().item()

    def dead_fraction(self):
        """Fraction of hidden units that are constant across every clip."""
        var = getattr(self.norm, "running_var", None)
        if var is None:
            return None
        return (var < 1e-4).float().mean().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
        if x.shape[2] != self.in_dim:
            raise ValueError(
                f"head was built for {self.in_dim} features, got {x.shape[2]}")

        # h is (directions, B, hidden): forwards first, then backwards
        _, h = self.gru(x)
        if self.directions == 1:
            summary = h[-1]
        else:
            summary = torch.cat([h[-2], h[-1]], dim=1)

        out = self.norm(summary)
        out = self.relu(self.fc1(out))
        out = self.drop(out)
        return self.fc2(out)


def _one_vector(x: torch.Tensor) -> torch.Tensor:
    """(B, T, D) averaged over time, or (B, D) as it is."""
    if x.dim() == 3:
        return x.mean(dim=1)
    if x.dim() == 2:
        return x
    raise ValueError(f"expected (B, T, D) or (B, D), got {tuple(x.shape)}")


class LinearHead(nn.Module):
    """One Linear on the clip's mean feature: the probe."""

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_dim = in_dim
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(_one_vector(x))


class MLPHead(nn.Module):
    """Linear -> ReLU -> Linear on the clip's mean feature. No time model."""

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES,
                 dense: int = DENSE, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, dense)
        self.relu = nn.ReLU()
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()
        self.fc2 = nn.Linear(dense, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.fc1(_one_vector(x)))
        return self.fc2(self.drop(out))


POOLS = ("mean", "max")

WIDTH = 256         # every new head shrinks a frame to this first
FRAMES = 15         # the most frames a clip has
CLUSTERS = 8        # NetVLAD's learned centres


def pool_frames(x: torch.Tensor, how: str) -> torch.Tensor:
    """(B, T, D) -> (B, D): the average, or the largest value, over frames."""
    if x.dim() != 3:
        raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
    if how == "mean":
        return x.mean(dim=1)
    if how == "max":
        return x.max(dim=1).values     # .max also returns where; drop it
    raise ValueError(f"unknown pool {how!r}; have {POOLS}")


def _check_frames(x: torch.Tensor, in_dim: int) -> None:
    if x.dim() != 3:
        raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
    if x.shape[2] != in_dim:
        raise ValueError(
            f"head was built for {in_dim} features, got {x.shape[2]}")


class _Tail(nn.Module):
    """GRUHead's ending, shared: BatchNorm -> Dense -> ReLU -> Dense.

    Every Family A head inherits it, so the only thing that differs between
    them is how the frames become one summary vector.
    """

    def _build_tail(self, width, num_classes, dense, dropout):
        self.norm = nn.BatchNorm1d(width, eps=BN_EPS, momentum=BN_MOMENTUM)
        self.fc1 = nn.Linear(width, dense)
        self.relu = nn.ReLU()
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()
        self.fc2 = nn.Linear(dense, num_classes)

    def _tail(self, summary: torch.Tensor) -> torch.Tensor:
        out = self.norm(summary)
        out = self.relu(self.fc1(out))
        out = self.drop(out)
        return self.fc2(out)


class PoolHead(_Tail):
    """GRUHead with the GRU swapped for a pool. Order-blind on purpose."""

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES,
                 how: str = "mean", dense: int = DENSE,
                 dropout: float = 0.0):
        super().__init__()
        if how not in POOLS:
            raise ValueError(f"unknown pool {how!r}; have {POOLS}")
        self.in_dim = in_dim
        self.how = how
        self._build_tail(in_dim, num_classes, dense, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_frames(x, self.in_dim)
        return self._tail(pool_frames(x, self.how))


class LSTMHead(_Tail):
    """GRUHead with an LSTM, initialised the Keras way too."""

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES,
                 hidden: int = HIDDEN, dense: int = DENSE,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True,
                            bidirectional=bidirectional)
        _init_lstm_like_keras(self.lstm)
        self._build_tail(hidden * self.directions, num_classes, dense, dropout)

    # the LSTM's output is o * tanh(c), so it pins at +-1 like the GRU's
    pinned_fraction = GRUHead.pinned_fraction
    dead_fraction = GRUHead.dead_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_frames(x, self.in_dim)
        _, (h, _) = self.lstm(x)        # the cell state c is not needed
        if self.directions == 1:
            summary = h[-1]
        else:
            summary = torch.cat([h[-2], h[-1]], dim=1)
        return self._tail(summary)


class TransformerHead(_Tail):
    """Shrink each frame, add its position, let every frame attend to every
    other, then average."""

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES,
                 layers: int = 2, attn_heads: int = 4, dense: int = DENSE,
                 dropout: float = 0.0):
        super().__init__()
        if WIDTH % attn_heads:
            raise ValueError(f"attn_heads must divide {WIDTH}, got {attn_heads}")
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, WIDTH)
        # one learned vector per frame slot, or frame 1 and 15 look the same
        self.pos = nn.Parameter(torch.zeros(1, FRAMES, WIDTH))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=WIDTH, nhead=attn_heads, dim_feedforward=2 * WIDTH,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self._build_tail(WIDTH, num_classes, dense, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_frames(x, self.in_dim)
        frames = x.shape[1]
        if frames > FRAMES:
            raise ValueError(f"at most {FRAMES} frames, got {frames}")
        h = self.proj(x) + self.pos[:, :frames]
        # torch's fused eval path misses XPU autocast and mixes bf16 with
        # fp32, so it is off for this call only
        was = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            h = self.encoder(h)
        finally:
            torch.backends.mha.set_fastpath_enabled(was)
        return self._tail(h.mean(dim=1))


class _TCNBlock(nn.Module):
    """One dilated convolution over time, added back onto its input."""

    def __init__(self, dilation: int, dropout: float):
        super().__init__()
        # padding = dilation keeps the number of frames the same
        self.conv = nn.Conv1d(WIDTH, WIDTH, kernel_size=3, dilation=dilation,
                              padding=dilation)
        self.relu = nn.ReLU()
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.relu(self.conv(x)))


class TCNHead(_Tail):
    """Three dilated convolutions (1, 2, 4): together they see 15 frames."""

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES,
                 dense: int = DENSE, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, WIDTH)
        self.blocks = nn.Sequential(_TCNBlock(1, dropout), _TCNBlock(2, dropout),
                                    _TCNBlock(4, dropout))
        self._build_tail(WIDTH, num_classes, dense, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_frames(x, self.in_dim)
        h = self.proj(x)                 # (B, T, W)
        h = h.transpose(1, 2)            # Conv1d wants (B, W, T)
        h = self.blocks(h)
        return self._tail(h.mean(dim=2))


class AttnPoolHead(_Tail):
    """A learned weight per frame; the clip is the weighted average.

    The last batch's weights are kept in `last_weights`, (B, T), so a figure
    can show which frames each class leans on.
    """

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES,
                 dense: int = DENSE, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, WIDTH)
        self.score = nn.Sequential(nn.Linear(WIDTH, WIDTH // 2), nn.Tanh(),
                                   nn.Linear(WIDTH // 2, 1))
        self.last_weights = None
        self._build_tail(WIDTH, num_classes, dense, dropout)

    def weights(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, W) -> (B, T), each row summing to 1."""
        return torch.softmax(self.score(h).squeeze(2), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_frames(x, self.in_dim)
        h = self.proj(x)
        w = self.weights(h)
        self.last_weights = w.detach()
        pooled = (w.unsqueeze(2) * h).sum(dim=1)
        return self._tail(pooled)


class NetVLADHead(_Tail):
    """Describe a clip by how its frames sit around 8 learned centres."""

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES,
                 dense: int = DENSE, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, WIDTH)
        self.assign = nn.Linear(WIDTH, CLUSTERS)
        self.centres = nn.Parameter(0.1 * torch.randn(CLUSTERS, WIDTH))
        self._build_tail(CLUSTERS * WIDTH, num_classes, dense, dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, CLUSTERS * WIDTH), length 1."""
        h = self.proj(x)                                  # (B, T, W)
        a = torch.softmax(self.assign(h), dim=2)          # (B, T, K)
        # sum over frames of a * (h - c) = sum(a * h) - sum(a) * c
        weighted = torch.bmm(a.transpose(1, 2), h)        # (B, K, W)
        mass = a.sum(dim=1).unsqueeze(2)                  # (B, K, 1)
        vlad = weighted - mass * self.centres.unsqueeze(0)
        vlad = nn.functional.normalize(vlad, dim=2)       # each centre
        return nn.functional.normalize(vlad.flatten(1), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_frames(x, self.in_dim)
        return self._tail(self.encode(x))


class _GraphBlock(nn.Module):
    """Each joint takes in its neighbours', then each joint looks across
    nearby frames. A shortcut around both, as in ST-GCN."""

    def __init__(self, cin: int, cout: int, kernel: int = 5):
        super().__init__()
        self.graph = nn.Conv2d(cin, cout, 1)       # channels, per joint
        self.time = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, (kernel, 1), padding=(kernel // 2, 0)),
            nn.BatchNorm2d(cout))
        if cin == cout:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1),
                                      nn.BatchNorm2d(cout))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x is (B, C, T, J); a mixes joint j into joint k along the bones
        y = torch.einsum("bctj,jk->bctk", self.graph(x), a)
        return self.relu(self.time(y) + self.skip(x))


class STGCNHead(_Tail):
    """Joints as graph nodes, edges along the skeleton (E02).

    (B, T, J * C) -> (B, classes). Three graph blocks, then the clip is the
    average over frames and joints, then the same tail as every Family A
    head. `adjacency` is (J, J) with self-loops; it is normalised here as
    D^-1/2 A D^-1/2, so a joint with many bones does not drown the others.
    """

    def __init__(self, in_dim: int, adjacency: torch.Tensor,
                 num_classes: int = NUM_CLASSES, channels: int = 3,
                 widths=(64, 64, 128), dense: int = DENSE,
                 dropout: float = 0.0):
        super().__init__()
        joints = adjacency.shape[0]
        if in_dim != joints * channels:
            raise ValueError(f"{in_dim} features is not {joints} joints x "
                             f"{channels}")
        self.in_dim = in_dim
        self.joints = joints
        self.channels = channels
        degree = adjacency.sum(dim=1).pow(-0.5)
        self.register_buffer("adjacency",
                             degree.unsqueeze(1) * adjacency
                             * degree.unsqueeze(0))
        self.data_norm = nn.BatchNorm1d(in_dim)
        blocks = []
        cin = channels
        for cout in widths:
            blocks.append(_GraphBlock(cin, cout))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self._build_tail(cin, num_classes, dense, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_frames(x, self.in_dim)
        b, t = x.shape[:2]
        x = self.data_norm(x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(b, t, self.joints, self.channels).permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x, self.adjacency)               # (B, C, T, J)
        return self._tail(x.mean(dim=(2, 3)))


class FrameNorm(nn.Module):
    """A LayerNorm on every frame, then any head (A13).

    The inner head is kept as `.head` so the saturation watch in train.py
    still finds the GRU's numbers through it.
    """

    def __init__(self, in_dim: int, head: nn.Module):
        super().__init__()
        self.frame_norm = nn.LayerNorm(in_dim)
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.frame_norm(x))
