"""cached_head: the probe's Linear, V's MLP and the authors' GRU, one file."""
import pytest
import torch

from src.methods import cached_head as M
from src.models.head import GRUHead, LinearHead, MLPHead
from src.train import DEFAULTS


class _Data:
    def __init__(self, dim):
        self.feature_dim = dim


def _cfg(**over):
    cfg = dict(DEFAULTS)
    cfg.update({"method": "cached_head", "view": "avg"})
    cfg.update(over)
    return cfg


def test_linear_averages_frames_and_takes_vectors_as_they_are():
    head = LinearHead(8, 3)
    frames = torch.randn(2, 5, 8)
    assert head(frames).shape == (2, 3)
    assert torch.allclose(head(frames), head(frames.mean(dim=1)))
    assert head(torch.randn(2, 8)).shape == (2, 3)


def test_mlp_has_one_hidden_layer_and_no_time_model():
    head = MLPHead(8, 3, dense=16)
    assert head(torch.randn(2, 5, 8)).shape == (2, 3)
    assert head(torch.randn(2, 8)).shape == (2, 3)
    assert head.fc1.out_features == 16


def test_a_4d_input_is_refused():
    with pytest.raises(ValueError):
        LinearHead(8, 3)(torch.randn(2, 5, 8, 1))


def test_the_config_picks_the_head():
    assert isinstance(M.model(_cfg(head="linear"), _Data(8), 3), LinearHead)
    assert isinstance(M.model(_cfg(head="mlp"), _Data(8), 3), MLPHead)
    assert isinstance(M.model(_cfg(head="gru"), _Data(8), 3), GRUHead)


def test_heads_not_built_yet_say_when():
    assert M.check(_cfg(head="tcn")) == "no tcn head yet (Day 10)"
    assert "Day 10" in M.check(_cfg(bidirectional=True))
    assert "Day 10" in M.check(_cfg(layers=2))
    assert "Day 10" in M.check(_cfg(feature_norm="layer"))
    with pytest.raises(ValueError):
        M.model(_cfg(head="tcn"), _Data(8), 3)


def test_a_gru_on_one_vector_per_clip_is_refused():
    assert "frames" in M.check(_cfg(head="gru", view="video"))
    assert M.check(_cfg(head="mlp", view="video")) is None


def test_every_key_it_reads_is_a_config_key():
    for key in M.READS:
        assert key in DEFAULTS
