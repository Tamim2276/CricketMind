"""cached_head: the probe's Linear, V's MLP and the authors' GRU, one file."""
import pytest
import torch

from src.methods import cached_head as M
from src.models.head import (GRUHead, LinearHead, MLPHead, PoolHead,
                             pool_frames)
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


# ---- 10.1: the two controls, mean and max --------------------------------

def test_mean_and_max_by_hand():
    # one clip, 3 frames, 2 numbers each: the example from the explanation
    x = torch.tensor([[[1.0, 8.0],
                       [3.0, 2.0],
                       [5.0, 2.0]]])
    assert pool_frames(x, "mean").tolist() == [[3.0, 4.0]]
    assert pool_frames(x, "max").tolist() == [[5.0, 8.0]]


def test_pooling_ignores_frame_order():
    torch.manual_seed(0)
    x = torch.randn(4, 15, 8)
    shuffled = x[:, torch.randperm(15), :]
    for how in ("mean", "max"):
        head = PoolHead(8, 3, how=how, dense=16).eval()
        assert torch.allclose(head(x), head(shuffled), atol=1e-6)


def test_a_gru_does_notice_frame_order():
    # the contrast that makes the controls worth having
    torch.manual_seed(0)
    x = torch.randn(4, 15, 8)
    shuffled = x[:, torch.randperm(15), :]
    head = GRUHead(8, 3, hidden=16, dense=16).eval()
    assert not torch.allclose(head(x), head(shuffled), atol=1e-4)


def test_the_pool_head_keeps_the_gru_heads_tail():
    pool = PoolHead(8, 3, how="mean", dense=16)
    gru = GRUHead(8, 3, hidden=16, dense=16)
    assert pool(torch.randn(2, 15, 8)).shape == (2, 3)
    assert pool.norm.eps == gru.norm.eps
    assert pool.norm.momentum == gru.norm.momentum
    assert pool.fc1.out_features == gru.fc1.out_features


def test_pooling_needs_frames():
    with pytest.raises(ValueError):
        pool_frames(torch.randn(2, 8), "mean")
    with pytest.raises(ValueError):
        PoolHead(8, 3, how="median")
    assert "frames" in M.check(_cfg(head="mean", view="video"))
    assert M.check(_cfg(head="max", view="avg")) is None


def test_the_config_picks_a_pool_head():
    head = M.model(_cfg(head="max"), _Data(8), 3)
    assert isinstance(head, PoolHead)
    assert head.how == "max"


def test_a01_and_a02_are_no_longer_blocked():
    from src.experiments import registry as R
    for exp_id in ("A01_mean", "A02_max"):
        assert M.check(R.resolve(R.get(exp_id), "author")) is None
