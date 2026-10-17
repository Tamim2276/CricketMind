"""cached_head: the probe's Linear, V's MLP and the authors' GRU, one file."""
import pytest
import torch

from src.methods import cached_head as M
from src.models.head import (CLUSTERS, WIDTH, AttnPoolHead, FrameNorm,
                             GRUHead, LinearHead, LSTMHead, MLPHead,
                             NetVLADHead, PoolHead, TCNHead, TransformerHead,
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


def test_a_setting_the_head_would_ignore_is_refused():
    # each of these would be written into the row but change nothing
    assert "unknown head" in M.check(_cfg(head="rnn"))
    assert "bidirectional" in M.check(_cfg(head="mean", bidirectional=True))
    assert "hidden" in M.check(_cfg(head="tcn", hidden=64))
    assert "layers" in M.check(_cfg(head="gru", layers=2))
    assert "attn_heads" in M.check(_cfg(head="gru", attn_heads=8))
    assert "divide" in M.check(_cfg(head="transformer", attn_heads=3))
    assert "feature_norm" in M.check(_cfg(feature_norm="batch"))
    with pytest.raises(ValueError):
        M.model(_cfg(head="rnn"), _Data(8), 3)


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


# ---- 10.2: the recurrent family -------------------------------------------

def _shuffled(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(4, 15, 8)
    return x, x[:, torch.randperm(15), :]


def test_one_direction_draws_the_same_weights_as_before():
    # A03 must equal cached_gru exactly, so the GRU head may not change
    # what it draws from the random generator when it is one-directional
    torch.manual_seed(42)
    a = GRUHead(8, 3, hidden=16, dense=16)
    torch.manual_seed(42)
    b = GRUHead(8, 3, hidden=16, dense=16, bidirectional=False)
    for (name, p), (_, q) in zip(a.state_dict().items(),
                                 b.state_dict().items()):
        assert torch.equal(p, q), name


def test_bidirectional_doubles_the_summary():
    for head in (GRUHead(8, 3, hidden=16, dense=16, bidirectional=True),
                 LSTMHead(8, 3, hidden=16, dense=16, bidirectional=True)):
        assert head.norm.num_features == 32
        assert head(torch.randn(2, 15, 8)).shape == (2, 3)


def test_the_backward_direction_is_initialised_too():
    head = GRUHead(8, 3, hidden=16, dense=16, bidirectional=True)
    w = head.gru.weight_hh_l0_reverse[:16]          # the first gate's block
    assert torch.allclose(w @ w.T, torch.eye(16), atol=1e-5)
    assert torch.count_nonzero(head.gru.bias_ih_l0_reverse) == 0


def test_the_lstm_forget_gate_starts_at_one():
    head = LSTMHead(8, 3, hidden=16, dense=16)
    bias = head.lstm.bias_ih_l0
    assert torch.all(bias[16:32] == 1.0)            # forget gate
    assert torch.count_nonzero(bias[:16]) == 0      # input gate
    assert torch.count_nonzero(bias[32:]) == 0      # cell and output gates
    assert torch.count_nonzero(head.lstm.bias_hh_l0) == 0


def test_the_lstm_reads_order_and_reports_saturation():
    x, shuffled = _shuffled()
    head = LSTMHead(8, 3, hidden=16, dense=16).eval()
    assert not torch.allclose(head(x), head(shuffled), atol=1e-4)
    assert head.pinned_fraction() is not None


# ---- 10.3: the transformer ------------------------------------------------

def test_the_transformer_shrinks_then_attends():
    head = TransformerHead(8, 3, layers=2, attn_heads=4, dense=16)
    assert head.proj.out_features == WIDTH
    assert len(head.encoder.layers) == 2
    assert head(torch.randn(2, 15, 8)).shape == (2, 3)
    assert head(torch.randn(2, 4, 8)).shape == (2, 3)      # fewer frames


def test_the_transformer_knows_frame_order():
    # without the position vectors this would be order-blind like a pool
    x, shuffled = _shuffled()
    head = TransformerHead(8, 3, dense=16).eval()
    with torch.no_grad():
        head.pos.normal_(std=1.0)
    assert not torch.allclose(head(x), head(shuffled), atol=1e-4)


def test_the_transformer_refuses_bad_shapes():
    with pytest.raises(ValueError):
        TransformerHead(8, 3, attn_heads=3)
    with pytest.raises(ValueError):
        TransformerHead(8, 3)(torch.randn(2, 16, 8))       # 16 > 15 frames


# ---- 10.4: TCN, attention pooling, NetVLAD --------------------------------

def test_the_tcn_sees_all_fifteen_frames():
    head = TCNHead(8, 3, dense=16).eval()
    dilations = []
    for block in head.blocks:
        dilations.append(block.conv.dilation[0])
    assert dilations == [1, 2, 4]
    reach = 1
    for d in dilations:
        reach += 2 * d                   # kernel 3 adds d on each side
    assert reach == 15
    assert head(torch.randn(2, 15, 8)).shape == (2, 3)


def test_attention_weights_sum_to_one_and_are_kept():
    head = AttnPoolHead(8, 3, dense=16).eval()
    out = head(torch.randn(4, 15, 8))
    assert out.shape == (4, 3)
    assert head.last_weights.shape == (4, 15)
    sums = head.last_weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)
    assert torch.all(head.last_weights >= 0)


def test_netvlad_gives_a_unit_length_description():
    head = NetVLADHead(8, 3, dense=16).eval()
    vlad = head.encode(torch.randn(4, 15, 8))
    assert vlad.shape == (4, CLUSTERS * WIDTH)
    lengths = vlad.norm(dim=1)
    assert torch.allclose(lengths, torch.ones(4), atol=1e-5)
    assert head(torch.randn(4, 15, 8)).shape == (4, 3)


def test_every_new_head_keeps_the_same_tail():
    gru = GRUHead(8, 3, hidden=16, dense=16)
    for head in (LSTMHead(8, 3, dense=16), TransformerHead(8, 3, dense=16),
                 TCNHead(8, 3, dense=16), AttnPoolHead(8, 3, dense=16),
                 NetVLADHead(8, 3, dense=16)):
        assert head.norm.eps == gru.norm.eps, type(head).__name__
        assert head.norm.momentum == gru.norm.momentum
        assert head.fc1.out_features == 16
        assert head.fc2.out_features == 3


# ---- 10.5: LayerNorm before the head --------------------------------------

def test_frame_norm_centres_every_frame():
    model = FrameNorm(8, GRUHead(8, 3, hidden=16, dense=16))
    x = 5.0 + 3.0 * torch.rand(2, 15, 8)          # all positive, off-centre
    normed = model.frame_norm(x)
    assert torch.allclose(normed.mean(dim=2), torch.zeros(2, 15), atol=1e-5)
    assert torch.allclose(normed.std(dim=2, unbiased=False),
                          torch.ones(2, 15), atol=1e-3)
    assert model(x).shape == (2, 3)


def test_the_saturation_watch_sees_through_frame_norm():
    from src.train import _saturation
    model = FrameNorm(8, GRUHead(8, 3, hidden=16, dense=16))
    pinned, dead = _saturation(model)
    assert pinned is not None and dead is not None


def test_feature_norm_wraps_any_head():
    model = M.model(_cfg(head="gru", feature_norm="layer"), _Data(8), 3)
    assert isinstance(model, FrameNorm)
    assert isinstance(model.head, GRUHead)


# ---- 10.6: every Family A config builds the head it names -----------------

def test_all_fourteen_family_a_ids_build():
    from src.experiments import registry as R
    want = {"A01_mean": PoolHead, "A02_max": PoolHead, "A03_gru128": GRUHead,
            "A04_gru64": GRUHead, "A05_bigru128": GRUHead,
            "A06_lstm64": LSTMHead, "A07_bilstm64": LSTMHead,
            "A08_tf2l4h": TransformerHead, "A09_tf4l8h": TransformerHead,
            "A10_tcn": TCNHead, "A11_attnpool": AttnPoolHead,
            "A12_netvlad": NetVLADHead, "A13_ln_gru128": FrameNorm,
            "A14_gru128_avg": GRUHead}
    for entry in R.entries("A"):
        cfg = R.resolve(entry, "author")
        assert M.check(cfg) is None, entry["id"]
        model = M.model(cfg, _Data(8), 15)
        assert isinstance(model, want[entry["id"]]), entry["id"]
    a07 = M.model(R.resolve(R.get("A07_bilstm64"), "author"), _Data(8), 15)
    assert a07.hidden == 64 and a07.directions == 2
    a09 = M.model(R.resolve(R.get("A09_tf4l8h"), "author"), _Data(8), 15)
    assert len(a09.encoder.layers) == 4
    assert a09.encoder.layers[0].self_attn.num_heads == 8
