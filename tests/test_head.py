import json
import math
import os
import zipfile

import pytest
import torch

from src.models.head import (BN_EPS, BN_MOMENTUM, DENSE, HIDDEN,
                             NUM_CLASSES, GRUHead)

B, T, D = 4, 15, 1280


def feats(b=B, t=T, d=D):
    return torch.randn(b, t, d)


@pytest.fixture(scope="module")
def head():
    return GRUHead(D).eval()


# ---- the shape contract -------------------------------------------------

def test_fifteen_vectors_become_one_prediction(head):
    with torch.no_grad():
        assert head(feats()).shape == (B, NUM_CLASSES)


@pytest.mark.parametrize("t", [1, 8, 15, 49])
def test_any_number_of_frames(head, t):
    with torch.no_grad():
        assert head(feats(2, t)).shape == (2, NUM_CLASSES)


def test_a_different_encoder_width(head):
    small = GRUHead(512).eval()
    with torch.no_grad():
        assert small(feats(2, T, 512)).shape == (2, NUM_CLASSES)


def test_the_wrong_feature_width_is_refused(head):
    with pytest.raises(ValueError, match="built for 1280"):
        head(feats(2, T, 512))


def test_a_missing_time_dimension_is_refused(head):
    with pytest.raises(ValueError, match=r"B, T, D"):
        head(torch.randn(B, D))


# ---- order is the whole point -------------------------------------------

def test_reversing_the_clip_changes_the_answer(head):
    """If order did not matter this would not be a temporal model at all."""
    x = feats(1, T)
    with torch.no_grad():
        forwards = head(x)
        backwards = head(torch.flip(x, dims=[1]))
    assert not torch.allclose(forwards, backwards, atol=1e-4)


def test_the_last_frame_is_not_the_only_one_that_counts(head):
    """The summary must carry the earlier frames, not just the final pose.

    Seeded, and asserting only that the effect is non-zero. An *untrained* GRU
    remembers frame 0 about a thousand times more weakly than frame 14 --
    measured over 40 seeds, median 2.2e-04 against 2.7e-01 -- so any tolerance
    picked here would be testing the seed. Training is what strengthens it.
    """
    torch.manual_seed(0)
    x = feats(1, T)
    changed = x.clone()
    changed[0, 0] = torch.randn(D)          # only the FIRST frame differs
    with torch.no_grad():
        effect = (head(x) - head(changed)).abs().max().item()
    assert effect > 0, "frame 0 reaches the output not at all"


# ---- logits, not probabilities ------------------------------------------

def test_the_output_is_not_a_probability_distribution(head):
    """CrossEntropyLoss wants raw scores; softmax belongs inside the loss."""
    with torch.no_grad():
        out = head(feats())
    assert not torch.allclose(out.sum(dim=1), torch.ones(B), atol=1e-3)
    assert (out < 0).any()


def test_an_untrained_head_sits_near_ln_num_classes(head):
    """The number 5.4 checks for. Random guessing over 15 classes is ln(15)."""
    torch.manual_seed(0)
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(
            head(feats(64)), torch.randint(0, NUM_CLASSES, (64,)))
    assert abs(loss.item() - math.log(NUM_CLASSES)) < 0.5, loss.item()


# ---- the stack the authors specified ------------------------------------

def test_no_dropout_by_default():
    """The authors have none; this is the reproduction before the improvement."""
    assert isinstance(GRUHead(D).drop, torch.nn.Identity)


def test_dropout_when_asked_for():
    assert isinstance(GRUHead(D, dropout=0.3).drop, torch.nn.Dropout)


def test_the_layer_sizes_are_the_authors():
    h = GRUHead(D)
    assert h.gru.hidden_size == HIDDEN
    assert h.fc1.out_features == DENSE
    assert h.fc2.out_features == NUM_CLASSES


def test_gradients_reach_the_gru():
    h = GRUHead(D).train()
    h(feats(2, T)).sum().backward()
    assert h.gru.weight_ih_l0.grad is not None
    assert h.gru.weight_ih_l0.grad.abs().sum() > 0


def test_batchnorm_needs_more_than_one_clip_in_training():
    """A real constraint on batch size, worth failing loudly for."""
    h = GRUHead(D).train()
    with pytest.raises(ValueError):
        h(feats(1, T))


# ---- initialisation, which is what cost the first real run -------------

def test_the_gru_is_not_saturated_at_a_wide_input():
    """PyTorch's default ignores fan-in. At 62720 inputs that kills the GRU."""
    torch.manual_seed(0)
    h = GRUHead(62720)
    x = torch.randn(64, 62720).abs()
    z = x @ h.gru.weight_ih_l0.t()
    saturated = (z.abs() > 6).float().mean().item()
    assert saturated < 0.01, f"{saturated:.1%} of gates saturated at init"


def test_torch_default_init_really_is_the_broken_one():
    """Kept as evidence: this is what the 32.39% run was running."""
    torch.manual_seed(0)
    h = GRUHead(62720, keras_init=False)
    x = torch.randn(64, 62720).abs()
    z = x @ h.gru.weight_ih_l0.t()
    assert (z.abs() > 6).float().mean().item() > 0.3


def test_the_input_weights_scale_with_fan_in():
    narrow = GRUHead(1280).gru.weight_ih_l0.std().item()
    wide = GRUHead(62720).gru.weight_ih_l0.std().item()
    assert wide < narrow / 4, (narrow, wide)


def test_the_recurrent_weights_are_orthogonal_per_gate():
    h = GRUHead(256)
    w = h.gru.weight_hh_l0
    for i in range(0, w.shape[0], h.hidden):
        block = w[i:i + h.hidden]
        eye = block @ block.t()
        assert torch.allclose(eye, torch.eye(h.hidden), atol=1e-4)


def test_the_biases_start_at_zero():
    h = GRUHead(512)
    assert torch.equal(h.gru.bias_ih_l0, torch.zeros_like(h.gru.bias_ih_l0))
    assert torch.equal(h.gru.bias_hh_l0, torch.zeros_like(h.gru.bias_hh_l0))


# ---- the saturation watch, which the loss column cannot see -------------

def test_a_fresh_head_is_not_saturated():
    h = GRUHead(D)
    assert h.pinned_fraction() == 0.0
    assert h.dead_fraction() == 0.0


def test_pinned_units_are_counted():
    """Reproduces what best.pt held at epoch 5: 20 of 128 at tanh's rail."""
    h = GRUHead(D)
    h.norm.running_mean[:20] = 1.0
    h.norm.running_mean[20:26] = -0.95
    assert h.pinned_fraction() == pytest.approx(20 / HIDDEN)


def test_dead_units_are_counted_separately():
    """A unit can be pinned at -1 and still be the one BatchNorm amplifies."""
    h = GRUHead(D)
    h.norm.running_var[:15] = 0.0
    assert h.dead_fraction() == pytest.approx(15 / HIDDEN)
    assert h.pinned_fraction() == 0.0


def test_the_watch_survives_a_head_without_batchnorm():
    """Phase 3 swaps the head; a missing layer must not crash the epoch loop."""
    h = GRUHead(D)
    h.norm = torch.nn.Identity()
    assert h.pinned_fraction() is None
    assert h.dead_fraction() is None


# ---- BatchNorm, the second Keras/torch default that cost a run ----------

AUTHORS_KERAS = os.path.join(
    "CricShoot10kModels",
    "Efficientnetv2-s_GRU_128_NEEDS_CROPPED_SEGMENTED_SHOTS.keras")


def test_batchnorm_uses_the_authors_epsilon_not_torchs():
    """1e-3 against torch's 1e-5. On a dead unit that is 31.6x instead of
    316x, and on the epoch-5 weights it was val loss 1.4819 against 3.1031."""
    h = GRUHead(D)
    assert h.norm.eps == BN_EPS == 1e-3


def test_batchnorm_momentum_is_keras_0_99_translated():
    """Keras keeps `momentum` of the old value; torch takes `momentum` of the
    new one. Their 0.99 is 0.01 here, not 0.99 and not torch's 0.1."""
    h = GRUHead(D)
    assert h.norm.momentum == BN_MOMENTUM == 0.01


@pytest.mark.skipif(not os.path.exists(AUTHORS_KERAS),
                    reason="the authors' .keras file is not in the tree")
def test_our_batchnorm_matches_the_one_in_their_shipped_model():
    """Read straight out of their file, so this cannot drift unnoticed."""
    with zipfile.ZipFile(AUTHORS_KERAS) as z:
        cfg = json.loads(z.read("config.json"))

    theirs = None
    for layer in cfg["config"]["layers"]:
        if layer.get("class_name") == "BatchNormalization":
            theirs = layer["config"]
    assert theirs is not None, "no BatchNormalization in their model"

    h = GRUHead(D)
    assert h.norm.eps == theirs["epsilon"]
    assert h.norm.momentum == pytest.approx(1.0 - theirs["momentum"])


@pytest.mark.skipif(not os.path.exists(AUTHORS_KERAS),
                    reason="the authors' .keras file is not in the tree")
def test_their_gru_settings_are_the_ones_torch_already_gives_us():
    """reset_after=True is what torch's GRU does -- it has a separate b_hn for
    exactly that. Worth asserting so nobody 'fixes' it later."""
    with zipfile.ZipFile(AUTHORS_KERAS) as z:
        cfg = json.loads(z.read("config.json"))

    gru = None
    for layer in cfg["config"]["layers"]:
        if layer.get("class_name") == "GRU":
            gru = layer["config"]
    assert gru is not None

    assert gru["reset_after"] is True
    assert gru["activation"] == "tanh"
    assert gru["recurrent_activation"] == "sigmoid"
    assert gru["units"] == HIDDEN
    assert gru["kernel_initializer"]["class_name"] == "GlorotUniform"
    assert gru["recurrent_initializer"]["class_name"] == "Orthogonal"
    assert gru["bias_initializer"]["class_name"] == "Zeros"
