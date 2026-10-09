import math

import pytest
import torch

from src.models.head import DENSE, HIDDEN, NUM_CLASSES, GRUHead

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
