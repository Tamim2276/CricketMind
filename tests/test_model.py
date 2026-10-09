"""5.4's checks, as tests: data flows end to end, the loss starts where an
untrained classifier should, and gradients reach the first convolution."""
import math

import pytest
import torch
import torch.nn as nn

from src.models.encoder import FrameEncoder
from src.models.head import GRUHead
from src.models.model import ShotModel, build_model

B, T, S, C = 4, 3, 64, 15


def clip(b=B, t=T, s=S):
    return torch.randn(b, t, 3, s, s)


@pytest.fixture(scope="module")
def model():
    return build_model(num_classes=C, pretrained=False)


def test_a_clip_goes_in_and_class_scores_come_out(model):
    with torch.no_grad():
        assert model.eval()(clip()).shape == (B, C)


def test_the_head_is_sized_from_the_encoder(model):
    assert model.head.in_dim == model.encoder.out_dim == 1280


def test_a_different_backbone_resizes_the_head():
    """Phase 3 swaps the encoder; nothing should need editing by hand."""
    m = build_model(num_classes=C, backbone="resnet18", pretrained=False)
    assert m.encoder.out_dim == 512 and m.head.in_dim == 512
    with torch.no_grad():
        assert m.eval()(clip(2)).shape == (2, C)


def test_the_two_halves_can_be_built_separately():
    enc = FrameEncoder(pretrained=False)
    m = ShotModel(enc, GRUHead(enc.out_dim, num_classes=C)).eval()
    with torch.no_grad():
        assert m(clip(2)).shape == (2, C)


# ---- 5.4's two checks ---------------------------------------------------

def test_an_untrained_model_loses_about_ln_num_classes(model):
    """ln(15) = 2.71 is what guessing uniformly among 15 classes costs.

    Far from it before training means something is wrong already -- a leak if
    much lower, a broken init or a label mismatch if much higher.
    """
    torch.manual_seed(0)
    model.eval()
    with torch.no_grad():
        loss = nn.functional.cross_entropy(
            model(clip(32)), torch.randint(0, C, (32,)))
    assert abs(loss.item() - math.log(C)) < 0.5, loss.item()


def test_gradients_reach_the_very_first_convolution(model):
    """The end-to-end check: a gradient has to travel the whole way back."""
    model.train()
    model.zero_grad()
    loss = nn.functional.cross_entropy(model(clip(2)),
                                       torch.randint(0, C, (2,)))
    loss.backward()
    first = None
    for module in model.encoder.backbone.modules():
        if isinstance(module, nn.Conv2d):
            first = module
            break
    assert first is not None
    assert first.weight.grad is not None
    assert first.weight.grad.abs().sum() > 0


def test_a_frozen_encoder_gets_no_gradients_but_the_head_does():
    m = build_model(num_classes=C, pretrained=False, freeze=True).train()
    nn.functional.cross_entropy(m(clip(2)), torch.randint(0, C, (2,))).backward()
    for p in m.encoder.parameters():
        assert p.grad is None
    assert m.head.gru.weight_ih_l0.grad.abs().sum() > 0


def test_train_and_eval_reach_both_halves(model):
    model.train()
    assert model.encoder.backbone.training and model.head.training
    model.eval()
    assert not model.encoder.backbone.training and not model.head.training
