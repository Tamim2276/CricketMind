import pytest
import torch

from src.models.encoder import FrameEncoder

B, T, S = 2, 3, 64


def clip(b=B, t=T, s=S):
    return torch.randn(b, t, 3, s, s)


@pytest.fixture(scope="module")
def enc():
    return FrameEncoder(pretrained=False).eval()


# ---- the shape contract -------------------------------------------------

def test_time_goes_in_and_comes_back_out(enc):
    with torch.no_grad():
        assert enc(clip()).shape == (B, T, enc.out_dim)


def test_efficientnet_is_1280_wide(enc):
    assert enc.out_dim == 1280


@pytest.mark.parametrize("t", [1, 5, 15])
def test_any_clip_length(enc, t):
    with torch.no_grad():
        assert enc(clip(1, t)).shape == (1, t, enc.out_dim)


def test_a_missing_time_dimension_is_refused(enc):
    with pytest.raises(ValueError, match=r"B, T, C, H, W"):
        enc(torch.randn(B, 3, S, S))


# ---- it really is the same model on every frame -------------------------

def test_identical_frames_give_identical_vectors(enc):
    """If the reshape were wrong, frames would be mixed and these would differ."""
    one = torch.randn(1, 1, 3, S, S)
    with torch.no_grad():
        out = enc(one.repeat(1, 4, 1, 1, 1))
    for t in range(1, 4):
        assert torch.allclose(out[0, 0], out[0, t], atol=1e-6)


def test_a_frame_is_encoded_as_if_it_were_alone(enc):
    """The unfold must put each frame's features back where they belong."""
    x = clip(1, 3)
    with torch.no_grad():
        together = enc(x)
        alone = enc(x[:, 1:2])
    assert torch.allclose(together[0, 1], alone[0, 0], atol=1e-5)


def test_clips_in_a_batch_do_not_leak_into_each_other(enc):
    x = clip(2, 2)
    with torch.no_grad():
        both = enc(x)
        first = enc(x[:1])
    assert torch.allclose(both[0], first[0], atol=1e-5)


# ---- freezing -----------------------------------------------------------

def test_freezing_stops_the_gradients():
    e = FrameEncoder(pretrained=False, freeze=True)
    assert not any(p.requires_grad for p in e.backbone.parameters())


def test_a_frozen_backbone_stays_in_eval_even_after_train():
    """Otherwise BatchNorm keeps updating and the features drift."""
    e = FrameEncoder(pretrained=False, freeze=True).train()
    assert e.training and not e.backbone.training


def test_an_unfrozen_backbone_trains_normally():
    e = FrameEncoder(pretrained=False).train()
    assert e.backbone.training
    assert all(p.requires_grad for p in e.backbone.parameters())


def test_gradients_reach_the_backbone():
    e = FrameEncoder(pretrained=False).train()
    e(clip(1, 2)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in e.backbone.parameters())


# ---- swapping the backbone, which Phase 3 does ten times ----------------

def test_another_backbone_with_a_different_head_attribute():
    """resnet uses .fc, efficientnet uses .classifier."""
    e = FrameEncoder("resnet18", pretrained=False).eval()
    assert e.out_dim == 512
    with torch.no_grad():
        assert e(clip(1, 2)).shape == (1, 2, 512)


def test_the_imagenet_head_is_gone(enc):
    with torch.no_grad():
        assert enc(clip(1, 1)).shape[-1] != 1000
