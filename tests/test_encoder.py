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


# ---- ConvNeXt: the last layer lives inside the "classifier" --------------

def test_efficientnet_head_still_becomes_identity(enc):
    """Dropout -> Linear leaves nothing to keep, so the baseline is untouched."""
    assert isinstance(enc.backbone.classifier, torch.nn.Identity)


def test_convnext_keeps_its_final_layernorm():
    """Its head is LayerNorm -> Flatten -> Linear. Only the Linear goes.

    Dropping the whole head returned (N, 768, 1, 1) with no LayerNorm, about
    6x too small. Measured 2026-10-15: the ImageNet logits then agree at
    cosine 0.87 instead of 1.0.
    """
    from torchvision.models import get_model

    e = FrameEncoder("convnext_tiny", pretrained=False).eval()
    full = get_model("convnext_tiny", weights=None).eval()
    result = e.backbone.load_state_dict(full.state_dict(), strict=False)
    assert result.missing_keys == []
    assert sorted(result.unexpected_keys) == ["classifier.2.bias",
                                              "classifier.2.weight"]
    assert e.out_dim == 768

    x = clip(1, 2)
    with torch.no_grad():
        mine = full.classifier[2](e(x))
        theirs = full(x.flatten(0, 1)).unflatten(0, (1, 2))
    assert torch.allclose(mine, theirs, atol=1e-5)


# --- Day 17: Family B's backbones -------------------------------------------

@pytest.mark.parametrize("name, flat", [
    ("convnext_tiny", 37632),        # 768 x 7 x 7
    ("mobilenet_v3_large", 47040),   # 960 x 7 x 7
    ("resnet50", 100352),            # 2048 x 7 x 7
    ("swin_t", 37632),               # 768 x 7 x 7
])
def test_every_torchvision_b_backbone_gives_its_whole_grid(name, flat):
    e = FrameEncoder(name, pretrained=False, pool="flatten")
    assert e.out_dim == flat


def test_swin_hands_over_its_grid_channels_first_like_a_cnn():
    """Swin works in (N, H, W, C); torchvision permutes before its pool, so
    the flattened row is channel-major like every CNN's."""
    from torchvision.models import get_model

    e = FrameEncoder("swin_t", pretrained=False, pool="flatten").eval()
    full = get_model("swin_t", weights=None).eval()
    e.backbone.load_state_dict(full.state_dict(), strict=False)
    x = clip(1, 1, 224)
    with torch.no_grad():
        grid = full.permute(full.norm(full.features(x.flatten(0, 1))))
        assert grid.shape == (1, 768, 7, 7)
        assert torch.equal(e(x)[0], grid.flatten(1))


@pytest.mark.parametrize("name, avg, flat", [
    ("tf_efficientnetv2_b3.in1k", 1536, 75264),
    ("legacy_xception.tf_in1k", 2048, 100352),
])
def test_the_timm_backbones_build_both_ways(name, avg, flat):
    pytest.importorskip("timm")
    assert FrameEncoder(name, pretrained=False).out_dim == avg
    assert FrameEncoder(name, pretrained=False, pool="flatten").out_dim == flat


def test_torchvision_wins_a_name_both_libraries_have():
    """resnet50 is in both. torchvision, or B06 would change under B01's
    protocol without anyone deciding it."""
    from src.models.encoder import is_timm

    assert not is_timm("resnet50")
    e = FrameEncoder("resnet50", pretrained=False)
    assert type(e.backbone).__module__.startswith("torchvision")


def test_xception_gets_its_own_colours():
    """Trained on 0.5 / 0.5. The dataset hands over ImageNet colours, so the
    encoder must turn them back into what Xception expects."""
    pytest.importorskip("timm")
    from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

    e = FrameEncoder("legacy_xception.tf_in1k", pretrained=False).eval()
    assert e.recolour
    assert not FrameEncoder("tf_efficientnetv2_b3.in1k",
                            pretrained=False).recolour
    raw = torch.rand(1, 1, 3, 96, 96)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    with torch.no_grad():
        mine = e((raw - mean) / std)
        theirs = e.backbone((raw[0] - 0.5) / 0.5)
    assert torch.allclose(mine[0], theirs, atol=1e-5)


def test_the_colour_fix_is_not_saved():
    """It follows from the name; a saved copy could only go stale."""
    pytest.importorskip("timm")
    e = FrameEncoder("legacy_xception.tf_in1k", pretrained=False)
    for key in e.state_dict():
        assert key not in ("scale", "shift")


def test_backbone_problem_names_what_is_missing(monkeypatch):
    pytest.importorskip("timm")
    import src.models.encoder as encoder

    assert encoder.backbone_problem("resnet50") is None
    assert encoder.backbone_problem("no_such_net") == \
        "neither torchvision nor timm has no_such_net"
    # Xception's weights come from a plain URL: they download themselves
    assert encoder.backbone_problem("legacy_xception.tf_in1k") is None
    # B3's live on Hugging Face, which this machine keeps offline
    monkeypatch.setattr(encoder, "_hf_cached", lambda repo, filename: False)
    told = encoder.backbone_problem("tf_efficientnetv2_b3.in1k")
    assert "HF_HUB_OFFLINE=0 hf download timm/tf_efficientnetv2_b3.in1k" in told
    assert encoder.backbone_problem("tf_efficientnetv2_b3.in1k",
                                    pretrained=False) is None
