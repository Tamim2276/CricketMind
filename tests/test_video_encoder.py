"""The 3D CNN encoder: one vector per clip, fed the way it was trained."""
import pytest
import torch

from src.models.video_encoder import VideoEncoder, input_spec, is_video


@pytest.fixture(scope="module")
def enc():
    return VideoEncoder("r3d_18", pretrained=False).eval()


def test_r3d_18_is_a_video_model_and_convnext_is_not():
    assert is_video("r3d_18")
    assert not is_video("convnext_tiny")
    assert not is_video("efficientnet_v2_s")


def test_a_frame_model_is_refused():
    with pytest.raises(ValueError, match="not a torchvision video model"):
        VideoEncoder("convnext_tiny", pretrained=False)


def test_size_and_colours_come_from_the_weights_not_imagenet():
    size, mean, std = input_spec("r3d_18")
    assert size == 112
    assert mean == pytest.approx((0.43216, 0.394666, 0.37645))
    assert std == pytest.approx((0.22803, 0.22145, 0.216989))


def test_one_vector_per_clip_from_15_frames(enc):
    assert enc.out_dim == 512
    with torch.no_grad():
        out = enc(torch.randn(2, 15, 3, 112, 112))
    assert out.shape == (2, 512)


def test_a_frozen_encoder_stays_in_eval(enc):
    enc.train()
    assert not enc.backbone.training
    enc.eval()


def test_it_is_the_full_model_without_its_last_linear():
    """Time goes second for a 3D CNN: (B, T, C, H, W) -> (B, C, T, H, W).

    Swapping the wrong two axes still runs whenever H == W == T, and gives
    a different answer. Comparing against the untouched model catches it.
    """
    from torchvision.models import get_model

    e = VideoEncoder("r3d_18", pretrained=False).eval()
    full = get_model("r3d_18", weights=None).eval()
    result = e.backbone.load_state_dict(full.state_dict(), strict=False)
    assert result.missing_keys == []
    assert sorted(result.unexpected_keys) == ["fc.bias", "fc.weight"]

    x = torch.randn(1, 15, 3, 112, 112)
    with torch.no_grad():
        mine = full.fc(e(x))
        theirs = full(x.transpose(1, 2))
    assert torch.allclose(mine, theirs, atol=1e-5)
