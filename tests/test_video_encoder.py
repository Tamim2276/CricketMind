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


# ---- VideoMAE: a transformer that needs exactly 16 frames -----------------

def test_15_frames_become_16_by_repeating_the_middle_one():
    from src.models.video_encoder import resample_frames

    x = torch.arange(15).view(1, 15)
    picked = resample_frames(x, 16)[0].tolist()
    assert picked == [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14]
    assert resample_frames(torch.zeros(2, 16, 3), 16).shape == (2, 16, 3)


def test_videomae_counts_as_a_video_model():
    assert is_video("videomae_base")


@pytest.fixture(scope="module")
def mae():
    pytest.importorskip("transformers")
    from src.models.video_encoder import VideoMAEEncoder

    return VideoMAEEncoder(pretrained=False).eval()


def test_videomae_gives_one_768_vector_from_15_frames(mae):
    from src.models.video_encoder import VideoMAEEncoder, video_encoder

    assert mae.out_dim == 768
    assert mae.frames == 16
    with torch.no_grad():
        assert mae(torch.randn(1, 15, 3, 224, 224)).shape == (1, 768)
    assert isinstance(video_encoder("r3d_18", pretrained=False), VideoEncoder)
    assert not isinstance(video_encoder("r3d_18", pretrained=False),
                          VideoMAEEncoder)


def test_videomae_is_the_full_model_without_its_classifier(mae):
    """Mean of the tokens, then fc_norm: what the classifier reads."""
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    from src.models.video_encoder import resample_frames

    torch.manual_seed(0)
    full = VideoMAEForVideoClassification(VideoMAEConfig(num_labels=400)).eval()
    result = mae.backbone.load_state_dict(full.state_dict(), strict=False)
    assert result.missing_keys == []
    assert sorted(result.unexpected_keys) == ["classifier.bias",
                                              "classifier.weight"]
    x = torch.randn(1, 15, 3, 224, 224)
    with torch.no_grad():
        mine = full.classifier(mae(x))
        theirs = full(pixel_values=resample_frames(x, 16)).logits
    assert torch.allclose(mine, theirs, atol=1e-4)


def _hf_cached(repo, filename):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo, filename), str)


@pytest.mark.skipif(
    not _hf_cached("MCG-NJU/videomae-base-finetuned-kinetics",
                   "preprocessor_config.json"),
    reason="VideoMAE not downloaded")
def test_videomae_reads_its_size_and_colours_from_its_own_config():
    size, mean, std = input_spec("videomae_base")
    assert size == 224
    assert mean == pytest.approx((0.485, 0.456, 0.406))
    assert std == pytest.approx((0.229, 0.224, 0.225))
