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
    from src.models.video_encoder import HFVideoEncoder

    return HFVideoEncoder(pretrained=False).eval()


def test_videomae_gives_one_768_vector_from_15_frames(mae):
    from src.models.video_encoder import HFVideoEncoder, video_encoder

    assert mae.out_dim == 768
    assert mae.frames == 16
    with torch.no_grad():
        assert mae(torch.randn(1, 15, 3, 224, 224)).shape == (1, 768)
    assert isinstance(video_encoder("r3d_18", pretrained=False), VideoEncoder)
    assert not isinstance(video_encoder("r3d_18", pretrained=False),
                          HFVideoEncoder)


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


# ---- Day 14: five more torchvision video models -----------------------------

def _same_as_full(name, size, frames_in=15):
    """The encoder, then the model's own last layer, against the whole model
    with random weights: same numbers, or the wrong vector is being kept."""
    from torchvision.models import get_model

    from src.models.video_encoder import (conv_as_linear, needed_frames,
                                          resample_frames)

    torch.manual_seed(0)
    full = get_model(name, weights=None).eval()
    enc = VideoEncoder(name, pretrained=False).eval()
    result = enc.backbone.load_state_dict(full.state_dict(), strict=False)
    assert result.missing_keys == []
    if name == "s3d":
        head = conv_as_linear(full.classifier[1])
    else:
        head = None
        for layer in full.modules():
            if isinstance(layer, torch.nn.Linear):
                head = layer
    x = torch.randn(1, frames_in, 3, size, size)
    frames = needed_frames(name)
    with torch.no_grad():
        mine = head(enc(x))
        if frames is not None:
            x = resample_frames(x, frames)
        theirs = full(x.transpose(1, 2))
    return enc, mine, theirs


@pytest.mark.parametrize("name", ["mc3_18", "r2plus1d_18"])
def test_the_other_resnet3ds_work_like_r3d(name):
    assert is_video(name)
    size, mean, _ = input_spec(name)
    assert size == 112
    assert mean == pytest.approx((0.43216, 0.394666, 0.37645))
    enc, mine, theirs = _same_as_full(name, 112)
    assert enc.out_dim == 512
    assert torch.allclose(mine, theirs, atol=1e-5)


def test_s3d_keeps_the_average_its_1x1x1_conv_reads():
    enc, mine, theirs = _same_as_full("s3d", 224)
    assert enc.out_dim == 1024
    assert enc.frames is None                  # 15 is enough; it wants 14
    assert torch.allclose(mine, theirs, atol=1e-5)


def test_conv_as_linear_is_the_same_map():
    from src.models.video_encoder import conv_as_linear

    torch.manual_seed(0)
    conv = torch.nn.Conv3d(8, 3, kernel_size=1)
    v = torch.randn(4, 8)
    with torch.no_grad():
        want = conv(v[:, :, None, None, None]).flatten(1)
        got = conv_as_linear(conv)(v)
    assert torch.allclose(got, want, atol=1e-6)


def test_swin3d_takes_15_frames_and_keeps_the_pooled_768():
    size, mean, _ = input_spec("swin3d_t")
    assert size == 224
    assert mean == pytest.approx((0.485, 0.456, 0.406))
    enc, mine, theirs = _same_as_full("swin3d_t", 224)
    assert enc.out_dim == 768
    assert enc.frames is None
    assert torch.allclose(mine, theirs, atol=1e-4)


def test_mvit_gets_exactly_16_frames_and_keeps_its_class_token():
    from src.models.video_encoder import needed_frames

    assert needed_frames("mvit_v2_s") == 16
    assert needed_frames("s3d") is None
    assert needed_frames("r3d_18") is None
    size, mean, std = input_spec("mvit_v2_s")
    assert size == 224
    assert mean == pytest.approx((0.45, 0.45, 0.45))
    assert std == pytest.approx((0.225, 0.225, 0.225))
    enc, mine, theirs = _same_as_full("mvit_v2_s", 224)
    assert enc.out_dim == 768
    assert enc.frames == 16
    assert torch.allclose(mine, theirs, atol=1e-4)


# ---- Day 14: TimeSformer, 8 frames from 15 ----------------------------------

def test_15_frames_become_8_by_keeping_every_other_one():
    from src.models.video_encoder import resample_frames

    x = torch.arange(15).view(1, 15)
    assert resample_frames(x, 8)[0].tolist() == [0, 2, 4, 6, 8, 10, 12, 14]


def test_timesformer_counts_as_a_video_model():
    assert is_video("timesformer_base")


def test_timesformer_is_the_full_model_without_its_classifier():
    """The first token after the final LayerNorm: what the classifier reads."""
    pytest.importorskip("transformers")
    from transformers import AutoConfig, AutoModelForVideoClassification

    from src.models.video_encoder import HFVideoEncoder, resample_frames

    torch.manual_seed(0)
    config = AutoConfig.for_model("timesformer", num_labels=400)
    full = AutoModelForVideoClassification.from_config(config).eval()
    enc = HFVideoEncoder("timesformer_base", pretrained=False).eval()
    assert enc.frames == 8
    assert enc.out_dim == 768
    result = enc.backbone.load_state_dict(full.state_dict(), strict=False)
    assert result.missing_keys == []
    assert sorted(result.unexpected_keys) == ["classifier.bias",
                                              "classifier.weight"]
    x = torch.randn(1, 15, 3, 224, 224)
    with torch.no_grad():
        mine = full.classifier(enc(x))
        theirs = full(pixel_values=resample_frames(x, 8)).logits
    assert torch.allclose(mine, theirs, atol=1e-4)


@pytest.mark.skipif(
    not _hf_cached("facebook/timesformer-base-finetuned-k400",
                   "preprocessor_config.json"),
    reason="TimeSformer not downloaded")
def test_timesformer_reads_its_size_and_colours_from_its_own_config():
    size, mean, std = input_spec("timesformer_base")
    assert size == 224
    assert mean == pytest.approx((0.45, 0.45, 0.45))
    assert std == pytest.approx((0.225, 0.225, 0.225))
