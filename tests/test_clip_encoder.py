"""CLIP's image tower per frame: the vector its projection reads, in its colours."""
import pytest
import torch

from src.models.clip_encoder import is_clip

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def enc():
    from src.models.clip_encoder import ClipFrameEncoder

    return ClipFrameEncoder(pretrained=False).eval()


def test_clip_is_known_and_the_cnns_are_not():
    assert is_clip("clip_vit_b16")
    assert not is_clip("convnext_tiny")
    assert not is_clip("r3d_18")


def test_one_768_vector_per_frame(enc):
    assert enc.out_dim == 768
    with torch.no_grad():
        assert enc(torch.randn(2, 3, 3, 224, 224)).shape == (2, 3, 768)


def test_a_frozen_encoder_stays_in_eval(enc):
    enc.train()
    assert not enc.backbone.training
    enc.eval()


def test_it_is_what_clips_projection_reads(enc):
    """Measured 2026-10-16 on real weights: the CLS token before the final
    LayerNorm gives cosine 0.48 against the true embedding, the patch mean
    0.41. Only post-LayerNorm CLS gives 1.0."""
    from transformers import CLIPConfig, CLIPModel

    torch.manual_seed(0)
    full = CLIPModel(CLIPConfig(vision_config={"patch_size": 16})).eval()
    enc.backbone.load_state_dict(full.vision_model.state_dict())
    x = torch.randn(1, 2, 3, 224, 224)
    with torch.no_grad():
        mine = full.visual_projection(enc(x))
        theirs = full.get_image_features(
            pixel_values=x.flatten(0, 1)).pooler_output.unflatten(0, (1, 2))
    assert torch.allclose(mine, theirs, atol=1e-5)


def test_quiet_loading_puts_the_logging_back():
    from transformers.utils import logging

    from src.models.hf import quiet_loading

    before = logging.get_verbosity()
    bar = logging.is_progress_bar_enabled()
    with quiet_loading():
        assert logging.get_verbosity() == logging.ERROR
    assert logging.get_verbosity() == before
    assert logging.is_progress_bar_enabled() == bar


def _cached(filename):
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("openai/clip-vit-base-patch16", filename)
    return isinstance(hit, str)


@pytest.mark.skipif(not _cached("preprocessor_config.json"),
                    reason="CLIP not downloaded")
def test_colours_come_from_clips_config_not_imagenet():
    from src.models.hf import hf_input_spec

    size, mean, std = hf_input_spec("openai/clip-vit-base-patch16")
    assert size == 224
    assert mean == pytest.approx((0.48145466, 0.4578275, 0.40821073))
    assert std == pytest.approx((0.26862954, 0.26130258, 0.27577711))
