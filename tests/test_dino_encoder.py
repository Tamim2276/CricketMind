"""DINOv2 per frame: the vector its ImageNet head reads, 1,536 wide."""
import pytest
import torch

from src.models.dino_encoder import is_dino

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def enc():
    from src.models.dino_encoder import DinoFrameEncoder

    return DinoFrameEncoder(pretrained=False).eval()


def test_dino_is_known_and_the_others_are_not():
    assert is_dino("dinov2_base")
    assert not is_dino("clip_vit_b16")
    assert not is_dino("r3d_18")


def test_one_1536_vector_per_frame(enc):
    assert enc.out_dim == 1536
    with torch.no_grad():
        assert enc(torch.randn(2, 3, 3, 224, 224)).shape == (2, 3, 1536)


def test_a_frozen_encoder_stays_in_eval(enc):
    enc.train()
    assert not enc.backbone.training
    enc.eval()


def test_it_is_what_the_imagenet_head_reads(enc):
    """cat(first token, mean of patch tokens), after the final LayerNorm."""
    from transformers import Dinov2Config, Dinov2ForImageClassification

    torch.manual_seed(0)
    full = Dinov2ForImageClassification(Dinov2Config(num_labels=1000)).eval()
    result = enc.backbone.load_state_dict(full.state_dict(), strict=False)
    assert result.missing_keys == []
    assert sorted(result.unexpected_keys) == ["classifier.bias",
                                              "classifier.weight"]
    x = torch.randn(1, 2, 3, 224, 224)
    with torch.no_grad():
        mine = full.classifier(enc(x))
        theirs = full(pixel_values=x.flatten(0, 1)).logits.unflatten(0, (1, 2))
    assert torch.allclose(mine, theirs, atol=1e-4)


def _cached(filename):
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("facebook/dinov2-base-imagenet1k-1-layer",
                                 filename)
    return isinstance(hit, str)


@pytest.mark.skipif(not _cached("preprocessor_config.json"),
                    reason="DINOv2 not downloaded")
def test_size_and_colours_come_from_its_config():
    from src.models.hf import hf_input_spec

    size, mean, std = hf_input_spec("facebook/dinov2-base-imagenet1k-1-layer")
    assert size == 224
    assert mean == pytest.approx((0.485, 0.456, 0.406))
    assert std == pytest.approx((0.229, 0.224, 0.225))
