import pytest
import torch

from src.data.augment import ClipFlip, build_transform

T_, C, H, W = 4, 3, 8, 10


def clip():
    return torch.arange(T_ * C * H * W, dtype=torch.float32).reshape(T_, C, H, W)


def test_a_flip_mirrors_left_to_right():
    x = clip()
    out = ClipFlip(p=1.0)(x)
    assert torch.equal(out, torch.flip(x, dims=[-1]))


def test_p_zero_never_flips():
    x = clip()
    assert torch.equal(ClipFlip(p=0.0)(x), x)


def test_every_frame_flips_together():
    """Flipping some frames and not others invents a camera jump mid-shot."""
    x = clip()
    out = ClipFlip(p=1.0)(x)
    for t in range(T_):
        assert torch.equal(out[t], torch.flip(x[t], dims=[-1]))


def test_it_flips_roughly_half_the_time():
    torch.manual_seed(0)
    flip = ClipFlip(p=0.5)
    x = clip()
    flipped = 0
    for _ in range(200):
        if not torch.equal(flip(x), x):
            flipped += 1
    assert 70 < flipped < 130, flipped


def test_shape_is_unchanged():
    assert ClipFlip(p=1.0)(clip()).shape == (T_, C, H, W)


def test_a_single_frame_is_refused():
    with pytest.raises(ValueError, match=r"T, C, H, W"):
        ClipFlip(p=1.0)(torch.zeros(C, H, W))


@pytest.mark.parametrize("p", [-0.1, 1.5])
def test_a_nonsense_probability_is_refused(p):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ClipFlip(p=p)


def test_build_transform_returns_nothing_when_off():
    assert build_transform(0.0) is None
    assert isinstance(build_transform(0.5), ClipFlip)
