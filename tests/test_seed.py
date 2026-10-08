"""Seeding, including the parts that are easy to get half-right."""
import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from src.utils.device import get_device          # noqa: E402
from src.utils.seed import (                     # noqa: E402
    DEFAULT_SEED,
    make_generator,
    seed_worker,
    set_seed,
)


def _draw():
    """One number from each of the three generators."""
    return random.random(), float(np.random.rand()), torch.randn(1).item()


def test_same_seed_gives_the_same_draws():
    set_seed(42)
    a = _draw()
    set_seed(42)
    b = _draw()
    assert a == b


def test_different_seeds_give_different_draws():
    set_seed(42)
    a = _draw()
    set_seed(43)
    b = _draw()
    assert all(x != y for x, y in zip(a, b)), "a generator ignored the seed"


def test_all_three_generators_are_covered():
    """Seeding torch alone would let this pass on torch and fail elsewhere."""
    set_seed(7)
    first = _draw()
    set_seed(7)
    for i, (x, y) in enumerate(zip(first, _draw())):
        assert x == y, f"generator {i} ('random', 'numpy', 'torch'[i]) not seeded"


def test_set_seed_returns_the_seed_for_logging():
    assert set_seed(123) == 123
    assert set_seed() == DEFAULT_SEED


def test_model_init_is_reproducible():
    set_seed(0)
    a = torch.nn.Linear(16, 4).weight.detach().clone()
    set_seed(0)
    b = torch.nn.Linear(16, 4).weight.detach().clone()
    assert torch.equal(a, b)


def test_model_init_is_reproducible_on_the_gpu():
    d = get_device()
    set_seed(0)
    a = torch.randn(64, device=d)
    set_seed(0)
    b = torch.randn(64, device=d)
    assert torch.equal(a, b)


def test_generator_makes_shuffling_repeatable():
    order1 = torch.randperm(20, generator=make_generator(5))
    order2 = torch.randperm(20, generator=make_generator(5))
    order3 = torch.randperm(20, generator=make_generator(6))
    assert torch.equal(order1, order2)
    assert not torch.equal(order1, order3)


def test_workers_get_different_seeds_from_each_other():
    """Same seed in every worker means duplicated augmentation, not variety."""
    draws = []
    for wid in range(4):
        torch.manual_seed(1000 + wid)     # what the DataLoader does per worker
        seed_worker(wid)
        draws.append((float(np.random.rand()), random.random()))
    assert len(set(draws)) == 4


def test_worker_seeding_is_stable_across_runs():
    def run():
        torch.manual_seed(999)
        seed_worker(0)
        return float(np.random.rand()), random.random()
    assert run() == run()


def test_deterministic_mode_does_not_crash_normal_ops():
    set_seed(1, deterministic=True)
    x = torch.randn(8, 4)
    (x @ x.T).sum().backward() if x.requires_grad else (x @ x.T).sum()
    set_seed(1)                            # back to default for later tests
    torch.use_deterministic_algorithms(False)
