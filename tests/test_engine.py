"""Most of these check the quiet failures: gradients that were never cleared,
an optimizer stepped the wrong number of times, a model left in eval."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine import EpochStats, make_scaler, train_epoch

N, D, C = 24, 8, 3


def data(n=N, batch=4):
    torch.manual_seed(0)
    x = torch.randn(n, D)
    y = torch.randint(0, C, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch)


def tiny():
    torch.manual_seed(0)
    return nn.Linear(D, C)


# ---- it runs and reports ------------------------------------------------

def test_returns_loss_accuracy_and_counts():
    model = tiny()
    stats = train_epoch(model, data(), torch.optim.SGD(model.parameters(), 0.1),
                        device=torch.device("cpu"))
    assert isinstance(stats, EpochStats)
    assert stats.clips == N
    assert 0.0 <= stats.top1 <= 1.0
    assert stats.loss > 0


def test_the_loss_goes_down_over_several_epochs():
    """6.1's whole check: a model that cannot learn a tiny set has a bug."""
    model = tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    first = train_epoch(model, data(), opt, device=torch.device("cpu")).loss
    for _ in range(20):
        last = train_epoch(model, data(), opt, device=torch.device("cpu")).loss
    assert last < first, (first, last)


def test_it_can_memorise_a_tiny_set():
    """Overfit on purpose. Near-100% here is the sign the wiring is right."""
    model = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, C))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loader = data(n=12, batch=4)
    for _ in range(80):
        stats = train_epoch(model, loader, opt, device=torch.device("cpu"))
    assert stats.top1 > 0.95, stats.top1


# ---- the quiet failures -------------------------------------------------

def test_gradients_are_cleared_between_batches():
    """Without zero_grad every batch trains on the sum of all before it."""
    model = tiny()
    seen = []

    class Spy(torch.optim.SGD):
        def step(self, *a, **k):
            seen.append(model.weight.grad.abs().sum().item())
            return super().step(*a, **k)

    train_epoch(model, data(), Spy(model.parameters(), 0.0),
                device=torch.device("cpu"))
    # lr=0 keeps the weights fixed, so a growing gradient means accumulation
    assert len(seen) == 6
    assert max(seen) < 3 * min(seen), seen


def test_the_model_is_put_into_train_mode():
    model = tiny().eval()
    train_epoch(model, data(), torch.optim.SGD(model.parameters(), 0.1),
                device=torch.device("cpu"))
    assert model.training


def test_weights_actually_change():
    model = tiny()
    before = model.weight.detach().clone()
    train_epoch(model, data(), torch.optim.SGD(model.parameters(), 0.5),
                device=torch.device("cpu"))
    assert not torch.allclose(before, model.weight)


# ---- gradient accumulation ----------------------------------------------

@pytest.mark.parametrize("accum,expected", [(1, 6), (2, 3), (3, 2), (4, 2)])
def test_the_optimizer_steps_once_per_group(accum, expected):
    """6 batches: accum 4 gives one full group plus a short final one."""
    model = tiny()
    stats = train_epoch(model, data(), torch.optim.SGD(model.parameters(), 0.1),
                        device=torch.device("cpu"), accum=accum)
    assert stats.steps == expected


def test_accumulating_matches_one_big_batch():
    """Four batches of 2 with accum=4 must give the gradient of one batch of 8."""
    torch.manual_seed(0)
    x = torch.randn(8, D)
    y = torch.randint(0, C, (8,))

    big = tiny()
    opt = torch.optim.SGD(big.parameters(), lr=0.0)
    train_epoch(big, DataLoader(TensorDataset(x, y), batch_size=8), opt,
                device=torch.device("cpu"))
    want = big.weight.grad.clone()

    small = tiny()
    opt = torch.optim.SGD(small.parameters(), lr=0.0)
    train_epoch(small, DataLoader(TensorDataset(x, y), batch_size=2), opt,
                device=torch.device("cpu"), accum=4)
    assert torch.allclose(want, small.weight.grad, atol=1e-5)


def test_a_nonsense_accum_is_refused():
    model = tiny()
    with pytest.raises(ValueError, match="at least 1"):
        train_epoch(model, data(), torch.optim.SGD(model.parameters(), 0.1),
                    device=torch.device("cpu"), accum=0)


# ---- the extras 6.3 will use --------------------------------------------

def test_max_batches_stops_early():
    model = tiny()
    stats = train_epoch(model, data(), torch.optim.SGD(model.parameters(), 0.1),
                        device=torch.device("cpu"), max_batches=2)
    assert stats.clips == 8 and stats.steps == 2


def test_the_scheduler_steps_with_the_optimizer():
    model = tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    train_epoch(model, data(), opt, device=torch.device("cpu"),
                accum=2, scheduler=sched)
    assert sched.last_epoch == 3              # once per optimizer step, not batch


def test_the_batch_callback_sees_every_batch():
    model = tiny()
    seen = []

    def note(i, total, loss):
        seen.append((i, total))

    train_epoch(model, data(), torch.optim.SGD(model.parameters(), 0.1),
                device=torch.device("cpu"), on_batch=note)
    assert seen == [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]


def test_an_empty_loader_is_an_error():
    model = tiny()
    empty = DataLoader(TensorDataset(torch.empty(0, D), torch.empty(0,
                                                                   dtype=torch.long)))
    with pytest.raises(RuntimeError, match="no batches"):
        train_epoch(model, empty, torch.optim.SGD(model.parameters(), 0.1),
                    device=torch.device("cpu"))


def test_no_scaler_on_a_device_that_does_not_need_one():
    """bf16 on the Arc needs no scaler; fp16 on CUDA does."""
    assert make_scaler(torch.device("cpu")) is None


# ---- evaluate -----------------------------------------------------------

from src.engine import EvalStats, evaluate, topk_correct   # noqa: E402


def test_topk_counts_a_hit_at_each_depth():
    # the same ranking three times (class 0 best, then 1, then 2), with the
    # true label placed at rank 1, 2 and 3 in turn
    logits = torch.tensor([[5.0, 4.0, 3.0],
                           [5.0, 4.0, 3.0],
                           [5.0, 4.0, 3.0]])
    y = torch.tensor([0, 1, 2])
    assert topk_correct(logits, y) == [1, 2, 3]


def test_topk_is_cumulative():
    """top-1 can never beat top-2, which can never beat top-3."""
    torch.manual_seed(0)
    got = topk_correct(torch.randn(40, C), torch.randint(0, C, (40,)))
    assert got[0] <= got[1] <= got[2]


def test_topk_survives_fewer_classes_than_k():
    logits = torch.tensor([[1.0, 2.0]])
    assert topk_correct(logits, torch.tensor([1]), ks=(1, 2, 3)) == [1, 1, 1]


def test_evaluate_reports_all_four_numbers():
    model = tiny()
    stats = evaluate(model, data(), device=torch.device("cpu"))
    assert isinstance(stats, EvalStats)
    assert stats.clips == N
    assert stats.top1 <= stats.top2 <= stats.top3 <= 1.0


def test_evaluate_puts_the_model_in_eval_and_leaves_weights_alone():
    model = tiny().train()
    before = model.weight.detach().clone()
    evaluate(model, data(), device=torch.device("cpu"))
    assert not model.training
    assert torch.equal(before, model.weight)


def test_evaluate_builds_no_graph():
    """no_grad is what keeps memory flat; without it .grad would fill in."""
    model = tiny()
    model.zero_grad(set_to_none=True)
    evaluate(model, data(), device=torch.device("cpu"))
    assert model.weight.grad is None


def test_evaluate_can_collect_predictions_for_a_confusion_matrix():
    model = tiny()
    stats = evaluate(model, data(), device=torch.device("cpu"), collect=True)
    assert stats.preds.shape == (N,) and stats.labels.shape == (N,)
    assert (stats.preds == stats.labels).mean() == pytest.approx(stats.top1)


def test_evaluate_collects_nothing_unless_asked():
    stats = evaluate(tiny(), data(), device=torch.device("cpu"))
    assert stats.preds is None and stats.labels is None


def test_a_perfect_model_scores_one():
    """A model that reads the label off the input. Catches a broken metric."""
    class Cheat(nn.Module):
        def forward(self, x):
            return torch.nn.functional.one_hot(
                x[:, 0].long().clamp(0, C - 1), C).float() * 10

    xs = torch.arange(9).float().unsqueeze(1).repeat(1, D) % C
    ys = (torch.arange(9) % C).long()
    loader = DataLoader(TensorDataset(xs, ys), batch_size=3)
    assert evaluate(Cheat(), loader, device=torch.device("cpu")).top1 == 1.0


def test_evaluate_stops_early_when_asked():
    stats = evaluate(tiny(), data(), device=torch.device("cpu"), max_batches=2)
    assert stats.clips == 8


def test_evaluate_reports_progress_too():
    seen = []
    evaluate(tiny(), data(), device=torch.device("cpu"),
             on_batch=lambda i, total, loss: seen.append((i, total)))
    assert seen == [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]
