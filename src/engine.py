"""One epoch of training. The five lines everything else serves.

Four things here are easy to get wrong and silent when you do:

**zero_grad comes first.** PyTorch *accumulates* gradients into `.grad` rather
than replacing them. Forget to clear and every batch trains on the sum of
itself and all its predecessors -- no crash, just a model that will not learn.

**model.train() matters.** BatchNorm uses the batch's own statistics in train
mode and its stored running ones in eval. Train with it in eval mode and the
normalisation never adapts.

**autocast, from Day 1.2, in its real place.** bf16 on the Arc with no scaler,
fp16 plus a GradScaler on CUDA, because fp16 gradients underflow and the
scaler multiplies them up before they do.

**Gradient accumulation** buys a bigger effective batch than the card holds:
back-propagate every batch, but step the optimizer only every N. Measured on
this machine, batch 6 fills 6.82 GB, so accum=2 gives the gradient of a batch
of 12 at the memory of 6. The loss is divided by N before backward, or the
summed gradient would be N times too large.

An out-of-memory here is fatal and must not be caught: an XPU OOM leaves every
later allocation in the process failing too, so the caller should checkpoint
and exit rather than retry smaller.
"""
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from src.utils.device import get_amp_settings, get_device

__all__ = ["train_epoch", "evaluate", "topk_correct",
           "EpochStats", "EvalStats", "make_scaler"]


class EpochStats(NamedTuple):
    loss: float         # mean over clips
    top1: float         # 0-1
    clips: int
    steps: int          # optimizer steps, fewer than batches when accumulating


class EvalStats(NamedTuple):
    loss: float
    top1: float
    top2: float
    top3: float
    clips: int
    preds: Optional[object] = None    # (N,) predicted labels, when collected
    labels: Optional[object] = None   # (N,) true labels, for a confusion matrix


def make_scaler(device=None):
    """A GradScaler where the device needs one, otherwise None."""
    device = device or get_device()
    amp = get_amp_settings(device)
    if not amp.use_scaler:
        return None
    return torch.amp.GradScaler(amp.device_type)


def train_epoch(model, loader, optimizer, criterion=None, device=None,
                scaler=None, accum: int = 1, scheduler=None,
                max_batches: Optional[int] = None,
                on_batch=None) -> EpochStats:
    """One pass over `loader`. Returns mean loss and top-1 over the epoch."""
    if accum < 1:
        raise ValueError(f"accum must be at least 1, got {accum}")

    device = device or get_device()
    amp = get_amp_settings(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    planned = len(loader)
    if max_batches is not None and max_batches < planned:
        planned = max_batches

    model.train()
    loss_sum = 0.0
    correct = 0
    clips = 0
    steps = 0

    for i, (x, y) in enumerate(loader):
        if i >= planned:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # first of an accumulation group: clear what the last step left behind
        if i % accum == 0:
            optimizer.zero_grad(set_to_none=True)

        if amp.dtype is None:
            logits = model(x)
            loss = criterion(logits, y)
        else:
            with torch.autocast(amp.device_type, dtype=amp.dtype):
                logits = model(x)
                loss = criterion(logits, y)

        # divide, or N batches of gradient add up to N times the real thing
        scaled = loss / accum
        if scaler is None:
            scaled.backward()
        else:
            scaler.scale(scaled).backward()

        last = (i + 1) == planned
        if (i + 1) % accum == 0 or last:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            steps += 1
            if scheduler is not None:
                scheduler.step()

        n = y.size(0)
        loss_sum += loss.item() * n
        correct += (logits.argmax(dim=1) == y).sum().item()
        clips += n

        if on_batch is not None:
            on_batch(i, planned, loss.item())

    if clips == 0:
        raise RuntimeError("the loader produced no batches")
    return EpochStats(loss_sum / clips, correct / clips, clips, steps)


def topk_correct(logits, y, ks=(1, 2, 3)):
    """How many of the batch had the true label among the top k, for each k.

    Top-3 is worth recording for 15 confusable shots: a Pull and a Hook differ
    by where the ball was, which the crop does not always show. A model that
    is wrong at top-1 but right at top-3 is confusing neighbours, which is a
    different problem from one that has no idea.
    """
    classes = logits.size(1)
    biggest = min(max(ks), classes)
    _, ranked = logits.topk(biggest, dim=1)
    hit = ranked == y.unsqueeze(1)

    out = []
    for k in ks:
        use = min(k, biggest)
        out.append(int(hit[:, :use].any(dim=1).sum().item()))
    return out


def evaluate(model, loader, criterion=None, device=None,
             max_batches: Optional[int] = None, use_amp: bool = True,
             collect: bool = False, on_batch=None) -> EvalStats:
    """Top-1/2/3 and loss, with nothing learned.

    Both `model.eval()` and `torch.no_grad()` are needed and they do different
    jobs. `eval()` changes layer behaviour -- BatchNorm switches to its stored
    running statistics instead of this batch's, so the answer for a clip stops
    depending on which clips happen to sit beside it. `no_grad()` stops the
    graph being built at all, which is what makes this fast and keeps memory
    flat.
    """
    device = device or get_device()
    amp = get_amp_settings(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    planned = len(loader)
    if max_batches is not None and max_batches < planned:
        planned = max_batches

    model.eval()
    loss_sum = 0.0
    hits = [0, 0, 0]
    clips = 0
    preds = []
    labels = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= planned:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if amp.dtype is None or not use_amp:
                logits = model(x)
                loss = criterion(logits, y)
            else:
                with torch.autocast(amp.device_type, dtype=amp.dtype):
                    logits = model(x)
                    loss = criterion(logits, y)

            logits = logits.float()
            got = topk_correct(logits, y)
            for k in range(3):
                hits[k] += got[k]
            loss_sum += loss.item() * y.size(0)
            clips += y.size(0)

            if collect:
                preds.append(logits.argmax(dim=1).cpu())
                labels.append(y.cpu())

            if on_batch is not None:
                on_batch(i, planned, loss.item())

    if clips == 0:
        raise RuntimeError("the loader produced no batches")

    out_preds = None
    out_labels = None
    if collect:
        out_preds = torch.cat(preds).numpy()
        out_labels = torch.cat(labels).numpy()

    return EvalStats(loss_sum / clips, hits[0] / clips, hits[1] / clips,
                     hits[2] / clips, clips, out_preds, out_labels)
