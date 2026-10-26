"""tools/memprobe.py: a few real batches, then peak memory and time (17.2)."""
import os

import pytest
import torch

from src.experiments import registry as R
from tools.memprobe import describe, probe

needs_clips = pytest.mark.skipif(
    not os.path.isdir(os.path.join("data", "processed", "box")),
    reason="no processed clips")

# B01's protocol, made small enough for the CPU
TINY = {"backbone": "mobilenet_v3_small", "pretrained": False, "pool": "avg",
        "limit_clips": 30, "batch_size": 2}


@needs_clips
def test_a_probe_trains_and_times_on_the_cpu():
    found = probe(R.get("B01_effnetv2s"), batches=2, overrides=TINY,
                  device=torch.device("cpu"))
    assert found["fits"] is True
    assert found["sec_per_batch"] > 0
    assert found["steps"] == 15          # 30 clips, batch 2, drop_last
    assert found["peak_gb"] is None      # the CPU has no peak to read
    assert "fits" in describe(found)


@needs_clips
def test_a_probe_on_the_gpu_reads_its_peak():
    """Found on its first real run: the XPU would not reset its counters
    before anything had been allocated on it."""
    from src.utils.device import get_device

    device = get_device()
    if device.type == "cpu":
        pytest.skip("no GPU")
    found = probe(R.get("B01_effnetv2s"), batches=1, overrides=TINY,
                  device=device)
    assert found["fits"] is True
    assert 0 < found["peak_gb"] < found["total_gb"]


def test_a_run_that_does_not_fit_says_what_to_try():
    found = {"id": "B07_regnety_16gf", "batch": 4, "fits": False,
             "peak_gb": None, "error": "XPU out of memory"}
    line = describe(found)
    assert "DOES NOT FIT" in line
    assert "batch 2 x accum 2" in line


def test_the_arcs_way_of_running_out_counts_as_running_out():
    """C08 at batch 4: the driver said OUT_OF_RESOURCES, not out of memory,
    so the probe crashed instead of advising batch 2."""
    from tools.memprobe import _out_of_memory
    told = RuntimeError("level_zero backend failed with error: 40 "
                        "(UR_RESULT_ERROR_OUT_OF_RESOURCES)")
    assert _out_of_memory(told)
    assert _out_of_memory(RuntimeError("XPU out of memory. Tried to ..."))
    assert not _out_of_memory(RuntimeError("shape mismatch"))


def test_too_close_to_full_is_not_called_a_fit():
    found = {"id": "B07_regnety_16gf", "batch": 4, "fits": False,
             "peak_gb": 11.5, "reserved_gb": 11.9, "total_gb": 12.0,
             "sec_per_batch": 1.0, "steps": 1580, "params_m": 140.0}
    assert "TOO CLOSE" in describe(found)
