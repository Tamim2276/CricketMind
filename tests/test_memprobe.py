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


def test_a_run_that_does_not_fit_says_what_to_try():
    found = {"id": "B07_regnety_16gf", "batch": 4, "fits": False,
             "peak_gb": None, "error": "XPU out of memory"}
    line = describe(found)
    assert "DOES NOT FIT" in line
    assert "batch 2 x accum 2" in line


def test_too_close_to_full_is_not_called_a_fit():
    found = {"id": "B07_regnety_16gf", "batch": 4, "fits": False,
             "peak_gb": 11.5, "reserved_gb": 11.9, "total_gb": 12.0,
             "sec_per_batch": 1.0, "steps": 1580, "params_m": 140.0}
    assert "TOO CLOSE" in describe(found)
