"""Device selection, and the precision policy for hardware we don't have.

get_device() depends on the machine, so it only gets smoke-tested. But
get_amp_settings() is a pure function of a device type, so the CUDA policy --
the one that can't run here -- is fully testable anyway.
"""
import pytest

torch = pytest.importorskip("torch")

from src.utils.device import (          # noqa: E402
    AmpSettings,
    describe,
    get_amp_settings,
    get_device,
)


# the policy

def test_xpu_uses_bf16_and_no_scaler():
    amp = get_amp_settings(torch.device("xpu", 0))
    assert amp.dtype is torch.bfloat16
    assert amp.use_scaler is False       # a scaler raises on XPU
    assert amp.device_type == "xpu"


def test_cuda_uses_fp16_with_a_scaler():
    amp = get_amp_settings(torch.device("cuda", 0))
    assert amp.dtype is torch.float16
    assert amp.use_scaler is True
    assert amp.device_type == "cuda"


def test_cpu_disables_autocast_entirely():
    amp = get_amp_settings(torch.device("cpu"))
    assert amp.dtype is None
    assert amp.use_scaler is False


def test_a_scaler_is_never_requested_without_fp16():
    """The real rule: a scaler only exists to rescue fp16."""
    for dev in ["xpu", "cuda", "cpu"]:
        amp = get_amp_settings(torch.device(dev))
        if amp.use_scaler:
            assert amp.dtype is torch.float16, f"{dev} wants a scaler without fp16"


def test_settings_unpack_as_a_plain_tuple():
    dtype, use_scaler, device_type = get_amp_settings(torch.device("xpu", 0))
    assert (dtype, use_scaler, device_type) == (torch.bfloat16, False, "xpu")
    assert isinstance(get_amp_settings(torch.device("cpu")), AmpSettings)


def test_device_type_is_what_autocast_actually_wants():
    amp = get_amp_settings(get_device())
    with torch.autocast(device_type=amp.device_type,
                        dtype=amp.dtype,
                        enabled=amp.dtype is not None):
        pass


# the machine

def test_get_device_is_real_and_stable():
    d = get_device()
    assert d.type in {"xpu", "cuda", "cpu"}
    assert get_device() == d, "two calls disagreed"
    torch.zeros(4, device=d)


def test_a_tensor_round_trips_through_the_device():
    d = get_device()
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert torch.equal(x.to(d).cpu(), x)


def test_describe_names_the_device_and_the_policy():
    line = describe()
    assert str(get_device()) in line
    assert "autocast=" in line and "grad_scaler=" in line


@pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()),
                    reason="no XPU on this machine")
def test_on_this_machine_it_is_the_arc():
    d = get_device()
    assert d.type == "xpu"
    assert get_amp_settings(d).use_scaler is False
