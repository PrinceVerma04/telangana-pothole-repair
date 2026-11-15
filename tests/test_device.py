"""CUDA usability check (DECISIONS D12): a GPU the torch wheel has no kernels for must fall back to CPU."""

from __future__ import annotations

import sys
import types

import pytest

from measure.depth_backends import base


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    base._CUDA_STATE.clear()
    monkeypatch.delenv("GM_DEVICE", raising=False)
    yield
    base._CUDA_STATE.clear()


def _fake_torch(cap, archs, available=True, free_gb=1.7):
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda i=0: cap,
        get_device_name=lambda i=0: "Fake GPU",
        get_arch_list=lambda: archs,
        mem_get_info=lambda: (int(free_gb * 1024 ** 3), 2 * 1024 ** 3),
    )

    class _T:
        def __mul__(self, o):
            return self

        def sum(self):
            return self

        def item(self):
            return 16.0

    return types.SimpleNamespace(cuda=cuda, ones=lambda *a, **k: _T())


def test_unsupported_arch_is_not_usable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch((6, 1), ["sm_75", "sm_80", "sm_120"]))
    st = base.cuda_status()
    assert st["usable"] is False and st["capability"] == "sm_61"
    assert base.gpu_free_gb() == 0.0
    assert base.pick_device("cuda", 1.0) == "cpu"
    with pytest.raises(RuntimeError):
        base.pick_device("cuda", 1.0, cpu_fallback=False)


def test_supported_arch_is_usable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch((8, 9), ["sm_80", "sm_86", "sm_89", "sm_90"], free_gb=6.0))
    assert base.cuda_usable()
    assert base.pick_device("cuda", 2.0) == "cuda"
    assert base.pick_device("cuda", 8.0) == "cpu"  # not enough free VRAM


def test_ptx_forward_compat_counts_as_usable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch((12, 1), ["sm_90", "compute_120"]))
    assert base.cuda_usable()


def test_env_override_forces_cpu(monkeypatch):
    monkeypatch.setenv("GM_DEVICE", "cpu")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch((8, 9), ["sm_89"]))
    assert not base.cuda_usable()
    assert base.cuda_status()["reason"] == "GM_DEVICE=cpu"


def test_half_precision_only_with_tensor_cores(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch((6, 1), ["sm_61"]))
    assert base.cuda_usable() and not base.cuda_half_ok()  # Pascal: fp32
    base._CUDA_STATE.clear()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch((8, 6), ["sm_86"]))
    assert base.cuda_half_ok()
