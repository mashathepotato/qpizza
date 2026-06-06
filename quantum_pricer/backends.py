"""Backend registry. Transpile quantum circuits to IQM native gates before Q50.

local_aer : qiskit-aer CPU (dev, statevector + shots)
lumi_aer  : qiskit-aer GPU/MPI (scale-up; CPU fallback)
q50_fake  : IQMFakeAphrodite (54q, native {r, cz, measure}) — noise, offline
q50_hw    : real VTT Q50 via qiskit-iqm — on-site only (needs IQM_SERVER_URL + IQM_TOKEN)
"""
import os

IQM_BASIS = ["r", "cz"]


class BackendUnavailable(RuntimeError):
    pass


def _local_aer():
    from qiskit_aer import AerSimulator
    return AerSimulator()


def _lumi_aer():
    from qiskit_aer import AerSimulator
    try:
        return AerSimulator(device="GPU")
    except Exception:
        return AerSimulator()


def _q50_fake():
    from iqm.qiskit_iqm import IQMFakeAphrodite
    return IQMFakeAphrodite()


def _q50_hw():
    token = os.environ.get("IQM_TOKEN") or os.environ.get("IQM_TOKENS_FILE")
    url = os.environ.get("IQM_SERVER_URL")
    if not token or not url:
        raise BackendUnavailable("q50_hw requires IQM_SERVER_URL and IQM_TOKEN (on-site only).")
    from iqm.qiskit_iqm import IQMProvider
    return IQMProvider(url).get_backend()


_REGISTRY = {"local_aer": _local_aer, "lumi_aer": _lumi_aer,
             "q50_fake": _q50_fake, "q50_hw": _q50_hw}


def get_backend(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend {name!r}; choose {sorted(_REGISTRY)}")
    try:
        return _REGISTRY[name]()
    except BackendUnavailable:
        raise
    except Exception as exc:
        raise BackendUnavailable(f"Backend {name!r} unavailable: {exc}") from exc
