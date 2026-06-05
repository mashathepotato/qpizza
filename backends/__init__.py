"""Backend abstraction — the Qiskit compatibility contract.

local_aer : qiskit-aer CPU (dev + tonight's builds)
lumi_aer  : qiskit-aer GPU/MPI (scale-up; falls back to CPU if no GPU)
q50_fake  : IQM fake backend — VTT Q50 native gates + noise, offline, no queue
             Uses IQMFakeAphrodite (54 qubits, closest to the target 50-qubit machine).
             Native gate set: {r, cz, measure}. Circuits must be transpiled with
             basis_gates=["r", "cz"] before submission.
q50_hw    : real VTT Q50 via qiskit-iqm — guarded, on-site only
"""
import os


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot be constructed in this environment."""


def _local_aer():
    from qiskit_aer import AerSimulator
    return AerSimulator()


def _lumi_aer():
    from qiskit_aer import AerSimulator
    try:
        return AerSimulator(device="GPU")
    except Exception:
        return AerSimulator()  # no GPU here; CPU. LUMI provides the GPU at scale.


def _q50_fake():
    # IQMFakeAphrodite: 54 qubits, native gates {r, cz, measure, id, delay}.
    # Chosen over IQMFakeApollo (20 q) and IQMFakeDeneb (6 q) because it is
    # the largest available fake chip and best emulates a ~50-qubit machine.
    from iqm.qiskit_iqm import IQMFakeAphrodite
    return IQMFakeAphrodite()


def _q50_hw():
    token = os.environ.get("IQM_TOKEN") or os.environ.get("IQM_TOKENS_FILE")
    url = os.environ.get("IQM_SERVER_URL")
    if not token or not url:
        raise BackendUnavailable(
            "q50_hw requires IQM_SERVER_URL and IQM_TOKEN (or IQM_TOKENS_FILE) (on-site only)."
        )
    from iqm.qiskit_iqm import IQMProvider
    return IQMProvider(url).get_backend()


_REGISTRY = {
    "local_aer": _local_aer,
    "lumi_aer": _lumi_aer,
    "q50_fake": _q50_fake,
    "q50_hw": _q50_hw,
}


def get_backend(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(_REGISTRY)}")
    try:
        return _REGISTRY[name]()
    except BackendUnavailable:
        raise
    except Exception as exc:
        raise BackendUnavailable(f"Backend {name!r} unavailable: {exc}") from exc
