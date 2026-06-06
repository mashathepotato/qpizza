import pytest


@pytest.fixture
def base_params():
    """Small, well-conditioned European-call parameters for fast tests."""
    return dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)
