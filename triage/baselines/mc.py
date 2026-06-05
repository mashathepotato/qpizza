"""Classical Monte-Carlo baseline for QAE: samples to reach accuracy eps."""
import math


def mc_samples_to_eps(p: float, eps: float, z: float = 1.96, seed: int = 0) -> int:
    """Analytic sample count for a Bernoulli(p) mean to a +/-eps CI at level z.
    n >= z^2 * p(1-p) / eps^2.  This is MC's O(1/eps^2) scaling, the QAE foil."""
    var = p * (1.0 - p)
    return int(math.ceil((z ** 2) * var / (eps ** 2)))
