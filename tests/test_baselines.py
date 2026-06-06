import numpy as np
from triage.baselines.mc import mc_samples_to_eps
from triage.baselines.classical_opt import exact_portfolio, sa_portfolio
from triage.baselines.classical_kernel import rbf_svm_auc

def test_mc_estimates_bernoulli_mean_within_eps():
    # E[X]=0.3; ask MC how many samples to reach eps=0.02 (95% CI)
    n = mc_samples_to_eps(p=0.3, eps=0.02, seed=0)
    assert 1000 < n < 100_000  # ~ p(1-p)/eps^2 * z^2

def test_exact_portfolio_picks_best_two_of_three():
    # returns favor assets 0 and 2; pick exactly 2
    mu = np.array([0.1, 0.01, 0.09])
    cov = np.eye(3) * 0.0001
    chosen, val = exact_portfolio(mu, cov, k=2, risk=1.0)
    assert sorted(chosen) == [0, 2]

def test_rbf_svm_separates_a_separable_toy():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.3, (30, 2)), rng.normal(2, 0.3, (30, 2))])
    y = np.array([0] * 30 + [1] * 30)
    auc = rbf_svm_auc(X, y, seed=0)
    assert auc > 0.95

def test_sa_portfolio_finds_good_solution():
    # Asset 0 and 2 dominate: mu=[0.1, 0.01, 0.09], near-zero variance.
    # A genuine SA with enough iters should find the exact optimum [0, 2].
    mu = np.array([0.1, 0.01, 0.09])
    cov = np.eye(3) * 1e-4
    chosen, val = sa_portfolio(mu, cov, k=2, risk=1.0, seed=0, iters=2000)
    assert sorted(chosen) == [0, 2], (
        f"SA failed to find optimum [0, 2]; got {sorted(chosen)} (obj={val:.6f})"
    )
