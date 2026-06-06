"""Quantum state-of-the-art (SOTA) baseline: the textbook oracle-QAE European
call pricer from Qiskit Finance (Stamatopoulos-style amplitude estimation).

WHY THIS IS THE 'SOTA' SLOT
---------------------------
The reference quantum option-pricing scheme (Stamatopoulos et al., Quantum 2020;
the Qiskit Finance tutorial implementation) prices a European call by:
  1. Loading the terminal price distribution into an n_p-qubit *price register*
     via a distribution-loading circuit (here a LogNormalDistribution -- the
     Black-Scholes terminal law). For an arbitrary distribution this loader is
     the expensive Grover-Rudolph / qGAN oracle; LogNormalDistribution is the
     analytic special case Qiskit ships.
  2. A comparator + controlled rotation that writes max(S_T - K, 0) into an
     objective qubit's amplitude.
  3. Iterative Amplitude Estimation (IAE) to read that amplitude -> O(1/eps).

KEY METHODOLOGICAL DIFFERENCE vs. OUR ROUTES (label this everywhere)
--------------------------------------------------------------------
  * OUR routes price the EXACT BINOMIAL TREE (loaded exactly with M R_y gates,
    no oracle).            -> reference = exact_tree_price(M)
  * THIS SOTA prices a LOGNORMAL (continuum Black-Scholes) law discretised onto
    n_p qubits.            -> reference = Black-Scholes price (continuum)
  They estimate two *different* (but converging) ground truths. The tree price
  and the BS price differ by an O(1/M) discretisation gap -- that gap is NOT an
  error of either method. So we score SOTA against BS, and our routes against the
  tree, and report both references side by side.

ALL NUMBERS HERE ARE SIMULATED (qiskit Sampler, finite shots) UNLESS LABELLED
'analytic'/'theoretical'. Nothing uses the known option answer.
"""
import numpy as np
from qiskit.primitives import Sampler
from qiskit_algorithms import IterativeAmplitudeEstimation
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit_finance.applications.estimation import EuropeanCallPricing

from quantum_pricer.classical import black_scholes_call


def _lognormal_model(S0, r, sigma, T, num_uncertainty_qubits):
    """Discretise the risk-neutral Black-Scholes terminal law S_T onto n_p qubits.

    Under risk-neutral GBM, ln S_T ~ Normal(mu, sigma_n^2) with
        mu      = ln S0 + (r - sigma^2/2) T
        sigma_n = sigma * sqrt(T)
    Qiskit's LogNormalDistribution takes the *variance* sigma_n^2 as its `sigma`.
    Truncation bounds are mean +/- 3 stddev of the lognormal (standard choice).
    """
    mu = np.log(S0) + (r - 0.5 * sigma ** 2) * T
    sigma_n = sigma * np.sqrt(T)
    mean = np.exp(mu + 0.5 * sigma_n ** 2)
    var = (np.exp(sigma_n ** 2) - 1.0) * np.exp(2 * mu + sigma_n ** 2)
    stddev = np.sqrt(var)
    low = float(max(0.0, mean - 3 * stddev))
    high = float(mean + 3 * stddev)
    model = LogNormalDistribution(
        num_uncertainty_qubits, mu=mu, sigma=sigma_n ** 2, bounds=(low, high)
    )
    return model, (low, high)


def price(S0, K, r, sigma, T, num_uncertainty_qubits=3, rescaling_factor=0.25,
          epsilon_target=0.01, alpha=0.05, shots=4096, seed=7):
    """Price a European call via Qiskit Finance oracle-QAE.

    Returns a dict with the SIMULATED estimate, the model's own analytic value
    (best the n_p-qubit lognormal can do), the Black-Scholes continuum reference,
    resource counts, and the error decomposition. Everything is discounted by
    e^{-rT} so it is directly comparable to a discounted option price.
    """
    model, (low, high) = _lognormal_model(S0, r, sigma, T, num_uncertainty_qubits)

    european = EuropeanCallPricing(
        num_state_qubits=num_uncertainty_qubits,
        strike_price=K,
        rescaling_factor=rescaling_factor,  # linearisation of the payoff (c_approx)
        bounds=(low, high),
        uncertainty_model=model,
    )
    problem = european.to_estimation_problem()

    # Finite-shot Sampler so IAE actually iterates and reports honest Grover
    # query counts (an exact statevector sampler converges at power 0 -> 0 queries).
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon_target, alpha=alpha,
        sampler=Sampler(options={"shots": shots, "seed": seed}),
    )
    result = iae.estimate(problem)

    # interpret() -> estimated E[max(S_T - K, 0)] (UNDISCOUNTED expected payoff)
    est_payoff = float(european.interpret(result))
    disc = float(np.exp(-r * T))
    est_price = disc * est_payoff

    # The best the discretised n_p-qubit lognormal can do (its own exact mean) --
    # this is the 'model-exact' the QAE is trying to estimate. Decomposes error
    # into (QAE+rescaling) vs (n_p discretisation).
    values = model.values            # the 2^n_p discretised price grid points
    probs = model.probabilities      # their probabilities (sum to 1)
    model_exact_payoff = float(np.sum(probs * np.maximum(values - K, 0.0)))
    model_exact_price = disc * model_exact_payoff

    bs_price = black_scholes_call(S0=S0, K=K, r=r, sigma=sigma, T=T)  # continuum ref

    qubits = int(problem.state_preparation.num_qubits)  # full circuit width

    return dict(
        price=est_price,                      # SIMULATED, discounted
        est_payoff=est_payoff,                # SIMULATED, undiscounted
        model_exact_price=model_exact_price,  # analytic (discretised lognormal)
        bs_price=bs_price,                     # analytic continuum reference
        num_oracle_queries=int(result.num_oracle_queries),  # SIMULATED
        qubits=qubits,
        num_uncertainty_qubits=int(num_uncertainty_qubits),
        rescaling_factor=float(rescaling_factor),
        epsilon_target=float(epsilon_target),
        shots=int(shots),
        bounds=(low, high),
        # error decomposition (all vs the appropriate reference)
        err_vs_bs=est_price - bs_price,
        err_qae_vs_model=est_price - model_exact_price,   # QAE + rescaling error
        err_model_vs_bs=model_exact_price - bs_price,     # n_p discretisation error
    )
