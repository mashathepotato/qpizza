"""QNDM amplitude-QAE route (paper Sec 5b). Build the preparation operator A
(load paths + payoff-into-amplitude), then run Iterative Amplitude Estimation.
Genuine O(1/eps) query scaling -> the quadratic speed-up over Monte Carlo."""
import numpy as np
from qiskit.primitives import Sampler
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from quantum_pricer import oracles, tree


def price(S0, K, r, sigma, T, M, option="european", kind="call",
         epsilon_target=0.01, alpha=0.05, shots=4096, seed=7):
    angles = tree.loading_angles(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    values = tree.payoff_variable_values(S0=S0, K=K, r=r, sigma=sigma, T=T, M=M,
                                         option=option)
    payoff = np.maximum(values - K, 0.0) if kind == "call" \
        else np.maximum(K - values, 0.0)
    Cmax = float(payoff.max()) * 1.0001 if payoff.max() > 0 else 1.0
    qc, target = oracles.payoff_amplitude_circuit(angles, payoff, Cmax)

    problem = EstimationProblem(state_preparation=qc, objective_qubits=[target])
    # Deviation from plan: the V1 Sampler with shots=None gives exact (statevector)
    # probabilities, so IAE converges at Grover power 0 and reports
    # num_oracle_queries == 0. A finite shot budget makes IAE actually iterate and
    # accumulate Grover queries -- the honest O(1/eps) query count the route is about.
    iae = IterativeAmplitudeEstimation(epsilon_target=epsilon_target, alpha=alpha,
                                       sampler=Sampler(options={"shots": shots,
                                                                "seed": seed}))
    res = iae.estimate(problem)
    a = res.estimation
    return dict(price=float(np.exp(-r * T) * Cmax * a),
                a=float(a),
                num_oracle_queries=int(res.num_oracle_queries),
                Cmax=Cmax)
