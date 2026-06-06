"""
The "human" data: sequential yes/no probabilities for two questions, in both orders.

IMPORTANT (read before the pitch)
---------------------------------
The dataset below is ILLUSTRATIVE — generated from the quantum model with a small
amount of noise so the whole pipeline runs end-to-end out of the box. For the real
submission, REPLACE it with genuine numbers, e.g.:
  * a behavioural survey you run on attendees (two framed yes/no questions, randomise
    which is asked first), or
  * a published question-order dataset (e.g. the Gallup polls analysed in
    Wang et al., PNAS 2014).
The scientific claim — quantum fits the order effect + satisfies the QQ-equality,
classical cannot — holds for real data; that's the point of the paper.

Schema
------
A dataset is a dict:
  {
    'AB': {'yy':, 'yn':, 'ny':, 'nn':},   # A asked first; key = (A-answer, B-answer)
    'BA': {'yy':, 'yn':, 'ny':, 'nn':},   # B asked first; key = (B-answer, A-answer)
    'questions': {'A': "...", 'B': "..."},
  }
Each order's four numbers sum to 1.
"""

import numpy as np

import quantum_model as qm

QUESTIONS = {
    "A": "Do you trust the market right now?",
    "B": "Will you invest your savings today?",
}

# "Ground-truth" human parameters (unknown to the models; they must fit to the data).
_TRUE_ALPHA = 0.62
_TRUE_BETA = 0.52
_NOISE = 0.01  # small measurement noise on the human responses


def _add_noise(joint, rng):
    vals = np.array([joint[c] for c in ("yy", "yn", "ny", "nn")])
    vals = np.clip(vals + rng.normal(0, _NOISE, size=4), 1e-6, None)
    vals = vals / vals.sum()
    return dict(zip(("yy", "yn", "ny", "nn"), vals.tolist()))


def human_data(seed=0):
    """Return the illustrative human dataset (replace with real data for submission)."""
    rng = np.random.default_rng(seed)
    truth = qm.predict(_TRUE_ALPHA, _TRUE_BETA)
    return {
        "AB": _add_noise(truth["AB"], rng),
        "BA": _add_noise(truth["BA"], rng),
        "questions": dict(QUESTIONS),
    }


def observed_order_effect(data):
    """p(B=yes | B asked first) - p(B=yes | A asked first), straight from the data."""
    pB_first = data["BA"]["yy"] + data["BA"]["yn"]      # B yes, A any
    pB_after_A = data["AB"]["yy"] + data["AB"]["ny"]    # A any, B yes
    return pB_first - pB_after_A
