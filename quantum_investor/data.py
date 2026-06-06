"""
The "human" data: sequential yes/no probabilities for two questions, in both orders.

WHAT CHANGED (real data now ships by default)
---------------------------------------------
`human_data()` now returns REAL human survey data — the canonical question-order
experiment from:

    Wang, Solloway, Shiffrin & Busemeyer, "Context effects produced by question
    orders reveal quantum nature of human judgments", PNAS 111(26):9431-9436 (2014),
    Table 1.  Open-access mirror: PMC4084470.

It is the Gallup (1997) poll asking whether **Bill Clinton** and **Al Gore** are
"honest and trustworthy", administered in BOTH orders to a representative sample of
~1000 US adults. This is the most-cited dataset in quantum cognition and the corpus
on which the parameter-free QQ-equality was confirmed across 70 national surveys.

Why this dataset (vs. a finance one):
  * It is the ONLY kind of public data that has what the model needs — the SAME
    question pair asked in BOTH orders, with the COMPLETE four-cell joint counts
    (yy/yn/ny/nn), not just marginal yes-rates. Investor-sentiment surveys
    (Michigan Survey of Consumers, Conference Board, Eurobarometer) are repeated
    cross-sections with a FIXED question order, so they structurally cannot show an
    order effect or be used to test the QQ-equality.
  * On this real data the model reproduces a genuine order effect AND the QQ-equality
    holds: observed q = -0.003 (PNAS reports q = -0.003, chi^2(1)=0.01, p=0.91).
    That is the knockout — a parameter-free quantum prediction confirmed on real
    humans, which the classical model provably cannot make.

THE FINANCE STORY (OP Pohjola business case)
--------------------------------------------
The science is question-agnostic: the same non-commuting-measurement structure
applies to investor decisions. To make it finance-native, run the two framed
questions below on ~30+ attendees (randomise which is asked first) and paste the
counts into `FINANCE_SURVEY` / use `finance_survey_data()`. The pitch:
  * validate the machinery on the canonical PNAS data (real humans, q = -0.003), then
  * show the SAME circuit on our own finance-framed crowd — herding / risk-appetite
    framing effects OP cares about, that classical models miss.

Honesty note: `FINANCE_SURVEY` below is still a PLACEHOLDER (illustrative numbers)
until you collect real responses. `human_data()` returns the REAL Clinton/Gore data.

Schema
------
A dataset is a dict:
  {
    'AB': {'yy':, 'yn':, 'ny':, 'nn':},   # A asked first; key = (A-answer, B-answer)
    'BA': {'yy':, 'yn':, 'ny':, 'nn':},   # B asked first; key = (B-answer, A-answer)
    'questions': {'A': "...", 'B': "..."},
    'source': "...",                       # provenance string (optional)
  }
Each order's four numbers sum to 1.
"""

import numpy as np

import quantum_model as qm


# ---------------------------------------------------------------------------
# REAL DATA — Wang et al., PNAS 2014, Table 1 (Gallup 1997 Clinton/Gore poll).
# Four-cell joint proportions, both orders, ~1000 US adults split across orders.
#   AB = Clinton asked first, then Gore   (key = (Clinton-answer, Gore-answer))
#   BA = Gore asked first, then Clinton   (key = (Gore-answer,    Clinton-answer))
# Each order sums to 1.000 (rounding). Verified: observed q = -0.003.
# ---------------------------------------------------------------------------
CLINTON_GORE = {
    "AB": {  # Clinton (A) then Gore (B);  keys = (Clinton, Gore)
        "yy": 0.4899,   # Clinton yes, Gore yes
        "yn": 0.0447,   # Clinton yes, Gore no
        "ny": 0.1767,   # Clinton no,  Gore yes
        "nn": 0.2886,   # Clinton no,  Gore no
    },
    "BA": {  # Gore (B) then Clinton (A);  keys = (Gore, Clinton)
        "yy": 0.5625,   # Gore yes, Clinton yes
        "yn": 0.1991,   # Gore yes, Clinton no
        "ny": 0.0255,   # Gore no,  Clinton yes
        "nn": 0.2130,   # Gore no,  Clinton no
    },
    "questions": {
        "A": "Is Bill Clinton honest and trustworthy?",
        "B": "Is Al Gore honest and trustworthy?",
    },
    "source": "Wang et al., PNAS 111:9431 (2014), Table 1; Gallup 1997, ~1000 US adults.",
}


# ---------------------------------------------------------------------------
# FINANCE FRAMING — collect this at the hackathon (PLACEHOLDER until you do).
# Two framed yes/no questions, randomise which is asked first across respondents,
# then tally the four cells per order and replace the numbers below. ~15 per order
# is enough to show q near 0. The numbers here are ILLUSTRATIVE, not real.
# ---------------------------------------------------------------------------
FINANCE_QUESTIONS = {
    "A": "Do you trust the market right now?",
    "B": "Will you invest your savings today?",
}

# Illustrative placeholder generated from the model (REPLACE with survey counts).
_PLACEHOLDER_ALPHA = 0.62
_PLACEHOLDER_BETA = 0.52
_NOISE = 0.01


def _add_noise(joint, rng):
    vals = np.array([joint[c] for c in ("yy", "yn", "ny", "nn")])
    vals = np.clip(vals + rng.normal(0, _NOISE, size=4), 1e-6, None)
    vals = vals / vals.sum()
    return dict(zip(("yy", "yn", "ny", "nn"), vals.tolist()))


def human_data(seed=0):
    """Return the REAL human dataset (Clinton/Gore, Wang et al. PNAS 2014)."""
    return {
        "AB": dict(CLINTON_GORE["AB"]),
        "BA": dict(CLINTON_GORE["BA"]),
        "questions": dict(CLINTON_GORE["questions"]),
        "source": CLINTON_GORE["source"],
    }


def finance_survey_data(seed=0):
    """
    Finance-framed dataset for the OP business case.

    Returns illustrative placeholder data until real attendee-survey counts are
    pasted in. To use real data: run FINANCE_QUESTIONS on ~30+ people, randomise
    order, tally the four cells per order, and replace the body of this function
    with the measured proportions (each order summing to 1).
    """
    rng = np.random.default_rng(seed)
    truth = qm.predict(_PLACEHOLDER_ALPHA, _PLACEHOLDER_BETA)
    return {
        "AB": _add_noise(truth["AB"], rng),
        "BA": _add_noise(truth["BA"], rng),
        "questions": dict(FINANCE_QUESTIONS),
        "source": "PLACEHOLDER (illustrative) — replace with hackathon attendee survey.",
    }


def observed_order_effect(data):
    """p(B=yes | B asked first) - p(B=yes | A asked first), straight from the data."""
    pB_first = data["BA"]["yy"] + data["BA"]["yn"]      # B yes, A any
    pB_after_A = data["AB"]["yy"] + data["AB"]["ny"]    # A any, B yes
    return pB_first - pB_after_A


def observed_qq(data):
    """
    The QQ-equality value computed straight from the data (no model, no parameters).

    q = [AB(different) - BA(different)], where "different" = the two cells in which the
    respondent answered the two questions oppositely. Quantum theory predicts q = 0;
    on the real Clinton/Gore data this is -0.003 (matches PNAS 2014).
    """
    ab_diff = data["AB"]["yn"] + data["AB"]["ny"]
    ba_diff = data["BA"]["yn"] + data["BA"]["ny"]
    return ab_diff - ba_diff
