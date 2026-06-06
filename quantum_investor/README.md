# Quantum Investor — starter code

> "The madness of people is quantum." A tiny, runnable proof-of-concept: human
> question-**order effects** in an investment decision are reproduced by a 1-qubit
> quantum model and are **impossible** for a classical (Bayesian) model — and the
> quantum model satisfies the *parameter-free* **QQ-equality** (PNAS 2014).

## Run it (≈30 seconds)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py          # prints the story + numbers, saves figure.png
python test_model.py    # self-checks (QQ-equality, order effect, normalisation)
```

## What each file does
| file | role |
|---|---|
| `quantum_model.py` | PennyLane belief-state circuit + sequential projective measurement of two non-commuting questions. Produces the order effect and the QQ-equality. **This is the science.** |
| `classical_model.py` | The Bayesian baseline that *structurally* cannot produce an order effect (its predicted effect is always 0). |
| `data.py` | The "human" responses. **Currently illustrative — swap in real survey/poll data for the submission** (see the note at the top of the file). |
| `fit.py` | Fits the quantum model's two parameters (α, β) to the data. |
| `plot.py` | The headline figure: humans vs classical vs quantum. |
| `main.py` | Runs everything end-to-end. |
| `test_model.py` | Fast self-checks so you trust the demo live. |

## The one-paragraph science
Two yes/no questions are projective measurements on a qubit "belief state". They
correspond to measurement axes that are *tilted* (non-commuting), so the order you
ask them in changes the answers — a genuine **order effect**. A classical joint
distribution is order-blind, so it can't represent this at all. The quantum model
also obeys the **QQ-equality** (the probability of giving *different* answers is the
same in both orders) — a parameter-free prediction confirmed on 70 national surveys
(Wang et al., PNAS 2014). That's the difference between a metaphor and a model.

## Guardrail (say it before a judge asks)
We do **not** claim the brain is a quantum computer. We claim human judgement
violates *classical* probability exactly as *quantum* probability predicts. It's
quantum-**like** math, not quantum neurons.

## Where to take it next (the "nice-to-have" stretch)
- **Real data:** run two framed yes/no questions on attendees, randomise order, drop
  the numbers into `data.py`.
- **Disjunction effect:** add a two-stage gamble (a second classic anomaly).
- **Crowd / bubble:** N coupled investor qubits; sweep the coupling and watch
  constructive *interference of intentions* form a bubble (Yukalov–Sornette).

## References
- Wang, Solloway, Shiffrin & Busemeyer, *PNAS* 111:9431 (2014) — QQ-equality, 70 surveys.
- Pothos & Busemeyer, *Proc. R. Soc. B* 276:2171 (2009) — quantum beats classical on sure-thing violations.
- Widdows, Rani & Pothos, *Entropy* 25:548 (2023) — these models as qubit circuits.
- Yukalov & Sornette, *Theory and Decision* 70:283 (2011) — quantum decision theory for risky prospects.
