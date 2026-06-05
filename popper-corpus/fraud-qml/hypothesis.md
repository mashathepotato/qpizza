---
slug: fraud-qml
created: 2026-06-06
status: active
falsifiability_gate: passed
literature_pass: completed
---

# Quantum-kernel SVM yields a meaningful, quantum-native advantage over classical RBF-SVM for credit-card fraud detection

## Original framing
> Quantum machine learning can detect fraud better than classical ML.

## Operational restatement
A quantum-kernel SVM (fidelity kernel via angle-embedding feature map) applied
to credit-card fraud detection achieves AUC >= that of a classical RBF-SVM
baseline on the same dataset and train/test split. Operationally: features
angle-embedded into a quantum circuit; kernel matrix K[i,j] = |<phi(x_i)|phi(x_j)>|^2
computed on `aer_simulator`; SVM trained with sklearn on the quantum kernel matrix.
Metric: `auc` (area under ROC curve) on held-out test set. Quantum-native claim:
the fidelity kernel is classically expensive to compute exactly when the feature
map is not efficiently classically simulable.

HONEST PRE-REGISTRATION: On standard tabular fraud-detection data (including the
ULB Kaggle creditcard.csv or the synthetic stand-in used here), we EXPECT the
falsifier to fire — classical RBF-SVM will likely match or beat the quantum
kernel AUC. The data is low-dimensional, the classes are near-separable, and
no quantum feature-map advantage is expected. This candidate is included for
its strong demo-naturalness (fraud is a compelling banking narrative) and OP
business fit (Nordea/SEB fraud prevention desks), not for expected quantum
advantage. The pre-registration is intentionally falsifier-positive to model
honest scientific practice.

## Falsifier(s)
- Classical RBF-SVM achieves `auc` >= quantum-kernel SVM on the test set at
  any simulable dataset size — this is the EXPECTED outcome and constitutes
  a clean falsification of the quantum advantage claim.
- The quantum kernel matrix is well-approximated by a classical kernel
  (e.g., cosine similarity on the raw features), confirming no quantum-specific
  inductive bias on this tabular dataset.

## Test design
- Methods: `auc` measured by triage harness (`triage/harnesses/fraud_qml.py`)
  on synthetic fraud data (or ULB Kaggle creditcard.csv if present at
  `data/raw/creditcard.csv`); 80/20 train/test split; class imbalance handled
  by stratified split.
- Design: quantum-kernel SVM (angle embedding, 4–8 qubits) vs classical
  RBF-SVM (sklearn, gamma='scale'); both trained on the same features and split.
- Comparison: `auc` with 95% CI (bootstrap); secondary metric: F1 at 0.5 threshold.
- Business tie: fraud detection is a Tier-1 use case for Nordic banking (Nordea,
  SEB); demo-naturalness is high even if advantage ties.

## Auxiliary assumptions
- The synthetic fraud data (or ULB stand-in) has statistical properties
  representative of real fraud data at the scale tested (N ~ 1000 samples,
  ~10 features post-PCA).
- `aer_simulator` faithfully computes fidelity kernel entries (no shot noise
  in statevector mode).
- Angle embedding maps each feature to a rotation angle; no data re-uploading
  is used (single-layer embedding).

## Distinctiveness
The quantum-kernel SVM makes a kernel-space prediction: the fidelity kernel
captures feature correlations that the RBF kernel cannot, IF the feature map
is non-classically-simulable. On tabular data, this distinctiveness collapses —
which is precisely what the pre-registration commits to discovering. Forbids
(in the honest direction): claiming quantum advantage when `auc` is within
noise of RBF-SVM. The finding is scientifically valuable regardless of outcome.

## References
- path: refs/havlicek-2019-quantum-kernel.md
  contribution: introduces quantum-kernel SVM with fidelity kernel; explicitly
                notes that advantage requires non-classically-simulable feature
                maps and may not arise on tabular data; motivates the honest
                pre-registration that the falsifier is expected to fire on
                standard fraud benchmarks

## Intake log
2026-06-06 — Hypothesis instantiated from triage-lab quantum-finance template.
Unusual honest pre-registration: explicitly predicts the falsifier will fire
(classical RBF-SVM ties) on tabular fraud data. Included for demo-naturalness
and OP banking business fit despite expected null advantage. This models
rigorous scientific practice — pre-committing to the null before running.
