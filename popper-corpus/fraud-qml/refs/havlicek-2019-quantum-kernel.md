# Supervised learning with quantum-enhanced feature spaces

**Authors:** Havlíček, V.; Córcoles, A. D.; Temme, K.; Harrow, A. W.; Kandala, A.;
             Chow, J. M.; Gambetta, J. M.

## Abstract
We propose two quantum machine learning algorithms that use quantum-enhanced
feature spaces: a quantum variational classifier and a quantum kernel estimator.
The quantum kernel is computed as the fidelity between quantum states produced
by a feature map circuit (angle embedding), and is classically expensive to
simulate when the feature map cannot be efficiently classically evaluated. An
SVM trained with the quantum kernel achieves competitive performance on certain
engineered datasets where the quantum feature map is the natural inductive bias.
However, on real tabular data (e.g., fraud detection), the quantum kernel often
ties with classical RBF-SVM because the feature map does not encode a
domain-specific advantage; the data is typically low-dimensional, easily
separable, and the kernel matrix is classically approximable. This honest
limitation is pre-registered: the falsifier (classical RBF-SVM matches or beats
quantum-kernel AUC) is expected to fire on standard fraud-detection benchmarks.
