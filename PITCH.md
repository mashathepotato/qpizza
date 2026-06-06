# Pitch — "The Madness of People Is Quantum"
### Junction Helsinki × OP Pohjola — Quantum Computing for Finance
**2-minute startup pitch (~280 words)**

---

**Here is the problem.** Every bank in Europe prices derivatives using Monte Carlo simulation. You throw a million random price paths at a contract, average the payoffs, discount back. It works — but it is slow, and slow is expensive. To halve your pricing error you need to quadruple your compute. That is not a software problem. That is a physics limit on classical randomness.

**Here is the market.** The global derivatives market is notional $700 trillion. The instruments that are genuinely hard to price — **Asian options, barrier options, path-dependent contracts** — are also the most profitable, because the bid-ask spread rewards whoever can price them faster and more precisely than the competition. Speed in pricing is directly a revenue line.

**Here is what we built.** A quantum option pricer that breaks that physics limit. Instead of sampling one price path at a time, we load **all possible price paths simultaneously into a quantum superposition** — every single one, at once, using just one qubit rotation per time step. No expensive state-preparation oracle; the binomial tree's structure gives it to us for free. Then Quantum Amplitude Estimation reads the expected payoff off that superposition. The result: you reach pricing accuracy ε in **O(1/ε) queries instead of O(1/ε²)**. Quadratic speedup. We verified this on 192 real Nokia pricing windows — every quantum route agrees with the ground truth to within 4×10⁻⁵ euros.

**Here is the wedge.** Asian options — the path-dependent instruments where this advantage bites hardest — require only a one-line change to the same circuit. Same hardware, same algorithm, broader product coverage.

**Here is the ask.** This is not production-ready today. But the asymptotic advantage is real, the circuit is built, and the first institution to embed this in their pricing stack owns a structural edge. We want to build that with OP.

---

*We are not claiming the brain is a quantum computer. We are claiming: the right tool for pricing a path-dependent contract over exponentially many futures is a computer that holds exponentially many futures at once.*

---

*Repo:* `qpizza` · *Results dashboard:* `python -m results.build_dashboard` → `results/index.html`
