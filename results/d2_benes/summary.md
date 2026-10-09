# D2 (extension): Beneš filter vs truncated signature

Model: dX = tanh(X) dt + dW, dY = X dt + 1.5 dV, X₀ = 0, T = 2, dt = 1e-3; N = 6000 paths (4000 train / 2000 test), seed 1.
Exact target: E[X_T|Y] = m_T + P_T tanh(m_T) (cosh × Brownian-signal Kalman–Bucy posterior; m_T linear in the signature words (y, tʲ)). Closed form checked against a 20,000-particle PF on 20 paths: RMSE 0.014 (target sd 1.91).
The part of the target that is nonlinear in m_T is 1.25% of its variance. This is the floor for any model linear in m_T, including the Kalman–Bucy-type linear-word representation of Cohen et al. (2023, Thm 1).

Held-out MSE / Var:

| M | linear words (y,tʲ) only | full truncated signature (ridge) |
|---|---|---|
| 1 | 1.44e-01 | 1.44e-01 |
| 2 | 1.35e-02 | 1.35e-02 |
| 3 | 1.25e-02 | 6.65e-03 |
| 4 | 1.25e-02 | 2.62e-03 |
| 5 | 1.25e-02 | 1.68e-03 |
| 6 | 1.25e-02 | 7.84e-04 |
| 7 | 1.25e-02 | 4.31e-04* |
| 8 | 1.25e-02 | 3.28e-04* |

\* Estimation-limited, not truncation-limited: at M=8 (510 features) train 1.33e-04 vs test 3.28e-04, and halving the training set raises test MSE to 4.28e-04. Up to M=6 train ≈ test and results are insensitive to training size (`estimation_check.json`).

Reading: the linear-word fit hits the 1.25% floor exactly from M = 3. Higher-order (multi-y) words are needed to go below it and recover ~94% of the nonlinear part by M = 6 (~97% by M = 8, estimation-limited). In the truncation-limited range the error roughly halves per level (step ratios 0.49, 0.39, 0.64, 0.47 for M = 3→6). That is geometric and much slower than the factorial-type decay in the linear-Gaussian case (where the error falls ~10–100× per level).
