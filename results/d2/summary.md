# D2: Kalman–Bucy mean vs truncated signature (scalar OU)

Parameters: a=1, σ=1, h=1, γ=0.5, T=1, P0=P∞=0.3090 (constant gain K=1.2361, λ=a+Kh=2.2361), m0=0; Euler grid dt=1e-3; N=6000 paths (4000 train / 2000 test), seed 0. Signatures: esig 1.0.0 (iisignature not installable on Py3.13/Windows), cross-checked against pysiglib 4.0.0 (max abs diff 2e-16).

Checks: target (KB mean for piecewise-linear observations) vs left-point Itô sum RMSE 4.5e-4; vs discrete Kalman filter RMSE 1.7e-4; discrete KF variance → 0.3094 (P∞ 0.3090).

Held-out MSE / Var(m_T) (Var = 0.165):

| M | analytic coefficients K(−λ)^j, truncated | restricted OLS on (y,tʲ) words | ridge, full signature |
|---|---|---|---|
| 1 | 1.87e+00 | 1.80e-01 | 1.80e-01 |
| 2 | 1.65e+00 | 1.16e-02 | 1.16e-02 |
| 3 | 7.29e-01 | 3.72e-04 | 3.72e-04 |
| 4 | 1.90e-01 | 7.02e-06 | 7.03e-06 |
| 5 | 3.26e-02 | 8.38e-08 | 8.41e-08 |
| 6 | 3.96e-03 | 7.31e-10 | 7.43e-10 |
| 7 | 3.59e-04 | 4.68e-12 | 4.99e-12 |
| 8 | 2.52e-05 | 2.10e-14 | 5.83e-13 |

Decay rates (step ratio of MSE, M−1 → M):
- analytic (Taylor) truncation → (λT/(M+1))² asymptotically (0.070 vs 0.062 at M=8)
- restricted OLS ≈ (λT/(4M))² (0.065/0.078, 0.032/0.035, 0.019/0.020, 0.012/0.013, 0.009/0.009, 0.0064/0.0064, 0.0045/0.0049 for M=2..8), consistent with best-L² polynomial (Legendre-type) approximation of the kernel e^{−λ(T−s)}, not Taylor.
- Fitted coefficients equal analytic K(−λ)^j for low j (M=8: 1.236, −2.759, 6.085, −12.62 vs 1.236, −2.764, 6.180, −13.82) and depart for j ≥ 4, where least squares re-optimises the polynomial.
