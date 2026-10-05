"""
D2: numerical verification of the Kalman–Bucy / signature representation and a
truncation study.

Model (scalar OU, continuous observation):
    dX = -a X dt + sigma dW,      X_0 ~ N(m0, P0)
    dY =  h X dt + gamma dV,      Y_0 = 0
With P0 = P_inf (the steady-state Riccati solution) the gain K = P_inf h / gamma^2
is constant and, with m0 = 0,
    m_T = K int_0^T exp(-lam (T - s)) dY_s,          lam = a + K h
        = sum_j K (-lam)^j  S^{(y, t^j)}_{0,T}(t, Y)
where S^{(y,t^j)} = int_0^T (T-s)^j / j! dY_s is the signature coordinate of the
time-augmented path (t, Y) on the word y t...t (Cohen et al. 2023, Thm 1 / Ex. 1).
Coefficient on the word (y, t^j) is K(-lam)^j; in the monomial basis
int (T-s)^j dY_s it is K(-lam)^j / j!.

Steps (plan D2):
 1. simulate N paths on a fine Euler grid (fixed seed and parameters, logged)
 2. exact KB mean (Riccati, constant-gain kernel) + discrete Kalman cross-check
 3. truncated signatures of the piecewise-linear (t, Y) path, M = 1..8 (esig)
 4. restricted least squares on the (y, t^j) words only, and ridge on the full
    truncated signature; held-out MSE vs M; fitted vs analytic coefficients
 5. plot and save results

Note on the target: for a piecewise-linear interpolation of the sampled Y, the
infinite signature series equals sum_k dY_k (1/dt) int_{seg_k} K e^{-lam(T-s)} ds
exactly. That is what the signature can represent, and it differs from the
left-point Ito sum by O(dt). Both are computed; the former is the regression
target, and the gap is reported.

Usage: python d2_kb_signature.py
"""
import json
import math
import os
import platform
import sys
import time

import esig
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARAMS = dict(a=1.0, sigma=1.0, h=1.0, gamma=0.5, T=1.0, n_steps=1000,
              N=6000, n_train=4000, m0=0.0, seed=0, M_max=8)
OUT = "results/d2"


def riccati_steady_state(a, sigma, h, gamma):
    # 0 = sigma^2 - 2 a P - (h^2/gamma^2) P^2
    c = h**2 / gamma**2
    return (-2 * a + math.sqrt(4 * a**2 + 4 * c * sigma**2)) / (2 * c)


def simulate(p, rng):
    dt = p["T"] / p["n_steps"]
    Pinf = riccati_steady_state(p["a"], p["sigma"], p["h"], p["gamma"])
    N, n = p["N"], p["n_steps"]
    X = np.empty((N, n + 1))
    X[:, 0] = p["m0"] + math.sqrt(Pinf) * rng.standard_normal(N)
    dW = math.sqrt(dt) * rng.standard_normal((N, n))
    dV = math.sqrt(dt) * rng.standard_normal((N, n))
    dY = np.empty((N, n))
    for k in range(n):  # Euler–Maruyama
        dY[:, k] = p["h"] * X[:, k] * dt + p["gamma"] * dV[:, k]
        X[:, k + 1] = X[:, k] - p["a"] * X[:, k] * dt + p["sigma"] * dW[:, k]
    return X, dY, dt, Pinf


def kb_means(p, dY, dt, Pinf):
    K = Pinf * p["h"] / p["gamma"]**2
    lam = p["a"] + K * p["h"]
    t = np.arange(dY.shape[1] + 1) * dt
    T = p["T"]
    # segment-averaged kernel (exact infinite-signature value for PL interpolation)
    seg = (np.exp(-lam * (T - t[1:])) - np.exp(-lam * (T - t[:-1]))) / (lam * dt)
    m_pl = K * dY @ seg
    # left-point Ito sum with the exact kernel
    m_ito = K * dY @ np.exp(-lam * (T - t[:-1]))
    return m_pl, m_ito, K, lam


def discrete_kalman(p, dY, dt):
    """Kalman filter for the Euler-discretised model (state X_k, obs dY_k = h X_k dt + gamma dV_k)."""
    N, n = dY.shape
    Pinf = riccati_steady_state(p["a"], p["sigma"], p["h"], p["gamma"])
    F, Q = 1 - p["a"] * dt, p["sigma"]**2 * dt
    H, R = p["h"] * dt, p["gamma"]**2 * dt
    m = np.full(N, p["m0"])
    P = Pinf
    for k in range(n):
        # update with dY_k (which depends on X_k), then predict X_{k+1}
        S = H * P * H + R
        G = P * H / S
        m = m + G * (dY[:, k] - H * m)
        P = (1 - G * H) * P
        m, P = F * m, F * P * F + Q
    return m, P


def signatures(dY, dt, M):
    N, n = dY.shape
    t = np.arange(n + 1) * dt
    Y = np.concatenate([np.zeros((N, 1)), np.cumsum(dY, axis=1)], axis=1)
    keys = esig.sigkeys(2, M).split()
    S = np.empty((N, len(keys)))
    for i in range(N):
        S[i] = esig.stream2sig(np.column_stack([t, Y[i]]), M)
    return S, keys, Y


def word_of(key):
    return tuple(int(c) for c in key.strip("()").split(",") if c)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    p = PARAMS
    rng = np.random.default_rng(p["seed"])

    # sanity check of esig's word convention on an L-shaped path: first t, then Y
    L = esig.stream2sig(np.array([[0., 0.], [1., 0.], [1., 1.]]), 2)
    lk = esig.sigkeys(2, 2).split()
    assert abs(L[lk.index("(1,2)")] - 1) < 1e-12 and abs(L[lk.index("(2,1)")]) < 1e-12, \
        "word (i1,i2) must mean: integrate dX^i1 first (earlier), then dX^i2"

    X, dY, dt, Pinf = simulate(p, rng)
    m_pl, m_ito, K, lam = kb_means(p, dY, dt, Pinf)
    m_dkf, P_dkf = discrete_kalman(p, dY, dt)
    S, keys, Y = signatures(dY, dt, p["M_max"])
    words = [word_of(k) for k in keys]
    level = np.array([len(w) for w in words])

    # cross-check esig against pysiglib on a few paths (optional dependency)
    xcheck = None
    try:
        import pysiglib
        tt = np.arange(dY.shape[1] + 1) * dt
        ps = np.asarray(pysiglib.sig(np.column_stack([tt, Y[0]]), p["M_max"]))
        xcheck = float(np.max(np.abs(ps.ravel() - S[0][1:])))  # pysiglib omits the level-0 term
    except Exception as e:  # report, don't fail
        xcheck = f"pysiglib check skipped: {e!r}"

    tr, te = np.arange(p["n_train"]), np.arange(p["n_train"], p["N"])
    y_tr, y_te = m_pl[tr], m_pl[te]
    var_te = float(np.var(y_te))

    # analytic coefficients on words (2, 1^j)
    ytj_idx = {j: words.index((2,) + (1,) * j) for j in range(p["M_max"])}
    analytic = {j: K * (-lam)**j for j in range(p["M_max"])}

    rows, coef_table = [], {}
    for M in range(1, p["M_max"] + 1):
        # (a) analytic truncated series, no fitting
        cols = [ytj_idx[j] for j in range(M)]
        pred_a = S[np.ix_(te, cols)] @ np.array([analytic[j] for j in range(M)])
        # (b) restricted OLS on (y, t^j) words, j < M, no intercept. Columns are
        # scaled (not centred) first: the raw words differ in size by ~T^j/j!, and
        # unscaled least squares hits a conditioning floor (~1e-9 relative MSE at M>=6).
        sd = S[np.ix_(tr, cols)].std(axis=0)
        ols = LinearRegression(fit_intercept=False).fit(S[np.ix_(tr, cols)] / sd, y_tr)
        pred_r = ols.predict(S[np.ix_(te, cols)] / sd)
        coef_table[M] = {"fitted": (ols.coef_ / sd).tolist(), "analytic": [analytic[j] for j in range(M)]}
        # (c) ridge on the full truncated signature (levels 1..M)
        full = np.where((level >= 1) & (level <= M))[0]
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-10, 3, 27)))
        ridge.fit(S[np.ix_(tr, full)], y_tr)
        pred_u = ridge.predict(S[np.ix_(te, full)])
        rows.append({
            "M": M, "n_restricted": len(cols), "n_full": int(len(full)),
            "mse_analytic_truncation": float(np.mean((pred_a - y_te)**2)),
            "mse_restricted_ols": float(np.mean((pred_r - y_te)**2)),
            "mse_full_ridge": float(np.mean((pred_u - y_te)**2)),
            "ridge_alpha": float(ridge[-1].alpha_),
            # reference: size of the next analytic term, |K| lam^M T^M / M!
            "next_term_scale": float(abs(K) * lam**M * p["T"]**M / math.factorial(M)),
        })
        print(f"M={M}: analytic {rows[-1]['mse_analytic_truncation']:.3e}  "
              f"restricted {rows[-1]['mse_restricted_ols']:.3e}  full-ridge {rows[-1]['mse_full_ridge']:.3e}",
              flush=True)

    summary = {
        "params": p, "dt": dt, "P_inf": Pinf, "K": K, "lambda": lam,
        "target": "KB mean for piecewise-linear interpolated observations (segment-averaged exact kernel)",
        "var_target_test": var_te,
        "check_pl_vs_ito_rmse": float(np.sqrt(np.mean((m_pl - m_ito)**2))),
        "check_pl_vs_discrete_kalman_rmse": float(np.sqrt(np.mean((m_pl - m_dkf)**2))),
        "check_discrete_kalman_P_end_vs_Pinf": [float(P_dkf), Pinf],
        "check_infinite_series_level8_vs_target_rmse": float(np.sqrt(np.mean(
            (S[:, [ytj_idx[j] for j in range(p["M_max"])]] @ np.array([analytic[j] for j in range(p["M_max"])]) - m_pl)**2))),
        "esig_vs_pysiglib_maxabs_path0": xcheck,
        "truncation": rows, "coefficients": coef_table,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── plot ──────────────────────────────────────────────────────────────
    Ms = [r["M"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].semilogy(Ms, [r["mse_analytic_truncation"] / var_te for r in rows], "o-", label="analytic coefficients, truncated")
    ax[0].semilogy(Ms, [r["mse_restricted_ols"] / var_te for r in rows], "s-", label="restricted OLS on (y, t^j) words")
    ax[0].semilogy(Ms, [r["mse_full_ridge"] / var_te for r in rows], "^-", label="ridge on full truncated signature")
    ax[0].semilogy(Ms, [(r["next_term_scale"]**2) * float(np.mean(S[te][:, ytj_idx[0]]**2)) / var_te for r in rows],
                   "k:", label="(K λ^M T^M / M!)² · E[Y_T²] / Var (scale)")
    ax[0].set_xlabel("truncation level M")
    ax[0].set_ylabel("held-out MSE / Var(m_T)")
    ax[0].set_title(f"KB mean vs truncated signature (a={p['a']}, σ={p['sigma']}, h={p['h']}, γ={p['gamma']}, T={p['T']})")
    ax[0].legend(fontsize=8)
    Mb = p["M_max"]
    j = np.arange(Mb)
    ax[1].plot(j, coef_table[Mb]["analytic"], "ko-", label="analytic K(-λ)^j")
    ax[1].plot(j, coef_table[Mb]["fitted"], "rx--", label=f"restricted OLS, M={Mb}")
    ax[1].set_xlabel("j  (word y t^j)")
    ax[1].set_ylabel("coefficient")
    ax[1].set_title("Fitted vs analytic coefficients")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "truncation.png"), dpi=150, bbox_inches="tight")
    plt.close()

    import importlib.metadata as md
    with open(os.path.join(OUT, "environment.txt"), "w") as f:
        f.write(f"python {platform.python_version()}\n" + "".join(
            f"{k} {md.version(k)}\n" for k in ["numpy", "scipy", "scikit-learn", "matplotlib", "esig", "roughpy", "pysiglib"]))
        f.write("iisignature: not installable on Python 3.13/Windows (no wheel, build failed); esig used instead\n")
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("check") or k in ("K", "lambda", "P_inf", "esig_vs_pysiglib_maxabs_path0", "runtime_s")}, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
