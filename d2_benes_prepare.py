"""
D2 (optional part): Beneš filter, prepare stage. Simulate, compute the exact posterior mean,
validate it against a large particle filter, and cache truncated signatures.

Model (h, gamma > 0, X_0 = 0 known):
    dX = tanh(X) dt + dW,      dY = h X dt + gamma dV,      Y_0 = 0
Since tanh' + tanh^2 = 1, Girsanov relative to Brownian motion gives the density factor
cosh(X_t) e^{-t/2}, so the posterior is cosh(x) times the Kalman–Bucy posterior N(m_t, P_t)
of a *Brownian* signal observed the same way:
    P_t = (gamma/h) tanh(h t / gamma)
    m_t = (1 / (gamma cosh(h t / gamma))) int_0^t sinh(h s / gamma) dY_s
and pi_t = mixture of N(m_t ± P_t, P_t) with weights ∝ e^{±m_t}, hence
    E[X_t | Y] = m_t + P_t tanh(m_t).
m_t is linear in the signature of (t, Y) (words y t^j); the Beneš mean is a nonlinear
function of it, so it needs higher-order (multi-y) signature terms.

As in d2_kb_signature.py, the regression target uses the segment-averaged kernel, which is
exact for the piecewise-linear interpolation of the sampled Y.

Output: results/d2_benes/cache.npz (paths' dY, targets, signatures to level M_MAX), validation.json
Usage:  python d2_benes_prepare.py
"""
import json
import math
import os
import sys
import time

import esig
import numpy as np

P = dict(h=1.0, gamma=1.5, T=2.0, n_steps=2000, N=6000, n_train=4000, seed=1, M_max=8,
         pf_paths=20, pf_particles=20000)
OUT = "results/d2_benes"


def simulate(p, rng):
    dt = p["T"] / p["n_steps"]
    N, n = p["N"], p["n_steps"]
    X = np.zeros(N)
    dY = np.empty((N, n))
    XT = None
    for k in range(n):  # Euler–Maruyama, dY_k depends on X_{t_k}
        dY[:, k] = p["h"] * X * dt + p["gamma"] * math.sqrt(dt) * rng.standard_normal(N)
        X = X + np.tanh(X) * dt + math.sqrt(dt) * rng.standard_normal(N)
    XT = X
    return dY, XT, dt


def exact_mean(p, dY, dt):
    c = p["h"] / p["gamma"]
    t = np.arange(dY.shape[1] + 1) * dt
    T = p["T"]
    # segment average of sinh(c s): (cosh(c t1) - cosh(c t0)) / (c dt)
    seg = (np.cosh(c * t[1:]) - np.cosh(c * t[:-1])) / (c * dt)
    m = (dY @ seg) / (p["gamma"] * math.cosh(c * T))
    PT = math.tanh(c * T) / c
    return m + PT * np.tanh(m), m, PT


def pf_mean(p, dY_path, dt, n_part, rng):
    """Bootstrap PF on the same Euler discretisation; posterior mean of X_T."""
    x = np.zeros(n_part)
    logw = np.zeros(n_part)
    for k in range(dY_path.shape[0]):
        logw += -0.5 * (dY_path[k] - p["h"] * x * dt) ** 2 / (p["gamma"] ** 2 * dt)
        w = np.exp(logw - logw.max()); w /= w.sum()
        if 1.0 / np.sum(w**2) < n_part / 2:  # resample when ESS drops
            x = x[rng.choice(n_part, n_part, p=w)]
            logw = np.zeros(n_part)
            w = np.full(n_part, 1.0 / n_part)
        x = x + np.tanh(x) * dt + math.sqrt(dt) * rng.standard_normal(n_part)
    return float(np.sum(w * x))


def main():
    os.makedirs(OUT, exist_ok=True)
    p = P
    rng = np.random.default_rng(p["seed"])
    t0 = time.time()
    dY, XT, dt = simulate(p, rng)
    target, m_lin, PT = exact_mean(p, dY, dt)

    # validation: exact mean vs a 20k-particle PF on a few paths
    vrng = np.random.default_rng(123)
    pf = np.array([pf_mean(p, dY[i], dt, p["pf_particles"], vrng) for i in range(p["pf_paths"])])
    ex = target[:p["pf_paths"]]
    val = {"pf_paths": p["pf_paths"], "pf_particles": p["pf_particles"],
           "max_abs_diff": float(np.max(np.abs(pf - ex))), "rmse": float(np.sqrt(np.mean((pf - ex) ** 2))),
           "target_sd": float(np.std(target)), "exact_first5": ex[:5].round(4).tolist(), "pf_first5": pf[:5].round(4).tolist(),
           # how nonlinear is the target? share of variance not explained by the linear part m
           "frac_var_beyond_linear_m": float(1 - np.corrcoef(target, m_lin)[0, 1] ** 2),
           "mse_X_T_vs_target": float(np.mean((XT - target) ** 2))}
    print("validation:", json.dumps(val, indent=1), flush=True)

    # signatures of the piecewise-linear (t, Y) path to level M_max (the slow part; ~25 ms/path)
    t = np.arange(p["n_steps"] + 1) * dt
    Y = np.concatenate([np.zeros((p["N"], 1)), np.cumsum(dY, axis=1)], axis=1)
    keys = esig.sigkeys(2, p["M_max"]).split()
    S = np.empty((p["N"], len(keys)))
    for i in range(p["N"]):
        S[i] = esig.stream2sig(np.column_stack([t, Y[i]]), p["M_max"])
    np.savez_compressed(os.path.join(OUT, "cache.npz"), S=S, keys=np.array(keys), target=target, m_lin=m_lin,
                        XT=XT, P_T=PT, params=json.dumps(p))
    val["runtime_s"] = round(time.time() - t0, 1)
    json.dump({"params": p, "validation": val}, open(os.path.join(OUT, "validation.json"), "w"), indent=2)
    print(f"cached {S.shape} signatures in {val['runtime_s']}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
