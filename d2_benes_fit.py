"""
D2 (optional part): Beneš filter, fit stage. Reads results/d2_benes/cache.npz (from
d2_benes_prepare.py) and measures held-out MSE of signature regressions vs truncation level M:
  (a) restricted: least squares on the linear words (y, t^j), j < M (the Kalman–Bucy-type part,
      Cohen et al. 2023 Thm 1). Cannot capture the nonlinear tanh term, so it should floor near
      the target's nonlinear variance share.
  (b) full: ridge on the whole truncated signature (all words up to level M).

Usage: python d2_benes_fit.py [--max-n N] [--max-m M] [--out DIR]   (defaults: all paths, M_max, results/d2_benes)
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def word_of(key):
    return tuple(int(c) for c in key.strip("()").split(",") if c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--max-m", type=int, default=None)
    ap.add_argument("--out", default="results/d2_benes")
    a = ap.parse_args()
    c = np.load("results/d2_benes/cache.npz")
    p = json.loads(str(c["params"]))
    S, y, m_lin = c["S"], c["target"], c["m_lin"]
    keys = [str(k) for k in c["keys"]]
    if a.max_n:
        S, y, m_lin = S[:a.max_n], y[:a.max_n], m_lin[:a.max_n]
    words = [word_of(k) for k in keys]
    level = np.array([len(w) for w in words])
    n_train = int(p["n_train"] * len(y) / p["N"]) if a.max_n else p["n_train"]
    tr, te = np.arange(n_train), np.arange(n_train, len(y))
    var_te = float(np.var(y[te]))
    nl = y - np.polyval(np.polyfit(m_lin[tr], y[tr], 1), m_lin)
    floor = float(np.var(nl[te]) / var_te)  # best possible for any model linear in m_T

    Mmax = a.max_m or p["M_max"]
    rows = []
    for M in range(1, Mmax + 1):
        lin = [words.index((2,) + (1,) * j) for j in range(M)]
        sd = S[np.ix_(tr, lin)].std(axis=0)
        ols = LinearRegression().fit(S[np.ix_(tr, lin)] / sd, y[tr])
        mse_r = float(np.mean((ols.predict(S[np.ix_(te, lin)] / sd) - y[te]) ** 2))
        full = np.where((level >= 1) & (level <= M))[0]
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-10, 3, 27))).fit(S[np.ix_(tr, full)], y[tr])
        mse_f = float(np.mean((ridge.predict(S[np.ix_(te, full)]) - y[te]) ** 2))
        rows.append({"M": M, "n_linear_words": len(lin), "n_full": int(len(full)),
                     "rel_mse_restricted": mse_r / var_te, "rel_mse_full_ridge": mse_f / var_te,
                     "ridge_alpha": float(ridge[-1].alpha_)})
        print(f"M={M}: restricted {mse_r/var_te:.3e}  full-ridge {mse_f/var_te:.3e}  (floor for linear-in-m {floor:.3e})", flush=True)

    os.makedirs(a.out, exist_ok=True)
    out = {"params": p, "n_train": int(n_train), "n_test": int(len(te)), "var_target_test": var_te,
           "nonlinear_share_floor": floor, "rows": rows,
           "dry_run": bool(a.max_n or a.max_m)}
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    Ms = [r["M"] for r in rows]
    plt.figure(figsize=(7, 4.8))
    plt.semilogy(Ms, [r["rel_mse_restricted"] for r in rows], "s-", label="linear words (y, t^j) only")
    plt.semilogy(Ms, [r["rel_mse_full_ridge"] for r in rows], "^-", label="full truncated signature (ridge)")
    plt.axhline(floor, color="k", ls=":", label="nonlinear variance share (floor for linear-in-m)")
    plt.xlabel("truncation level M")
    plt.ylabel("held-out MSE / Var")
    plt.title(f"Beneš filter mean vs truncated signature (h={p['h']}, γ={p['gamma']}, T={p['T']})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "truncation_benes.png"), dpi=150)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
