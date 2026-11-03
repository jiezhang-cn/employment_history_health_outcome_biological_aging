"""m3b: omnibus test of each exposure term against all features of a layer (one test per sex x sample x
layer x modality x term).

Per-feature BH has little power when a signal is weak and spread over many features; the omnibus test
pools the evidence of the whole family. Which features differ is read from m3.

Method
  1. Take the per-feature z statistics from m3 (same Z, HC1). T = mean_j z_j^2 (expectation 1 under H0).
  2. Under H0, z ~ N(0, C) with C the correlation matrix of the features after residualising on Z
     (pairwise complete observations, scaled by each feature's n). Then m * T ~ sum_i lambda_i chi2_1 with
     lambda the eigenvalues of C; the p value comes from a saddlepoint (Lugannani-Rice) approximation.
     C depends on the features only, so it is estimated once per sex x sample x modality and shared by all
     terms.
  3. Secondary statistic max_j |z_j|: n_sim draws from N(0, C) give p_max; the number of features above the
     single-step max-T 95% threshold is n_fwer.
  4. m_eff = (sum lambda)^2 / sum lambda^2, the effective number of independent features (reported only).
  BH across terms within sex x sample x layer x modality x analysis x model; sig = q < alpha.
  A term is grey if fewer than omnibus.min_features non-grey features remain (WMH, a single feature, has no
  omnibus test). For exposure-years (X2) the Z used for C omits the pre-measurement CAMSIS column; its effect
  on C is negligible.

Output
  tables/m3b_omnibus.csv   m, m_eff, T, null_sd, p, max_abs_z, p_max, n_fwer, n_nom (features with nominal
                           p < 0.05), exp_nom (= 0.05 m), grey, q, sig

  python molecular/py/m3b_omnibus.py [--layers LB,LN,LP,LI] [--samples full]
  Requires m3 (reads m3_assoc).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import optimize, stats

from common_mol import (ARM_OF, add_q, design_mol, h1, h2, load_layer, need, ok, resid_on_Z, setup, sexes, tee,
                        warn)

OMNI_BY = ["sex", "sample", "layer", "modality", "analysis", "model"]


# ── Null distribution ────────────────────────────────────────────────
def z_cov(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Covariance of the per-feature z under H0: C_jk = sum over common observations of u_j u_k / sqrt(n_j n_k),
    u = residual on Z standardised over its own observations."""
    R = resid_on_Z(X, Z)
    fin = np.isfinite(R)
    nj = fin.sum(0).astype("float64")
    mu = np.where(nj > 0, np.nansum(R, 0) / np.maximum(nj, 1), 0.0)
    U = np.where(fin, R - mu, 0.0)
    sd = np.sqrt((U ** 2).sum(0) / np.maximum(nj, 1))
    U /= np.where(sd > 0, sd, 1.0)
    C = (U.T @ U) / np.sqrt(np.outer(np.maximum(nj, 1), np.maximum(nj, 1)))
    np.fill_diagonal(C, np.where(sd > 0, 1.0, 0.0))
    return C


def sp_sf(q: float, lam: np.ndarray) -> float:
    """P(sum lambda_i chi2_1 >= q) by the Lugannani-Rice saddlepoint approximation; near the mean, the
    Satterthwaite scaled chi-square is used instead."""
    lam = lam[lam > 1e-10 * lam.max()]
    mu, s2 = lam.sum(), 2 * (lam ** 2).sum()
    a, d = s2 / 2 / mu, mu ** 2 / (s2 / 2)
    if q <= 0:
        return 1.0
    if abs(q - mu) < 0.05 * np.sqrt(s2):
        return float(stats.chi2.sf(q / a, d))
    K = lambda s: -0.5 * np.sum(np.log1p(-2 * lam * s))           # noqa: E731
    K1 = lambda s: np.sum(lam / (1 - 2 * lam * s))                 # noqa: E731
    K2 = lambda s: np.sum(2 * lam ** 2 / (1 - 2 * lam * s) ** 2)   # noqa: E731
    if q > mu:
        lo, hi = 0.0, (1 - 1e-12) / (2 * lam.max())
    else:
        hi, lo = 0.0, -1.0
        while K1(lo) > q:
            lo *= 2
    s = optimize.brentq(lambda x: K1(x) - q, lo, hi, xtol=1e-14, maxiter=500)
    w = np.sign(s) * np.sqrt(max(2 * (s * q - K(s)), 0.0))
    v = s * np.sqrt(K2(s))
    p = stats.norm.sf(w) + stats.norm.pdf(w) * (1 / v - 1 / w)
    return float(np.clip(p, 0.0, 1.0))


class Null:
    """Null distribution on one feature set: eigenvalues (mean z^2) and simulated max |z|."""

    def __init__(self, C, n_sim, seed, batch=2000):
        w, V = np.linalg.eigh(C)
        keep = w > 1e-10 * max(w.max(), 1e-300)
        self.lam = w[keep]
        self.m = C.shape[0]
        self.m_eff = float(self.lam.sum() ** 2 / (self.lam ** 2).sum())
        self.null_sd = float(np.sqrt(2 * (self.lam ** 2).sum()) / self.m)
        L = V[:, keep] * np.sqrt(self.lam)                          # z* = L g
        rng = np.random.default_rng(seed)
        mx = []
        for b0 in range(0, n_sim, batch):
            g = rng.standard_normal((L.shape[1], min(batch, n_sim - b0)))
            mx.append(np.abs(L @ g).max(0))
        self.maxabs = np.sort(np.concatenate(mx))

    def p_mean(self, T):
        return sp_sf(T * self.m, self.lam)

    def p_max(self, x):
        n = len(self.maxabs)
        return (n - np.searchsorted(self.maxabs, x, side="left") + 1) / (n + 1)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    args, cfg, up, p = setup(__doc__, [(("--layers",), {"default": None}),
                                       (("--samples",), {"default": None})])
    oc = cfg["omnibus"]
    layers = args.layers.split(",") if args.layers else list(oc["layers"])
    samples = args.samples.split(",") if args.samples else list(oc["samples"])
    nom_p = cfg["nominal_p"]
    with tee(p, "m3b_omnibus"):
        h1(f"m3b omnibus tests (mean z^2, analytic null; max|z| from {oc['n_sim']} draws)")
        m3 = pd.read_csv(need(p["tables"] / "m3_assoc.csv.gz", "m3_assoc.py"), low_memory=False,
                         usecols=["sex", "layer", "modality", "sample", "analysis", "model", "term", "feature", "z",
                                  "p", "grey"])
        m3["grey"] = m3["grey"].astype(str).str.lower().isin(["true", "1"])
        rows = []
        t0 = time.time()
        for si, sex in enumerate(sexes(up)):
            for layer in layers:
                for smp in samples:
                    if smp != "full" and smp not in cfg["samples"][ARM_OF[layer]]:
                        continue
                    d = m3[(m3["sex"] == sex) & (m3["layer"] == layer) & (m3["sample"] == smp)]
                    if not len(d):
                        warn(f"{sex} {layer} {smp}: no m3 results; skipped")
                        continue
                    df, X, meta = load_layer(p, cfg, layer, sex, smp)
                    Z = design_mol(df, cfg)["Z"]
                    for mi, (mod, mm) in enumerate(meta.groupby("modality", sort=False)):
                        feats = [f for f in mm["feature"] if f in X.columns]
                        if len(feats) < oc["min_features"]:
                            continue
                        C = z_cov(X[feats].to_numpy("float64"), Z)
                        pos = {f: i for i, f in enumerate(feats)}
                        cache = {}
                        dm = d[(d["modality"] == mod) & d["feature"].isin(pos)]
                        h2(f"{sex} / {layer} / {mod} / {smp}: {len(df):,} participants x {len(feats)} features "
                           f"(C estimated, {time.time() - t0:.0f}s)")
                        for (an, mo, term), g in dm.groupby(["analysis", "model", "term"], sort=False):
                            g = g[~g["grey"] & np.isfinite(g["z"])]
                            base = {"sex": sex, "sample": smp, "layer": layer, "modality": mod, "analysis": an,
                                    "model": mo, "term": term, "m": len(g),
                                    "n_nom": int((g["p"] < nom_p).sum()), "exp_nom": nom_p * len(g)}
                            if len(g) < oc["min_features"]:
                                rows.append({**base, "grey": True})
                                continue
                            idx = tuple(sorted(pos[f] for f in g["feature"]))
                            if idx not in cache:
                                seed = int(oc["seed"]) + 1000 * si + 100 * mi + len(cache)
                                cache[idx] = Null(C[np.ix_(idx, idx)], int(oc["n_sim"]), seed)
                            nl = cache[idx]
                            z = g.set_index("feature")["z"].reindex([feats[i] for i in idx]).to_numpy()
                            T, mx = float(np.mean(z ** 2)), float(np.max(np.abs(z)))
                            q95 = float(np.quantile(nl.maxabs, 0.95))
                            rows.append({**base, "grey": False, "m_eff": nl.m_eff, "T": T, "null_sd": nl.null_sd,
                                         "p": nl.p_mean(T), "max_abs_z": mx, "p_max": float(nl.p_max(mx)),
                                         "n_fwer": int((np.abs(z) >= q95).sum())})
        if not rows:
            warn("no testable families")
            return
        ot = add_q(pd.DataFrame(rows), cfg["alpha"], by=OMNI_BY, col="p")
        cols = OMNI_BY + ["term", "m", "m_eff", "T", "null_sd", "p", "max_abs_z", "p_max", "n_fwer", "n_nom",
                          "exp_nom", "grey", "q", "sig"]
        ot = ot.reindex(columns=cols)
        ot.to_csv(p["tables"] / "m3b_omnibus.csv", index=False)
        g = ot[~ot["grey"].astype(bool)]
        h2(f"Summary (nominal p < 0.05 / BH q < {cfg['alpha']})")
        print(g.groupby(["sex", "sample", "layer", "modality", "analysis"]).agg(
            terms=("term", "size"), nominal=("p", lambda s: int((s < .05).sum())), fdr=("sig", "sum"),
            min_p=("p", "min")).to_string(float_format=lambda v: f"{v:.3g}"))
        ok(f"m3b done: {len(g)} family tests ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
