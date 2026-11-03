#!/usr/bin/env python3
"""b4: topic burden and topic share by career type, with a global permutation test.

For person s with N_s projected diagnoses and topic weights theta_sk, the
outcome for topic k is y = theta_sk * N_s (expected number of diagnoses
attributed to the topic).
  burden: all cohort members (y = 0 without a projected diagnosis), no offset
          -> exp(beta) = topic burden ratio;
  share:  members with >= ob.min_dx_fit diagnoses, offset log N_s
          -> exp(beta) = topic share ratio (proportion of diagnoses in topic k).
More disease of every kind raises the burden of all topics without changing
shares; a shift towards particular diseases raises their shares.

Both use a log-link quasi-Poisson model with robust (sandwich) variance (the
same estimator as o2; the score and sandwich only involve y - mu, so
non-integer y is valid), one model per type: y ~ covariates + I(type == k),
i.e. each type vs all other types of the same sex. Cells of types with fewer
than ob.assoc.min_cell_n members are suppressed. BH-FDR within sex x estimand.

Global test: type labels permuted within deciles of age at time zero, using
covariate-adjusted observed/expected ratios with the covariate model fitted
once (robust SE per group: var(log O/E) ~ sum (y - mu)^2 / O^2).

Inputs: derived/op_person.parquet (o1), derived/<fit>/weights.csv.gz (b2),
        derived/ob_topics.json (b3).
Outputs: tables/OB4_assoc.csv, tables/OB5_global_null.csv, qc/ob_design.json,
         qc/ob_null_{sex}_{estimand}.npy.

  python b4_assoc.py [--fit ob_fit_atm_K10_published] [--stage all|fit|null]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from common_out import (Z975, bh_fdr, build_design, fail, h1, h2, irls_poisson,
                        load_cfg_out, load_cfg_up, ok, paths_out, sexes, tee,
                        write_json, z_to_p)

DEFAULT_FIT = "ob_fit_atm_K10_published"


# -- Inputs ------------------------------------------------------------------
def load_inputs(p, cfg, args):
    """Cohort covariates (o1), topic weights (b2) and topic names (b3)."""
    pf = p["derived"] / "op_person.parquet"
    if not pf.exists():
        fail(f"{pf.name} missing; run o1_assemble.py --stage timeline first")
        sys.exit(1)
    per = pd.read_parquet(pf).set_index("eid")
    wf = p["derived"] / args.fit / "weights.csv.gz"
    if not wf.exists():
        fail(f"{wf} missing; run R/b2_atm_fit.R first")
        sys.exit(1)
    w = pd.read_csv(wf).set_index("eid")
    tc = [c for c in w.columns if c.startswith("theta_")]
    K = len(tc)
    tj = p["derived"] / "ob_topics.json"
    if not tj.exists():
        fail("ob_topics.json missing; run b3_topics.py first")
        sys.exit(1)
    topics = pd.DataFrame(json.loads(tj.read_text(encoding="utf-8"))["topics"])
    if len(topics) != K:
        fail(f"ob_topics.json has {len(topics)} topics, weights have {K}")
        sys.exit(1)
    # Cohort members without projected weights (< 2 diagnoses) have burden 0
    # and are excluded from the share analysis.
    th = w[tc].reindex(per.index)
    n_dx = w["n_dx"].reindex(per.index).fillna(0).astype("float64")
    per = per.assign(n_dx=n_dx)
    Y = (th.to_numpy("float64") * n_dx.to_numpy()[:, None])
    Y = np.where(np.isfinite(Y), Y, 0.0)
    print(f"  cohort {len(per):,}; with topic weights {int(th.notna().all(1).sum()):,} "
          f"({th.notna().all(1).mean():.1%}); K={K}")
    return per, Y, topics, K


def sex_view(per, Y, sex):
    rows = np.flatnonzero((per["sex"] == sex).to_numpy())
    ps = per.iloc[rows].copy()
    types = sorted(int(t) for t in ps["type"].unique())
    lab = np.array([types.index(int(t)) for t in ps["type"]], dtype="int32")
    return {"rows": rows, "person": ps, "Y": Y[rows], "lab": lab, "types": types}


def _analysis_set(cfg, ps, estimand):
    """Rows and offset of an estimand: share = >= min_dx_fit diagnoses with log N offset."""
    n_dx = ps["n_dx"].to_numpy("float64")
    if estimand == "share":
        keep = n_dx >= float(cfg["ob"]["min_dx_fit"])
        off = np.log(np.clip(n_dx, 1e-9, None))
    else:
        keep = np.ones(len(ps), dtype=bool)
        off = None
    return keep, off


# -- Estimation ----------------------------------------------------------------
def fit_cells(cfg, v, sex, estimand) -> tuple[list, dict]:
    """All type x topic cells of one sex and estimand."""
    ob = cfg["ob"]["assoc"]
    ps, Y, lab = v["person"], v["Y"], v["lab"]
    keep, off = _analysis_set(cfg, ps, estimand)
    des = build_design(ps.loc[keep], cfg)
    X = des["X"]
    K = len(v["types"])
    out = []
    for j in range(Y.shape[1]):
        y = Y[keep, j]
        o = off[keep] if off is not None else None
        for k in range(K):
            m = lab[keep] == k
            n_k = int(m.sum())
            base = {"sex": sex, "estimand": estimand, "topic": j + 1,
                    "type_index": k, "type": v["types"][k], "n_type": n_k,
                    "n_analysed": int(keep.sum()),
                    "n_type_cohort": int((lab == k).sum()),
                    "mass": float(y[m].sum()), "mass_rest": float(y[~m].sum()),
                    "mean_y": float(y[m].mean()) if n_k else np.nan,
                    "mean_y_rest": float(y[~m].mean()) if n_k < keep.sum() else np.nan}
            if n_k < int(ob["min_cell_n"]) or n_k == int(keep.sum()):
                out.append({**base, "log_ratio": np.nan, "se": np.nan,
                            "ratio": np.nan, "lo": np.nan, "hi": np.nan,
                            "z": np.nan, "p": np.nan, "q": np.nan,
                            "converged": False, "suppressed": True})
                continue
            r = irls_poisson(y, np.column_stack([X, m.astype("float64")]),
                             offset=o, firth=False, robust=True)
            b, se = float(r["beta"][-1]), float(r["se"][-1])
            z = b / se if se > 0 else np.nan
            out.append({**base, "log_ratio": b, "se": se, "ratio": float(np.exp(b)),
                        "lo": float(np.exp(b - Z975 * se)),
                        "hi": float(np.exp(b + Z975 * se)), "z": z,
                        "p": float(z_to_p(np.array([z]))[0]), "q": np.nan,
                        "converged": bool(r["converged"]), "suppressed": False})
    return out, des


def stage_fit(p, cfg, cup, args, per, Y, topics) -> pd.DataFrame:
    ob = cfg["ob"]["assoc"]
    rows, design_log = [], {}
    for sex in sexes(cup):
        v = sex_view(per, Y, sex)
        h2(f"{sex} (K_type={len(v['types'])})")
        for est in ob["estimands"]:
            r, des = fit_cells(cfg, v, sex, est)
            rows += r
            design_log[f"{sex}|{est}"] = {"n_cov_cols": len(des["names"]),
                                          "dropped": des["dropped"],
                                          "age_knots": des["knots"],
                                          "refs": des["refs"]}
            nn = sum(1 for x in r if not x["suppressed"])
            print(f"    {est:<7} {len(r):>4} cells ({nn} estimated)")
    df = pd.DataFrame(rows)
    df["topic_name"] = df["topic"].map(dict(zip(topics["topic"], topics["name"])))
    df["fit"] = args.fit
    df["window"] = "lifetime"

    h2("FDR")
    df["family"] = df["sex"] + "|" + df["estimand"]
    for fam, g in df.groupby("family"):
        df.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
        sig = int((df.loc[g.index, "q"] < ob["fdr_alpha"]).sum())
        print(f"    {fam:<20} m={int(g['p'].notna().sum()):>4}  "
              f"q<{ob['fdr_alpha']}: {sig:>3}")

    f = p["tables"] / "OB4_assoc.csv"
    df.to_csv(f, index=False)
    ok(f"{f.name}  {len(df):,} rows")
    write_json(p["qc"] / "ob_design.json", design_log)
    return df


# -- Global permutation test ----------------------------------------------------
def _group(lab, K):
    from scipy.sparse import csr_matrix
    n = len(lab)
    return csr_matrix((np.ones(n, dtype="float64"), (lab, np.arange(n))),
                      shape=(K, n))


def _stat(lab, K, Y, MU, R2, alpha, min_n):
    """Energy (sum of squared log ratios) and number of FDR-significant cells."""
    G = _group(lab, K)
    O, E, S = G @ Y, G @ MU, G @ R2
    n_k = np.asarray(G.sum(1))
    Or, Er, Sr = O.sum(0) - O, E.sum(0) - E, S.sum(0) - S
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(np.clip(O, 1e-9, None) / np.clip(E, 1e-9, None)) \
            - np.log(np.clip(Or, 1e-9, None) / np.clip(Er, 1e-9, None))
        se = np.sqrt(S / np.clip(O, 1e-9, None) ** 2
                     + Sr / np.clip(Or, 1e-9, None) ** 2)
    m = (n_k >= min_n) & np.isfinite(lr) & np.isfinite(se) & (se > 0)
    z = np.where(m, lr / se, np.nan)
    q = bh_fdr(z_to_p(z))
    return (float(np.nansum(np.where(m, lr, np.nan) ** 2)),
            int(np.nansum(q < alpha)), lr, m)


def permute_within(lab, strata, rng):
    out = lab.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = lab[rng.permutation(idx)]
    return out


def stage_null(p, cfg, cup, per, Y) -> pd.DataFrame:
    ob = cfg["ob"]["assoc"]
    B = int(ob["null_n"])
    rng = np.random.default_rng(int(ob["null_seed"]))
    rows = []
    for sex in sexes(cup):
        v = sex_view(per, Y, sex)
        K = len(v["types"])
        for est in ob["estimands"]:
            ps = v["person"]
            keep, off = _analysis_set(cfg, ps, est)
            off = off[keep] if off is not None else None
            X = build_design(ps.loc[keep], cfg)["X"]
            Yk = v["Y"][keep]
            MU = np.empty_like(Yk)
            for j in range(Yk.shape[1]):
                r = irls_poisson(Yk[:, j], X, offset=off, robust=False)
                MU[:, j] = np.exp(np.clip(X @ r["beta"]
                                          + (off if off is not None else 0), -30, 30))
            R2 = (Yk - MU) ** 2
            lab = v["lab"][keep]
            strata = pd.qcut(ps.loc[keep, "tz_age"], int(ob["null_age_strata"]),
                             labels=False, duplicates="drop").to_numpy()
            obs, obs_sig, _, mask = _stat(lab, K, Yk, MU, R2,
                                          ob["fdr_alpha"], ob["min_cell_n"])
            nulls = np.empty((B, 2))
            for b in range(B):
                e, s, _, _ = _stat(permute_within(lab, strata, rng), K, Yk, MU, R2,
                                   ob["fdr_alpha"], ob["min_cell_n"])
                nulls[b] = (e, s)
            pe = (1 + int((nulls[:, 0] >= obs).sum())) / (B + 1)
            psig = (1 + int((nulls[:, 1] >= obs_sig).sum())) / (B + 1)
            rows.append({"sex": sex, "estimand": est, "n_cells": int(mask.sum()),
                         "energy_obs": obs, "energy_null_mean": nulls[:, 0].mean(),
                         "energy_null_sd": nulls[:, 0].std(ddof=1),
                         "energy_ratio": obs / max(nulls[:, 0].mean(), 1e-12),
                         "energy_p": pe, "nsig_obs": obs_sig,
                         "nsig_null_mean": nulls[:, 1].mean(), "nsig_p": psig,
                         "null_n": B})
            print(f"    {sex:<7}{est:<7} energy {obs:.3f} vs null "
                  f"{nulls[:, 0].mean():.3f} +/- {nulls[:, 0].std(ddof=1):.3f} "
                  f"(P={pe:.3f}); significant cells {obs_sig} vs "
                  f"{nulls[:, 1].mean():.1f} (P={psig:.3f})")
            np.save(p["qc"] / f"ob_null_{sex}_{est}.npy", nulls)
    nl = pd.DataFrame(rows)
    nl.to_csv(p["tables"] / "OB5_global_null.csv", index=False)
    ok("OB5_global_null.csv")
    return nl


def main():
    ap = argparse.ArgumentParser(description="Topic burden and share ratios by career type")
    ap.add_argument("--fit", default=DEFAULT_FIT, help="fit directory under derived/")
    ap.add_argument("--stage", default="all", choices=["fit", "null", "all"])
    args = ap.parse_args()
    cfg, cup = load_cfg_out(), load_cfg_up()
    p = paths_out(cfg)

    with tee(p, f"b4_assoc_{args.stage}"):
        h1(f"b4 topic associations: {args.fit}")
        h2("Inputs")
        per, Y, topics, K = load_inputs(p, cfg, args)
        if args.stage in ("fit", "all"):
            h1("Burden and share ratios")
            stage_fit(p, cfg, cup, args, per, Y, topics)
        if args.stage in ("null", "all"):
            h1("Global permutation test (labels permuted within age deciles)")
            stage_null(p, cfg, cup, per, Y)
        h2("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
