"""a4: career types and career axes vs clock age gaps, and the share of extreme agers by type.

Models, per sex x family x sample x clock: gapz ~ exposure + covariates (common_aging.design),
estimated by FWL with HC1 standard errors. Samples: for blood-measured families (proteomic and
blood-based), the left-work sample (restricted) and all participants (full); for MRI families, full.
Analyses (column analysis):
  type_vs_rest  each career type vs all other types of the same sex (one 0/1 indicator per model)
  continuous    each career axis in axes.continuous per SD (one axis per model)
Contrasts with fewer than qa.min_type_n participants in the type or among the others are not tested
(grey). BH-FDR within sex x family x sample x analysis x clock over tested cells, alpha = qa.fdr_alpha.
All clocks are analysed; column main carries the a3 validation status.

Extreme agers, per sex x family (blood-measured families) x sample x clock x type (+ All): participants
with gapz above the threshold (C-C: protein.ding.extreme_z; C-P: protein.oh.extreme_z; C-L:
blood.extreme_z; older side only), their share in the type and among the other types, two-sided Fisher's
exact test, BH within sex x family x sample x clock, same grey rule.

Tables (tables/): a4_assoc.csv, a4_extreme_share.csv

  python aging/py/a4_assoc_type.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from common_aging import (FAMILIES, assoc, design, fdr_within, h1, h2, load_frame, load_person, load_status, ok,
                          restrict, samples_for, setup, tee, type_names, vs_rest, warn, zstd)

TYPE_AN = "type_vs_rest"


def type_term(k) -> str:
    return f"type{int(k)}"


def exposure_sets(df, cfg, type_levels):
    """Yields (analysis, E, n_term rule). Types are compared with all other types, one model per type."""
    t = df["type"].astype("Int64")
    for k in type_levels:
        yield TYPE_AN, vs_rest(t, k, type_term(k)), "nonzero"
    for ax in cfg["axes"]["continuous"]:
        if ax not in df.columns:
            continue
        e = zstd(df[ax])
        if e.notna().sum() < 3:
            continue
        yield "continuous", pd.DataFrame({ax: e}, index=df.index), None


def type_levels_of(p, sex):
    """All type levels of the sex in the whole work-history cohort."""
    return sorted(load_person(p, sex)["type"].dropna().astype(int).unique())


def extreme_share(df, clocks, levels, thr, meta, min_type):
    """Per clock x type (+ All): n and % with gapz > thr, % among the other types, Fisher p, mean gapz."""
    t = df["type"].astype("Int64")
    tv = t.fillna(-1).astype(int).to_numpy()
    has_t = t.notna().to_numpy()
    rows = []
    for ck in clocks:
        if f"gapz__{ck}" not in df.columns:
            continue
        z = pd.to_numeric(df[f"gapz__{ck}"], errors="coerce").to_numpy("float64")
        base = np.isfinite(z) & has_t
        if base.sum() < 30:
            continue
        ext = z > thr
        N, E = int(base.sum()), int((ext & base).sum())
        rows.append({**meta, "clock": ck, "type": "All", "n": N, "n_extreme": E, "share_pct": 100 * E / N,
                     "n_rest": 0, "share_rest_pct": np.nan, "mean_gapz": float(z[base].mean()), "p": np.nan})
        for k in levels:
            sel = base & (tv == k)
            n, e = int(sel.sum()), int((ext & sel).sum())
            nr, er = N - n, E - e
            p_ = float(fisher_exact([[e, n - e], [er, nr - er]])[1]) if n and nr else np.nan
            rows.append({**meta, "clock": ck, "type": str(int(k)), "n": n, "n_extreme": e,
                         "share_pct": 100 * e / n if n else np.nan, "n_rest": nr,
                         "share_rest_pct": 100 * er / nr if nr else np.nan,
                         "mean_gapz": float(z[sel].mean()) if n else np.nan, "p": p_})
    out = pd.DataFrame(rows)
    if len(out):
        out["grey"] = (out["type"] != "All") & ((out["n"] < min_type) | (out["n_rest"] < min_type))
    return out


def extreme_threshold(cfg, fam):
    if fam == "C-L":
        return cfg["blood"]["extreme_z"]
    return cfg["protein"]["ding" if fam == "C-C" else "oh"]["extreme_z"]


# ── main ───────────────────────────────────────────────────────────
def main():
    args, cfg, up, p = setup(__doc__)
    qa = cfg["qa"]
    min_type = qa["min_type_n"]
    with tee(p, "a4_assoc_type"):
        h1("a4 career types and career axes x clocks; extreme agers")
        status = load_status(p)
        res_parts, xs_parts = [], []
        sexes_ = ["Female", "Male"]
        for sex in sexes_:
            levels = type_levels_of(p, sex)
            print(f"    {sex}: {len(levels)} types, each compared with all other types")
            for fam in FAMILIES:
                fs = status.get(fam, {}).get("_family", {})
                if not fs.get("present"):
                    warn(f"{fam}: absent from clock_status.json; skipped")
                    continue
                df0, clocks = load_frame(p, cfg, fam, sex)
                for smp in samples_for(fam, cfg):
                    df = restrict(df0, fam, smp)
                    h2(f"{sex} / {fam} / {smp}: n = {len(df):,}, {len(clocks)} clocks")
                    if len(df) < 50:
                        warn("sample too small; skipped")
                        continue
                    D = design(df, cfg, fam)
                    if D["dropped"]:
                        print(f"    collinear design columns dropped: {D['dropped'][:6]}"
                              f"{' ...' if len(D['dropped']) > 6 else ''}")
                    meta0 = {"sex": sex, "family": fam, "sample": smp, "arm": FAMILIES[fam]["arm"],
                             "timing": FAMILIES[fam]["timing"]}
                    if FAMILIES[fam]["arm"] in ("protein", "blood"):
                        thr = extreme_threshold(cfg, fam)
                        xs_parts.append(extreme_share(df, clocks, levels, thr, {**meta0, "extreme_z": thr}, min_type))
                    for an, E, nt in exposure_sets(df, cfg, levels):
                        r = assoc(df, D, clocks, E, {**meta0, "analysis": an}, status, fam, n_term=nt)
                        if len(r):
                            res_parts.append(r)

        h2("Type and axis associations: FDR")
        res = pd.concat(res_parts, ignore_index=True)
        res["grey"] = (res["analysis"] == TYPE_AN) & ((res["n_term"] < min_type)
                                                      | (res["n"] - res["n_term"] < min_type))
        res["q"] = fdr_within(res, ["sex", "family", "sample", "analysis", "clock"], ~res["grey"])
        for sex in sexes_:
            tn = type_names(p, sex)
            s = res["sex"] == sex
            tt = res.loc[s, "term"].str.extract(r"^type(\d+)$")[0]
            res.loc[s, "type_name"] = pd.to_numeric(tt, errors="coerce").map(tn)
        res["sig"] = (res["q"] < qa["fdr_alpha"]) & ~res["grey"]
        res.to_csv(p["tables"] / "a4_assoc.csv", index=False)
        print(res.groupby(["sex", "family", "sample", "analysis"])
              .agg(n_cells=("p", "size"), n_grey=("grey", "sum"), n_sig=("sig", "sum")).reset_index()
              .to_string(index=False))

        h2("Extreme agers by type (Fisher's exact test, BH within clock)")
        if xs_parts:
            xs = pd.concat(xs_parts, ignore_index=True)
            xs["main"] = [bool(status.get(f, {}).get(c, {}).get("main")) for f, c in zip(xs["family"], xs["clock"])]
            xs["q"] = fdr_within(xs, ["sex", "family", "sample", "clock"], ~xs["grey"] & (xs["type"] != "All"))
            xs["sig"] = (xs["q"] < qa["fdr_alpha"]) & ~xs["grey"]
            for sex in sexes_:
                tn = type_names(p, sex)
                s = xs["sex"] == sex
                xs.loc[s, "type_name"] = pd.to_numeric(xs.loc[s, "type"], errors="coerce").map(tn)
            xs.loc[xs["type"] == "All", "type_name"] = "All"
            xs.to_csv(p["tables"] / "a4_extreme_share.csv", index=False)
            print(xs[xs["type"] != "All"].groupby(["sex", "family", "sample"]).agg(
                n_cells=("p", "size"), n_grey=("grey", "sum"), n_sig=("sig", "sum")).to_string())
        ok("a4 done")


if __name__ == "__main__":
    main()
