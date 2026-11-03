#!/usr/bin/env python3
"""Career types from the frozen work-history representations, their profiles and labels.

Stage `cluster` (per sex):
  * unweighted k-nearest-neighbour graph (k = cluster.knn_k) on the raw
    representation z. Clustering is done on z, not on a 2-d UMAP embedding,
    which preserves neither distances nor densities;
  * Leiden (RBConfiguration modularity) at a fixed, prespecified resolution
    (no search over resolutions);
  * communities smaller than max(min_cluster_frac * n, min_cluster_n) are merged
    into the community with the nearest centroid;
  * descriptive diagnostics: size entropy, average silhouette width (ASW), NMI of
    types with career-end decile and with modal occupation;
  * stability: bootstrap ARI (the full partition rule refitted on resamples) and
    ARI of partitions refitted within birth-cohort strata.

Stage `profiles` (per sex, uses only work-history inputs, never outcomes):
  P2 occupational composition and lift, P3 exposure enrichment, P4 CAMSIS status
  level and age profile, P5 work breaks, P7 between-type share of variance (eta^2)
  of candidate profile axes, then a fixed rule turns P2-P5 into a type label
  saved with a timestamp (`type_names_{sex}.json`).

Inputs (results/typology/derived/): latent_{sex}.parquet (+ .meta.json),
  cohort.parquet, worklife_annual.parquet, worklife_exposure.parquet,
  camsis.parquet, camsis_by_age.parquet, worklife_gaps.parquet; and
  data/<files.cov_baseline> (column year_birth).
Outputs:
  derived/types_{sex}.parquet          eid, type
  derived/type_names_{sex}.json        rule-based labels with `named_at`
  qc/cluster_summary_main.json         partition, dominance and stability numbers
  tables/P2_soc_mix_{sex}.csv          SOC major-group shares and lift
  tables/P2_soc_mix_l4_{sex}.csv       SOC unit-group shares and lift
  tables/P3_exposure_{sex}.csv         exposure profile and enrichment
  tables/P4_camsis_{sex}.csv           CAMSIS level, slopes, rising/flat/declining shares
  tables/P5_gaps_{sex}.csv             work-break and retirement summary
  tables/P7_axis_discrimination_{sex}.csv

Run:
  python typology/04_typology.py --stage cluster
  python typology/04_typology.py --stage profiles
  python typology/04_typology.py                    # both stages
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from common import (SOC_DIVISOR, fail, h1, h2, load_cfg, load_latent, ok, paths,
                    require_leiden, sexes, tee, warn)

SOC_L1 = {1: "Managers", 2: "Professional", 3: "AssocProf/Technical",
          4: "Admin/Secretarial", 5: "SkilledTrades", 6: "PersonalService",
          7: "Sales/Customer", 8: "Operatives", 9: "Elementary"}
GAP_RETIREMENT = 108


# ==================================================================
# A: Leiden partition
# ==================================================================
def knn_graph(z, knn, seed=42):
    """Unweighted, undirected kNN graph of z (igraph)."""
    import igraph as ig
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(z, knn, mode="connectivity", include_self=False)
    s, d = A.nonzero()
    g = ig.Graph(n=len(z), edges=list(zip(s.tolist(), d.tolist())), directed=False)
    g.simplify()
    return g


def leiden(z, res, knn=30, seed=42, graph=None):
    """Leiden with RBConfiguration (modularity with a resolution parameter)."""
    import leidenalg
    g = graph if graph is not None else knn_graph(z, knn, seed)
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=res, seed=seed)
    return np.asarray(part.membership)


def merge_small(lab, z, min_frac, min_n=0):
    """Merge clusters below max(min_frac * n, min_n) into the nearest cluster by
    centroid distance, then relabel 0..K-1.

    Leiden on a kNN graph always returns a few tiny communities; the absolute
    floor (min_n) is what keeps every type large enough to profile.
    """
    lab = np.asarray(lab).copy()
    n = len(lab)
    floor = max(min_frac * n, float(min_n))
    while True:
        vc = pd.Series(lab).value_counts()
        small = vc[vc < floor]
        if small.empty or len(vc) <= 2:
            break
        s = small.index[-1]
        big = [c for c in vc.index if c != s and vc[c] >= floor]
        if not big:
            break
        cs = z[lab == s].mean(0)
        lab[lab == s] = min(big,
                            key=lambda c: np.linalg.norm(z[lab == c].mean(0) - cs))
    u = {v: i for i, v in enumerate(sorted(set(lab)))}
    return np.array([u[v] for v in lab])


def asw(z, lab, max_n=20000, seed=42):
    """Average silhouette width, on a random subsample when n is large (O(n^2)).

    Descriptive only: distance concentration in high dimensions lowers ASW, so
    its absolute value is not comparable with values from other metric spaces.
    """
    from sklearn.metrics import silhouette_score
    if len(set(lab)) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    i = (rng.choice(len(z), max_n, replace=False) if len(z) > max_n
         else np.arange(len(z)))
    return (float(silhouette_score(z[i], lab[i])) if len(set(lab[i])) > 1
            else float("nan"))


def partition(z, cfg, seed, graph=None):
    """One full partition: Leiden at the fixed resolution, then merge_small."""
    cc = cfg["cluster"]
    raw = leiden(z, cc["resolution"], cc["knn_k"], seed, graph=graph)
    lab = merge_small(raw, z, cc["min_cluster_frac"], cc["min_cluster_n"])
    frac = pd.Series(lab).value_counts(normalize=True)
    cnt = pd.Series(lab).value_counts()
    return lab, {"k_raw": int(len(set(raw))), "k": int(len(set(lab))),
                 "min_n": int(cnt.min()), "max_frac": float(frac.max()),
                 "min_frac": float(frac.min()),
                 "entropy_ratio": _entropy_ratio(lab)}


def _entropy_ratio(lab):
    """Size entropy H / log(K) in (0, 1]; 1 = equally sized types."""
    f = pd.Series(np.asarray(lab)).value_counts(normalize=True).to_numpy()
    k = len(f)
    return 0.0 if k < 2 else float(-(f * np.log(f)).sum() / np.log(k))


def cluster_dominance(lab, p, cfg, eids):
    """Cluster-level dominance diagnostics (reported, never used to filter).

    Asks whether the partition is a restatement of career length (NMI with
    career-end decile) or of occupation title (NMI with modal SOC code, and the
    largest within-type share of a single modal code).
    """
    from sklearn.metrics import normalized_mutual_info_score as nmi
    d = p["derived"]
    lab = np.asarray(lab)
    n = len(lab)
    res = {}

    vc = pd.Series(lab).value_counts()
    top = float(vc.iloc[0] / n)
    res["largest_cluster_frac"] = top
    print(f"  Largest type: {top:.1%} (K={len(vc)})")
    if top > cfg["cluster"]["warn_largest_cluster_frac"]:
        warn(f"largest type holds {top:.1%} of the sample")

    try:
        coh = pd.read_parquet(d / "cohort.parquet").set_index("eid").reindex(eids)
        ce = coh["career_end"].to_numpy()
        good = np.isfinite(ce)
        dec = (pd.qcut(ce[good], 10, labels=False, duplicates="drop")
               if good.sum() > 100 else None)
        if dec is not None and pd.isna(dec).all():
            warn("career_end is near-constant; NMI with its deciles not computed")
        elif dec is not None:
            dec = np.where(pd.isna(dec), -1, dec)     # unassigned values form one class
            v = float(nmi(lab[good], dec))
            res["nmi_career_end_decile"] = v
            thr = cfg["cluster"]["warn_nmi_career_end"]
            (warn if v > thr else ok)(
                f"NMI(types, career-end decile) = {v:.3f} (reference <= {thr})")
    except Exception as e:
        warn(f"career-end NMI skipped: {e}")

    try:
        msoc = mode_soc(p, eids)
        good = pd.notna(msoc)
        if good.sum() > 100:
            v = float(nmi(lab[good], msoc[good].astype(str)))
            res["nmi_mode_soc"] = v
            df = pd.DataFrame({"lab": lab[good], "soc": msoc[good]})
            pur = df.groupby("lab")["soc"].agg(
                lambda s: s.value_counts().iloc[0] / len(s))
            res["max_cluster_soc_purity"] = float(pur.max())
            thr = cfg["cluster"]["warn_nmi_mode_soc"]
            (warn if v > thr else ok)(
                f"NMI(types, modal SOC) = {v:.3f} (reference <= {thr})")
            pthr = cfg["cluster"]["warn_cluster_soc_purity"]
            hi = pur[pur > pthr]
            if len(hi):
                warn(f"{len(hi)} type(s) with modal-SOC share > {pthr:.0%} "
                     f"(max {pur.max():.0%}; types {list(hi.index)})")
    except Exception as e:
        warn(f"modal-SOC NMI skipped: {e}")
    return res


def mode_soc(p, eids):
    """Each person's modal SOC code over in-job person-years."""
    ann = pd.read_parquet(p["derived"] / "worklife_annual.parquet",
                          columns=["eid", "soc4", "in_job"])
    ann = ann[ann["in_job"] == 1]
    return (ann.groupby(["eid", "soc4"]).size().reset_index(name="n")
            .sort_values("n").groupby("eid").tail(1)
            .set_index("eid")["soc4"].reindex(eids).to_numpy())


# --- stability ---
def s1_bootstrap(z, lab, refit, n, seed=42):
    """Bootstrap stability: refit the full partition rule on each resample and
    compare with the main partition by ARI on the unique resampled individuals.

    `refit` must be the main rule (same resolution, kNN and merge_small);
    otherwise the comparison is between two different procedures.
    """
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        b = rng.choice(len(z), len(z), replace=True)
        lb = refit(z[b])
        u, first = np.unique(b, return_index=True)
        out.append(adjusted_rand_score(lab[u], lb[first]))
    a = np.array(out)
    return float(np.median(a)), float(a.std())


def s3_cohort(z, coh, lab, refit, split=1950):
    """Birth-cohort stability: refit within those born before / from `split`
    and compare each with the main partition by ARI; returns the mean ARI."""
    from sklearn.metrics import adjusted_rand_score
    yb = pd.to_numeric(
        coh["birth_cohort_5y"].astype(str).str.extract(r"(\d{4})")[0],
        errors="coerce").to_numpy()
    out = {}
    for nm, m in (("early", yb < split), ("late", yb >= split)):
        if m.sum() < 500:
            continue
        lb = refit(z[m])
        out[nm] = {"n": int(m.sum()), "k": int(len(set(lb))),
                   "ari": float(adjusted_rand_score(lab[m], lb))}
    if not out:
        return float("nan"), out
    return float(np.mean([v["ari"] for v in out.values()])), out


def run_cluster(p, cfg):
    """Stage 1: partition, diagnostics and stability; writes types_{sex}.parquet."""
    seed = cfg["seed"]
    cc = cfg["cluster"]
    thr = cc["min_ari"]
    coh_all = pd.read_parquet(p["derived"] / "cohort.parquet").set_index("eid")
    summary = {}

    for sex in sexes(cfg):
        f = p["derived"] / f"latent_{sex}.parquet"
        if not f.exists():
            fail(f"missing {f.name}; run 03_pretrain.py first")
            continue
        eids, z, _ = load_latent(f)
        coh = coh_all.reindex(eids)
        h2(f"{sex} ({len(z):,} x {z.shape[1]})")

        graph = knn_graph(z, cc["knn_k"], seed)
        lab, info = partition(z, cfg, seed, graph=graph)
        K = info["k"]
        vc = pd.Series(lab).value_counts().sort_index()
        print(f"  Leiden res={cc['resolution']}, {cc['knn_k']}-NN graph: "
              f"{info['k_raw']} communities -> K={K} after merging")
        print(f"    sizes {[int(v) for v in vc.to_numpy()]}; "
              f"size entropy H/logK={info['entropy_ratio']:.3f}")
        small = vc[vc < cc["small_cluster_warn_n"]]
        if len(small):
            warn(f"{len(small)} type(s) with < {cc['small_cluster_warn_n']} members")
        # merge_small stops at K=2, so the floor can remain unmet there
        floor = max(cc["min_cluster_frac"] * len(z), cc["min_cluster_n"])
        if info["min_n"] < floor:
            warn(f"smallest type {info['min_n']} < floor {floor:.0f} (K={K})")
        a_val = asw(z, lab, seed=seed)
        print(f"  ASW = {a_val:.3f}")

        h2(f"{sex}: dominance diagnostics")
        dom = cluster_dominance(lab, p, cfg, eids)

        h2(f"{sex}: bootstrap and birth-cohort stability")

        def refit(zz, _c=cfg, _s=seed):
            return partition(zz, _c, _s)[0]

        m, sd = s1_bootstrap(z, lab, refit, cc["bootstrap_n"], seed)
        st = {"K": K, "asw": a_val,
              "sizes": {int(k): int(v) for k, v in vc.items()},
              **{f"info_{k}": v for k, v in info.items()},
              **{f"dom_{k}": v for k, v in dom.items()},
              "S1_bootstrap_ari": m, "S1_bootstrap_sd": sd}
        (ok if m >= thr else warn)(
            f"bootstrap ARI median {m:.3f} +/- {sd:.3f} "
            f"({cc['bootstrap_n']} resamples; reference >= {thr})")
        s3m, s3d = s3_cohort(z, coh, lab, refit, cc["birth_cohort_split"])
        st["S3_birth_cohort_ari"] = s3m
        st["S3_strata"] = s3d
        print(f"  birth-cohort strata (split {cc['birth_cohort_split']}): mean ARI "
              f"{s3m:.3f}; " + "; ".join(
                  f"{k} n={v['n']:,} K={v['k']} ARI={v['ari']:.3f}"
                  for k, v in s3d.items()))

        out = pd.DataFrame({"eid": eids, "type": lab})
        out.to_parquet(p["derived"] / f"types_{sex}.parquet", index=False)
        ok(f"types_{sex}.parquet  K={K}")
        summary[sex] = st

    (p["qc"] / "cluster_summary_main.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    ok("cluster_summary_main.json")
    return summary


# ==================================================================
# B: type profiles
# ==================================================================
def exposure_axes(expo, cfg, eids):
    """Exposure breadth (number of agents with any exposure year) and the
    longest single-agent exposure (years).

    People without any exposure row (no in-job years) are missing, not zero.
    """
    names = list(cfg["employment"]["exposures"])
    e = expo.reindex(eids)
    yrs = e[[f"yrs_exp_{n}" for n in names]].to_numpy(dtype="float64")
    allna = np.isnan(yrs).all(axis=1)
    yrs0 = np.nan_to_num(yrs, nan=0.0)
    breadth = (yrs0 > 0).sum(axis=1).astype("float64")
    longest = yrs0.max(axis=1)
    breadth[allna] = np.nan
    longest[allna] = np.nan
    return breadth, longest, int(allna.sum())


def p2_soc(ann, coh, types, p, sex, cfg):
    """P2: occupational composition at SOC major group (L1) and at the model's
    code level, with lift = share in the type / share in the same-sex cohort.

    Lift identifies the characteristic rather than the most common occupations;
    a minimum within-type share (profiles.soc_lift_min_frac) guards against
    rare codes with inflated lift. Person-years are right-truncated at career
    end, as in the tokens.
    """
    a = ann[(ann["in_job"] == 1) & ann["eid"].isin(types.index)].copy()
    ce = coh["career_end"].reindex(a["eid"]).to_numpy()
    a = a[a["age"].to_numpy() <= ce]
    a["type"] = types.reindex(a["eid"]).to_numpy()
    soc = pd.to_numeric(a["soc4"], errors="coerce")
    a["l1"] = soc // 1000
    a["fine"] = soc // SOC_DIVISOR.get(cfg["model"]["soc_level"], 1)

    tab = a.groupby(["type", "l1"]).size().rename("n").reset_index()
    tab["frac"] = tab["n"] / tab.groupby("type")["n"].transform("sum")
    base1 = a.groupby("l1").size() / len(a)
    tab["lift"] = tab["frac"] / tab["l1"].map(base1)
    tab["l1_name"] = tab["l1"].map(SOC_L1)
    tab.to_csv(p["tables"] / f"P2_soc_mix_{sex}.csv", index=False)
    piv = tab.pivot(index="type", columns="l1_name", values="frac").fillna(0)

    lvl = cfg["model"]["soc_level"].replace("soc_", "").upper()
    t2 = a.groupby(["type", "fine"]).size().rename("n").reset_index()
    t2["frac"] = t2["n"] / t2.groupby("type")["n"].transform("sum")
    base2 = a.groupby("fine").size() / len(a)
    t2["lift"] = t2["frac"] / t2["fine"].map(base2)
    t2["code_name"] = t2["fine"].map(
        lambda v: f"SOC{int(v)}" if pd.notna(v) else "NA")
    t2["soc_level"] = lvl
    t2.to_csv(p["tables"] / f"P2_soc_mix_{lvl.lower()}_{sex}.csv", index=False)

    fmin = cfg["profiles"]["soc_lift_min_frac"]
    lift_top = {}
    for t in sorted(types.unique()):
        q = t2[(t2["type"] == t) & (t2["frac"] >= fmin)].nlargest(3, "lift")
        lift_top[int(t)] = q
        print(f"    type {t}: " + "; ".join(
            f"{r.code_name} lift {r.lift:.1f} ({r.frac:.0%})" for r in q.itertuples()))
    return piv, t2, lift_top


def p3_exposure(expo, types, p, cfg, sex):
    """P3: exposure profile per type.

    'High exposure' is defined by enrichment relative to the same-sex cohort
    (share ever rated 'often' in the type / in the cohort), because the cohort
    prevalence of the ten agents differs by two orders of magnitude and an
    absolute share threshold would flag none of them.
    """
    names = list(cfg["employment"]["exposures"])
    e = expo.reindex(types.index)
    base_rate = {nm: float((e[f"peak_{nm}"] == 2).mean()) for nm in names}
    breadth, longest, n_na = exposure_axes(expo, cfg, types.index)
    if n_na:
        warn(f"{n_na} people without exposure records (treated as missing)")
    b_idx = pd.Series(breadth, index=e.index)
    l_idx = pd.Series(longest, index=e.index)
    rows = []
    for t in sorted(types.unique()):
        m = types == t
        r = {"type": int(t), "n": int(m.sum()),
             "work_years_median": float(e.loc[m, "work_years"].median()),
             "breadth_median": float(b_idx[m.to_numpy()].median()),
             "longest_agent_years_median": float(l_idx[m.to_numpy()].median())}
        for nm in names:
            pk = float((e.loc[m, f"peak_{nm}"] == 2).mean())
            r[f"mean_{nm}"] = float(e.loc[m, f"mean_{nm}"].mean())
            r[f"peak_{nm}"] = pk
            br = base_rate[nm]
            r[f"enrich_{nm}"] = float(pk / br) if br > 0 else float("nan")
            r[f"yrs_high_{nm}"] = float(e.loc[m, f"yrs_high_{nm}"].median())
        for c in ("long_hours_years", "night_shift_years"):
            if c in e.columns:
                r[c] = float(e.loc[m, c].median())
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.attrs["base_rate"] = base_rate
    tab.attrs["breadth_cohort_median"] = float(np.nanmedian(breadth))
    tab.attrs["longest_cohort_median"] = float(np.nanmedian(longest))
    tab.to_csv(p["tables"] / f"P3_exposure_{sex}.csv", index=False)

    ethr = cfg["profiles"]["peak_exposure_enrichment"]
    nmin = cfg["profiles"]["peak_exposure_min_frac"]
    for r in tab.itertuples():
        hi = [nm for nm in names
              if (getattr(r, f"enrich_{nm}") >= ethr
                  and getattr(r, f"peak_{nm}") >= nmin)]
        print(f"    type {r.type}: breadth {r.breadth_median:.0f}; enriched: "
              + (", ".join(f"{nm} {getattr(r, f'enrich_{nm}'):.1f}x" for nm in hi)
                 or "none"))
    return tab


ZERO_SLOPE_EPS = 1e-9    # polyfit on a constant series returns ~1e-16, not 0
CAMSIS_SPLIT_AGE = 35    # conventional early/later career boundary, not tuned


def _wslope(x, y, w, min_pts=5, min_span=5.0):
    """Weighted least-squares slope cov_w(x, y) / var_w(x).

    Weights are the number of type members in work at each age, so sparse ages
    at the ends of the curve cannot dominate the slope.
    """
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < min_pts or np.ptp(x[m]) < min_span:
        return np.nan
    x, y, w = x[m], y[m], w[m]
    w = w / w.sum()
    mx, my = float((w * x).sum()), float((w * y).sum())
    vx = float((w * (x - mx) ** 2).sum())
    return float((w * (x - mx) * (y - my)).sum() / vx) if vx > 0 else np.nan


def camsis_age_curves(p, types, min_n=100):
    """Mean CAMSIS by age per type (ages with >= min_n members in work) and the
    cohort mean by age."""
    f = p["derived"] / "camsis_by_age.parquet"
    if not f.exists():
        return None, None
    d = pd.read_parquet(f)
    d = d[d["eid"].isin(types.index)].copy()
    if d.empty:
        return None, None
    d["type"] = types.reindex(d["eid"]).to_numpy()
    g = (d.groupby(["type", "age"])["camsis"].agg(["mean", "size"])
         .reset_index())
    g = g[g["size"] >= min_n]
    return (g if len(g) else None), d.groupby("age")["camsis"].mean()


def _curve_stats(g, split_age=CAMSIS_SPLIT_AGE):
    """Per-type curve slopes (whole, early < split_age, later >= split_age) and
    the change from the first to the last age shown (total gain)."""
    rows = {}
    for t, q in g.groupby("type"):
        q = q.sort_values("age")
        x, y, w = (q["age"].to_numpy(float), q["mean"].to_numpy(float),
                   q["size"].to_numpy(float))
        e, l = x < split_age, x >= split_age
        rows[t] = {
            "slope_curve": _wslope(x, y, w),
            "slope_curve_early": _wslope(x[e], y[e], w[e], min_pts=4,
                                         min_span=4.0),
            "slope_curve_late": _wslope(x[l], y[l], w[l]),
            "curve_first_age": float(x[0]), "curve_last_age": float(x[-1]),
            "curve_change": float(y[-1] - y[0]),
            "curve_min": float(y.min()), "curve_max": float(y.max())}
    return pd.DataFrame(rows).T.rename_axis("type")


def p4_camsis(p, types, cfg, sex, min_n=100):
    """P4: CAMSIS occupational status per type.

    The type `pattern` (rising / flat / declining) is the weighted OLS slope of
    the type's mean-CAMSIS-by-age curve, with |slope| < profiles.camsis_slope_flat
    read as flat. This curve is cross-sectional (each age averages those still
    in work), so the within-person view is reported alongside it: the mean
    individual slope and the shares of members with rising / flat / declining
    individual slopes (denominator: members with a defined slope). The median
    individual slope is not used for labelling because members with a single
    lifetime occupation code (`pct_zero_slope`) have a slope of exactly 0 and
    often occupy the median.
    """
    f = p["derived"] / "camsis.parquet"
    if not f.exists():
        warn("missing camsis.parquet; P4 skipped")
        return None
    cam = pd.read_parquet(f).set_index("eid").reindex(types.index)
    tab = pd.DataFrame({"type": types}).join(cam)
    flat = cfg["profiles"]["camsis_slope_flat"]
    if "camsis_pattern" not in tab.columns:
        sl_ = tab["camsis_slope"]
        tab["camsis_pattern"] = np.where(
            sl_ > flat, "rising", np.where(sl_ < -flat, "declining", "flat"))
        tab.loc[sl_.isna(), "camsis_pattern"] = np.nan
    agg = tab.groupby("type").agg(
        camsis_mean=("camsis_mean", "mean"), camsis_sd=("camsis_mean", "std"),
        slope_median=("camsis_slope", "median"),
        slope_mean_within=("camsis_slope", "mean"),
        slope_sd=("camsis_slope", "std"), n=("camsis_mean", "size"),
        n_slope_defined=("camsis_slope", "count"),
        pct_zero_slope=("camsis_slope",
                        lambda s: float((s.abs() < ZERO_SLOPE_EPS).mean())))
    # value_counts drops NaN, so the denominator is members with a defined slope
    comp = (tab.groupby("type")["camsis_pattern"]
            .value_counts(normalize=True).unstack(fill_value=0.0)
            .reindex(columns=["rising", "flat", "declining"], fill_value=0.0)
            .rename(columns=lambda c: f"pct_{c}"))
    agg = agg.join(comp)

    curves, _ = camsis_age_curves(p, types, min_n)
    if curves is not None:
        agg = agg.join(_curve_stats(curves))
        agg["pattern"] = np.where(agg["slope_curve"] > flat, "rising",
                                  np.where(agg["slope_curve"] < -flat,
                                           "declining", "flat"))
        agg.loc[~np.isfinite(agg["slope_curve"]), "pattern"] = "undetermined"
    else:
        warn("missing camsis_by_age.parquet; pattern falls back to the median "
             "individual slope")
        agg["slope_curve"] = np.nan
        agg["pattern"] = np.where(agg["slope_median"] > flat, "rising",
                                  np.where(agg["slope_median"] < -flat,
                                           "declining", "flat"))
    agg.attrs["cohort_mean"] = float(cam["camsis_mean"].mean())
    agg.attrs["cohort_sd"] = float(cam["camsis_mean"].std())
    agg.attrs["slope_flat"] = float(flat)
    agg.attrs["split_age"] = float(CAMSIS_SPLIT_AGE)
    agg.to_csv(p["tables"] / f"P4_camsis_{sex}.csv")
    for t, r in agg.iterrows():
        print(f"    type {t}: mean {r['camsis_mean']:.1f}, curve slope "
              f"{r.get('slope_curve', np.nan):+.3f}/yr -> {r['pattern']}; "
              f"rising {r.get('pct_rising', np.nan):.0%} / "
              f"flat {r.get('pct_flat', np.nan):.0%} / "
              f"declining {r.get('pct_declining', np.nan):.0%}")
    return agg


def p5_gaps(p, cfg, types, coh, base, sex):
    """P5: work breaks per type.

    Breaks are right-truncated at career end, as in the tokens. Retirement
    (gap code 108) is the end-point event, not a break: it is excluded from
    break counts and years and summarised separately as observed retirement.
    """
    f = p["derived"] / "worklife_gaps.parquet"
    if not f.exists():
        warn("missing worklife_gaps.parquet; P5 skipped")
        return None
    g = pd.read_parquet(f)
    g = g[g["eid"].isin(types.index)].copy()
    g["type"] = types.reindex(g["eid"]).to_numpy()
    yb = base["year_birth"].reindex(g["eid"]).to_numpy()
    g["age_start"] = pd.to_numeric(g["start"], errors="coerce") - yb
    g["age_end"] = pd.to_numeric(g["end"], errors="coerce") - yb
    g["cut"] = coh["career_end"].reindex(g["eid"]).to_numpy()
    # keep only the part before career end; drop spells starting after it
    g = g[g["age_start"] <= g["cut"]].copy()
    g["years"] = (np.minimum(g["age_end"], g["cut"])
                  - g["age_start"] + 1).clip(lower=0)
    is_ret = g["gap_code"] == GAP_RETIREMENT
    brk, ret = g[~is_ret], g[is_ret]

    hc = cfg["codings"]["gap_health"]
    rows = []
    for t in sorted(types.unique()):
        ids = types.index[types == t]
        n = len(ids)
        sb, sr = brk[brk["type"] == t], ret[ret["type"] == t]
        obs_ret = coh["observed_retirement"].reindex(ids)
        rows.append({
            "type": int(t), "n_people": n,
            "breaks_per_person": len(sb) / max(n, 1),
            "break_years_median": (float(sb.groupby("eid")["years"].sum().median())
                                   if len(sb) else 0.0),
            "break_years_p75": (float(sb.groupby("eid")["years"].sum()
                                      .quantile(.75)) if len(sb) else 0.0),
            "first_break_age_median": (
                float(sb.groupby("eid")["age_start"].min().median())
                if len(sb) else np.nan),
            "pct_observed_retirement": float(obs_ret.mean()),
            "retire_age_median": (
                float(sr.groupby("eid")["age_start"].min().median())
                if len(sr) else np.nan),
            "pct_health_gap": float(sb.loc[sb["gap_code"] == hc, "eid"].nunique()
                                    / max(n, 1)),
            **{f"gap_{c}_frac": float(
                sb["gap_code"].value_counts(normalize=True).get(c, 0.0))
               for c in cfg["codings"]["gap_codes"]
               if c > 0 and c != GAP_RETIREMENT},
        })
    tab = pd.DataFrame(rows)
    # long breaks: at least 1 year above the median across types (a P75 rule
    # flags most types when values are tied)
    med = float(tab["break_years_median"].median())
    thr = med + 1.0
    tab["long_breaks"] = (tab["break_years_median"] >= thr).astype(int)
    tab.attrs["long_break_threshold"] = thr
    tab.to_csv(p["tables"] / f"P5_gaps_{sex}.csv", index=False)
    print(f"  long-break threshold >= {thr:.1f} years: "
          f"{int(tab['long_breaks'].sum())}/{len(tab)} types")
    return tab


def _eta2(v, g):
    """Between-type share of variance (one-way eta^2) of v."""
    v = np.asarray(v, dtype="float64")
    g = np.asarray(g)
    m = np.isfinite(v)
    v, g = v[m], g[m]
    if len(v) < 10:
        return float("nan")
    gm = v.mean()
    tot = float(((v - gm) ** 2).sum())
    if tot <= 0:
        return float("nan")
    ssb = sum(len(v[g == k]) * (v[g == k].mean() - gm) ** 2 for k in np.unique(g))
    return float(ssb / tot)


def axis_discrimination(types, coh, expo, p, cfg, sex):
    """P7: eta^2 of content axes (status, exposure) and timing axes (age,
    career length) across types. Reported only; never used to choose the
    partition, since selecting on CAMSIS separation would make the status
    profiles circular.
    """
    g = types.to_numpy()
    idx = types.index
    co = coh.reindex(idx)
    ex = expo.reindex(idx)
    breadth, longest, _ = exposure_axes(expo, cfg, idx)
    axes = {
        "exposure_breadth": (breadth, "content"),
        "longest_agent_years": (longest, "content"),
        "work_years": (ex["work_years"].to_numpy("float64"), "timing"),
        "career_end": (co["career_end"].to_numpy("float64"), "timing"),
        "age_recruit": (co["age_recruit"].to_numpy("float64"), "timing"),
        "years_out_at_recruit": ((co["age_recruit"] - co["career_end"])
                                 .to_numpy("float64"), "timing"),
    }
    # retirement age among observed retirees only (career_end mixes retirement
    # with administrative censoring)
    if "retire_age" in co.columns:
        axes["retire_age_retirees_only"] = (
            pd.to_numeric(co["retire_age"], errors="coerce").to_numpy("float64"),
            "timing")
    cf = p["derived"] / "camsis.parquet"
    if cf.exists():
        cam = pd.read_parquet(cf).set_index("eid").reindex(idx)
        axes["camsis_mean"] = (cam["camsis_mean"].to_numpy("float64"), "content")
        axes["camsis_slope"] = (cam["camsis_slope"].to_numpy("float64"), "content")
    else:
        warn("missing camsis.parquet; CAMSIS axes omitted from P7")

    prim = list(cfg["profiles"].get("primary_axes") or [])
    rows = []
    for nm, (v, kind) in axes.items():
        vv = np.asarray(v, dtype="float64")
        fin = vv[np.isfinite(vv)]
        med = pd.Series(vv).groupby(g).median()
        rows.append({"axis": nm, "kind": kind, "eta2": _eta2(vv, g),
                     "type_median_range": float(med.max() - med.min()),
                     "cohort_sd": float(fin.std()) if len(fin) else float("nan"),
                     "is_primary": nm in prim})
    tab = pd.DataFrame(rows).sort_values("eta2", ascending=False)
    tab.to_csv(p["tables"] / f"P7_axis_discrimination_{sex}.csv", index=False)
    for r in tab.itertuples():
        print(f"    {r.axis:<24} {r.kind:<8} eta2={r.eta2:.3f}")
    con = tab[tab["kind"] == "content"]["eta2"]
    tim = tab[tab["kind"] == "timing"]["eta2"]
    if len(con) and len(tim) and np.isfinite(con.max()) and np.isfinite(tim.max()):
        (ok if con.max() >= tim.max() else warn)(
            f"max eta2: content {con.max():.3f}, timing {tim.max():.3f}")
    return tab


def suggest_names(soc_l1, lift_top, expo, cam, gaps, cfg):
    """Fixed labelling rule built only from P2-P5 (no outcome is ever used).

    Label = top-2 unit groups by lift [dominant major group], then, if met:
    enriched agents (top 2 by enrichment), broad/narrow exposure (breadth at
    least 1 agent above/below the cohort median), high/low status (type mean
    CAMSIS >= +0.5 / <= -0.5 SD of the same-sex cohort), rising/declining status
    trajectory, and long breaks. Career end is not used.
    """
    names_ = list(cfg["employment"]["exposures"])
    ethr = cfg["profiles"]["peak_exposure_enrichment"]
    nmin = cfg["profiles"]["peak_exposure_min_frac"]
    out = {}
    for t in soc_l1.index:
        l1 = str(soc_l1.loc[t].idxmax())
        q = lift_top.get(int(t))
        if q is not None and len(q):
            bits = ["+".join(str(r.code_name) for r in q.head(2).itertuples())
                    + f" [{l1}]"]
        else:
            bits = [l1]
        if expo is not None:
            r = expo[expo["type"] == t]
            if len(r):
                cand = [(n, float(r[f"enrich_{n}"].iloc[0])) for n in names_
                        if (f"enrich_{n}" in r.columns
                            and float(r[f"enrich_{n}"].iloc[0]) >= ethr
                            and float(r[f"peak_{n}"].iloc[0]) >= nmin)]
                cand.sort(key=lambda kv: -kv[1])
                if cand:
                    bits.append("high " + "+".join(f"{n}({v:.1f}x)"
                                                   for n, v in cand[:2]))
                bm = expo.attrs.get("breadth_cohort_median")
                if bm is not None and "breadth_median" in r.columns:
                    bv = float(r["breadth_median"].iloc[0])
                    unit = "agent" if abs(bv - 1) < 1e-9 else "agents"
                    if bv >= bm + 1:
                        bits.append(f"broad exposure ({bv:.0f} {unit})")
                    elif bv <= bm - 1:
                        bits.append(f"narrow exposure ({bv:.0f} {unit})")
        if cam is not None and t in cam.index:
            # status level relative to the same-sex cohort (CAMSIS is sex-specific)
            cm, cs = cam.attrs.get("cohort_mean"), cam.attrs.get("cohort_sd")
            cv = float(cam.loc[t, "camsis_mean"])
            if cm is not None and cs and cs > 0:
                dz = (cv - cm) / cs
                if dz >= 0.5:
                    bits.append(f"high status (CAMSIS {cv:.0f})")
                elif dz <= -0.5:
                    bits.append(f"low status (CAMSIS {cv:.0f})")
            if str(cam.loc[t, "pattern"]) not in ("flat", "undetermined", "nan"):
                bits.append(f"{cam.loc[t, 'pattern']} status")
        if gaps is not None:
            r = gaps[gaps["type"] == t]
            if len(r) and int(r["long_breaks"].iloc[0]):
                bits.append("long breaks")
        out[str(int(t))] = ", ".join(bits)
    return out


def run_profiles(p, cfg):
    """Stage 2: P2-P5, P7 and the timestamped label file (written before any
    downstream analysis)."""
    d = p["derived"]
    ann = pd.read_parquet(d / "worklife_annual.parquet",
                          columns=["eid", "age", "soc4", "in_job"])
    expo = pd.read_parquet(d / "worklife_exposure.parquet").set_index("eid")
    coh = pd.read_parquet(d / "cohort.parquet").set_index("eid")
    b = pd.read_csv(p["data"] / cfg["files"]["cov_baseline"]).set_index("eid")
    base = coh.join(b[["year_birth"]], how="left")

    for sex in sexes(cfg):
        tf = d / f"types_{sex}.parquet"
        if not tf.exists():
            fail(f"missing {tf.name}; run --stage cluster first")
            continue
        types = pd.read_parquet(tf).set_index("eid")["type"]
        h2(f"{sex}: K={types.nunique()}, n={len(types):,}")

        h2(f"{sex}: P2 occupational composition")
        soc_l1, _, lift_top = p2_soc(ann, coh, types, p, sex, cfg)
        h2(f"{sex}: P3 exposure profile")
        ex = p3_exposure(expo, types, p, cfg, sex)
        h2(f"{sex}: P4 CAMSIS status")
        cam = p4_camsis(p, types, cfg, sex)
        h2(f"{sex}: P5 work breaks")
        gg = p5_gaps(p, cfg, types, coh, base, sex)
        h2(f"{sex}: P7 axis discrimination")
        axis_discrimination(types, coh, expo, p, cfg, sex)

        h2(f"{sex}: labels")
        nm = suggest_names(soc_l1, lift_top, ex, cam, gg, cfg)
        (d / f"type_names_{sex}.json").write_text(json.dumps({
            "sex": sex,
            "named_at": datetime.now().isoformat(timespec="seconds"),
            "basis": "P2 SOC lift + P3 exposure enrichment + P4 CAMSIS + P5 breaks",
            "soc_level": cfg["model"]["soc_level"], "outcome_blind": True,
            "names": nm,
            "note": ("rule-based labels; may be edited after review, but never "
                     "with reference to any downstream result"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        for k, v in nm.items():
            print(f"    type {k}: {v}")
        ok(f"type_names_{sex}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cluster", "profiles", "all"],
                    default="all")
    args = ap.parse_args()
    cfg = load_cfg()
    p = paths(cfg)

    with tee(p, f"04_typology_{args.stage}"):
        h1(f"04: career typology (stage={args.stage})")
        if args.stage in ("cluster", "all"):
            if not require_leiden():
                return 1
            h2("Stage 1: Leiden partition and stability")
            run_cluster(p, cfg)
        if args.stage in ("profiles", "all"):
            h2("Stage 2: profiles and labels")
            run_profiles(p, cfg)
        return 0


if __name__ == "__main__":
    sys.exit(main())
