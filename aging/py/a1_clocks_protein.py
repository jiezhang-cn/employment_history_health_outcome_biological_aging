"""a1: proteomic clocks at baseline (instance 0) -- C-P organ clocks (Oh et al. 2025, published
coefficients) and C-C cell-type and lineage clocks (Ding et al. 2026, published coefficients).

All Olink participants are scored (not only the work-history cohort): the age baseline of the gap and the
mortality validation in a3 use the whole sample.

The two families define the age gap differently:
  C-P  z = (NPX - training mean) / training SD from Oh's supplementary table ST3 (the published scaler,
       not sample z-scores); pred = intercept + sum(w * z); gap = residual of pred on a linear regression
       on age, then z-scored per clock; sexes pooled.
  C-C  z = sample z-scores (Ding did not publish a scaler); pred = intercept + sum(beta * z)
       + beta_sex * female (sex is a model feature, female = 1); gap = pred - LOWESS(age) fitted in CN
       participants only (span 2/3) and applied to everyone; then z-scored per clock.

Input: the UKB-PPP olink_data long table (NPX); protein_id is mapped to gene symbols with UKB Data-Coding
143. Participants missing >= max_person_missing_frac of the model proteins (union of both families) are
excluded; remaining gaps are filled once by KNN imputation (k = 152) in the ST3-standardised space, as in
Oh et al., and shared by both families.

Reference files (references/, see README): Oh ST4 weights (oh2025_weights_long.csv), Oh ST3 scaler
(oh2025_training_scaler_ST3.csv), Ding coefficients (ding_cellage_coef_long.csv, ding_cellage_models.csv),
Data-Coding 143 (uniprot_meaning).

Outputs (derived/): clock_CP.parquet / clock_CC.parquet  eid, sex, age, cn, pred__*, gapy__* (years), gapz__*
                    olink_model_proteins_long_i0.parquet  cache of the model proteins
Tables (tables/):   a1_CP_coverage.csv  model-protein coverage and missing-weight share per organ (V7)
                    a1_CP_perf.csv / a1_CC_perf.csv  r(pred, age), MAE, published benchmarks (V1)
                    a1_CC_branch.csv  C-C accuracy check: use_weights / retrain / fix_mapping

  python aging/py/a1_clocks_protein.py [--olink-long PATH]
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common_aging import (clock_file, fail, h1, h2, load_measure, ok, olink_long_file, olink_long_sep,
                          ref_file, setup, tee, warn)


def read_ref(cfg, name):
    return pd.read_csv(ref_file(cfg, name, required=True), encoding="utf-8-sig")


def _key(s) -> str:
    """Name key: drop - / _ / spaces and upper-case (aligns ST3, ST4, Ding and Data-Coding 143 names)."""
    return re.sub(r"[-_ ]", "", str(s)).upper()


def _cache_ok(f_cache: Path, f_src: Path, cols) -> pd.DataFrame | None:
    if f_cache.exists() and f_cache.stat().st_mtime > f_src.stat().st_mtime:
        X = pd.read_parquet(f_cache)
        if set(cols) <= set(X.columns):
            return X[list(cols)]
    return None


def _dedup(X: pd.DataFrame, what: str) -> pd.DataFrame:
    X = X[X.index > 0]                                     # negative eid = withdrawn
    dup = X.index.duplicated().sum()
    if dup:
        warn(f"{what}: {dup} duplicated eids; first row kept")
        X = X[~X.index.duplicated()]
    return X


# ── Olink long table ───────────────────────────────────────────────
def coding143(cfg, prots):
    """Data-Coding 143: coding -> 'GENE;protein name'. Returns Series coding -> model protein column."""
    c = pd.read_csv(ref_file(cfg, cfg["protein"]["coding143"], required=True), sep="\t")
    c["gene"] = c["meaning"].astype(str).str.split(";").str[0]
    kmap = {_key(x): x for x in prots}
    c["local_col"] = c["gene"].map(_key).map(kmap)
    m = c.dropna(subset=["local_col"]).set_index("coding")["local_col"]
    absent = sorted(set(prots) - set(m))
    (warn if absent else ok)(f"Data-Coding 143: {len(c):,} assays, {m.nunique()}/{len(prots)} model proteins matched"
                             + (f"; unmatched {absent[:10]}" if absent else ""))
    return m


def olink_long(p, cfg, f, prots, instances):
    """Long table -> {instance: eid x model-protein NPX table}. Read in chunks, keeping model proteins only."""
    pc = cfg["protein"]
    code2col = coding143(cfg, prots)
    stem = pc["cache_stem"]
    caches = {k: p["derived"] / f"{stem}_long_i{k}.parquet" for k in instances}
    have = sorted(set(code2col))
    got = {k: _cache_ok(caches[k], f, have) for k in instances}
    if all(v is not None for v in got.values()):
        for k, X in got.items():
            ok(f"read cache {caches[k].name}: {len(X):,} x {X.shape[1]} proteins")
        return got
    sep, hdr = olink_long_sep(f)
    need_cols = ["eid", "ins_index", "protein_id", "result"]
    if not set(need_cols) <= set(hdr):
        fail(f"{f.name} is not an olink_data long table (header {hdr[:8]}); needs {need_cols}")
        raise SystemExit(1)
    parts = {k: [] for k in instances}
    n_read = 0
    reader = pd.read_csv(f, sep=sep, usecols=need_cols, chunksize=pc["long_chunk_rows"],
                         dtype={"eid": "int64", "ins_index": "int16", "protein_id": "int32"})
    for ch in reader:
        n_read += len(ch)
        ch = ch[ch["protein_id"].isin(code2col.index) & ch["ins_index"].isin(instances)]
        ch["result"] = pd.to_numeric(ch["result"], errors="coerce").astype("float32")
        for k, g in ch.groupby("ins_index"):
            parts[int(k)].append(g[["eid", "protein_id", "result"]])
    print(f"    long table {f.name}: {n_read:,} rows read")
    out = {}
    for k in instances:
        if not parts[k]:
            out[k] = pd.DataFrame(columns=have, dtype="float32")
            print(f"    instance {k}: no records")
            continue
        d = pd.concat(parts[k], ignore_index=True)
        dup = d.duplicated(["eid", "protein_id"]).sum()
        if dup:
            warn(f"instance {k}: {dup:,} duplicated (eid, protein) records; first kept")
            d = d.drop_duplicates(["eid", "protein_id"])
        X = d.pivot(index="eid", columns="protein_id", values="result")
        X.columns = [code2col[c] for c in X.columns]
        X = _dedup(X.reindex(columns=have).astype("float32"), f"long table instance {k}")
        X.to_parquet(caches[k])
        out[k] = X
        print(f"    instance {k}: {len(X):,} participants x {int(X.notna().any().sum())} model proteins")
    return out


def prezscored(X, cfg):
    """True if the input looks column-wise z-scored (median |mean| < mean_tol and median |SD - 1| < sd_tol);
    the ST3 scaler would then be applied to the wrong scale. Medians, not per-column hit rates, are used
    because sampling error pushes individual columns past the tolerance."""
    c = cfg["protein"]["prezscored_check"]
    mu, sd = X.mean(), X.std()
    med_m, med_s = float(mu.abs().median()), float((sd - 1).abs().median())
    return (med_m < c["mean_tol"] and med_s < c["sd_tol"]), med_m, med_s


def fit_imputer(Z: pd.DataFrame, cfg):
    """KNN imputer (Oh et al.: KNNImputer, k ~ sqrt(training n) = 152). The neighbour reference set is
    ref_n participants drawn at random from the 2 * ref_n with the fewest missing values."""
    from sklearn.impute import KNNImputer
    ic = cfg["protein"]["impute"]
    miss = Z.isna().sum(axis=1)
    rng = np.random.default_rng(ic["seed"])
    ref_n = min(ic["ref_n"], len(Z))
    pool = miss.sort_values(kind="mergesort").index[: ref_n * 2]
    ref = rng.choice(pool.to_numpy(), size=min(ref_n, len(pool)), replace=False)
    k = min(ic["k"], len(ref) - 1)
    imp = KNNImputer(n_neighbors=k, keep_empty_features=True).fit(Z.loc[ref].to_numpy("float64"))
    imp.n_ref_, imp.k_ = len(ref), k
    return imp


def knn_impute(Z: pd.DataFrame, imp, cfg) -> pd.DataFrame:
    """Impute only rows with missing values, in chunks."""
    miss = Z.isna().sum(axis=1)
    need = miss[miss > 0].index
    if not len(need):
        return Z
    out = Z.copy()
    ch = cfg["protein"]["impute"]["chunk"]
    t0 = time.time()
    print(f"    KNN imputation: {len(need):,} rows, chunks of {ch:,}", flush=True)
    for i in range(0, len(need), ch):
        idx = need[i:i + ch]
        out.loc[idx] = imp.transform(Z.loc[idx].to_numpy("float64")).astype("float32")
        done = min(i + ch, len(need))
        el = time.time() - t0
        print(f"      {done:,}/{len(need):,}  {el / 60:.1f} min elapsed, "
              f"~{el / done * (len(need) - done) / 60:.1f} min left", flush=True)
    print(f"    KNN imputation: k = {imp.k_}, reference set {imp.n_ref_:,}, {len(need):,} rows imputed "
          f"({float(Z.isna().to_numpy().mean()):.2%} of cells missing)")
    return out


# ── Age baselines and accuracy ─────────────────────────────────────
def base_linear(pred, age):
    """C-P age baseline: linear regression of pred on age."""
    ok_ = np.isfinite(pred) & np.isfinite(age)
    b = np.polyfit(age[ok_], pred[ok_], 1)
    return lambda a: np.polyval(b, a)


def base_lowess(pred, age, cn, frac):
    """C-C age baseline: LOWESS fitted in CN participants only; constant beyond the fitted range."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    ok_ = np.isfinite(pred) & np.isfinite(age) & cn
    if ok_.sum() < 50:
        warn(f"only {int(ok_.sum())} CN participants; LOWESS fitted in the whole sample")
        ok_ = np.isfinite(pred) & np.isfinite(age)
    a, y = age[ok_], pred[ok_]
    fit = lowess(y, a, frac=frac, delta=0.01 * (a.max() - a.min()), return_sorted=True)
    xs, idx = np.unique(fit[:, 0], return_index=True)
    ys = fit[idx, 1]
    return lambda v: np.interp(v, xs, ys)


def perf(pred, age):
    ok_ = np.isfinite(pred) & np.isfinite(age)
    return {"n": int(ok_.sum()), "r": float(np.corrcoef(pred[ok_], age[ok_])[0, 1]),
            "mae": float(np.mean(np.abs(pred[ok_] - age[ok_])))}


# ── Predictions ────────────────────────────────────────────────────
def predict_cp(W, Z):
    out = {}
    for organ, g in W.groupby("organ", sort=False):
        gg = g[g["local_col"].isin(Z.columns)]
        out[organ] = float(g["intercept"].iloc[0]) + Z[gg["local_col"]].to_numpy("float64") @ gg["weight"].to_numpy()
    return out


def predict_cc(DC, DM, Zs, female):
    out = {}
    for model, g in DC.groupby("model", sort=False):
        gg = g[g["local_col"].isin(Zs.columns)]
        mi = DM.loc[model]
        out[model] = (float(mi["intercept"]) + Zs[gg["local_col"]].to_numpy("float64") @ gg["coef"].to_numpy()
                      + float(mi["coef_sex_female1"]) * female)
    return out


def gap_cols(preds, age, bases, scale):
    """pred -> pred__ / gapy__ / gapz__; bases[clock] = age baseline, scale[clock] = (mean, SD) of the gap."""
    cols = {}
    for ck, pr in preds.items():
        gy = pr - bases[ck](age)
        mu, sd = scale[ck]
        cols[f"pred__{ck}"], cols[f"gapy__{ck}"], cols[f"gapz__{ck}"] = pr, gy, (gy - mu) / sd
    return cols


def frame_base(m, eids, inst):
    return pd.DataFrame({"eid": np.asarray(eids), "sex": m["sex"].to_numpy(),
                         "age": m[f"age_i{inst}"].to_numpy("float64"), "cn": m[f"cn_i{inst}"].to_numpy()})


# ── main ───────────────────────────────────────────────────────────
def main():
    args, cfg, up, p = setup(__doc__, [(("--olink-long",), {"default": None,
                                        "help": "path of the olink_data long table (overrides files.olink_long)"})])
    with tee(p, "a1_clocks_protein"):
        h1("a1 proteomic clocks: C-P (Oh 2025) + C-C (Ding 2026)")
        pc, vc_ = cfg["protein"], cfg["validation"]
        W = read_ref(cfg, pc["oh"]["weights"])
        S = read_ref(cfg, pc["oh"]["scaler"]).drop_duplicates("local_col").set_index("local_col")
        DC = read_ref(cfg, pc["ding"]["coef"])
        DM = read_ref(cfg, pc["ding"]["models"]).set_index("model")
        if pc["ding"]["retained_only"]:
            DM = DM[DM["retained_48"].astype(bool)]
        DC = DC[DC["model"].isin(DM.index)]
        print(f"    Oh: {W['organ'].nunique()} clocks, {W['local_col'].nunique()} proteins; "
              f"Ding: {len(DM)} models, {DC['local_col'].nunique()} proteins")
        prots = sorted(set(W["local_col"]) | set(DC["local_col"]))
        miss_sc = sorted(set(prots) - set(S.index))
        if miss_sc:
            fail(f"ST3 has no scaling parameters for these model proteins: {miss_sc[:10]}")
            raise SystemExit(1)

        h2("Olink model proteins")
        f_long = olink_long_file(p, cfg, args.olink_long)
        if f_long is None:
            fail(f"Olink long table not found ({args.olink_long or cfg['files']['olink_long']} in {p['data']})")
            raise SystemExit(1)
        k0 = cfg["measure"]["protein_instance"]
        X = olink_long(p, cfg, f_long, prots, [k0])[k0]
        mea = load_measure(p).set_index("eid")
        X = X[X.index.isin(mea.index[mea["age_i0"].notna()])]
        X = X.drop(columns=[c for c in X.columns if X[c].isna().all()])
        isz, med_m, med_s = prezscored(X, cfg)
        print(f"    pre-standardisation check: median |mean| = {med_m:.4f}, median |SD - 1| = {med_s:.4f}; "
              f"missing cells {float(X.isna().to_numpy().mean()):.2%}")
        if isz:
            fail("Olink input appears column-wise z-scored; the ST3 training scaler requires raw NPX values")
            raise SystemExit(1)
        pm = X.isna().mean(axis=1)
        keep = pm < pc["max_person_missing_frac"]
        print(f"    excluded for missing >= {pc['max_person_missing_frac']:.3f} of model proteins: "
              f"{int((~keep).sum()):,}; remaining {int(keep.sum()):,}")
        X = X[keep]
        mu3, sd3 = S.loc[X.columns, "mean"].to_numpy(), S.loc[X.columns, "stdev"].to_numpy()
        Z = ((X - mu3) / sd3).astype("float32")
        imp = fit_imputer(Z, cfg)
        Z = knn_impute(Z, imp, cfg)
        Xn = Z * sd3 + mu3                                              # imputed NPX

        m = mea.loc[Z.index]
        age = m["age_i0"].to_numpy("float64")
        cn = (m["cn_i0"] == True).to_numpy()                                     # noqa: E712
        female = (m["sex"] == "Female").to_numpy("float64")
        base = frame_base(m, Z.index, 0)

        # ── C-P ──
        h2("C-P: Oh 2025 (ST3 scaling + ST4 weights)")
        cov_rows, perf_rows = [], []
        for organ, g in W.groupby("organ", sort=False):
            pres = g["local_col"].isin(Z.columns)
            wabs = g["weight"].abs()
            cov_rows.append({"clock": organ, "n_prot": len(g), "n_present": int(pres.sum()),
                             "coverage": float(pres.mean()),
                             "missing_weight_frac": float(wabs[~pres].sum() / wabs.sum()),
                             "missing_prots": ";".join(g.loc[~pres, "local_col"]),
                             "person_missing_mean": float(X[g.loc[pres, "local_col"]].isna().mean(axis=1).mean())})
        pr_cp = predict_cp(W, Z)
        bases_cp = {ck: base_linear(v, age) for ck, v in pr_cp.items()}
        scale_cp = {}
        for ck, v in pr_cp.items():
            gy = v - bases_cp[ck](age)
            scale_cp[ck] = (float(np.nanmean(gy)), float(np.nanstd(gy)))
        cols = gap_cols(pr_cp, age, bases_cp, scale_cp)
        for ck, v in pr_cp.items():
            r = perf(v, age)
            bm = pc["oh"]["benchmark_r"].get(ck, np.nan)
            perf_rows.append({"clock": ck, **r, "benchmark_r": bm, "r_minus_benchmark": r["r"] - bm,
                              "extreme_frac": float(np.mean(np.abs(cols[f"gapz__{ck}"]) > pc["oh"]["extreme_z"]))})
        cp = pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1)
        cp.to_parquet(clock_file(p, "C-P"), index=False)
        cv = pd.DataFrame(cov_rows)
        cv.to_csv(p["tables"] / "a1_CP_coverage.csv", index=False)
        pf = pd.DataFrame(perf_rows)
        pf.to_csv(p["tables"] / "a1_CP_perf.csv", index=False)
        print(pf.merge(cv[["clock", "coverage", "missing_weight_frac"]], on="clock")
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

        # ── C-C ──
        h2("C-C: Ding 2026 (sample z-scores + sex feature + LOWESS baseline in CN)")
        mu_s, sd_s = Xn.mean(), Xn.std()               # no published scaler: sample z-scores
        Zs = (Xn - mu_s) / sd_s
        pr_cc = predict_cc(DC, DM, Zs, female)
        bases_cc = {ck: base_lowess(v, age, cn, pc["ding"]["lowess_frac"]) for ck, v in pr_cc.items()}
        scale_cc = {}
        for ck, v in pr_cc.items():
            gy = v - bases_cc[ck](age)
            scale_cc[ck] = (float(np.nanmean(gy)), float(np.nanstd(gy)))
        cols = gap_cols(pr_cc, age, bases_cc, scale_cc)
        perf_rows = []
        for model, g in DC.groupby("model", sort=False):
            pres = g["local_col"].isin(Zs.columns)
            mi = DM.loc[model]
            r = perf(pr_cc[model], age)
            perf_rows.append({"clock": model, "source": mi["Source"], **r,
                              "paper_test_r": float(mi["Test_Correlation"]),
                              "paper_train_r": float(mi["Train_Correlation"]),
                              "r_minus_paper": r["r"] - float(mi["Test_Correlation"]),
                              "n_prot": len(g), "coverage": float(pres.mean()),
                              "missing_weight_frac": float(g.loc[~pres, "coef"].abs().sum() / g["coef"].abs().sum()),
                              "extreme_frac": float(np.mean(cols[f"gapz__{model}"] > pc["ding"]["extreme_z"]))})
        cc = pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1)
        cc.to_parquet(clock_file(p, "C-C"), index=False)
        pf = pd.DataFrame(perf_rows)
        pf.to_csv(p["tables"] / "a1_CC_perf.csv", index=False)
        print(pf[["clock", "source", "n", "r", "paper_test_r", "r_minus_paper", "coverage"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

        # Accuracy check: r within v1_r_tol of the published test r -> use the weights; lower but with the
        # family's ranking preserved -> would need retraining; r near 0 -> protein mapping is wrong.
        from scipy.stats import spearmanr
        rho = float(spearmanr(pf["r"], pf["paper_test_r"]).statistic) if len(pf) > 2 else np.nan
        tol = vc_["v1_r_tol"]
        br = []
        for r in pf.itertuples():
            if r.r <= 0.05:
                b = "fix_mapping"
            elif r.r >= r.paper_test_r - tol:
                b = "use_weights"
            else:
                b = "retrain" if rho > 0.5 else "fix_mapping"
            br.append({"clock": r.clock, "r": r.r, "paper_test_r": r.paper_test_r, "branch": b})
        br = pd.DataFrame(br)
        br["rank_rho_family"] = rho
        br.to_csv(p["tables"] / "a1_CC_branch.csv", index=False)
        print(f"    accuracy check: {br['branch'].value_counts().to_dict()}; "
              f"Spearman rho of r with published r = {rho:.2f}")
        ok("a1 done")


if __name__ == "__main__":
    main()
