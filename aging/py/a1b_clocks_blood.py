"""a1b: blood-based clocks (C-L) at baseline (instance 0) -- PhenoAge, KDM, HD and LTL.

All UKB participants with an age at instance 0 are scored (not only the Olink subsample): KDM training,
the HD reference and the age baseline of the gap use the whole sample. Input: e1_blood.csv.gz.

  PhenoAge  Levine et al. 2018: nine markers + age -> Gompertz mortality score -> phenotypic age, with the
            published coefficients (not refitted). UKB units match Levine's except CRP (mg/L -> mg/dL,
            divided by 10, before taking ln). Closed form, algebraically equal to the M form and free of
            overflow as M -> 1:
              PhenoAge = 141.50225 + [ln 0.00553 + ln((e^{120 g} - 1)/g) + xb] / 0.090165,  g = 0.0076927
            Glucose is non-fasting, so fasting time (field 74) is a technical covariate in a4 / a5.
  KDM       Klemera-Doubal method as in BioAge::kdm_calc, fitted by sex in CN participants:
            each marker x_j = q_j + k_j * CA with residual SD s_j and r_j^2;
              BAe   = sum((x - q) k / s^2) / sum((k / s)^2)
              rchar = sum(|k / s| r) / sum(|k / s|)
              s2_BA = var(BAe - CA) - (1 - rchar^2) / rchar^2 * (CAmax - CAmin)^2 / (12 m)
              KDM   = [sum((x - q) k / s^2) + CA / s2_BA] / [sum((k / s)^2) + 1 / s2_BA]
            If s2_BA <= 0, BAe is used.
  HD        homeostatic dysregulation (Cohen et al. 2013; BioAge::hd_calc): markers standardised to a
            reference of same-sex CN participants younger than hd_ref_age; ln Mahalanobis distance under the
            reference covariance. Complete cases.
  LTL       pred = -ltl_z (field 22192), so that higher = shorter telomeres = older.

Gap: per clock and sex, residual of pred on a linear regression on age in the whole sample (gapy), then
z-scored within sex (gapz > 0 = older). gapy units: PhenoAge / KDM years, HD ln distance, LTL SD; clocks are
compared on gapz only. KDM / HD inputs are log-transformed (log_markers) and clipped to +/- winsor_sd SD
within sex; PhenoAge uses raw values.

Outputs (derived/): clock_CL.parquet       eid, sex, age, cn, pred__* / gapy__* / gapz__*
                    qb_i0all_{sex}.parquet lifetime intensities at instance 0 (cohort with a C-L clock)
Tables (tables/):   a1b_CL_perf.csv        r(pred, age) overall and by sex, MAE (year-scaled clocks), n
                    a1b_CL_kdm_params.csv  by sex x marker: q, k, s, r^2, rchar, s2_BA, training n
                    a1b_CL_hd_ref.csv      by sex x marker: reference mean / SD, reference n
                    a1b_CL_markers.csv     per marker: unit, n, P1 / median / P99 (unit check)

  python aging/py/a1b_clocks_blood.py
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from a0_assemble import lifetime_intensity
from common_aging import clock_file, fail, h1, h2, load_measure, ok, read_extract, setup, tee, warn

SEXES = ["Female", "Male"]
# Levine et al. 2018 (Aging 10:573), Table 1; units in the module docstring
PHENO_COEF = {"albumin": -0.0336, "creatinine": 0.0095, "glucose": 0.1953, "ln_crp_mgdl": 0.0954,
              "lymph_pct": -0.0120, "mcv": 0.0268, "rdw": 0.3306, "alp": 0.00188, "wbc": 0.0554}
PHENO_AGE_COEF, PHENO_INTERCEPT, PHENO_GAMMA = 0.0804, -19.9067, 0.0076927
CRP_FLOOR_MGL = 0.01          # avoids ln(0); below the UKB detection limit
UNITS = {"wbc": "10^9/L", "mcv": "fL", "rdw": "%", "lymph_pct": "%", "albumin": "g/L", "alp": "U/L",
         "urea": "mmol/L", "cholesterol": "mmol/L", "creatinine": "umol/L", "crp": "mg/L",
         "glucose": "mmol/L", "hba1c": "mmol/mol", "sbp": "mmHg", "fev1": "L", "fasting_time": "h",
         "ltl_z": "z(log T/S)"}


# ── Markers ────────────────────────────────────────────────────────
def _cols(raw, name):
    """{name}_i0 or, for multi-array fields, {name}_i0_a{k}."""
    return [c for c in raw.columns if re.fullmatch(rf"{re.escape(name)}_i0(_a\d+)?", c)]


def build_markers(raw: pd.DataFrame) -> pd.DataFrame:
    """e1_blood -> one row per participant with short-name columns. SBP = mean of the two automated
    readings (manual if missing); FEV1 = field 20150 (maximum over blows of field 3063 if missing)."""
    num = raw.apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(index=raw.index)
    for k in UNITS:
        if k in ("sbp", "fev1"):
            continue
        c = _cols(num, k)
        out[k] = num[c].mean(axis=1) if c else np.nan
        if not c:
            warn(f"e1_blood has no {k}_i0; column left missing")
    auto, man = _cols(num, "sbp_auto"), _cols(num, "sbp_manual")
    out["sbp"] = (num[auto].mean(axis=1) if auto else np.nan)
    if man:
        out["sbp"] = out["sbp"].fillna(num[man].mean(axis=1))
    best, raw3063 = _cols(num, "fev1_best"), _cols(num, "fev1")
    out["fev1"] = num[best].mean(axis=1) if best else np.nan
    if raw3063:
        out["fev1"] = out["fev1"].fillna(num[raw3063].max(axis=1))
    return out


def marker_table(M: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in M.columns:
        v = M[k].dropna()
        rows.append({"marker": k, "unit": UNITS.get(k, ""), "n": len(v),
                     "p01": v.quantile(.01) if len(v) else np.nan, "median": v.median() if len(v) else np.nan,
                     "p99": v.quantile(.99) if len(v) else np.nan})
    return pd.DataFrame(rows)


def prep_inputs(M, markers, sex, cfg):
    """KDM / HD inputs: ln (log_markers), then clipped to mean +/- winsor_sd SD within sex."""
    bc = cfg["blood"]
    X = M[markers].astype("float64").copy()
    for k in markers:
        if k in bc["log_markers"]:
            X[k] = np.log(X[k].clip(lower=CRP_FLOOR_MGL if k == "crp" else 1e-6))
    w = bc["winsor_sd"]
    if w:
        for s in SEXES:
            m = sex == s
            mu, sd = X.loc[m].mean(), X.loc[m].std()
            X.loc[m] = X.loc[m].clip(lower=mu - w * sd, upper=mu + w * sd, axis=1)
    return X


# ── PhenoAge ───────────────────────────────────────────────────────
def phenoage(M: pd.DataFrame, age: np.ndarray) -> np.ndarray:
    crp = np.maximum(M["crp"].to_numpy("float64"), CRP_FLOOR_MGL)
    xb = PHENO_INTERCEPT + PHENO_AGE_COEF * age + PHENO_COEF["ln_crp_mgdl"] * np.log(crp / 10.0)
    for k, b in PHENO_COEF.items():
        if k != "ln_crp_mgdl":
            xb = xb + b * M[k].to_numpy("float64")
    g = PHENO_GAMMA
    return 141.50225 + (np.log(0.00553) + np.log((np.exp(120 * g) - 1) / g) + xb) / 0.090165


# ── KDM ────────────────────────────────────────────────────────────
def kdm_fit(X: np.ndarray, age: np.ndarray) -> dict:
    """Per-marker regression x = q + k * CA in the training set (complete cases), then rchar and s2_BA."""
    m = X.shape[1]
    q, k, s, r2 = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)
    for j in range(m):
        kk, qq = np.polyfit(age, X[:, j], 1)
        res = X[:, j] - (qq + kk * age)
        q[j], k[j] = qq, kk
        s[j] = np.sqrt(res @ res / (len(age) - 2))                 # = summary(lm)$sigma
        r2[j] = np.corrcoef(age, X[:, j])[0, 1] ** 2
    ks = np.abs(k / s)
    rchar = float((ks * np.sqrt(r2)).sum() / ks.sum())
    bae = ((X - q) * k / s ** 2).sum(1) / ((k / s) ** 2).sum()
    s_r = (1 - rchar ** 2) / rchar ** 2 * (age.max() - age.min()) ** 2 / (12 * m)
    s2 = float(np.var(bae - age, ddof=1) - s_r)
    return {"q": q, "k": k, "s": s, "r2": r2, "rchar": rchar, "s2": s2, "n": len(age)}


def kdm_apply(X: np.ndarray, age: np.ndarray, fit: dict) -> np.ndarray:
    """Missing markers are left out of both sums (only relevant when max_missing > 0)."""
    q, k, s = fit["q"], fit["k"], fit["s"]
    av = np.isfinite(X)
    n1 = np.where(av, (np.nan_to_num(X) - q) * k / s ** 2, 0).sum(1)
    n2 = np.where(av, (k / s) ** 2, 0).sum(1)
    if fit["s2"] > 0:
        return (n1 + age / fit["s2"]) / (n2 + 1 / fit["s2"])
    return n1 / n2


# ── HD ─────────────────────────────────────────────────────────────
def hd_fit(R: np.ndarray) -> dict:
    mu, sd = R.mean(0), R.std(0, ddof=1)
    Z = (R - mu) / sd
    C = np.cov(Z, rowvar=False)
    return {"mu": mu, "sd": sd, "center": Z.mean(0), "icov": np.linalg.pinv(C), "n": len(R)}


def hd_apply(X: np.ndarray, fit: dict) -> np.ndarray:
    D = (X - fit["mu"]) / fit["sd"] - fit["center"]
    d2 = np.einsum("ij,jk,ik->i", D, fit["icov"], D)
    return np.log(np.sqrt(np.maximum(d2, 1e-12)))


# ── Gap ────────────────────────────────────────────────────────────
def gap_by_sex(pred, age, sex):
    """Within sex: residual of pred on a linear fit on age (gapy), then z-scored (gapz)."""
    gy, gz = np.full(len(pred), np.nan), np.full(len(pred), np.nan)
    for s in SEXES:
        m = (sex == s) & np.isfinite(pred) & np.isfinite(age)
        if m.sum() < 30:
            continue
        b = np.polyfit(age[m], pred[m], 1)
        gy[m] = pred[m] - np.polyval(b, age[m])
        gz[m] = (gy[m] - gy[m].mean()) / gy[m].std()
    return gy, gz


def perf_row(ck, pred, age, sex, unit, extreme_z, gz, **extra):
    ok_ = np.isfinite(pred) & np.isfinite(age)
    row = {"clock": ck, "n": int(ok_.sum()), "units": unit,
           "r": float(np.corrcoef(pred[ok_], age[ok_])[0, 1]) if ok_.sum() > 2 else np.nan,
           "mae": float(np.mean(np.abs(pred[ok_] - age[ok_]))) if unit == "years" and ok_.any() else np.nan,
           "mean_pred_minus_age": float(np.mean(pred[ok_] - age[ok_])) if unit == "years" and ok_.any() else np.nan,
           "extreme_frac": float(np.mean(gz[np.isfinite(gz)] > extreme_z)) if np.isfinite(gz).any() else np.nan}
    for s in SEXES:
        m = ok_ & (sex == s)
        row[f"n_{s}"] = int(m.sum())
        row[f"r_{s}"] = float(np.corrcoef(pred[m], age[m])[0, 1]) if m.sum() > 2 else np.nan
    return {**row, **extra}


# ── main ───────────────────────────────────────────────────────────
def main():
    args, cfg, up, p = setup(__doc__)
    with tee(p, "a1b_clocks_blood"):
        h1("a1b blood-based clocks: PhenoAge + KDM + HD + LTL")
        bc = cfg["blood"]
        clocks = list(bc["clocks"])
        mea = load_measure(p).set_index("eid")
        mea = mea[mea["age_i0"].notna() & mea["sex"].isin(SEXES)]
        raw = read_extract(p, "blood").set_index("eid")
        raw = raw[raw.index.isin(mea.index)]
        if raw.index.duplicated().any():
            warn(f"e1_blood has {int(raw.index.duplicated().sum())} duplicated eids; first row kept")
            raw = raw[~raw.index.duplicated()]
        raw = raw.sort_index()
        print(f"    e1_blood with an age at i0: {len(raw):,} participants ({raw.shape[1]} columns)")
        M = build_markers(raw)
        mt = marker_table(M)
        mt.to_csv(p["tables"] / "a1b_CL_markers.csv", index=False)
        print(mt.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        m = mea.loc[M.index]
        age = m["age_i0"].to_numpy("float64")
        sex = m["sex"].to_numpy()
        cn = (m["cn_i0"] == True).to_numpy()                                      # noqa: E712
        preds, perf_extra, kdm_rows, hd_rows = {}, {}, [], []

        if "PhenoAge" in clocks:
            h2("PhenoAge (Levine 2018, published formula)")
            pm = bc["pheno_markers"]
            comp = M[pm].notna().all(1).to_numpy()
            pa = np.where(comp, phenoage(M[pm].fillna(1.0), age), np.nan)  # fillna avoids log warnings; masked
            preds["PhenoAge"] = pa
            perf_extra["PhenoAge"] = {"n_train": 0, "train_note": "published formula"}
            print(f"    complete cases {int(comp.sum()):,}; mean PhenoAge - age {np.nanmean(pa - age):+.2f} years")

        if "KDM" in clocks:
            h2("KDM (Klemera-Doubal, by sex, trained in CN)")
            km = bc["kdm_markers"]
            X = prep_inputs(M, km, sex, cfg)
            nmiss = X.isna().sum(1).to_numpy()
            usable = nmiss <= bc["max_missing"]
            kd = np.full(len(X), np.nan)
            ntr = {}
            for s in SEXES:
                ms = sex == s
                tr = ms & (nmiss == 0) & (cn if bc["kdm_train"] == "cn" else True)
                if tr.sum() < bc["min_train_n"]:
                    warn(f"KDM {s}: {int(tr.sum())} CN complete cases < {bc['min_train_n']}; "
                         "training on all complete cases of this sex")
                    tr = ms & (nmiss == 0)
                if tr.sum() < bc["min_train_n"]:
                    warn(f"KDM {s}: only {int(tr.sum())} complete cases; skipped")
                    continue
                fit = kdm_fit(X.to_numpy("float64")[tr], age[tr])
                if fit["s2"] <= 0:
                    warn(f"KDM {s}: s2_BA = {fit['s2']:.3f} <= 0; using BAe (no CA term)")
                ap = ms & usable
                kd[ap] = kdm_apply(X.to_numpy("float64")[ap], age[ap], fit)
                ntr[s] = fit["n"]
                for j, mk in enumerate(km):
                    kdm_rows.append({"sex": s, "marker": mk, "q": fit["q"][j], "k": fit["k"][j], "s": fit["s"][j],
                                     "r2": fit["r2"][j], "rchar": fit["rchar"], "s2_ba": fit["s2"],
                                     "n_train": fit["n"]})
                print(f"    {s}: trained on {fit['n']:,}, rchar = {fit['rchar']:.3f}, s2_BA = {fit['s2']:.2f}, "
                      f"scored {int(ap.sum()):,}")
            preds["KDM"] = kd
            perf_extra["KDM"] = {"n_train": sum(ntr.values()),
                                 "train_note": f"{bc['kdm_train']} by sex; complete cases; CA term included"}

        if "HD" in clocks:
            h2(f"HD (ln Mahalanobis; reference = CN and age < {bc['hd_ref_age']}, by sex)")
            hm = bc["hd_markers"]
            X = prep_inputs(M, hm, sex, cfg)
            comp = X.notna().all(1).to_numpy()
            hd = np.full(len(X), np.nan)
            nref = {}
            for s in SEXES:
                ms = sex == s
                ref = ms & comp & cn & (age < bc["hd_ref_age"])
                if ref.sum() < bc["min_train_n"]:
                    warn(f"HD {s}: reference group {int(ref.sum())} < {bc['min_train_n']}; skipped")
                    continue
                fit = hd_fit(X.to_numpy("float64")[ref])
                ap = ms & comp
                hd[ap] = hd_apply(X.to_numpy("float64")[ap], fit)
                nref[s] = fit["n"]
                for j, mk in enumerate(hm):
                    hd_rows.append({"sex": s, "marker": mk, "ref_mean": fit["mu"][j], "ref_sd": fit["sd"][j],
                                    "n_ref": fit["n"]})
                print(f"    {s}: reference {fit['n']:,}, scored {int(ap.sum()):,}")
            preds["HD"] = hd
            perf_extra["HD"] = {"n_train": sum(nref.values()),
                                "train_note": f"reference = CN & age < {bc['hd_ref_age']} by sex"}

        if "LTL" in clocks:
            h2(f"LTL (pred = -{bc['ltl_field']})")
            v = M[bc["ltl_field"]].to_numpy("float64")
            preds["LTL"] = -v
            perf_extra["LTL"] = {"n_train": 0, "train_note": f"direct measurement, sign flipped ({bc['ltl_field']})"}
            print(f"    non-missing {int(np.isfinite(v).sum()):,}")

        h2("Gap (within sex, linear residual on age), r and MAE")
        base = pd.DataFrame({"eid": M.index.to_numpy(), "sex": sex, "age": age, "cn": m["cn_i0"].to_numpy()})
        cols, perf_rows = {}, []
        for ck in clocks:
            if ck not in preds:
                continue
            pr = preds[ck]
            gy, gz = gap_by_sex(pr, age, sex)
            cols[f"pred__{ck}"], cols[f"gapy__{ck}"], cols[f"gapz__{ck}"] = pr, gy, gz
            perf_rows.append(perf_row(ck, pr, age, sex, bc["units"].get(ck, ""), bc["extreme_z"], gz,
                                      **perf_extra.get(ck, {})))
        if not cols:
            fail("no C-L clock could be computed; check the warnings above (missing e1_blood columns?)")
            raise SystemExit(1)
        cl = pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1)
        cl = cl[cl[[c for c in cl.columns if c.startswith("gapz__")]].notna().any(axis=1)].reset_index(drop=True)
        cl.to_parquet(clock_file(p, "C-L"), index=False)
        pf = pd.DataFrame(perf_rows)
        pf.to_csv(p["tables"] / "a1b_CL_perf.csv", index=False)
        pd.DataFrame(kdm_rows).to_csv(p["tables"] / "a1b_CL_kdm_params.csv", index=False)
        pd.DataFrame(hd_rows).to_csv(p["tables"] / "a1b_CL_hd_ref.csv", index=False)
        print(pf[["clock", "units", "n", "r", "r_Female", "r_Male", "mae", "mean_pred_minus_age", "extreme_frac"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        ok(f"clock_CL.parquet: {len(cl):,} participants x {len(cols) // 3} clocks")

        h2("Lifetime intensities qb_i0all (cohort with a C-L clock, age at i0)")
        have = set(cl["eid"])
        for s in SEXES:
            fp, fq = p["derived"] / f"person_{s}.parquet", p["derived"] / f"qb_annual_{s}.parquet"
            if not (fp.exists() and fq.exists()):
                warn(f"missing {fp.name} / {fq.name} -- run a0 first; qb_i0all for {s} not built")
                continue
            d = pd.read_parquet(fp, columns=["eid"])
            am = mea["age_i0"].reindex(d["eid"][d["eid"].isin(have)].to_numpy()).dropna()
            q = lifetime_intensity(cfg, pd.read_parquet(fq), am)
            q.to_parquet(p["derived"] / f"qb_i0all_{s}.parquet", index=False)
            print(f"    {s}: {len(q):,} participants")
        ok("a1b done")


if __name__ == "__main__":
    main()
