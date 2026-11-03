"""m0: analysis frames for the blood arm (baseline, instance 0) and the imaging arm (instance 2).

Builds, per sex and arm, one row per work-history participant with a measurement visit: career type,
career axes, exposure-years before the measurement, pre-measurement mean CAMSIS, age / date / centre at
measurement, disease-free flag (cn_meas), left-work flag (left_before) and ethnic group. No feature values
are read.

Inputs
  aging module (results/aging/derived): person_{sex}.parquet, measure.parquet
  typology (results/typology/derived):  worklife_annual.parquet, camsis_by_age.parquet
  this module's extracts:               m_e1_cov_extra, m_e1_blood_bc, m_e2_nmr_i0 (participant counts only)
Outputs
  derived/frame_{blood,brain}_{sex}.parquet
  tables/m0_sample_flow.csv    sex x layer: cohort / measured / with layer data / left-work / disease-free
  tables/m0_type_counts.csv    sex x layer x type: n, n in the rest, n left-work, grey flag

  python molecular/py/m0_assemble.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common_mol import (first_nonnull, h1, h2, load_measure, load_person, multi_arr, ok, read_m,
                        setup, sexes, tee, warn)

ETHNIC = {1: "White", 2: "Mixed", 3: "Asian", 4: "Black", 5: "Chinese", 6: "Other"}


def ethnic_group(v: pd.Series) -> pd.Series:
    """Field 21000: top-level group (codes 1001-1003 -> 1, 2001-2004 -> 2, ...)."""
    x = pd.to_numeric(v, errors="coerce")
    top = np.where(x >= 1000, x // 1000, x)
    return pd.Series(pd.Series(top, index=v.index).map(ETHNIC), index=v.index)


# ── 1. Extra covariates ──────────────────────────────────────────────
def covariates(p) -> pd.DataFrame:
    cx = read_m(p, "m_e1_cov_extra")
    out = pd.DataFrame({"eid": cx["eid"]})
    out["cat__ethnic"] = ethnic_group(first_nonnull(cx, multi_arr(cx, "ethnic", 0)))
    return out


# ── 2. Exposure-years before the measurement and pre-measurement CAMSIS ─
def build_ex(annual: pd.DataFrame, cam: pd.DataFrame, age_meas: pd.Series, cfg) -> pd.DataFrame:
    """Over in-job years (in_job == 1) at ages below floor(age at measurement): number of years with each
    exposure > 0, and the mean CAMSIS of those years. Participants with no such year get 0 years (a true
    zero) and missing camsis_pre."""
    ex = cfg["exposure"]
    am = age_meas.dropna()
    a = annual.loc[(annual["in_job"] == 1) & annual["eid"].isin(am.index)]
    a = a[a["age"].to_numpy() < np.floor(a["eid"].map(am).to_numpy())]
    res = pd.DataFrame(index=am.index)
    res.index.name = "eid"
    flags = {}
    for x in ex["agents"]:
        c = f"exp_{x}"
        flags[x] = (a[c] > 0) if c in a.columns else pd.Series(False, index=a.index)
    sl = a["shift_level"] if "shift_level" in a.columns else pd.Series(np.nan, index=a.index)
    flags["shift"] = sl >= 1
    flags["night"] = sl == 2
    hr = a["hours"] if "hours" in a.columns else pd.Series(np.nan, index=a.index)
    flags["longhours"] = hr > ex["long_hours_threshold"]
    F = pd.DataFrame({f"x__{k}": v.astype("int16") for k, v in flags.items()}, index=a.index)
    S = F.groupby(a["eid"]).sum()
    res = res.join(S).fillna(0.0)
    res["work_years_pre"] = a.groupby("eid").size().reindex(res.index).fillna(0.0)
    c = a[["eid", "age"]].merge(cam, on=["eid", "age"], how="left")
    res["camsis_pre"] = c.groupby("eid")["camsis"].mean().reindex(res.index)
    return res.astype("float32").reset_index()


# ── 3. Frame ─────────────────────────────────────────────────────────
def frame(cfg, per, mea, cov, annual, cam, inst):
    m = mea[["eid", f"date_i{inst}", f"age_i{inst}", f"centre_i{inst}", f"cn_i{inst}"]].rename(
        columns={f"date_i{inst}": "date_meas", f"age_i{inst}": "age_meas",
                 f"centre_i{inst}": "centre_meas", f"cn_i{inst}": "cn_meas"})
    df = per.merge(m, on="eid", how="left")
    df = df[df["age_meas"].notna()].copy()
    ce = df["career_end"].astype(float)
    df["exposed_frac"] = ((np.minimum(ce, df["age_meas"]) - 16) / (ce - 16)).clip(0, 1)
    df["left_before"] = (ce <= df["age_meas"]).to_numpy()
    ex = build_ex(annual, cam, df.set_index("eid")["age_meas"], cfg)
    df = df.merge(ex, on="eid", how="left").merge(cov, on="eid", how="left")
    return df.reset_index(drop=True)


# ── 4. Layer availability ────────────────────────────────────────────
def availability(p, mea):
    h2("Layer availability (participant counts only)")
    av = {}
    bc = read_m(p, "m_e1_blood_bc", required=False)
    if bc is not None:
        fields = [c for c in bc.columns if c != "eid" and not c.endswith(("_missing_reason_i0", "_reportability_i0"))
                  and not c.startswith(("fasting_time", "draw_time")) and "_flag" not in c]
        av["LB"] = set(bc.loc[bc[fields].notna().any(axis=1), "eid"])
    nm = read_m(p, "m_e2_nmr_i0", usecols=["eid"], required=False)
    if nm is not None:
        av["LN"] = set(nm["eid"])
    else:
        warn("m_e2_nmr_i0.csv.gz not found (run R/m_e2_nmr_ukbnmr.R); NMR layer counted as 0")
    av["LP"] = set(mea.loc[mea["has_olink"].astype(bool), "eid"]) if "has_olink" in mea else set()
    av["LI"] = set(mea.loc[mea["date_i2"].notna(), "eid"])
    for k, v in av.items():
        print(f"    {k}: {len(v):,} UK Biobank participants")
    return av


def main():
    args, cfg, up, p = setup(__doc__)
    with tee(p, "m0_assemble"):
        h1("m0 analysis frames")
        mea = load_measure(p)
        annual = pd.read_parquet(p["up_derived"] / "worklife_annual.parquet")
        cam = pd.read_parquet(p["up_derived"] / "camsis_by_age.parquet")[["eid", "age", "camsis"]]
        cov = covariates(p)
        av = availability(p, mea)
        flow, tcount = [], []
        gmin = cfg["exposure"]["grey_min_n"]
        for sex in sexes(up):
            h2(sex)
            per = load_person(p, sex)
            for arm, inst in (("blood", 0), ("brain", 2)):
                df = frame(cfg, per, mea, cov, annual, cam, inst)
                df.to_parquet(p["derived"] / f"frame_{arm}_{sex}.parquet", index=False)
                print(f"    frame_{arm}_{sex}: {len(df):,} participants; left work before measurement "
                      f"{df['left_before'].mean():.1%}; disease-free {df['cn_meas'].astype('float').mean():.1%}")
                layers = ("LB", "LN", "LP") if arm == "blood" else ("LI",)
                for L in layers:
                    has = df["eid"].isin(av.get(L, set()))
                    d = df[has]
                    flow.append({"sex": sex, "layer": L, "n_cohort": len(per), "n_measured": len(df),
                                 "n_layer": int(has.sum()), "n_restricted": int(d["left_before"].sum()),
                                 "n_cn": int(d["cn_meas"].astype("boolean").fillna(False).sum())})
                    vc = d["type"].value_counts()
                    for t in sorted(per["type"].dropna().unique()):
                        n1 = int(vc.get(t, 0))
                        n0 = len(d) - n1
                        nr1 = int(((d["type"] == t) & d["left_before"]).sum())
                        tcount.append({"sex": sex, "layer": L, "type": int(t), "n": n1, "n_rest": n0,
                                       "n_restricted": nr1, "grey": (n1 < gmin) or (n0 < gmin)})
        t = pd.DataFrame(flow)
        t.to_csv(p["tables"] / "m0_sample_flow.csv", index=False)
        print(t.to_string(index=False))
        tc = pd.DataFrame(tcount)
        tc.to_csv(p["tables"] / "m0_type_counts.csv", index=False)
        if tc["grey"].any():
            print(f"    not tested (type or rest < {gmin}):\n"
                  + tc[tc["grey"]].groupby(["sex", "layer"]).size().to_string())
        ok("m0 done")


if __name__ == "__main__":
    main()
