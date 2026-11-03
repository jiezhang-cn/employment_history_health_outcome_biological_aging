#!/usr/bin/env python3
"""01: employment-history wide table -> event tables, annual panel, career end,
exposure summaries, CAMSIS and descriptive variables.

Inputs (`paths.data`, file names in `config.files`):
  employment     UKB online employment-history fields (array columns
                 <FieldName>__<instance>_<index>; see `config.employment`)
  cov_baseline   baseline covariates (eid, sex_label, year_birth, age_recruit,
                 birth_cohort_5y, date_assess, centre, townsend, ethnic_group)
  cov_raw        raw Qualifications (field 6138) columns
  work_env       date the employment questionnaire was completed (Category 123)
  camsis (in `paths.refs`)  CAMSIS scores per SOC2000 unit group

Outputs (results/typology/derived/):
  worklife_jobs.parquet      job events (eid, slot, start, end, soc4, hours, shift, exposures)
  worklife_gaps.parquet      gap events (eid, slot, start, end, gap_code)
  worklife_annual.parquet    annual panel, one row per person-year (not truncated)
  worklife_exposure.parquet  exposure summaries, truncated at career_end
  cohort.parquet             cohort, sex, career_end and its source, fold
  camsis.parquet             CAMSIS mean / slope / first / last / pattern
  camsis_by_age.parquet      CAMSIS per (eid, age)
  descriptives.parquet       education, Townsend, ethnic group, birth-year group

Career end is the earliest of definitive retirement, the year after the last
job and age at questionnaire completion (capped at `cohort.age_max`).

Implementation notes:
(a) The panel is built from the union of job and gap years. Expanding jobs and
    then merging gaps would drop gap years without a concurrent job, which
    includes nearly all retirement years.
(b) UKB records calendar years only, so a job ending and another starting in
    the same year look concurrent. A year counts as truly parallel only if it
    has >= 2 jobs and at least one of them strictly spans it (start < y < end).
(c) Exposure summaries use the same right truncation as the tokens.
(d) The coverage denominator is career_end, not age at recruitment.

  python 01_worklife.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import (clean_year, entry_year, fail, fold_of, h1, h2,
                    hours_from_cat, load_cfg, map_exposure, ok, paths,
                    slot_cols, tee, to_long, warn)

COHORT_RE = r"(\d{4})"

# UKB field 6138 Qualifications (coding 100305). Multiple choice: answers are
# pipe-separated (e.g. "1|2|3|6").
EDU_CODES = {1: "Degree", 2: "A level", 3: "O level/GCSE", 4: "CSE",
             5: "NVQ/HND/HNC", 6: "Other professional", -7: "None of the above"}
EDU_RANK = {1: 6, 2: 5, 5: 4, 6: 3, 3: 2, 4: 1, -7: 0}   # higher = higher qualification

# Main-analysis definitions written to cohort.parquet.
TRUNCATE_AT = "career_end"
CAREER_END_CAP = "questionnaire"


# ==========================================================================
# A: work history -> panel
# ==========================================================================
def read_employment(p, cfg):
    f = p["data"] / cfg["files"]["employment"]
    if not f.exists():
        fail(f"missing {f}")
        sys.exit(1)
    allc = list(pd.read_csv(f, nrows=2).columns)
    e = cfg["employment"]
    want = ["eid", f"{e['n_jobs']}__0_0", f"{e['n_gaps']}__0_0"]
    for fld in (e["job_start"], e["job_end"], e["soc4"], e["oscar8"],
                e["hours_cat"], e["hours_exact"], e["shift_any"], e["shift_day"],
                e["shift_mixed"], e["shift_night"], e["gap_code"],
                e["gap_start"], e["gap_end"], e["breath_problem"]):
        want += slot_cols(allc, fld)
    for name in e["exposures"].values():
        want += slot_cols(allc, name)
    want = [c for c in dict.fromkeys(want) if c in allc]
    df = pd.read_csv(f, usecols=want, low_memory=False)
    print(f"  read {len(df):,} rows x {len(want)} columns")
    return df, allc


def build_jobs(df, allc, cfg, fill_year=None):
    """Job event table. The array index is the job index."""
    e = cfg["employment"]
    fields = {"start": e["job_start"], "end": e["job_end"], "soc4": e["soc4"],
              "oscar8": e["oscar8"], "hours_cat": e["hours_cat"],
              "hours_exact": e["hours_exact"], "shift_any": e["shift_any"],
              "shift_day": e["shift_day"], "shift_mixed": e["shift_mixed"],
              "shift_night": e["shift_night"], "breath": e["breath_problem"]}
    for k, v in e["exposures"].items():
        fields[f"exp_{k}"] = v
    L = to_long(df, allc, fields).reset_index()
    # row_id is the index label of df (not a position), so align by label
    L["eid"] = df["eid"].reindex(L["row_id"]).to_numpy()

    n_ongoing = int((pd.to_numeric(L["end"], errors="coerce")
                     == cfg["codings"]["ongoing"]).sum())
    fill = (pd.Series(fill_year.reindex(L["eid"]).to_numpy(), index=L.index)
            if fill_year is not None else cfg["cohort"]["fill_year"])
    L["start"] = clean_year(L["start"], cfg)
    L["end"] = clean_year(L["end"], cfg, ongoing_fill=fill)
    for k in e["exposures"]:
        L[f"exp_{k}"] = map_exposure(L[f"exp_{k}"], cfg)

    # Hours: the lumped category and the exact value are answered for disjoint
    # sets of jobs, so take whichever is present.
    hc = hours_from_cat(L["hours_cat"], cfg)
    he = pd.to_numeric(L["hours_exact"], errors="coerce")
    L["hours"] = he.where(he.notna(), hc)

    # Shift work, three ordered levels: 0 never / 1 shifts without nights / 2 nights
    nd = cfg["codings"]["shift_notdone"]
    night = pd.to_numeric(L["shift_night"], errors="coerce")
    mixed = pd.to_numeric(L["shift_mixed"], errors="coerce")
    any_s = pd.to_numeric(L["shift_any"], errors="coerce")
    has_night = ((night.notna() & (night != nd)) | (mixed.notna() & (mixed != nd)))
    L["shift_level"] = np.where(has_night, 2, np.where(any_s == 1, 1, 0))
    L.loc[any_s.isna(), "shift_level"] = np.nan

    jobs = L[L["start"].notna()].copy()
    n0 = len(jobs)
    bad = jobs["end"] < jobs["start"]
    jobs = jobs[~bad]
    print(f"  job records {n0:,}; dropped end < start {int(bad.sum()):,} -> {len(jobs):,}")
    print(f"  ongoing jobs {n_ongoing:,}, end filled with "
          + ("per-person questionnaire year" if isinstance(fill, pd.Series)
             else f"constant {fill}"))
    return jobs.drop(columns=["row_id"])


def build_gaps(df, allc, cfg, fill_year=None):
    """Gap event table."""
    e = cfg["employment"]
    L = to_long(df, allc, {"gap_code": e["gap_code"], "start": e["gap_start"],
                           "end": e["gap_end"]}).reset_index()
    L["eid"] = df["eid"].reindex(L["row_id"]).to_numpy()
    fill = (pd.Series(fill_year.reindex(L["eid"]).to_numpy(), index=L.index)
            if fill_year is not None else cfg["cohort"]["fill_year"])
    L["start"] = clean_year(L["start"], cfg)
    L["end"] = clean_year(L["end"], cfg, ongoing_fill=fill)
    g = L[L["gap_code"].notna() & L["start"].notna()].copy()
    g = g[g["end"] >= g["start"]]
    print(f"  gap periods {len(g):,} ({g['eid'].nunique():,} participants)")
    return g.drop(columns=["row_id"])


def to_annual(jobs, gaps, birth, cfg):
    """Jobs and gaps -> annual panel, built from their union (note (a))."""
    a_min, a_max = cfg["cohort"]["age_min"], cfg["cohort"]["age_max"]
    exp_names = list(cfg["employment"]["exposures"])

    recs = []
    for src, kind in ((jobs, "job"), (gaps, "gap")):
        if src is None or not len(src):
            continue
        d = src.copy()
        d["kind"] = kind
        yb = birth.reindex(d["eid"]).to_numpy()
        d["a0"] = np.ceil(d["start"].to_numpy() - yb)
        d["a1"] = np.floor(d["end"].to_numpy() - yb)
        d = d[np.isfinite(d["a0"]) & np.isfinite(d["a1"])]
        d["a0"] = d["a0"].clip(lower=a_min)
        d["a1"] = d["a1"].clip(upper=a_max)
        d = d[d["a1"] >= d["a0"]]
        n = (d["a1"] - d["a0"] + 1).astype(int)
        rep = d.loc[d.index.repeat(n)].copy()
        rep["age"] = rep["a0"].astype(int) + rep.groupby(level=0).cumcount()
        recs.append(rep)
    rep = pd.concat(recs, ignore_index=True)

    job_rows = rep[rep["kind"] == "job"].copy()
    # Main job of a year: the one with the longest hours (missing hours last)
    job_rows["_pri"] = job_rows["hours"].fillna(-1)
    job_rows = job_rows.sort_values(["eid", "age", "_pri"])
    n_year = job_rows.groupby(["eid", "age"]).size().rename("n_jobs_year")
    main = job_rows.groupby(["eid", "age"]).tail(1).set_index(["eid", "age"])

    keep = (["soc4", "oscar8", "hours", "shift_level", "breath"]
            + [f"exp_{n}" for n in exp_names])
    A = main[keep].copy()
    A["n_jobs_year"] = n_year

    # Gap years: separate index aligned by reindex on the union of indices
    gr = rep[rep["kind"] == "gap"]
    gmask = gr.groupby(["eid", "age"]).agg(gap_code=("gap_code", "first"))
    idx = A.index.union(gmask.index)
    A = A.reindex(idx)
    A["gap_code"] = gmask["gap_code"].reindex(idx)
    A["n_jobs_year"] = A["n_jobs_year"].fillna(0)
    A["in_job"] = (A["n_jobs_year"] > 0).astype("int8")
    A["in_gap"] = A.index.isin(gmask.index).astype("int8")

    # (b) true parallel jobs vs transition years
    yb_j = birth.reindex(job_rows["eid"]).to_numpy()
    interior = ((job_rows["start"] < job_rows["age"] + yb_j)
                & (job_rows["age"] + yb_j < job_rows["end"]))
    has_int = job_rows.assign(_i=interior).groupby(["eid", "age"])["_i"].any()
    multi = A["n_jobs_year"] > 1
    has_int = has_int.reindex(A.index)
    par = multi & has_int.where(has_int.notna(), False).astype(bool)
    A["is_parallel"] = par.astype("int8")
    A["is_transition"] = (multi & ~par).astype("int8")
    return A.reset_index()


def career_end(annual, age_recruit, cfg, cap=None):
    """Career end age = min(definitive retirement, last working year + 1, cap).

    `cap` is the age at questionnaire completion, capped at `cohort.age_max`.
    1. Retirement is definitive only if no working year follows it; some
       participants return to work after a first retirement record.
    2. Participants without a retirement record end at the last working year
       + 1, so that the end of the sequence remains visible.
    Returns (career_end, components, source); `source` names the component
    that determined the end point (retire / last_job_plus1 / cap).
    """
    ret_code = cfg["codings"]["gap_retirement"]
    ar = age_recruit
    cap = (ar if cap is None else cap).reindex(ar.index)
    a = annual.copy()
    a["_cap"] = cap.reindex(a["eid"]).to_numpy()
    a = a[a["age"].to_numpy() <= a["_cap"].to_numpy()].sort_values(["eid", "age"])
    gc = pd.to_numeric(a["gap_code"], errors="coerce")
    a["_is_ret"] = ((a["in_job"] == 0) & (gc == ret_code)).astype("int8")

    # Rule 1: any working year after this one? (reverse cumulative sum)
    a["_rev"] = a.iloc[::-1].groupby("eid")["in_job"].cumsum().iloc[::-1]
    nxt = a.groupby("eid")["_rev"].shift(-1).fillna(0)
    ret_age = a[(a["_is_ret"] == 1) & (nxt == 0)].groupby("eid")["age"].min()
    last_job = a[a["in_job"] == 1].groupby("eid")["age"].max()

    idx = ar.index
    ce = pd.DataFrame({"retire": ret_age.reindex(idx),
                       "last_job_plus1": last_job.reindex(idx) + 1,
                       "cap": cap.reindex(idx)})
    out = ce.min(axis=1, skipna=True)
    out = out.where(out.notna(), cap.reindex(idx))   # no job and no retirement -> cap
    src = ce.idxmin(axis=1)
    src = src.where(out.notna() & ce.notna().any(axis=1), "cap")
    return (out.clip(lower=cfg["cohort"]["age_min"]).rename("career_end"),
            ce, src.rename("career_end_source"))


def truncate_annual(annual, cut) -> pd.DataFrame:
    """Keep person-years up to each person's cut age (shared with the tokens)."""
    c = cut.reindex(annual["eid"]).to_numpy()
    return annual[annual["age"].to_numpy() <= c]


def exposure_summary(annual_trunc, cfg):
    """Exposure summaries over working years (in_job == 1) up to career_end.

    mean_* = (d1 + 2*d2) / (d0 + d1 + d2), range [0, 2], where dk is the number
    of working years at intensity k with a non-missing answer. peak_* keeps NaN
    when every answer was 'do not know'; cumulative yrs_* counts are filled
    with 0 so that they remain additive.
    """
    exp_names = list(cfg["employment"]["exposures"])
    g = annual_trunc[annual_trunc["in_job"] == 1].groupby("eid")
    out = {}
    for n in exp_names:
        c = f"exp_{n}"
        d1 = g[c].apply(lambda s: (s == 1).sum())
        d2 = g[c].apply(lambda s: (s == 2).sum())
        dn = g[c].apply(lambda s: s.notna().sum())      # d0 + d1 + d2
        out[f"yrs_exp_{n}"] = d1 + d2
        out[f"yrs_high_{n}"] = d2
        out[f"peak_{n}"] = g[c].max()                   # keeps NaN
        out[f"mean_{n}"] = (d1 + 2 * d2) / dn.replace(0, np.nan)
    o = pd.DataFrame(out)
    o["work_years"] = g.size()
    o["long_hours_years"] = g["hours"].apply(
        lambda s: (s >= cfg["cohort"]["long_hours_threshold"]).sum())
    o["shift_years"] = g["shift_level"].apply(lambda s: (s > 0).sum())
    o["night_shift_years"] = g["shift_level"].apply(lambda s: (s == 2).sum())
    cum = [c for c in o.columns
           if c.startswith(("yrs_", "long_", "shift_", "night_", "work_"))]
    o[cum] = o[cum].fillna(0)
    return o


# ==========================================================================
# B: occupational status and descriptive variables
# ==========================================================================
def _camsis_maps(f):
    cm = pd.read_csv(f)
    cm.columns = [c.lower() for c in cm.columns]
    return cm


def _camsis_person_years(cm, coh, annual):
    """CAMSIS of the main job in each working year up to career_end, by sex."""
    soc_col = pd.to_numeric(cm["soc2000"], errors="coerce")
    mmap = dict(zip(soc_col, pd.to_numeric(cm["mcamsis"], errors="coerce")))
    fmap = dict(zip(soc_col, pd.to_numeric(cm["fcamsis"], errors="coerce")))
    a = annual[annual["in_job"] == 1][["eid", "age", "soc4"]].copy()
    ce = coh["career_end"].reindex(a["eid"]).to_numpy()
    a = a[a["age"].to_numpy() <= ce]
    sex = coh["sex"].reindex(a["eid"]).to_numpy()
    soc = pd.to_numeric(a["soc4"], errors="coerce")
    a["camsis"] = np.where(sex == "Female", soc.map(fmap), soc.map(mmap))
    return a


def build_camsis(p, cfg, coh, annual):
    """CAMSIS occupational status: mean, slope with age, first/last value, pattern.

    CAMSIS is not a model input (it is a deterministic function of the SOC
    code), so it can serve as an independent check of the representation.
    Sex-specific scores (mcamsis / fcamsis) are used.
    """
    f = p["refs"] / cfg["files"]["camsis"]
    if not f.exists():
        warn(f"missing {f.name}; CAMSIS skipped")
        return None
    cm = _camsis_maps(f)
    need = {"soc2000", "mcamsis", "fcamsis"}
    if not need <= set(cm.columns):
        warn(f"CAMSIS table lacks columns (needs {need}, has {set(cm.columns)}); skipped")
        return None
    a = _camsis_person_years(cm, coh, annual)
    cover = a["camsis"].notna().mean()
    (ok if cover > 0.95 else warn)(f"CAMSIS coverage of working person-years {cover:.4f}")

    g = a.dropna(subset=["camsis"]).groupby("eid")

    def _slope(d):
        x, y = d["age"].to_numpy(float), d["camsis"].to_numpy(float)
        if len(x) < 5 or np.ptp(x) < 3:
            return np.nan
        return float(np.polyfit(x, y, 1)[0])

    out = pd.DataFrame({
        "camsis_mean": g["camsis"].mean(),
        "camsis_first": g.apply(lambda d: d.sort_values("age")["camsis"].iloc[0],
                                include_groups=False),
        "camsis_last": g.apply(lambda d: d.sort_values("age")["camsis"].iloc[-1],
                               include_groups=False),
        "camsis_slope": g.apply(_slope, include_groups=False),
    }).reindex(coh.index)
    out["camsis_change"] = out["camsis_last"] - out["camsis_first"]
    flat = cfg["profiles"]["camsis_slope_flat"]
    sl = out["camsis_slope"]
    out["camsis_pattern"] = np.where(sl > flat, "rising",
                                     np.where(sl < -flat, "declining", "flat"))
    out.loc[sl.isna(), "camsis_pattern"] = np.nan
    return out


def build_camsis_by_age(p, cfg, coh, annual, camsis_cols):
    """CAMSIS per (eid, age), for age trajectories by career type."""
    f = p["refs"] / cfg["files"]["camsis"]
    if not f.exists() or camsis_cols is None:
        return None
    a = _camsis_person_years(_camsis_maps(f), coh, annual)
    return a.dropna(subset=["camsis"])[["eid", "age", "camsis"]]


def build_education(p, cfg, index):
    """Highest qualification rebuilt from all answers to field 6138.

    The question is multiple choice; taking the highest option across all
    answers and instances avoids losing pipe-separated multi-answer values.
    """
    f = p["data"] / cfg["files"]["cov_raw"]
    if not f.exists():
        warn(f"missing {f.name}; education not rebuilt")
        return None
    qcols = [c for c in pd.read_csv(f, nrows=0).columns
             if c.startswith("Qualifications__")]
    if not qcols:
        warn("no Qualifications columns in raw table; education not rebuilt")
        return None
    d = pd.read_csv(f, usecols=["eid"] + qcols, low_memory=False).set_index("eid")

    # wide -> one row per selected option -> maximum rank per eid
    long = (d[qcols].stack().astype(str).str.split("|").explode()
            .str.strip().rename("code"))
    ranks = pd.to_numeric(long, errors="coerce").map(EDU_RANK)
    best = ranks.dropna().groupby(level=0).max()        # level 0 = eid

    inv = {v: k for k, v in EDU_RANK.items()}
    out = pd.DataFrame({
        "edu_rank": best,
        "edu_highest": best.map(
            lambda r: EDU_CODES.get(inv.get(r)) if pd.notna(r) else np.nan),
    }).reindex(index)
    out["edu_degree"] = (out["edu_rank"] == EDU_RANK[1]).astype("float")
    out.loc[out["edu_rank"].isna(), "edu_degree"] = np.nan
    cov = out["edu_highest"].notna().mean()
    (ok if cov > 0.95 else warn)(f"edu_highest available for {cov:.1%}")
    return out


def build_descriptives(p, cfg, coh, edu):
    """Descriptive variables used for profiles and stratification."""
    src = pd.read_csv(p["data"] / cfg["files"]["cov_baseline"]
                      ).set_index("eid").reindex(coh.index)
    keep = {}
    if "birth_cohort_5y" in coh.columns:
        keep["birth_year_group"] = pd.to_numeric(
            coh["birth_cohort_5y"].astype(str).str.extract(COHORT_RE)[0],
            errors="coerce")
    for c in ("townsend", "ethnic_group"):
        if c in src.columns:
            keep[c] = src[c]
        else:
            warn(f"missing {c}")
    out = pd.DataFrame(keep, index=coh.index)
    if edu is not None:
        for c in edu.columns:
            out[c] = edu[c]
    return out


# ==========================================================================
def main():
    cfg = load_cfg()
    p = paths(cfg)

    with tee(p, "01_worklife"):
        h1("01: work history -> panel, career end, exposures, CAMSIS, descriptives")

        h2("Baseline covariates")
        base = pd.read_csv(p["data"] / cfg["files"]["cov_baseline"])
        base["date_assess"] = pd.to_datetime(base["date_assess"], errors="coerce")
        base = base.set_index("eid")
        print(f"  {len(base):,} participants")

        h2("Employment-history table")
        df, allc = read_employment(p, cfg)
        njf = f"{cfg['employment']['n_jobs']}__0_0"
        df = df[df[njf].notna() & (df[njf] > 0)]
        print(f"  participants reporting >= 1 job: {len(df):,}")

        h2("Questionnaire completion year")
        ey = entry_year(p, cfg, index=pd.Index(df["eid"]))

        h2("Job events")
        jobs = build_jobs(df, allc, cfg, fill_year=ey)
        yb = base["year_birth"].reindex(jobs["eid"]).to_numpy()
        too_young = jobs["start"].to_numpy() < yb + cfg["cohort"]["min_work_age"]
        print(f"  dropped jobs starting before age {cfg['cohort']['min_work_age']}: "
              f"{int(too_young.sum()):,}")
        jobs = jobs[~too_young]

        h2("Gap events")
        gaps = build_gaps(df, allc, cfg, fill_year=ey)

        h2("Annual panel (union of job and gap years)")
        birth = base["year_birth"]
        ann = to_annual(jobs, gaps, birth, cfg)
        print(f"  {len(ann):,} person-years, {ann['eid'].nunique():,} participants")

        h2("Career end")
        idx = base.index.intersection(pd.Index(ann["eid"].unique()))
        ar = base["age_recruit"].reindex(idx)
        amax = cfg["cohort"]["age_max"]
        cap = (ey.reindex(idx) - base["year_birth"].reindex(idx)).clip(upper=amax)
        ce, ce_parts, ce_src = career_end(ann, ar, cfg, cap=cap)
        print("  end point determined by: " + ", ".join(
            f"{k} {v:,}" for k, v in ce_src.value_counts().items()))

        ann_t = truncate_annual(ann, ce)
        L1 = ann_t.groupby("eid").size()
        print(f"  person-years after truncation {len(ann_t):,}; "
              f"sequence length median {L1.median():.0f}, max {int(L1.max())}")
        ml = cfg["model"]["max_len"]
        n_over = int((L1 > ml - 1).sum())
        (ok if n_over == 0 else warn)(
            f"sequences longer than max_len - 1 = {ml - 1}: {n_over:,}"
            + ("" if n_over == 0 else " (earliest years will be dropped)"))
        obs_ret = ce_parts["retire"].notna().astype(int).reindex(idx)

        h2("Exposure summaries (same truncation as tokens)")
        expo = exposure_summary(ann_t, cfg)
        print(f"  {expo.shape[1]} features x {len(expo):,} participants")

        h2("Cohort table")
        # (d) coverage denominator is career_end
        nobs = ann_t.groupby("eid")["age"].nunique().reindex(idx)
        span = (ce - cfg["cohort"]["age_min"] + 1).clip(lower=1)
        ratio = (nobs / span).clip(upper=1.0)
        mxp = ann_t.groupby("eid")["n_jobs_year"].max().reindex(idx)
        coh = pd.DataFrame({
            "sex": base["sex_label"].reindex(idx),
            "age_recruit": ar,
            "birth_cohort_5y": base["birth_cohort_5y"].reindex(idx),
            "assess_year": base["date_assess"].reindex(idx).dt.year,
            "entry_year": ey.reindex(idx),
            "centre": base["centre"].reindex(idx),
            "career_end": ce,
            "retire_age": ce_parts["retire"],
            "observed_retirement": obs_ret.astype("int8"),
            "career_end_source": ce_src,
            "truncate_at": TRUNCATE_AT,
            "career_end_cap": CAREER_END_CAP,
            "coverage": ratio,
            "max_parallel": mxp,
            "flag_incomplete": (ratio < cfg["cohort"]["min_coverage"]
                                ).fillna(True).astype("int8"),
            "flag_overlap": (mxp > cfg["cohort"]["max_overlap"]
                             ).fillna(False).astype("int8"),
        })
        coh["fold"] = fold_of(idx, cfg["cohort"]["n_folds"])
        print(f"  cohort {len(coh):,} participants")
        for s, n in coh["sex"].value_counts().items():
            print(f"    {s}: {n:,}")

        h2("CAMSIS occupational status (not a model input)")
        cam = build_camsis(p, cfg, coh, ann)
        cam_age = build_camsis_by_age(p, cfg, coh, ann, cam)

        h2("Education (rebuilt from field 6138)")
        edu = build_education(p, cfg, coh.index)

        h2("Descriptive variables")
        desc = build_descriptives(p, cfg, coh, edu)

        h2("Write")
        d = p["derived"]
        jobs.to_parquet(d / "worklife_jobs.parquet", index=False)
        gaps.to_parquet(d / "worklife_gaps.parquet", index=False)
        ann.to_parquet(d / "worklife_annual.parquet", index=False)
        expo.reset_index(names="eid").to_parquet(
            d / "worklife_exposure.parquet", index=False)
        coh.reset_index(names="eid").to_parquet(d / "cohort.parquet", index=False)
        for f in ("worklife_jobs", "worklife_gaps", "worklife_annual",
                  "worklife_exposure", "cohort"):
            ok(f"{f}.parquet")
        if cam is not None:
            cam.reset_index(names="eid").to_parquet(d / "camsis.parquet", index=False)
            ok("camsis.parquet")
        if cam_age is not None:
            cam_age.to_parquet(d / "camsis_by_age.parquet", index=False)
            ok("camsis_by_age.parquet")
        desc.reset_index(names="eid").to_parquet(
            d / "descriptives.parquet", index=False)
        ok("descriptives.parquet")
        return 0


if __name__ == "__main__":
    sys.exit(main())
