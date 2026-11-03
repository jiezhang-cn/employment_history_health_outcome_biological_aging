"""a0: assemble measurement time points, the disease-free (CN) reference subset, modality availability,
person-level career exposures and lifetime intensities. No clock or age gap is touched here.

Inputs: typology outputs (results/typology/derived: cohort, types_{sex}, camsis, camsis_by_age,
seqmove_{sex}, worklife_exposure, worklife_annual, descriptives), e1 extracts (covariates, imaging_qc,
mri_multi, brain_tian, fc_d25p, dictionary), UKB tables (cov_baseline, first occurrences, cancer
registry, cov_raw, Olink long table) and Tian et al. Table S7 (references/).

Outputs (results/aging/derived/):
  measure.parquet          one row per UKB participant: dates, continuous ages and centres at instances
                           0 and 2, CN status at each, modality availability, cohort membership
  cn.parquet               earliest date of any Table S7 disease and of any malignancy
  person_{sex}.parquet     work-history cohort x career type x career axes x education
  qb_annual_{sex}.parquet  annual intensities (0-1) of each exposure in working person-years
  qb_i0_{sex}.parquet / qb_i2_{sex}.parquet
                           lifetime intensities ({axis}__life): mean annual intensity over working years
                           before measurement (Olink participants at instance 0; imaging at instance 2)
Tables (results/aging/tables/):
  a0_intersection.csv      cohort x modality x sex: n, n who had left work, n disease-free
  a0_type_counts.csv       participants per career type and modality

  python aging/py/a0_assemble.py [--olink-long PATH]
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common_aging import (birth_date, brain_features, build_education, data_file, fail, h1, h2, load_fc,
                          mask_multi_errors, multi_features, ok, olink_long_file, olink_long_sep,
                          read_dictionary, read_extract, read_wide, ref_file, setup, sexes, tee, to_date,
                          warn, years_between)


# ── 1. Measurement time points ─────────────────────────────────────
def build_timepoints(p, cfg):
    h2("Measurement time points: i0 = baseline (blood draw), i2 = first imaging visit")
    base = pd.read_csv(data_file(p, cfg, "cov_baseline"),
                       usecols=["eid", "sex", "date_assess", "centre", "birth_date"], low_memory=False)
    base["eid"] = pd.to_numeric(base["eid"], errors="coerce")
    base = base.dropna(subset=["eid"]).astype({"eid": "int64"})
    base = base[base["eid"] > 0].drop_duplicates("eid")
    m = pd.DataFrame({"eid": base["eid"].to_numpy()})
    m["sex"] = base["sex"].map({0: "Female", 1: "Male"}).to_numpy()
    m["birth_date"] = to_date(base["birth_date"]).to_numpy()
    m["date_i0"] = to_date(base["date_assess"]).to_numpy()
    m["centre_i0"] = base["centre"].to_numpy()

    e1 = read_extract(p, "covariates")
    bd1 = birth_date(e1["birth_year_i0"], e1["birth_month_i0"], cfg["measure"]["birth_day"])
    e1 = e1.assign(_bd=bd1.to_numpy())
    e1["date_i2"] = to_date(e1["date_attend_i2"])
    m = m.merge(e1[["eid", "_bd", "date_i2", "centre_i2"]], on="eid", how="outer")
    m["birth_date"] = m["birth_date"].fillna(m["_bd"])
    m = m.drop(columns="_bd")
    if "sex_i0" in e1.columns:
        s1 = e1.set_index("eid")["sex_i0"].map({0: "Female", 1: "Male"})
        m["sex"] = m["sex"].fillna(m["eid"].map(s1))
    for k in (0, 2):
        m[f"age_i{k}"] = years_between(m["birth_date"], m[f"date_i{k}"])
        print(f"    i{k}: {m[f'date_i{k}'].notna().sum():,} with a date, "
              f"age {np.nanmin(m[f'age_i{k}']):.1f}-{np.nanmax(m[f'age_i{k}']):.1f}")
    return m


# ── 2. Disease-free (CN) reference subset ──────────────────────────
def _parse_ranges(ranges):
    out = []
    for r in ranges or []:
        a, b = r.split("-")
        out.append((a.strip(), b.strip()))
    return out


def _all_three(a, b):
    """Expand a range such as 'I00'-'I99' into its 3-character codes (same letter)."""
    if a[0] != b[0]:
        raise ValueError(f"extra_ranges must stay within one letter: {a}-{b}")
    return {f"{a[0]}{i:02d}" for i in range(int(a[1:3]), int(b[1:3]) + 1)}


def build_cn(p, cfg, m):
    """CN = no Table S7 disease and no malignancy on or before the measurement date.

    First occurrences only carry 3-character ICD-10 codes, so S7 codes are truncated to 3 characters; where
    S7 lists only sub-codes this over-includes (e.g. K70 for K702-K704). Those codes are logged.
    """
    h2("Disease-free reference subset (Tian Table S7 + cancer registry)")
    cc = cfg["cn"]
    s7 = pd.read_csv(ref_file(cfg, cc["s7_table"], required=True))
    s7 = s7[s7["code_type"].astype(str).str.replace(" ", "").str.upper() == "ICD10"]
    if cc["categories"] != "all":
        s7 = s7[s7["category"].isin(cc["categories"])]
    codes = s7["code"].astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
    three = set(codes.str[:3])
    for a, b in _parse_ranges(cc.get("extra_ranges")):
        three |= _all_three(a, b)
    listed3 = set(codes[codes.str.len() == 3])
    partial = sorted(t for t in three if t not in listed3)
    if partial:
        print(f"    3-character codes covering S7 sub-codes only (over-inclusive): {', '.join(partial)}")
    fo = data_file(p, cfg, "first_occ")
    hdr = pd.read_csv(fo, nrows=0).columns
    pat = re.compile(r"^Date_([A-Z]\d\d)_first_reported")
    dcols = {c: pat.match(c).group(1) for c in hdr if pat.match(c) and pat.match(c).group(1) in three}
    miss = sorted(three - set(dcols.values()))
    if miss:
        warn(f"3-character codes absent from first occurrences: {miss}")
    d = pd.read_csv(fo, usecols=["eid"] + list(dcols), dtype=str)
    d["eid"] = pd.to_numeric(d["eid"], errors="coerce")
    d = d.dropna(subset=["eid"]).astype({"eid": "int64"}).set_index("eid")
    prev = set(cc["prevalent_dates"])
    und = cc["undated_date"]
    vals = d.to_numpy(dtype=object)
    is_und = (vals == und)
    early = np.isin(vals, list(prev))
    dt = pd.DataFrame(vals).apply(lambda s: pd.to_datetime(s, errors="coerce")).to_numpy("datetime64[ns]")
    dt = np.where(early, np.datetime64("1900-01-01"), dt)
    dt = np.where(is_und, np.datetime64("NaT"), dt)
    s7_min = pd.Series(np.nanmin(np.where(np.isnat(dt), np.datetime64("2262-01-01"), dt), axis=1),
                       index=d.index)
    s7_min = s7_min.where(s7_min < pd.Timestamp("2262-01-01"))
    s7_und = pd.Series(is_und.any(axis=1), index=d.index)
    cn = pd.DataFrame({"s7_first": s7_min, "s7_undated": s7_und})

    if cc["use_cancer_register"]:
        cf = data_file(p, cfg, "cancer", required=False)
        if cf is None:
            warn("cancer registry not found; CN does not exclude malignancies")
            cn["cancer_first"] = pd.NaT
        else:
            ch = pd.read_csv(cf, nrows=0).columns
            ks = sorted({int(re.search(r"__(\d+)_0$", c).group(1)) for c in ch
                         if c.startswith("Date_of_cancer_diagnosis__")})
            use = ["eid"] + [c for k in ks for c in
                             (f"Date_of_cancer_diagnosis__{k}_0", f"Type_of_cancer_ICD10__{k}_0",
                              f"Type_of_cancer_ICD9__{k}_0", f"Behaviour_of_cancer_tumour__{k}_0")
                             if c in ch]
            c = pd.read_csv(cf, usecols=use, dtype=str)
            c["eid"] = pd.to_numeric(c["eid"], errors="coerce")
            c = c.dropna(subset=["eid"]).astype({"eid": "int64"})
            keep_b = {str(b) for b in cc["cancer_behaviour_keep"]}
            parts = []
            for k in ks:
                dc, ic10, ic9, bc = (f"Date_of_cancer_diagnosis__{k}_0", f"Type_of_cancer_ICD10__{k}_0",
                                     f"Type_of_cancer_ICD9__{k}_0", f"Behaviour_of_cancer_tumour__{k}_0")
                if dc not in c.columns:
                    continue
                i10 = c.get(ic10, pd.Series(np.nan, index=c.index)).astype(str).str.upper()
                i9 = pd.to_numeric(c.get(ic9, pd.Series(np.nan, index=c.index)).astype(str).str[:3],
                                   errors="coerce")
                mal = i10.str.startswith("C") | i9.between(140, 208)
                beh = c.get(bc, pd.Series(np.nan, index=c.index))
                bok = beh.isna() | beh.astype(str).str.split(".").str[0].isin(keep_b)
                sel = mal & bok & c[dc].notna()
                parts.append(pd.DataFrame({"eid": c.loc[sel, "eid"], "d": to_date(c.loc[sel, dc])}))
            cd = pd.concat(parts) if parts else pd.DataFrame(columns=["eid", "d"])
            cfirst = cd.dropna().groupby("eid")["d"].min()
            print(f"    cancer registry: {len(cfirst):,} participants with a malignancy "
                  f"(ICD-10 C / ICD-9 140-208, behaviour in {sorted(keep_b)} or missing)")
            cn = cn.join(cfirst.rename("cancer_first"), how="outer")
    cn["s7_undated"] = cn["s7_undated"].fillna(False).astype(bool)
    cn.index.name = "eid"
    cn = cn.reset_index()

    mm = m[["eid"]].merge(cn, on="eid", how="left")
    first = mm[["s7_first", "cancer_first"]].min(axis=1)
    und_d = mm["s7_undated"].eq(True) & bool(cc["undated_as_disease"])
    in_fo = mm["eid"].isin(d.index)
    for k in (0, 2):
        dk = m[f"date_i{k}"]
        has = dk.notna() & in_fo
        sick = (first.notna() & (first <= dk)) | und_d
        v = pd.Series(np.where(has, ~sick, np.nan), index=m.index, dtype="object")
        m[f"cn_i{k}"] = v.map({True: True, False: False}).astype("boolean")
        n = int(has.sum())
        share = float((~sick[has]).mean()) if n else np.nan
        print(f"    i{k}: CN share {share:.1%} (n={n:,})")
    if not in_fo.all():
        warn(f"{(~in_fo).sum():,} participants not in the first-occurrence file -> CN status missing")
    cn.to_parquet(p["derived"] / "cn.parquet", index=False)
    return m


# ── 3. Modality availability (same complete-case rule as a2) ───────
def olink_eids(p, cfg, override=None):
    """Participants with Olink data: {instance: set of eids}, from the eid / ins_index columns of the
    long table (read in chunks; cached in derived/olink_eids.parquet while newer than the source)."""
    f = olink_long_file(p, cfg, override)
    if f is None:
        return None
    cache = p["derived"] / "olink_eids.parquet"
    if cache.exists() and cache.stat().st_mtime > f.stat().st_mtime:
        c = pd.read_parquet(cache)
        ok(f"Olink long table {f.name}: read cache {cache.name}")
    else:
        sep, hdr = olink_long_sep(f)
        if not {"eid", "ins_index"} <= set(hdr):
            fail(f"{f} is not an olink_data long table (header {hdr[:8]}); needs eid / ins_index")
            raise SystemExit(1)
        parts = []
        for ch in pd.read_csv(f, sep=sep, usecols=["eid", "ins_index"], chunksize=cfg["protein"]["long_chunk_rows"],
                              dtype={"eid": "int64", "ins_index": "int16"}):
            parts.append(ch.drop_duplicates())
        c = pd.concat(parts, ignore_index=True).drop_duplicates()
        c = c[c["eid"] > 0]
        c.to_parquet(cache, index=False)
    out = {int(k): set(g["eid"]) for k, g in c.groupby("ins_index")}
    print("    Olink participants by instance: " + ", ".join(f"i{k} {len(v):,}" for k, v in sorted(out.items())))
    return out


def build_availability(p, cfg, m, olink_override=None):
    h2("Modality availability")
    ids = m["eid"]
    oe = olink_eids(p, cfg, olink_override)
    if oe is None:
        warn("Olink long table not found (files.olink_long / --olink-long); no proteomic participants")
        m["has_olink"] = False
    else:
        m["has_olink"] = ids.isin(oe.get(0, set()))            # baseline only
    dic = read_dictionary(p)
    qc = read_extract(p, "imaging_qc", dtype=str).set_index("eid")
    k = 2                                                      # first imaging visit only
    for organ in cfg["multi"]["organs"]:
        cols = multi_features(dic, organ, k)
        if not cols:
            m[f"has_CM_{organ}_i{k}"] = False
            continue
        X = mask_multi_errors(read_wide(p, "mri_multi", cols), qc, cfg, k)
        m[f"has_CM_{organ}_i{k}"] = ids.isin(X.index[X.notna().all(axis=1)])
    fc = load_fc(p, k)
    allc = sorted({c for s in cfg["tian"]["models"].values() for c in brain_features(dic, s["subgroups"], k)})
    XB = read_wide(p, "brain_tian", allc) if allc else None
    for name, spec in cfg["tian"]["models"].items():
        cols = [c for c in brain_features(dic, spec["subgroups"], k) if XB is not None and c in XB.columns]
        idx = set(XB.index[XB[cols].notna().all(axis=1)]) if cols else None
        if spec["subgroups"] and not cols:
            idx = set()                                        # as in a2: no structural features -> not available
        elif spec["fc"]:
            fidx = set(fc.index[fc.notna().all(axis=1)]) if len(fc) else set()
            idx = fidx if idx is None else idx & fidx
        m[f"has_CB_{name}_i{k}"] = ids.isin(idx or set())
    for c in [c for c in m.columns if c.startswith("has_")]:
        print(f"    {c:<24} {int(m[c].sum()):>7,}")
    return m


# ── 4. Person level ────────────────────────────────────────────────
def build_person(p, cfg, sex):
    ud = p["up_derived"]
    co = pd.read_parquet(ud / "cohort.parquet")
    co = co[co["sex"] == sex]
    ty = pd.read_parquet(ud / f"types_{sex}.parquet")
    d = co[["eid", "sex", "age_recruit", "birth_cohort_5y", "centre", "career_end"]].merge(
        ty[["eid", "type"]], on="eid", how="inner")
    cam = pd.read_parquet(ud / "camsis.parquet")[["eid", "camsis_mean", "camsis_slope", "camsis_change"]]
    sm = pd.read_parquet(ud / f"seqmove_{sex}.parquet")[["eid", "n_moves", "net_displacement"]]
    agents = cfg["axes"]["agents"]
    ex = pd.read_parquet(ud / "worklife_exposure.parquet")
    ex = ex[["eid"] + [c for c in (f"yrs_exp_{a}" for a in agents) if c in ex.columns]]
    for x in (cam, sm, ex):
        d = d.merge(x, on="eid", how="left")
    ye = d[[f"yrs_exp_{a}" for a in agents]]
    d["exposure_breadth"] = (ye > 0).sum(axis=1).where(ye.notna().any(axis=1))
    d["longest_agent_years"] = ye.max(axis=1)
    d["edu_highest"] = build_education(p, cfg, pd.Index(d["eid"])).to_numpy()
    return d


# ── 5. Annual intensities and lifetime intensities ─────────────────
def build_qb_annual(p, cfg, annual, sex_eids, sex):
    """Annual intensity (0-1) in each working person-year: low status = (p95 - CAMSIS)/(p95 - p5) within
    sex; agents = rating / 2; night shifts and long hours = 0/1; agent_breadth = share of agents > 0."""
    qb = cfg["qb"]
    a = annual.loc[(annual["in_job"] == 1) & annual["eid"].isin(sex_eids)].copy()
    cam = pd.read_parquet(p["up_derived"] / "camsis_by_age.parquet")
    a = a.merge(cam, on=["eid", "age"], how="left")
    lo, hi = np.nanquantile(a["camsis"], qb["camsis_low_q"])
    out = pd.DataFrame({"eid": a["eid"].to_numpy(), "age": a["age"].to_numpy().astype("int16")})
    out["camsis_low"] = ((hi - a["camsis"]) / (hi - lo)).clip(0, 1).to_numpy("float32")
    agents = cfg["axes"]["agents"]
    E = a[[f"exp_{x}" for x in agents]]
    out["agent_breadth"] = ((E > 0).sum(axis=1) / len(agents)).where(E.notna().any(axis=1)).to_numpy("float32")
    out["night_share"] = (a["shift_level"] == 2).astype("float32").where(a["shift_level"].notna()).to_numpy()
    out["long_hours"] = (a["hours"] > cfg["axes"]["long_hours_threshold"]).astype("float32") \
        .where(a["hours"].notna()).to_numpy()
    for x in agents:
        out[x] = (a[f"exp_{x}"] / 2.0).clip(0, 1).to_numpy("float32")
    print(f"    {sex}: {len(out):,} working person-years; CAMSIS p5/p95 = {lo:.1f}/{hi:.1f}")
    out.to_parquet(p["derived"] / f"qb_annual_{sex}.parquet", index=False)
    return out


def qb_axes(cfg):
    return list(dict.fromkeys(cfg["qb"]["axes"] + cfg["qb"]["aux_axes"]))


def lifetime_intensity(cfg, qa, age_meas: pd.Series) -> pd.DataFrame:
    """{axis}__life = mean annual intensity over working years before measurement (age < floor(age at
    measurement)); missing when fewer than qb.min_years such years are available."""
    mb = cfg["qb"]["min_years"]
    axes = [x for x in qb_axes(cfg) if x in qa.columns]
    am = age_meas.dropna()
    a = qa[qa["eid"].isin(am.index)].copy()
    a["am"] = a["eid"].map(am).to_numpy()
    a = a[a["age"] < np.floor(a["am"])]
    g = a.groupby("eid")[axes]
    life, life_n = g.mean(), g.count()
    res = pd.DataFrame({f"{x}__life": life[x].where(life_n[x] >= mb).reindex(am.index) for x in axes},
                       index=am.index)
    res.index.name = "eid"
    return res.astype("float32").reset_index()


def write_lifetime(p, cfg, m, d, qa, sex):
    mi = m.set_index("eid")
    for k in (0, 2):
        am = mi[f"age_i{k}"].reindex(d["eid"])
        if k == 0:
            am = am.where(mi["has_olink"].reindex(d["eid"], fill_value=False).astype(bool))
        q = lifetime_intensity(cfg, qa, am)
        q.to_parquet(p["derived"] / f"qb_i{k}_{sex}.parquet", index=False)
        x0 = cfg["qb"]["axes"][0]
        print(f"    {sex} i{k}: {len(q):,} participants; {int(q[x0 + '__life'].notna().sum()):,} with {x0}")


# ── 6. Sample-size tables ──────────────────────────────────────────
def modality_flags(m):
    f = {"olink": m["has_olink"]}
    for c in m.columns:
        mm = re.match(r"^has_(C[MB]_\w+)_i2$", c)
        if mm:
            f[mm.group(1)] = m[c]
    f["imaging_any"] = m["date_i2"].notna()
    return f


def sample_tables(p, cfg, m, persons):
    h2("Sample sizes by modality")
    min_type = cfg["qa"]["min_type_n"]
    flags = modality_flags(m)
    rows, trows = [], []
    for sex, d in persons.items():
        mm = m.set_index("eid").reindex(d["eid"])
        for name, fl in flags.items():
            has = fl.set_axis(m["eid"]).reindex(d["eid"]).fillna(False).to_numpy(bool)
            inst = 0 if name == "olink" else 2
            age = mm[f"age_i{inst}"].to_numpy()
            left = d["career_end"].to_numpy() <= age
            rows.append({"sex": sex, "modality": name, "instance": inst, "n_cohort": len(d),
                         "n": int(has.sum()), "n_left_before": int((has & left).sum()),
                         "n_cn": int((has & (mm[f"cn_i{inst}"] == True).to_numpy()).sum())})   # noqa: E712
            if name in ("olink", "CB_whole", "imaging_any") or name.startswith("CM_"):
                vc = pd.Series(d["type"].to_numpy()[has]).value_counts()
                for t in sorted(d["type"].unique()):
                    n = int(vc.get(t, 0))
                    trows.append({"sex": sex, "modality": name, "type": int(t), "n": n,
                                  "below_min": n < min_type})
    it = pd.DataFrame(rows)
    both = []
    for sex, d in persons.items():
        e = d["eid"]
        ol = m.set_index("eid")["has_olink"].reindex(e, fill_value=False).to_numpy(bool)
        im = m.set_index("eid")["date_i2"].reindex(e).notna()
        both.append({"sex": sex, "modality": "olink_and_imaging", "instance": -1, "n_cohort": len(d),
                     "n": int((ol & im).sum())})
    it = pd.concat([it, pd.DataFrame(both)], ignore_index=True)
    it.to_csv(p["tables"] / "a0_intersection.csv", index=False)
    print(it.to_string(index=False))
    pd.DataFrame(trows).to_csv(p["tables"] / "a0_type_counts.csv", index=False)


# ── main ───────────────────────────────────────────────────────────
def main():
    args, cfg, up, p = setup(__doc__, [(("--olink-long",), {"default": None,
                                        "help": "path of the olink_data long table (overrides files.olink_long)"})])
    with tee(p, "a0_assemble"):
        h1("a0 assemble: time points, CN subset, availability, person-level exposures")
        print(f"    typology: {p['upstream']}\n    e1: {p['extract']}\n    output: {p['out']}")
        m = build_timepoints(p, cfg)
        m = build_cn(p, cfg, m)
        m = build_availability(p, cfg, m, args.olink_long)
        h2("Annual work-history table")
        annual = pd.read_parquet(p["up_derived"] / "worklife_annual.parquet")
        print(f"    worklife_annual: {len(annual):,} rows ({int(annual['in_job'].sum()):,} in work)")
        persons = {}
        for sex in sexes(up):
            h2(f"Person level: {sex}")
            d = build_person(p, cfg, sex)
            persons[sex] = d
            print(f"    {sex}: {len(d):,} participants, {d['type'].nunique()} types")
        h2("Annual and lifetime intensities")
        for sex, d in persons.items():
            d.to_parquet(p["derived"] / f"person_{sex}.parquet", index=False)
            qa = build_qb_annual(p, cfg, annual, set(d["eid"]), sex)
            write_lifetime(p, cfg, m, d, qa, sex)
        m["in_cohort"] = m["eid"].isin(pd.concat([d["eid"] for d in persons.values()]))
        m.to_parquet(p["derived"] / "measure.parquet", index=False)
        sample_tables(p, cfg, m, persons)
        ok("a0 done")


if __name__ == "__main__":
    main()
