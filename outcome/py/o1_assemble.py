#!/usr/bin/env python3
"""o1: assemble time zero, follow-up, disease matrices and the code catalogue.

No associations are estimated here.

Time zero is the date the online work-history questionnaire was completed.
Diagnoses are three-character ICD-10 codes from the UK Biobank first-occurrence
fields plus malignant tumours from the cancer registry (first occurrences
contain no C codes, so chapter II comes from the registry alone).

Inputs (paths in config_outcome.yaml):
  results/typology/derived/types_{sex}.parquet   career types (typology output)
  data/ files.cov_baseline     eid, birth_date, centre, ethnic_group
  data/ files.cov_raw          Qualifications__*, Date_lost_to_follow_up__0_0
  data/ files.work_env         eid, When_occupational_data_entered__0_0
  data/ files.death            eid, date_of_death
  data/ files.first_occ        Date_<code>_first_reported_*__0_0 columns
  data/ files.cancer           cancer registry instances (date, ICD-10, ICD-9, behaviour)

Outputs (results/outcome/):
  derived/op_person.parquet            one row per participant: type, time zero,
                                       follow-up, death, covariates
  derived/op_disease/{prevalent,incident}.npz
                                       sparse person x code boolean matrices
                                       (first record on/before time zero; after
                                       time zero up to the end of follow-up)
  derived/op_disease/meta.json, eid.npy, code_chapters.csv
  tables/OP0_code_catalog.csv          per code and sex: lifetime prevalence and
                                       the >= op.min_prevalence selection
  tables/OP0_feasibility.csv           per sex and type: counts
  qc/op_timeline.json, qc/op_disease_qc.json

  python o1_assemble.py --stage all
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common_out import (build_education, chapters_of, data_file, fail, h1, h2,
                        load_cfg_out, load_cfg_up, ok, parse_code_columns,
                        paths_out, sexes, tee, warn, write_json)

FAR = np.datetime64("2099-12-31")


# -- Person table -------------------------------------------------------------
def load_upstream_types(p, cup) -> pd.DataFrame:
    """Career types from the typology pipeline (read only)."""
    parts = []
    for sex in sexes(cup):
        f = p["up_derived"] / f"types_{sex}.parquet"
        if not f.exists():
            fail(f"{f} missing; run typology/04_typology.py first")
            sys.exit(1)
        t = pd.read_parquet(f)[["eid", "type"]]
        t["sex"] = sex
        parts.append(t)
        ok(f"types_{sex}.parquet  n={len(t):,}  K={t['type'].nunique()}")
    return pd.concat(parts, ignore_index=True)


def build_person(p, cfg, cup) -> pd.DataFrame:
    tl = cfg["timeline"]
    df = load_upstream_types(p, cup).set_index("eid")

    base = pd.read_csv(data_file(p, cfg, "cov_baseline")).set_index("eid")
    want = ["birth_date", "centre", "ethnic_group"]
    miss = [c for c in want if c not in base.columns]
    if miss:
        warn(f"cov_baseline lacks {miss}")
    df = df.join(base[[c for c in want if c in base.columns]], how="left")
    df["edu_highest"] = build_education(p, cfg, df.index)

    we = pd.read_csv(data_file(p, cfg, "work_env"),
                     usecols=["eid", tl["tz_col"]]).set_index("eid")
    df["tz_date"] = pd.to_datetime(we[tl["tz_col"]], errors="coerce").reindex(df.index)
    n_no_tz = int(df["tz_date"].isna().sum())
    if n_no_tz:
        fail(f"{n_no_tz:,} participants have no time zero; check the cohort and files")
        sys.exit(1)
    df["tz_year"] = df["tz_date"].dt.year
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["tz_age"] = (df["tz_date"] - df["birth_date"]).dt.days / 365.25
    return df


def attach_followup(df, p, cfg) -> tuple[pd.DataFrame, dict]:
    """Death, loss to follow-up and administrative censoring.

    `fu_years` is the potential follow-up (censoring date minus time zero),
    not the observed time, which for decedents would be the outcome itself.
    """
    tl = cfg["timeline"]
    d = pd.read_csv(data_file(p, cfg, "death"), sep="\t")
    d["date_of_death"] = pd.to_datetime(d["date_of_death"], errors="coerce")
    # The death table can hold several records per participant: keep the earliest.
    n_raw, n_eid = len(d), d["eid"].nunique()
    dd = d.groupby("eid")["date_of_death"].min()
    qc = {"death_rows": n_raw, "death_eids": int(n_eid),
          "death_dup_eids": int(n_raw - n_eid),
          "death_date_min": str(dd.min().date()), "death_date_max": str(dd.max().date())}
    cfg_cens = pd.Timestamp(tl["death_censor"])
    if dd.max() != cfg_cens:
        warn(f"last death date in the data {dd.max().date()} differs from "
             f"timeline.death_censor {cfg_cens.date()}; update the config")
    df["death_date"] = dd.reindex(df.index)

    lost = pd.Series(pd.NaT, index=df.index)
    lc = tl.get("lost_col")
    f_raw = data_file(p, cfg, "cov_raw", required=False)
    if f_raw is not None and lc in pd.read_csv(f_raw, nrows=0).columns:
        lv = pd.read_csv(f_raw, usecols=["eid", lc]).set_index("eid")[lc]
        lost = pd.to_datetime(lv, errors="coerce").reindex(df.index)
        ok(f"loss to follow-up dates: {int(lost.notna().sum()):,} participants")
    else:
        warn(f"{lc} (field 191) not available; no loss to follow-up assumed")
    df["lost_date"] = lost
    qc["lost_available"] = bool(lost.notna().any())

    # censoring date = min(administrative, lost to follow-up); death is the event
    cens = pd.Series(cfg_cens, index=df.index)
    cens = cens.where(lost.isna() | (lost > cens), lost)
    df["censor_date"] = cens
    df["died"] = ((df["death_date"].notna()) & (df["death_date"] <= cens)).astype("int8")
    df["fu_years"] = (cens - df["tz_date"]).dt.days / 365.25
    qc["died_n"] = int(df["died"].sum())
    qc["fu_years_median"] = float(df["fu_years"].median())
    return df, qc


def stage_timeline(p, cfg, cup) -> pd.DataFrame:
    h2("Career types and covariates")
    df = build_person(p, cfg, cup)
    h2("Death, loss to follow-up, censoring")
    df, qc = attach_followup(df, p, cfg)

    h2("Summary")
    print(f"  n = {len(df):,}")
    for sex, g in df.groupby("sex"):
        vc = g["type"].value_counts()
        print(f"  {sex}: n={len(g):,}  K={g['type'].nunique()}  "
              f"largest type {vc.iloc[0]:,}  smallest type {vc.iloc[-1]:,}")
        print(f"        age at time zero median {g['tz_age'].median():.1f}  "
              f"deaths {int(g['died'].sum()):,}  "
              f"potential follow-up median {g['fu_years'].median():.1f} y")
    vc = df["tz_year"].value_counts().sort_index()
    print("  time-zero year: " + ", ".join(f"{int(k)}: {v:,}" for k, v in vc.items()))
    print("  covariate missingness:")
    for c in ("ethnic_group", "edu_highest", "centre"):
        if c in df.columns:
            print(f"    {c:<16} {df[c].isna().mean():>6.2%}")

    # Rows sorted by eid: the disease matrices use this row order.
    df = df.sort_index()
    f = p["derived"] / "op_person.parquet"
    df.reset_index(names="eid").to_parquet(f, index=False)
    ok(f"{f.name}  {len(df):,} x {df.shape[1]}")
    qc["n"] = int(len(df))
    qc["tz_year"] = {int(k): int(v) for k, v in vc.items()}
    write_json(p["qc"] / "op_timeline.json", qc)
    return df


# -- Disease matrices ----------------------------------------------------------
class _Acc:
    """(row, column) coordinate accumulator -> CSR boolean matrix."""

    def __init__(self, n_row, n_col):
        self.n_row, self.n_col = n_row, n_col
        self.r, self.c = [], []

    def add(self, rows, col):
        if rows.size:
            self.r.append(rows.astype("int32"))
            self.c.append(np.full(rows.size, col, dtype="int32"))

    def add_pairs(self, rows, cols):
        if len(rows):
            self.r.append(np.asarray(rows, dtype="int32"))
            self.c.append(np.asarray(cols, dtype="int32"))

    def csr(self):
        from scipy.sparse import csr_matrix
        if not self.r:
            return csr_matrix((self.n_row, self.n_col), dtype=bool)
        r = np.concatenate(self.r)
        c = np.concatenate(self.c)
        M = csr_matrix((np.ones(r.size, dtype=bool), (r, c)),
                       shape=(self.n_row, self.n_col))
        M.sum_duplicates()
        return M


def read_cancer(p, cfg, person: pd.DataFrame) -> pd.DataFrame:
    """Cancer registry -> long table (row, code, date, src), earliest date per person and code.

    - Instances are not in date order, so the earliest date is taken over all
      instances.
    - ICD-10 codes are four characters without a dot; the first three are used.
    - Only malignant behaviour (ICD-O-3 behaviour 3 or 6) is kept; in situ,
      benign and uncertain tumours are registered inconsistently. Where
      behaviour is missing, C codes are kept and D codes dropped.
    - ICD-9-only records (registrations before ICD-10) have no usable site; if
      the ICD-9 code is malignant (140-208) they get a pseudo-code that counts
      towards chapter II only.
    """
    cc = cfg["cancer"]
    f = data_file(p, cfg, "cancer")
    have = set(pd.read_csv(f, nrows=0).columns)
    slots = [i for i in range(cc["n_slots"])
             if f"{cc['date']}__{i}_0" in have and f"{cc['icd10']}__{i}_0" in have]
    use = ["eid"] + [f"{k}__{i}_0" for i in slots
                     for k in (cc["date"], cc["icd10"], cc["icd9"], cc["behaviour"])
                     if f"{k}__{i}_0" in have]
    raw = pd.read_csv(f, usecols=use, dtype=str)
    keep_b = {str(int(x)) for x in cc["behaviour_keep"]}
    lo, hi = cc["icd9_malignant"]
    pseudo = cc["icd9_pseudo_code"]
    row_of = pd.Series(np.arange(len(person), dtype="int32"),
                       index=person.index.to_numpy("int64"))
    rows = row_of.reindex(pd.to_numeric(raw["eid"], errors="coerce")).to_numpy("float64")
    inb = np.isfinite(rows)
    parts, qc = [], {"slots": len(slots), "records": 0, "kept_icd10": 0,
                     "kept_icd9": 0, "drop_behaviour": 0, "drop_no_code": 0,
                     "drop_icd9_nonmalignant": 0, "beh_missing_c_kept": 0,
                     "beh_missing_d_dropped": 0, "not_in_cohort": 0}
    for i in slots:
        d = pd.to_datetime(raw.get(f"{cc['date']}__{i}_0"), errors="coerce")
        c10 = raw.get(f"{cc['icd10']}__{i}_0")
        c9 = raw.get(f"{cc['icd9']}__{i}_0")
        beh = raw.get(f"{cc['behaviour']}__{i}_0")
        rec = d.notna().to_numpy()
        qc["records"] += int(rec.sum())
        qc["not_in_cohort"] += int((rec & ~inb).sum())
        m = rec & inb
        if not m.any():
            continue
        s10 = c10[m].astype("object")
        s9 = (c9[m].astype("object") if c9 is not None
              else pd.Series(index=s10.index, dtype="object"))
        sb = (beh[m].astype("object").str.strip() if beh is not None
              else pd.Series(index=s10.index, dtype="object"))
        r, dt = rows[m].astype("int32"), d[m].to_numpy("datetime64[ns]")
        has10 = s10.notna().to_numpy()
        is_c = has10 & s10.fillna("").str.startswith("C").to_numpy()
        beh_ok = sb.isin(keep_b).to_numpy()
        beh_na = sb.isna().to_numpy() | (sb.fillna("") == "").to_numpy()
        take10 = has10 & (beh_ok | (beh_na & is_c
                                    if cc["keep_c_when_behaviour_missing"]
                                    else beh_na & False))
        qc["drop_behaviour"] += int((has10 & ~take10 & ~beh_na).sum())
        qc["beh_missing_c_kept"] += int((has10 & beh_na & is_c).sum())
        qc["beh_missing_d_dropped"] += int((has10 & beh_na & ~is_c).sum())
        if take10.any():
            parts.append(pd.DataFrame({
                "row": r[take10], "code": s10[take10].str[:3].to_numpy(),
                "date": dt[take10], "src": "cancer"}))
            qc["kept_icd10"] += int(take10.sum())
        n9 = pd.to_numeric(s9.str[:3], errors="coerce").to_numpy()
        only9 = ~has10 & np.isfinite(n9)
        mal9 = only9 & (n9 >= lo) & (n9 <= hi)
        qc["drop_icd9_nonmalignant"] += int((only9 & ~mal9).sum())
        qc["drop_no_code"] += int((~has10 & ~np.isfinite(n9)).sum())
        if mal9.any():
            parts.append(pd.DataFrame({"row": r[mal9], "code": pseudo,
                                       "date": dt[mal9], "src": "cancer_icd9"}))
            qc["kept_icd9"] += int(mal9.sum())
    if not parts:
        fail("no cancer registry record parsed; check the cancer column names in the config")
        sys.exit(1)
    long = pd.concat(parts, ignore_index=True)
    long = (long.sort_values("date").drop_duplicates(["row", "code"], keep="first")
            .reset_index(drop=True))
    qc["persons"] = int(long["row"].nunique())
    qc["person_code_pairs"] = int(len(long))
    qc["date_min"] = str(long["date"].min().date())
    qc["date_max"] = str(long["date"].max().date())
    qc["n_codes"] = int(long["code"].nunique())
    # Malignant D codes (e.g. D46, D47 registered with behaviour 3) stay in chapter II.
    nonc = sorted(x for x in long["code"].unique()
                  if not x.startswith("C") and x != pseudo)
    qc["non_c_codes_kept"] = nonc
    print(f"  {qc['slots']} instances / {qc['records']:,} registrations -> "
          f"kept ICD-10 {qc['kept_icd10']:,} + ICD-9-only {qc['kept_icd9']:,}; "
          f"dropped as non-malignant {qc['drop_behaviour']:,}")
    print(f"    {qc['persons']:,} cohort members x {qc['n_codes']} codes, "
          f"dates {qc['date_min']} to {qc['date_max']}")
    return long, qc


def stage_disease(p, cfg, person: pd.DataFrame) -> dict:
    from scipy.sparse import save_npz
    fo = cfg["first_occ"]
    tl = cfg["timeline"]
    src = data_file(p, cfg, "first_occ")

    head = pd.read_csv(src, nrows=0)
    dcol, scol = parse_code_columns(head.columns, cfg)
    fo_codes = sorted(dcol)
    print(f"  first occurrences: {len(head.columns):,} columns -> {len(fo_codes):,} codes")
    if not fo_codes:
        fail("no code parsed; check first_occ.date_re")
        sys.exit(1)
    if any(c.startswith("C") for c in fo_codes):
        warn("first occurrences contain C codes; they would double count with the "
             "cancer registry")

    h2("Cancer registry")
    can, can_qc = read_cancer(p, cfg, person)
    codes = sorted(set(fo_codes) | set(can["code"].unique()))
    src_of = {c: "first_occ" for c in fo_codes}
    for c, g in can.groupby("code"):
        src_of[c] = "first_occ+cancer" if c in src_of else g["src"].iloc[0]
    dup = [c for c in codes if src_of[c] == "first_occ+cancer"]
    if dup:
        warn(f"{len(dup)} codes appear in both sources ({dup[:6]}); "
             "the earliest date of the two is used")

    ch = chapters_of(codes, cfg)
    ch["source"] = ch["code"].map(src_of)
    nch = ch.loc[ch["chapter"] != "?", "chapter"].nunique()
    print(f"  {len(codes):,} codes in {nch} ICD-10 chapters")

    cix = {c: i for i, c in enumerate(codes)}
    eids = person.index.to_numpy("int64")
    n = len(eids)
    row_of = pd.Series(np.arange(n, dtype="int32"), index=eids)
    tz = person["tz_date"].to_numpy("datetime64[ns]")

    h2("End of follow-up for incident diagnoses")
    # One common window for all codes: the earliest of the cancer-registry and
    # death-register end dates, so that no column has a shorter window.
    cancer_max = pd.Timestamp(can_qc["date_max"])
    cancer_cens = pd.Timestamp(tl["cancer_censor"] or cancer_max)
    death_cens = pd.Timestamp(tl["death_censor"])
    adm = min(cancer_cens, death_cens)
    print(f"  common window end = min(cancer registry {cancer_cens.date()}, "
          f"death register {death_cens.date()}) = {adm.date()}")
    gap = (death_cens - adm).days / 365.25
    if gap > tl["max_window_gap_years"]:
        warn(f"common window ends {gap:.1f} years before the death register")
    # incident cap = min(window end, loss to follow-up, death)
    cap = np.minimum.reduce([
        np.full(n, np.datetime64(adm), dtype="datetime64[ns]"),
        person["lost_date"].fillna(pd.Timestamp(FAR)).to_numpy("datetime64[ns]"),
        person["death_date"].fillna(pd.Timestamp(FAR)).to_numpy("datetime64[ns]")])
    sp_prev = list(fo["special_prevalent"])
    sp_und = list(fo["special_undated"])

    keys = ("prevalent", "incident")
    acc = {k: _Acc(n, len(codes)) for k in keys}
    qc = {"n_codes": len(codes), "n_chapters": int(nch), "recorded": 0,
          "special_prevalent": 0, "special_undated": 0, "after_cap": 0,
          "unparsed_date": 0, "cancer": can_qc,
          "incident_window_end": str(adm.date())}

    h2("Cancer registry -> matrices")
    crow = can["row"].to_numpy("int32")
    ccol = can["code"].map(cix).to_numpy("int32")
    cdate = can["date"].to_numpy("datetime64[ns]")
    c_prev = cdate <= tz[crow]
    c_inc = (cdate > tz[crow]) & (cdate <= cap[crow])
    acc["prevalent"].add_pairs(crow[c_prev], ccol[c_prev])
    acc["incident"].add_pairs(crow[c_inc], ccol[c_inc])
    qc["cancer_prevalent"] = int(c_prev.sum())
    qc["cancer_incident"] = int(c_inc.sum())
    qc["cancer_after_cap"] = int(((cdate > tz[crow]) & (cdate > cap[crow])).sum())
    print(f"  prevalent {qc['cancer_prevalent']:,} / incident {qc['cancer_incident']:,} / "
          f"after window {qc['cancer_after_cap']:,} person-code pairs")

    h2("First occurrences -> matrices")
    # Special placeholder dates (UKB coding 819): 1900-01-01, 1901-01-01,
    # 1902-02-02 and 1903-03-03 all precede time zero and count as prevalent;
    # 2037-07-07 (future date) is unusable and counts in neither matrix.
    dx_max = np.datetime64("1900-01-01")
    per_pass = int(fo["codes_per_pass"])
    n_pass = (len(fo_codes) + per_pass - 1) // per_pass
    for ip, b0 in enumerate(range(0, len(fo_codes), per_pass), start=1):
        blk = fo_codes[b0:b0 + per_pass]
        use = ["eid"] + [dcol[c] for c in blk]
        # read in column blocks to bound memory
        for chunk in pd.read_csv(src, usecols=use, dtype=str, chunksize=100_000):
            ids = pd.to_numeric(chunk["eid"], errors="coerce")
            rows = row_of.reindex(ids).to_numpy(dtype="float64")
            m = np.isfinite(rows)
            if not m.any():
                continue
            sub = chunk.loc[m]
            r = rows[m].astype("int32")
            tzs, caps = tz[r], cap[r]
            for code in blk:
                j = cix[code]
                sv = sub[dcol[code]].to_numpy()
                rec = pd.notna(sv)
                if not rec.any():
                    continue
                und = np.isin(sv, sp_und)
                spv = np.isin(sv, sp_prev)
                usable = rec & ~und & ~spv
                d = pd.to_datetime(pd.Series(np.where(usable, sv, None)),
                                   format="%Y-%m-%d", errors="coerce"
                                   ).to_numpy("datetime64[ns]")
                dated = usable & ~pd.isna(d)
                if dated.any():
                    dx_max = max(dx_max, d[dated].max())
                prev = spv | (dated & (d <= tzs))
                inc = dated & (d > tzs) & (d <= caps)
                qc["recorded"] += int(rec.sum())
                qc["special_prevalent"] += int(spv.sum())
                qc["special_undated"] += int(und.sum())
                qc["unparsed_date"] += int((usable & ~dated).sum())
                qc["after_cap"] += int((dated & (d > tzs) & (d > caps)).sum())
                acc["prevalent"].add(r[prev], j)
                acc["incident"].add(r[inc], j)
        print(f"    pass {ip}/{n_pass}: codes {blk[0]}-{blk[-1]}")

    dxm = pd.Timestamp(dx_max)
    print(f"  last first-occurrence date {dxm.date()}; cancer registry {cancer_max.date()}; "
          f"death register {death_cens.date()}; incident window end {adm.date()}")
    if dxm < adm:
        fail(f"first-occurrence data end {dxm.date()} before the common window end "
             f"{adm.date()}; set timeline.cancer_censor/death_censor accordingly")
        sys.exit(1)
    qc["dx_censor_measured"] = str(dxm.date())
    qc["cancer_censor_measured"] = str(cancer_max.date())
    qc["death_censor_config"] = tl["death_censor"]
    print(f"  {qc['recorded']:,} records: special dates before time zero "
          f"{qc['special_prevalent']:,}, future placeholder {qc['special_undated']:,}, "
          f"unparsable {qc['unparsed_date']:,}, after end of follow-up {qc['after_cap']:,}")

    d = p["derived"] / "op_disease"
    d.mkdir(parents=True, exist_ok=True)
    nnz = {}
    for k in keys:
        M = acc[k].csr()
        nnz[k] = M.nnz
        save_npz(d / f"{k}.npz", M)
        print(f"    {k:<10} nnz {M.nnz:>10,}  per person {M.nnz / n:>5.1f}")
    if nnz["prevalent"] and not nnz["incident"]:
        fail("no incident diagnosis at all; check the follow-up window against time zero")
        sys.exit(1)
    meta = {"codes": codes, "eid_order": "row order of op_person.parquet (ascending eid)",
            "n": int(n), "dx_censor": str(dxm.date()),
            "cancer_censor": str(cancer_max.date()),
            "incident_window_end": str(adm.date()),
            "matrices": list(keys), **{k: v for k, v in qc.items() if k != "cancer"}}
    write_json(d / "meta.json", meta)
    np.save(d / "eid.npy", eids)
    ch.to_csv(d / "code_chapters.csv", index=False)
    write_json(p["qc"] / "op_disease_qc.json", qc)
    ok(f"op_disease/  {len(codes):,} codes x {n:,} participants")
    return meta


# -- Code catalogue and feasibility -------------------------------------------
def load_matrices(p, person, keys=("prevalent", "incident")):
    """Load the sparse disease matrices and check that their row order matches person."""
    import json
    from scipy.sparse import load_npz
    d = p["derived"] / "op_disease"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    M = {k: load_npz(d / f"{k}.npz") for k in keys}
    eids = np.load(d / "eid.npy")
    if len(eids) != len(person) or not np.array_equal(eids,
                                                     person.index.to_numpy("int64")):
        fail("eid order of op_disease differs from op_person.parquet; "
             "rerun o1 --stage disease")
        sys.exit(1)
    for k, m in M.items():
        if m.shape[0] != len(person):
            fail(f"{k}.npz has {m.shape[0]:,} rows, expected {len(person):,}")
            sys.exit(1)
    return M, meta, list(meta["codes"])


def group_counts(M, rows, labels) -> tuple[np.ndarray, np.ndarray]:
    """(type x code) case counts and type sizes; labels are 0..K-1 on `rows`."""
    from scipy.sparse import csr_matrix
    K = int(labels.max()) + 1
    G = csr_matrix((np.ones(len(rows), dtype="int32"),
                    (labels, np.arange(len(rows)))), shape=(K, len(rows)))
    cnt = np.asarray((G @ M[rows].astype("int32")).todense())
    return cnt, np.asarray(G.sum(1)).ravel()


def stage_feasibility(p, cfg, cup, person: pd.DataFrame) -> None:
    thr = cfg["op"]["min_prevalence"]
    mcc = cfg["op"]["min_cell_cases"]
    M, meta, codes = load_matrices(p, person)
    ch = pd.read_csv(p["derived"] / "op_disease" / "code_chapters.csv")
    prev, inc = M["prevalent"], M["incident"]
    ever = (prev + inc).astype(bool)

    cat, feas = [], []
    for sex in sexes(cup):
        sel_rows = np.flatnonzero((person["sex"] == sex).to_numpy())
        ns = len(sel_rows)
        n_ever = np.asarray(ever[sel_rows].sum(0)).ravel()
        n_prev = np.asarray(prev[sel_rows].sum(0)).ravel()
        n_inc = np.asarray(inc[sel_rows].sum(0)).ravel()
        c = ch.copy()
        c["sex"] = sex
        c["n_sex"] = ns
        c["n_ever"] = n_ever
        c["prev_ever"] = n_ever / ns
        c["n_prevalent"] = n_prev
        c["n_incident"] = n_inc
        c["n_at_risk_incident"] = ns - n_prev
        # code-level selection: sex-specific lifetime prevalence >= min_prevalence
        c["selected"] = c["prev_ever"] >= thr
        # the ICD-9-only pseudo-code has no site and is never a code-level column
        pseudo = cfg["cancer"]["icd9_pseudo_code"]
        c.loc[c["code"] == pseudo, "selected"] = False
        cat.append(c)

        codes_sel = np.flatnonzero(c["selected"].to_numpy())
        raw = person.loc[person["sex"] == sex, "type"].to_numpy()
        uniq = np.unique(raw)
        lab = np.searchsorted(uniq, raw)
        cnt_p, n_type = group_counts(prev, sel_rows, lab)
        cnt_i, _ = group_counts(inc, sel_rows, lab)
        died = person.loc[person["sex"] == sex, "died"].to_numpy()
        for k, t in enumerate(uniq):
            feas.append({
                "sex": sex, "type": int(t), "n": int(n_type[k]),
                "pct": 100 * n_type[k] / ns,
                "deaths": int(died[lab == k].sum()),
                "n_codes_selected": int(len(codes_sel)),
                "cells_colourable_prevalent":
                    int((cnt_p[k, codes_sel] >= mcc).sum()),
                "cells_colourable_incident":
                    int((cnt_i[k, codes_sel] >= mcc).sum()),
                "median_prevalent_codes":
                    float(np.median(np.asarray(
                        prev[sel_rows[lab == k]].sum(1)).ravel()))})
        print(f"  {sex}: {int(c['selected'].sum())} codes with lifetime prevalence "
              f">= {thr:.0%}")

    cat = pd.concat(cat, ignore_index=True)
    cat.to_csv(p["tables"] / "OP0_code_catalog.csv", index=False)
    fe = pd.DataFrame(feas)
    fe.to_csv(p["tables"] / "OP0_feasibility.csv", index=False)
    ok("OP0_code_catalog.csv / OP0_feasibility.csv")


def main():
    ap = argparse.ArgumentParser(description="Assemble time zero, follow-up and "
                                             "disease matrices (no associations)")
    ap.add_argument("--stage", default="all",
                    choices=["timeline", "disease", "feasibility", "all"])
    args = ap.parse_args()
    cfg, cup = load_cfg_out(), load_cfg_up()
    p = paths_out(cfg)

    with tee(p, f"o1_assemble_{args.stage}"):
        h1(f"o1 assemble: {args.stage}")
        pf = p["derived"] / "op_person.parquet"
        person = None
        if args.stage in ("timeline", "all"):
            h1("Timeline (time zero = questionnaire completion)")
            person = stage_timeline(p, cfg, cup)
        if person is None:
            if not pf.exists():
                fail(f"{pf.name} missing; run --stage timeline first")
                sys.exit(1)
            person = pd.read_parquet(pf).set_index("eid")
        if args.stage in ("disease", "all"):
            h1("Disease matrices")
            stage_disease(p, cfg, person)
        if args.stage in ("feasibility", "all"):
            h1("Code catalogue and feasibility")
            if not (p["derived"] / "op_disease" / "meta.json").exists():
                fail("op_disease/meta.json missing; run --stage disease first")
                sys.exit(1)
            stage_feasibility(p, cfg, cup, person)
        h2("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
