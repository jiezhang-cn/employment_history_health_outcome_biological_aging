#!/usr/bin/env python3
"""b1: hospital diagnoses -> (eid, phecode, age at diagnosis) records for topic projection.

Lifetime hospital diagnoses (HES, all diagnosis positions) of all UK Biobank
participants are mapped from ICD-10 to phecodes with the AgeTopicModels map
(exported by R/b0_export_phecode.R) and reduced to the earliest age per person
and phecode. These records are projected onto the published comorbidity
topics by R/b2_atm_fit.R.

hesin_diag has no dates; the diagnosis date comes from the matching hesin
episode (epistart, else admidate), joined on (eid, ins_index). Age uses the
birth date of every UK Biobank participant, not only the work-history cohort.

Inputs: data/ files.hesin, files.hesin_diag, files.cov_baseline (eid, birth_date);
        references/phecode_icd10.csv (from b0).
Outputs (results/outcome/):
  derived/ob_records_raw.parquet      all mapped records
  derived/ob_records.parquet/.csv.gz  records restricted to the vocabulary (input to b2)
  derived/ob_code_to_icd3.csv         phecode -> most frequent 3-character ICD-10 code
                                      (used for chapter assignment)
  derived/ob_vocab.csv, tables/OB0_code_catalog.csv
  tables/OB0_feasibility.csv          per sex and type: participants with >= 1 / >= 2 diagnoses
  qc/ob_records_qc.json

  python b1_records.py --stage all
  python b1_records.py --stage vocab    # rebuild vocabulary from ob_records_raw.parquet
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common_out import (chapters_of_vocab, data_file, fail, h1, h2, load_cfg_out,
                        load_cfg_up, ok, paths_out, ref_file, sexes, tee, warn,
                        write_json)

DAYS_PER_YEAR = 365.25


# -- Birth dates ---------------------------------------------------------------
def birth_dates(p, cfg) -> pd.Series:
    """Birth dates of all UK Biobank participants, indexed by int64 eid."""
    d = pd.read_csv(data_file(p, cfg, "cov_baseline"), usecols=["eid", "birth_date"])
    d["eid"] = pd.to_numeric(d["eid"], errors="coerce").astype("int64")
    b = pd.to_datetime(d.set_index("eid")["birth_date"], errors="coerce")
    n_bad = int(b.isna().sum())
    if n_bad:
        warn(f"{n_bad:,} participants have no birth date; their diagnoses are dropped")
    ok(f"birth dates for {int(b.notna().sum()):,} participants")
    return b


def letters_ok(codes, spec: str) -> np.ndarray:
    """First letter within spec (e.g. "A-N"), as in Jiang et al. 2023."""
    lo, hi = (spec.split("-") + [spec])[:2]
    c = np.asarray([str(x)[:1] for x in np.asarray(codes)])
    return (c >= lo.strip()) & (c <= hi.strip())


# -- HES -------------------------------------------------------------------------
def read_episodes(p, cfg) -> tuple[pd.DataFrame, dict]:
    """hesin -> one row per episode (eid, ins_index, date).

    Dates are ISO (YYYY-MM-DD) and parsed with an explicit format; epistart is
    used, falling back to admidate.
    """
    hs = cfg["ob"]["hes"]
    fmain = data_file(p, cfg, "hesin")
    have = set(pd.read_csv(fmain, sep="\t", nrows=0).columns)
    miss_date = [c for c in hs["date_cols"] if c not in have]
    if len(miss_date) == len(hs["date_cols"]):
        fail(f"{fmain.name} has none of the date columns {hs['date_cols']}")
        sys.exit(1)
    if miss_date:
        warn(f"{fmain.name} lacks {miss_date}")
    use = [c for c in ["eid", "ins_index"] + list(hs["date_cols"]) if c in have]
    m = pd.read_csv(fmain, sep="\t", usecols=use, low_memory=False)
    qc = {"episodes": int(len(m)), "columns_used": use}
    print(f"  hesin: {len(m):,} episodes")
    dates = []
    for c in hs["date_cols"]:
        if c in m.columns:
            dates.append(pd.to_datetime(m[c], format=hs["date_format"],
                                        errors="coerce"))
            qc[f"{c}_missing"] = int(dates[-1].isna().sum())
    date = dates[0]
    for nxt in dates[1:]:
        date = date.fillna(nxt)
    m = m.assign(date=date).dropna(subset=["date"])
    qc["episodes_with_date"] = int(len(m))
    print(f"    episodes with a date: {len(m):,} "
          f"({len(m) / max(qc['episodes'], 1):.2%})")
    m = m.dropna(subset=["eid", "ins_index"])
    m["eid"] = pd.to_numeric(m["eid"], errors="coerce").astype("int64")
    m["ins_index"] = pd.to_numeric(m["ins_index"], errors="coerce").astype("int32")
    m = m[["eid", "ins_index", "date"]]
    # (eid, ins_index) should be unique; keep the earliest date if not.
    n0 = len(m)
    m = m.sort_values("date").drop_duplicates(["eid", "ins_index"], keep="first")
    qc["duplicate_episode_keys"] = int(n0 - len(m))
    if n0 != len(m):
        warn(f"{n0 - len(m):,} duplicate (eid, ins_index) keys in hesin; earliest date kept")
    return m, qc


def _fold_proxy(acc: dict, code, icd3) -> None:
    """Accumulate {phecode: {icd3: count}}."""
    for c, i in zip(code, icd3):
        acc.setdefault(c, {})
        acc[c][i] = acc[c].get(i, 0) + 1


def scan_hes(p, cfg, birth: pd.Series) -> tuple[pd.DataFrame, dict]:
    """hesin_diag + hesin -> (eid, code, age), earliest age per person and phecode.

    The diagnosis table is read in chunks and folded to (eid, code) minima per
    chunk to bound memory.
    """
    ob = cfg["ob"]
    hs = ob["hes"]
    fdiag = data_file(p, cfg, "hesin_diag")
    epi, qc = read_episodes(p, cfg)
    epi = epi.set_index(["eid", "ins_index"])["date"]
    pmap, ppar = phecode_map(cfg)
    lo_age, hi_age = ob["age_range"]
    cols = ["eid", "ins_index", "diag_icd9", "diag_icd10"]
    qc.update({"diag_rows": 0, "icd9_only_rows": 0, "no_icd10": 0,
               "merge_attempted": 0, "unmatched_episode": 0, "no_birth": 0,
               "age_out_of_range": 0, "unmapped_code": 0, "dropped_by_letter": 0})
    parts, proxy, unmapped_hist = [], {}, {}
    for ic, d in enumerate(pd.read_csv(fdiag, sep="\t", usecols=lambda c: c in cols,
                                       dtype={"diag_icd9": str, "diag_icd10": str},
                                       chunksize=int(hs["chunksize"]),
                                       low_memory=False), start=1):
        qc["diag_rows"] += len(d)
        if "diag_icd10" not in d.columns:
            fail(f"{fdiag.name} has no diag_icd10 column")
            sys.exit(1)
        if "diag_icd9" in d.columns:
            qc["icd9_only_rows"] += int((d["diag_icd10"].isna()
                                         & d["diag_icd9"].notna()).sum())
        n0 = len(d)
        d = d.dropna(subset=["diag_icd10"])
        qc["no_icd10"] += n0 - len(d)
        if not len(d):
            continue
        d = d.assign(
            eid=pd.to_numeric(d["eid"], errors="coerce").astype("int64"),
            ins_index=pd.to_numeric(d["ins_index"], errors="coerce").astype("int32"))
        dt = epi.reindex(pd.MultiIndex.from_arrays(
            [d["eid"].to_numpy(), d["ins_index"].to_numpy()])).to_numpy(
            "datetime64[ns]")
        qc["merge_attempted"] += len(d)
        qc["unmatched_episode"] += int(pd.isna(dt).sum())
        raw = d["diag_icd10"].astype(str).str.replace(".", "", regex=False).str.strip()
        ok_l = letters_ok(raw, ob["code_letters"])
        qc["dropped_by_letter"] += int((~ok_l).sum())
        code = raw.map(pmap)
        # Codes absent from the map fall back to the longest parent code that
        # is present (4, then 3 characters); phecodes are coarser than ICD-10,
        # so the parent's phecode is the correct group.
        for plen in (4, 3):
            miss = code.isna()
            if not miss.any():
                break
            fb = raw[miss].str[:plen].map(ppar)
            qc[f"mapped_via_parent{plen}"] = qc.get(
                f"mapped_via_parent{plen}", 0) + int(fb.notna().sum())
            code = code.where(~miss, fb)
        unmapped = code.isna().to_numpy() & ok_l
        qc["unmapped_code"] += int(unmapped.sum())
        if unmapped.any():
            for c0, n0_ in raw[unmapped].value_counts().items():
                unmapped_hist[c0] = unmapped_hist.get(c0, 0) + int(n0_)
        code = code.to_numpy()
        good_p = ok_l & pd.notna(code)
        _fold_proxy(proxy, code[good_p], raw.str[:3].to_numpy()[good_p])
        b = birth.reindex(d["eid"]).to_numpy("datetime64[ns]")
        qc["no_birth"] += int(pd.isna(b).sum())
        age = ((dt - b) / np.timedelta64(1, "D") / DAYS_PER_YEAR).astype("float64")
        keep = (ok_l & pd.notna(code) & np.isfinite(age)
                & (age >= lo_age) & (age <= hi_age))
        qc["age_out_of_range"] += int((np.isfinite(age) & ~(
            (age >= lo_age) & (age <= hi_age))).sum())
        if not keep.any():
            continue
        parts.append(pd.DataFrame({"eid": d["eid"].to_numpy()[keep],
                                   "code": code[keep],
                                   "age": age[keep]})
                     .groupby(["eid", "code"], sort=False)["age"].min().reset_index())
        print(f"    chunk {ic}: {qc['diag_rows']:,} diagnosis rows -> "
              f"{sum(len(x) for x in parts):,} person-code pairs")
    if not parts:
        fail("no HES diagnosis kept; see the QC counts above")
        sys.exit(1)
    rate = 1.0 - qc["unmatched_episode"] / max(qc["merge_attempted"], 1)
    print(f"  diagnosis-episode match rate {rate:.2%}")
    # A low match rate means hesin and hesin_diag come from different releases.
    if rate < float(hs["min_merge_rate"]):
        fail(f"match rate {rate:.2%} < {hs['min_merge_rate']:.0%}; "
             "hesin and hesin_diag do not match")
        sys.exit(1)
    long = (pd.concat(parts, ignore_index=True)
            .groupby(["eid", "code"], sort=False)["age"].min().reset_index())
    long["age"] = long["age"].astype("float32")
    long["src"] = 40                     # hospital (first-occurrence source coding)
    qc["kept"] = int(len(long))
    qc["merge_rate"] = float(rate)
    pr = pd.DataFrame([{"code": c, "icd3": max(d, key=d.get),
                        "n": max(d.values())} for c, d in proxy.items()])
    pr[["code", "icd3"]].to_csv(p["derived"] / "ob_code_to_icd3.csv", index=False)
    ok(f"ob_code_to_icd3.csv  {len(pr):,} phecodes")
    print(f"  ICD-9-only rows {qc['icd9_only_rows']:,} (not used); "
          f"outside A-N {qc['dropped_by_letter']:,}; age out of range "
          f"{qc['age_out_of_range']:,}; unmapped {qc['unmapped_code']:,}")
    top = sorted(unmapped_hist.items(), key=lambda kv: -kv[1])[:25]
    qc["unmapped_top"] = {k: v for k, v in top}
    ur = qc["unmapped_code"] / max(qc["merge_attempted"], 1)
    if ur > 0.05:
        warn(f"{ur:.1%} of A-N diagnoses have no phecode; most frequent: "
             + ", ".join(f"{k} x{v:,}" for k, v in top[:8]))
    return long, qc


def phecode_map(cfg: dict) -> tuple[dict, dict]:
    """{ICD-10 (no dot) -> phecode} from the AgeTopicModels map, plus the parent table.

    Where an ICD-10 code maps to several phecodes, the phecode covering the
    most ICD-10 codes is taken (rule of the AgeTopicModels documentation).
    """
    pc = cfg["ob"]["phecode"]
    f = ref_file(cfg, pc["map_file"], required=False)
    if f is None:
        fail(f"{pc['map_file']} missing; run Rscript outcome/R/b0_export_phecode.R")
        sys.exit(1)
    m = pd.read_csv(f, dtype=str)
    icd_col = next((c for c in m.columns if "icd" in c.lower()), None)
    ph_col = next((c for c in m.columns if "phecode" in c.lower()), None)
    if icd_col is None or ph_col is None:
        fail(f"{f.name}: ICD-10 and phecode columns not found ({list(m.columns)})")
        sys.exit(1)
    m = m[[icd_col, ph_col]].dropna()
    m.columns = ["icd", "phecode"]
    m["icd"] = m["icd"].str.replace(".", "", regex=False).str.strip().str.upper()
    m["phecode"] = m["phecode"].str.strip()
    cover = m["phecode"].value_counts()
    m = (m.assign(cover=m["phecode"].map(cover))
         .sort_values(["icd", "cover", "phecode"], ascending=[True, False, True])
         .drop_duplicates("icd", keep="first"))
    out = dict(zip(m["icd"], m["phecode"]))
    # Parent fallback table: genuine 3- and 4-character entries of the map only.
    par = {k: v for k, v in out.items() if len(k) in (3, 4)}
    print(f"  phecode map: {len(out):,} ICD-10 codes -> {m['phecode'].nunique():,} "
          f"phecodes; {len(par):,} parent entries")
    return out, par


# -- Vocabulary ------------------------------------------------------------------
def build_vocab(p, long: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep codes with at least ob.min_occurrence records in UK Biobank."""
    ob = cfg["ob"]
    cnt = long["code"].value_counts()
    v = (chapters_of_vocab(sorted(cnt.index), cfg, p)
         .assign(n_occurrence=lambda d: d["code"].map(cnt).astype("int64")))
    v["selected"] = v["n_occurrence"] >= int(ob["min_occurrence"])
    print(f"  vocabulary: {len(v):,} codes observed, {int(v['selected'].sum())} with "
          f">= {ob['min_occurrence']} records")
    v = v.sort_values(["chapter_index", "code"]).reset_index(drop=True)
    return v


def stage_records(p, cfg) -> pd.DataFrame:
    ob = cfg["ob"]
    h2("Birth dates (all UK Biobank)")
    birth = birth_dates(p, cfg)
    h2("Hospital diagnoses")
    long, qc = scan_hes(p, cfg, birth)
    long = long.sort_values(["eid", "age"]).reset_index(drop=True)
    n_person = int(long["eid"].nunique())
    print(f"  {len(long):,} person-code records / {n_person:,} participants "
          f"({len(long) / max(n_person, 1):.1f} per participant)")
    f = p["derived"] / "ob_records_raw.parquet"
    long.to_parquet(f, index=False)
    ok(f"{f.name}  {len(long):,} rows (eid / code / age / src)")
    qc["n_person"] = n_person
    qc["n_records"] = int(len(long))
    qc["age_range"] = list(ob["age_range"])
    write_json(p["qc"] / "ob_records_qc.json", qc)
    return long


def stage_vocab(p, cfg, cup, long: pd.DataFrame) -> None:
    ob = cfg["ob"]
    h2("Vocabulary")
    v = build_vocab(p, long, cfg)
    sel = set(v.loc[v["selected"], "code"])
    rec = long[long["code"].isin(sel)].reset_index(drop=True)
    print(f"  records in vocabulary: {len(rec):,} of {len(long):,}")
    v.to_csv(p["tables"] / "OB0_code_catalog.csv", index=False)
    v[v["selected"]].reset_index(drop=True).to_csv(
        p["derived"] / "ob_vocab.csv", index=False)
    f = p["derived"] / "ob_records.parquet"
    rec.to_parquet(f, index=False)
    ok(f"{f.name}  {len(rec):,} rows (input to b2)")
    # csv.gz copy for R, so that b2 does not depend on the arrow package
    gz = p["derived"] / "ob_records.csv.gz"
    rec[["eid", "code", "age"]].to_csv(gz, index=False, compression="gzip")
    ok(gz.name)

    ndx = rec.groupby("eid").size()
    fit = ndx[ndx >= int(ob["min_dx_fit"])]
    print(f"  {len(ndx):,} participants with >= 1 diagnosis, {len(fit):,} with "
          f">= {ob['min_dx_fit']}")

    h2("Work-history cohort")
    pf = p["derived"] / "op_person.parquet"
    if not pf.exists():
        warn(f"{pf.name} missing; feasibility table skipped (run o1_assemble.py first)")
        return
    per = pd.read_parquet(pf).set_index("eid")
    per = per.assign(n_dx=ndx.reindex(per.index).fillna(0).astype("int32"))
    mn = int(ob["min_dx_fit"])
    feas = []
    for sex in sexes(cup):
        g = per[per["sex"] == sex]
        for t in sorted(g["type"].unique()):
            gt = g[g["type"] == t]
            feas.append({
                "sex": sex, "type": int(t), "n": len(gt),
                "n_with_dx": int((gt["n_dx"] > 0).sum()),
                "n_share_eligible": int((gt["n_dx"] >= mn).sum()),
                "pct_share_eligible": 100 * float((gt["n_dx"] >= mn).mean()),
                "median_n_dx": float(gt["n_dx"].median()),
                "mean_n_dx": float(gt["n_dx"].mean())})
        print(f"  {sex}: n={len(g):,}, {float((g['n_dx'] >= mn).mean()):.1%} with "
              f">= {mn} diagnoses")
    pd.DataFrame(feas).to_csv(p["tables"] / "OB0_feasibility.csv", index=False)
    ok("OB0_code_catalog.csv / OB0_feasibility.csv")


def main():
    ap = argparse.ArgumentParser(description="Hospital diagnoses -> phecode records")
    ap.add_argument("--stage", default="all", choices=["records", "vocab", "all"])
    args = ap.parse_args()
    cfg, cup = load_cfg_out(), load_cfg_up()
    p = paths_out(cfg)
    raw = p["derived"] / "ob_records_raw.parquet"

    with tee(p, f"b1_records_{args.stage}"):
        h1(f"b1 diagnosis records: {args.stage}")
        long = None
        if args.stage in ("records", "all"):
            long = stage_records(p, cfg)
        if args.stage in ("vocab", "all"):
            if long is None:
                if not raw.exists():
                    fail(f"{raw.name} missing; run --stage records first")
                    return 1
                long = pd.read_parquet(raw)
            stage_vocab(p, cfg, cup, long)
        h2("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
