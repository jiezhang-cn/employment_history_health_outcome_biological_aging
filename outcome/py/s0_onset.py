#!/usr/bin/env python3
"""s0: first-onset dates (long table), the input of the prospective analyses.

o1 stores diagnoses as binary matrices without dates. Incidence (s1),
disease-free life (s2) and the accumulation of diagnoses (s3) need event
times, so this script scans the same sources again for dates:

  * UK Biobank first-occurrence fields (one date column per 3-character
    ICD-10 code; primary care, hospital, death registry and self-report);
  * the cancer registry, the only source of chapter II (neoplasms), parsed
    with `o1_assemble.read_cancer` so both scripts share one rule set.

Inputs:  results/outcome/derived/op_person.parquet (o1), files.first_occ and
         files.cancer under data/
Outputs: results/outcome/derived/op_onset.parquet  (row, code, age, post_tz, src)
             one row per person and code, the earliest record;
             `row` indexes op_person.parquet, `post_tz` = after time zero
         results/outcome/qc/s0_onset_qc.json       aggregate counts

Run:  python outcome/py/s0_onset.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common_out import (fail, h1, h2, load_cfg_out, ok, parse_code_columns,
                        paths_out, tee, warn, write_json)
from o1_assemble import read_cancer
from surv_common import DAYS_PER_YEAR, ONSET, load_person


def scan_first_occ(p, cfg, person, birth, tz):
    """First-occurrence date columns -> (row, code, age, post_tz, src).

    Special dates 1900/1901/1902/1903-xx-xx mean "before birth / undated but
    prevalent" and 2037-07-07 is a future placeholder; neither yields an age,
    so they are only counted here (prevalence is taken from o1's matrices).
    The date format is given explicitly so pandas does not infer it per row.
    """
    fo = cfg["first_occ"]
    src = p["data"] / cfg["files"]["first_occ"]
    head = pd.read_csv(src, nrows=0)
    dcol, _ = parse_code_columns(head.columns, cfg)
    codes = sorted(dcol)
    print(f"  {len(head.columns):,} columns -> date columns for {len(codes):,} codes")
    if not codes:
        fail("no code columns parsed: check first_occ.date_re")
        sys.exit(1)
    row_of = pd.Series(np.arange(len(person), dtype="int32"),
                       index=person.index.to_numpy("int64"))
    sp_prev = list(fo["special_prevalent"])
    sp_und = list(fo["special_undated"])
    per = int(fo["codes_per_pass"])
    n_pass = (len(codes) + per - 1) // per
    parts = []
    qc = {"recorded": 0, "dated": 0, "special_prevalent": 0,
          "special_undated": 0, "unparsed_date": 0, "negative_age": 0}
    for ip, b0 in enumerate(range(0, len(codes), per), start=1):
        blk = codes[b0:b0 + per]
        use = ["eid"] + [dcol[c] for c in blk]
        for chunk in pd.read_csv(src, usecols=use, dtype=str, chunksize=100_000):
            ids = pd.to_numeric(chunk["eid"], errors="coerce")
            rows = row_of.reindex(ids).to_numpy("float64")
            m = np.isfinite(rows)
            if not m.any():
                continue
            sub = chunk.loc[m]
            r = rows[m].astype("int32")
            for code in blk:
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
                qc["recorded"] += int(rec.sum())
                qc["special_prevalent"] += int(spv.sum())
                qc["special_undated"] += int(und.sum())
                qc["unparsed_date"] += int((usable & ~dated).sum())
                if not dated.any():
                    continue
                rr = r[dated]
                age = ((d[dated] - birth[rr]) / np.timedelta64(1, "D")
                       ) / DAYS_PER_YEAR
                bad = ~np.isfinite(age) | (age < 0)
                qc["negative_age"] += int(bad.sum())
                parts.append(pd.DataFrame({
                    # float64: window-boundary comparisons need sub-day resolution
                    "row": rr[~bad], "code": code,
                    "age": age[~bad].astype("float64"),
                    "post_tz": (d[dated][~bad] > tz[rr[~bad]]), "src": "first_occ"}))
                qc["dated"] += int((~bad).sum())
        print(f"    pass {ip}/{n_pass}: codes {blk[0]}-{blk[-1]}")
    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=["row", "code", "age", "post_tz", "src"])), qc


def add_cancer(p, cfg, person, birth, tz, fo_codes) -> tuple[pd.DataFrame, dict]:
    """Cancer registry -> same format (earliest malignant record per code)."""
    can, qc = read_cancer(p, cfg, person)
    r = can["row"].to_numpy("int32")
    d = can["date"].to_numpy("datetime64[ns]")
    age = ((d - birth[r]) / np.timedelta64(1, "D")) / DAYS_PER_YEAR
    bad = ~np.isfinite(age) | (age < 0)
    if bad.any():
        warn(f"{int(bad.sum()):,} cancer records without computable age: dropped")
    both = sorted(set(can["code"].unique()) & set(fo_codes))
    qc["codes_in_both_sources"] = both
    if both:
        warn(f"{len(both)} codes present in both sources ({both[:6]}): "
             "the earliest date is used")
    return pd.DataFrame({"row": r[~bad], "code": can["code"].to_numpy()[~bad],
                         "age": age[~bad].astype("float64"),
                         "post_tz": (d[~bad] > tz[r[~bad]]), "src": "cancer"}), qc


def build(p, cfg, person) -> pd.DataFrame:
    birth = pd.to_datetime(person["birth_date"]).to_numpy("datetime64[ns]")
    tz = pd.to_datetime(person["tz_date"]).to_numpy("datetime64[ns]")
    h2("first occurrences -> onset dates")
    fo, qfo = scan_first_occ(p, cfg, person, birth, tz)
    h2("cancer registry -> onset dates")
    ca, qca = add_cancer(p, cfg, person, birth, tz,
                         set(fo["code"].unique()) if len(fo) else set())
    long = pd.concat([fo, ca], ignore_index=True)
    n0 = len(long)
    # earliest record per (person, code)
    long = (long.sort_values("age").drop_duplicates(["row", "code"], keep="first")
            .reset_index(drop=True))
    if len(long) != n0:
        print(f"  merged across sources: {n0:,} -> {len(long):,} person-code pairs")
    qc = {"first_occ": qfo, "cancer": qca, "pairs": int(len(long)),
          "persons": int(long["row"].nunique()),
          "codes": int(long["code"].nunique()),
          "post_tz_pairs": int(long["post_tz"].sum()),
          "age_min": float(long["age"].min()) if len(long) else None,
          "age_max": float(long["age"].max()) if len(long) else None}
    print(f"  {qc['pairs']:,} person-code pairs, {qc['persons']:,} persons, "
          f"{qc['codes']:,} codes; {qc['post_tz_pairs']:,} after time zero")
    f = p["derived"] / ONSET
    long.to_parquet(f, index=False)
    ok(f"{f.name}  {len(long):,} rows")
    write_json(p["qc"] / "s0_onset_qc.json", qc)
    return long


def main():
    cfg = load_cfg_out()
    p = paths_out(cfg)
    with tee(p, "s0_onset"):
        h1("s0: first-onset dates")
        person = load_person(p)
        print(f"  n = {len(person):,}")
        build(p, cfg, person)
    return 0


if __name__ == "__main__":
    sys.exit(main())
