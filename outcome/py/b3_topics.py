#!/usr/bin/env python3
"""b3: name the published topics and assign diagnoses to topics.

Run before any association is estimated. For every diagnosis record the topic
responsibility r_nk is proportional to theta_sk * beta_k,j(t_n) (person s,
disease j, age t_n), normalised over topics. From these:
  - each disease's average topic assignment (mean responsibility over its
    diagnoses; Jiang et al. 2023) and the number of diseases assigned mainly
    (> ob.naming.assign_threshold) to each topic;
  - topic names: the acronyms of Jiang et al. 2023 Table 1, matched to the
    loading array by disease names (Hungarian assignment on the overlap
    between each topic's top diseases and the three representative diseases
    in Table 1). If the match fails, names fall back to the ICD-10 chapters
    with the largest summed assignment.

This uses the converged (theta, beta) in mean-field form and is descriptive
only; no test uses it.

Inputs: derived/ob_fit_atm_K10_published/ (b2), derived/ob_records.* and
        derived/ob_code_to_icd3.csv (b1), references/phecode_info.csv (b0).
Outputs: derived/ob_topics.json (names read by b4),
         tables/OB2_topic_profile.csv, tables/OB2_disease_topic_assignment.csv.

  python b3_topics.py [--fit ob_fit_atm_K10_published]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common_out import (chapters_of_vocab, fail, h1, h2, load_cfg_out, ok,
                        paths_out, read_records, ref_file, tee, warn, write_json)

DEFAULT_FIT = "ob_fit_atm_K10_published"


def code_labels(cfg, codes, trunc: int = 20) -> dict:
    """phecode -> phenotype name (from phecode_info.csv); trunc=0 keeps full names."""
    f = ref_file(cfg, "phecode_info.csv", required=False)
    if f is None:
        warn("phecode_info.csv not found; run Rscript outcome/R/b0_export_phecode.R")
        return {str(c): str(c) for c in codes}
    d = pd.read_csv(f, dtype=str)
    kc = next((c for c in d.columns if "phecode" in c.lower()), d.columns[0])
    nc = next((c for c in d.columns if c.lower() in ("phenotype", "description",
                                                     "name")), None)
    if nc is None:
        return {str(c): str(c) for c in codes}
    m = dict(zip(d[kc].astype(str).str.strip(), d[nc].astype(str)))
    return {str(c): (m.get(str(c), str(c))[:trunc] if trunc else
                     m.get(str(c), str(c))) for c in codes}


# The ten topics of Jiang et al. Nat Genet 2023, Table 1: acronym, disease
# systems, number of associated diseases, expected position in the third
# dimension of UKB_HES_10topics (cross-check only), three representative
# diseases. Table 1 is ordered by phecode system, not by array position, so
# names are matched by disease names rather than by order.
PUBLISHED_TOPICS_JIANG2023 = [
    ("NRI", "Neoplasms, respiratory, infectious diseases", 53, 4,
     ["Secondary malignancy of lymph nodes", "Pneumococcal pneumonia",
      "Bacterial infection NOS"]),
    ("CER", "Circulatory, endocrine/metabolic, respiratory", 41, 5,
     ["Type 2 diabetes", "Obesity", "Chronic airway obstruction"]),
    ("SRD", "Sense organs, respiratory, dermatologic", 38, 10,
     ["Cataract", "Septal Deviations/Turbinate Hypertrophy",
      "Benign neoplasm of skin"]),
    ("CVD", "Cardiovascular disease", 27, 7,
     ["Hypercholesterolemia", "Coronary atherosclerosis",
      "Myocardial infarction"]),
    ("UGI", "Upper gastrointestinal disease", 22, 8,
     ["Diaphragmatic hernia",
      "Benign neoplasm of other parts of digestive system",
      "Gastritis and duodenitis"]),
    ("LGI", "Lower gastrointestinal disease", 13, 9,
     ["Irritable bowel syndrome", "Benign neoplasm of colon",
      "Anal and rectal polyp"]),
    ("FGND", "Female genitourinary, neoplasms, digestive", 34, 3,
     ["Uterine leiomyoma", "Malignant neoplasm of female breast",
      "Hypothyroidism NOS"]),
    ("MGND", "Male genitourinary, neoplasms, digestive", 33, 6,
     ["Urinary tract infection", "Cancer of prostate",
      "Other disorders of bladder"]),
    ("MDS", "Musculoskeletal, digestive, symptoms", 29, 1,
     ["Back pain", "Cholelithiasis", "Other disorders of soft tissues"]),
    ("ARP", "Arthropathy-related disease", 26, 2,
     ["Arthropathy NOS", "Rheumatoid arthritis", "Enthesopathy"]),
]


def published_names(p, cfg, fit) -> dict:
    """{topic: (acronym, systems, n_assoc)} matched by disease names, or {} if no match."""
    if fit["K"] != len(PUBLISHED_TOPICS_JIANG2023):
        warn(f"K={fit['K']} differs from 10; published names not used")
        return {}
    full = code_labels(cfg, fit["codes"], trunc=0)
    if all(str(v) == str(k) for k, v in full.items()):
        warn("no phenotype names; published names cannot be matched")
        return {}
    # 25 diseases with the highest age-averaged loading per topic
    mb = fit["beta"].mean(axis=0)                       # (D, K)
    top = {k: {str(full.get(fit["codes"][j], "")).strip().lower()
               for j in np.argsort(-mb[:, k])[:25]} for k in range(fit["K"])}
    M = np.zeros((len(PUBLISHED_TOPICS_JIANG2023), fit["K"]), dtype=int)
    for i, (_, _, _, _, reps) in enumerate(PUBLISHED_TOPICS_JIANG2023):
        for k in range(fit["K"]):
            M[i, k] = sum(r.strip().lower() in top[k] for r in reps)
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-M)
    hit = int(M[ri, ci].sum())
    lo = int(M[ri, ci].min())
    print(f"  match with Table 1 representative diseases: {hit}/30, weakest topic {lo}/3")
    # accept if >= 20/30 matched and every topic matches at least one disease
    if hit < 20 or lo < 1:
        warn("match too weak; falling back to chapter-based names")
        return {}
    out, mismatch = {}, []
    for i, k in zip(ri, ci):
        ab, sysname, n_assoc, expect, _ = PUBLISHED_TOPICS_JIANG2023[i]
        out[k + 1] = (ab, sysname, n_assoc)
        if expect != k + 1:
            mismatch.append(f"{ab}: matched topic {k + 1}, expected {expect}")
    if mismatch:
        warn("matched order differs from the expected array order: " + "; ".join(mismatch))
    print("  topic names: " + " ".join(f"T{k}={v[0]}" for k, v in sorted(out.items())))
    return out


# -- Fit outputs ---------------------------------------------------------------
def load_fit(p, name: str) -> dict:
    d = p["derived"] / name
    if not d.is_dir():
        fail(f"{d} not found; run Rscript outcome/R/b2_atm_fit.R first")
        sys.exit(1)
    # phecodes such as "8.5" or "41" must stay strings to match the records
    g = pd.read_csv(d / "loadings_long.csv.gz", dtype={"code": str})
    g["code"] = g["code"].str.strip()
    meta = dict(zip(*pd.read_csv(d / "fit_meta.csv")[["key", "value"]].to_numpy().T))
    ages = np.sort(g["age"].unique())
    codes = list(dict.fromkeys(g.loc[g["age"] == ages[0], "code"]))
    K = int(g["topic"].max())
    B = (g.pivot_table(index=["topic", "age"], columns="code", values="beta")
         .reindex(columns=codes))
    beta = np.stack([B.loc[k].reindex(ages).to_numpy("float32") for k in
                     range(1, K + 1)], axis=2)          # (A, D, K)
    w = pd.read_csv(d / "weights.csv.gz")
    th = w[[c for c in w.columns if c.startswith("theta_")]].to_numpy("float32")
    print(f"  {name}: K={K}, {len(codes)} diseases, ages {ages.min()}-{ages.max()}, "
          f"weights for {len(w):,} participants")
    return {"name": name, "dir": d, "beta": beta, "ages": ages, "codes": codes,
            "K": K, "theta": th, "eid": w["eid"].to_numpy("int64"),
            "n_dx": w["n_dx"].to_numpy("int32"), "meta": meta}


def record_responsibilities(fit, rec: pd.DataFrame, chunk=2_000_000):
    """r_nk proportional to theta_sk * beta_k,j(t_n), normalised per diagnosis."""
    cix = {c: i for i, c in enumerate(fit["codes"])}
    rec = rec[rec["code"].isin(cix)].copy()
    pix = pd.Series(np.arange(len(fit["eid"]), dtype="int32"), index=fit["eid"])
    rec["pi"] = pix.reindex(rec["eid"]).to_numpy("float64")
    rec = rec[np.isfinite(rec["pi"])].copy()
    rec["pi"] = rec["pi"].astype("int32")
    rec["ci"] = rec["code"].map(cix).to_numpy("int32")
    ages = fit["ages"]
    # integer age grid: use the cell containing the age (floor)
    rec["ai"] = np.clip(np.searchsorted(ages, rec["age"].to_numpy(), side="right") - 1,
                        0, len(ages) - 1).astype("int32")
    if not len(rec):
        fail("no diagnosis matches the fit's vocabulary or participants; "
             "ob_records and the fit come from different runs")
        sys.exit(1)
    print(f"  {len(rec):,} records / {rec['eid'].nunique():,} participants matched")
    out = np.empty((len(rec), fit["K"]), dtype="float32")
    B, TH = fit["beta"], fit["theta"]
    for s in range(0, len(rec), chunk):
        sl = slice(s, min(s + chunk, len(rec)))
        a = rec["ai"].to_numpy()[sl]
        c = rec["ci"].to_numpy()[sl]
        q = TH[rec["pi"].to_numpy()[sl]] * B[a, c, :]
        out[sl] = q / np.clip(q.sum(1, keepdims=True), 1e-30, None)
    return rec.reset_index(drop=True), out


def name_topics(p, cfg, fit, rec, R) -> pd.DataFrame:
    """Topic profile table: names, dominant chapter, associated diseases, top diseases."""
    st = cfg["ob"]["naming"]
    ch = chapters_of_vocab(fit["codes"], cfg, p).set_index("code")
    bad = float((ch["chapter"] == "?").mean())
    if bad > 0.20:
        fail(f"{bad:.0%} of codes have no ICD-10 chapter; derived/ob_code_to_icd3.csv "
             "missing or stale (rerun b1_records.py)")
        sys.exit(1)
    names = code_labels(cfg, fit["codes"])
    pub = published_names(p, cfg, fit)
    D, K = len(fit["codes"]), fit["K"]
    # disease x topic: mean responsibility over the disease's diagnoses
    A = np.zeros((D, K), dtype="float64")
    cnt = np.zeros(D, dtype="float64")
    np.add.at(A, rec["ci"].to_numpy(), R.astype("float64"))
    np.add.at(cnt, rec["ci"].to_numpy(), 1.0)
    Aavg = A / np.clip(cnt[:, None], 1, None)
    chs = ch.loc[fit["codes"], "chapter"].to_numpy()
    ch_idx = dict(zip(ch["chapter"], ch["chapter_index"]))
    rows = []
    for k in range(K):
        by_ch = pd.Series(A[:, k], index=chs).groupby(level=0).sum()
        top = by_ch.sort_values(ascending=False)
        top = top[top > 0].head(int(st["name_top_chapters"]))
        dom = str(top.index[0]) if len(top) else "?"
        assoc = int((Aavg[:, k] > float(st["assign_threshold"])).sum())
        ord_d = np.argsort(-Aavg[:, k])
        chap_name = "+".join(str(x) for x in top.index)
        rows.append({
            "topic": k + 1,
            "name": pub[k + 1][0] if k + 1 in pub else chap_name,
            "chapter_name": chap_name,
            "published_acronym": pub[k + 1][0] if k + 1 in pub else "",
            "published_systems": pub[k + 1][1] if k + 1 in pub else "",
            "published_n_assoc": pub[k + 1][2] if k + 1 in pub else np.nan,
            "dominant_chapter": dom,
            "dominant_chapter_index": int(ch_idx.get(dom, 99)),
            "chapter_share_top": float(top.iloc[0] / max(A[:, k].sum(), 1e-9))
            if len(top) else np.nan,
            "n_assoc_diseases": assoc,
            "mass_diagnoses": float(A[:, k].sum()),
            "share_of_all_diagnoses": float(A[:, k].sum() / max(A.sum(), 1e-9)),
            "mean_age_of_diagnoses": float(np.average(
                rec["age"].to_numpy(), weights=R[:, k])),
            "top_diseases": ";".join(f"{fit['codes'][j]}({Aavg[j, k]:.2f})"
                                     for j in ord_d[:8]),
            "top_disease_names": ";".join(
                names.get(fit["codes"][j], fit["codes"][j]) for j in ord_d[:8])})
    tab = pd.DataFrame(rows)
    tab = tab.sort_values(["dominant_chapter_index", "topic"]).reset_index(drop=True)
    tab["display_order"] = np.arange(1, len(tab) + 1)
    return tab, Aavg


def write_topics_json(p, fit, tab: pd.DataFrame, extra: dict) -> None:
    f = p["derived"] / "ob_topics.json"
    write_json(f, {
        "fit": fit["name"], "K": fit["K"], "engine": fit["meta"].get("engine"),
        "named_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "note": "topic names fixed before any association was estimated",
        "topics": tab.to_dict("records"), **extra})
    ok(f.name)


def main():
    ap = argparse.ArgumentParser(description="Topic names and diagnosis-topic assignment")
    ap.add_argument("--fit", default=DEFAULT_FIT, help="fit directory under derived/")
    args = ap.parse_args()
    cfg = load_cfg_out()
    p = paths_out(cfg)
    name = args.fit

    with tee(p, "b3_topics"):
        h1(f"b3 topics: {name}")
        h2("Fit")
        fit = load_fit(p, name)
        for k, v in fit["meta"].items():
            if k in ("engine", "K", "n_person", "n_records", "role",
                     "n_records_dropped_age_gt_81", "vocab_coverage_of_published"):
                print(f"    {k:<28} {v}")
        h2("Diagnosis-level topic assignment")
        rec, R = record_responsibilities(fit, read_records(p))
        print(f"  median of the maximum responsibility per diagnosis: "
              f"{np.median(R.max(1)):.2f}")
        h2("Topic names")
        tab, Aavg = name_topics(p, cfg, fit, rec, R)
        f = p["tables"] / "OB2_topic_profile.csv"
        tab.to_csv(f, index=False)
        ok(f.name)
        for r in tab.itertuples():
            print(f"    T{int(r.topic):<2} {r.name:<6} chapter {r.dominant_chapter:<5}"
                  f" share {r.share_of_all_diagnoses:>5.1%}"
                  f" assigned diseases {int(r.n_assoc_diseases):>3}"
                  f" mean age {r.mean_age_of_diagnoses:>5.1f}")
        pd.DataFrame(Aavg, index=fit["codes"],
                     columns=[f"T{i + 1}" for i in range(fit["K"])]).to_csv(
            p["tables"] / "OB2_disease_topic_assignment.csv")
        ok("OB2_disease_topic_assignment.csv")
        write_topics_json(p, fit, tab, {
            "records_used": int(len(rec)),
            "mean_max_responsibility": float(np.median(R.max(1)))})
        h2("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
