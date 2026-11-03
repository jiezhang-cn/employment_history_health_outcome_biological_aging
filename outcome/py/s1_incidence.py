#!/usr/bin/env python3
"""s1: incident disease, rate ratios for every code with enough events.

For each sex and each three-character ICD-10 code with at least
`surv.min_code_events` incident events after time zero, a cause-specific
piecewise-exponential Poisson model on attained-age bands (delayed entry at
time zero) estimates the adjusted rate ratio of each career type versus all
other types of the same sex. Participants with the code recorded on or
before time zero (including undated prevalent records) are excluded from
that code's risk set; follow-up ends at diagnosis, death, loss to follow-up
or the end of the common diagnosis window. Chapters XV-XVII (pregnancy,
perinatal, congenital) are not analysed, because a record after age 45 is
new recording rather than new onset. Cells with fewer than
`surv.min_cell_events` events are suppressed, q values are BH-FDR within
sex x analysis, and the number of nominal associations is compared with the
number expected under the null.

Inputs:  derived/op_person.parquet, derived/op_disease/ (o1),
         tables/OP0_code_catalog.csv (o1 --stage feasibility),
         derived/op_onset.parquet (s0); all under results/outcome/
Outputs: results/outcome/tables/S1_incidence.csv    one row per sex x code x type
         results/outcome/tables/S1_code_screen.csv  events per candidate code

Run:  python outcome/py/s1_incidence.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common_out import (fail, h1, h2, load_cfg_out, load_cfg_up, ok, paths_out,
                        sexes, tee, warn)
from o1_assemble import load_matrices
from surv_common import (EPS_YEAR, add_fdr, add_type_cols, band_edges,
                         dx_window_end, load_onset, load_person, rate_ratio_rest,
                         sex_setup, split_bands, suppress_events, timeline)


SKIP_CHAPTERS = ("XV", "XVI", "XVII")
NOMINAL_P = 0.05


def candidate_codes(p, cfg, sex) -> pd.DataFrame:
    """Pre-screen codes on o1's per-sex incident count (cheap). That count has
    no left truncation or window constraint, so it is an upper bound on the
    event count computed per code in `fit`."""
    f = p["tables"] / "OP0_code_catalog.csv"
    if not f.exists():
        fail(f"missing {f.name}: run o1_assemble.py --stage feasibility")
        sys.exit(1)
    cat = pd.read_csv(f)
    cat["code"] = cat["code"].astype(str)
    c = cat[cat["sex"] == sex].copy()
    if not len(c):
        fail(f"OP0_code_catalog.csv has no rows for sex={sex}")
        sys.exit(1)
    thr = int(cfg["surv"]["min_code_events"])
    pseudo = str(cfg["cancer"]["icd9_pseudo_code"])
    n0 = len(c)
    c = c[c["code"] != pseudo]
    c = c[~c["chapter"].isin(SKIP_CHAPTERS)]
    n1 = len(c)
    c = c[c["n_incident"] >= thr]
    print(f"  {sex} pre-screen: {n0} codes -> {n1} after removing the ICD-9 "
          f"pseudo-code and chapters {'/'.join(SKIP_CHAPTERS)} -> "
          f"{len(c)} with n_incident >= {thr}")
    return (c.sort_values(["chapter_index", "code"])
            [["code", "chapter", "chapter_index", "chapter_label", "n_incident",
              "prev_ever"]].reset_index(drop=True))


def onset_by_code(onset: pd.DataFrame) -> dict:
    """Post-time-zero first records split by code (one groupby)."""
    post = onset[onset["post_tz"].to_numpy()]
    return {str(c): g for c, g in post.groupby("code", sort=False)}


def onset_ages(by_code: dict, code: str, n: int) -> np.ndarray:
    """Post-time-zero onset age of one code, length n (NaN = no event)."""
    a = np.full(n, np.nan)
    g = by_code.get(str(code))
    if g is not None and len(g):
        a[g["row"].to_numpy("int64")] = g["age"].to_numpy("float64")
    return a


def outcome_arrays(v, tl_all, onset_age_all, prev_col):
    """-> (entry, exit, cause, at_risk); cause 1 = event, 2 = death, 0 = censored.

    `at_risk` comes from o1's prevalent matrix, which also holds the undated
    records known to precede time zero.
    """
    r = v["rows"]
    entry = tl_all["entry"].to_numpy("float64")[r]
    ex = tl_all["exit_dx"].to_numpy("float64")[r]
    isd = tl_all["dx_exit_is_death"].to_numpy(bool)[r]
    oa = onset_age_all[r]
    ev = np.isfinite(oa) & (oa > entry) & (oa <= ex + EPS_YEAR)
    exit_ = np.where(ev, oa, ex)
    cause = np.where(ev, 1, np.where(isd, 2, 0))
    at_risk = ~prev_col[r].astype(bool)
    return entry, exit_, cause.astype("int64"), at_risk


def fit(p, cfg, cup) -> pd.DataFrame:
    """Per code: rebuild the risk set, count events, fit if above threshold;
    then suppress small cells and add FDR."""
    sv = cfg["surv"]
    thr = int(sv["min_code_events"])
    person = load_person(p)
    tl = timeline(person, cfg, dx_window_end(p))
    by_code = onset_by_code(load_onset(p))
    M, meta, codes = load_matrices(p, person, ("prevalent",))
    prev = M["prevalent"]
    cix = {str(c): i for i, c in enumerate(codes)}
    rows, screens = [], []
    for sex in sexes(cup):
        h2(f"{sex}")
        v = sex_setup(p, cfg, person, tl, sex)
        K = len(v["types"])
        cand = candidate_codes(p, cfg, sex)
        edges = band_edges(cfg, tl["entry"].to_numpy()[v["rows"]],
                           tl["exit_dx"].to_numpy()[v["rows"]])
        print(f"    {len(edges) - 1} age bands ({edges[0]:.0f}-{edges[-1]:.0f}, "
              f"width {sv['model_band']:.1f}); up to {len(cand)} codes x {K} fits")
        scr, kept = [], []
        for r in cand.itertuples():
            if r.code not in cix:
                warn(f"{r.code} is in the catalog but not in op_disease: skipped")
                continue
            oa = onset_ages(by_code, r.code, len(person))
            pc = np.asarray(prev[:, cix[r.code]].todense()).ravel()
            entry, exit_, cause, at_risk = outcome_arrays(v, tl, oa, pc)
            m = at_risk & (exit_ > entry)
            nev = int((cause[m] == 1).sum())
            scr.append({"sex": sex, "code": r.code, "chapter": r.chapter,
                        "chapter_index": int(r.chapter_index),
                        "chapter_label": r.chapter_label,
                        "n_incident_prescreen": int(r.n_incident),
                        "n_at_risk": int(m.sum()), "events": nev,
                        "competing_deaths": int((cause[m] == 2).sum()),
                        "passed": bool(nev >= thr and m.sum() >= 100)})
            if nev < thr or m.sum() < 100:
                continue
            b, bb, pt, ev = split_bands(entry[m], exit_[m],
                                        (cause[m] == 1).astype("float64"), edges)
            meta_row = {"sex": sex, "analysis": "incidence", "outcome": r.code,
                        "chapter": r.chapter, "chapter_index": int(r.chapter_index),
                        "chapter_label": r.chapter_label,
                        "n_at_risk": int(m.sum()), "events_code": nev,
                        "n_prevalent_excluded": int((~at_risk).sum())}
            got = rate_ratio_rest(bb, v["strata"][m][b], v["lab"][m][b], pt, ev,
                                  v["P"], len(edges) - 1, K, cfg, meta_row,
                                  ph=True)
            rows += add_type_cols(got, v)
            kept.append(r.code)
        screens.append(pd.DataFrame(scr))
        if not len(kept):
            warn(f"{sex}: no code reached {thr} events")
            continue
        print(f"    {len(cand)} pre-screened -> {len(kept)} codes with >= {thr} events")
    df = pd.DataFrame(rows)
    if not len(df):
        fail(f"no code estimated: every code has < surv.min_code_events = {thr} "
             "events (see S1_code_screen.csv)")
        sys.exit(1)
    h2("suppression and FDR")
    df = suppress_events(df, sv["min_cell_events"])
    print(f"  suppressed {int(df['suppressed'].sum()):,}/{len(df):,} cells "
          f"(events < {sv['min_cell_events']})")
    df = add_fdr(df, cfg)
    sc = pd.concat(screens, ignore_index=True) if screens else pd.DataFrame()
    df.to_csv(p["tables"] / "S1_incidence.csv", index=False)
    sc.to_csv(p["tables"] / "S1_code_screen.csv", index=False)
    ok(f"S1_incidence.csv ({len(df):,} rows) / S1_code_screen.csv ({len(sc):,} rows)")
    return df


def report(df, cfg) -> None:
    """Nominal associations versus the number expected under the null, per sex."""
    h2("nominal vs expected, FDR, convergence")
    a = float(cfg["surv"]["fdr_alpha"])
    for sex, g0 in df.groupby("sex"):
        g = g0[g0["log_rr"].notna()]
        m = len(g)
        if not m:
            continue
        n_nom = int((g["p"] < NOMINAL_P).sum())
        n_q = int((g["q"] < a).sum())
        print(f"  {sex}: {g['outcome'].nunique()} codes, {m} estimable cells; "
              f"P<{NOMINAL_P:g}: {n_nom} (expected {NOMINAL_P * m:.0f}); "
              f"q<{a:g}: {n_q}")
    nc = int((~df.loc[df["log_rr"].notna(), "converged"].astype(bool)).sum())
    if nc:
        warn(f"{nc} cells did not converge")


def main():
    cfg, cup = load_cfg_out(), load_cfg_up()
    p = paths_out(cfg)
    with tee(p, "s1_incidence"):
        h1("s1: incident disease, rate ratios (each type vs the rest)")
        df = fit(p, cfg, cup)
        report(df, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
