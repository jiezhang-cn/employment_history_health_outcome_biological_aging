#!/usr/bin/env python3
"""s3: accumulation of new diagnoses after time zero.

The outcome is the number of distinct three-character ICD-10 codes, from any
chapter, first recorded after time zero (each code counts once per person,
so this measures disease accumulation, not health-care contact).

  * Adjusted rate ratio of each career type versus all other types of the
    same sex: piecewise-exponential Poisson model for recurrent events on
    attained-age bands with delayed entry, with a person-clustered sandwich
    variance (Lin, Wei, Yang & Ying 2000).
  * Unadjusted mean cumulative function (MCF) by type, with death as a
    terminating event (Ghosh-Lin).
  * Years-earlier shift: the age at which each type's MCF reaches
    `dxrate.target_count` diagnoses per person, compared with the age at
    which the whole cohort of that sex reaches it, with a bootstrap CI.

Inputs:  derived/op_person.parquet, derived/op_disease/meta.json (o1),
         derived/op_onset.parquet (s0); all under results/outcome/
Outputs: results/outcome/tables/S3_dx_rate_ratio.csv  rate ratios per type
         results/outcome/tables/S3_mcf.csv            MCF by type and age
         results/outcome/tables/S3_dx_shift.csv       years-earlier shift

Run:  python outcome/py/s3_dxrate.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common_out import (h1, h2, load_cfg_out, load_cfg_up, ok, paths_out, sexes,
                        tee, warn)
from surv_common import (CURVE_COLS, EPS_YEAR, SHIFT_COLS, add_fdr,
                         add_type_cols, band_edges, boot_cross, cross_age,
                         dx_window_end, frame, load_onset, load_person, mcf,
                         rate_ratio_rest, sex_setup, suppress_events, timeline)


def event_list(onset, tl):
    """-> (cohort row, age) of every new diagnosis after time zero and within
    disease follow-up."""
    sub = onset[onset["post_tz"].to_numpy()]
    r = sub["row"].to_numpy("int64")
    a = sub["age"].to_numpy("float64")
    ex = tl["exit_dx"].to_numpy("float64")
    en = tl["entry"].to_numpy("float64")
    keep = (a > en[r]) & (a <= ex[r] + EPS_YEAR)
    print(f"  {len(r):,} post-time-zero diagnoses -> {int(keep.sum()):,} within "
          f"follow-up in {int(pd.unique(r[keep]).size):,} persons")
    return r[keep], a[keep]


def band_counts(ev_row, ev_age, entry, exit_, edges, sel):
    """Recurrent events -> (person index, band index, person-years, events in
    that band). `ev_row` must already be positions within the sex subset;
    returned person indices are positions within `sel`."""
    idx = np.flatnonzero(sel)
    pos = np.full(len(sel), -1, dtype="int64")
    pos[idx] = np.arange(len(idx))
    lo = np.maximum(entry[idx][:, None], edges[None, :-1])
    hi = np.minimum(exit_[idx][:, None], edges[None, 1:])
    pt = np.clip(hi - lo, 0.0, None)
    nb = len(edges) - 1
    m = pt > 0
    r, b = np.nonzero(m)
    key = r * nb + b
    order = np.argsort(key, kind="mergesort")
    r, b = r[order], b[order]
    ptv = pt[m][order]
    # band of each event; right-open intervals, consistent with the pt split
    keep = pos[ev_row] >= 0
    er, ea = pos[ev_row[keep]], ev_age[keep]
    eb = np.clip(np.searchsorted(edges, ea, side="right") - 1, 0, nb - 1)
    cnt = np.bincount(er * nb + eb, minlength=len(idx) * nb)
    ev = cnt[r * nb + b].astype("float64")
    miss = cnt.sum() - ev.sum()
    if miss > 0:
        warn(f"{miss:.0f} diagnoses fall in zero person-time pieces: dropped")
    return r, b, ptv, ev


def fit(p, cfg, cup):
    sv = cfg["surv"]
    person = load_person(p)
    tl = timeline(person, cfg, dx_window_end(p))
    onset = load_onset(p)
    gr, ga = event_list(onset, tl)
    g0, g1, gs_ = sv["curve_grid"]
    grid = np.arange(g0, g1 + 1e-9, gs_)
    rows, curves, shifts = [], [], []
    for sex in sexes(cup):
        h2(f"{sex}")
        v = sex_setup(p, cfg, person, tl, sex)
        r, K = v["rows"], len(v["types"])
        loc = np.full(len(person), -1, dtype="int64")
        loc[r] = np.arange(len(r))
        entry = tl["entry"].to_numpy("float64")[r]
        exit_ = tl["exit_dx"].to_numpy("float64")[r]
        died = (tl["dx_exit_is_death"].to_numpy(bool)[r]).astype("int64")
        keep = loc[gr] >= 0
        er, ea = loc[gr[keep]], ga[keep]
        sel = exit_ > entry
        edges = band_edges(cfg, entry[sel], exit_[sel])
        pr, pb, pt, ev = band_counts(er, ea, entry, exit_, edges, sel)
        print(f"    {pt.sum():,.0f} person-years; {ev.sum():,.0f} new diagnoses "
              f"({ev.sum() / max(sel.sum(), 1):.2f} per person, "
              f"{ev.sum() / max(pt.sum(), 1e-9):.3f} per person-year)")
        idx = np.flatnonzero(sel)
        got = rate_ratio_rest(pb, v["strata"][idx][pr], v["lab"][idx][pr], pt, ev,
                              v["P"], len(edges) - 1, K, cfg,
                              {"sex": sex, "analysis": "dx_rate",
                               "outcome": "new_diagnoses",
                               "n_at_risk": int(sel.sum())},
                              person_r=pr, ph=True)
        rows += add_type_cols(got, v)
        curves += mcf_rows(v, entry, exit_, died, er, ea, sel, grid, cfg, sex)
        shifts += shift_rows(v, entry, exit_, died, er, ea, sel, grid, cfg, sex)
    df = suppress_events(pd.DataFrame(rows), sv["min_cell_events"])
    df = add_fdr(df, cfg)
    cv = frame(curves, CURVE_COLS + ("mcf", "events"))
    sh = frame(shifts, SHIFT_COLS + ("n",))
    for name, d in (("S3_dx_rate_ratio", df), ("S3_mcf", cv), ("S3_dx_shift", sh)):
        d.to_csv(p["tables"] / f"{name}.csv", index=False)
    ok("S3_dx_rate_ratio / S3_mcf / S3_dx_shift")
    return df, cv, sh


def mcf_rows(v, entry, exit_, died, er, ea, sel, grid, cfg, sex) -> list:
    """MCF of the whole cohort ("ALL") and of each type; ages with fewer than
    `min_at_risk` people at risk are omitted."""
    out = []
    for k, t in enumerate([None] + list(v["types"])):
        s = sel if t is None else (sel & (v["lab"] == k - 1))
        idx = np.flatnonzero(s)
        if len(idx) < 20:
            continue
        keep = np.isin(er, idx)
        M, nr = mcf(entry[idx], exit_[idx], died[idx], ea[keep], grid)
        M = np.where(nr < cfg["surv"]["min_at_risk"], np.nan, M)
        for a, m_, n_ in zip(grid, M, nr):
            if np.isfinite(m_):
                out.append({"sex": sex, "type": "ALL" if t is None else int(t),
                            "age": float(a), "mcf": float(m_), "n_risk": int(n_),
                            "n_group": int(len(idx)),
                            "events": int(keep.sum())})
    return out


def shift_rows(v, entry, exit_, died, er, ea, sel, grid, cfg, sex) -> list:
    """Age at which each type reaches `target_count` new diagnoses per person,
    and the shift in years relative to the whole cohort (positive = earlier).

    If the cohort MCF never reaches the target within follow-up, the level
    falls back to 80% of its maximum; if some types' curves stop below the
    level, it falls back to 95% of the lowest type maximum. Both are logged.
    """
    sv, dr = cfg["surv"], cfg["surv"]["dxrate"]
    allidx = np.flatnonzero(sel)
    keep0 = np.isin(er, allidx)
    M0, nr0 = mcf(entry[allidx], exit_[allidx], died[allidx], ea[keep0], grid)
    M0 = np.where(nr0 < sv["min_at_risk"], np.nan, M0)
    lvl = float(dr["target_count"])
    if not np.isfinite(np.nanmax(M0)) or np.nanmax(M0) < lvl:
        lvl = 0.8 * float(np.nanmax(M0))
        warn(f"{sex}: cohort MCF max {np.nanmax(M0):.2f} < target "
             f"{dr['target_count']}; level set to {lvl:.2f}")
    kc, kd = {}, {}
    for k in range(len(v["types"])):
        idx = np.flatnonzero(sel & (v["lab"] == k))
        if len(idx) < 50:
            continue
        keep = np.isin(er, idx)
        pos = np.full(len(sel), -1, dtype="int64")
        pos[idx] = np.arange(len(idx))
        by = [np.zeros(0) for _ in idx]
        o = np.argsort(pos[er[keep]], kind="mergesort")
        pp, aa = pos[er[keep]][o], ea[keep][o]
        for g, sl in zip(*_runs(pp)):
            by[g] = aa[sl]
        M, nr = mcf(entry[idx], exit_[idx], died[idx], aa, grid)
        kc[k] = np.where(nr < sv["min_at_risk"], np.nan, M)
        kd[k] = (idx, entry[idx], exit_[idx], died[idx], by)
    caps = [float(np.nanmax(F)) for F in kc.values() if np.isfinite(F).any()]
    if caps and min(caps) < lvl:
        lvl = 0.95 * min(caps)
        warn(f"{sex}: some type MCFs stop below the level; level set to {lvl:.2f}")
    ref = cross_age(grid, M0, lvl)
    mar = sv["min_at_risk"]
    out = []
    for k, t in enumerate(v["types"]):
        idx = np.flatnonzero(sel & (v["lab"] == k))
        row = {"sex": sex, "type": int(t), "n": int(len(idx)), "level": lvl,
               "ref_age": ref, "age_at_level": np.nan, "shift_years": np.nan,
               "shift_lo": np.nan, "shift_hi": np.nan}
        if k in kc:
            _, en, ex, dd, by = kd[k]

            def fn(i, en=en, ex=ex, dd=dd, by=by):
                """Resampled curve, with the same risk-set rule as the estimate."""
                ags = (np.concatenate([by[j] for j in i]) if len(i)
                       else np.zeros(0))
                M, nr = mcf(en[i], ex[i], dd[i], ags, grid)
                return np.where(nr < mar, np.nan, M)

            a = cross_age(grid, kc[k], lvl)
            row["age_at_level"] = a
            row["shift_years"] = (ref - a) if np.isfinite(a) else np.nan
            lo, hi = boot_cross(fn, len(idx), lvl, grid,
                                int(dr.get("boot", 200)) if np.isfinite(a) else 0,
                                cfg["op"]["null_seed"] + 7 * k)
            row["shift_lo"] = ref - hi if np.isfinite(hi) else np.nan
            row["shift_hi"] = ref - lo if np.isfinite(lo) else np.nan
        out.append(row)
    print(f"    level {lvl:.2f} diagnoses (cohort reaches it at {ref:.1f}); "
          "years earlier (+) / later (-): "
          + ", ".join(f"{r['type']}:{r['shift_years']:+.1f}"
                      for r in out if np.isfinite(r["shift_years"])))
    return out


def _runs(sorted_keys):
    """Sorted keys -> (key of each run, slice of each run); splits events by
    person for the bootstrap."""
    if not len(sorted_keys):
        return [], []
    b = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate([[0], b])
    ends = np.concatenate([b, [len(sorted_keys)]])
    return ([int(sorted_keys[s]) for s in starts],
            [slice(int(s), int(e)) for s, e in zip(starts, ends)])


def main():
    cfg, cup = load_cfg_out(), load_cfg_up()
    p = paths_out(cfg)
    with tee(p, "s3_dxrate"):
        h1("s3: accumulation of new diagnoses (recurrent events)")
        fit(p, cfg, cup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
