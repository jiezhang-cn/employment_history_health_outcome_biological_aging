#!/usr/bin/env python3
"""s2: disease-free years between ages 60 and 80 and their decomposition.

Disease-free life ends at the first of the components in
`surv.healthspan.components` (cancer excluding C44, diabetes, myocardial
infarction, heart failure, stroke, COPD, dementia, death). The analysis is
restricted to participants free of all components at time zero; follow-up
ends at the first component, loss to follow-up or the end of the common
diagnosis window (the same end is used for death, so death does not get a
longer observation window than the diseases).

Method (Andersen, Stat Med 2013; Erlangsen et al., Lancet Psychiatry 2017):
for each career type, an Aalen-Johansen estimator on attained age with left
truncation gives the cumulative incidence F_c of disease-free life ending
with component c, among those disease-free at `years_from`. Years lost to c
are RMTL_c = int F_c over [years_from, years_to], and disease-free years =
(years_to - years_from) - sum_c RMTL_c exactly, so the cause-specific
contributions add up to the total difference.

Reference type: among types with at least `worst_min_events` events, the one
with the fewest disease-free years. Each other type is compared with it.
95% CIs and two-sided p values come from `decomp_boot` bootstrap samples
drawn within type (percentile method); `ref_rank_prob` is the share of
bootstrap samples in which the reference is still the type with the fewest
disease-free years. Estimates are unadjusted.

Inputs:  derived/op_person.parquet, derived/op_disease/ (o1),
         derived/op_onset.parquet (s0); all under results/outcome/
Outputs: results/outcome/tables/S2_healthspan.csv   analysis = healthspan_vsworst
             (difference in disease-free years per type) and
             healthspan_decomp_vsworst (years lost per component)
         results/outcome/tables/S2_components.csv  codes of each component

Run:  python outcome/py/s2_healthspan.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common_out import (fail, h1, h2, load_cfg_out, load_cfg_up, ok, paths_out,
                        sexes, tee, warn)
from o1_assemble import load_matrices
from surv_common import (EPS_YEAR, add_fdr, add_type_cols, aj_rmtl,
                         dx_window_end, load_onset, load_person, sex_setup,
                         suppress_events, timeline)


def component_codes(comp: dict, codes) -> list:
    """Component -> code list: explicit `codes`, plus all codes starting with
    a `prefix`, minus `exclude` (C44, non-melanoma skin cancer, would
    otherwise dominate the cancer component)."""
    out = set(str(c) for c in comp.get("codes", []))
    for pre in comp.get("prefix", []):
        out |= {c for c in codes if str(c).startswith(str(pre))}
    return sorted(out - {str(x) for x in comp.get("exclude", [])})


def build_components(cfg, onset, prev, codes, person, tl):
    """-> ({component: (onset age, present at time zero)}, component table).
    Arrays have one entry per cohort member; death comes from the time line."""
    n = len(person)
    cix = {c: i for i, c in enumerate(codes)}
    post = onset[onset["post_tz"].to_numpy()]
    by_code = {c: g for c, g in post.groupby("code", sort=False)}
    out, info = {}, []
    for comp in cfg["surv"]["healthspan"]["components"]:
        name = comp["name"]
        if comp.get("death"):
            age = tl["death_age"].to_numpy("float64").copy()
            pre = np.zeros(n, dtype=bool)
            info.append({"component": name, "label": comp.get("label", name),
                         "n_codes": 0, "codes": "(death registry)"})
        else:
            mem = [c for c in component_codes(comp, codes) if c in cix]
            if not mem:
                warn(f"component {name}: none of its codes is in the code table, skipped")
                continue
            age = np.full(n, np.inf)
            for c in mem:
                g = by_code.get(c)
                if g is None:
                    continue
                r = g["row"].to_numpy("int64")
                age[r] = np.minimum(age[r], g["age"].to_numpy("float64"))
            age = np.where(np.isfinite(age), age, np.nan)
            j = [cix[c] for c in mem]
            pre = np.asarray(prev[:, j].sum(1)).ravel() > 0
            info.append({"component": name, "label": comp.get("label", name),
                         "n_codes": len(mem),
                         "codes": ",".join(mem[:12]) + ("..." if len(mem) > 12 else "")})
        out[name] = (age, pre)
    print(f"  {len(out)} components:")
    for r in info:
        print(f"    {r['component']:<9} {r['n_codes']:>4} codes  {r['codes']}")
    return out, pd.DataFrame(info)


def first_event(comps: dict, entry, exit_):
    """-> (event age, index of the first component, event flag). Ties go to
    the component listed first in the config."""
    names = list(comps)
    A = np.stack([comps[k][0] for k in names])            # component x person
    A = np.where(np.isfinite(A) & (A > entry[None, :])
                 & (A <= exit_[None, :] + EPS_YEAR), A, np.inf)
    who = np.argmin(A, axis=0)
    age = A[who, np.arange(A.shape[1])]
    ev = np.isfinite(age)
    return np.where(ev, age, np.nan), np.where(ev, who, -1), ev


def fit(p, cfg, cup):
    sv = cfg["surv"]
    person = load_person(p)
    tl = timeline(person, cfg, dx_window_end(p))
    onset = load_onset(p)
    M, meta, codes = load_matrices(p, person, ("prevalent",))
    comps, cinfo = build_components(cfg, onset, M["prevalent"], codes, person, tl)
    if len(comps) < 2:
        fail("fewer than 2 components: composite endpoint undefined")
        sys.exit(1)
    names = list(comps)
    rows = []
    for sex in sexes(cup):
        h2(f"{sex}")
        v = sex_setup(p, cfg, person, tl, sex, design=False)
        r = v["rows"]
        entry = tl["entry"].to_numpy("float64")[r]
        adm = tl["exit_dx"].to_numpy("float64")[r]
        sub = {k: (a[r], pre[r]) for k, (a, pre) in comps.items()}
        free = ~np.any(np.stack([pre for _, pre in sub.values()]), axis=0)
        age, who, ev = first_event(sub, entry, adm)
        m = free & (adm > entry)
        exit_ = np.where(ev, age, adm)
        print(f"    disease-free at time zero {int(free.sum()):,}/{len(r):,} "
              f"({free.mean():.1%}); analysed {int(m.sum()):,}, "
              f"events {int(ev[m].sum()):,}")
        base = {"sex": sex, "n_at_risk": int(m.sum()),
                "n_prevalent_excluded": int((~free).sum())}
        rows += vs_reference(v, m, entry, exit_, ev, who, names, cfg, base, sex)
    df = pd.DataFrame(rows)
    if not len(df):
        fail("no estimates produced")
        sys.exit(1)
    h2("suppression and FDR")
    df = suppress_events(df, sv["min_cell_events"])
    df = add_fdr(df, cfg)
    df.to_csv(p["tables"] / "S2_healthspan.csv", index=False)
    cinfo.to_csv(p["tables"] / "S2_components.csv", index=False)
    ok("S2_healthspan.csv / S2_components.csv")
    return df


def vs_reference(v, m, entry, exit_, ev, who, names, cfg, base, sex) -> list:
    """Disease-free years of each type minus those of the reference type, and
    the decomposition into years lost to each component, with bootstrap CIs."""
    hs = cfg["surv"]["healthspan"]
    K, C = len(v["types"]), len(names)
    a0, a1 = float(hs["years_from"]), float(hs["years_to"])
    idx = np.flatnonzero(m)
    lab = v["lab"][idx]
    en, ex = entry[idx], exit_[idx]
    cz = np.where(ev[idx], who[idx] + 1, 0).astype("int64")
    by = [np.flatnonzero(lab == k) for k in range(K)]

    def one(ii):
        return aj_rmtl(en[ii], ex[ii], cz[ii], np.ones(len(ii)), a0, a1, C)

    est = [one(by[k]) for k in range(K)]
    nB = int(hs["decomp_boot"])
    rng = np.random.default_rng(int(cfg["op"]["null_seed"]))
    L = np.full((nB, K, C), np.nan)
    for b_ in range(nB):
        # resample persons within type (type sizes fixed)
        draw = [rng.choice(by[k], size=len(by[k]), replace=True) if len(by[k])
                else by[k] for k in range(K)]
        for k in range(K):
            if len(draw[k]):
                L[b_, k] = one(draw[k])["rmtl"]
    print(f"    {nB} bootstrap samples")
    ok_b = np.isfinite(L).all(axis=(1, 2))
    L = L[ok_b]
    nB = int(ok_b.sum())
    T = L.sum(2)                                   # bootstrap x type: total years lost
    ev_k = np.bincount(lab, weights=(cz > 0).astype(float), minlength=K)
    elig = ev_k >= float(hs["worst_min_events"])
    lost = np.array([e_["rmtl"].sum() for e_ in est])
    print(f"    disease-free years {a0:.0f}-{a1:.0f} (Aalen-Johansen, unadjusted): "
          + ", ".join(f"{int(v['types'][k])}:{a1 - a0 - lost[k]:.2f}" for k in range(K)))
    for k in range(K):
        if est[k]["truncated"] or est[k]["n_a1"] < int(hs["aj_min_at_risk_end"]):
            warn(f"{sex} type {int(v['types'][k])}: only {est[k]['n_a1']} at risk "
                 f"at age {a1:.0f}; estimate unstable"
                 + ("; risk set emptied inside the window" if est[k]["truncated"] else ""))
    if not elig.any():
        warn(f"{sex}: no type has >= {hs['worst_min_events']} events; "
             "no reference type, no estimates")
        return []
    w_ = int(np.flatnonzero(elig)[np.argmax(lost[elig])])
    wt = int(v["types"][w_])
    Te = np.where(elig[None, :], T, -np.inf)
    hit = np.bincount(np.argmax(Te, 1), minlength=K) / nB
    prob = float(hit[w_])
    order = [k for k in np.argsort(-hit) if k != w_ and hit[k] > 0]
    ru = order[0] if order else None
    jumped = [int(v["types"][k]) for k in np.flatnonzero(~elig) if lost[k] > lost[w_]]
    if jumped:
        warn(f"{sex}: types {jumped} lose more years than {wt} but have < "
             f"{hs['worst_min_events']} events and are not eligible as reference")
    print(f"    reference type = {wt} ({int(ev_k[w_]):,} events; fewest disease-free "
          f"years in {prob:.2f} of bootstrap samples"
          + (f", next type {int(v['types'][ru])} ({hit[ru]:.2f})" if ru is not None else "")
          + ")")
    if prob < float(hs["worst_rank_warn"]):
        warn(f"{sex}: reference choice unstable ({prob:.2f} < {hs['worst_rank_warn']})")

    def pct(x):
        x = x[np.isfinite(x)]
        if not len(x):
            return np.nan, np.nan, np.nan
        p_ = 2 * min((x <= 0).mean(), (x >= 0).mean())
        return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)),
                float(min(1.0, max(p_, 1.0 / len(x)))))

    meta = {"ref_type": wt, "ref_type_index": w_, "comparator": f"type {wt}",
            "ref_rank_prob": prob,
            "ref_runner_up": int(v["types"][ru]) if ru is not None else np.nan,
            "ref_runner_prob": float(hit[ru]) if ru is not None else np.nan,
            "method": "aalen_johansen", "n_boot": nB,
            "hs_years_from": a0, "hs_years_to": a1,
            "worst_min_events": float(hs["worst_min_events"])}
    cev = [np.bincount(lab, weights=(cz == j + 1).astype(float), minlength=K)
           for j in range(C)]
    out = []
    for k in range(K):
        if k == w_:
            continue
        d = lost[w_] - lost[k]                     # disease-free years, type - reference
        lo, hi, p_ = pct(T[:, w_] - T[:, k])
        out.append({**base, **meta, "analysis": "healthspan_vsworst",
                    "outcome": "healthspan", "type_index": k,
                    "events_type": float(ev_k[k]), "events_rest": float(ev_k[w_]),
                    "n_type": int(len(by[k])),
                    "n_risk_from": est[k]["n_a0"], "n_risk_to": est[k]["n_a1"],
                    "eligible_ref": bool(elig[k]),
                    "hs_years_type": a1 - a0 - lost[k],
                    "hs_years_rest": a1 - a0 - lost[w_], "hs_years_diff": d,
                    "hs_years_lo": lo, "hs_years_hi": hi, "p": p_, "q": np.nan})
        for j, cname in enumerate(names):
            lt, lr = float(est[k]["rmtl"][j]), float(est[w_]["rmtl"][j])
            clo, chi, cp = pct(L[:, w_, j] - L[:, k, j])
            out.append({"sex": sex, "analysis": "healthspan_decomp_vsworst",
                        "outcome": cname, "type_index": k, **meta,
                        "years_lost_type": lt, "years_lost_rest": lr,
                        "contribution_years": -(lt - lr),
                        "contribution_lo": clo, "contribution_hi": chi,
                        "p": cp, "q": np.nan,
                        "events_type": float(cev[j][k]),
                        "events_rest": float(cev[j][w_]),
                        "years_from": a0, "years_to": a1,
                        "years_diff_direct": d, "years_diff_decomposed": d})
    out = add_type_cols(out, v)
    print(f"    disease-free years vs type {wt}: " + ", ".join(
        f"{x['type']}:{x['hs_years_diff']:+.2f}" for x in out
        if x["analysis"] == "healthspan_vsworst"))
    return out


def main():
    cfg, cup = load_cfg_out(), load_cfg_up()
    p = paths_out(cfg)
    with tee(p, "s2_healthspan"):
        hs = cfg["surv"]["healthspan"]
        h1(f"s2: disease-free years {hs['years_from']:.0f}-{hs['years_to']:.0f} "
           "and decomposition by terminating event")
        fit(p, cfg, cup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
