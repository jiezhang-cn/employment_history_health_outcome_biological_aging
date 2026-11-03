"""a3: clock validation (accuracy V1, mortality V2, protein coverage V7), run before any association
analysis. a4 and a5 read the validation status only from clock_status.json.

  V1  accuracy
        C-P  r(pred, age) >= benchmark - v1_r_tol (a1_CP_perf)
        C-C  a1 accuracy check = use_weights (r >= published test r - v1_r_tol)
        C-M  out-of-bag MAE in CN (before correction, as published) <= published MAE + v1_mae_tol (a2_CM_perf)
        C-B  not applicable (no published UKB-IDP benchmark); own cross-validated performance reported
        C-L  not applicable (published formula / trained in this sample / direct measure); r and MAE reported
  V2  all-cause mortality, Cox model (statsmodels PHReg, Breslow ties): gapz + age at measurement + sex
        (female = 1); time zero = the family's measurement date, censoring at measure.death_censor;
        passes if HR per SD > 1 and p < v2_alpha. Fewer than v2_min_events deaths -> not assessable
        (treated as failed). Everyone with the clock is included (not only the work-history cohort).
        For C-P the published HR (Oh et al., ST9) is reported alongside.
  V7  published-coefficient proteomic families: model-protein coverage >= v7_coverage_min and missing
        model weight <= v7_missing_weight_max (a1_CP_coverage; for C-C the same columns of a1_CC_perf).

main = V1 (where applicable) and V2 and V7 (where applicable). A family in which no clock passes V2 is
flagged (family_exit_on_v2). All clocks are analysed downstream; main is reported with the results.

Outputs (derived/): clock_status.json
Tables (tables/):   a3_T1_validation.csv (per clock: V1 / V2 / V7 and overall status), a3_v2_cox.csv

  python aging/py/a3_validate.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.duration.hazard_regression import PHReg

from common_aging import (FAMILIES, clock_file, data_file, h1, h2, load_clocks, load_measure, ref_file,
                          setup, sha256_of, tee, warn, write_json, ok)

ROUND = 6


def clean(o):
    """JSON-safe: numpy scalars -> Python, NaN -> null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def read_table(p, name):
    f = p["tables"] / name
    if not f.exists():
        return None
    try:
        return pd.read_csv(f)
    except pd.errors.EmptyDataError:                             # all clocks of the family skipped upstream
        return None


# ── Mortality ──────────────────────────────────────────────────────
def death_dates(p, cfg) -> pd.Series:
    d = pd.read_csv(data_file(p, cfg, "death"), sep="\t", usecols=["eid", "date_of_death"])
    d["eid"] = pd.to_numeric(d["eid"], errors="coerce")
    d = d.dropna(subset=["eid"])
    dd = pd.to_datetime(d["date_of_death"], errors="coerce").groupby(d["eid"].astype("int64")).min()
    cens = pd.Timestamp(cfg["measure"]["death_censor"])
    if dd.max() > cens:
        warn(f"deaths recorded after the censoring date {cens.date()} are treated as censored")
    print(f"    death registry: {len(dd):,} participants (earliest date kept)")
    return dd.dropna()


def survival(eids, date_meas, dod, cens):
    """(time in years, event). Death before the measurement date (registry error) -> NaN."""
    dm = pd.to_datetime(pd.Series(date_meas, index=eids))
    dd = dod.reindex(eids)
    event = (dd.notna() & (dd <= cens)).to_numpy()
    end = dd.where(event, cens)
    t = (end.to_numpy("datetime64[ns]") - dm.to_numpy("datetime64[ns]")) / np.timedelta64(1, "D") / 365.25
    t = np.where(t > 0, t, np.nan)
    return t, event.astype(float)


def cox(t, e, X):
    """PHReg (Breslow). Falls back to BFGS when Newton fails or diverges; NaN if both fail.
    Returns (params, bse)."""
    k = X.shape[1]
    mod = PHReg(t, X, status=e, ties="breslow")
    for method in ("newton", "bfgs"):
        try:
            with np.errstate(all="ignore"):
                r = mod.fit(method=method, disp=False)
            b, s = np.asarray(r.params), np.asarray(r.bse)
            if np.isfinite(b).all() and np.isfinite(s).all() and np.abs(b).max() < 20:
                return b, s
        except Exception as ex:                                 # noqa: BLE001
            last = f"{type(ex).__name__}: {ex}"
        else:
            last = "coefficients NaN / diverged"
    warn(f"Cox fit failed (newton and bfgs): {last}")
    return np.full(k, np.nan), np.full(k, np.nan)


def run_v2(p, cfg, fam, ck, clocks, mea, dod, oh_hr):
    vc = cfg["validation"]
    inst = FAMILIES[fam]["inst"]
    cens = pd.Timestamp(cfg["measure"]["death_censor"])
    eids = ck["eid"].to_numpy()
    t, e = survival(eids, mea.reindex(eids)[f"date_i{inst}"].to_numpy(), dod, cens)
    age = ck["age"].to_numpy("float64")
    fem = (ck["sex"] == "Female").to_numpy("float64")
    rows = []
    for c in clocks:
        g = ck[f"gapz__{c}"].to_numpy("float64")
        m = np.isfinite(t) & np.isfinite(g) & np.isfinite(age)
        ne = int(e[m].sum())
        row = {"family": fam, "clock": c, "n": int(m.sum()), "events": ne,
               "fu_median": float(np.nanmedian(t[m])) if m.any() else np.nan}
        if ne < vc["v2_min_events"]:
            rows.append({**row, "hr": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "v2": None,
                         "v2_note": f"{ne} deaths < v2_min_events = {vc['v2_min_events']}"})
            continue
        b, s = cox(t[m], e[m], np.column_stack([g[m], age[m], fem[m]]))
        z = b[0] / s[0] if s[0] > 0 else np.nan
        pv = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
        passed = bool(np.isfinite(pv) and b[0] > 0 and pv < vc["v2_alpha"])
        rows.append({**row, "hr": float(np.exp(b[0])), "lo": float(np.exp(b[0] - 1.959964 * s[0])),
                     "hi": float(np.exp(b[0] + 1.959964 * s[0])), "p": pv, "v2": passed,
                     "v2_note": "" if passed else ("wrong direction" if b[0] <= 0 else "not significant"),
                     "paper_hr": oh_hr.get(c, np.nan) if fam == "C-P" else np.nan})
    return rows


# ── V1 / V7 ────────────────────────────────────────────────────────
def run_v1_v7(p, cfg, fam, clocks):
    vc = cfg["validation"]
    out = {c: {"v1": None, "v1_note": "", "v7": None, "v7_note": "", "perf_r": np.nan, "perf_mae": np.nan}
           for c in clocks}
    if fam == "C-P":
        pf, cv = read_table(p, "a1_CP_perf.csv"), read_table(p, "a1_CP_coverage.csv")
        for r in (pf.itertuples() if pf is not None else []):
            if r.clock in out:
                ok_ = bool(r.r >= r.benchmark_r - vc["v1_r_tol"])
                out[r.clock].update(v1=ok_, perf_r=r.r, perf_mae=r.mae, benchmark=r.benchmark_r,
                                    v1_note="" if ok_ else f"r {r.r:.3f} < benchmark {r.benchmark_r:.3f} - {vc['v1_r_tol']}")
        cov = cv
    elif fam == "C-C":
        pf, br = read_table(p, "a1_CC_perf.csv"), read_table(p, "a1_CC_branch.csv")
        brm = dict(zip(br["clock"], br["branch"])) if br is not None else {}
        for r in (pf.itertuples() if pf is not None else []):
            if r.clock in out:
                b = brm.get(r.clock, "missing")
                out[r.clock].update(v1=b == "use_weights", perf_r=r.r, perf_mae=r.mae, benchmark=r.paper_test_r,
                                    v1_note="" if b == "use_weights" else f"accuracy check = {b}")
        cov = pf
    elif fam == "C-M":
        pf = read_table(p, "a2_CM_perf.csv")
        for r in (pf.itertuples() if pf is not None else []):
            if r.clock in out:
                ok_ = bool(r.mae_oob <= r.paper_mae + vc["v1_mae_tol"])
                out[r.clock].update(v1=ok_, perf_r=r.r_oob, perf_mae=r.mae_oob, benchmark=r.paper_mae,
                                    v1_note="" if ok_ else f"MAE {r.mae_oob:.2f} > published {r.paper_mae} + {vc['v1_mae_tol']}")
        cov = None
    elif fam == "C-L":
        pf = read_table(p, "a1b_CL_perf.csv")
        for r in (pf.itertuples() if pf is not None else []):
            if r.clock in out:
                out[r.clock].update(perf_r=r.r, perf_mae=r.mae,
                                    v1_note=f"not applicable ({r.train_note}); r / MAE only, units {r.units}")
        cov = None
    else:                                                        # C-B: own cross-validated performance
        pf = read_table(p, "a2_CB_perf.csv")
        if pf is not None:
            for c, g in pf.groupby("clock"):
                if c in out:
                    w = g["n_cn"] / g["n_cn"].sum()
                    out[c].update(perf_r=float((g["r_oob"] * w).sum()), perf_mae=float((g["mae_oob"] * w).sum()),
                                  v1_note="no published benchmark; own CV performance (CN-weighted over sexes)")
        cov = None
    if cov is not None and "coverage" in cov.columns:
        for r in cov.itertuples():
            if r.clock in out:
                ok_ = bool(r.coverage >= vc["v7_coverage_min"] and r.missing_weight_frac <= vc["v7_missing_weight_max"])
                out[r.clock].update(v7=ok_, coverage=r.coverage, missing_weight_frac=r.missing_weight_frac,
                                    v7_note="" if ok_ else f"coverage {r.coverage:.2f} / missing weight {r.missing_weight_frac:.2f}")
    return out


def published_hr(cfg) -> dict:
    """Oh et al. supplementary table ST9: HR of each organ age gap for all-cause mortality."""
    f = ref_file(cfg, cfg["protein"]["oh"]["st9"])
    if f is None:
        warn(f"{cfg['protein']['oh']['st9']} not found; published HRs not reported")
        return {}
    try:
        st9 = pd.read_excel(f, sheet_name="ST9_agegap_vs_mortality_cox_sta")
    except Exception as ex:                                      # noqa: BLE001
        warn(f"cannot read ST9 from {f.name}: {ex}")
        return {}
    return dict(zip(st9["organ"], st9["exp(coef)"]))


# ── main ───────────────────────────────────────────────────────────
def main():
    args, cfg, up, p = setup(__doc__)
    vc = cfg["validation"]
    with tee(p, "a3_validate"):
        h1("a3 clock validation: V1 / V2 / V7")
        mea = load_measure(p).set_index("eid")
        dod = death_dates(p, cfg)
        oh_hr = published_hr(cfg)
        status = {"_meta": {"thresholds": dict(vc)}}
        t1, v2_rows = [], []
        for fam in FAMILIES:
            ck, clocks = load_clocks(p, fam, required=False)
            if ck is None or not clocks or not len(ck):
                warn(f"{fam}: {clock_file(p, fam).name} missing or empty; family skipped")
                status[fam] = {"_family": {"present": False, "exit": True, "reason": "clock file missing"}}
                continue
            h2(f"{fam}: {FAMILIES[fam]['label']} ({len(clocks)} clocks, n = {len(ck):,})")
            v17 = run_v1_v7(p, cfg, fam, clocks)
            r2 = run_v2(p, cfg, fam, ck, clocks, mea, dod, oh_hr)
            v2_rows += r2
            d2 = {r["clock"]: r for r in r2}
            fam_st = {}
            for c in clocks:
                a, b = v17[c], d2[c]
                reasons = [x for x in (a["v1_note"] if a["v1"] is False else "",
                                       b["v2_note"] if not b["v2"] else "",
                                       a["v7_note"] if a["v7"] is False else "") if x]
                main_ = (a["v1"] is not False) and bool(b["v2"]) and (a["v7"] is not False)
                fam_st[c] = {"v1": a["v1"], "v2": b["v2"], "v7": a["v7"],
                             "hr": b.get("hr"), "p_v2": b.get("p"), "main": main_, "reasons": reasons}
                t1.append({"family": fam, "clock": c, "arm": FAMILIES[fam]["arm"],
                           "timing": FAMILIES[fam]["timing"], "n": b["n"],
                           "r": a["perf_r"], "mae": a["perf_mae"], "benchmark": a.get("benchmark", np.nan),
                           "v1": a["v1"], "coverage": a.get("coverage", np.nan),
                           "missing_weight_frac": a.get("missing_weight_frac", np.nan), "v7": a["v7"],
                           "events": b["events"], "hr": b.get("hr"), "hr_lo": b.get("lo"), "hr_hi": b.get("hi"),
                           "p_v2": b.get("p"), "paper_hr": b.get("paper_hr", np.nan), "v2": b["v2"],
                           "main": main_,
                           "reasons": "; ".join(fam_st[c]["reasons"])})
            exit_ = vc["family_exit_on_v2"] and not any(bool(d2[c]["v2"]) for c in clocks)
            if exit_:
                warn(f"{fam}: no clock passes V2; family flagged")
                for c in clocks:
                    fam_st[c]["main"] = False
                    fam_st[c]["reasons"].append("family flagged (no clock passes V2)")
                for r in t1:
                    if r["family"] == fam:
                        r["main"] = False
            fam_st["_family"] = {"present": True, "exit": bool(exit_), "n_clocks": len(clocks),
                                 "n_main": int(sum(fam_st[c]["main"] for c in clocks)),
                                 "timing": FAMILIES[fam]["timing"]}
            status[fam] = fam_st
            print(f"    V1 passed {sum(v17[c]['v1'] is not False for c in clocks)}/{len(clocks)}  "
                  f"V2 passed {sum(bool(d2[c]['v2']) for c in clocks)}/{len(clocks)}  "
                  f"-> all criteria {fam_st['_family']['n_main']}")

        T1 = pd.DataFrame(t1)
        T1.to_csv(p["tables"] / "a3_T1_validation.csv", index=False)
        pd.DataFrame(v2_rows).to_csv(p["tables"] / "a3_v2_cox.csv", index=False)
        status["_meta"]["sha256_T1"] = sha256_of(T1.round(ROUND).astype(str).to_dict("records"))
        write_json(p["derived"] / "clock_status.json", clean(status))
        if len(T1):
            print(T1.groupby("family", sort=False)[["main"]].sum().astype(int).to_string())
        ok("a3 done -> clock_status.json")


if __name__ == "__main__":
    main()
