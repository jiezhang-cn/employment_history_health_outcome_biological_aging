"""Shared machinery for the prospective analyses (s0-s3).

Time scale, person-time splitting, piecewise-exponential rate models,
Aalen-Johansen restricted mean time lost, mean cumulative function and
bootstrap helpers.

Design notes
------------
* Time scale: attained age with delayed entry (left truncation) at time zero,
  so every risk set contains people of the same age and age is absorbed by
  the baseline hazard.
* Rate models: piecewise-exponential models on 5-year attained-age bands,
  fitted as Poisson regressions with log person-years as offset. The two
  likelihoods differ only by a constant, and with all covariates categorical
  the data can be aggregated by (band x covariate pattern x type indicator)
  without loss (sufficient statistics), which makes K "type vs rest" fits per
  outcome feasible.
* Variance: for a single first event the Poisson likelihood is the true
  likelihood, so the model variance is used. For recurrent events (s3) a
  person-clustered sandwich is used; its meat must be computed on individual
  rows because aggregated cells span persons. Firth and sandwich are never
  combined.
* Competing risks: cumulative incidence uses Aalen-Johansen with death as a
  competing event; 1-KM would overstate risk, more so in types with higher
  mortality.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

from common_out import (Z975, bh_fdr, fail, independent_cols, irls_poisson,
                        pick_reference, warn, z_to_p)
from common_out import _dummies as _cat_dummies


ONSET = "op_onset.parquet"
DAYS_PER_YEAR = 365.25
# Tolerance for age comparisons at window boundaries (dates are daily, so
# 1e-6 years is far below any meaningful resolution; it absorbs float error).
EPS_YEAR = 1e-6


# -- inputs: per-person time line ------------------------------------------
def load_person(p) -> pd.DataFrame:
    f = p["derived"] / "op_person.parquet"
    if not f.exists():
        fail(f"missing {f.name}: run o1_assemble.py first")
        sys.exit(1)
    return pd.read_parquet(f).set_index("eid")


def load_onset(p) -> pd.DataFrame:
    """First-onset long table written by s0: (row, code, age, post_tz, src)."""
    f = p["derived"] / ONSET
    if not f.exists():
        fail(f"missing {f.name}: run s0_onset.py first")
        sys.exit(1)
    d = pd.read_parquet(f)
    d["code"] = d["code"].astype(str)
    return d


def dx_window_end(p) -> pd.Timestamp:
    """Administrative end of disease follow-up: the right end of the window
    common to all diagnosis sources, as determined by o1. It differs from the
    end of mortality follow-up and must not be mixed with it."""
    f = p["derived"] / "op_disease" / "meta.json"
    if not f.exists():
        fail("missing op_disease/meta.json: run o1_assemble.py --stage disease")
        sys.exit(1)
    return pd.Timestamp(json.loads(f.read_text(encoding="utf-8"))
                        ["incident_window_end"])


def timeline(person: pd.DataFrame, cfg, dx_end: pd.Timestamp) -> pd.DataFrame:
    """Per-person entry and exit on the attained-age scale.

    Disease and mortality follow-up have different administrative ends, so
    there are two exit ages. `censor_date` (mortality side) is computed by o1
    as min(administrative end, loss to follow-up).
    """
    b = pd.to_datetime(person["birth_date"])
    tz = pd.to_datetime(person["tz_date"])
    d = pd.DataFrame(index=person.index)
    d["tz_age"] = (tz - b).dt.days / DAYS_PER_YEAR
    dth = pd.to_datetime(person["death_date"])
    d["death_age"] = (dth - b).dt.days / DAYS_PER_YEAR
    end_dx = pd.concat([pd.Series(dx_end, index=person.index),
                        pd.to_datetime(person["lost_date"]), dth],
                       axis=1).min(axis=1)
    end_dt = pd.concat([pd.to_datetime(person["censor_date"]), dth],
                       axis=1).min(axis=1)
    d["exit_dx"] = (end_dx - b).dt.days / DAYS_PER_YEAR
    d["exit_death"] = (end_dt - b).dt.days / DAYS_PER_YEAR
    d["died"] = person["died"].to_numpy()
    # whether disease follow-up ended by death (the competing event)
    d["dx_exit_is_death"] = (np.isfinite(d["death_age"])
                             & (np.abs(d["exit_dx"] - d["death_age"]) < EPS_YEAR))
    d["entry"] = d["tz_age"]
    bad = ~np.isfinite(d["entry"]) | ~np.isfinite(d["exit_death"])
    if bad.any():
        warn(f"{int(bad.sum()):,} persons without computable entry/exit "
             "(missing birth or censoring date): excluded")
    short = (d["exit_death"] <= d["entry"]) & ~bad
    if short.any():
        warn(f"{int(short.sum()):,} persons with follow-up <= 0: excluded")
    d["usable"] = ~bad & ~short
    return d


# -- covariate design -------------------------------------------------------
def cov_design(df: pd.DataFrame, cfg: dict) -> dict:
    """Categorical covariates used for aggregation: questionnaire year,
    assessment centre (or nation), ethnic group, highest qualification.

    No age-at-entry term: on the attained-age scale age is absorbed by the
    banded baseline hazard. Level merging and missing-as-level follow
    `common_out._dummies`.
    """
    ga = cfg["covariates"]["group_a"]
    min_n = cfg["covariates"]["min_level_n"]
    mal = cfg["covariates"]["missing_as_level"]
    n = len(df)
    X, names, refs = [], [], {}
    if ga.get("entry_year", True):
        X.append((df["tz_year"].to_numpy("float64") > 2015).reshape(-1, 1))
        names.append("tz_year>2015")
    cat = list(ga["categorical"])
    if ga.get("region") == "centre":
        cat += ["centre"]
    elif ga.get("region") == "nation":
        cat += ["nation"]
    for c in cat:
        if c not in df.columns:
            warn(f"design: column {c} missing, covariate set incomplete")
            continue
        Xc, nm, ref = _cat_dummies(df[c], c, min_n, mal)
        if Xc is None:
            warn(f"{c} has a single usable level: skipped")
            continue
        X.append(Xc)
        names += nm
        refs[c] = ref
    if not X:
        return {"X": np.zeros((n, 0)), "names": [], "refs": {}, "dropped": []}
    Xm = np.column_stack(X)
    keep, drop = independent_cols(np.column_stack([np.ones(n), Xm]),
                                  cfg["covariates"]["design_rank_tol"])
    keep = [j - 1 for j in keep if j > 0]
    return {"X": np.ascontiguousarray(Xm[:, keep]),
            "names": [names[j] for j in keep], "refs": refs,
            "dropped": [names[j - 1] for j in drop if j > 0]}


def cov_patterns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix -> (pattern index per person, pattern matrix).

    Aggregation is exact only if the design is constant within a cell, so
    patterns are defined by the design matrix itself.
    """
    if X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype="int64"), np.zeros((1, 0))
    P, inv = np.unique(X, axis=0, return_inverse=True)
    return np.asarray(inv).ravel().astype("int64"), P


# -- person-time splitting ----------------------------------------------------
def band_edges(cfg, entry, exit_) -> np.ndarray:
    """Attained-age band edges covering the data (empty end bands removed)."""
    sv = cfg["surv"]
    e = np.arange(float(sv["band_min"]),
                  float(sv["band_max"]) + 1e-9, float(sv["model_band"]))
    lo, hi = float(np.nanmin(entry)), float(np.nanmax(exit_))
    if lo < e[0] or hi > e[-1]:
        fail(f"age bands [{e[0]:.0f}, {e[-1]:.0f}] do not cover the data "
             f"[{lo:.1f}, {hi:.1f}]: widen surv.band_min/band_max")
        sys.exit(1)
    i0 = max(0, int(np.searchsorted(e, lo, side="right") - 1))
    i1 = min(len(e) - 1, int(np.searchsorted(e, hi, side="left")))
    return e[i0:i1 + 1]


def split_bands(entry, exit_, event, edges):
    """Person x band -> (person index, band index, person-years, events),
    keeping only pieces with positive person-time. The event is placed in the
    band containing the exit time."""
    lo = np.maximum(entry[:, None], edges[None, :-1])
    hi = np.minimum(exit_[:, None], edges[None, 1:])
    pt = np.clip(hi - lo, 0.0, None)
    nb = len(edges) - 1
    b_ev = np.clip(np.searchsorted(edges, exit_, side="right") - 1, 0, nb - 1)
    ev = np.zeros_like(pt)
    ev[np.arange(len(entry)), b_ev] = np.asarray(event, dtype="float64")
    m = pt > 0
    r, b = np.nonzero(m)
    lost = float(ev.sum() - ev[m].sum())
    if lost > 0:
        warn(f"{lost:.0f} events fall in zero person-time pieces: dropped")
    return r.astype("int64"), b.astype("int64"), pt[m], ev[m]


def aggregate(b, strata_r, ind_r, pt, ev, nb, ns):
    """Aggregate by (band x covariate pattern x type indicator); exact for
    Poisson. A binary type indicator suffices for "type vs rest"."""
    key = (b * ns + strata_r) * 2 + ind_r
    n = nb * ns * 2
    PT = np.bincount(key, weights=pt, minlength=n)
    EV = np.bincount(key, weights=ev, minlength=n)
    idx = np.flatnonzero(PT > 0)
    return (idx // (ns * 2)).astype("int64"), ((idx // 2) % ns).astype("int64"), \
        (idx % 2).astype("float64"), PT[idx], EV[idx]


def _cell_design(bb, ss, tt, P, nb, band_ref):
    """Cell design: intercept + band dummies (reference = band with most
    person-time) + covariate pattern columns + type indicator (last)."""
    cols = [np.ones(len(bb))]
    for j in range(nb):
        if j == band_ref:
            continue
        cols.append((bb == j).astype("float64"))
    if P.shape[1]:
        cols.append(P[ss])
    cols.append(tt.reshape(-1, 1))
    return np.column_stack([c.reshape(len(bb), -1) for c in cols])


# -- rate model: each type vs the rest ---------------------------------------
def _pois_ll(y, eta) -> float:
    """Poisson log-likelihood up to terms free of beta (cancel in LR tests)."""
    return float(np.sum(y * eta - np.exp(eta)))


def rate_ratio_rest(b, strata_r, typ_r, pt, ev, P, nb, K, cfg, meta_row,
                    person_r=None, ph=False) -> list:
    """K rows: adjusted rate ratio of each type versus all other types.

    `person_r` (recurrent events only) switches to a person-clustered sandwich
    SE. `ph=True` adds the LR test of a type x age-band interaction, the
    piecewise-exponential analogue of a proportional-hazards check.
    """
    sv = cfg["surv"]
    ns = int(P.shape[0]) if P.shape[1] else 1
    pt_band = np.bincount(b, weights=pt, minlength=nb)
    band_ref = int(np.argmax(pt_band))
    out = []
    for k in range(K):
        ind = (typ_r == k)
        ev_k, pt_k = float(ev[ind].sum()), float(pt[ind].sum())
        ev_r, pt_r = float(ev[~ind].sum()), float(pt[~ind].sum())
        base = {**meta_row, "type_index": k, "events_type": ev_k,
                "pyears_type": pt_k, "events_rest": ev_r, "pyears_rest": pt_r,
                "rate_type": ev_k / pt_k * 1000 if pt_k else np.nan,
                "rate_rest": ev_r / pt_r * 1000 if pt_r else np.nan,
                "n_band": nb, "band_ref": band_ref}
        nan = {"log_rr": np.nan, "se": np.nan, "rr": np.nan, "lo": np.nan,
               "hi": np.nan, "z": np.nan, "p": np.nan, "q": np.nan,
               "firth": False, "converged": False, "ph_p": np.nan}
        if pt_k <= 0 or pt_r <= 0 or (ev_k + ev_r) <= 0:
            out.append({**base, **nan})
            continue
        bb, ss, tt, PT, EV = aggregate(b, strata_r, ind.astype("int64"),
                                       pt, ev, nb, ns)
        X0 = _cell_design(bb, ss, tt, P, nb, band_ref)
        rk, ck, dropped = drop_uninformative(X0, PT, EV)
        X, PT, EV = X0[np.ix_(rk, ck)], PT[rk], EV[rk]
        bb, tt = bb[rk], tt[rk]
        base["cells"] = int(len(EV))
        base["cells_dropped"] = dropped["cells"]
        base["pyears_dropped"] = dropped["pyears"]
        # after dropping zero-event strata the type indicator may be constant
        # (not estimable) or one side may have no events (needs Firth)
        ev_k2 = float((EV * X[:, -1]).sum())
        ev_r2 = float(EV.sum() - ev_k2)
        if not len(EV) or len(np.unique(X[:, -1])) < 2 or EV.sum() <= 0:
            out.append({**base, **nan})
            continue
        firth = bool(min(ev_k, ev_r) < sv["firth_below"]
                     or min(ev_k2, ev_r2) < sv["firth_below"])
        off = np.log(np.clip(PT, 1e-12, None))
        r = irls_poisson(EV, X, offset=off, firth=firth, robust=False)
        bi, si = float(r["beta"][-1]), float(r["se"][-1])
        if person_r is not None and not firth:
            si = _cluster_se(b, strata_r, ind, pt, ev, P, nb, band_ref,
                             r["beta"], r["cov"], person_r, ck)
        zi = bi / si if si > 0 else np.nan
        row = {**base, "log_rr": bi, "se": si, "rr": float(np.exp(bi)),
               "lo": float(np.exp(bi - Z975 * si)),
               "hi": float(np.exp(bi + Z975 * si)), "z": zi,
               "p": float(z_to_p(np.array([zi]))[0]), "q": np.nan,
               "firth": firth, "converged": r["converged"], "ph_p": np.nan}
        if ph and nb > 1:
            row["ph_p"] = _ph_lr(X, EV, off, tt, bb, nb, band_ref, r)
        out.append(row)
    return out


def drop_uninformative(X, PT, EV):
    """Drop cells and columns of zero-event strata.

    If a 0/1 dummy has no events, its profile MLE is -inf and IRLS does not
    converge; the remaining coefficients' MLE equals the MLE after removing
    that stratum (as in Cox, risk sets without events do not enter the
    partial likelihood). Iterated because removing rows can create new
    zero-event columns. The intercept and the type indicator (last column)
    are never dropped; a zero-event type is handled by Firth.

    Returns row and column masks so the caller can filter auxiliary arrays.
    """
    X = np.asarray(X, dtype="float64")
    PT = np.asarray(PT, dtype="float64")
    EV = np.asarray(EV, dtype="float64")
    rows = np.ones(len(EV), dtype=bool)
    cols = np.ones(X.shape[1], dtype=bool)
    is01 = np.all((X == 0) | (X == 1), axis=0)
    for _ in range(20):
        if not rows.any():
            break
        ev_col = X[rows].T @ EV[rows]
        n_col = X[rows].sum(0)
        bad = np.flatnonzero(is01 & (ev_col <= 0) & (n_col > 0) & cols)
        bad = bad[(bad > 0) & (bad < X.shape[1] - 1)]
        if not bad.size:
            break
        rows = rows & ~(X[:, bad] > 0).any(1)
        cols[bad] = False
    if rows.any():
        cols = cols & ((X[rows].sum(0) > 0)
                       | (np.arange(X.shape[1]) == X.shape[1] - 1))
        cols[0] = True
    return rows, cols, {"cells": int((~rows).sum()),
                        "pyears": float(PT[~rows].sum())}


def _ph_lr(X, EV, off, tt, bb, nb, band_ref, red) -> float:
    """LR test of type x age-band interaction (not a Schoenfeld test)."""
    inter = [((bb == j) & (tt > 0)).astype("float64") for j in range(nb)
             if j != band_ref]
    inter = [c for c in inter if c.sum() > 0]
    if not inter:
        return np.nan
    Xf = np.column_stack([X] + inter)
    keep, _ = independent_cols(Xf, 1e-10, protect=X.shape[1])
    Xf = Xf[:, keep]
    df = Xf.shape[1] - X.shape[1]
    if df <= 0:
        return np.nan
    full = irls_poisson(EV, Xf, offset=off, robust=False)
    ll1 = _pois_ll(EV, Xf @ full["beta"] + off)
    ll0 = _pois_ll(EV, X @ red["beta"] + off)
    stat = max(0.0, 2.0 * (ll1 - ll0))
    return float(chi2.sf(stat, df))


def _cluster_se(b, strata_r, ind, pt, ev, P, nb, band_ref, beta, cov, person_r,
                ck=None):
    """Person-clustered sandwich SE of the type coefficient (recurrent events).

    Point estimate and bread come from the aggregated fit; the meat
    sum_person (sum_band x (y - mu))^2 is accumulated over individual rows,
    in chunks to bound memory. Rows in strata dropped by `drop_uninformative`
    are removed so the meat matches the fitted likelihood.
    """
    p = len(beta)
    G = int(person_r.max()) + 1
    S = np.zeros((G, p))
    logpt = np.log(np.clip(pt, 1e-12, None))
    for i0 in range(0, len(b), 200_000):
        sl = slice(i0, i0 + 200_000)
        Xc = _cell_design(b[sl], strata_r[sl], ind[sl].astype("float64"),
                          P, nb, band_ref)
        keep = np.ones(len(Xc), dtype=bool)
        if ck is not None:
            gone = np.flatnonzero(~np.asarray(ck))
            gone = gone[gone < Xc.shape[1] - 1]
            if gone.size:
                keep = ~(Xc[:, gone] > 0).any(1)
            Xc = Xc[:, np.asarray(ck)]
        Xc = Xc[keep]
        mu = np.exp(np.clip(Xc @ beta, -30.0, 30.0) + logpt[sl][keep])
        res = ev[sl][keep] - mu
        gi = person_r[sl][keep]
        for j in range(p):
            S[:, j] += np.bincount(gi, weights=Xc[:, j] * res, minlength=G)
    V = cov @ (S.T @ S) @ cov
    return float(math.sqrt(max(float(V[-1, -1]), 0.0)))


# -- restricted mean time lost by cause (Andersen 2013) ----------------------
def aj_rmtl(entry, exit_, cause, w, a0, a1, n_cause) -> dict:
    """Weighted Aalen-Johansen -> restricted mean time lost to each cause
    between ages a0 and a1, among those event-free at a0.

    `cause`: 0 = censored, 1..n_cause = the first event that ended
    event-free life. Entry is truncated at a0 (left truncation handled by the
    risk set), events after a1 are censored. Then

        RMTL_c = int_{a0}^{a1} F_c(t) dt,  RMST = (a1 - a0) - sum_c RMTL_c

    and the second identity is exact under Aalen-Johansen, so cause-specific
    contributions add up to the total difference. `w` = person weights
    (all ones = unadjusted).
    Returns {"rmtl": (C,), "rmst", "n_a0", "n_a1", "truncated"}; truncated
    means the risk set emptied before a1.
    """
    e = np.maximum(np.asarray(entry, "float64"), a0)
    x = np.minimum(np.asarray(exit_, "float64"), a1)
    c = np.where(np.asarray(exit_, "float64") > a1, 0, np.asarray(cause, "int64"))
    w = np.asarray(w, "float64")
    keep = x > e
    e, x, c, w = e[keep], x[keep], c[keep], w[keep]
    n_a0, n_a1 = int((e <= a0).sum()), int((x >= a1).sum())
    out = {"rmtl": np.zeros(n_cause), "rmst": float(a1 - a0), "n_a0": n_a0,
           "n_a1": n_a1, "truncated": n_a1 == 0}
    ev = c > 0
    if not ev.any():
        return out
    t = np.unique(x[ev])
    oe, ox = np.argsort(e), np.argsort(x)
    ce = np.concatenate([[0.0], np.cumsum(w[oe])])
    cx = np.concatenate([[0.0], np.cumsum(w[ox])])
    # risk set = sum w[entry < t] - sum w[exit < t] (left truncation)
    Y = (ce[np.searchsorted(e[oe], t, side="left")]
         - cx[np.searchsorted(x[ox], t, side="left")])
    J = len(t)
    pos = np.searchsorted(t, x[ev])
    dN = np.bincount(pos * n_cause + (c[ev] - 1), weights=w[ev],
                     minlength=J * n_cause).reshape(J, n_cause)
    Y = np.clip(Y, 1e-12, None)
    S = np.cumprod(1.0 - dN.sum(1) / Y)
    Sprev = np.concatenate([[1.0], S[:-1]])
    F = np.cumsum(Sprev[:, None] * dN / Y[:, None], axis=0)      # time x cause
    dt = np.diff(np.concatenate([t, [a1]]))
    out["rmtl"] = (F * dt[:, None]).sum(0)
    out["rmst"] = float(a1 - a0 - out["rmtl"].sum())
    return out


# -- mean cumulative function -------------------------------------------------
def _risk_sets(entry, exit_, cause):
    """(event times, risk set, events of cause 1, events of cause 2) under
    left truncation: risk set = #{entry < t} - #{exit < t}."""
    t = np.unique(exit_[cause > 0])
    if not t.size:
        return t, t, t, t
    es, xs = np.sort(entry), np.sort(exit_)
    n_risk = (np.searchsorted(es, t, side="left")
              - np.searchsorted(xs, t, side="left")).astype("float64")
    pos = np.searchsorted(t, exit_)
    d1 = np.bincount(pos[cause == 1], minlength=len(t))[:len(t)].astype("float64")
    d2 = np.bincount(pos[cause == 2], minlength=len(t))[:len(t)].astype("float64")
    return t, n_risk, d1, d2


def mcf(entry, exit_, died, ev_age, grid):
    """Mean cumulative number of new diagnoses by attained age.

    Ghosh-Lin / Cook-Lawless estimator: death is a terminating event (the
    Nelson-Aalen increments are multiplied by the Kaplan-Meier probability of
    being alive), so the curve estimates the expected count actually
    accumulated rather than the count if nobody died.
    Returns (mcf on grid, risk set on grid).
    """
    g = np.asarray(grid, dtype="float64")
    entry = np.asarray(entry, float)
    exit_ = np.asarray(exit_, float)
    ev_age = np.asarray(ev_age, float)
    ev_age = ev_age[np.isfinite(ev_age)]
    if not ev_age.size:
        return np.zeros_like(g), np.zeros_like(g)
    u, cnt = np.unique(ev_age, return_counts=True)
    es, xs = np.sort(entry), np.sort(exit_)
    Y = (np.searchsorted(es, u, side="left")
         - np.searchsorted(xs, u, side="left")).astype("float64")
    Sprev = np.ones(len(u))
    td, nd, dd, _ = _risk_sets(entry, exit_, np.asarray(died, int))
    if td.size:
        Sd = np.cumprod(1.0 - dd / np.clip(nd, 1.0, None))
        jd = np.searchsorted(td, u, side="left") - 1
        Sprev = np.where(jd >= 0, Sd[np.clip(jd, 0, len(td) - 1)], 1.0)
    M = np.cumsum(np.where(Y > 0, Sprev * cnt / np.clip(Y, 1.0, None), 0.0))
    j = np.searchsorted(u, g, side="right") - 1
    out = np.where(j >= 0, M[np.clip(j, 0, len(u) - 1)], 0.0)
    nrisk = (np.searchsorted(es, g, side="left")
             - np.searchsorted(xs, g, side="left")).astype("float64")
    return out, nrisk


def cross_age(grid, y, level) -> float:
    """First age at which a curve reaches `level` (linear interpolation);
    NaN if never reached (never the end point of the grid)."""
    g, y = np.asarray(grid, float), np.asarray(y, float)
    m = np.isfinite(y) & np.isfinite(g)
    if m.sum() < 2 or np.nanmax(y[m]) < level or not np.isfinite(level):
        return float("nan")
    i = int(np.argmax((y >= level) & m))
    if i == 0:
        return float(g[0])
    y0, y1 = y[i - 1], y[i]
    if not np.isfinite(y0) or y1 <= y0:
        return float(g[i])
    return float(g[i - 1] + (level - y0) / (y1 - y0) * (g[i] - g[i - 1]))


def boot_cross(fn, n, level, grid, n_boot, seed) -> tuple[float, float]:
    """Percentile bootstrap CI of `cross_age`; `fn(idx)` returns the curve
    for a resample of person indices."""
    if n_boot <= 0 or n < 20:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.full(n_boot, np.nan)
    for i in range(n_boot):
        vals[i] = cross_age(grid, fn(rng.integers(0, n, n)), level)
    v = vals[np.isfinite(vals)]
    if v.size < n_boot * 0.5:
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# -- small-cell suppression and FDR -------------------------------------------
def suppress_events(df: pd.DataFrame, min_ev: int) -> pd.DataFrame:
    """Blank event counts and effect estimates of cells with fewer than
    `min_ev` events in the type or in its comparator (UK Biobank small-cell
    rule). Denominators are kept."""
    d = df.copy()
    small = (d["events_type"].fillna(0) < min_ev) | (d["events_rest"].fillna(0) < min_ev)
    d["suppressed"] = small
    for c in ("events_type", "rate_type", "log_rr", "se", "rr", "lo", "hi",
              "z", "p", "q"):
        if c in d.columns:
            d.loc[small, c] = np.nan
    return d


def add_fdr(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Benjamini-Hochberg q values within families defined by surv.fdr_by."""
    sv = cfg["surv"]
    d = df.copy()
    keys = [k for k in sv["fdr_by"] if k in d.columns]
    d["family"] = d[keys].astype(str).agg("|".join, axis=1) if keys else "all"
    for fam, g in d.groupby("family"):
        d.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
    for fam, g in d.groupby("family"):
        n = int(g["p"].notna().sum())
        sig = int((g["q"] < sv["fdr_alpha"]).sum())
        print(f"    {str(fam):<30} m={n:>5}  q<{sv['fdr_alpha']}: {sig:>4}")
    return d


# -- per-sex setup ------------------------------------------------------------
def camsis_of(p, cfg, types_series: pd.Series, sex: str) -> dict:
    """{type: mean CAMSIS}, from the same definition as the O-P analyses."""
    _, info = pick_reference(p, cfg, types_series, sex)
    return {int(r["type"]): r.get("camsis_mean")
            for r in info.get("table", [])}


def sex_setup(p, cfg, person, tl, sex, design=True) -> dict:
    """Rows of one sex, type labels (0..K-1) and covariate patterns.

    Types are indexed in ascending order of mean CAMSIS (missing last). This
    order defines `type_index` and the order of bootstrap draws.
    `design=False` skips the covariate design (unadjusted analyses).
    """
    m = (person["sex"] == sex).to_numpy() & tl["usable"].to_numpy()
    rows = np.flatnonzero(m)
    ps = person.iloc[rows].copy()
    cam = camsis_of(p, cfg, ps["type"], sex)
    raw = ps["type"].to_numpy()
    types = sorted({int(t) for t in raw},
                   key=lambda t: (cam.get(t) if cam.get(t) is not None else np.inf, t))
    lab = np.array([types.index(int(t)) for t in raw], dtype="int64")
    n_of = {int(t): int((raw == t).sum()) for t in types}
    out = {"rows": rows, "person": ps, "tl": tl.iloc[rows], "types": types,
           "lab": lab, "camsis": cam, "n_of": n_of, "sex": sex}
    if design:
        des = cov_design(ps, cfg)
        strata, P = cov_patterns(des["X"])
        out.update({"strata": strata, "P": P, "design": des})
        print(f"  {sex}: n={len(rows):,}  K={len(types)}  covariate columns "
              f"{des['X'].shape[1]} -> {P.shape[0]:,} patterns"
              + (f"; {len(des['dropped'])} collinear columns dropped"
                 if des["dropped"] else ""))
    else:
        print(f"  {sex}: n={len(rows):,}  K={len(types)}")
    return out


def add_type_cols(rs, v) -> list:
    """Attach the type number and its mean CAMSIS to result rows."""
    for x in rs:
        x["type"] = int(v["types"][x["type_index"]])
        c = v["camsis"].get(x["type"])
        x["camsis"] = float(c) if c is not None and np.isfinite(c) else np.nan
    return rs


# Column sets for tables that can be empty (so an empty CSV still has a header)
CURVE_COLS = ("sex", "type", "age", "n_risk", "n_group")
SHIFT_COLS = ("sex", "type", "level", "ref_age", "age_at_level", "shift_years",
              "shift_lo", "shift_hi")


def frame(rows, cols) -> pd.DataFrame:
    """Rows -> DataFrame that keeps its column names when empty."""
    d = pd.DataFrame(rows)
    if not len(d):
        return pd.DataFrame(columns=list(cols))
    return d
