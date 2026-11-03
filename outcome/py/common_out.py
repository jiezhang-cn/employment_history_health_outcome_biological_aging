"""Shared utilities for the health-outcome analyses.

Paths and configuration, ICD-10 chapter mapping, education rebuilt from the
qualifications question, the covariate design matrix, the modified Poisson
estimator (log-link Poisson with robust variance), Benjamini-Hochberg FDR and
small-cell suppression. Cohort definitions, sex split and logging come from the
typology package (`typology/common.py`); this module only adds what the outcome
analyses need. All thresholds and codings live in `outcome/config_outcome.yaml`.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
OUTCOME = HERE.parent                      # outcome/
CODES = OUTCOME.parent                     # repository root

sys.path.insert(0, str(CODES / "typology"))
from common import fail, h1, h2, load_cfg, ok, sexes, tee, warn  # noqa: E402

_ = (tee, h1, h2, ok, warn, fail, sexes)   # re-exported for the scripts


# -- Configuration and paths ------------------------------------------------
def load_cfg_out(path=None) -> dict:
    p = Path(path) if path else OUTCOME / "config_outcome.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cfg_up() -> dict:
    """Typology config at the repository root (sex split, cohort definition)."""
    return load_cfg(CODES / "config.yaml")


def _resolve(entry) -> Path:
    """Relative paths are taken from the repository root."""
    d = Path(entry)
    return d if d.is_absolute() else CODES / d


def _as_list(x) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _first_dir(entry, what, make=False) -> Path:
    cands = [_resolve(c) for c in _as_list(entry)]
    for d in cands:
        if d.is_dir():
            return d
    if make:
        cands[0].mkdir(parents=True, exist_ok=True)
        return cands[0]
    raise FileNotFoundError(f"{what} directory not found: {[str(c) for c in cands]}")


def paths_out(cfg: dict) -> dict:
    p = {"data": _first_dir(cfg["paths"]["data"], "data"),
         "upstream": _first_dir(cfg["paths"]["upstream"], "typology output"),
         "out": _first_dir(cfg["paths"]["out"], "output", make=True)}
    for sub in ("derived", "tables", "qc", "logs"):
        d = p["out"] / sub
        d.mkdir(parents=True, exist_ok=True)
        p[sub] = d
    p["up_derived"] = p["upstream"] / "derived"
    return p


def data_file(p: dict, cfg: dict, key: str, required=True):
    """Locate `files.<key>` in paths.data (then any optional paths.data_extra)."""
    name = str(cfg["files"][key])
    if Path(name).is_absolute():
        f = Path(name)
        if f.exists():
            return f
        if required:
            fail(f"files.{key} not found: {f}")
            sys.exit(1)
        return None
    dirs = [p["data"]] + [_resolve(d) for d in _as_list(cfg["paths"].get("data_extra"))]
    tried = []
    for d in dirs:
        f = Path(d) / name
        tried.append(str(f))
        if f.exists():
            return f
    if required:
        fail(f"{name} not found; tried: " + ", ".join(tried))
        sys.exit(1)
    return None


def ref_file(cfg: dict, name: str, required=False):
    """Published reference table (phecode maps etc.) under paths.refs."""
    dirs = [_resolve(d) for d in _as_list(cfg["paths"].get("refs"))]
    for d in dirs:
        f = d / name
        if f.exists():
            return f
    if required:
        fail(f"reference file {name} not found in {[str(d) for d in dirs]}")
        sys.exit(1)
    return None


def read_records(p: dict, stem="ob_records") -> pd.DataFrame:
    """Read the diagnosis records written by b1 (parquet or csv.gz)."""
    d = Path(p["derived"])
    for f, how in ((d / f"{stem}.parquet", pd.read_parquet),
                   (d / f"{stem}.csv.gz", pd.read_csv)):
        if f.exists():
            df = how(f)
            need = {"eid", "code", "age"}
            if not need <= set(df.columns):
                fail(f"{f.name} lacks columns {need - set(df.columns)}")
                sys.exit(1)
            df["code"] = df["code"].astype(str)
            return df
    fail(f"{stem}.parquet / {stem}.csv.gz missing; run b1_records.py --stage all first")
    sys.exit(1)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


# -- ICD-10 chapters --------------------------------------------------------
_RANGE = re.compile(r"^([A-Z])(\d{2})-([A-Z])(\d{2})$")


def chapter_map(cfg: dict) -> dict:
    """{three-character code: (chapter index, roman numeral, label)} for A00-Q99.

    Chapters cannot be assigned by first letter alone: D spans chapters II
    (D00-D48) and III (D50-D89), and H spans VII (H00-H59) and VIII (H60-H95).
    """
    out = {}
    for i, ch in enumerate(cfg["icd_chapters"], start=1):
        meta = (i, ch["roman"], ch["label"])
        for L in ch.get("letters", []):
            for n in range(100):
                out[f"{L}{n:02d}"] = meta
        for rg in ch.get("ranges", []):
            m = _RANGE.match(rg)
            if not m:
                raise ValueError(f"invalid chapter range: {rg}")
            l1, n1, l2, n2 = m[1], int(m[2]), m[3], int(m[4])
            for L in range(ord(l1), ord(l2) + 1):
                lo = n1 if L == ord(l1) else 0
                hi = n2 if L == ord(l2) else 99
                for n in range(lo, hi + 1):
                    out[f"{chr(L)}{n:02d}"] = meta
    return out


def chapters_of(codes, cfg: dict) -> pd.DataFrame:
    """One row per code with its chapter; unknown codes are reported, not dropped."""
    cmap = chapter_map(cfg)
    # Cancer-registry ICD-9-only records carry no site; they count as chapter II.
    pseudo = cfg.get("cancer", {}).get("icd9_pseudo_code")
    if pseudo:
        cmap[pseudo] = (2, "II", "Neoplasms")
    rows = []
    for c in codes:
        idx, roman, label = cmap.get(c, (99, "?", "unmapped"))
        rows.append({"code": c, "chapter_index": idx, "chapter": roman,
                     "chapter_label": label})
    df = pd.DataFrame(rows)
    bad = df[df["chapter"] == "?"]["code"].tolist()
    if bad:
        warn(f"{len(bad)} codes have no chapter (first 10: {bad[:10]}); "
             "they are kept at code level only")
    return df


def chapters_of_vocab(codes, cfg: dict, p: dict) -> pd.DataFrame:
    """Chapter assignment for a phecode vocabulary via its representative ICD-10 code.

    b1 writes `derived/ob_code_to_icd3.csv` (phecode -> most frequent
    three-character ICD-10 code); without it, codes are treated as ICD-10.
    """
    f = Path(p["derived"]) / "ob_code_to_icd3.csv"
    if not f.exists():
        return chapters_of(codes, cfg)
    proxy = pd.read_csv(f, dtype=str).set_index("code")["icd3"].to_dict()
    df = chapters_of([proxy.get(str(c), str(c)) for c in codes], cfg)
    df["icd3_proxy"] = df["code"]
    df["code"] = [str(c) for c in codes]
    return df


def parse_code_columns(cols, cfg: dict) -> tuple[dict, dict]:
    """First-occurrence column names -> {code: date column} and {code: source column}."""
    dre = re.compile(cfg["first_occ"]["date_re"])
    sre = re.compile(cfg["first_occ"]["source_re"])
    dcol, scol = {}, {}
    for c in cols:
        m = dre.match(c)
        if m:
            dcol[m[1]] = c
            continue
        m = sre.match(c)
        if m:
            scol[m[1]] = c
    return dcol, scol


# -- Education ---------------------------------------------------------------
def build_education(p, cfg, index) -> pd.Series:
    """Highest qualification, rebuilt from all answers to the multiple-choice question.

    Qualifications (field 6138) is multiple choice and stored as pipe-separated
    values ("1|2|3|6"). Mapping only single answers leaves participants with
    several qualifications missing, and they are not missing at random. The
    typology pipeline writes the rebuilt variable to descriptives.parquet; if
    it is absent or incomplete, it is rebuilt here from the raw field with the
    same ranking (config `covariates.edu_rank`).
    """
    cc = cfg["covariates"]
    for src in cc["edu_source"]:
        if src == "descriptives":
            f = p["up_derived"] / "descriptives.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f).set_index("eid")
            if "edu_highest" not in d.columns:
                continue
            s = d["edu_highest"].reindex(index)
            cov = s.notna().mean()
            if cov < cc["edu_min_coverage"]:
                warn(f"edu_highest in descriptives.parquet covers {cov:.1%}; "
                     "rebuilding from the raw field")
                continue
            ok(f"education from descriptives.parquet, coverage {cov:.1%}")
            return s.rename("edu_highest")
        if src == "cov_raw":
            f = data_file(p, cfg, "cov_raw")
            qc = [c for c in pd.read_csv(f, nrows=0).columns
                  if c.startswith("Qualifications__")]
            if not qc:
                continue
            # eid must be int64, otherwise reindexing to the cohort matches nothing.
            d = pd.read_csv(f, usecols=["eid"] + qc,
                            dtype={c: str for c in qc})
            d = d.assign(eid=pd.to_numeric(d["eid"], errors="coerce")
                         ).dropna(subset=["eid"])
            d["eid"] = d["eid"].astype("int64")
            d = d.set_index("eid")
            rank = {int(k): int(v) for k, v in cc["edu_rank"].items()}
            lab = {int(k): v for k, v in cc["edu_labels"].items()}
            # wide -> one row per ticked answer -> highest rank per person
            long = (d[qc].stack().astype(str).str.split("|").explode()
                    .str.strip().rename("code"))
            # nullable Int64 so that the integer keys of `rank` match
            best = (pd.to_numeric(long, errors="coerce").astype("Int64").map(rank)
                    .dropna().groupby(level=0).max())
            s = best.map(lab).reindex(index)
            cov = s.notna().mean()
            (ok if cov >= cc["edu_min_coverage"] else fail)(
                f"education rebuilt from {len(qc)} Qualifications instances, "
                f"coverage {cov:.1%}")
            if cov < cc["edu_min_coverage"]:
                fail("coverage below edu_min_coverage; stopping")
                sys.exit(1)
            return s.rename("edu_highest")
    fail("no usable source for education")
    sys.exit(1)


# -- CAMSIS ranking of types ------------------------------------------------
def pick_reference(p, cfg, types: pd.Series, sex: str) -> tuple[int, dict]:
    """CAMSIS gradient of the career types of one sex.

    All analyses compare each type with all other types of the same sex, so
    there is no reference type. The mean CAMSIS score per type is still
    computed: the survival scripts use it to order types. `types` is indexed
    by eid. Returns (type with the median CAMSIS mean, info dict with the
    per-type table); for an even number of types the lower median is taken.
    """
    vc = types.value_counts()
    f = p["up_derived"] / "camsis.parquet"
    if not f.exists():
        fail(f"{f.name} missing; run typology/01_worklife.py first")
        sys.exit(1)
    cam = pd.read_parquet(f).set_index("eid")
    col = cfg["op"]["reference_camsis_col"]
    if col not in cam.columns:
        cands = [c for c in cam.columns if "camsis" in c.lower()]
        fail(f"camsis.parquet has no column {col} (available: {cands})")
        sys.exit(1)
    v = pd.to_numeric(cam[col], errors="coerce").reindex(types.index)
    tab = (pd.DataFrame({"type": types.to_numpy(), "camsis": v.to_numpy()})
           .groupby("type")["camsis"]
           .agg(camsis_mean="mean", n_camsis="count").reset_index())
    tab["n"] = tab["type"].map(vc).astype(int)
    miss = tab["camsis_mean"].isna()
    if miss.any():
        warn(f"{int(miss.sum())} types have no CAMSIS mean; they are not ranked")
    t = tab[~miss].sort_values(["camsis_mean", "type"]).reset_index(drop=True)
    if not len(t):
        fail("no type has a CAMSIS mean")
        sys.exit(1)
    pos = (len(t) - 1) // 2
    ref = int(t.loc[pos, "type"])
    tab["camsis_rank"] = tab["type"].map(
        {int(r.type): i + 1 for i, r in enumerate(t.itertuples())})
    tab["is_reference"] = False
    print(f"    {sex} CAMSIS gradient (ascending): " + " -> ".join(
        f"{int(r.type)}:{r.camsis_mean:.0f}" for r in t.itertuples()))
    return ref, {"rule": "rest", "ref": None, "camsis_median_type": ref,
                 "k_ranked": int(len(t)), "table": tab.to_dict("records")}


# -- Design matrix ------------------------------------------------------------
def rcs(x, n_knots=4, knots=None):
    """Restricted cubic spline basis (Harrell); n_knots knots -> n_knots-1 columns."""
    x = np.asarray(x, dtype="float64")
    if knots is None:
        qs = {3: [.10, .50, .90], 4: [.05, .35, .65, .95],
              5: [.05, .275, .50, .725, .95]}.get(n_knots)
        if qs is None:
            qs = list(np.linspace(.05, .95, n_knots))
        knots = np.quantile(x[np.isfinite(x)], qs)
    knots = np.unique(np.asarray(knots, dtype="float64"))
    k = len(knots)
    if k < 3:
        return x.reshape(-1, 1), knots
    tk, tk1 = knots[-1], knots[-2]
    den = (tk - tk1)
    cols = [x]
    for j in range(k - 2):
        tj = knots[j]
        b = (np.clip(x - tj, 0, None) ** 3
             - np.clip(x - tk1, 0, None) ** 3 * (tk - tj) / den
             + np.clip(x - tk, 0, None) ** 3 * (tk1 - tj) / den)
        cols.append(b / (tk - knots[0]) ** 2)
    return np.column_stack(cols), knots


def _dummies(s: pd.Series, name: str, min_n: int, missing_as_level: bool):
    """Categorical -> dummy columns (reference = largest level).

    Levels smaller than min_n are merged into "Other"; missing values form an
    explicit "Missing" level so that every outcome uses the same denominator.
    """
    v = s.astype("object")
    v = v.where(~pd.isna(v) & (v.astype(str).str.strip() != ""), np.nan)
    if missing_as_level:
        v = v.astype("object").where(v.notna(), "Missing")
    vc = v.value_counts(dropna=True)
    small = [k for k, n in vc.items() if n < min_n]
    if small:
        v = v.where(~v.isin(small), "Other")
        vc = v.value_counts(dropna=True)
    if len(vc) < 2:
        return None, None, None
    ref = str(vc.index[0])
    lev = [str(k) for k in vc.index[1:]]
    X = np.column_stack([(v.astype(str) == L).to_numpy("float64") for L in lev])
    return X, [f"{name}={L}" for L in lev], ref


def independent_cols(X, tol=1e-8, protect=1) -> list:
    """Linearly independent columns by incremental Cholesky on the correlation matrix.

    Age spline, entry year and centre dummies can be nearly collinear (most
    questionnaires were completed in one year, so age at time zero is close to
    a function of birth year). Collinear columns are dropped and named rather
    than left to produce large offsetting coefficients. The first `protect`
    columns are always kept.
    """
    p = X.shape[1]
    G = X.T @ X
    d = np.sqrt(np.clip(np.diag(G), 1e-300, None))
    R = G / np.outer(d, d)
    keep, dropped = [], []
    for j in range(p):
        if np.diag(G)[j] <= 0:
            dropped.append(j)
            continue
        if not keep:
            keep.append(j)
            continue
        A = R[np.ix_(keep, keep)]
        b = R[np.ix_(keep, [j])]
        try:
            c = np.linalg.solve(A + 1e-12 * np.eye(len(keep)), b)
        except np.linalg.LinAlgError:
            dropped.append(j)
            continue
        resid = float(R[j, j] - (b[:, 0] @ c[:, 0]))
        if resid > tol or j < protect:
            keep.append(j)
        else:
            dropped.append(j)
    return keep, dropped


def build_design(df: pd.DataFrame, cfg: dict, extra=()) -> dict:
    """Covariate design matrix. Returns {X, names, knots, dropped, refs}.

    Intercept, restricted cubic spline of age at time zero, questionnaire year
    (after 2015), ethnic group, highest qualification and assessment centre.
    Career-type indicators are appended by the caller (the permutation test
    shuffles them). `extra` adds continuous columns.
    """
    ga = cfg["covariates"]["group_a"]
    min_n = cfg["covariates"]["min_level_n"]
    mal = cfg["covariates"]["missing_as_level"]
    n = len(df)
    X = [np.ones((n, 1))]
    names = ["(intercept)"]
    refs = {}

    B, knots = rcs(df["tz_age"].to_numpy("float64"), ga["age_spline_knots"])
    X.append(B)
    names += [f"tz_age_rcs{i + 1}" for i in range(B.shape[1])]

    if ga.get("entry_year", True):
        X.append((df["tz_year"].to_numpy("float64") > 2015).reshape(-1, 1))
        names.append("tz_year>2015")

    cat = list(ga["categorical"])
    if ga.get("region") == "centre":
        cat = cat + ["centre"]

    for c in cat:
        if c not in df.columns:
            warn(f"design matrix: column {c} missing, covariate not adjusted for")
            continue
        Xc, nm, ref = _dummies(df[c], c, min_n, mal)
        if Xc is None:
            warn(f"{c} has a single level; skipped")
            continue
        X.append(Xc)
        names += nm
        refs[c] = ref

    for c in list(extra):
        if c not in df.columns:
            warn(f"design matrix: continuous column {c} missing")
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy("float64")
        miss = ~np.isfinite(v)
        mu = np.nanmean(v[~miss]) if (~miss).any() else 0.0
        sd = np.nanstd(v[~miss]) or 1.0
        X.append(np.where(miss, 0.0, (v - mu) / sd).reshape(-1, 1))
        names.append(f"{c}_z")
        if miss.any():
            X.append(miss.astype("float64").reshape(-1, 1))
            names.append(f"{c}_missing")

    Xm = np.column_stack(X)
    keep, drop = independent_cols(Xm, cfg["covariates"]["design_rank_tol"])
    if drop:
        warn(f"rank check dropped {len(drop)} columns: "
             + ", ".join(names[j] for j in drop[:12])
             + (" ..." if len(drop) > 12 else ""))
    return {"X": np.ascontiguousarray(Xm[:, keep]),
            "names": [names[j] for j in keep], "knots": knots.tolist(),
            "dropped": [names[j] for j in drop], "refs": refs}


# -- Modified Poisson (Zou 2004) with optional Firth penalty -----------------
def irls_poisson(y, X, offset=None, firth=False, robust=True,
                 max_iter=80, tol=1e-9, ridge=0.0) -> dict:
    """Log-link Poisson fitted by IRLS; robust=True gives the HC0 sandwich variance.

    For binary outcomes this is Zou's modified Poisson regression: the Poisson
    point estimate of the log prevalence ratio is consistent and the sandwich
    corrects the variance, avoiding the convergence failures of log-binomial
    models. For non-negative continuous outcomes (topic burden and share) the
    same estimator is quasi-Poisson with robust variance.

    Firth: the working response y is replaced by y + h/2 (h = hat-matrix
    diagonal), which keeps coefficients finite when a group has zero cases.
    With Firth the penalised information matrix is used instead of the
    sandwich, because in separated cells the residuals vanish and HC0 would
    give a spuriously small SE.
    """
    y = np.asarray(y, dtype="float64")
    X = np.asarray(X, dtype="float64")
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype="float64")
    beta = np.zeros(p)
    ybar = max(y.mean(), 0.5 / n)
    beta[0] = math.log(ybar) - off.mean()
    I = np.eye(p)
    conv, it = False, 0
    for it in range(1, max_iter + 1):
        eta = np.clip(X @ beta + off, -30.0, 30.0)
        mu = np.clip(np.exp(eta), 1e-10, None)
        XW = X * mu[:, None]
        A = X.T @ XW + ridge * I
        yy = y
        if firth:
            try:
                Ai = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                Ai = np.linalg.pinv(A)
            h = np.clip(mu * np.einsum("ij,jk,ik->i", X, Ai, X), 0.0, 1.0)
            yy = y + h / 2.0
        rhs = X.T @ (XW @ beta + (yy - mu)) + ridge * (beta if ridge else 0.0)
        try:
            new = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(A, rhs, rcond=None)[0]
        step = np.max(np.abs(new - beta))
        beta = new
        if step < tol:
            conv = True
            break
    eta = np.clip(X @ beta + off, -30.0, 30.0)
    mu = np.clip(np.exp(eta), 1e-10, None)
    A = X.T @ (X * mu[:, None]) + ridge * I
    try:
        Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Ai = np.linalg.pinv(A)
    if robust and not firth:
        r2 = (y - mu) ** 2
        cov = Ai @ (X.T @ (X * r2[:, None])) @ Ai
    else:
        cov = Ai
    return {"beta": beta, "cov": cov,
            "se": np.sqrt(np.clip(np.diag(cov), 0, None)),
            "converged": bool(conv), "iters": int(it), "firth": bool(firth)}


# -- Inference helpers ------------------------------------------------------
Z975 = 1.959963984540054


try:
    from scipy.special import erfc as _erfc
except ImportError:
    _erfc = np.frompyfunc(math.erfc, 1, 1)


def z_to_p(z):
    """Two-sided normal P value (vectorised)."""
    z = np.asarray(z, dtype="float64")
    with np.errstate(invalid="ignore"):
        p = np.asarray(_erfc(np.abs(z) / math.sqrt(2.0)), dtype="float64")
    return np.where(np.isfinite(z), p, np.nan)


def bh_fdr(p):
    """Benjamini-Hochberg q values. NaN stays NaN and does not count towards m."""
    p = np.asarray(p, dtype="float64")
    q = np.full(p.shape, np.nan)
    m = np.isfinite(p)
    if not m.any():
        return q
    pv = p[m]
    n = pv.size
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order] * n / (np.arange(n) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(n)
    qq[order] = np.clip(ranked, 0, 1)
    q[m] = qq
    return q


def suppress(df: pd.DataFrame, min_cases: int) -> pd.DataFrame:
    """Blank cells with fewer than min_cases cases in the type or in the other types.

    Applies to the cell's case count and its estimates (UK Biobank small-cell
    rule); type sizes are kept.
    """
    d = df.copy()
    small = (d["cases_type"].fillna(0) < min_cases) | (d["cases_ref"].fillna(0) < min_cases)
    d["suppressed"] = small
    for c in ("cases_type", "log_pr", "se", "pr", "lo", "hi", "z", "p", "q"):
        if c in d.columns:
            d.loc[small, c] = np.nan
    return d


# -- Profile reliability and hierarchical ordering ---------------------------
