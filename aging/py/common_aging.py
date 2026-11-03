"""Shared utilities for the biological-aging analyses (a0-a5).

Contents: paths and configuration, the registry of the five clock families, readers for the e1 extracts
and for the a0-a2 outputs, the covariate design matrix, and the Frisch-Waugh-Lovell (FWL) regression with
HC1 standard errors used for every association estimate.

Education, restricted cubic splines, dummy coding, the rank check and BH-FDR are imported from
outcome/py/common_out.py, so the outcome and aging analyses adjust for identically defined covariates.

Measurement time points differ by family (proteomic and blood-based clocks: instance 0; MRI clocks:
instance 2). Age, centre, season and the "before measurement" cut-off all follow the family's instance,
which is defined once in FAMILIES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
AGING = HERE.parent                         # aging/
CODES = AGING.parent                        # repository root

sys.path.insert(0, str(CODES / "typology"))
sys.path.insert(0, str(CODES / "outcome" / "py"))
from common import fail, h1, h2, load_cfg, ok, sexes, tee, warn  # noqa: E402
from common_out import (Z975, _dummies, bh_fdr, build_education, data_file,  # noqa: E402
                        independent_cols, rcs, ref_file, write_json, z_to_p)

# ── Clock families ─────────────────────────────────────────────────
# inst = measurement instance; arm = platform; timing = whether the work history precedes measurement
# (blood drawn at recruitment, when most participants were still working, is "poor").
# Optional keys: qb = which lifetime-intensity file to use (default i{inst}); tech_src = e1 group holding
# the technical covariates (default imaging_qc).
FAMILIES = {
    "C-P": {"stem": "CP", "inst": 0, "arm": "protein", "timing": "poor",
            "label": "Protein organ clocks (Oh 2025)"},
    "C-C": {"stem": "CC", "inst": 0, "arm": "protein", "timing": "poor",
            "label": "Protein cell clocks (Ding 2026)"},
    "C-M": {"stem": "CM", "inst": 2, "arm": "imaging", "timing": "good",
            "label": "MRI organ clocks (MULTI method, retrained)"},
    "C-B": {"stem": "CB", "inst": 2, "arm": "imaging", "timing": "good",
            "label": "Brain clocks (Tian method, retrained)"},
    "C-L": {"stem": "CL", "inst": 0, "arm": "blood", "timing": "poor",
            "label": "Blood-chemistry & telomere ageing (PhenoAge / KDM / HD / LTL)",
            "qb": "i0all", "tech_src": "blood"},
}
CLOCK_SCRIPT = {"protein": "a1_clocks_protein", "imaging": "a2_clocks_mri", "blood": "a1b_clocks_blood"}


# ── Configuration and paths ────────────────────────────────────────
def load_cfg_aging(path=None) -> dict:
    p = Path(path) if path else AGING / "config_aging.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cfg_up() -> dict:
    """Typology config (repository root): sex split and cohort definition."""
    return load_cfg(CODES / "config.yaml")


def repo_path(c) -> Path:
    """Config path -> absolute path (relative entries are relative to the repository root)."""
    c = Path(c)
    return c if c.is_absolute() else CODES / c


def paths_aging(cfg: dict) -> dict:
    pc = cfg["paths"]
    p = {"upstream": repo_path(pc["upstream"]), "extract": repo_path(pc["extract"]),
         "data": repo_path(pc["data"]), "out": repo_path(pc["out"])}
    p["up_derived"] = p["upstream"] / "derived"
    if not (p["up_derived"] / "cohort.parquet").exists():
        fail(f"missing {p['up_derived'] / 'cohort.parquet'} -- run the typology pipeline first")
        sys.exit(1)
    if not (p["extract"] / "e1_covariates.csv.gz").exists():
        fail(f"missing {p['extract'] / 'e1_covariates.csv.gz'} -- run R/e1_extract_fields.R first")
        sys.exit(1)
    for sub in ("derived", "tables", "logs"):
        d = p["out"] / sub
        d.mkdir(parents=True, exist_ok=True)
        p[sub] = d
    return p


def setup(doc: str, extra_args=None):
    """Common entry point: parse arguments, read both configs, resolve paths."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config_aging.yaml")
    for a in (extra_args or []):
        ap.add_argument(*a[0], **a[1])
    args = ap.parse_args()
    cfg = load_cfg_aging(args.config)
    up = load_cfg_up()
    p = paths_aging(cfg)
    return args, cfg, up, p


def sha256_of(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── Readers: e1 extracts ───────────────────────────────────────────
def read_extract(p, name, usecols=None, dtype=None) -> pd.DataFrame:
    f = p["extract"] / f"e1_{name}.csv.gz"
    if not f.exists():
        fail(f"missing {f.name} in {p['extract']} -- run R/e1_extract_fields.R first")
        sys.exit(1)
    d = pd.read_csv(f, usecols=usecols, dtype=dtype, low_memory=False)
    d["eid"] = pd.to_numeric(d["eid"], errors="coerce")
    return d.dropna(subset=["eid"]).astype({"eid": "int64"})


def read_dictionary(p) -> pd.DataFrame:
    """e1 dictionary. If a wide table exists but its group is absent from the dictionary, the dictionary
    was overwritten by a partial e1 run; stop rather than treat the modality as having no features."""
    d = pd.read_csv(p["extract"] / "e1_dictionary.csv", low_memory=False)
    lost = [g for g in ("brain_tian", "mri_multi")
            if (p["extract"] / f"e1_{g}.csv.gz").exists() and g not in set(d["group"])]
    if lost:
        fail(f"e1_dictionary.csv has no rows for {lost} although the extracts exist; "
             f"rerun: Rscript e1_extract_fields.R --groups {','.join(lost)}")
        sys.exit(1)
    return d


def _sg_tokens(s) -> list:
    """Dictionary subgroup -> list of subgroups. "adipose;pancreas" is split; annotated variants such as
    "GM(...)" are merged into the part before the bracket."""
    return [t.split("(")[0].strip() for t in str(s).split(";")]


def multi_features(dic: pd.DataFrame, organ: str, inst: int) -> list:
    d = dic[(dic["group"] == "mri_multi") & (dic["instance"] == inst)]
    return [r.out_col for r in d.itertuples() if organ in _sg_tokens(r.subgroup)]


def brain_features(dic: pd.DataFrame, subgroups, inst: int) -> list:
    d = dic[(dic["group"] == "brain_tian") & (dic["instance"] == inst)]
    want = set(subgroups)
    return [r.out_col for r in d.itertuples() if want & set(_sg_tokens(r.subgroup))]


def olink_long_file(p, cfg, override=None):
    """The UKB-PPP olink_data long table (files.olink_long in paths.data, or --olink-long)."""
    f = Path(override or cfg["files"]["olink_long"])
    if not f.is_absolute():
        f = p["data"] / f
    return f if f.exists() else None


def olink_long_sep(f: Path):
    """Delimiter and header of the long table (olink_data.txt is tab-delimited)."""
    import gzip
    with (gzip.open(f, "rt") if f.suffix == ".gz" else open(f, encoding="utf-8")) as fh:
        first = fh.readline()
    sep = "\t" if "\t" in first else ","
    return sep, [h.strip().strip('"') for h in first.rstrip("\n").split(sep)]


def read_wide(p, name, cols, dtype="float32") -> pd.DataFrame:
    """Read only the requested columns of an e1 wide table (indexed by eid)."""
    have = set(pd.read_csv(p["extract"] / f"e1_{name}.csv.gz", nrows=0).columns)
    use = [c for c in cols if c in have]
    if len(use) < len(cols):
        warn(f"e1_{name} lacks {len(cols) - len(use)} columns (e.g. {[c for c in cols if c not in have][:3]})")
    d = read_extract(p, name, usecols=["eid"] + use, dtype={c: dtype for c in use})
    return d.set_index("eid")


def mask_multi_errors(X: pd.DataFrame, qc: pd.DataFrame, cfg, inst) -> pd.DataFrame:
    """Set abdominal measurements to missing when their error indicator is not in error_ok_values.
    qc is indexed by eid and read as strings."""
    okv = set(str(v) for v in cfg["multi"]["error_ok_values"])
    X = X.copy()
    for ecol, feats in cfg["multi"]["error_map"].items():
        qn = f"{ecol}_i{inst}"
        cols = [f"{f}_i{inst}" for f in feats if f"{f}_i{inst}" in X.columns]
        if not cols:
            continue
        if qn not in qc.columns:
            warn(f"e1_imaging_qc has no {qn}; {cols} not screened for errors")
            continue
        v = qc[qn].reindex(X.index)
        bad = v.notna() & ~v.astype(str).str.strip().isin(okv)
        X.loc[bad, cols] = np.nan
    return X


def load_fc(p, inst) -> pd.DataFrame:
    f = p["extract"] / "e1_fc_d25p.csv.gz"
    if not f.exists():
        warn(f"missing {f.name}; connectivity models skipped")
        return pd.DataFrame()
    d = read_extract(p, "fc_d25p")
    d = d[d["instance"] == inst].drop(columns="instance")
    return d.drop_duplicates("eid").set_index("eid").astype("float32")


def need(f: Path, who: str):
    if not Path(f).exists():
        fail(f"missing {Path(f).name} -- run {who} first")
        sys.exit(1)
    return Path(f)


# ── Readers: clock files and validation status ────────────────────
def clock_file(p, fam) -> Path:
    return p["derived"] / f"clock_{FAMILIES[fam]['stem']}.parquet"


def clock_names(df: pd.DataFrame) -> list:
    return [c[len("gapz__"):] for c in df.columns if c.startswith("gapz__")]


def load_clocks(p, fam, required=True):
    f = clock_file(p, fam)
    if not f.exists():
        if required:
            fail(f"missing {f.name} -- run {CLOCK_SCRIPT[FAMILIES[fam]['arm']]} first")
            sys.exit(1)
        return None, []
    d = pd.read_parquet(f)
    return d, clock_names(d)


def load_status(p) -> dict:
    """Per-clock validation status written by a3 (V1 / V2 / V7 and whether all criteria are met)."""
    f = p["derived"] / "clock_status.json"
    if not f.exists():
        fail("missing clock_status.json -- run a3_validate.py first")
        sys.exit(1)
    return json.loads(f.read_text(encoding="utf-8"))


def is_main(status: dict, fam: str, clock: str) -> bool:
    return bool(status.get(fam, {}).get(clock, {}).get("main", False))


# ── Dates and ages ─────────────────────────────────────────────────
def to_date(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def birth_date(year, month, day=15) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce")
    m = pd.to_numeric(month, errors="coerce")
    ok_ = y.notna() & m.notna()
    out = pd.Series(pd.NaT, index=y.index, dtype="datetime64[ns]")
    out[ok_] = pd.to_datetime(dict(year=y[ok_].astype(int), month=m[ok_].astype(int),
                                   day=int(day)), errors="coerce")
    return out


def years_between(a, b) -> pd.Series:
    """(b - a) / 365.25 in years."""
    return (to_date(b) - to_date(a)).dt.days / 365.25


# ── Analysis frame: one family x one sex ───────────────────────────
def load_person(p, sex) -> pd.DataFrame:
    return pd.read_parquet(need(p["derived"] / f"person_{sex}.parquet", "a0_assemble"))


def load_measure(p) -> pd.DataFrame:
    return pd.read_parquet(need(p["derived"] / "measure.parquet", "a0_assemble"))


def load_frame(p, cfg, fam, sex, inst=None):
    """person x measurement time point x clocks -> one analysis table (work-history cohort members with
    a clock of this family).

    Returns (df, clocks). df carries age_meas / date_meas / centre_meas / cn_meas / exposed_frac /
    left_before and the family's technical covariates (tech__*). Clocks listed in exclude_clocks are dropped.
    """
    inst = FAMILIES[fam]["inst"] if inst is None else inst
    per = load_person(p, sex)
    mea = load_measure(p)
    ck, clocks = load_clocks(p, fam)
    drop = [c for c in (cfg.get("exclude_clocks") or {}).get(fam, []) if c in clocks]
    if drop:
        ck = ck.drop(columns=[f"{k}__{c}" for c in drop for k in ("pred", "gapy", "gapz") if f"{k}__{c}" in ck])
        clocks = [c for c in clocks if c not in drop]
    ck = ck.drop(columns=[c for c in ("sex", "age", "cn") if c in ck.columns])
    m = mea[["eid", f"date_i{inst}", f"age_i{inst}", f"centre_i{inst}", f"cn_i{inst}"]].rename(
        columns={f"date_i{inst}": "date_meas", f"age_i{inst}": "age_meas",
                 f"centre_i{inst}": "centre_meas", f"cn_i{inst}": "cn_meas"})
    df = per.merge(m, on="eid", how="left").merge(ck, on="eid", how="inner")
    df = df[df["age_meas"].notna()].copy()
    ce = df["career_end"].astype(float)
    df["exposed_frac"] = ((np.minimum(ce, df["age_meas"]) - 16) / (ce - 16)).clip(0, 1)
    df["left_before"] = ce <= df["age_meas"]
    tech = cfg["covariates"]["tech"].get(fam) or []
    if tech:
        src = FAMILIES[fam].get("tech_src", "imaging_qc")
        qc = read_extract(p, src, usecols=lambda c: c == "eid" or any(c == f"{t}_i{inst}" for t in tech), dtype=str)
        cols = [f"{t}_i{inst}" for t in tech if f"{t}_i{inst}" in qc.columns]
        miss = [t for t in tech if f"{t}_i{inst}" not in qc.columns]
        if miss:
            warn(f"{fam}: technical covariates not in e1_{src}: {miss}")
        q = qc[["eid"] + cols].rename(columns={c: "tech__" + c[: -len(f"_i{inst}")] for c in cols})
        df = df.merge(q, on="eid", how="left")
    return df.reset_index(drop=True), clocks


def restrict(df, fam, sample) -> pd.DataFrame:
    """Sample: full, or restricted (left work before measurement; the left-work sample)."""
    if sample == "full":
        return df
    if sample == "restricted":
        return df[df["left_before"]].reset_index(drop=True)
    raise ValueError(sample)


def samples_for(fam, cfg) -> list:
    """Blood-measured families (proteomic and blood-based) are analysed in the left-work and full samples;
    MRI families, measured after almost all careers had ended, in the full sample only."""
    if FAMILIES[fam]["arm"] in ("protein", "blood") and cfg["protein"]["restrict_to_left_before_draw"]:
        return ["restricted", "full"]
    return ["full"]


def qb_file(p, fam, sex) -> Path:
    """Lifetime-intensity file of a family: qb_i{inst}_{sex} from a0; C-L uses qb_i0all_{sex} from a1b."""
    key = FAMILIES[fam].get("qb", f"i{FAMILIES[fam]['inst']}")
    return p["derived"] / f"qb_{key}_{sex}.parquet"


# ── Design matrix ──────────────────────────────────────────────────
def _as_cat_or_num(s: pd.Series, max_levels=10):
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().sum() == 0:
        return "cat", s
    if v.dropna().nunique() <= max_levels:
        return "cat", s.astype("object")
    return "num", v


def _cont_block(v, name):
    v = np.asarray(pd.to_numeric(v, errors="coerce"), dtype="float64")
    miss = ~np.isfinite(v)
    if (~miss).sum() < 2:
        return [], []
    mu, sd = np.nanmean(v[~miss]), np.nanstd(v[~miss]) or 1.0
    cols, names = [np.where(miss, 0.0, (v - mu) / sd)], [f"{name}_z"]
    if miss.any():
        cols.append(miss.astype("float64"))
        names.append(f"{name}_missing")
    return cols, names


def design(df: pd.DataFrame, cfg: dict, fam: str, extra=()) -> dict:
    """Covariates: rcs(age at measurement) + 5-year birth cohort + assessment centre + education
    + technical covariates (tech__*) + season and year of measurement; extra adds continuous columns.

    Returns {Z, names, dropped}. Z has no missing values (missing categorical = own level; missing
    continuous = 0 plus an indicator column).
    """
    cc = cfg["covariates"]
    n = len(df)
    X, names = [np.ones((n, 1))], ["(intercept)"]
    B, _ = rcs(df["age_meas"].to_numpy("float64"), cc["age_knots"])
    X.append(B)
    names += [f"age_rcs{i + 1}" for i in range(B.shape[1])]
    cats = ["birth_cohort_5y", "centre_meas", "edu_highest"]
    for c in cats:
        if c not in df.columns:
            warn(f"design matrix: {c} missing")
            continue
        Xc, nm, _ref = _dummies(df[c], c, cc["min_level_n"], cc["missing_as_level"])
        if Xc is not None:
            X.append(Xc)
            names += nm
    conts = list(extra)
    for c in conts:
        cols, nm = _cont_block(df[c], c)
        X += [x.reshape(-1, 1) for x in cols]
        names += nm
    for c in [c for c in df.columns if c.startswith("tech__")]:
        kind, v = _as_cat_or_num(df[c])
        if kind == "cat":
            Xc, nm, _ref = _dummies(v, c, cc["min_level_n"], True)
            if Xc is not None:
                X.append(Xc)
                names += nm
        else:
            cols, nm = _cont_block(v, c)
            X += [x.reshape(-1, 1) for x in cols]
            names += nm
    if cc.get("season", True) and "date_meas" in df.columns:
        d = to_date(df["date_meas"])
        doy = d.dt.dayofyear.to_numpy("float64")
        yr = d.dt.year.to_numpy("float64")
        for nm, v in (("season_sin", np.sin(2 * np.pi * doy / 365.25)),
                      ("season_cos", np.cos(2 * np.pi * doy / 365.25)), ("meas_year", yr)):
            cols, nn = _cont_block(v, nm)
            X += [x.reshape(-1, 1) for x in cols]
            names += nn
    Xm = np.column_stack(X)
    keep, drop = independent_cols(Xm, cc["design_rank_tol"])
    return {"Z": np.ascontiguousarray(Xm[:, keep]), "names": [names[j] for j in keep],
            "dropped": [names[j] for j in drop]}


# ── FWL + HC1 ──────────────────────────────────────────────────────
def _basis(Z, tol=1e-9):
    """Orthonormal basis of the column space of Z (pivoted QR, so empty levels in a subsample are dropped)."""
    from scipy.linalg import qr
    Q, R, _piv = qr(Z, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    r = int((d > tol * max(d[0], 1e-300)).sum()) if len(d) else 0
    return Q[:, :r]


def residualise(A, Q):
    A = np.asarray(A, dtype="float64")
    return A - Q @ (Q.T @ A)


def fwl(y, E, Z=None, Q=None, want_if=False) -> dict:
    """Coefficients of E in y ~ E + Z with HC1 covariance, estimated on residuals (FWL; identical to the
    full model). Columns of E that are degenerate in the subsample return NaN.
    want_if=True also returns the influence functions IF (n x k, HC0 scale: beta_hat - beta ~ sum IF_i).
    """
    y = np.asarray(y, dtype="float64")
    E = np.asarray(E, dtype="float64")
    if E.ndim == 1:
        E = E[:, None]
    if Q is None:
        Q = _basis(Z)
    n, k = E.shape
    ry = residualise(y, Q)
    rE = residualise(E, Q)
    ss = (rE ** 2).sum(0)
    good = ss > 1e-8 * np.maximum((E ** 2).sum(0), 1e-12)
    out = {"beta": np.full(k, np.nan), "se": np.full(k, np.nan), "n": n,
           "cov": np.full((k, k), np.nan), "df": n - Q.shape[1] - int(good.sum())}
    if not good.any() or out["df"] <= 1:
        return out
    G = rE[:, good]
    A = G.T @ G
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return out
    b = Ainv @ (G.T @ ry)
    e = ry - G @ b
    S = G * e[:, None]
    V = Ainv @ (S.T @ S) @ Ainv * n / max(out["df"], 1)
    out["beta"][good] = b
    idx = np.where(good)[0]
    out["cov"][np.ix_(idx, idx)] = V
    out["se"][good] = np.sqrt(np.clip(np.diag(V), 0, None))
    out["sigma"] = float(np.sqrt((e ** 2).sum() / max(out["df"], 1)))
    out["cov_ols"] = np.full((k, k), np.nan)                      # homoskedastic sigma^2 (G'G)^-1
    out["cov_ols"][np.ix_(idx, idx)] = Ainv * out["sigma"] ** 2
    if want_if:
        IF = np.zeros((n, k))
        IF[:, good] = S @ Ainv
        out["IF"] = IF
    return out


def assoc(df, D, clocks, E: pd.DataFrame, meta: dict, status=None, fam=None, min_n=30, n_term=None):
    """gapz ~ E + Z for each clock; one output row per clock x column of E.

    Rows with missing E or missing gap are dropped per clock (Z has no missing values).
    n_term = participants contributing to the term ("nonzero": members of the type for an indicator;
    otherwise the non-missing n), used for the n < min_type_n rule.
    """
    rows = []
    Ev = E.to_numpy("float64")
    eok = np.isfinite(Ev).all(1)
    for ck in clocks:
        y = df[f"gapz__{ck}"].to_numpy("float64")
        m = eok & np.isfinite(y)
        if m.sum() < min_n:
            continue
        r = fwl(y[m], Ev[m], D["Z"][m])
        sdy = float(np.nanstd(df.loc[m, f"gapy__{ck}"])) if f"gapy__{ck}" in df.columns else np.nan
        for j, t in enumerate(E.columns):
            nt = int((Ev[m, j] != 0).sum()) if n_term == "nonzero" else int(m.sum())
            b, s = r["beta"][j], r["se"][j]
            rows.append({**meta, "clock": ck, "term": t, "n": int(m.sum()), "n_term": nt,
                         "beta": b, "se": s, "z": b / s if s and np.isfinite(s) else np.nan,
                         "gap_sd_years": sdy, "beta_years": b * sdy,
                         "main": is_main(status, fam, ck) if status is not None else True})
    out = pd.DataFrame(rows)
    if len(out):
        out["p"] = z_to_p(out["z"].to_numpy())
    return out


def fdr_within(df: pd.DataFrame, by: list, tested: pd.Series, col="p") -> pd.Series:
    """BH q values within groups `by`, computed over the rows where `tested` is True (others: NaN)."""
    q = pd.Series(np.nan, index=df.index)
    for _, g in df[tested.astype(bool)].groupby(by, dropna=False):
        q.loc[g.index] = bh_fdr(g[col].to_numpy())
    return q


def zstd(s) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * np.nan


def vs_rest(col: pd.Series, level, name: str) -> pd.DataFrame:
    """One type versus all other types: a 0/1 indicator; missing type -> NaN (not counted as "rest")."""
    return pd.DataFrame({name: (col == level).astype(float).where(col.notna())}, index=col.index)


def type_names(p, sex) -> dict:
    f = p["up_derived"] / f"type_names_{sex}.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text(encoding="utf-8"))
    return {int(k): v for k, v in d.get("names", {}).items()}


__all__ = ["CODES", "FAMILIES", "Z975", "_basis", "_cont_block", "_dummies", "assoc", "bh_fdr", "birth_date",
           "brain_features", "build_education", "clock_file", "data_file", "design", "fail", "fdr_within",
           "fwl", "h1", "h2", "independent_cols", "load_cfg_up", "load_clocks", "load_frame", "load_measure",
           "load_person", "load_status", "mask_multi_errors", "multi_features", "ok", "olink_long_file",
           "olink_long_sep", "qb_file", "rcs", "read_dictionary", "read_extract", "read_wide", "ref_file",
           "residualise", "restrict", "samples_for", "setup", "sexes", "sha256_of", "tee", "to_date",
           "type_names", "vs_rest", "warn", "write_json", "years_between", "z_to_p", "zstd"]
