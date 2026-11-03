"""Shared helpers for the molecular and neuroimaging analysis (career types x features).

Reuses only pure functions from the aging module (design matrix, QR basis, BH, z -> p); it never calls the
aging module's path or setup helpers.

Paths (config_molecular.yaml `paths.*`, relative entries resolved against the repository root):
  p["up_derived"]     typology outputs (cohort, worklife_annual, camsis_by_age)
  p["aging_derived"]  aging-module outputs read here (person_{sex}, measure)
  p["mextract"]       this module's extracts (R/m_e1_extract.R, R/m_e2_nmr_ukbnmr.R) = <out>/extract
  p["data"]           UKB raw tables (Olink long table)
  p["derived"] / ["tables"] / ["logs"]   outputs of this module

Association engine `Assoc`: one model per feature, y_j ~ e + Z (Frisch-Waugh-Lovell + HC1 standard errors).
Features are grouped by missingness pattern; Z is QR-decomposed once on the full sample and the residual
operator for each row subset is derived from that single decomposition.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
MOL = HERE.parent                          # molecular/
CODES = MOL.parent                         # repository root
sys.path.insert(0, str(CODES / "aging" / "py"))

from common_aging import (_basis, _dummies, bh_fdr, design, fail, h1, h2, independent_cols,  # noqa: E402
                          load_cfg_up, ok, sexes, tee, warn, z_to_p, zstd)

_ = (fail, h1, h2, ok, sexes, tee, warn)   # re-exported for the scripts

LAYERS = ("LB", "LN", "LP", "LI")
ARM_OF = {"LB": "blood", "LN": "blood", "LP": "blood", "LI": "brain"}
# BH families for the per-feature tests
FDR_BY = ["sex", "layer", "modality", "sample", "analysis", "model", "term"]


# ── Config and paths ─────────────────────────────────────────────────
def load_cfg_mol(path=None) -> dict:
    p = Path(path) if path else MOL / "config_molecular.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def from_root(c) -> Path:
    """Config path -> absolute path (relative entries are relative to the repository root)."""
    c = Path(c)
    return c if c.is_absolute() else CODES / c


def _need_dir(c, marker, what) -> Path:
    d = from_root(c)
    if not (d / marker).exists():
        raise FileNotFoundError(f"{what} not found: {d / marker}")
    return d


def paths_mol(cfg: dict) -> dict:
    pc = cfg["paths"]
    out = from_root(pc["out"])
    p = {"upstream": _need_dir(pc["upstream"], Path("derived") / "cohort.parquet", "typology outputs"),
         "aging_derived": _need_dir(pc["aging_derived"], "measure.parquet", "aging-module outputs"),
         "mextract": _need_dir(out / "extract", "m_e1_dictionary.csv",
                               "m_e1 extracts (run R/m_e1_extract.R first)"),
         "data": from_root(pc["data"]),
         "out": out}
    for sub in ("derived", "tables", "logs"):
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        p[sub] = d
    p["up_derived"] = p["upstream"] / "derived"
    p["ref"] = MOL / "ref"
    return p


def ref_path(cfg: dict, name: str) -> Path:
    """Published reference file under `paths.refs` (not individual-level data)."""
    f = from_root(cfg["paths"]["refs"]) / name
    if not f.exists():
        fail(f"reference file not found: {f}")
        sys.exit(1)
    return f


def setup(doc: str, extra_args=None):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config_molecular.yaml")
    for a in (extra_args or []):
        ap.add_argument(*a[0], **a[1])
    args = ap.parse_args()
    cfg = load_cfg_mol(args.config)
    up = load_cfg_up()
    p = paths_mol(cfg)
    return args, cfg, up, p


def need(f: Path, who: str) -> Path:
    if not Path(f).exists():
        fail(f"missing {Path(f).name}; run {who} first")
        sys.exit(1)
    return Path(f)


# ── Reading inputs ───────────────────────────────────────────────────
def read_m(p, name, usecols=None, dtype=None, required=True) -> pd.DataFrame | None:
    """Read an extract of this module; `name` is the file stem (e.g. m_e1_blood_bc, m_e2_nmr_i0)."""
    f = p["mextract"] / f"{name}.csv.gz"
    if not f.exists():
        if required:
            fail(f"missing {f.name} in {p['mextract']}; run "
                 f"R/{'m_e2_nmr_ukbnmr' if 'e2' in name else 'm_e1_extract'}.R first")
            sys.exit(1)
        return None
    d = pd.read_csv(f, usecols=usecols, dtype=dtype, low_memory=False)
    d["eid"] = pd.to_numeric(d["eid"], errors="coerce")
    return d.dropna(subset=["eid"]).astype({"eid": "int64"}).drop_duplicates("eid")


def brain_ref(p) -> pd.DataFrame:
    return pd.read_csv(need(p["ref"] / "brain_features.csv", "py/build_brain_ref.py"))


def load_person(p, sex) -> pd.DataFrame:
    return pd.read_parquet(need(p["aging_derived"] / f"person_{sex}.parquet", "the aging module (a0)"))


def load_measure(p) -> pd.DataFrame:
    return pd.read_parquet(need(p["aging_derived"] / "measure.parquet", "the aging module (a0)"))


def type_levels(p, sex) -> list:
    per = load_person(p, sex)
    return sorted(per["type"].dropna().astype(int).unique())


def multi_arr(df: pd.DataFrame, short: str, inst: int) -> list:
    """m_e1 column names: {short}_i{inst} or {short}_i{inst}_a{k}."""
    rx = re.compile(rf"^{re.escape(short)}_i{inst}(_a\d+)?$")
    return [c for c in df.columns if rx.match(c)]


def first_nonnull(df: pd.DataFrame, cols) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].bfill(axis=1).iloc[:, 0]


# ── Analysis frames and feature tables ───────────────────────────────
def frame_file(p, arm, sex) -> Path:
    return p["derived"] / f"frame_{arm}_{sex}.parquet"


def feat_file(p, layer, sex) -> Path:
    return p["derived"] / f"feat_{layer}_{sex}.parquet"


def feat_meta(p, layer) -> pd.DataFrame:
    return pd.read_csv(need(p["derived"] / f"feat_meta_{layer}.csv", "m1_features.py"))


def restrict(df, sample) -> pd.DataFrame:
    """full; restricted = career ended before the measurement (blood layers); cn = free of chronic disease
    at measurement (brain MRI)."""
    if sample == "full":
        return df
    if sample == "restricted":
        return df[df["left_before"].astype(bool)]
    if sample == "cn":
        return df[df["cn_meas"].astype("boolean").fillna(False).astype(bool)]
    raise ValueError(sample)


def load_layer(p, cfg, layer, sex, sample="full"):
    """Frame (m0) x features (m1) -> (df, X, meta). df holds covariates, exposures, tech__* and cat__* columns;
    X holds the feature columns."""
    arm = ARM_OF[layer]
    fr = pd.read_parquet(need(frame_file(p, arm, sex), "m0_assemble.py")).set_index("eid")
    ft = pd.read_parquet(need(feat_file(p, layer, sex), "m1_features.py"))
    if "eid" in ft.columns:
        ft = ft.set_index("eid")
    meta = feat_meta(p, layer)
    feats = [f for f in meta["feature"] if f in ft.columns]
    side = [c for c in ft.columns if c.startswith(("tech__", "cat__"))]
    df = fr.join(ft[side], how="inner")
    df = restrict(df, sample)
    X = ft.loc[df.index, feats].astype("float64")
    return df.reset_index(), X.reset_index(drop=True), meta[meta["feature"].isin(feats)].reset_index(drop=True)


def design_mol(df: pd.DataFrame, cfg: dict, extra=()) -> dict:
    """Aging-module design (age spline, birth cohort, centre, education, tech__*, season / year) plus this
    module's cat__* covariates (ethnic group, Olink plate; levels < cat_min_level_n pooled into Other)."""
    D = design(df, cfg, "mol", extra=extra)
    cc = cfg["covariates"]
    blocks, names = [D["Z"]], list(D["names"])
    for c in [c for c in df.columns if c.startswith("cat__")]:
        Xc, nm, _ref = _dummies(df[c], c, cc["cat_min_level_n"], True)
        if Xc is not None:
            blocks.append(Xc)
            names += nm
    if len(blocks) == 1:
        return D
    Xm = np.column_stack(blocks)
    keep, drop = independent_cols(Xm, cc["design_rank_tol"])
    return {"Z": np.ascontiguousarray(Xm[:, keep]), "names": [names[j] for j in keep],
            "dropped": D["dropped"] + [names[j] for j in drop]}


def x_names(cfg) -> list:
    return list(cfg["exposure"]["agents"]) + ["shift", "night", "longhours"]


def term_specs(df, cfg, types, analyses=("type", "axis", "x")):
    """Yield (analysis, model, term, e, kind). kind: binary (type vs rest), cont (axis, +1 SD) or exposed
    (exposure-years; n1 = participants with any exposure). Model X2 adds pre-measurement mean CAMSIS to Z."""
    ex = cfg["exposure"]
    if "type" in analyses:
        t = pd.to_numeric(df["type"], errors="coerce")
        for k in types:
            yield "type", "base", f"type:{int(k)}", (t == k).astype(float).where(t.notna()).to_numpy(), "binary"
    if "axis" in analyses:
        for ax in ex["axes"]:
            if ax in df.columns:
                e = zstd(df[ax]).to_numpy()
                if np.isfinite(e).sum() >= 3:
                    yield "axis", "base", f"axis:{ax}", e, "cont"
    if "x" in analyses:
        for m in ex["x_models"]:
            for x in x_names(cfg):
                c = f"x__{x}"
                if c in df.columns:
                    yield "x", m, f"x:{x}", pd.to_numeric(df[c], errors="coerce").to_numpy() / ex["per_years"], "exposed"


# ── Association engine ───────────────────────────────────────────────
class _Proj:
    """Residual-on-Z operator for a row subset r, derived from one full-sample QR (Z = Q0 R).

    span(Z_r) = span(Q0_r), so the projection is Q0_r G^+ Q0_r' with G = Q0_r' Q0_r = I - Q0_S' Q0_S
    (S = dropped rows). G is k x k; eigenvalues ~0 (a dummy whose members were all dropped) are discarded, so
    the retained rank equals the rank of a pivoted QR of Z_r."""

    TOL = 1e-10            # eigenvalues of G lie in [0, 1]; lost directions ~1e-16, valid minimum ~1 / level size

    def __init__(self, Q0, r):
        self.Q0, self.r = Q0, r
        n, k = Q0.shape
        if len(r) == n:
            self.M, self.rank = None, k
            return
        drop = np.ones(n, dtype=bool)
        drop[r] = False
        if drop.sum() < len(r):
            QS = Q0[drop]
            G = np.eye(k) - QS.T @ QS
        else:
            Qr = Q0[r]
            G = Qr.T @ Qr
        w, V = np.linalg.eigh((G + G.T) / 2)
        keep = w > self.TOL
        self.M = (V[:, keep] / w[keep]) @ V[:, keep].T
        self.rank = int(keep.sum())

    def resid(self, A):
        A = np.asarray(A, dtype="float64")
        Q0 = self.Q0
        if self.M is None:
            return A - Q0 @ (Q0.T @ A)
        full = np.zeros((Q0.shape[0],) + A.shape[1:])
        full[self.r] = A
        return A - (Q0 @ (self.M @ (Q0.T @ full)))[self.r]


class Assoc:
    """y_j ~ e + Z for j = 1..m (FWL + HC1).

    Missing Y values are dropped feature by feature; missing e values term by term. Features sharing a
    missingness pattern are processed together, and all terms with the same row set are residualised in one
    matrix product."""

    def __init__(self, Y, Z, min_n=30):
        self.Y = np.asarray(Y, dtype="float64")
        self.Z = np.asarray(Z, dtype="float64")
        self.min_n = min_n
        self.Q0 = _basis(self.Z)
        M = np.isfinite(self.Y)
        packed = np.packbits(M, axis=0)
        grp = {}
        for j in range(self.Y.shape[1]):
            grp.setdefault(packed[:, j].tobytes(), []).append(j)
        self.groups = [(np.where(M[:, c[0]])[0], np.asarray(c)) for c in grp.values()]

    def _iter(self, fins):
        """Per feature group: (cols, {row pattern -> (r, P, RY)}, pattern key of each term); P is None when
        the row set is too small."""
        for rows, cols in self.groups:
            sub = {}
            keys = []
            for fin in fins:
                fr = fin[rows]
                k = b"" if fr.all() else np.packbits(fr).tobytes()
                keys.append(k)
                if k not in sub:
                    r = rows if k == b"" else rows[fr]
                    if len(r) < max(self.min_n, 3):
                        sub[k] = (r, None, None)
                        continue
                    P = _Proj(self.Q0, r)
                    sub[k] = (r, P, P.resid(self.Y[np.ix_(r, cols)]))
            yield cols, sub, keys

    def fit_many(self, terms):
        """terms = [(e, kind), ...] -> [DataFrame[col, beta, se, n, n1, n0, z, p]] in the same order."""
        m = self.Y.shape[1]
        E = [np.asarray(e, dtype="float64") for e, _ in terms]
        kinds = [k for _, k in terms]
        res = [{k: np.full(m, np.nan) for k in ("beta", "se")} for _ in terms]
        cnt = [{k: np.zeros(m, dtype="int64") for k in ("n", "n1", "n0")} for _ in terms]
        Emat = np.column_stack(E) if E else np.zeros((len(self.Y), 0))
        for cols, sub, keys in self._iter([np.isfinite(e) for e in E]):
            RE = {}
            for k, (r, P, _RY) in sub.items():
                if P is not None:
                    ts = [t for t, kk in enumerate(keys) if kk == k]
                    RE[k] = (ts, P.resid(Emat[np.ix_(r, ts)]))
            for t, (e, kind) in enumerate(zip(E, kinds)):
                r, P, RY = sub[keys[t]]
                n = len(r)
                ev = e[r]
                n1 = int((ev == 1).sum()) if kind == "binary" else int((ev > 0).sum()) if kind == "exposed" else n
                cnt[t]["n"][cols], cnt[t]["n1"][cols], cnt[t]["n0"][cols] = n, n1, n - n1
                if P is None:
                    continue
                df = n - P.rank - 1
                if n < self.min_n or df <= 1:
                    continue
                ts, REk = RE[keys[t]]
                re_ = REk[:, ts.index(t)]
                ss = float(re_ @ re_)
                if ss <= 1e-8 * max(float(ev @ ev), 1e-12):
                    continue
                b = (re_ @ RY) / ss
                U = RY - np.outer(re_, b)
                meat = (re_ ** 2) @ (U ** 2)
                res[t]["beta"][cols] = b
                res[t]["se"][cols] = np.sqrt(np.clip(meat, 0, None) / ss ** 2 * n / df)   # HC1
        outs = []
        for t in range(len(terms)):
            o = pd.DataFrame({"col": np.arange(m), **res[t], **cnt[t]})
            o["z"] = o["beta"] / o["se"]
            o["p"] = z_to_p(o["z"].to_numpy())
            outs.append(o)
        return outs

    def residuals(self):
        """Residuals of each feature on Z (missing values dropped per column). Used by the omnibus test."""
        out = np.full_like(self.Y, np.nan)
        for cols, sub, keys in self._iter([np.ones(len(self.Y), dtype=bool)]):
            r, P, RY = sub[keys[0]]
            if P is not None:
                out[np.ix_(r, cols)] = RY
        return out


def grey_flag(kind, n1, n0, min_n):
    """Contrasts with < min_n participants in the type or the rest (or < min_n exposed) are not tested."""
    n1, n0 = np.asarray(n1), np.asarray(n0)
    if kind == "binary":
        return (n1 < min_n) | (n0 < min_n)
    if kind == "exposed":
        return n1 < min_n
    return np.zeros(len(n1), dtype=bool)


def run_terms(df, X, meta, cfg, types, layer, sex, sample, analyses=("type", "axis", "x"), min_n=30):
    """All exposure terms for one (sex, layer, sample) -> long table. Two versions of Z: base (type, axis)
    and X2 (base + pre-measurement mean CAMSIS, for exposure-years)."""
    feats = X.columns.to_numpy()
    Y = X.to_numpy("float64")
    mod = meta.set_index("feature").reindex(feats)["modality"].fillna(layer).to_numpy()
    specs = list(term_specs(df, cfg, types, analyses))
    gmin = cfg["exposure"]["grey_min_n"]
    rows = []
    for zver in ("base", "X2"):
        sp = [s for s in specs if ("X2" if s[1] == "X2" else "base") == zver]
        if not sp:
            continue
        D = design_mol(df, cfg, extra=("camsis_pre",) if zver == "X2" else ())
        outs = Assoc(Y, D["Z"], min_n=min_n).fit_many([(s[3], s[4]) for s in sp])
        for t, (analysis, model, term, _e, kind) in enumerate(sp):
            r = outs[t]
            r["feature"] = feats[r["col"]]
            r["modality"] = mod[r["col"]]
            r = r.drop(columns="col").assign(sex=sex, layer=layer, sample=sample, analysis=analysis, model=model,
                                             term=term, kind=kind)
            r["grey"] = grey_flag(kind, r["n1"], r["n0"], gmin)
            rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_q(df: pd.DataFrame, alpha, by=None, col="p") -> pd.DataFrame:
    """BH within families `by`, over non-grey cells only; sig = q < alpha and not grey."""
    by = by or FDR_BY
    df = df.copy()
    df["q"] = np.nan
    ok_ = ~df["grey"].astype(bool) & np.isfinite(df[col])
    for _, g in df[ok_].groupby(by, dropna=False):
        df.loc[g.index, "q"] = bh_fdr(g[col].to_numpy())
    df["sig"] = (df["q"] < alpha) & ~df["grey"].astype(bool)
    return df


def nominal(df: pd.DataFrame, cfg: dict, col="p") -> pd.Series:
    """Nominal significance: p < nominal_p and not grey."""
    grey = df["grey"].astype(str).str.lower().isin(["true", "1"]) if "grey" in df else False
    return (pd.to_numeric(df[col], errors="coerce") < cfg["nominal_p"]) & ~grey


# ── Transforms ───────────────────────────────────────────────────────
def mad_trim(v: np.ndarray, k=6.0):
    """Values more than k x 1.4826 MAD from the median -> NaN. Returns (v, n_trimmed). No trimming when
    MAD = 0 (discrete measures)."""
    v = np.asarray(v, dtype="float64").copy()
    f = np.isfinite(v)
    if f.sum() < 10:
        return v, 0
    med = np.median(v[f])
    mad = 1.4826 * np.median(np.abs(v[f] - med))
    if not mad > 0:
        return v, 0
    bad = f & (np.abs(v - med) > k * mad)
    v[bad] = np.nan
    return v, int(bad.sum())


def zcol(v):
    v = np.asarray(v, dtype="float64")
    f = np.isfinite(v)
    if f.sum() < 3:
        return v * np.nan
    sd = v[f].std()
    return (v - v[f].mean()) / sd if sd > 0 else v * np.nan


def resid_on_Z(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Column-wise residuals on Z (missing values dropped per column). Used by m3b; no exposure involved."""
    return Assoc(X, Z, min_n=3).residuals()
