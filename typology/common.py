"""Shared utilities for the typology pipeline.

Path resolution, parsing of UK Biobank wide-format array fields, UKB coding
conversions, reproducibility settings, the representation-freeze hash check,
representation diagnostics and logging. All coding values and thresholds live
in `config.yaml` at the repository root.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Deterministic cuBLAS requires this variable before the first CUDA matmul
# creates a cuBLAS handle; setting it at import time is early enough.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
CODES = HERE.parent          # repository root

# SOC2000 aggregation levels -> divisor applied to the 4-digit unit-group code.
# L1 major (9) / L2 sub-major (25) / L3 minor (81) / L4 unit group (353, main model)
SOC_DIVISOR = {"soc_l1": 1000, "soc_l2": 100, "soc_l3": 10, "soc_l4": 1}


# -- Configuration and paths ------------------------------------------------
def load_cfg(path=None) -> dict:
    p = Path(path) if path else CODES / "config.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_dir(entry, what, make=False) -> Path:
    """Resolve a configured directory; relative entries are taken from the repo root."""
    d = Path(entry)
    if not d.is_absolute():
        d = CODES / d
    if d.is_dir():
        return d
    if make:
        d.mkdir(parents=True, exist_ok=True)
        return d
    raise FileNotFoundError(f"{what} directory not found: {d}")


def paths(cfg: dict) -> dict:
    p = {"data": _resolve_dir(cfg["paths"]["data"], "data"),
         "out": _resolve_dir(cfg["paths"]["out"], "output", make=True),
         "refs": _resolve_dir(cfg["paths"]["refs"], "references", make=True)}
    for sub in ("derived", "tables", "qc", "models", "logs"):
        d = p["out"] / sub
        d.mkdir(parents=True, exist_ok=True)
        p[sub] = d
    return p


def sexes(cfg: dict) -> tuple:
    return (tuple(cfg["sex_split"]["labels"]) if cfg["sex_split"]["enabled"]
            else ("Pooled",))


# -- Wide-table array parsing -----------------------------------------------
# Columns are named <FieldName>__<instance>_<array index>.
_SLOT = re.compile(r"^(?P<name>.+?)__(?P<inst>\d+)_(?P<slot>\d+)$")


def parse_col(col: str):
    m = _SLOT.match(col)
    return (m["name"], int(m["inst"]), int(m["slot"])) if m else None


def slot_cols(all_cols, field, inst=0) -> list:
    """All array columns of a field, in ascending array index.

    The employment-history arrays are sparse and keep the job index: array
    index k always refers to job k, so the index can be used as a job index.
    """
    found = {}
    for c in all_cols:
        pc = parse_col(c)
        if pc and pc[0] == field and pc[1] == inst:
            found[pc[2]] = c
    return [found[k] for k in sorted(found)]


def to_long(df, all_cols, fields: dict, inst=0) -> pd.DataFrame:
    """Several wide array fields -> one long table aligned on (row_id, slot).

    Each field is re-indexed to positional slots before stacking; stacking the
    raw columns and concatenating would not align, because the column labels
    differ between fields.
    """
    parts = {}
    for out_name, fld in fields.items():
        sc = slot_cols(all_cols, fld, inst)
        if not sc:
            continue
        sub = df[sc].apply(pd.to_numeric, errors="coerce")
        sub.columns = range(len(sc))
        sub.index.name = "row_id"
        s = sub.stack(future_stack=True)
        s.index = s.index.set_names(["row_id", "slot"])
        parts[out_name] = s
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


# -- UKB coding conversions -------------------------------------------------
def map_exposure(series, cfg):
    """Workplace exposure frequency -> ordinal 0/1/2; 'Do not know' (-121) -> NaN.

    The negative codes are frequency levels, not missing values
    (ordering 0 < -131 < -141), so they must not be cleaned as missing.
    """
    m = cfg["codings"]["exposure_ordinal"]
    s = pd.to_numeric(series, errors="coerce")
    out = s.map(m)
    out[s == cfg["codings"]["exposure_dk"]] = np.nan
    return out


def clean_year(series, cfg, ongoing_fill=None):
    """Year column: the 'ongoing' code (-313) is replaced by `ongoing_fill`.

    `ongoing_fill` is the per-person questionnaire year (a Series aligned with
    `series`) or a scalar. Without this, ongoing records give negative
    durations. Years before 1900 are set to missing.
    """
    s = pd.to_numeric(series, errors="coerce")
    og = cfg["codings"]["ongoing"]
    if ongoing_fill is None:
        s = s.mask(s == og, np.nan)
    elif isinstance(ongoing_fill, pd.Series):
        s = s.where(s != og, ongoing_fill.reindex(s.index))
    else:
        s = s.mask(s == og, ongoing_fill)
    return s.mask(s < 1900)


def entry_year(p: dict, cfg: dict, index=None) -> pd.Series:
    """Per-person year in which the employment-history questionnaire was completed.

    Read from the 'When occupational data entered' field (UKB Category 123).
    Used to fill ongoing records and as the third term of `career_end`.
    Missing values fall back to `cohort.fill_year`.
    """
    f = p["data"] / cfg["files"].get("work_env", "")
    col = cfg["cohort"]["entry_date_col"]
    fb = cfg["cohort"]["fill_year"]
    if not f.exists():
        warn(f"missing {f.name}; ongoing records filled with constant {fb}")
        return pd.Series(fb, index=index if index is not None else [],
                         dtype="float64")
    w = pd.read_csv(f, usecols=["eid", col]).set_index("eid")
    y = pd.to_datetime(w[col], errors="coerce").dt.year.astype("float64")
    if index is not None:
        y = y.reindex(index)
    n_miss = int(y.isna().sum())
    if n_miss:
        warn(f"{n_miss:,} participants without entry date; filled with {fb}")
        y = y.fillna(fb)
    return y.rename("entry_year")


def hours_from_cat(series, cfg):
    """Lumped weekly-hours category -> midpoint in hours per week."""
    return pd.to_numeric(series, errors="coerce").map(cfg["codings"]["hours_midpoint"])


def soc_levels(df, col="soc4"):
    """Add soc_l1..soc_l4 columns (integer division of the 4-digit code)."""
    s = pd.to_numeric(df[col], errors="coerce")
    return df.assign(**{k: (s // v) for k, v in SOC_DIVISOR.items()})


def fold_of(eids, n_folds=5) -> np.ndarray:
    """Fold number = md5(eid) % n_folds.

    `eid % n_folds` is avoided because eids carry recruitment-batch structure;
    Python's `hash()` is avoided because string hashing is randomised per process.
    """
    return np.array([int(hashlib.md5(str(int(e)).encode()).hexdigest(), 16) % n_folds
                     for e in eids], dtype="int8")


# -- Reproducibility --------------------------------------------------------
def set_seed(seed, deterministic=True):
    """Seed all RNGs and disable non-deterministic kernels (FP32, no TF32).

    Mixed precision is not used: a non-fixed accumulation order makes the
    validation curve drift between runs, which can move early stopping to a
    different epoch and change the frozen weights. `warn_only=True` because a
    few CUDA kernels have no deterministic implementation.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:      # older torch versions
            warn(f"use_deterministic_algorithms unavailable ({e})")


def env_stamp() -> dict:
    def _v(m):
        try:
            return __import__(m).__version__
        except Exception:
            return None
    git = None
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=CODES,
                             capture_output=True, text=True, timeout=5
                             ).stdout.strip() or None
    except Exception:
        pass
    return {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "git": git, "numpy": _v("numpy"), "pandas": _v("pandas"),
            "torch": _v("torch"), "sklearn": _v("sklearn"),
            "precision": "fp32", "deterministic": True}


# -- Representation freeze: weight hash and verification --------------------
def model_hash(model) -> str:
    """SHA256 over all parameter tensors, in sorted name order."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def save_latent(path: Path, eids, z, meta: dict):
    df = pd.DataFrame(z, columns=[f"z{i:03d}" for i in range(z.shape[1])])
    df.insert(0, "eid", np.asarray(eids, dtype="int64"))
    df.to_parquet(path, index=False)
    meta = {**meta, "env": env_stamp(), "n": int(len(df)), "d": int(z.shape[1])}
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    ok(f"{path.name}  {len(df):,} x {z.shape[1]}  "
       f"sha {meta.get('model_sha256', 'NA')[:12]}...")


def load_latent(path: Path, expect_hash=None, enforce=True):
    """Read a frozen representation and verify its weight hash.

    The hash check prevents downstream analyses from running on a
    representation other than the one that was frozen after training.
    """
    if not path.exists():
        fail(f"representation {path.name} not found; run 03_pretrain.py first")
        sys.exit(1)
    mp = path.with_suffix(".meta.json")
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    if enforce:
        if "model_sha256" not in meta:
            fail(f"{mp.name} has no model_sha256; provenance cannot be verified")
            sys.exit(1)
        if expect_hash and meta["model_sha256"] != expect_hash:
            fail(f"weight hash mismatch (expected {expect_hash[:12]}..., got "
                 f"{meta['model_sha256'][:12]}...); the representation has changed")
            sys.exit(1)
    df = pd.read_parquet(path)
    zc = [c for c in df.columns if c.startswith("z")]
    return df["eid"].to_numpy(), df[zc].to_numpy(dtype="float32"), meta


# -- Representation diagnostics ---------------------------------------------
def effective_rank(z) -> float:
    """Participation ratio of the covariance spectrum: (sum l)^2 / sum l^2."""
    zc = z - z.mean(0, keepdims=True)
    ev = np.linalg.svd(zc, compute_uv=False) ** 2
    return float(ev.sum() ** 2 / (ev ** 2).sum())


# -- Logging ----------------------------------------------------------------
class Tee:
    """Write to stdout and to a UTF-8 log file at the same time.

    Characters the console cannot encode are replaced on the console only.
    Every write is flushed so that long runs can be followed in the log.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def __enter__(self):
        self._o = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *a):
        sys.stdout = self._o
        self.f.close()

    def write(self, s):
        try:
            self._o.write(s)
        except UnicodeEncodeError:
            enc = getattr(self._o, "encoding", None) or "ascii"
            self._o.write(s.encode(enc, errors="replace").decode(enc))
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self._o.flush()
        self.f.flush()


def tee(p: dict, name: str):
    return Tee(p["logs"] / f"{name}.log")


def h1(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def h2(t):
    print(f"\n-- {t} " + "-" * max(0, 66 - len(t)))


def ok(m):
    print(f"  [OK]   {m}")


def warn(m):
    print(f"  [WARN] {m}")


def fail(m):
    print(f"  [FAIL] {m}")


def import_numbered(stem: str):
    """Import a module whose file name starts with a digit (e.g. 03_pretrain)."""
    import importlib.util
    p = HERE / f"{stem}.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location(stem.replace("_", "") + "_m", p)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def require_leiden():
    """Leiden is the only clustering algorithm; exit if its dependencies are missing."""
    try:
        import igraph
        import leidenalg
        del igraph, leidenalg
        return True
    except ImportError:
        fail("python-igraph / leidenalg not installed (required; no fallback). "
             "conda install -c conda-forge python-igraph leidenalg")
        return False
