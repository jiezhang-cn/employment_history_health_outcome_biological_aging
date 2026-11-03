"""m1: feature preprocessing for the four layers -> feat_{layer}_{sex}.parquet + feat_meta_{layer}.csv.

Transformation rules (log / binarisation) are set on both sexes pooled; trimming and z-scoring are done
within sex. No exposure is used.

  LB  clinical biochemistry + blood counts (m_e1_blood_bc, instance 0)
      Values flagged as not reportable -> missing; oestradiol and rheumatoid factor enter only as
      "detectable" binaries; eGFR (CKD-EPI 2021, creatinine + cystatin C) and urine albumin/creatinine ratio
      (albumin below the detection limit filled with LOD x 0.5); measures whose modal value covers > 50% of
      participants -> binary "> mode"; skewness > 1 -> log; trim at 6 robust SD; z-score.
  LN  107 non-derived NMR measures (m_e2_nmr_i0, after ukbnmr technical-variation removal)
      -> log(x + offset) -> trim -> z-score.
  LP  Olink NPX from the raw long table (olink_data) x UKB coding 143; no imputation; assays missing in
      > 50% of a sex removed -> trim -> z-score.
  LI  139 regional grey-matter volumes (FAST), FA of 48 tracts (TBSS) and white-matter hyperintensity
      volume (log1p) -> trim -> z-score.

Layer-specific technical covariates are stored in the feature tables (tech__* enter design(), cat__* enter
design_mol()):
  LB / LN / LP: tech__fasting_time, tech__draw_hour; LP also tech__olink_nprot and cat__olink_plate
  LI: config li.tech (head-size scaling, T1 quality, head motion, scanner position)

Outputs
  derived/feat_{layer}_{sex}.parquet, derived/feat_meta_{layer}.csv
  tables/m1_vm1_qc.csv       sex x layer x feature: raw n, not reportable, trimmed, final n, transform
  tables/m1_vm1_dropped.csv  features dropped and why

  python molecular/py/m1_features.py [--layers LB,LN,LP,LI] [--olink-long FILE]
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common_mol import (LAYERS, brain_ref, feat_file, first_nonnull, frame_file, h1, h2, mad_trim, multi_arr,
                        need, ok, read_m, ref_path, setup, sexes, tee, warn, zcol)

QC_ROWS, DROPPED = [], []


# ── Shared ───────────────────────────────────────────────────────────
def cohort(p, arm, up):
    """Both sexes' analysis frames (m0): eid -> (sex, age_meas)."""
    fr = []
    for sex in sexes(up):
        d = pd.read_parquet(need(frame_file(p, arm, sex), "m0_assemble.py"), columns=["eid", "age_meas"])
        fr.append(d.assign(sex=sex))
    return pd.concat(fr, ignore_index=True).drop_duplicates("eid").set_index("eid")


def finalize(X: pd.DataFrame, scale: dict, kmad, layer, sex, n_raw=None, n_rep=None, transform=None):
    """Within sex: trim + z-score continuous features; binaries kept as 0/1. Records the QC row."""
    out = {}
    for c in X.columns:
        v = X[c].to_numpy("float64")
        nr = int(np.isfinite(v).sum())
        if scale.get(c) == "binary":
            out[c], ntr = v, 0
        else:
            v, ntr = mad_trim(v, kmad)
            out[c] = zcol(v)
        QC_ROWS.append({"sex": sex, "layer": layer, "feature": c,
                        "n_raw": int(n_raw.get(c, nr)) if n_raw else nr,
                        "n_unreportable": int(n_rep.get(c, 0)) if n_rep else 0,
                        "n_trim": ntr, "n_final": int(np.isfinite(out[c]).sum()),
                        "scale": scale.get(c, "z"), "transform": (transform or {}).get(c, "")})
    return pd.DataFrame(out, index=X.index).astype("float32")


def skew_log(v: pd.Series, thr):
    """Skewness > thr -> log (shifted to half the smallest positive value if zeros / negatives occur).
    Returns (v, transform name)."""
    from scipy.stats import skew
    x = v.dropna().to_numpy("float64")
    if len(x) < 20 or not np.isfinite(skew(x)) or skew(x) <= thr:
        return v, "identity"
    pos = x[x > 0]
    if not len(pos):
        return v, "identity"
    if x.min() > 0:
        return np.log(v), "log"
    sh = pos.min() / 2 - min(x.min(), 0)
    return np.log(v + sh), f"log(x+{sh:.3g})"


def tech_blood(p, cfg) -> pd.DataFrame:
    """Fasting time (field 74) and time of blood draw (field 3166 -> hour of day); shared by LB / LN / LP."""
    bc = read_m(p, "m_e1_blood_bc", required=False)
    if bc is None:
        return pd.DataFrame()
    bc = bc.set_index("eid")
    out = pd.DataFrame(index=bc.index)
    if "fasting_time" in cfg["lb"]["tech_fields"]:
        out["tech__fasting_time"] = pd.to_numeric(first_nonnull(bc, multi_arr(bc, "fasting_time", 0)),
                                                  errors="coerce").where(lambda s: s >= 0)
    if "draw_time" in cfg["lb"]["tech_fields"]:
        t = pd.to_datetime(first_nonnull(bc, multi_arr(bc, "draw_time", 0)), errors="coerce")
        out["tech__draw_hour"] = t.dt.hour + t.dt.minute / 60
    return out.astype("float32")


# ── LB ───────────────────────────────────────────────────────────────
def egfr_ckdepi2021(scr_umol, cys, age, female):
    """CKD-EPI 2021 creatinine-cystatin C equation (Inker et al. 2021). Creatinine umol/L -> mg/dL (/ 88.4)."""
    scr = np.asarray(scr_umol, dtype="float64") / 88.4
    cys = np.asarray(cys, dtype="float64")
    age = np.asarray(age, dtype="float64")
    female = np.asarray(female, dtype=bool)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.219, -0.144)
    r = scr / k
    e = (135 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -0.544
         * np.minimum(cys / 0.8, 1) ** -0.323 * np.maximum(cys / 0.8, 1) ** -0.778
         * 0.9961 ** age * np.where(female, 0.963, 1.0))
    return np.where((scr > 0) & (cys > 0), e, np.nan)


def layer_lb(p, cfg, up, tech):
    h2("LB clinical biochemistry + blood counts")
    lb = cfg["lb"]
    co = cohort(p, "blood", up)
    bc = read_m(p, "m_e1_blood_bc").set_index("eid")
    bc = bc[bc.index.isin(co.index)]
    fields = cfg["extract"]["groups"]["blood_bc"]["fields"]
    shorts = [s for s in fields.values() if not s.endswith("_flag") and s not in lb["tech_fields"]]
    low = set(float(x) for x in lb["reportability_low"])
    okv = set(float(x) for x in lb["reportable_ok"])
    raw, n_raw, n_rep, scale, trans = {}, {}, {}, {}, {}
    for s in shorts:
        c = f"{s}_i0"
        if c not in bc.columns:
            DROPPED.append({"layer": "LB", "feature": s, "reason": "column not in m_e1 extract"})
            continue
        v = pd.to_numeric(bc[c], errors="coerce")
        n_raw[s] = int(v.notna().sum())
        rc = f"{s}_reportability_i0"
        rep = pd.to_numeric(bc[rc], errors="coerce") if rc in bc.columns else pd.Series(np.nan, index=bc.index)
        bad = v.notna() & rep.notna() & ~rep.isin(okv)
        n_rep[s] = int(bad.sum())
        v = v.mask(bad)
        if s in lb["detect_markers"]:
            # detectable = 1 if a value exists; 0 if not reportable because below the assay range
            d = pd.Series(np.where(v.notna(), 1.0, np.where(rep.isin(low), 0.0, np.nan)), index=bc.index)
            raw[f"{s}_detect"], scale[f"{s}_detect"], trans[f"{s}_detect"] = d, "binary", "detectable"
            n_raw[f"{s}_detect"] = int(d.notna().sum())
            continue
        raw[s] = v
    raw = pd.DataFrame(raw)
    # Derived measures
    if lb.get("egfr") and {"creatinine", "cystatin"} <= set(raw.columns):
        fem = (co.loc[raw.index, "sex"] == "Female").to_numpy()
        raw["egfr"] = egfr_ckdepi2021(raw["creatinine"], raw["cystatin"], co.loc[raw.index, "age_meas"], fem)
        trans["egfr"] = "CKD-EPI 2021 (Scr + CysC)"
    if {"u_alb", "u_creat"} <= set(raw.columns):
        ua = raw["u_alb"].copy()
        fl = first_nonnull(bc, multi_arr(bc, "u_alb_flag", 0))
        below = ua.isna() & fl.notna() & raw["u_creat"].notna()
        ua[below] = lb["acr"]["lod"] * lb["acr"]["below_lod_fill"]
        raw["acr"] = ua / (raw["u_creat"] / 1000.0)           # mg/mmol
        trans["acr"] = (f"u_alb/u_creat (< LOD filled with {lb['acr']['lod'] * lb['acr']['below_lod_fill']:g}, "
                        f"{int(below.sum()):,} participants)")
    for s in lb["exclude"]:
        if s in raw.columns:
            raw = raw.drop(columns=s)
            DROPPED.append({"layer": "LB", "feature": s, "reason": "config lb.exclude"})
    # Transformation rules: set on both sexes pooled
    for c in [c for c in raw.columns if scale.get(c) != "binary"]:
        v = raw[c]
        if v.notna().sum() < 50:
            DROPPED.append({"layer": "LB", "feature": c, "reason": f"non-missing {int(v.notna().sum())} < 50"})
            raw = raw.drop(columns=c)
            continue
        vc = v.value_counts(normalize=True)
        if vc.iloc[0] > lb["binarise_mode_share"]:
            md = vc.index[0]
            raw[c] = (v > md).astype(float).where(v.notna())
            scale[c], trans[c] = "binary", f"> mode ({md:g}; share {vc.iloc[0]:.0%})"
            continue
        raw[c], trans[c] = skew_log(v, lb["skew_log"])
    meta = pd.DataFrame({"feature": raw.columns, "layer": "LB", "modality": "LB", "label": list(raw.columns),
                         "group": ["blood count" if re.match(r"^(wbc|rbc|hgb|hct|mc|rdw|plt|pct|mpv|pdw|lymph|mono|neut|"
                                                             r"eos|baso|nrbc|retic|mrv|mscv|irf|hlr)", c) else
                                   "urine" if c == "acr" else "biochemistry" for c in raw.columns],
                         "scale": [scale.get(c, "z") for c in raw.columns],
                         "transform": [trans.get(c, "") for c in raw.columns]})
    write_layer(p, up, "LB", raw, meta, scale, tech, co, cfg["lb"]["trim_mad"], n_raw, n_rep, trans)


# ── LN ───────────────────────────────────────────────────────────────
def nmr_offsets(p, names) -> dict:
    """Log offsets reported by ukbnmr; NaN where unavailable (then half the smallest positive value is used)."""
    f = p["mextract"] / "m_e2_nmr_log_offset.csv"
    off = {}
    if f.exists():
        lo = pd.read_csv(f)
        oc = next((c for c in lo.columns if "offset" in c.lower()), None)
        bc = next((c for c in lo.columns if "biomarker" in c.lower()), None) or lo.columns[0]
        if oc is not None:
            off = dict(zip(lo[bc].astype(str), pd.to_numeric(lo[oc], errors="coerce")))
        else:
            warn(f"{f.name} has no offset column ({list(lo.columns)}); using half the smallest positive value")
    return {n: off.get(n, np.nan) for n in names}


def layer_ln(p, cfg, up, tech):
    h2("LN NMR metabolomics (107 non-derived, ukbnmr-corrected)")
    co = cohort(p, "blood", up)
    nm = read_m(p, "m_e2_nmr_i0").set_index("eid")
    nm = nm[nm.index.isin(co.index)]
    info_f = p["mextract"] / "m_e2_nmr_info.csv"
    info = pd.read_csv(info_f) if info_f.exists() else pd.DataFrame({"Biomarker": nm.columns})
    info = info[info["Biomarker"].isin(nm.columns)]
    names = list(info["Biomarker"])
    if len(names) != cfg["ln"]["expected_n"]:
        warn(f"LN has {len(names)} features, expected {cfg['ln']['expected_n']}")
    off = nmr_offsets(p, names)
    raw, trans, n_raw = {}, {}, {}
    for b in names:
        v = pd.to_numeric(nm[b], errors="coerce")
        n_raw[b] = int(v.notna().sum())
        o = off.get(b)
        if not (o is not None and np.isfinite(o)):
            pos = v[v > 0]
            o = float(pos.min() / 2) if len(pos) else 1.0
            src = "half-min"
        else:
            src = "ukbnmr"
        o = o if (v.min() + o) > 0 else float(-v.min() + (v[v > 0].min() / 2 if (v > 0).any() else 1.0))
        raw[b] = np.log(v + o)
        trans[b] = f"log(x+{o:.3g}) [{src}]"
    raw = pd.DataFrame(raw)
    g = lambda c: info[c].tolist() if c in info.columns else [""] * len(info)       # noqa: E731
    meta = pd.DataFrame({"feature": names, "layer": "LN", "modality": "LN",
                         "label": [d if isinstance(d, str) and d else b for b, d in zip(names, g("Description"))],
                         "group": g("Group"), "subgroup": g("Sub.Group"), "units": g("Units"),
                         "scale": "z", "transform": [trans[b] for b in names]})
    write_layer(p, up, "LN", raw, meta, {}, tech, co, cfg["ln"]["trim_mad"], n_raw, None, trans)


# ── LP ───────────────────────────────────────────────────────────────
def olink_long_file(p, cfg, override=None) -> Path:
    f = Path(override) if override else p["data"] / cfg["files"]["olink_long"]
    if not f.exists():
        raise SystemExit(f"Olink long table not found: {f} (set files.olink_long or use --olink-long)")
    return f


def olink_long_sep(f: Path):
    """Delimiter and header of the long table (the UKB-PPP olink_data.txt is tab-delimited)."""
    with (gzip.open(f, "rt") if f.suffix == ".gz" else open(f, encoding="utf-8")) as fh:
        first = fh.readline()
    sep = "\t" if "\t" in first else ","
    return sep, [h.strip().strip('"') for h in first.rstrip("\n").split(sep)]


def coding143_map(cfg) -> pd.DataFrame:
    """UKB coding 143: assay code -> gene symbol and protein name (meaning = "GENE;Protein name")."""
    c = pd.read_csv(ref_path(cfg, cfg["files"]["coding143"]), sep="\t")
    if not {"coding", "meaning"} <= set(c.columns):
        raise SystemExit(f"coding 143 file lacks coding / meaning columns: {list(c.columns)}")
    c["gene"] = c["meaning"].astype(str).str.split(";").str[0].str.strip().str.upper()
    dup = c["gene"].duplicated(keep=False)
    if dup.any():                                         # same gene on several assays -> numbered suffix
        c.loc[dup, "gene"] = c.loc[dup, "gene"] + "_" + (c[dup].groupby("gene").cumcount() + 1).astype(str)
        warn(f"coding 143: {int(dup.sum())} assays share a gene symbol; suffixed (e.g. {c.loc[dup, 'gene'].iloc[0]})")
    c["name"] = c["meaning"].astype(str).str.split(";").str[1:].str.join(";").str.strip()
    return c.set_index("coding")[["gene", "name"]]


def olink_wide(p, cfg, eids, override=None) -> pd.DataFrame:
    """Long table -> eids x assay NPX matrix (filled chunk by chunk into a preallocated array). Cached in
    derived/."""
    f = olink_long_file(p, cfg, override)
    cmap = coding143_map(cfg)
    cache = p["derived"] / "olink_i0_cohort.parquet"
    stamp = {"src": str(f), "mtime": f.stat().st_mtime, "n_eid": len(eids)}
    sf = cache.with_suffix(".json")
    if cache.exists() and sf.exists() and json.loads(sf.read_text(encoding="utf-8")) == stamp:
        ok(f"reading cache {cache.name}")
        return pd.read_parquet(cache)
    sep, hdr = olink_long_sep(f)
    need_cols = ["eid", "ins_index", "protein_id", "result"]
    if not set(need_cols) <= set(hdr):
        raise SystemExit(f"{f.name} is not an olink_data long table: header {hdr[:8]}")
    eids = np.asarray(sorted(eids), dtype="int64")
    codes = np.asarray(sorted(cmap.index), dtype="int64")
    A = np.full((len(eids), len(codes)), np.nan, dtype="float32")
    inst = int(cfg["lp"]["instance"])
    n_read = n_hit = 0
    for ch in pd.read_csv(f, sep=sep, usecols=need_cols, chunksize=cfg["lp"]["long_chunk_rows"],
                          dtype={"eid": "int64", "ins_index": "int16", "protein_id": "int32"}):
        n_read += len(ch)
        ch = ch[ch["ins_index"] == inst]
        ri = np.searchsorted(eids, ch["eid"].to_numpy())
        ci = np.searchsorted(codes, ch["protein_id"].to_numpy())
        okm = ((ri < len(eids)) & (eids[np.minimum(ri, len(eids) - 1)] == ch["eid"].to_numpy())
               & (ci < len(codes)) & (codes[np.minimum(ci, len(codes) - 1)] == ch["protein_id"].to_numpy()))
        A[ri[okm], ci[okm]] = pd.to_numeric(ch["result"], errors="coerce").to_numpy("float32")[okm]
        n_hit += int(okm.sum())
    print(f"    {f.name}: {n_read:,} rows read, {n_hit:,} values in the cohort")
    X = pd.DataFrame(A, index=pd.Index(eids, name="eid"), columns=cmap.loc[codes, "gene"].to_numpy())
    X = X[X.notna().any(axis=1)]
    absent = X.columns[~X.notna().any(axis=0)]
    if len(absent):                                       # in coding 143 but without any value in the table
        print(f"    {len(absent):,} of {len(codes):,} coding-143 assays have no values; not kept")
        X = X.drop(columns=absent)
    X.to_parquet(cache)
    sf.write_text(json.dumps(stamp), encoding="utf-8")
    return X


def layer_lp(p, cfg, up, tech, override=None):
    h2("LP Olink proteomics (NPX from the raw long table)")
    co = cohort(p, "blood", up)
    X = olink_wide(p, cfg, set(co.index), override)
    print(f"    cohort participants with Olink data: {len(X):,}")
    cmap = coding143_map(cfg).set_index("gene")["name"]
    om = read_m(p, "m_e1_olink_meta", required=False)
    side = pd.DataFrame(index=X.index)
    if om is not None:
        om = om.set_index("eid").reindex(X.index)
        for s, nm in (("olink_nprot", "tech__olink_nprot"), ("olink_plate", "cat__olink_plate")):
            cols = multi_arr(om, s, 0)
            if cols:
                v = first_nonnull(om, cols)
                side[nm] = pd.to_numeric(v, errors="coerce") if nm.startswith("tech__") else v.astype("object")
    else:
        warn("m_e1_olink_meta not found; LP models are not adjusted for plate / number of proteins")
    meta = pd.DataFrame({"feature": X.columns, "layer": "LP", "modality": "LP",
                         "label": [cmap.get(g, g) for g in X.columns], "scale": "z", "transform": "NPX"})
    write_layer(p, up, "LP", X, meta, {}, tech, co, cfg["ln"]["trim_mad"], None, None, None,
                side=side, max_missing=cfg["lp"]["max_missing_frac"])


# ── LI ───────────────────────────────────────────────────────────────
def layer_li(p, cfg, up):
    h2("LI brain MRI (GMV 139 / FA 48 / WMH)")
    li = cfg["li"]
    co = cohort(p, "brain", up)
    br = brain_ref(p)
    be = read_m(p, "m_e1_brain_extra").set_index("eid")
    be = be[be.index.isin(co.index)]
    raw, trans, scale = {}, {}, {}
    for r in br.itertuples():
        c = f"{r.short_name}_i2"
        if c not in be.columns:
            DROPPED.append({"layer": "LI", "feature": r.short_name, "reason": "column not in m_e1 extract"})
            continue
        raw[r.short_name] = pd.to_numeric(be[c], errors="coerce")
    extra = cfg["extract"]["groups"]["brain_extra"]["fields"]
    for s in extra.values():
        if s in li["tech"]:
            continue
        c = f"{s}_i2"
        if c in be.columns:
            v = pd.to_numeric(be[c], errors="coerce")
            raw[s], trans[s] = (np.log1p(v.clip(lower=0)), "log1p") if s in li["log1p"] else (v, "identity")
    raw = pd.DataFrame(raw, index=be.index)
    side = pd.DataFrame(index=be.index)
    for s in li["tech"]:
        c = f"{s}_i2"
        if c in be.columns:
            side[f"tech__{s}"] = pd.to_numeric(be[c], errors="coerce")
        else:
            warn(f"LI technical covariate {s} not in m_e1_brain_extra")
    rows = []
    bmeta = br.set_index("short_name")
    for c in raw.columns:
        if c in bmeta.index:
            b = bmeta.loc[c]
            rows.append({"feature": c, "modality": b["modality"], "label": b["label"], "group": b["region"],
                         "side": b["side"], "field": b["field"]})
        else:
            rows.append({"feature": c, "modality": c.upper(), "label": c.upper(), "group": c.upper()})
    meta = pd.DataFrame(rows).assign(layer="LI", scale="z",
                                     transform=lambda d: d["feature"].map(trans).fillna("identity"))
    meta = meta[meta["modality"].isin(li["modalities"])]
    raw = raw[meta["feature"].tolist()]
    write_layer(p, up, "LI", raw, meta, scale, None, co, li["trim_mad"], None, None, trans, side=side)


# ── Write ────────────────────────────────────────────────────────────
def write_layer(p, up, layer, raw, meta, scale, tech, co, kmad, n_raw, n_rep, trans, side=None,
                max_missing=None):
    raw = raw[raw.notna().any(axis=1)]
    keep_all = set()
    for sex in sexes(up):
        ids = co.index[co["sex"] == sex]
        Xs = raw[raw.index.isin(ids)]
        if not len(Xs):
            warn(f"{layer} {sex}: 0 participants")
            continue
        if max_missing is not None:
            mf = Xs.isna().mean()
            drop = mf[mf > max_missing].index
            for c in drop:
                DROPPED.append({"layer": layer, "sex": sex, "feature": c,
                                "reason": f"missing {mf[c]:.0%} > {max_missing:.0%}"})
            Xs = Xs.drop(columns=drop)
            print(f"    {sex}: {len(drop)} assays with > {max_missing:.0%} missing removed")
        Z = finalize(Xs, scale, kmad, layer, sex, n_raw, n_rep, trans)
        keep_all |= set(Z.columns)
        if tech is not None and len(tech):
            Z = Z.join(tech.reindex(Z.index))
        if side is not None and len(side.columns):
            Z = Z.join(side.reindex(Z.index))
        Z.index.name = "eid"
        Z.reset_index().to_parquet(feat_file(p, layer, sex), index=False)
        nt = sum(1 for r in QC_ROWS if r["layer"] == layer and r["sex"] == sex)
        trim = sum(r["n_trim"] for r in QC_ROWS if r["layer"] == layer and r["sex"] == sex)
        print(f"    feat_{layer}_{sex}: {len(Z):,} participants x {nt} features ({trim:,} values trimmed)")
    meta = meta[meta["feature"].isin(keep_all)]
    meta.to_csv(p["derived"] / f"feat_meta_{layer}.csv", index=False)
    ok(f"feat_meta_{layer}.csv: {len(meta)} features"
       + (f" ({meta['modality'].value_counts().to_dict()})" if layer == "LI" else ""))


def main():
    args, cfg, up, p = setup(__doc__, [(("--layers",), {"default": ",".join(LAYERS)}),
                                       (("--olink-long",), {"default": None})])
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    with tee(p, "m1_features"):
        h1("m1 feature preprocessing")
        tech = tech_blood(p, cfg) if {"LB", "LN", "LP"} & set(layers) else None
        if "LB" in layers:
            layer_lb(p, cfg, up, tech)
        if "LN" in layers:
            layer_ln(p, cfg, up, tech)
        if "LP" in layers:
            layer_lp(p, cfg, up, tech, args.olink_long)
        if "LI" in layers:
            layer_li(p, cfg, up)
        qc = pd.DataFrame(QC_ROWS)
        f = p["tables"] / "m1_vm1_qc.csv"
        if f.exists() and set(layers) != set(LAYERS):            # partial run: keep rows of other layers
            old = pd.read_csv(f)
            qc = pd.concat([old[~old["layer"].isin(layers)], qc], ignore_index=True)
        qc.to_csv(f, index=False)
        pd.DataFrame(DROPPED).to_csv(p["tables"] / "m1_vm1_dropped.csv", index=False)
        if len(qc):
            print(qc.groupby(["sex", "layer"]).agg(n_feat=("feature", "size"), n_trim=("n_trim", "sum"),
                                                   n_unrep=("n_unreportable", "sum"),
                                                   median_n=("n_final", "median")).to_string())
        ok("m1 done")


if __name__ == "__main__":
    main()
