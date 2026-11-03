"""a5: lifetime exposure intensities vs clock age gaps.

Exposure: {axis}__life from a0 (qb_i{inst}_{sex}) or, for C-L, from a1b (qb_i0all_{sex}) -- the mean annual
intensity over working years before measurement (missing if fewer than qb.min_years such years).
Axes: qb.axes (low occupational status, night shifts, long working hours and eight agents; axis_group
"fig") and qb.aux_axes (the annual share of the ten agents with any exposure; axis_group "aux", its own
FDR family).

Models, per sex x family x sample x axis x clock: gapz ~ z(axis__life) + covariates (common_aging.design),
FWL with HC1 standard errors. Samples as in a4 (left-work and full for blood-measured families; full for
MRI). Grey (not tested): fewer than qb.grey_min_n participants with exposure > 0, or no variance.
BH-FDR within sex x family x sample x axis group x clock over tested cells, alpha = qb.fdr_alpha.

Table (tables/): a5_life.csv

  python aging/py/a5_lifecourse.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common_aging import (FAMILIES, _basis, design, fdr_within, fwl, h1, h2, load_frame, load_status, ok,
                          qb_file, restrict, samples_for, setup, tee, warn, z_to_p)


class QCache:
    """Orthonormal basis of Z per row mask, computed once (most clocks share the missingness pattern)."""

    def __init__(self, Z):
        self.Z, self.c = Z, {}

    def get(self, m):
        key = hash(m.tobytes())
        if key not in self.c:
            self.c[key] = _basis(self.Z[m])
        return self.c[key]


def life_assoc(df, D, cfg, fam, clocks, axes, status, meta):
    """gapz ~ z(axis__life) + covariates, one axis at a time."""
    qb = cfg["qb"]
    qc = QCache(D["Z"])
    rows = []
    for ax in axes:
        c = f"{ax}__life"
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy("float64")
        for ck in clocks:
            y = df[f"gapz__{ck}"].to_numpy("float64")
            m = np.isfinite(y) & np.isfinite(v)
            if m.sum() < 50:
                continue
            x = v[m]
            sd = float(x.std())
            n_exp = int((x > 0).sum())
            grey = n_exp < qb["grey_min_n"] or not sd > 0
            b = s = np.nan
            if sd > 0:
                r = fwl(y[m], (x - x.mean()) / sd, Q=qc.get(m))
                b, s = float(r["beta"][0]), float(r["se"][0])
            sdy = float(np.nanstd(df.loc[m, f"gapy__{ck}"]))
            rows.append({**meta, "axis": ax, "axis_group": "fig" if ax in qb["axes"] else "aux",
                         "clock": ck, "n": int(m.sum()), "n_exposed": n_exp, "exposure_sd": sd,
                         "beta": b, "se": s, "z": b / s if s and np.isfinite(s) else np.nan,
                         "gap_sd_years": sdy, "beta_years": b * sdy, "grey": grey,
                         "main": bool(status.get(fam, {}).get(ck, {}).get("main"))})
    return rows


def main():
    args, cfg, up, p = setup(__doc__)
    qb = cfg["qb"]
    axes = list(dict.fromkeys(qb["axes"] + qb["aux_axes"]))
    with tee(p, "a5_lifecourse"):
        h1("a5 lifetime exposure intensities x clocks")
        status = load_status(p)
        life_rows = []
        for sex in ["Female", "Male"]:
            for fam in FAMILIES:
                if not status.get(fam, {}).get("_family", {}).get("present"):
                    continue
                qf = qb_file(p, fam, sex)
                if not qf.exists():
                    warn(f"missing {qf.name}; {fam} skipped")
                    continue
                qp = pd.read_parquet(qf)
                df0, clocks = load_frame(p, cfg, fam, sex)
                df0 = df0.merge(qp, on="eid", how="left")
                for smp in samples_for(fam, cfg):
                    df = restrict(df0, fam, smp)
                    h2(f"{sex} / {fam} / {smp}: n = {len(df):,}")
                    if len(df) < 50:
                        warn("sample too small; skipped")
                        continue
                    meta = {"sex": sex, "family": fam, "sample": smp, "timing": FAMILIES[fam]["timing"]}
                    D = design(df, cfg, fam)
                    life_rows += life_assoc(df, D, cfg, fam, clocks, axes, status, meta)

        h2("FDR (within clock)")
        lf = pd.DataFrame(life_rows)
        if len(lf):
            lf["p"] = z_to_p(lf["z"].to_numpy())
            lf["q"] = fdr_within(lf, ["sex", "family", "sample", "axis_group", "clock"], ~lf["grey"])
            lf["sig"] = (lf["q"] < qb["fdr_alpha"]) & ~lf["grey"]
            lf.to_csv(p["tables"] / "a5_life.csv", index=False)
            print(lf.groupby(["sex", "family", "sample"]).agg(
                cells=("p", "size"), grey=("grey", "sum"), sig=("sig", "sum"),
                p05=("p", lambda v: int((v < .05).sum()))).to_string())
        ok("a5 done")


if __name__ == "__main__":
    main()
