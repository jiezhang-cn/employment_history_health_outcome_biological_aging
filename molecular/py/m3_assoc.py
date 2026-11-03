"""m3: per-feature associations of career types, career axes and exposure-years with the four layers.

  Terms   type vs all other types of the same sex; career axis (+1 SD); exposure-years before the
          measurement (per 10 years; model X2 additionally adjusts for pre-measurement mean CAMSIS)
  Samples blood layers: full and restricted (career ended before the blood draw);
          brain MRI: full and cn (free of chronic disease at the imaging visit)
  Model   feature_z ~ exposure + Z, by sex, one model per feature (FWL + HC1 standard errors)
  FDR     BH within sex x layer x modality x sample x analysis x model x term; cells with < 50 participants
          in the type or the rest (or < 50 exposed) are grey: not tested and not counted in the family
  Label   robust = full-sample q < alpha and same sign with p < 0.05 in the secondary sample;
          discovery = full-sample q < alpha only

Outputs
  tables/m3_assoc.csv.gz   every cell (beta / se / z / p / q / sig / label / n / n1 / n0 / grey, plus the
                           secondary-sample beta and p on full-sample rows)
  tables/m3_summary.csv    sex x layer x modality x analysis x model x term (full sample): features tested,
                           nominal p < 0.05 (all / beta > 0; exp_nom = expected under the null),
                           FDR-significant (all / beta > 0), robust

  python molecular/py/m3_assoc.py [--layers LB,LN,LP,LI]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common_mol import (ARM_OF, FDR_BY, LAYERS, add_q, h1, load_layer, nominal, ok, run_terms, setup, sexes,
                        tee, type_levels, warn)

SECOND = {"blood": "restricted", "brain": "cn"}


def label_cells(res):
    """Add label (robust / discovery / "") and the secondary-sample beta / p to full-sample rows."""
    key = ["sex", "layer", "modality", "analysis", "model", "term", "feature"]
    out = []
    for arm, sec in SECOND.items():
        L = [l for l, a in ARM_OF.items() if a == arm]
        f = res[(res["layer"].isin(L)) & (res["sample"] == "full")].copy()
        s = res[(res["layer"].isin(L)) & (res["sample"] == sec)][key + ["beta", "p"]].rename(
            columns={"beta": "beta_second", "p": "p_second"})
        f = f.merge(s, on=key, how="left")
        rep = (np.sign(f["beta_second"]) == np.sign(f["beta"])) & (f["p_second"] < 0.05)
        f["second_sample"] = sec
        f["label"] = np.where(f["sig"] & rep, "robust", np.where(f["sig"], "discovery", ""))
        out.append(f)
    lab = pd.concat(out, ignore_index=True)
    rest = res[res["sample"] != "full"].assign(label="")
    return pd.concat([lab, rest], ignore_index=True)


def estimate(p, cfg, up, layers):
    parts = []
    for sex in sexes(up):
        types = type_levels(p, sex)
        for layer in layers:
            for smp in cfg["samples"][ARM_OF[layer]]:
                try:
                    df, X, meta = load_layer(p, cfg, layer, sex, smp)
                except SystemExit:
                    warn(f"{layer} {sex}: feature table missing; skipped")
                    break
                if len(df) < 100:
                    warn(f"{layer} {sex} {smp}: {len(df)} participants; skipped")
                    continue
                r = run_terms(df, X, meta, cfg, types, layer, sex, smp)
                parts.append(r)
                print(f"    {sex} {layer} {smp}: {len(df):,} participants x {X.shape[1]} features x "
                      f"{r['term'].nunique() if len(r) else 0} terms -> {len(r):,} cells")
    res = add_q(pd.concat(parts, ignore_index=True), cfg["alpha"], FDR_BY)
    if set(layers) != set(LAYERS) and (p["tables"] / "m3_assoc.csv.gz").exists():
        # partial run: keep other layers from the existing table (BH families never span layers)
        old = pd.read_csv(p["tables"] / "m3_assoc.csv.gz", low_memory=False)
        res = pd.concat([old[~old["layer"].isin(layers)].drop(columns=["label", "beta_second", "p_second",
                                                                       "second_sample"], errors="ignore"),
                         res], ignore_index=True)
        warn(f"only {layers} were estimated; other layers taken from the existing m3_assoc")
    return res


def main():
    args, cfg, up, p = setup(__doc__, [(("--layers",), {"default": ",".join(LAYERS)})])
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    with tee(p, "m3_assoc"):
        h1("m3 per-feature associations")
        res = label_cells(estimate(p, cfg, up, layers))
        res.to_csv(p["tables"] / "m3_assoc.csv.gz", index=False)
        ok(f"m3_assoc.csv.gz: {len(res):,} cells")
        f = res[res["sample"] == "full"]
        f = f.assign(nom=nominal(f, cfg), up=f["beta"] > 0)
        summ = f.groupby(["sex", "layer", "modality", "analysis", "model", "term"]).agg(
            n_feat=("feature", "size"), n_grey=("grey", "sum"),
            n_nom=("nom", "sum"), n_nom_up=("nom", lambda s: int((s & f.loc[s.index, "up"]).sum())),
            n_sig=("sig", "sum"), n_up=("sig", lambda s: int((s & f.loc[s.index, "up"]).sum())),
            n_robust=("label", lambda s: int((s == "robust").sum()))).reset_index()
        summ["exp_nom"] = cfg["nominal_p"] * (summ["n_feat"] - summ["n_grey"])
        summ.to_csv(p["tables"] / "m3_summary.csv", index=False)
        print(summ.groupby(["sex", "layer", "analysis"])[["n_nom", "exp_nom", "n_sig", "n_robust"]].sum()
              .to_string(float_format=lambda v: f"{v:.0f}"))
        ok("m3 done")


if __name__ == "__main__":
    main()
