"""m5 (Python part): preranked t table of the Olink layer for R/m5_fgsea.R. Requires m3.

Input: m3 Olink cells in the full sample, not grey, for the analyses in config enrich.analyses (type by
default). One t = beta / SE vector per sex x term; gene symbol = feature name without the _N suffix; when a
gene is measured by several assays the one with the largest |t| is kept. The background is every Olink gene
measured for that term. The enrichment itself (GO BP / CC / MF and KEGG) runs in R.

Output
  derived/m5_t.csv.gz   sex, term, analysis, gene, t, beta, se

  python molecular/py/m5_enrich.py
  Rscript molecular/R/m5_fgsea.R          # next
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common_mol import h1, need, ok, setup, tee, warn


def gene_of(feature):
    return pd.Series(feature).astype(str).str.replace(r"_\d+$", "", regex=True).str.upper().to_numpy()


def t_table(res, cfg):
    d = res[(res["layer"] == "LP") & (res["sample"] == "full") & ~res["grey"].astype(bool)]
    d = d[d["analysis"].isin(cfg["enrich"]["analyses"])].copy()
    d["t"] = d["beta"] / d["se"]
    d = d[np.isfinite(d["t"])]
    d["gene"] = gene_of(d["feature"])
    d["abs_t"] = d["t"].abs()
    d = d.sort_values("abs_t", ascending=False).drop_duplicates(["sex", "term", "gene"])
    return d[["sex", "term", "analysis", "gene", "t", "beta", "se"]].reset_index(drop=True)


def main():
    args, cfg, up, p = setup(__doc__)
    with tee(p, "m5_enrich"):
        h1("m5 Olink t table for fgsea")
        res = pd.read_csv(need(p["tables"] / "m3_assoc.csv.gz", "m3_assoc.py"), low_memory=False)
        tt = t_table(res, cfg)
        if not len(tt):
            warn("no non-grey Olink cells in m3; nothing to do")
            return
        tt.to_csv(p["derived"] / "m5_t.csv.gz", index=False)
        ok(f"derived/m5_t.csv.gz: {tt.groupby(['sex', 'term']).ngroups} sex x term vectors, "
           f"median {int(tt.groupby(['sex', 'term']).size().median())} genes each")


if __name__ == "__main__":
    main()
