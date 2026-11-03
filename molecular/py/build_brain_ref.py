"""Build the brain feature list molecular/ref/brain_features.csv: 139 regional grey-matter volumes (FAST)
and FA of 48 white-matter tracts (TBSS).

This list is the single source of brain field IDs for R/m_e1_extract.R and py/m1_features.py. Short names
follow the UK Biobank Showcase labels (pair / triple ordering described below); the script asserts 139 + 48
features with contiguous field IDs.

  GMV  fields 25782-25920: 48 cortical regions (left, right), 7 subcortical regions (left, right),
       brain stem, cerebellar lobules I-IV and V (left, right), and VI-X (left, vermis, right)
  FA   fields 25056-25103: 6 midline tracts, then 21 tracts (right, left)

Reads no individual-level data:
  python molecular/py/build_brain_ref.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "ref" / "brain_features.csv"

CORTEX = [
    "Frontal Pole", "Insular Cortex", "Superior Frontal Gyrus", "Middle Frontal Gyrus",
    "Inferior Frontal Gyrus, pars triangularis", "Inferior Frontal Gyrus, pars opercularis",
    "Precentral Gyrus", "Temporal Pole", "Superior Temporal Gyrus, anterior division",
    "Superior Temporal Gyrus, posterior division", "Middle Temporal Gyrus, anterior division",
    "Middle Temporal Gyrus, posterior division", "Middle Temporal Gyrus, temporooccipital part",
    "Inferior Temporal Gyrus, anterior division", "Inferior Temporal Gyrus, posterior division",
    "Inferior Temporal Gyrus, temporooccipital part", "Postcentral Gyrus", "Superior Parietal Lobule",
    "Supramarginal Gyrus, anterior division", "Supramarginal Gyrus, posterior division", "Angular Gyrus",
    "Lateral Occipital Cortex, superior division", "Lateral Occipital Cortex, inferior division",
    "Intracalcarine Cortex", "Frontal Medial Cortex",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)", "Subcallosal Cortex",
    "Paracingulate Gyrus", "Cingulate Gyrus, anterior division", "Cingulate Gyrus, posterior division",
    "Precuneous Cortex", "Cuneal Cortex", "Frontal Orbital Cortex", "Parahippocampal Gyrus, anterior division",
    "Parahippocampal Gyrus, posterior division", "Lingual Gyrus", "Temporal Fusiform Cortex, anterior division",
    "Temporal Fusiform Cortex, posterior division", "Temporal Occipital Fusiform Cortex",
    "Occipital Fusiform Gyrus", "Frontal Operculum Cortex", "Central Opercular Cortex",
    "Parietal Operculum Cortex", "Planum Polare", "Heschl's Gyrus (includes H1 and H2)", "Planum Temporale",
    "Supracalcarine Cortex", "Occipital Pole",
]
SUBCORT = ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Ventral Striatum"]
CEREB_PAIR = ["I-IV", "V"]                                          # 25893-25896: (left), (right)
CEREB_TRIPLE = ["VI", "Crus I", "Crus II", "VIIb", "VIIIa", "VIIIb", "IX", "X"]   # 25897-25920: (left), (vermis), (right)
FA_SINGLE = ["middle cerebellar peduncle", "pontine crossing tract", "genu of corpus callosum",
             "body of corpus callosum", "splenium of corpus callosum", "fornix"]
FA_PAIR = ["corticospinal tract", "medial lemniscus", "inferior cerebellar peduncle",
           "superior cerebellar peduncle", "cerebral peduncle", "anterior limb of internal capsule",
           "posterior limb of internal capsule", "retrolenticular part of internal capsule",
           "anterior corona radiata", "superior corona radiata", "posterior corona radiata",
           "posterior thalamic radiation", "sagittal stratum", "external capsule",
           "cingulum cingulate gyrus", "cingulum hippocampus", "fornix cres+stria terminalis",
           "superior longitudinal fasciculus", "superior fronto-occipital fasciculus",
           "uncinate fasciculus", "tapetum"]                       # 25062-25103: (right), (left)
SIDE_WORD = {"L": "left", "R": "right", "V": "vermis", "": ""}


def slug(s: str) -> str:
    s = s.lower().replace("'", "").replace("+", "_")
    s = re.sub(r"\(formerly.*?\)|\(includes.*?\)", "", s)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def row(field, modality, region, side, prefix):
    side_tag = {"L": "_l", "R": "_r", "V": "_vermis", "": ""}[side]
    label = f"{region} ({SIDE_WORD[side]})" if side else region
    return {"field": field, "short_name": f"{prefix}_{slug(region)}{side_tag}", "label": label,
            "modality": modality, "region": region, "side": side}


def build() -> pd.DataFrame:
    rows = []
    f = 25782
    for reg in CORTEX:
        for s in ("L", "R"):
            rows.append(row(f, "GMV", reg, s, "gmv"))
            f += 1
    for reg in SUBCORT:
        for s in ("L", "R"):
            rows.append(row(f, "GMV", reg, s, "gmv"))
            f += 1
    rows.append(row(f, "GMV", "Brain-Stem", "", "gmv"))
    f += 1
    for reg in CEREB_PAIR:
        for s in ("L", "R"):
            rows.append(row(f, "GMV", f"{reg} Cerebellum", s, "gmv"))
            f += 1
    for reg in CEREB_TRIPLE:
        for s in ("L", "V", "R"):
            rows.append(row(f, "GMV", f"{reg} Cerebellum", s, "gmv"))
            f += 1
    assert f == 25921, f
    f = 25056
    for reg in FA_SINGLE:
        rows.append(row(f, "FA", reg, "", "fa"))
        f += 1
    for reg in FA_PAIR:
        for s in ("R", "L"):
            rows.append(row(f, "FA", reg, s, "fa"))
            f += 1
    assert f == 25104, f
    return pd.DataFrame(rows)


def main():
    d = build()
    n = d["modality"].value_counts().to_dict()
    assert n == {"GMV": 139, "FA": 48}, n
    assert d["field"].is_unique and d["short_name"].is_unique
    assert list(d.loc[d.modality == "GMV", "field"]) == list(range(25782, 25921))
    assert list(d.loc[d.modality == "FA", "field"]) == list(range(25056, 25104))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False, encoding="utf-8")
    print(f"brain_features.csv: GMV {n['GMV']} + FA {n['FA']} -> {OUT}")


if __name__ == "__main__":
    main()
