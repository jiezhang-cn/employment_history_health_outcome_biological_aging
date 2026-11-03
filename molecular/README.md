# molecular: career types x molecular and neuroimaging features

Relates career types (type vs all other types of the same sex), seven career axes (+1 SD) and
exposure-years before the measurement (per 10 years) to four feature layers, by sex:

| Layer | Features | Visit |
| ----- | -------- | ----- |
| LB | clinical biochemistry and blood counts (incl. eGFR, urine ACR) | baseline (instance 0) |
| LN | 107 non-derived NMR metabolomic measures, technical variation removed with ukbnmr | baseline |
| LP | Olink plasma proteins (NPX) | baseline |
| LI | 139 regional grey-matter volumes, FA of 48 tracts, white-matter hyperintensity volume | first imaging visit (instance 2) |

Samples: full sample (main) and, in parallel, participants whose career ended before the blood draw (blood
layers) or who were free of chronic disease at the imaging visit (brain MRI).

## Run order

Run from the repository root after the typology and aging modules (this module reads
`results/typology/derived/` and `results/aging/derived/person_{sex}.parquet`, `measure.parquet`).

```bash
python  molecular/py/build_brain_ref.py      # writes molecular/ref/brain_features.csv (no data needed)
Rscript molecular/R/m_e1_extract.R           # extraction from the main table -> results/molecular/extract/
Rscript molecular/R/m_e2_nmr_ukbnmr.R        # NMR technical-variation removal (ukbnmr)
python  molecular/py/m0_assemble.py          # analysis frames, exposure-years before measurement, sample flow
python  molecular/py/m1_features.py          # feature preprocessing (--olink-long FILE if not in data/)
python  molecular/py/m3_assoc.py             # per-feature associations, BH-FDR
python  molecular/py/m3b_omnibus.py          # omnibus tests (mean z^2, saddlepoint p)
python  molecular/py/m5_enrich.py            # preranked t table for the Olink layer
Rscript molecular/R/m5_fgsea.R --threads 4   # GO BP / CC / MF and KEGG enrichment
```

Paths are set in `config_molecular.yaml` (`paths.*`, relative to the repository root).

## Inputs

UK Biobank (application required; place under `data/`):
- `data/ukb_main_dataset.csv`: main phenotype table with field 53 (assessment date), 74 (fasting time),
  3166 (blood draw time), 21000 (ethnic group), blood counts 30000-30300, urine assays 30500-30535,
  biochemistry 30600-30890 with their companion fields (+5, +6), Olink technical fields 30900-30901,
  NMR fields 20280-20283, 23400-23660, 23700-23948, brain IDPs 25056-25103 (FA) and 25782-25920 (GMV),
  25781 (WMH), 25000, 25732, 25734, 25741, 25746, 25756-25759 (imaging technical covariates).
- `data/olink_data.txt`: Olink long table (eid, ins_index, protein_id, result).
- `data/withdraw.csv`: current list of withdrawn participants.

Published resources (place under `references/`):
- `uniprot_meaning`: UK Biobank data coding 143 (Olink assay -> "GENE;Protein name"), tab-separated with
  columns `coding` and `meaning`, from the UK Biobank Showcase.
- Gene sets for fgsea: either install the R package `msigdbr`, or download from MSigDB
  (https://www.gsea-msigdb.org) the gene-symbol GMT files listed in `files.gmt`
  (C5 GO:BP, GO:CC, GO:MF and C2 CP:KEGG_LEGACY, v2024.1) into `references/`.

## Outputs (`results/molecular/`)

- `tables/m0_sample_flow.csv`, `tables/m0_type_counts.csv`: sample flow and participants per type and layer
- `tables/m1_vm1_qc.csv`, `tables/m1_vm1_dropped.csv`: feature transforms, trimming and exclusions
- `tables/m3_assoc.csv.gz`: every association (beta, SE, p, q, label, n); `tables/m3_summary.csv`: counts
- `tables/m3b_omnibus.csv`: omnibus tests
- `tables/m5_fgsea.csv`: enrichment results

## Dependencies

- Python: numpy, pandas, pyarrow, scipy, PyYAML (plus the `aging`, `outcome` and `typology` helper modules).
- R: data.table, yaml, ukbnmr (Ritchie et al. 2023), fgsea (Bioconductor); optional msigdbr and
  BiocParallel (`--threads > 1`).
