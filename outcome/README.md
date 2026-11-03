# Health outcomes

Career types (from `typology/`) related to prevalent disease, age-dependent
comorbidity topics, incident disease, accumulation of diagnoses and
disease-free life. All settings are in `config_outcome.yaml`; paths are
relative to the repository root. Commands below are run from the repository
root, after the typology pipeline has written `results/typology/derived/`
(`types_{sex}.parquet`, `descriptives.parquet`, `camsis.parquet`).

## Run order

```bash
# 1. Time zero, follow-up, disease matrices, code catalogue
python outcome/py/o1_assemble.py --stage all

# 2. Prevalent disease: prevalence ratios (chapter and code level) + permutation test
python outcome/py/o2_atlas.py --stage all

# 3. Comorbidity topics (published topics, no refitting)
Rscript outcome/R/b0_export_phecode.R     # phecode map and names from AgeTopicModels
python  outcome/py/b1_records.py --stage all
Rscript outcome/R/b2_atm_fit.R            # projection onto UKB_HES_10topics
python  outcome/py/b3_topics.py           # topic names, disease-topic assignment
python  outcome/py/b4_assoc.py            # burden and share ratios + permutation test
```

### Prospective analyses (s0–s3)

After o1, run `python outcome/py/s0_onset.py`, then `s1_incidence.py`,
`s2_healthspan.py` and `s3_dxrate.py` (no arguments; s1–s3 are independent of
each other once s0 has run). Outputs: `tables/S1_incidence.csv`,
`tables/S2_healthspan.csv`, `tables/S3_dx_rate_ratio.csv`, `tables/S3_mcf.csv`,
`tables/S3_dx_shift.csv`.

## Main outputs (`results/outcome/tables/`)

| File | Content |
| --- | --- |
| `OP0_code_catalog.csv` | per code and sex: lifetime prevalence, selection (>= 2%) |
| `OP1_cells.csv` | prevalence ratios, each type vs all other types of the same sex, chapter and code level, BH q |
| `OP2_global_null.csv` | permutation test of the type x disease matrix |
| `OB2_topic_profile.csv`, `OB2_disease_topic_assignment.csv` | topic names and per-disease average topic assignment |
| `OB4_assoc.csv` | topic burden and topic share ratios, BH q |
| `OB5_global_null.csv` | permutation test of the type x topic matrices |

Cells with fewer than 10 cases (or types with fewer than 10 members for
topics) are suppressed in the output tables.

## Required inputs

UK Biobank extracts in `data/` (file names set under `files:` in the config):

- first-occurrence fields (category 1712; `Date_<ICD-10>_first_reported_*` and
  `Source_of_report_of_*` columns);
- cancer registry, all instances: date of diagnosis (40005), ICD-10 (40006),
  ICD-9 (40013), behaviour (40012);
- death register table (`eid`, `ins_index`, `date_of_death`);
- hospital inpatient record tables `hesin` (`eid`, `ins_index`, `epistart`,
  `admidate`) and `hesin_diag` (`eid`, `ins_index`, `diag_icd10`, `diag_icd9`);
- baseline covariates: `birth_date`, assessment centre (54), ethnic group
  (21000); qualifications (6138, all instances, raw multiple-choice answers);
  date lost to follow-up (191);
- work-history questionnaire completion date
  (`When_occupational_data_entered__0_0`).

Published resources, not distributed here:

- R package **AgeTopicModels** (Jiang X. et al., Age-dependent topic modeling
  of comorbidities in UK Biobank identifies disease subtypes with differential
  genetic risk. *Nat. Genet.* 55, 1854–1865 (2023); CRAN, source at
  github.com/Xilin-Jiang/ATM). It provides `loading2weights()`, the published
  topic loadings `UKB_HES_10topics` and vocabulary `UKB_349_disease`, and the
  ICD-10 to phecode map. `b0_export_phecode.R` writes `phecode_icd10.csv`,
  `UKB_349_disease.csv` and `phecode_info.csv` to `references/`. If the
  installed version lacks the two data sets, download `UKB_HES_10topics.rda`
  and `UKB_349_disease.rda` from the `data/` folder of that repository into
  `references/`; `b2_atm_fit.R` reads them from there.

## Software

Python 3.11 with numpy, pandas, scipy, pyyaml and pyarrow (parquet). R with
yaml and AgeTopicModels (arrow optional).
