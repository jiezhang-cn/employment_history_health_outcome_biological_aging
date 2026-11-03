# aging: career types and biological aging clocks

Computes 73 biological aging clocks in five families and relates their age gaps to career types,
career axes and lifetime exposure intensities (paper sections "Biological aging clocks" and
"Organ- and cell-type-specific aging across careers").

| Family | Clocks | Source | Measurement |
| --- | --- | --- | --- |
| C-P | proteomic organ clocks | Oh et al. 2025, published coefficients | Olink, instance 0 |
| C-C | proteomic cell-type and lineage clocks (48) | Ding et al. 2026, published coefficients | Olink, instance 0 |
| C-M | MRI organ clocks (heart, adipose, liver, spleen, kidney, pancreas) | retrained with the MULTI method | instance 2 |
| C-B | brain MRI clocks (GM, WM, FC, whole brain) | retrained following Tian et al. 2023 | instance 2 |
| C-L | PhenoAge, KDM, HD, LTL | Levine 2018 formula; KDM / HD fitted by sex in disease-free participants; LTL sign-reversed | blood, instance 0 |

## Run order

Run from the repository root after the typology pipeline (outputs in `results/typology/derived/`).

```bash
Rscript aging/R/e1_extract_fields.R --cores 8     # field extraction -> results/aging/extract/
python aging/py/a0_assemble.py                    # time points, CN subset, availability, person-level exposures
python aging/py/a1_clocks_protein.py              # C-P and C-C
python aging/py/a2_clocks_mri.py                  # C-M and C-B (--only CM | CB)
python aging/py/a1b_clocks_blood.py               # C-L
python aging/py/a3_validate.py                    # accuracy, mortality, coverage -> clock_status.json
python aging/py/a4_assoc_type.py                  # type vs rest, career axes, extreme agers
python aging/py/a5_lifecourse.py                  # lifetime exposure intensities
```

Main outputs (`results/aging/tables/`): `a3_T1_validation.csv`, `a3_v2_cox.csv`, `a4_assoc.csv`,
`a4_extreme_share.csv`, `a5_life.csv`, `a0_type_counts.csv`. The molecular module reads
`results/aging/derived/person_{sex}.parquet`, `measure.parquet` and `results/aging/extract/`.

## Inputs

UKB data (paths in `config_aging.yaml`, relative to the repository root):

- `data/ukb_main_dataset.csv`: main phenotype table with the fields listed in `config_aging.yaml`
  (`extract.extra_groups`) and in the two feature lists below (brain IDPs, abdominal and cardiac MRI
  fields), plus field 53 (attendance date) and the field 25752 pointer columns.
- `data/rfMRI/`: field 25752 bulk files (`{eid}_25752_{instance}_0.txt`, via ukbfetch) and
  `data/rfMRI_GoodComponents_d25_v1.txt` (UKB good-component list, used to label edges).
- `data/olink_data.txt`: UKB-PPP Olink long table (`eid`, `ins_index`, `protein_id`, `result` = NPX).
  Raw NPX is required; pre-standardised data are rejected.
- `data/`: `ukb_covariates_baseline.csv` (eid, sex, date_assess, centre, birth_date),
  `ukb_covariates_raw.csv` (Qualifications), `ukb_first_occurrences_data.csv` (`Date_<ICD10>_first_reported...`),
  `ukb_cancer_register_data.csv`, `ukb_death_data.tsv` (eid, date_of_death),
  `withdraw.csv` (optional).

Published resources in `references/` (not included; rebuild them from the papers' supplements):

| File | Content | Source |
| --- | --- | --- |
| `clock_feature_lists/tian_brain_2309.csv` | columns `field`, `short_name`, `label`, `system` (GM / WM / whole_brain_only / FC) | Tian et al. 2023, Table S2 |
| `clock_feature_lists/multi_mri_6organs.csv` | `field`, `short_name`, `label`, `organ` | MULTI Consortium 2026, eFile1 |
| `clock_feature_lists/tian_s7_disease_codes.csv` | `category`, `code_type` (ICD10 rows used), `code` | Tian et al. 2023, Table S7 |
| `clock_feature_lists/oh2025_weights_long.csv` | `organ`, `local_col` (gene symbol), `weight`, `intercept` | Oh et al. 2025, ST4 |
| `clock_feature_lists/oh2025_training_scaler_ST3.csv` | `local_col`, `mean`, `stdev` | Oh et al. 2025, ST3 |
| `oh2025_organage_media-1.xlsx` | sheet `ST9_agegap_vs_mortality_cox_sta` (`organ`, `exp(coef)`) | Oh et al. 2025 supplement |
| `clock_feature_lists/ding_cellage_models.csv` | `model`, `intercept`, `coef_sex_female1`, `Source` (original / aggregate), `Train_Correlation`, `Test_Correlation`, `retained_48` | Ding et al. 2026 |
| `clock_feature_lists/ding_cellage_coef_long.csv` | `model`, `local_col`, `coef` | Ding et al. 2026 |
| `uniprot_meaning` | UKB Data-Coding 143 (tab-separated `coding`, `meaning`) | UK Biobank Showcase |

PhenoAge coefficients (Levine et al. 2018, Table 1) are constants in `a1b_clocks_blood.py`.

## Notes

- Connectivity uses field 25752 (partial correlation, dimension 25, 210 edges) because field 25751 used by
  Tian et al. is no longer available; the FC and whole-brain clocks therefore use the same method with a
  different input.
- Age gaps are z-scored within clock (`gapz`, positive = older); `beta_years` = coefficient x SD of the gap
  in the clock's own units (years for all clocks except HD, ln distance, and LTL, SD).
- The two non-organ C-P clocks (Conventional, Organismal) are computed and validated but excluded from the
  association analyses (`exclude_clocks`).
- Python: numpy, pandas, scipy, scikit-learn, statsmodels, pyarrow, pyyaml, openpyxl. R: data.table, yaml.
