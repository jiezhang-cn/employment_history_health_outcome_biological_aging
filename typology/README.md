# typology

Builds year-by-year work histories from the UK Biobank online employment-history
questionnaire, trains sex-specific self-supervised Transformer encoders on them,
and derives career types (Leiden clustering of the frozen 192-dimensional [CLS]
representation) and career axes. No health variable is used anywhere in this
module. Its outputs in `results/typology/derived/` are the inputs of the
`outcome`, `aging` and `molecular` modules.

Settings and thresholds are in `../config.yaml`. Paths are relative to the
repository root; run the commands from there.

## Run order

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8          # deterministic cuBLAS (also set in common.py)
python typology/01_worklife.py                  # panel, career end, exposures, CAMSIS, education
python typology/02_tokenize.py                  # person-year tokens
python typology/03_pretrain.py                  # encoders, frozen representation, diagnostics
python typology/04_typology.py --stage cluster  # Leiden types, bootstrap and birth-cohort stability, NMI
python typology/04_typology.py --stage profiles # type profiles (tables) and type labels
python typology/05_derived.py                   # occupation embedding, n_moves, net_displacement
```

`03_pretrain.py` needs a GPU for practical run times. `04_typology.py` and
`05_derived.py` verify the SHA256 of the encoder weights stored with the
representation and stop if it does not match.

## Inputs

Place the UK Biobank extracts in `data/` (names set in `config.files`). Array
fields are expected as columns `<FieldName>__<instance>_<array index>`
(e.g. `Year_job_started__0_3`).

| file | content |
|---|---|
| `ukb_employment_history_data.csv` | `eid` and the employment-history fields 22599, 22601-22617, 22620, 22630, 22640, 22650, 22660, 22661, 22663, 22664 (column prefixes in `config.employment`) |
| `ukb_work_environment_data.csv` | `eid`, `When_occupational_data_entered__0_0` (Category 123) |
| `ukb_covariates_baseline.csv` | `eid`, `sex_label`, `year_birth`, `age_recruit`, `birth_cohort_5y`, `date_assess`, `centre`, `townsend`, `ethnic_group` |
| `ukb_covariates_raw.csv` | `eid` and all `Qualifications__<i>_<j>` columns (field 6138, raw pipe-separated answers) |

Place in `references/`:

| file | content |
|---|---|
| `ukbjobs_soc2000_occ_status.csv` | CAMSIS scores per SOC2000 unit group, columns `soc2000`, `mcamsis`, `fcamsis` (sex-specific Cambridge Social Interaction and Stratification scale; Prandy & Lambert, *Sociology* 37, 397-411 (2003); scores from the CAMSIS project, SOC2000 version for Britain) |

## Outputs (`results/typology/`)

- `derived/worklife_jobs.parquet`, `worklife_gaps.parquet`: job and gap events
- `derived/worklife_annual.parquet`: one row per person-year (not truncated)
- `derived/worklife_exposure.parquet`: exposure summaries up to career end
- `derived/cohort.parquet`: sex, age at recruitment, questionnaire year, career end and its source, fold
- `derived/camsis.parquet`, `camsis_by_age.parquet`: CAMSIS summaries and per-age values
- `derived/descriptives.parquet`: rebuilt highest qualification, Townsend, ethnic group, birth-year group
- `derived/tokens_{sex}.pt`, `vocab.json`: model inputs
- `derived/latent_{sex}.parquet` (+ `.meta.json` with the weight hash): frozen representation
- `models/encoder_{sex}.pt`: encoder weights
- `qc/latent_diagnostics_{sex}.json`: effective rank and sequence-length dominance
- `derived/types_{sex}.parquet`, `type_names_{sex}.json`: career types and their labels
- `tables/P2_soc_mix*`, `P3_exposure*`, `P4_camsis*`, `P5_gaps*`, `P7_axis_discrimination*`: type profiles
- `qc/cluster_summary_main.json`: partition sizes, bootstrap and birth-cohort ARI, NMI
- `derived/seqmove_{sex}.parquet`: number of occupational moves and net displacement in the occupation embedding
- `logs/`: one log per script

## Dependencies

Python 3.10+, numpy, pandas, pyarrow, PyYAML, PyTorch, scikit-learn,
python-igraph, leidenalg.
