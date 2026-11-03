# Employment history, health outcomes and biological aging in UK Biobank

Analysis code for the study relating lifetime work histories of 117,588 UK Biobank participants to
prevalent and incident disease, comorbidity topics, disease-free life, biological aging clocks and
molecular and neuroimaging features.

The code covers the main analyses reported in the paper. No data are included: UK Biobank data are available to approved researchers through
https://www.ukbiobank.ac.uk/enable-your-research.

## Modules

| Directory | Paper section | What it does |
| --- | --- | --- |
| [`typology/`](typology/) | Representation learning; career types, axes and exposures | Year-by-year work histories, self-supervised Transformer encoders, Leiden career types, type profiles, occupation embedding and mobility measures |
| [`outcome/`](outcome/) | Health outcomes | Prevalence ratios, published age-dependent comorbidity topics, incident disease rate ratios, accumulation of diagnoses, disease-free years 60–80 |
| [`aging/`](aging/) | Biological aging clocks | Proteomic organ and cell-type clocks, MRI organ and brain clocks, blood-based measures; associations with career types, axes and lifetime exposure intensities |
| [`molecular/`](molecular/) | Molecular and neuroimaging features | Blood biochemistry, NMR metabolomics, Olink proteins and brain MRI features; per-feature and omnibus tests; protein set enrichment |

Each module has a README with its run order, required inputs and outputs. Settings and thresholds are
in the YAML files (`config.yaml` for `typology/`, one `config_*.yaml` per other module).

## Software

Python 3.11 (see `requirements.txt`) and R 4.x with `data.table`, `yaml`, `AgeTopicModels`, `ukbnmr`
and `fgsea` (optional: `msigdbr`, `BiocParallel`, `arrow`). Training the encoders in
`typology/03_pretrain.py` needs a GPU for practical run times.

The data-processing and modelling pipeline for UK Biobank work histories is also available as the R
package [ukbcareer](https://github.com/jiezhang-cn/ukbcareer).
