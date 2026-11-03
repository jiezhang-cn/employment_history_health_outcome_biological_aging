#!/usr/bin/env Rscript
# b0: export the ICD-10 -> phecode map and phecode names from AgeTopicModels.
#
# The map is produced by the package's own icd2phecode() (the mapping used by
# Jiang et al. 2023, including its one-to-many rule) rather than re-implemented
# in Python. Written to paths.refs (default references/):
#   phecode_icd10.csv      ICD-10 (no dot) -> phecode, one phecode per ICD-10 code
#   UKB_349_disease.csv    the 349 phecodes of the published topics
#   phecode_info.csv       phecode -> phenotype name (used to match topic names)
#
#   Rscript outcome/R/b0_export_phecode.R

suppressWarnings(suppressMessages({
  ok_yaml <- requireNamespace("yaml", quietly = TRUE)
  has_atm <- requireNamespace("AgeTopicModels", quietly = TRUE)
}))
say <- function(...) cat(sprintf(...), "\n", sep = "")
die <- function(...) { say("  [FAIL] %s", sprintf(...)); quit(status = 1) }

if (!ok_yaml) die("R package yaml is required: install.packages('yaml')")
if (!has_atm) die("R package AgeTopicModels is required: install.packages('AgeTopicModels')")

here <- tryCatch(dirname(normalizePath(sub("^--file=", "", grep("^--file=",
  commandArgs(trailingOnly = FALSE), value = TRUE)[1]))), error = function(e) ".")
root <- normalizePath(file.path(here, "..", ".."), mustWork = FALSE)
cfg <- yaml::read_yaml(file.path(here, "..", "config_outcome.yaml"))
resolve <- function(x) if (grepl("^(/|[A-Za-z]:)", x)) x else file.path(root, x)
refs <- resolve(cfg$paths$refs[[1]])
dir.create(refs, showWarnings = FALSE, recursive = TRUE)
say("  writing to %s", refs)

# icd2phecode() looks up its data sets as free variables, so the package must
# be attached, not only loaded.
suppressMessages(library(AgeTopicModels))

grab <- function(name) {
  e <- new.env()
  okd <- tryCatch({ utils::data(list = name, package = "AgeTopicModels", envir = e)
                    TRUE }, warning = function(w) FALSE, error = function(x) FALSE)
  if (!okd || !exists(name, envir = e)) return(NULL)
  get(name, envir = e)
}

# -- ICD-10 -> phecode via the package's icd2phecode() --
# Every short ICD-10 code of the package is passed through icd2phecode() once;
# the output is used as a lookup table. Input columns: (eid, diag_icd10, age_diag).
short <- grab("short_icd10")
raw <- grab("phecode_icd10")
codes <- NULL
if (!is.null(short)) {
  s <- as.data.frame(short)
  codes <- unique(as.character(s[[grep("icd", names(s), ignore.case = TRUE)[1]]]))
} else if (!is.null(raw)) {
  r0 <- as.data.frame(raw)
  codes <- unique(as.character(r0[[grep("icd", names(r0), ignore.case = TRUE)[1]]]))
} else {
  die("neither short_icd10 nor phecode_icd10 found in AgeTopicModels")
}
codes <- gsub("[^A-Za-z0-9]", "", codes)
codes <- unique(codes[nzchar(codes)])
probe <- data.frame(eid = seq_along(codes), diag_icd10 = codes,
                    age_diag = 50, stringsAsFactors = FALSE)
mapped <- tryCatch(AgeTopicModels::icd2phecode(probe),
                   error = function(e) { die("icd2phecode failed: %s", conditionMessage(e)) })
mapped <- as.data.frame(mapped)
# icd2phecode replaces diag_icd10 by the (numeric) phecode; eid is the probe row.
ph <- mapped[[grep("icd|phecode", names(mapped), ignore.case = TRUE)[1]]]
# Phecodes as strings with one canonical format ("496", "411.4") on both tables.
canon <- function(x) as.character(as.numeric(as.character(x)))
out <- data.frame(icd10 = probe$diag_icd10[as.integer(mapped$eid)],
                  phecode = canon(ph), stringsAsFactors = FALSE)
out <- out[!is.na(out$phecode) & nzchar(out$phecode) & out$phecode != "NA", ]
out <- out[!duplicated(out$icd10), ]
utils::write.csv(out, file.path(refs, "phecode_icd10.csv"), row.names = FALSE)
say("  phecode_icd10.csv: %s ICD-10 codes -> %s phecodes",
    format(nrow(out), big.mark = ","),
    format(length(unique(out$phecode)), big.mark = ","))

# -- The 349 phecodes of the published topics --
d349 <- grab("UKB_349_disease")
if (is.null(d349)) {
  say("  [WARN] UKB_349_disease not in this package version")
} else {
  d349 <- as.data.frame(d349)
  d349[[1]] <- canon(d349[[1]])
  utils::write.csv(d349, file.path(refs, "UKB_349_disease.csv"), row.names = FALSE)
  say("  UKB_349_disease.csv: %d phecodes", nrow(d349))
}

# -- Phecode names --
info <- grab("disease_info_phecode_icd10")
if (is.null(info)) {
  say("  [WARN] disease_info_phecode_icd10 not in this package version")
} else {
  info <- as.data.frame(info)
  pk <- grep("phecode", names(info), ignore.case = TRUE)[1]
  if (!is.na(pk)) info[[pk]] <- canon(info[[pk]])
  utils::write.csv(info, file.path(refs, "phecode_info.csv"), row.names = FALSE)
  say("  phecode_info.csv: %s phecodes", format(nrow(info), big.mark = ","))
}
