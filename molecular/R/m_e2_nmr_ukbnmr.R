#!/usr/bin/env Rscript
# m_e2: removal of technical variation from the NMR metabolomics data (ukbnmr::remove_technical_variation;
# Ritchie et al. 2023, Sci Data), keeping the 107 non-derived biomarkers (ukbnmr::nmr_info$Type ==
# "Non-derived" with a QC-flag field); derived ratios, percentages and composite measures are not analysed.
#
# Input: <paths.out>/extract/m_e1_nmr.csv.gz (all participants, all instances, RAP column names
# p{field}_i{inst}_a{arr}). ukbnmr estimates technical variation by plate and measurement date over all
# samples, so no participants are filtered beforehand. After correction only the baseline visit
# (visit_index == 0) is kept. Outputs (same directory):
#   m_e2_nmr_i0.csv.gz          eid + 107 non-derived biomarkers (original units; log and z-score in py/m1)
#   m_e2_nmr_log_offset.csv     log offsets reported by ukbnmr (m1 uses log(x + offset))
#   m_e2_nmr_info.csv           name / units / group / subgroup of the 107 biomarkers
#   m_e2_nmr_qc.csv             non-missing count per biomarker at instance 0, before and after correction
#   m_e2_nmr_outlier_plates.csv outlier plates detected by ukbnmr
#
#   Rscript molecular/R/m_e2_nmr_ukbnmr.R

suppressPackageStartupMessages({
  library(data.table)
  library(yaml)
})
say <- function(...) cat(sprintf(...), "\n", sep = "")
warn <- function(...) say("  [WARN] %s", sprintf(...))
die <- function(...) { say("  [FAIL] %s", sprintf(...)); quit(status = 1) }
t_start <- Sys.time()
elapsed <- function() sprintf("%.1f min", as.numeric(difftime(Sys.time(), t_start, units = "mins")))

here <- tryCatch(dirname(normalizePath(sub("^--file=", "", grep("^--file=",
  commandArgs(trailingOnly = FALSE), value = TRUE)[1]))), error = function(e) ".")
root <- normalizePath(file.path(here, "..", ".."), mustWork = FALSE)     # repository root
from_root <- function(p) if (grepl("^(/|[A-Za-z]:)", p)) p else file.path(root, p)
cfg <- read_yaml(file.path(here, "..", "config_molecular.yaml"))

if (!requireNamespace("ukbnmr", quietly = TRUE))
  die("R package ukbnmr (>= 2.0) is required: install.packages('ukbnmr')")
suppressPackageStartupMessages(library(ukbnmr))
say("ukbnmr %s", as.character(packageVersion("ukbnmr")))

mex <- file.path(from_root(cfg$paths$out), "extract")
if (!file.exists(file.path(mex, "m_e1_dictionary.csv"))) die("%s has no m_e1_dictionary.csv (run m_e1_extract.R first)", mex)
f_in <- file.path(mex, "m_e1_nmr.csv.gz")
if (!file.exists(f_in)) die("%s not found (run m_e1_extract.R --groups nmr)", f_in)
x <- fread(f_in, showProgress = FALSE)
say("read %s: %d participants x %d columns (%s)", basename(f_in), nrow(x), ncol(x) - 1, elapsed())

# ── 1. Check the fields ukbnmr needs ──────────────────────────────────
# remove_technical_variation strictly requires five sample-processing fields; missing QC flags are filled
# with NA, and a missing Processing.Batch (20282) is inferred from Shipment.Plate. The script therefore
# stops only if (1) one of the five required sample fields is absent, or (2) fewer than expected_n (107)
# non-derived biomarkers are present. Other gaps are reported.
# Non-derived = Type == "Non-derived" with a QC-flag field: newer ukbnmr versions also label
# Glucose_Lactate (20280) and Corrected_Ala (20281) non-derived, but they have no QC-flag field and are
# not among the 107.
info <- as.data.table(nmr_info)
sqc <- as.data.table(sample_qc_info)
have <- unique(as.integer(sub("^p(\\d+)_.*$", "\\1", grep("^p\\d+_", names(x), value = TRUE))))
req_names <- c("Well.Position.Within.Plate", "Sample.Measured.Date.and.Time", "Sample.Prepared.Date.and.Time",
               "Spectrometer", "Shipment.Plate")
req <- sqc[Name %in% req_names & !is.na(UKB.Field.ID)]
lack_req <- req[!(as.integer(UKB.Field.ID) %in% have)]
if (nrow(lack_req)) die("sample-processing fields required by ukbnmr are missing: %s (add them to extract.groups.nmr.ranges and rerun m_e1 --groups nmr)",
                        paste(sprintf("%s %s", lack_req$UKB.Field.ID, lack_req$Name), collapse = ", "))

nd_all <- info[Type == cfg$ln$type_keep]
nd <- nd_all[!is.na(QC.Flag.Field.ID)]
if (nrow(nd) < nrow(nd_all))
  say("non-derived biomarkers without a QC-flag field excluded: %s", paste(setdiff(nd_all$Biomarker, nd$Biomarker), collapse = ", "))
nd_have <- nd[as.integer(UKB.Field.ID) %in% have]
nd_lack <- nd[!(as.integer(UKB.Field.ID) %in% have)]
keep <- nd_have$Biomarker
say("non-derived: %d listed by ukbnmr %s, %d in the main table%s", nrow(nd), as.character(packageVersion("ukbnmr")),
    length(keep), if (nrow(nd_lack)) sprintf(" (absent: %s)", paste(sprintf("%s %s", nd_lack$UKB.Field.ID,
                                                                         nd_lack$Biomarker), collapse = ", ")) else "")
if (length(keep) < as.integer(cfg$ln$expected_n))
  die("only %d non-derived biomarkers in the main table, fewer than the expected %d", length(keep), as.integer(cfg$ln$expected_n))
if (length(keep) > as.integer(cfg$ln$expected_n))
  warn("%d non-derived biomarkers, more than the expected %d; check the ukbnmr version", length(keep),
       as.integer(cfg$ln$expected_n))

lack_flag <- setdiff(na.omit(as.integer(nd_have$QC.Flag.Field.ID)), have)
lack_sqc <- setdiff(na.omit(as.integer(sqc$UKB.Field.ID)), have)
if (length(lack_flag)) warn("%d QC-flag fields of non-derived biomarkers missing (filled with NA by ukbnmr; flags are not used): %s",
                            length(lack_flag), paste(head(sort(lack_flag), 10), collapse = ", "))
if (length(lack_sqc)) warn("sample processing / QC fields missing: %s%s", paste(sort(lack_sqc), collapse = ", "),
                           if (20282 %in% lack_sqc) " (Processing.Batch 20282 missing; ukbnmr infers it from Shipment.Plate)" else "")

# Non-missing counts at instance 0 before correction
pre_n <- sapply(keep, function(b) {
  fid <- nd_have[Biomarker == b, UKB.Field.ID]
  c0 <- sprintf("p%d_i0_a0", fid)
  if (c0 %in% names(x)) sum(!is.na(x[[c0]])) else NA_integer_
})

# ── 2. Technical-variation removal ───────────────────────────────────
# Rename columns to the original UKB form {field}-{instance}.{array}: ukbnmr 2.x does not recognise RAP
# names, whereas the original form is accepted by every version.
setnames(x, sub("^p(\\d+)_i(\\d+)_a(\\d+)$", "\\1-\\2.\\3", names(x)))
proc <- remove_technical_variation(x)
rm(x); invisible(gc())
say("remove_technical_variation done (%s)", elapsed())
bio <- as.data.table(proc$biomarkers)
if (!all(c("eid", "visit_index") %in% names(bio))) die("ukbnmr output lacks eid / visit_index")
bio0 <- bio[visit_index == 0]
miss_keep <- setdiff(keep, names(bio0))
if (length(miss_keep)) die("non-derived biomarkers missing from the corrected output: %s", paste(miss_keep, collapse = ", "))
has_any <- Reduce(`|`, lapply(keep, function(b) !is.na(bio0[[b]])), FALSE)
bio0 <- bio0[has_any]
fwrite(bio0[, c("eid", keep), with = FALSE], file.path(mex, "m_e2_nmr_i0.csv.gz"))
say("m_e2_nmr_i0.csv.gz: %d participants x %d non-derived biomarkers", nrow(bio0), length(keep))

lo <- proc$log_offset
if (!is.null(lo)) {
  fwrite(as.data.table(lo), file.path(mex, "m_e2_nmr_log_offset.csv"))
  say("log_offset: %d rows, columns %s", nrow(lo), paste(names(lo), collapse = ", "))
} else warn("ukbnmr returned no log_offset (m1 falls back to half the smallest positive value)")
if (!is.null(proc$outlier_plate_detection))
  fwrite(as.data.table(proc$outlier_plate_detection), file.path(mex, "m_e2_nmr_outlier_plates.csv"))

info_cols <- intersect(c("Biomarker", "Description", "Units", "Group", "Sub.Group", "Type", "UKB.Field.ID"), names(info))
fwrite(info[Biomarker %in% keep, ..info_cols], file.path(mex, "m_e2_nmr_info.csv"))
post_n <- sapply(keep, function(b) sum(!is.na(bio0[[b]])))
qc <- data.table(biomarker = keep, n_i0_raw = pre_n, n_i0_corrected = post_n)
fwrite(qc, file.path(mex, "m_e2_nmr_qc.csv"))
say("median non-missing count: %s before, %s after correction; done (%s)", median(pre_n, na.rm = TRUE),
    median(post_n), elapsed())
