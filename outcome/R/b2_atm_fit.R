#!/usr/bin/env Rscript
# b2: project hospital diagnoses onto the published age-dependent comorbidity topics.
#
# Topics are NOT refitted. The ten topics of Jiang et al. (Nat Genet 2023;
# 55:1854-1865; R package AgeTopicModels), estimated on UK Biobank HES with
# K = 10 and a one-knot cubic spline in age over 349 phecodes
# (`UKB_HES_10topics`, age x disease x topic), are held fixed and each
# participant's diagnosis history is projected onto them with the package's
# loading2weights() (the collapsed variational E-step at fixed loadings).
#
# Input:  results/outcome/derived/ob_records.parquet (or .csv.gz) from b1
# Output: results/outcome/derived/ob_fit_atm_K10_published/
#           weights.csv.gz        eid, theta_1..10, n_dx
#           loadings_long.csv.gz  age, code, topic, beta (published loadings)
#           fit_meta.csv          key, value
#
# `UKB_HES_10topics` and `UKB_349_disease` are taken from the installed
# package; if this package version lacks them, the two .rda files of the
# package repository (github.com/Xilin-Jiang/ATM, data/) are read from
# paths.refs.
#
#   Rscript outcome/R/b2_atm_fit.R

suppressWarnings(suppressMessages({
  ok_yaml <- requireNamespace("yaml", quietly = TRUE)
  has_arrow <- requireNamespace("arrow", quietly = TRUE)
  has_atm <- requireNamespace("AgeTopicModels", quietly = TRUE)
}))

say <- function(...) cat(sprintf(...), "\n", sep = "")
die <- function(...) { say("  [FAIL] %s", sprintf(...)); quit(status = 1) }
h1 <- function(t) { say("\n%s", strrep("=", 72)); say("%s", t); say("%s", strrep("=", 72)) }
h2 <- function(t) say("\n-- %s %s", t, strrep("-", max(0, 60 - nchar(t))))

if (!ok_yaml) die("R package yaml is required: install.packages('yaml')")
if (!has_atm) die("R package AgeTopicModels is required: install.packages('AgeTopicModels')")

# -- Configuration and paths --
here <- tryCatch(dirname(normalizePath(sub("^--file=", "", grep("^--file=",
  commandArgs(trailingOnly = FALSE), value = TRUE)[1]))), error = function(e) ".")
root <- normalizePath(file.path(here, "..", ".."), mustWork = FALSE)
cfg_path <- file.path(here, "..", "config_outcome.yaml")
if (!file.exists(cfg_path)) die("config not found: %s", cfg_path)
cfg <- yaml::read_yaml(cfg_path)
resolve <- function(x) if (grepl("^(/|[A-Za-z]:)", x)) x else file.path(root, x)
out <- resolve(cfg$paths$out[[1]])
refs_dir <- resolve(cfg$paths$refs[[1]])
dirs <- list(derived = file.path(out, "derived"))
dir.create(dirs$derived, showWarnings = FALSE, recursive = TRUE)
ob <- cfg$ob

h1("b2: projection onto the published topics (AgeTopicModels)")
say("  package version %s at %s", as.character(utils::packageVersion("AgeTopicModels")),
    tryCatch(find.package("AgeTopicModels"), error = function(e) "?"))

# -- Records from b1 --
read_records <- function() {
  pq <- file.path(dirs$derived, "ob_records.parquet")
  cs <- file.path(dirs$derived, "ob_records.csv.gz")
  if (has_arrow && file.exists(pq)) {
    d <- as.data.frame(arrow::read_parquet(pq))
  } else if (file.exists(cs)) {
    d <- utils::read.csv(gzfile(cs))
  } else {
    die("ob_records.parquet / ob_records.csv.gz missing; run python outcome/py/b1_records.py first")
  }
  d <- d[, c("eid", "code", "age")]
  # column names expected by the package: id, disease code, age at diagnosis
  names(d) <- c("eid", "diag_icd10", "age_diag")
  # The package requires ages in (0, 90].
  bad <- !is.finite(d$age_diag) | d$age_diag <= 0 | d$age_diag > 90
  if (any(bad)) {
    say("  [WARN] %d records with age outside (0, 90] dropped", sum(bad))
    d <- d[!bad, ]
  }
  # Topic weights are compositions: participants need >= min_dx_fit diagnoses.
  n_by <- table(d$eid)
  keep <- names(n_by)[n_by >= as.integer(ob$min_dx_fit)]
  d <- d[as.character(d$eid) %in% keep, ]
  say("  %s records / %s participants", format(nrow(d), big.mark = ","),
      format(length(keep), big.mark = ","))
  d
}

# The first argument of loading2weights() is named differently across package
# versions (`data` or `rec_data`); pass it by position and check the others.
call_l2w <- function(rec, ds_list, topics) {
  fm <- names(formals(AgeTopicModels::loading2weights))
  if (length(fm) < 3 || !all(c("ds_list", "topics") %in% fm))
    die("unexpected loading2weights() signature: %s", paste(fm, collapse = ", "))
  da <- setdiff(fm, c("ds_list", "topics"))[1]
  do.call(AgeTopicModels::loading2weights,
          stats::setNames(list(rec, ds_list, topics), c(da, "ds_list", "topics")))
}

# loading2weights() returns topic_weights = data.frame(eid, topic_weights.1..K).
weights_from_l2w <- function(w, rec_used) {
  W <- as.data.frame(w$topic_weights)
  if (!"eid" %in% names(W))
    die("topic_weights has no eid column (columns: %s)", paste(names(W), collapse = ", "))
  eids <- as.integer(as.character(W$eid))
  th <- as.matrix(W[, setdiff(names(W), "eid"), drop = FALSE])
  colnames(th) <- sprintf("theta_%d", seq_len(ncol(th)))
  rs <- rowSums(th)
  if (any(!is.finite(rs)) || max(abs(rs - 1)) > 1e-3)
    die("topic weights do not sum to 1 (max deviation %.4g)", max(abs(rs - 1)))
  out <- data.frame(eid = eids, th, check.names = FALSE)
  # n_dx from the records actually projected (it is the offset of the share model)
  n_by <- table(rec_used$eid)
  out$n_dx <- as.integer(n_by[as.character(out$eid)])
  if (anyNA(out$n_dx))
    die("%d participants have no record count", sum(is.na(out$n_dx)))
  out
}

load_rda <- function(nm) {
  f <- file.path(refs_dir, paste0(nm, ".rda"))
  if (!file.exists(f)) return(NULL)
  e <- new.env()
  loaded <- load(f, envir = e)
  if (!nm %in% loaded) {
    if (!length(loaded)) return(NULL)
    say("  [WARN] %s.rda contains {%s}; using the first object",
        nm, paste(loaded, collapse = ", "))
    return(get(loaded[1], envir = e))
  }
  get(nm, envir = e)
}

stage_published <- function(rec) {
  pv <- as.character(utils::packageVersion("AgeTopicModels"))
  need <- c("UKB_349_disease", "UKB_HES_10topics")
  avail <- tryCatch(utils::data(package = "AgeTopicModels")$results[, "Item"],
                    error = function(e) character(0))
  avail <- sub("\\s.*$", "", avail)
  for (nm in need) {
    if (nm %in% avail) {
      data(list = nm, package = "AgeTopicModels", envir = environment())
    } else {
      v <- load_rda(nm)
      if (is.null(v))
        die("%s is neither in AgeTopicModels %s nor at %s/%s.rda", nm, pv, refs_dir, nm)
      assign(nm, v, envir = environment())
      say("  %s read from %s.rda", nm, nm)
    }
  }
  if (!"loading2weights" %in% getNamespaceExports("AgeTopicModels"))
    die("AgeTopicModels %s has no loading2weights(); install a newer version", pv)
  if (length(dim(UKB_HES_10topics)) != 3)
    die("UKB_HES_10topics is not a 3-dimensional array")
  n_ds <- if (is.data.frame(UKB_349_disease)) nrow(UKB_349_disease) else
    length(UKB_349_disease)
  if (dim(UKB_HES_10topics)[2] != n_ds)
    die("loading array has %d diseases but UKB_349_disease has %d",
        dim(UKB_HES_10topics)[2], n_ds)
  say("  published loadings: %s (age x disease x topic)",
      paste(dim(UKB_HES_10topics), collapse = " x "))
  # Loadings must sum to 1 over diseases at each age, i.e. dimension order
  # is age x disease x topic.
  cs <- colSums(UKB_HES_10topics[50, , , drop = TRUE])
  if (max(abs(cs - 1)) > 1e-3)
    die("loadings at age 50 do not sum to 1 over diseases; unexpected dimension order")

  ds <- as.character(UKB_349_disease[[1]])
  A_pub <- dim(UKB_HES_10topics)[1]          # ages 1..A_pub
  mine <- unique(as.character(rec$diag_icd10))
  n0 <- nrow(rec)
  # Codes outside the 349 published phecodes cannot be projected.
  extra <- setdiff(mine, ds)
  if (length(extra)) rec <- rec[as.character(rec$diag_icd10) %in% ds, ]
  # The published loadings end at age A_pub; later diagnoses are dropped here
  # (the package would drop them silently) and counted in fit_meta.
  old <- rec$age_diag > A_pub
  if (any(old)) {
    say("  [WARN] %s diagnoses (%.2f%%) after age %d dropped",
        format(sum(old), big.mark = ","), 100 * mean(old), A_pub)
    rec <- rec[!old, ]
  }
  n_by <- table(rec$eid)
  drop_lt2 <- names(n_by)[n_by < as.integer(ob$min_dx_fit)]
  if (length(drop_lt2))
    rec <- rec[!as.character(rec$eid) %in% drop_lt2, ]
  cover <- length(intersect(mine, ds)) / length(ds)
  say("  vocabulary covers %.1f%% of the published phecodes (%d/%d)",
      100 * cover, length(intersect(mine, ds)), length(ds))
  if (cover < 0.5)
    die("coverage %.1f%%: records are not phecodes?", 100 * cover)
  say("  projecting %s records / %s participants (%.1f%% of input records dropped)",
      format(nrow(rec), big.mark = ","),
      format(length(unique(rec$eid)), big.mark = ","),
      100 * (1 - nrow(rec) / n0))

  t0 <- Sys.time()
  w <- call_l2w(rec, UKB_349_disease, UKB_HES_10topics)
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  dir_out <- file.path(dirs$derived, "ob_fit_atm_K10_published")
  dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)
  W <- weights_from_l2w(w, rec)
  gz <- gzfile(file.path(dir_out, "weights.csv.gz"), "w")
  utils::write.csv(W, gz, row.names = FALSE); close(gz)
  L <- UKB_HES_10topics
  # as.vector() runs over age, then disease, then topic, matching expand.grid
  g <- expand.grid(age = seq_len(dim(L)[1]),
                   code = as.character(UKB_349_disease[[1]]),
                   topic = seq_len(dim(L)[3]), KEEP.OUT.ATTRS = FALSE,
                   stringsAsFactors = FALSE)
  g$beta <- as.vector(L)
  gz <- gzfile(file.path(dir_out, "loadings_long.csv.gz"), "w")
  utils::write.csv(g, gz, row.names = FALSE); close(gz)
  meta <- list(
    engine = "atm", K = dim(L)[3], package_version = pv,
    role = "published_loadings",
    note = "UKB_HES_10topics (Jiang et al. 2023); loadings fixed, no fitting",
    age_basis = "spline1_fixed_by_package",
    degree_free_num = 5,
    fit_procedure = "loading2weights: E-step at fixed loadings",
    n_person = nrow(W), n_records = nrow(rec), n_disease = dim(L)[2],
    n_age_axis = dim(L)[1], age_axis_note = "row i = age i years",
    vocab_coverage_of_published = round(cover, 4),
    n_records_in = n0,
    n_records_dropped_age_gt_81 = sum(old),
    n_codes_not_in_published = length(extra),
    data_age_min = round(min(rec$age_diag), 2),
    data_age_max = round(max(rec$age_diag), 2),
    seconds = round(secs, 1),
    vocab = "phecode", source = "hes")
  utils::write.csv(data.frame(key = names(meta),
                              value = vapply(meta, function(x)
                                paste(as.character(x), collapse = ";"), "")),
                   file.path(dir_out, "fit_meta.csv"), row.names = FALSE)
  say("  -> %s (%s participants, %.0f s)", dir_out,
      format(nrow(W), big.mark = ","), secs)
}

rec <- read_records()
stage_published(rec)
h2("Done")
