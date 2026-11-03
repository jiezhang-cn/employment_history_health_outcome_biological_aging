#!/usr/bin/env Rscript
# e1: extract from the UKB main table every field needed by the clocks -- brain IDPs for the C-B clocks
# (Tian et al. 2023, Table S2) and non-brain MRI fields for the C-M clocks (MULTI Consortium 2026, eFile1)
# -- plus covariates, imaging technical covariates and blood markers. Fields are extracted and renamed
# only; no cleaning or derivation happens here.
#
# Feature lists are read from paths.feature_lists (CSV files with columns field / short_name / label /
# system or organ), built from the source papers' supplements; they are not copied into the code.
# Additional fields (field -> short name) are listed in config_aging.yaml, extract.extra_groups.
#
# Scale: the main table is read column-block-wise (fread(select=), extract.chunk_cols columns at a time)
# and each block is cut to the row gate at once (participants with an imaging attendance date, field 53).
# Groups with gate: false (blood) keep all participants. Several main-table files can be combined
# (--main a.csv,b.csv); ukbconv `25056-2.0`, ukbtools `f.25056.2.0` and RAP `p25056_i2_a0` headers are
# recognised.
#
# Connectivity (config fc:) is field 25752, rfMRI partial correlation at dimension 25 (210 edges among 21
# good components). In the main table the field holds file names only; values come from the ukbfetch bulk
# files {eid}_{field}_{instance}_0.txt in paths.fc_bulk_dir, value k of a file = edge k. A file must hold
# exactly n_edges values, otherwise it is listed in e1_fc_bad.csv and skipped (the matrix layout is never
# guessed).
#
# Outputs (<paths.out>/extract/):
#   e1_{group}.csv.gz      one wide table per group: eid + {short_name}_i{instance}[_a{array}]
#   e1_{fc.prefix}.csv.gz  eid, instance, {prefix}_0001 ... (long by instance)
#   e1_fc_bad.csv          connectivity files that could not be parsed, with the reason
#   e1_fc_refetch.bulk     (eid, {field}_{instance}_0) with a main-table pointer but no file (ukbfetch -b)
#   e1_dictionary.csv      output column -> field / instance / array / label / subgroup / source column
#   e1_missing.csv         (field, instance) requested by a list but absent from the main table
#   e1_availability.csv    subgroup x instance: n features, n with any value, n complete cases
#
#   Rscript aging/R/e1_extract_fields.R
#   Rscript aging/R/e1_extract_fields.R --main a.csv,b.csv --out /tmp/x --groups brain_tian,covariates
#   Rscript aging/R/e1_extract_fields.R --groups fc --cores 8 [--fc-dir DIR]
#   Rscript aging/R/e1_extract_fields.R --groups blood
#   Rscript aging/R/e1_extract_fields.R --no-gate          # no imaging row gate (memory heavy)
# With --groups, the dictionary / availability / missing files keep the rows of the other groups.

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
cfg <- read_yaml(file.path(here, "..", "config_aging.yaml"))
repo_root <- normalizePath(file.path(here, "..", ".."))
# Config paths are relative to the repository root unless absolute
rp <- function(x) if (grepl("^(/|[A-Za-z]:)", x)) x else file.path(repo_root, x)

args <- commandArgs(trailingOnly = TRUE)
opt <- function(name, default = NULL) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) return(default)
  if (i == length(args) || startsWith(args[i + 1], "--")) return(TRUE)
  args[i + 1]
}

# ── Paths ──────────────────────────────────────────────────────────
main_files <- opt("main")
main_files <- if (is.null(main_files)) rp(cfg$paths$main_file) else strsplit(main_files, ",")[[1]]
miss <- main_files[!file.exists(main_files)]
if (length(miss)) die("main table not found: %s (set paths.main_file or use --main)",
                      paste(miss, collapse = " | "))
lists_dir <- opt("lists", rp(cfg$paths$feature_lists))
if (!dir.exists(lists_dir)) die("feature-list directory not found: %s", lists_dir)
out_root <- opt("out", rp(cfg$paths$out))
out_dir <- file.path(out_root, "extract")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
say("main table: %s", paste(main_files, collapse = " | "))
say("feature lists: %s", lists_dir)
say("output: %s", out_dir)

ex <- cfg$extract
fcc <- cfg$fc
img_inst <- as.integer(unlist(ex$instances))
groups_sel <- opt("groups")
groups_sel <- if (is.null(groups_sel)) NULL else strsplit(groups_sel, ",")[[1]]
all_groups <- c(names(ex$feature_groups), names(ex$extra_groups), "fc")
if (!is.null(groups_sel)) {
  bad <- setdiff(groups_sel, all_groups)
  if (length(bad)) die("undefined groups in --groups: %s (available: %s)", paste(bad, collapse = ", "),
                       paste(all_groups, collapse = ", "))
}
do_fc <- is.null(groups_sel) || "fc" %in% groups_sel

# ── 1. Requested fields ────────────────────────────────────────────
# insts is stored as a string: "all" or "2,3"
want <- list()
for (g in names(ex$feature_groups)) {
  spec <- ex$feature_groups[[g]]
  d <- fread(file.path(lists_dir, spec$list), encoding = "UTF-8")
  sub_col <- intersect(c("system", "organ"), names(d))[1]
  d <- d[!field %in% unlist(spec$skip_fields)]
  # A field listed under two organs (MULTI 21090) becomes one row with subgroup "adipose;pancreas"
  d <- d[, .(short_name = short_name[1], label = label[1],
             subgroup = paste(unique(get(sub_col)), collapse = ";")), by = field]
  want[[g]] <- d[, .(group = g, field, short_name, label, subgroup,
                     insts = paste(img_inst, collapse = ","), ord = .I)]
}
for (g in names(ex$extra_groups)) {
  spec <- ex$extra_groups[[g]]
  f <- spec$fields
  insts <- if (identical(spec$instances, "all")) "all" else paste(unlist(spec$instances), collapse = ",")
  want[[g]] <- data.table(group = g, field = as.integer(names(f)), short_name = unlist(f),
                          label = unlist(f), subgroup = g, insts = insts, ord = seq_along(f))
}
want <- rbindlist(want)
if (!is.null(groups_sel)) want <- want[group %in% groups_sel]
say("requested: %s", paste(c(if (nrow(want)) want[, sprintf("%s %d", group[1], .N), by = group]$V1,
                             if (do_fc) sprintf("fc (field %s) %d edges", fcc$field, fcc$n_edges)), collapse = ", "))

# ── 2. Parse headers into (field, instance, array) ─────────────────
parse_header <- function(cols) {
  out <- data.table(col = cols, field = NA_integer_, instance = NA_integer_, array = NA_integer_)
  # pattern, capture-group index of (field, instance, array); 0 = not encoded (recorded as 0)
  pats <- list(
    list("^(\\d+)-(\\d+)\\.(\\d+)$",       c(1, 2, 3)),   # ukbconv csv
    list("^f\\.(\\d+)\\.(\\d+)\\.(\\d+)$", c(1, 2, 3)),   # ukbconv r / ukbtools
    list("^p(\\d+)_i(\\d+)_a(\\d+)$",      c(1, 2, 3)),   # RAP
    list("^p(\\d+)_i(\\d+)$",              c(1, 2, 0)),
    list("^p(\\d+)_a(\\d+)$",              c(1, 0, 2)),
    list("^p(\\d+)$",                      c(1, 0, 0)),
    list("_f(\\d+)_(\\d+)_(\\d+)$",        c(1, 2, 3)))   # ukbtools descriptive names
  for (p in pats) {
    todo <- is.na(out$field) & grepl(p[[1]], cols)
    if (!any(todo)) next
    m <- regmatches(cols[todo], regexec(p[[1]], cols[todo]))
    grab <- function(k) if (k == 0) 0L else as.integer(vapply(m, `[`, "", k + 1))
    out[todo, `:=`(field = grab(p[[2]][1]), instance = grab(p[[2]][2]), array = grab(p[[2]][3]))]
  }
  out
}
hdr <- rbindlist(lapply(main_files, function(f) {
  h <- names(fread(f, nrows = 0))
  eid_col <- h[grepl("^(f\\.|participant\\.)?eid$", h)][1]
  if (is.na(eid_col)) die("no eid column in %s (eid / f.eid / participant.eid)", f)
  p <- parse_header(setdiff(h, eid_col))
  n_bad <- sum(is.na(p$field))
  say("  %s: %d columns, %d parsed%s", basename(f), length(h) - 1, nrow(p) - n_bad,
      if (n_bad) sprintf(" (%d without a field id, ignored)", n_bad) else "")
  p[!is.na(field)][, `:=`(file = f, eid_col = eid_col)]
}))
# The same (field, instance, array) in several files: the earlier file wins
hdr <- unique(hdr, by = c("field", "instance", "array"))

# ── 3. Requested x available ───────────────────────────────────────
split_insts <- function(r) as.integer(strsplit(r, ",")[[1]])
hit <- merge(want, hdr, by = "field", allow.cartesian = TRUE)
hit <- hit[insts == "all" | vapply(seq_len(.N), function(k)
  insts[k] != "all" && instance[k] %in% split_insts(insts[k]), TRUE)]
hit[, multi_arr := max(c(0L, array)) > 0, by = .(group, field, instance)]
hit[, out_col := paste0(short_name, "_i", instance, fifelse(multi_arr, paste0("_a", array), ""))]
dup <- hit[, .N, by = .(group, out_col)][N > 1]
if (nrow(dup)) die("duplicated output columns (check short_name in the lists): %s",
                   paste(head(dup$out_col, 5), collapse = ", "))
setorder(hit, group, instance, ord, array)

fc_hdr <- if (do_fc) hdr[field == as.integer(fcc$field) & instance %in% img_inst] else hdr[0]

# Missing: groups with explicit instances are checked per (field, instance); "all" groups per field
wf <- want[insts != "all"]
ni <- lengths(lapply(wf$insts, split_insts))
exp_fixed <- data.table(wf[rep(seq_len(nrow(wf)), ni), .(group, field, short_name, label)],
                        instance = as.integer(unlist(lapply(wf$insts, split_insts))))
missing <- rbind(
  exp_fixed[!hit, on = .(group, field, instance)],
  want[insts == "all"][!hit, on = .(group, field)][, .(group, field, short_name, label, instance = NA_integer_)])
# With --groups, summary files keep the rows of the groups not run this time; otherwise a partial run
# would overwrite the dictionary and a0 / a2 would see no features for the other modalities.
keep_other_groups <- function(new, fname) {
  f <- file.path(out_dir, fname)
  if (is.null(groups_sel) || !file.exists(f) || file.size(f) == 0) return(new)
  old <- fread(f)
  if (!"group" %in% names(old)) return(new)
  old <- old[!group %in% groups_sel]
  if (nrow(old)) say("  %s: kept %d rows of groups %s", fname, nrow(old), paste(unique(old$group), collapse = ", "))
  rbind(old, new, fill = TRUE)
}
# Connectivity completeness is checked against the bulk directory in step 6, not here
fwrite(keep_other_groups(missing, "e1_missing.csv"), file.path(out_dir, "e1_missing.csv"))
say("%d columns found%s; %d requested (field, instance) absent -> e1_missing.csv", nrow(hit),
    if (do_fc) sprintf(" + %d connectivity pointer columns", nrow(fc_hdr)) else "", nrow(missing))
if (nrow(missing)) print(missing[, .(n_missing = .N), by = .(group, instance)])

# ── 4. Row gate: participants with an imaging attendance date ──────
read_block <- function(f, eid_col, cols, ...) {
  x <- fread(f, select = c(eid_col, cols), na.strings = c("", "NA"), showProgress = FALSE, ...)
  setnames(x, eid_col, "eid")
  x
}
# gate: false groups keep all participants (negative eids and withdrawals still removed). When only such
# groups run, field 53 is not read.
no_gate <- isTRUE(opt("no-gate"))
ungated <- names(ex$extra_groups)[vapply(ex$extra_groups, function(s) isFALSE(s$gate), TRUE)]
grp_run <- unique(want$group)
need_gate <- !no_gate && (do_fc || any(!grp_run %in% ungated))
need_all <- no_gate || any(grp_run %in% ungated)
wd_file <- opt("withdraw", rp(cfg$paths$withdraw))
if (isTRUE(wd_file) || !file.exists(wd_file)) {
  warn("withdrawal list not found (paths.withdraw); only negative eids removed")
  wd <- integer(0)
} else {
  wd <- fread(wd_file, header = FALSE)[[1]]
  say("withdrawal list %s: %d eids", basename(wd_file), length(wd))
}
clean_eids <- function(x, what) {
  x <- as.integer(x)
  x <- x[!is.na(x) & x > 0]    # negative eid = withdrawn participant placeholder
  if (length(wd)) say("  %s: %d removed by the withdrawal list", what, sum(x %in% wd))
  sort(setdiff(x, wd))
}
e_all <- integer(0)
if (need_all) {
  e_all <- unique(unlist(lapply(unique(hdr$file), function(f)
    fread(f, select = hdr[file == f]$eid_col[1], showProgress = FALSE)[[1]])))
  e_all <- clean_eids(e_all, "all participants")
  say("ungated groups (%s): %d participants", if (no_gate) "--no-gate, all groups" else
      paste(intersect(grp_run, ungated), collapse = ", "), length(e_all))
}
if (no_gate) {
  warn("--no-gate: reading all participants")
  e <- e_all
} else if (need_gate) {
  gate <- hdr[field == as.integer(ex$gate_field) & instance %in% img_inst]
  if (!nrow(gate)) die("field %s at instance %s (row gate) not in the main table; add it or use --no-gate",
                       ex$gate_field, paste(img_inst, collapse = "/"))
  e <- integer(0)
  for (f in unique(gate$file)) {
    g <- read_block(f, gate[file == f]$eid_col[1], gate[file == f]$col, colClasses = "character")
    e <- union(e, g$eid[rowSums(!is.na(g[, -1])) > 0])
  }
  e <- clean_eids(e, "imaging cohort")
  say("row gate: %d participants with an attendance date (field %s) at instance %s (%s)",
      length(e), ex$gate_field, paste(img_inst, collapse = "/"), elapsed())
} else {
  e <- integer(0)
}
eids_of <- function(g) if (no_gate || g %in% ungated) e_all else e

# ── 5. Tabular fields: all groups read together in column blocks ───
chunk <- as.integer(ex$chunk_cols)
res <- setNames(lapply(unique(hit$group), function(g) data.table(eid = eids_of(g))), unique(hit$group))
for (f in unique(hit$file)) {
  hf <- unique(hit[file == f], by = "col")         # each source column read once
  idx <- split(seq_len(nrow(hf)), ceiling(seq_len(nrow(hf)) / chunk))
  for (b in seq_along(idx)) {
    cols <- hf$col[idx[[b]]]
    x <- read_block(f, hf$eid_col[1], cols)
    hk <- hit[file == f & col %in% cols]
    for (g in unique(hk$group)) {
      hg <- hk[group == g]
      rows <- match(res[[g]]$eid, x$eid)
      res[[g]][, (hg$out_col) := x[rows, hg$col, with = FALSE]]
    }
    say("  %s block %d/%d (%d columns) read (%s)", basename(f), b, length(idx), length(cols), elapsed())
    rm(x); invisible(gc())
  }
}

av_list <- list()
count_nonmissing <- function(dt, cols) Reduce(`+`, lapply(cols, function(c) !is.na(dt[[c]])), 0L)
for (g in names(res)) {
  h <- hit[group == g]
  setcolorder(res[[g]], c("eid", h$out_col))
  out <- file.path(out_dir, sprintf("e1_%s.csv.gz", g))
  fwrite(res[[g]], out)
  say("  %-12s %6d participants x %4d columns -> %s", g, nrow(res[[g]]), ncol(res[[g]]) - 1, basename(out))
  # Availability by subgroup; annotated subgroups such as "GM(...)" are counted under GM
  if (g %in% names(ex$feature_groups)) {
    tok <- h[, .(sg = sub("\\(.*$", "", strsplit(subgroup, ";")[[1]])), by = .(out_col, instance)]
    tok <- rbind(tok, h[, .(out_col, instance, sg = "ALL")])
    av <- tok[, {
      k <- count_nonmissing(res[[g]], out_col)     # column-wise counts; no person x column matrix
      .(n_features = length(out_col), n_any = sum(k > 0), n_complete = sum(k == length(out_col)))
    }, by = .(subgroup = sg, instance)]
    av_list[[g]] <- av[, .(group = g, subgroup, instance, n_features, n_any, n_complete)]
    print(av_list[[g]][, -1])
  }
  res[[g]] <- NULL
  invisible(gc())
}

# ── 6. Connectivity from bulk files ────────────────────────────────
# Non-missing main-table pointers ({field}-2.0 / -3.0) mean a file should exist; pointers without a file are
# written to a ukbfetch re-fetch list.
read_fc_file <- function(p, n) {
  v <- tryCatch({
    con <- if (endsWith(p, ".zip")) {
      inner <- unzip(p, list = TRUE)$Name
      inner <- inner[!endsWith(inner, "/")]
      if (length(inner) != 1) stop(sprintf("zip holds %d files", length(inner)))
      unz(p, inner)
    } else file(p)
    on.exit(close(con))
    tok <- unlist(strsplit(readLines(con, warn = FALSE), "[,;[:space:]]+"))
    suppressWarnings(as.numeric(tok[nzchar(tok)]))
  }, error = function(err) conditionMessage(err))
  if (is.character(v)) return(v)
  if (length(v) != n) return(sprintf("%d values (expected %d)", length(v), n))
  if (any(!is.finite(v))) return("non-numeric / NA / Inf values")
  v
}
dict_fc <- NULL
if (do_fc) {
  n <- as.integer(fcc$n_edges)
  fld <- as.integer(fcc$field)
  pre <- fcc$prefix
  edge_cols <- sprintf("%s_%04d", pre, seq_len(n))
  say("  connectivity: field %d (%s), %d values per file", fld, fcc$label, n)
  fc_dir <- opt("fc-dir", rp(cfg$paths$fc_bulk_dir))
  if (isTRUE(fc_dir) || !dir.exists(fc_dir))
    die("connectivity bulk directory not found (set paths.fc_bulk_dir or use --fc-dir)")
  bf <- data.table(file = list.files(fc_dir, pattern = "^\\d+_\\d+_\\d+_\\d+\\.(txt|zip)$"))
  if (!nrow(bf)) die("no {eid}_{field}_{instance}_{array}.txt / .zip files in %s", fc_dir)
  bf[, c("eid", "field", "instance", "array") :=
       lapply(tstrsplit(sub("\\.(txt|zip)$", "", file), "_"), as.integer)]
  say("  bulk directory %s: %s", fc_dir, paste(bf[, .N, by = .(field, instance)][order(field, instance),
      sprintf("field %d i%d %d files", field, instance, N)], collapse = ", "))
  if (any(bf$field != fld))
    warn("files of field %s ignored (fc.field = %d)", paste(setdiff(unique(bf$field), fld), collapse = "/"), fld)
  bf <- bf[field == fld & instance %in% img_inst]
  if (!nrow(bf)) die("no files of field %d at instance %s in %s", fld, paste(img_inst, collapse = "/"), fc_dir)
  setorder(bf, eid, instance, file)          # .txt preferred over .zip for the same (eid, instance)
  bf <- unique(bf, by = c("eid", "instance"))
  n_out <- sum(!bf$eid %in% e)
  bf <- bf[eid %in% e]
  if (n_out) say("  %d files of participants outside the cohort ignored", n_out)

  # Reconcile main-table pointers (cohort, non-missing) with the files present
  refetch <- file.path(out_dir, "e1_fc_refetch.bulk")
  unlink(refetch)
  if (nrow(fc_hdr)) {
    if (uniqueN(fc_hdr$file) > 1) die("pointer columns of field %d spread over several main-table files", fld)
    ptr <- read_block(fc_hdr$file[1], fc_hdr$eid_col[1], fc_hdr$col, colClasses = list(character = fc_hdr$col))
    ptr <- melt(ptr[eid %in% e], id.vars = "eid", measure.vars = fc_hdr$col, variable.name = "col",
                value.name = "ptr", na.rm = TRUE, variable.factor = FALSE)[nzchar(ptr)]
    ptr[fc_hdr, on = "col", instance := i.instance]
    lack <- ptr[!bf, on = .(eid, instance)]
    extra <- bf[!ptr, on = .(eid, instance)]
    say("  pointers %d (cohort), files present %d, missing %d; files without pointer %d",
        nrow(ptr), nrow(ptr) - nrow(lack), nrow(lack), nrow(extra))
    by_i <- merge(ptr[, .(pointer = .N), by = instance], lack[, .(lack = .N), by = instance], all = TRUE)
    by_i <- merge(by_i, extra[, .(no_pointer = .N), by = instance], all = TRUE)
    for (k in setdiff(names(by_i), "instance")) set(by_i, which(is.na(by_i[[k]])), k, 0L)
    print(by_i[, .(instance, pointer, have = pointer - lack, lack, no_pointer)])
    if (nrow(lack)) {
      writeLines(sprintf("%d %d_%d_0", lack$eid, fld, lack$instance), refetch)
      warn("%d missing files written to e1_fc_refetch.bulk (ukbfetch -b)", nrow(lack))
    }
  } else {
    warn("no pointer columns of field %d in the main table; download completeness not checked", fld)
  }

  # Each file must contain exactly n finite values; otherwise it is recorded as bad and skipped
  cores <- as.integer(opt("cores", "1"))
  lap <- if (cores > 1 && .Platform$OS.type == "unix") {
    function(x, f) parallel::mclapply(x, f, mc.cores = cores)
  } else lapply
  say("  reading %d connectivity files (%d cores)", nrow(bf), cores)
  r <- lap(file.path(fc_dir, bf$file), function(p) read_fc_file(p, n))
  ok <- vapply(r, is.numeric, TRUE)
  bad <- bf[!ok, .(eid, instance, file, why = vapply(r[!ok], function(z) as.character(z)[1], ""))]
  fwrite(bad, file.path(out_dir, "e1_fc_bad.csv"))
  if (nrow(bad)) print(bad[, .N, by = why])
  if (!any(ok)) die("no connectivity file could be parsed; see e1_fc_bad.csv")
  m <- matrix(unlist(r[ok]), ncol = n, byrow = TRUE, dimnames = list(NULL, edge_cols))
  rm(r); invisible(gc())
  fc <- cbind(bf[ok, .(eid, instance)], as.data.table(m))
  setorder(fc, instance, eid)
  say("  connectivity: %d parsed, %d bad; range [%.3f, %.3f], mean %.4f (%s)", sum(ok), nrow(bad),
      min(m), max(m), mean(m), elapsed())
  fc_file <- sprintf("e1_%s.csv.gz", pre)
  fwrite(fc, file.path(out_dir, fc_file))
  say("  fc -> %s (%d rows, long by instance)", fc_file, nrow(fc))
  for (i in sort(unique(bf$instance)))
    av_list[[paste0("fc", i)]] <- data.table(group = "fc", subgroup = "FC", instance = i, n_features = n,
                                             n_any = sum(bf$instance == i), n_complete = sum(bf$instance[ok] == i))
  dict_fc <- data.table(group = "fc", out_col = edge_cols, field = fld, instance = NA_integer_,
                        array = seq_len(n) - 1L, label = sprintf("%s, edge %d", fcc$label, seq_len(n)),
                        subgroup = "FC", source_file = basename(fc_dir),
                        source_col = sprintf("{eid}_%d_{instance}_0.txt value %d", fld, seq_len(n)))
  # Edge -> node pair: upper triangle in column-major order (FMRIB netmat convention), which is the order
  # of which(upper.tri())
  gc_file <- rp(fcc$good_components)
  if (!file.exists(gc_file)) {
    warn("fc.good_components not found; dictionary has edge numbers only")
  } else {
    ic <- scan(gc_file, quiet = TRUE)
    D <- length(ic)
    if (D * (D - 1) / 2 != n) die("%s lists %d good components -> %d edges, but fc.n_edges = %d",
                                  basename(gc_file), D, D * (D - 1) / 2, n)
    rc <- which(upper.tri(diag(D)), arr.ind = TRUE)
    dict_fc[, `:=`(node_row = rc[, 1], node_col = rc[, 2], ic_row = ic[rc[, 1]], ic_col = ic[rc[, 2]])]
    dict_fc[, label := sprintf("%s, IC%d-IC%d", fcc$label, ic_row, ic_col)]
    say("  good components: %d (%s); edge 1 = IC%d-IC%d, edge %d = IC%d-IC%d", D, basename(gc_file),
        dict_fc$ic_row[1], dict_fc$ic_col[1], n, dict_fc$ic_row[n], dict_fc$ic_col[n])
  }
  rm(m, fc); invisible(gc())
}

if (length(av_list)) fwrite(keep_other_groups(rbindlist(av_list), "e1_availability.csv"),
                            file.path(out_dir, "e1_availability.csv"))
dict <- rbind(hit[, .(group, out_col, field, instance, array, label, subgroup, source_file = basename(file), source_col = col)],
              dict_fc, fill = TRUE)
dict <- keep_other_groups(dict, "e1_dictionary.csv")
fwrite(dict, file.path(out_dir, "e1_dictionary.csv"))
say("done (%s). Dictionary %d rows (%s) -> e1_dictionary.csv", elapsed(), nrow(dict),
    paste(dict[, .N, by = group][, sprintf("%s %d", group, N)], collapse = ", "))
