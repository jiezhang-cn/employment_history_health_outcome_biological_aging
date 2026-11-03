#!/usr/bin/env Rscript
# m5 (R part): preranked gene-set enrichment of the Olink layer with fgsea. Run py/m5_enrich.py first
# (writes <out>/derived/m5_t.csv.gz).
#
# Per sex x term: stats = t (beta / SE); the background is the Olink genes measured for that term, not the
# genome (the Olink panel is itself a selection, so a genome background would mistake panel membership for
# enrichment). Four gene-set libraries (source):
#   GO_BP / GO_CC / GO_MF (MSigDB C5 GO:BP / GO:CC / GO:MF) and KEGG (C2 CP:KEGG_LEGACY, else CP:KEGG_MEDICUS)
#   Source priority: --gmt files -> msigdbr package -> GMT files listed in files.gmt under paths.refs.
#   For GMT files the library is taken from the file name (go.bp / go.cc / go.mf / kegg).
# Gene-set size enrich.min_size-max_size (members within the background); fgseaMultilevel with
# nPermSimple = enrich.nperm and eps = enrich.eps; fixed seed. padj = BH within sex x term x source.
#
# Output: <out>/tables/m5_fgsea.csv  sex, term, analysis, source, pathway, size, ES, NES, pval, padj, leading_edge
#
#   Rscript molecular/R/m5_fgsea.R [--threads 4] [--gmt a.gmt,b.gmt]

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
args <- commandArgs(trailingOnly = TRUE)
opt <- function(name, default = NULL) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) return(default)
  if (i == length(args) || startsWith(args[i + 1], "--")) return(TRUE)
  args[i + 1]
}
threads <- as.integer(opt("threads", 1))

if (!requireNamespace("fgsea", quietly = TRUE)) die("R package fgsea is required: BiocManager::install('fgsea')")
suppressPackageStartupMessages(library(fgsea))

# ── Paths ─────────────────────────────────────────────────────────────
out <- from_root(cfg$paths$out)
f_t <- file.path(out, "derived", "m5_t.csv.gz")
f_out <- file.path(out, "tables", "m5_fgsea.csv")
dir.create(dirname(f_out), recursive = TRUE, showWarnings = FALSE)
cols_out <- c("sex", "term", "analysis", "source", "pathway", "size", "ES", "NES", "pval", "padj", "leading_edge")
write_empty <- function(why) {
  warn("%s; writing empty %s", why, basename(f_out))
  fwrite(setnames(data.table(matrix(character(0), 0, length(cols_out))), cols_out), f_out)
  quit(status = 0)
}
if (!file.exists(f_t)) die("%s not found (run py/m5_enrich.py first)", f_t)
tt <- tryCatch(fread(f_t, showProgress = FALSE),     # some R.utils versions fail on .gz; fall back to gzfile
               error = function(e) {
                 warn("fread failed on .gz (%s); reading via gzfile", conditionMessage(e))
                 con <- gzfile(f_t, "rt"); on.exit(close(con))
                 fread(text = readLines(con), showProgress = FALSE)
               })
say("m5_t: %d rows, %d sex x term vectors (%s)", nrow(tt), uniqueN(tt[, .(sex, term)]), elapsed())

# ── Gene sets ─────────────────────────────────────────────────────────
ec <- cfg$enrich
pw <- list()
src_of <- function(f) {
  b <- tolower(basename(f))
  if (grepl("go[._]?bp", b)) return("GO_BP")
  if (grepl("go[._]?cc", b)) return("GO_CC")
  if (grepl("go[._]?mf", b)) return("GO_MF")
  if (grepl("kegg", b)) return("KEGG")
  NA_character_
}
add_gmt <- function(f) {
  s <- src_of(f)
  if (is.na(s)) { warn("cannot tell which library %s belongs to (GO_BP / GO_CC / GO_MF / KEGG); skipped", basename(f)); return() }
  g <- lapply(gmtPathways(f), toupper)
  pw[[s]] <<- c(pw[[s]], g[setdiff(names(g), names(pw[[s]]))])
  say("GMT %s -> %s: %d sets", basename(f), s, length(g))
}
gmt_arg <- opt("gmt", NULL)
if (is.character(gmt_arg)) {
  for (f in strsplit(gmt_arg, ",")[[1]]) {
    if (!file.exists(f)) die("--gmt file not found: %s", f)
    add_gmt(f)
  }
} else if (requireNamespace("msigdbr", quietly = TRUE)) {
  m <- function(cat, sub) {      # newer msigdbr: collection / subcollection; older: category / subcategory
    d <- tryCatch(msigdbr::msigdbr(species = "Homo sapiens", collection = cat, subcollection = sub),
                  error = function(e) tryCatch(msigdbr::msigdbr(species = "Homo sapiens", category = cat,
                                                                subcategory = sub), error = function(e) NULL))
    if (is.null(d) || !nrow(d)) return(NULL)
    d <- as.data.table(d)
    split(toupper(d$gene_symbol), d$gs_name)
  }
  pw$GO_BP <- m("C5", "GO:BP")
  pw$GO_CC <- m("C5", "GO:CC")
  pw$GO_MF <- m("C5", "GO:MF")
  pw$KEGG <- m("C2", "CP:KEGG_LEGACY")
  if (is.null(pw$KEGG)) pw$KEGG <- m("C2", "CP:KEGG_MEDICUS")
  pw <- pw[!vapply(pw, is.null, TRUE)]
  say("msigdbr %s: %s", as.character(packageVersion("msigdbr")),
      paste(sprintf("%s %d", names(pw), lengths(pw)), collapse = ", "))
} else {
  refs <- from_root(cfg$paths$refs)
  for (g in unlist(cfg$files$gmt)) {
    f <- file.path(refs, g)
    if (!file.exists(f)) { warn("%s not found in paths.refs", g); next }
    add_gmt(f)
  }
}
miss <- setdiff(c("GO_BP", "GO_CC", "GO_MF", "KEGG"), names(pw))
if (length(miss)) warn("gene-set libraries missing: %s", paste(miss, collapse = ", "))
if (!length(pw)) write_empty("no msigdbr package and none of the files.gmt GMT files found")

# ── fgsea ─────────────────────────────────────────────────────────────
nperm <- as.integer(ec$nperm)
eps <- if (is.null(ec$eps)) 1e-50 else as.numeric(ec$eps)
if (threads > 1) suppressPackageStartupMessages(library(BiocParallel))
res <- list()
for (k in unique(tt[, paste(sex, term, sep = "|")])) {
  g <- tt[paste(sex, term, sep = "|") == k]
  st <- setNames(g$t, toupper(g$gene))
  st <- st[is.finite(st)]
  st <- st[!duplicated(names(st))]
  st <- st + seq_along(st) * 1e-12 * sd(st)            # break ties (far below the precision of t)
  parts <- list()
  for (src in names(pw)) {
    set.seed(as.integer(ec$seed))
    f <- suppressWarnings(fgseaMultilevel(pathways = pw[[src]], stats = st, minSize = ec$min_size,
                                          maxSize = ec$max_size, nPermSimple = nperm, eps = eps,
                                          nproc = threads))
    if (!nrow(f)) next
    f[, source := src]
    parts[[src]] <- f
  }
  if (!length(parts)) next
  f <- rbindlist(parts, fill = TRUE)
  f[, padj := p.adjust(pval, "BH"), by = source]
  f[, leading_edge := vapply(leadingEdge, function(x) paste(x, collapse = ";"), "")]
  f[, `:=`(sex = g$sex[1], term = g$term[1], analysis = g$analysis[1])]
  res[[k]] <- f[, ..cols_out]
  say("  %-28s %4d genes -> %4d sets, padj < %.2f: %d (%s)", k, length(st), nrow(f), cfg$alpha,
      sum(f$padj < cfg$alpha, na.rm = TRUE), elapsed())
}
if (!length(res)) write_empty("no sex x term had gene sets within the size range")
r <- rbindlist(res)
fwrite(r, f_out)
say("  [OK] %s: %d rows; %d with padj < alpha (%s)", basename(f_out), nrow(r), sum(r$padj < cfg$alpha, na.rm = TRUE),
    elapsed())
