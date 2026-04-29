#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript export_gene_lists.R input.rds output_prefix")
}

input <- args[1]
prefix <- args[2]

genesets <- readRDS(input)

# ensure named list
if (is.null(names(genesets))) {
    stop("Gene list has no names")
}

# write JSON (best for Python)
jsonlite::write_json(
    genesets,
    paste0(prefix, ".json"),
    pretty = TRUE,
    auto_unbox = TRUE
)

# write GMT (useful for enrichment tools)
gmt <- unlist(lapply(names(genesets), function(n) {
    paste(c(n, "NA", genesets[[n]]), collapse = "\t")
}))
writeLines(gmt, paste0(prefix, ".gmt"))

message("Exported: ", prefix)