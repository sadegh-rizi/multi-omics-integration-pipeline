#!/usr/bin/env Rscript

# Run the FactoMineR MFA cross-check and control reference.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L || length(args) > 3L) {
  stop("Usage: mfa.R <integration-root> <output-root> [all|control_reference]")
}

suppressPackageStartupMessages(library(FactoMineR))

input_root <- normalizePath(args[[1]], mustWork = TRUE)
output_root <- args[[2]]
mode <- if (length(args) == 3L) args[[3]] else "all"
if (!mode %in% c("all", "control_reference")) stop("mode must be all or control_reference")
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

read_block <- function(path) {
  table <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  samples <- table$Sample
  table$Sample <- NULL
  rownames(table) <- samples
  values <- as.data.frame(lapply(table, as.numeric), check.names = FALSE)
  rownames(values) <- samples
  values
}

blocks <- lapply(c("lipidomics", "metabolomics", "proteomics"), function(view) {
  read_block(file.path(input_root, "00_inputs", "blocks", paste0(view, ".csv")))
})
names(blocks) <- c("lipidomics", "metabolomics", "proteomics")
metadata <- read.csv(file.path(input_root, "00_inputs", "sample_metadata.csv"), check.names = FALSE, stringsAsFactors = FALSE)
metadata <- metadata[match(rownames(blocks[[1]]), metadata$Sample), , drop = FALSE]
if (any(is.na(metadata$Sample))) stop("Metadata does not cover every MFA sample.")

run_mfa <- function(block_list, sample_metadata, supplementary = integer(0), ncp = 10L) {
  combined <- as.data.frame(do.call(cbind, block_list), check.names = FALSE)
  arguments <- list(
    base = combined,
    group = vapply(block_list, ncol, integer(1)),
    type = rep("s", length(block_list)),
    name.group = names(block_list),
    ncp = ncp,
    graph = FALSE
  )
  if (length(supplementary)) arguments$ind.sup <- supplementary
  do.call(MFA, arguments)
}

write_mfa_scores <- function(fit, sample_metadata, output_path) {
  active <- fit$ind$coord
  rows <- lapply(seq_len(ncol(active)), function(component) {
    data.frame(
      Sample = sample_metadata$Sample[seq_len(nrow(active))],
      `Score name` = paste0("MFA", component),
      Score = active[, component]
    )
  })
  write.csv(do.call(rbind, rows), output_path, row.names = FALSE)
}

if (mode == "all") {
  fit <- run_mfa(blocks, metadata)
  write_mfa_scores(fit, metadata, file.path(output_root, "scores.csv"))
  write.csv(fit$eig, file.path(output_root, "eigenvalues.csv"), row.names = FALSE)
  write.csv(fit$group$contrib, file.path(output_root, "group_contributions.csv"), row.names = TRUE)
}

if (mode == "control_reference") {
  control <- metadata$`Bio group` == "Saline Control"
  active_blocks <- lapply(blocks, function(block) block[control, , drop = FALSE])
  supplementary_blocks <- lapply(blocks, function(block) block[!control, , drop = FALSE])
  combined_blocks <- Map(rbind, active_blocks, supplementary_blocks)
  active_metadata <- metadata[control, , drop = FALSE]
  combined_metadata <- rbind(active_metadata, metadata[!control, , drop = FALSE])
  fit <- run_mfa(combined_blocks, combined_metadata, supplementary = seq_len(sum(!control)) + sum(control))
  write_mfa_scores(fit, active_metadata, file.path(output_root, "control_scores.csv"))
  write.csv(fit$ind.sup$coord, file.path(output_root, "mia_supplementary_scores.csv"), row.names = TRUE)
  write.csv(fit$eig, file.path(output_root, "control_eigenvalues.csv"), row.names = FALSE)
}
