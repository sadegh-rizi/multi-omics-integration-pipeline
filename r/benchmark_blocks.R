# Run the R-side MFA and block.spls benchmark arms.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: benchmark_blocks.R <integration-root> <output-root>")
input_root <- normalizePath(args[[1]], mustWork = TRUE)
output_root <- args[[2]]
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

read_block <- function(path) {
  table <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  samples <- table$Sample
  table$Sample <- NULL
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

suppressPackageStartupMessages(library(FactoMineR))
combined <- as.data.frame(do.call(cbind, blocks), check.names = FALSE)
mfa <- MFA(
  base = combined,
  group = vapply(blocks, ncol, integer(1)),
  type = rep("s", length(blocks)),
  name.group = names(blocks),
  ncp = 10,
  graph = FALSE
)
write.csv(data.frame(Sample = metadata$Sample, mfa$ind$coord, check.names = FALSE), file.path(output_root, "mfa_scores.csv"), row.names = FALSE)
write.csv(mfa$eig, file.path(output_root, "mfa_eigenvalues.csv"), row.names = FALSE)
write.csv(mfa$group$contrib, file.path(output_root, "mfa_group_contributions.csv"), row.names = TRUE)

suppressPackageStartupMessages(library(mixOmics))
response <- model.matrix(~ 0 + metadata$`Sample type`)
colnames(response) <- c("Cerebellum", "Cortex", "Hippocampus")
block_fit <- block.spls(
  X = blocks,
  Y = response,
  ncomp = 2,
  keepX = lapply(blocks, function(block) rep(min(50L, ncol(block)), 2L)),
  design = matrix(0.1, nrow = length(blocks), ncol = length(blocks))
)
write.csv(do.call(rbind, lapply(seq_along(block_fit$loadings$X), function(index) {
  values <- block_fit$loadings$X[[index]]
  data.frame(View = names(blocks)[index], Feature = rownames(values), values, check.names = FALSE)
})), file.path(output_root, "block_spls_loadings.csv"), row.names = FALSE)
write.csv(data.frame(Sample = metadata$Sample, block_fit$variates$X[[1]], check.names = FALSE), file.path(output_root, "block_spls_scores.csv"), row.names = FALSE)
