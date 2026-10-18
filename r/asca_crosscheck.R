args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: Rscript asca_crosscheck.R complete_matrix.csv metadata.csv output.csv")
}

matrix_path <- args[[1L]]
metadata_path <- args[[2L]]
output_path <- args[[3L]]

values <- as.matrix(read.csv(matrix_path, row.names = 1L, check.names = FALSE))
storage.mode(values) <- "double"
metadata <- read.csv(metadata_path, check.names = FALSE, stringsAsFactors = FALSE)
metadata <- metadata[match(colnames(values), metadata$Sample), , drop = FALSE]
if (anyNA(metadata$Sample)) stop("Metadata does not cover every matrix sample.")
if (any(!is.finite(values))) stop("The cross-check requires a complete finite matrix.")

metadata$Region <- factor(
  metadata[["Sample type"]],
  levels = c("Cerebellum", "Cortex", "Hippocampus")
)
metadata$Group <- factor(
  metadata[["Bio group"]],
  levels = c("Saline Control", "MIA")
)
metadata$Sex <- factor(metadata$Sex, levels = c("Male", "Female"))
metadata$Mouse <- factor(metadata[["Mouse number"]])

# ALASCA is not on CRAN and is not present in the pinned renv lockfile.  Keep
# the check explicit: when a local ALASCA installation is available, the
# provenance records it; otherwise this independent R fit is the declared
# estimation cross-check, not a permutation test or a substitute for ALASCA.
alasca_available <- requireNamespace("ALASCA", quietly = TRUE) ||
  requireNamespace("alasca", quietly = TRUE)
method <- if (alasca_available) "ALASCA package detected; base-R reference fit used for reproducible term extraction" else "base-R sum-coded ASCA reference; ALASCA package unavailable"

options(contrasts = c("contr.sum", "contr.poly"))
formula <- stats::as.formula("response ~ Region * Group * Sex + Mouse")
model_matrix <- stats::model.matrix(stats::delete.response(stats::terms(formula)), metadata)
assignments <- attr(model_matrix, "assign")
term_labels <- attr(stats::terms(formula), "term.labels")
term_names <- c("Region", "Group", "Sex", "Group:Sex", "Mouse", "Region:Group", "Region:Sex", "Region:Group:Sex")

pinv <- function(x, tolerance = 1e-10) {
  decomposition <- svd(x)
  keep <- decomposition$d > max(decomposition$d) * tolerance
  if (!any(keep)) return(matrix(0, ncol(x), nrow(x)))
  decomposition$v[, keep, drop = FALSE] %*%
    diag(1 / decomposition$d[keep], nrow = sum(keep)) %*%
    t(decomposition$u[, keep, drop = FALSE])
}

model_pinv <- pinv(model_matrix)
rows <- list()
profile_rows <- list()
for (feature_index in seq_len(nrow(values))) {
  response <- as.numeric(values[feature_index, ])
  coefficients <- model_pinv %*% response
  fitted <- as.numeric(model_matrix %*% coefficients)
  centered_response <- response - mean(response)
  total_ssq <- sum(centered_response^2)
  for (term in term_names) {
    term_index <- match(term, term_labels)
    if (is.na(term_index)) next
    columns <- which(assignments == term_index)
    contribution <- as.numeric(model_matrix[, columns, drop = FALSE] %*% coefficients[columns, , drop = FALSE])
    rows[[length(rows) + 1L]] <- data.frame(
      Feature = rownames(values)[feature_index],
      Effect = term,
      ssq = sum(contribution^2),
      variance_share = if (total_ssq > 0) sum(contribution^2) / total_ssq else NA_real_,
      method = method,
      alasca_package_available = alasca_available,
      stringsAsFactors = FALSE
    )
  }
  rows[[length(rows) + 1L]] <- data.frame(
    Feature = rownames(values)[feature_index],
    Effect = "Model fitted value",
    ssq = sum((fitted - mean(response))^2),
    variance_share = if (total_ssq > 0) sum((fitted - mean(response))^2) / total_ssq else NA_real_,
    method = method,
    alasca_package_available = alasca_available,
    stringsAsFactors = FALSE
  )
  region_columns <- which(assignments == match("Region", term_labels))
  interaction_columns <- which(assignments == match("Region:Group", term_labels))
  region_effect <- as.numeric(model_matrix[, region_columns, drop = FALSE] %*% coefficients[region_columns, , drop = FALSE])
  interaction_effect <- as.numeric(model_matrix[, interaction_columns, drop = FALSE] %*% coefficients[interaction_columns, , drop = FALSE])
  region_difference <- function(effect, group_mask, region_a, region_b) {
    mean(effect[group_mask & metadata$Region == region_a]) -
      mean(effect[group_mask & metadata$Region == region_b])
  }
  all_samples <- rep(TRUE, nrow(metadata))
  profile_rows[[length(profile_rows) + 1L]] <- data.frame(
    Feature = rownames(values)[feature_index],
    Region_Cortex_minus_Cerebellum = region_difference(region_effect, all_samples, "Cortex", "Cerebellum"),
    Region_Cortex_minus_Hippocampus = region_difference(region_effect, all_samples, "Cortex", "Hippocampus"),
    Region_Hippocampus_minus_Cerebellum = region_difference(region_effect, all_samples, "Hippocampus", "Cerebellum"),
    RegionGroup_Cortex_minus_Cerebellum = region_difference(interaction_effect, metadata$Group == "MIA", "Cortex", "Cerebellum") -
      region_difference(interaction_effect, metadata$Group == "Saline Control", "Cortex", "Cerebellum"),
    RegionGroup_Cortex_minus_Hippocampus = region_difference(interaction_effect, metadata$Group == "MIA", "Cortex", "Hippocampus") -
      region_difference(interaction_effect, metadata$Group == "Saline Control", "Cortex", "Hippocampus"),
    RegionGroup_Hippocampus_minus_Cerebellum = region_difference(interaction_effect, metadata$Group == "MIA", "Hippocampus", "Cerebellum") -
      region_difference(interaction_effect, metadata$Group == "Saline Control", "Hippocampus", "Cerebellum"),
    stringsAsFactors = FALSE
  )
}
result <- do.call(rbind, rows)
write.csv(result, output_path, row.names = FALSE)
write.csv(do.call(rbind, profile_rows), sub("\\.csv$", "_profiles.csv", output_path), row.names = FALSE)

alasca_status <- data.frame(
  method = method,
  package_available = alasca_available,
  status = "base-R reference written",
  message = "",
  stringsAsFactors = FALSE
)
if (alasca_available) {
  wide <- cbind(
    metadata[, c("Region", "Group", "Sex", "Mouse"), drop = FALSE],
    as.data.frame(t(values), check.names = FALSE)
  )
  alasca_fit <- tryCatch(
    ALASCA::ALASCA(
      wide,
      formula = ~ Region * Group * Sex + (1 | Mouse),
      wide = TRUE,
      participant_column = "Mouse",
      x_column = "Region",
      effects = c("Region", "Region:Group"),
      separate_effects = TRUE,
      validate = FALSE,
      use_Rfast = FALSE
    ),
    error = function(error) error
  )
  if (inherits(alasca_fit, "error")) {
    alasca_status$status <- "ALASCA call failed; base-R reference retained"
    alasca_status$message <- conditionMessage(alasca_fit)
  } else {
    saveRDS(alasca_fit, sub("\\.csv$", "_model.rds", output_path))
    alasca_status$status <- "ALASCA model saved; base-R term table retained for comparison"
  }
}
write.csv(alasca_status, sub("\\.csv$", "_status.csv", output_path), row.names = FALSE)
