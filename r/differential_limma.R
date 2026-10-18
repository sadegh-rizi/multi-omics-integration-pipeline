#!/usr/bin/env Rscript

# Region-wise limma models using the same BioGroup-by-Sex parameterization as
# src/mia_multiomics/stats/differential.py. Input values are already log2-transformed and normalized.

library(limma)

effects <- c(
  "MIA vs Saline Control" = "BioGroupMIA",
  "Female vs Male" = "SexFemale",
  "MIA x Female interaction" = "BioGroupMIA:SexFemale"
)

build_design <- function(metadata) {
  metadata$BioGroup <- relevel(factor(metadata$BioGroup), ref = "Saline Control")
  metadata$Sex <- relevel(factor(metadata$Sex), ref = "Male")
  model.matrix(~ BioGroup * Sex, data = metadata)
}

validate_inputs <- function(values, metadata) {
  required_metadata <- c("Sample", "Region", "BioGroup", "Sex", "Mouse number")
  missing_metadata <- setdiff(required_metadata, names(metadata))
  if (length(missing_metadata) > 0) {
    stop("Metadata is missing columns: ", paste(missing_metadata, collapse = ", "))
  }
  if (!"Feature" %in% names(values)) {
    stop("Expression matrix is missing the Feature column.")
  }
  if (anyDuplicated(values$Feature)) {
    stop("Feature identifiers must be unique.")
  }
  if (anyDuplicated(metadata$Sample)) {
    stop("Metadata sample identifiers must be unique.")
  }
  missing_samples <- setdiff(names(values)[-1], metadata$Sample)
  if (length(missing_samples) > 0) {
    stop("Metadata is missing intensity samples: ", paste(missing_samples, collapse = ", "))
  }
}

fit_region <- function(values, metadata, region, min_per_cell = 3, fdr_denominator = "fitted", effect_scale = "normalized_log2") {
  region_metadata <- metadata[metadata$Region == region, , drop = FALSE]
  sample_names <- region_metadata$Sample
  expression <- as.matrix(values[, sample_names, drop = FALSE])
  storage.mode(expression) <- "numeric"
  rownames(expression) <- values$Feature

  cells <- expand.grid(
    BioGroup = c("Saline Control", "MIA"), Sex = c("Male", "Female"), stringsAsFactors = FALSE
  )
  cell_counts <- vapply(
    seq_len(nrow(expression)),
    function(i) {
      vapply(
        seq_len(nrow(cells)),
        function(cell) {
          sum(!is.na(expression[i, ]) & region_metadata$BioGroup == cells$BioGroup[cell] & region_metadata$Sex == cells$Sex[cell])
        },
        integer(1)
      )
    },
    integer(4)
  )
  if (is.null(dim(cell_counts))) {
    cell_counts <- matrix(cell_counts, ncol = 1)
  }
  keep <- apply(cell_counts, 2, min) >= min_per_cell
  if (!any(keep)) {
    return(data.frame())
  }

  fit <- eBayes(lmFit(expression[keep, , drop = FALSE], build_design(region_metadata)))
  counts <- t(cell_counts[, keep, drop = FALSE])
  colnames(counts) <- c(
    "n_saline_control_male", "n_mia_male", "n_saline_control_female", "n_mia_female"
  )
  results <- lapply(names(effects), function(effect) {
    coefficient <- effects[[effect]]
    table <- topTable(fit, coef = coefficient, number = Inf, sort.by = "none", adjust.method = "BH")
    denominator <- if (fdr_denominator == "all_features") nrow(expression) else nrow(table)
    table$adj.P.Val <- p.adjust(table$P.Value, method = "BH", n = denominator)
    standard_error <- fit$stdev.unscaled[, coefficient] * sqrt(fit$s2.post)
    confidence_multiplier <- qt(0.975, df = fit$df.total)
    data.frame(
      Feature = rownames(table),
      Region = region,
      Effect = effect,
      estimate_log2 = table$logFC,
      effect_scale = effect_scale,
      standard_error = standard_error,
      ci_lower_log2 = table$logFC - confidence_multiplier * standard_error,
      ci_upper_log2 = table$logFC + confidence_multiplier * standard_error,
      fold_change = if (effect_scale == "normalized_log2") 2 ^ table$logFC else NA_real_,
      moderated_t = table$t,
      p_value = table$P.Value,
      p_value_fdr = table$adj.P.Val,
      residual_df = fit$df.residual,
      prior_df = fit$df.prior,
      posterior_df = fit$df.total,
      n_usable = rowSums(!is.na(expression[keep, , drop = FALSE])),
      counts,
      fdr_denominator = denominator,
      check.names = FALSE
    )
  })
  do.call(rbind, results)
}

run_differential_moderated <- function(input_matrix, input_metadata, output_file, min_per_cell = 3, fdr_denominator = "fitted", effect_scale = "normalized_log2") {
  values <- read.csv(input_matrix, check.names = FALSE, stringsAsFactors = FALSE)
  metadata <- read.csv(input_metadata, check.names = FALSE, stringsAsFactors = FALSE)
  validate_inputs(values, metadata)
  values <- values[, c("Feature", metadata$Sample), drop = FALSE]
  results <- lapply(unique(metadata$Region), function(region) {
    fit_region(values, metadata, region, min_per_cell, fdr_denominator, effect_scale)
  })
  results <- Filter(function(result) nrow(result) > 0, results)
  if (length(results) == 0) {
    stop("No features met the per-cell eligibility requirement.")
  }
  write.csv(do.call(rbind, results), output_file, row.names = FALSE)
}

if (sys.nframe() == 0) {
  arguments <- commandArgs(trailingOnly = TRUE)
  if (length(arguments) == 0) {
    script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
    if (length(script_argument) != 1) {
      stop("Run this analysis with Rscript r/differential_limma.R.")
    }
    script_path <- sub("^--file=", "", script_argument)
    project_root <- normalizePath(file.path(dirname(script_path), ".."))
    results_dir <- file.path(project_root, "results", "mia_vs_control")
    arguments <- c(
      file.path(results_dir, "00_inputs", "differential_input_matrix.csv"),
      file.path(results_dir, "00_inputs", "differential_input_metadata.csv"),
      file.path(results_dir, "03_differential", "differential_effects_limma.csv")
    )
  }
  if (length(arguments) < 3 || length(arguments) > 5) {
    stop("Expected INPUT_MATRIX INPUT_METADATA OUTPUT_FILE [FDR_DENOMINATOR] [EFFECT_SCALE], or no arguments for the primary pipeline.")
  }
  denominator <- if (length(arguments) >= 4) arguments[[4]] else "fitted"
  if (!denominator %in% c("fitted", "all_features")) {
    stop("FDR_DENOMINATOR must be 'fitted' or 'all_features'.")
  }
  effect_scale <- if (length(arguments) == 5) arguments[[5]] else "normalized_log2"
  if (!effect_scale %in% c("normalized_log2", "pareto_scaled_log2")) {
    stop("EFFECT_SCALE must be 'normalized_log2' or 'pareto_scaled_log2'.")
  }
  run_differential_moderated(arguments[[1]], arguments[[2]], arguments[[3]], fdr_denominator = denominator, effect_scale = effect_scale)
}
