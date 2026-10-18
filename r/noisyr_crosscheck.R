args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(3, 5)) stop("Usage: Rscript noisyr_crosscheck.R input.csv curve.csv thresholds.csv [window step]")

suppressPackageStartupMessages(library(noisyr))
values <- as.matrix(read.csv(args[[1]], row.names = 1, check.names = FALSE))
storage.mode(values) <- "double"
if (any(is.infinite(values))) stop("Input matrix may contain finite values or missing values only.")

MIN_PAIRWISE_OBSERVATIONS <- 10L
MIN_COMPARATORS_PER_REFERENCE <- 3L
MIN_PAIRWISE_CORRELATIONS_PER_POINT <- 10L

calculate_guarded_expression_similarity_counts <- function(
    expression.matrix,
    n.elements.per.window,
    n.step,
    min.pairwise.observations = MIN_PAIRWISE_OBSERVATIONS,
    min.comparators.per.reference = MIN_COMPARATORS_PER_REFERENCE) {
  n.sequence <- seq.int(
    1L,
    nrow(expression.matrix) - n.elements.per.window + 1L,
    by = n.step
  )
  expression.levels <- matrix(
    NA_real_,
    nrow = length(n.sequence),
    ncol = ncol(expression.matrix)
  )
  expression.levels.similarity <- expression.levels
  n.pairwise.correlations <- matrix(0L, nrow = length(n.sequence), ncol = ncol(expression.matrix))
  n.usable.comparators <- n.pairwise.correlations

  for (j in seq_len(ncol(expression.matrix))) {
    sorted.positions <- order(expression.matrix[, j], na.last = TRUE)
    expression.matrix.sorted <- expression.matrix[sorted.positions, , drop = FALSE]
    for (idx in seq_along(n.sequence)) {
      positions <- n.sequence[[idx]]:(n.sequence[[idx]] + n.elements.per.window - 1L)
      col.j <- expression.matrix.sorted[positions, j]
      expression.levels[idx, j] <- mean(col.j, na.rm = TRUE)
      similarity.vector <- rep(NA_real_, ncol(expression.matrix))

      for (k in seq_len(ncol(expression.matrix))) {
        if (j == k) next
        col.k <- expression.matrix.sorted[positions, k]
        complete <- is.finite(col.j) & is.finite(col.k)
        if (sum(complete) < min.pairwise.observations) next
        cols.in <- cbind(col.j[complete], col.k[complete])
        similarity.vector[k] <- suppressWarnings(
          stats::cor(cols.in, method = "pearson", use = "pairwise.complete.obs")[1, 2]
        )
      }

      usable <- is.finite(similarity.vector)
      n.usable.comparators[idx, j] <- sum(usable)
      n.pairwise.correlations[idx, j] <- sum(usable)
      if (sum(usable) >= min.comparators.per.reference) {
        expression.levels.similarity[idx, j] <- mean(similarity.vector[usable], na.rm = TRUE)
      }
    }
  }

  list(
    expression.matrix = expression.matrix,
    expression.levels = expression.levels,
    expression.levels.similarity = expression.levels.similarity,
    n.pairwise.correlations = n.pairwise.correlations,
    n.usable.comparators = n.usable.comparators
  )
}

window <- if (length(args) == 5) as.integer(args[[4]]) else round(nrow(values) / 10)
step <- if (length(args) == 5) as.integer(args[[5]]) else max(floor(window * 0.05), 1L)
summary <- calculate_guarded_expression_similarity_counts(
  expression.matrix = values,
  n.elements.per.window = window,
  n.step = step
)

thresholds <- rep(NA_real_, ncol(values))
for (j in seq_len(ncol(values))) {
  usable <- is.finite(summary$expression.levels[, j]) &
    is.finite(summary$expression.levels.similarity[, j])
  if (!any(usable)) next
  sample_summary <- list(
    expression.matrix = values[, j, drop = FALSE],
    expression.levels = matrix(summary$expression.levels[usable, j], ncol = 1),
    expression.levels.similarity = matrix(summary$expression.levels.similarity[usable, j], ncol = 1)
  )
  threshold <- calculate_noise_threshold(
    expression = sample_summary,
    similarity.threshold = 0.25,
    method.chosen = "Boxplot-IQR"
  )
  thresholds[j] <- threshold[[1]]
}
thresholds[!is.finite(thresholds)] <- NA_real_

safe_median <- function(x) {
  median(x, na.rm = TRUE)
}

point_pairwise_counts <- rowSums(summary$n.pairwise.correlations)
supported_points <- point_pairwise_counts >= MIN_PAIRWISE_CORRELATIONS_PER_POINT
curve <- data.frame(
  window_index = (seq_len(nrow(summary$expression.levels)) - 1L)[supported_points],
  median_log2_abundance = apply(summary$expression.levels, 1, safe_median)[supported_points],
  median_pearson = apply(summary$expression.levels.similarity, 1, safe_median)[supported_points],
  n_reference_samples = rowSums(is.finite(summary$expression.levels.similarity))[supported_points],
  n_usable_comparators = rowSums(summary$n.usable.comparators)[supported_points],
  n_pairwise_correlations = point_pairwise_counts[supported_points],
  window_size = window,
  window_step = step,
  min_pairwise_observations = MIN_PAIRWISE_OBSERVATIONS,
  min_comparators_per_reference = MIN_COMPARATORS_PER_REFERENCE,
  min_pairwise_correlations_per_point = MIN_PAIRWISE_CORRELATIONS_PER_POINT
)
write.csv(curve, args[[2]], row.names = FALSE)
write.csv(data.frame(sample = colnames(values), threshold_log2_abundance = thresholds), args[[3]], row.names = FALSE)
