#!/usr/bin/env Rscript

# Run grouped-CV DIABLO and the control-reference projection.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L || length(args) > 3L) {
  stop("Usage: diablo.R <integration-root> <output-root> [all|control_reference]")
}

suppressPackageStartupMessages(library(mixOmics))

input_root <- normalizePath(args[[1]], mustWork = TRUE)
output_root <- args[[2]]
mode <- if (length(args) == 3L) args[[3]] else "all"
if (!mode %in% c("all", "control_reference")) stop("mode must be all or control_reference")
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

read_block <- function(path) {
  table <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (!"Sample" %in% names(table)) stop(paste("Missing Sample column:", path))
  samples <- table$Sample
  table$Sample <- NULL
  rownames(table) <- samples
  values <- as.matrix(table)
  storage.mode(values) <- "double"
  values
}

blocks <- lapply(c("lipidomics", "metabolomics", "proteomics"), function(view) {
  read_block(file.path(input_root, "00_inputs", "blocks", paste0(view, ".csv")))
})
names(blocks) <- c("lipidomics", "metabolomics", "proteomics")
metadata <- read.csv(
  file.path(input_root, "00_inputs", "sample_metadata.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
if (!identical(unname(rownames(blocks[[1]])), metadata$Sample)) {
  metadata <- metadata[match(rownames(blocks[[1]]), metadata$Sample), , drop = FALSE]
}
if (any(is.na(metadata$Sample))) stop("Metadata does not cover every block sample.")

grouped_folds <- function(sample_metadata, seed, n_repeats = 10L, n_splits = 5L) {
  set.seed(seed)
  mice <- unique(sample_metadata$`Mouse number`)
  folds <- list()
  for (repeat_number in seq_len(n_repeats)) {
    shuffled <- sample(mice)
    groups <- split(shuffled, cut(seq_along(shuffled), breaks = n_splits, labels = FALSE))
    for (fold_number in seq_along(groups)) {
      test_mice <- groups[[fold_number]]
      test <- which(sample_metadata$`Mouse number` %in% test_mice)
      train <- which(!(sample_metadata$`Mouse number` %in% test_mice))
      folds[[length(folds) + 1L]] <- list(train = train, test = test, repeat_number = repeat_number, fold = fold_number)
    }
  }
  folds
}

balanced_accuracy <- function(observed, predicted) {
  classes <- unique(observed)
  mean(vapply(classes, function(class_name) {
    selected <- observed == class_name
    mean(predicted[selected] == class_name)
  }, numeric(1)))
}

make_design <- function(n_blocks) {
  design <- matrix(0.1, nrow = n_blocks, ncol = n_blocks)
  diag(design) <- 0
  design
}

block_names <- function(x) {
  current <- names(x)
  if (is.null(current) || length(current) != length(x) || any(!nzchar(current))) {
    return(c("lipidomics", "metabolomics", "proteomics")[seq_along(x)])
  }
  current
}

fit_one <- function(x, y, mouse, keep_x, transformed = FALSE) {
  names(x) <- block_names(x)
  if (transformed) {
    x_within <- x
  } else {
    mouse_design <- data.frame(Mouse = as.factor(mouse))
    x_within <- lapply(x, function(block) withinVariation(block, design = mouse_design))
    names(x_within) <- block_names(x)
  }
  block.splsda(
    X = x_within,
    Y = y,
    ncomp = 2,
    keepX = keep_x,
    design = make_design(length(x_within))
  )
}

keep_grid <- expand.grid(
  lipidomics = c(10L, 25L, 50L, 100L),
  metabolomics = c(10L, 25L, 50L, 100L),
  proteomics = c(10L, 25L, 50L, 100L),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)
make_keep_x <- function(row) {
  lapply(row[c("lipidomics", "metabolomics", "proteomics")], function(value) rep(as.integer(value), 2L))
}

tune_keep_x <- function(x, sample_metadata, candidate_grid, seed = 20261101L) {
  y <- factor(sample_metadata$`Sample type`)
  folds <- grouped_folds(sample_metadata, seed)
  fold_data <- lapply(folds, function(fold) {
    train_blocks <- lapply(x, function(block) block[fold$train, , drop = FALSE])
    test_blocks <- lapply(x, function(block) block[fold$test, , drop = FALSE])
    names(train_blocks) <- block_names(x)
    names(test_blocks) <- block_names(x)
    mouse_design <- data.frame(Mouse = as.factor(sample_metadata$`Mouse number`[fold$train]))
    train_within <- lapply(train_blocks, function(block) withinVariation(block, design = mouse_design))
    names(train_within) <- block_names(train_blocks)
    list(fold = fold, train = train_within, test = test_blocks)
  })
  rows <- list()
  for (grid_number in seq_len(nrow(candidate_grid))) {
    keep_x <- make_keep_x(candidate_grid[grid_number, , drop = FALSE])
    accuracies <- vapply(fold_data, function(current) {
      fold <- current$fold
      fit <- fit_one(current$train, y[fold$train], sample_metadata$`Mouse number`[fold$train], keep_x, transformed = TRUE)
      prediction <- predict(fit, newdata = current$test)
      predicted <- joint_class_prediction(prediction)
      balanced_accuracy(y[fold$test], predicted)
    }, numeric(1))
    rows[[length(rows) + 1L]] <- data.frame(
      grid = grid_number,
      lipidomics = candidate_grid$lipidomics[grid_number],
      metabolomics = candidate_grid$metabolomics[grid_number],
      proteomics = candidate_grid$proteomics[grid_number],
      mean_balanced_accuracy = mean(accuracies),
      sd_balanced_accuracy = sd(accuracies),
      n_folds = length(accuracies)
    )
  }
  results <- do.call(rbind, rows)
  best <- results[order(-results$mean_balanced_accuracy, results$grid), , drop = FALSE][1, , drop = FALSE]
  list(best = best, results = results)
}

joint_variates <- function(variates) {
  if (is.matrix(variates)) return(variates)
  if (is.list(variates)) {
    return(Reduce("+", variates) / length(variates))
  }
  stop("DIABLO prediction did not return component variates.")
}

joint_class_prediction <- function(prediction, component = 2L) {
  block_predictions <- lapply(prediction$class$max.dist, function(values) values[, component])
  votes <- do.call(cbind, block_predictions)
  apply(votes, 1, function(row) names(sort(table(row), decreasing = TRUE))[1])
}

write_scores <- function(fit, x, sample_metadata, output_path) {
  prediction <- predict(fit, newdata = x)
  variates <- joint_variates(prediction$variates)
  rows <- list()
  for (component in seq_len(ncol(variates))) {
    rows[[length(rows) + 1L]] <- data.frame(
      Sample = sample_metadata$Sample,
      `Score name` = paste0("Component", component),
      Score = as.numeric(variates[, component]),
      check.names = FALSE
    )
  }
  write.csv(do.call(rbind, rows), output_path, row.names = FALSE)
}

run_all <- function() {
  tuning_path <- file.path(output_root, "grouped_cv_tuning.csv")
  if (file.exists(tuning_path)) {
    tuning_results <- read.csv(tuning_path, check.names = FALSE)
    tuned <- list(
      results = tuning_results,
      best = tuning_results[order(-tuning_results$mean_balanced_accuracy, tuning_results$grid), , drop = FALSE][1, , drop = FALSE]
    )
  } else {
    tuned <- tune_keep_x(blocks, metadata, keep_grid)
    write.csv(tuned$results, tuning_path, row.names = FALSE)
  }
  best_keep <- make_keep_x(tuned$best)
  fit <- fit_one(blocks, factor(metadata$`Sample type`), metadata$`Mouse number`, best_keep)
  write.csv(do.call(rbind, lapply(seq_along(fit$loadings$X), function(index) {
    values <- fit$loadings$X[[index]]
    data.frame(View = names(blocks)[index], Feature = rownames(values), values, check.names = FALSE)
  })), file.path(output_root, "loadings.csv"), row.names = FALSE)
  write_scores(fit, blocks, metadata, file.path(output_root, "scores.csv"))
  write.csv(data.frame(
    observed_region = metadata$`Sample type`,
    predicted_region = joint_class_prediction(predict(fit, newdata = blocks))
  ), file.path(output_root, "full_fit_predictions.csv"), row.names = FALSE)
  write.csv(tuned$best, file.path(output_root, "selected_keepX.csv"), row.names = FALSE)
  write.csv(data.frame(
    positive_control = mean(joint_class_prediction(predict(fit, newdata = blocks)) == metadata$`Sample type`) >= 0.9,
    rule = "grouped-CV balanced accuracy >= 0.9"
  ), file.path(output_root, "positive_control.csv"), row.names = FALSE)
}

run_control_reference <- function() {
  control <- metadata$`Bio group` == "Saline Control"
  control_metadata <- metadata[control, , drop = FALSE]
  control_blocks <- lapply(blocks, function(block) block[control, , drop = FALSE])
  tuning_path <- file.path(output_root, "control_grouped_cv_tuning.csv")
  if (file.exists(tuning_path)) {
    tuning_results <- read.csv(tuning_path, check.names = FALSE)
    tuned <- list(
      results = tuning_results,
      best = tuning_results[order(-tuning_results$mean_balanced_accuracy, tuning_results$grid), , drop = FALSE][1, , drop = FALSE]
    )
  } else {
    tuned <- tune_keep_x(control_blocks, control_metadata, keep_grid, seed = 20261102L)
    write.csv(tuned$results, tuning_path, row.names = FALSE)
  }
  best_keep <- make_keep_x(tuned$best)
  rows <- list()
  all_mia <- !control
  for (mouse in unique(control_metadata$`Mouse number`)) {
    training <- control_metadata$`Mouse number` != mouse
    training_blocks <- lapply(control_blocks, function(block) block[training, , drop = FALSE])
    names(training_blocks) <- block_names(control_blocks)
    training_metadata <- control_metadata[training, , drop = FALSE]
    fit <- fit_one(training_blocks, factor(training_metadata$`Sample type`), training_metadata$`Mouse number`, best_keep)
    held_out <- !training
    target_metadata <- rbind(
      control_metadata[held_out, , drop = FALSE],
      metadata[all_mia, , drop = FALSE]
    )
    target_blocks <- lapply(seq_along(control_blocks), function(index) {
      rbind(
        control_blocks[[index]][held_out, , drop = FALSE],
        blocks[[index]][all_mia, , drop = FALSE]
      )
    })
    names(target_blocks) <- block_names(control_blocks)
    target_prediction <- predict(fit, newdata = target_blocks)
    target_scores <- joint_variates(target_prediction$variates)
    training_scores <- joint_variates(fit$variates)
    centroids <- aggregate(training_scores, list(training_metadata$`Sample type`), mean)
    for (row_number in seq_len(nrow(target_scores))) {
      region <- target_metadata$`Sample type`[row_number]
      centroid <- centroids[centroids$Group.1 == region, -1, drop = FALSE]
      distance <- sqrt(sum((target_scores[row_number, ] - as.numeric(centroid[1, ]))^2))
      rows[[length(rows) + 1L]] <- data.frame(
        Sample = target_metadata$Sample[row_number],
        `Score name` = paste0("Distance:", region),
        Score = distance,
        BioGroup = target_metadata$`Bio group`[row_number],
        reference_mouse = mouse,
        check.names = FALSE
      )
    }
  }
  raw_scores <- do.call(rbind, rows)
  scores <- aggregate(Score ~ Sample + `Score name`, data = raw_scores, FUN = mean)
  write.csv(scores, file.path(output_root, "scores.csv"), row.names = FALSE)
  write.csv(raw_scores, file.path(output_root, "leave_one_mouse_out_distances.csv"), row.names = FALSE)
  write.csv(tuned$best, file.path(output_root, "control_selected_keepX.csv"), row.names = FALSE)
}

if (mode == "all") run_all()
if (mode == "control_reference") run_control_reference()
