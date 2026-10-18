library(testthat)

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
test_directory <- if (length(file_argument) == 1) {
  dirname(normalizePath(sub("^--file=", "", file_argument)))
} else {
  normalizePath("r")
}
source(file.path(test_directory, "differential_limma.R"))

test_that("limma models use the planned design and return all effects", {
  metadata <- expand.grid(
    BioGroup = c("Saline Control", "MIA"),
    Sex = c("Male", "Female"),
    Replicate = 1:3,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  metadata$Sample <- sprintf("sample_%02d", seq_len(nrow(metadata)))
  metadata$Region <- "Cortex"
  metadata$`Mouse number` <- sprintf("mouse_%02d", seq_len(nrow(metadata)))
  metadata <- metadata[, c("Sample", "Region", "BioGroup", "Sex", "Mouse number")]

  design <- build_design(metadata)
  expect_equal(colnames(design), c("(Intercept)", "BioGroupMIA", "SexFemale", "BioGroupMIA:SexFemale"))

  values <- data.frame(Feature = c("mia_signal", "null_feature"), check.names = FALSE)
  values[metadata$Sample] <- rbind(
    8 + 2 * (metadata$BioGroup == "MIA") + rep(c(-0.1, 0, 0.1), 4),
    8 + rep(c(-0.1, 0, 0.1), 4)
  )
  matrix_file <- tempfile(fileext = ".csv")
  metadata_file <- tempfile(fileext = ".csv")
  output_file <- tempfile(fileext = ".csv")
  write.csv(values, matrix_file, row.names = FALSE)
  write.csv(metadata, metadata_file, row.names = FALSE)

  run_differential_moderated(matrix_file, metadata_file, output_file)
  results <- read.csv(output_file, check.names = FALSE)
  expect_setequal(results$Effect, c("MIA vs Saline Control", "Female vs Male", "MIA x Female interaction"))
  expect_true(all(c("ci_lower_log2", "ci_upper_log2") %in% names(results)))
  expect_true(all(results$moderated_t[results$Feature == "mia_signal" & results$Effect == "MIA vs Saline Control"] > 0))
  expect_true(all(results$p_value[results$Feature == "mia_signal" & results$Effect == "MIA vs Saline Control"] < 0.05))
})
