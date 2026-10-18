"""mia_multiomics: analysis pipeline for the MIA multi-omics mouse study.

The package is organised in layers (see the README for the full map):

1. data        - config, toy_data, inputs, preprocessing, imputation
2. methods     - stats/, multivariate/, outliers/, integration/, qc, lipids,
                 noise_floor, plotting (tables in, tables out)
3. stages      - stages/: one module per analysis step, writing results/;
                 pipeline.py runs them in order
4. exploration - explore/: figures recomputed from user-chosen settings

Typical use:

    from mia_multiomics.config import Settings
    from mia_multiomics import pipeline
    pipeline.run_all(Settings.fast("path/to/project"))
"""
