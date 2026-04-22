# Sleep Apnea Detection — Pipeline Overview

ECG-based sleep apnea detection pipeline using HRV feature engineering and machine learning.

## Directory Structure

```
apnea-project/
├── data/
│   ├── raw/          # Standardized raw recordings
│   ├── rpeaks/       # Detected R-peaks and RR intervals
│   └── features/     # Extracted HRV feature matrices
├── notebooks/        # Analysis notebooks (run in order)
├── models/           # Trained model artifacts
├── reports/
│   └── figures/      # Generated plots and figures
├── tableau/          # Tableau-ready exports
└── README.md
```

## Notebook Pipeline

| # | Notebook | Purpose | Inputs | Outputs |
|---|----------|---------|--------|---------|
| 00 | `00_environment_check.ipynb` | Verify Python version and all required packages are installed | None | Console confirmation of environment readiness |
| 01 | `01_data_ingestion.ipynb` | Load raw ECG/PSG files, validate integrity, standardize format | Raw source files (EDF, CSV, or vendor format) | `data/raw/` recordings; file manifest CSV |
| 02 | `02_signal_inspection.ipynb` | Visually and statistically assess signal quality; flag artifacts | `data/raw/`; manifest from 01 | Quality report in `reports/`; figures in `reports/figures/`; exclusion list |
| 03 | `03_rpeak_detection.ipynb` | Detect R-peaks and compute RR intervals using validated algorithm | `data/raw/`; quality flags from 02 | R-peak timestamps and RR series in `data/rpeaks/` |
| 04 | `04_eda.ipynb` | Explore RR distributions, apnea labels, class balance, and demographics | `data/rpeaks/`; apnea annotations; subject metadata | EDA figures in `reports/figures/`; class balance summary |
| 05 | `05_feature_extraction.ipynb` | Compute time-domain, frequency-domain, and nonlinear HRV features per window | `data/rpeaks/`; apnea labels; window parameters | `data/features/features.parquet`; feature description table |
| 06 | `06_modeling.ipynb` | Train and tune classifiers (LR, RF, XGBoost) with cross-validation | `data/features/`; train/val split | Trained model artifacts in `models/`; CV scores; tuning results |
| 07 | `07_evaluation.ipynb` | Evaluate final model on held-out test set; compute clinical metrics | `models/`; test split from `data/features/` | Metrics report in `reports/`; ROC/PR curves, SHAP plots in `reports/figures/` |
| 08 | `08_tableau_export.ipynb` | Prepare and export datasets for Tableau dashboard creation | `data/features/`; `reports/` metrics; model predictions | CSV / `.hyper` files in `tableau/`; data dictionary |

## Status

All notebooks: **NOT STARTED**
