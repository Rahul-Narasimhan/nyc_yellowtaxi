# NYC Taxi Fare Prediction: Temporal XGBoost Pipeline

## Overview

This project builds a **multi-year tabular ML pipeline** for predicting NYC Yellow Taxi fares using **2023–2025 monthly parquet files**. The work focuses on three things:

1. **Reliable data engineering** across messy real-world parquet files  
2. **Leakage-safe temporal evaluation** instead of random train/test splits  
3. **Controlled XGBoost experiments** to understand which features and hyperparameters actually help future generalization  

---

## Problem Statement

Given trip-level taxi metadata, can we predict `fare_amount` using only features that are valid at prediction time?

The goal was not just to train an XGBoost model, but to build a workflow that answers questions like:

- How clean and consistent is the raw data across years?
- What data quality issues must be handled before modeling?
- How should a fare model be evaluated in a **temporally realistic** way?
- Which spatial features actually help?
- Where does the model fail?

---

## Dataset

The project uses **NYC Yellow Taxi monthly parquet files** for:

- **2023**: 12 monthly files
- **2024**: 12 monthly files
- **2025**: 11 monthly files (Jan–Nov)
- **123M+ normalized rows** across 2023–2025

## Repository Structure

```text
nyc_taxi/
├── reports/
│   └── data_quality/             # generated audits / summaries
├── notebooks/
│   ├── 03_baseline_split_and_sanity_checks.ipynb
│   ├── 04_xgboost_baseline.ipynb
│   └── 05_xgboost_hyperparameter_study.ipynb
├── src/
│   └── data/
│       ├── schema_audit.py
│       ├── make_dataset_multi_year.py
│       ├── data_quality_audit.py
│       └── build_model_df_multi_year.py
├── .gitignore
└── README.md
```

## What I built

### Data pipeline
- Audited schema across all raw parquet files
- Normalized column-name and dtype inconsistencies across years
- Built yearly normalized parquet outputs
- Created a model-ready dataset with temporal validity and first-pass cleaning rules

### Modeling workflow
- Defined a leakage-safe temporal split:
  - **Train:** 2023–2024
  - **Validation:** Jan–Jun 2025
  - **Test:** Jul–Nov 2025
- Built an XGBoost baseline with train-fitted preprocessing
- Ran controlled feature ablations
- Ran a targeted hyperparameter study
- Performed segment-level error analysis

## Stable baseline result

Using:
- numeric: `trip_distance`, `passenger_count`, `pickup_hour`, `pickup_weekday`, `pickup_month`
- categorical: `VendorID`, `RatecodeID`, `store_and_fwd_flag`, `PULocationID`, `DOLocationID`

| Split | MAE | RMSE |
|---|---:|---:|
| Validation | 2.2227 | 4.7689 |
| Test | 2.8350 | 5.7415 |

## Key feature experiments

| Experiment | Valid MAE | Test MAE |
|---|---:|---:|
| Baseline with `PULocationID` + `DOLocationID` | 2.2227 | 2.8350 |
| Add naive `PU_DO_pair` | 2.2432 | 2.8754 |
| Add route-frequency feature | 2.2744 | 3.0127 |
| Remove `PULocationID` / `DOLocationID` | 2.4113 | 3.0647 |

### Main takeaway
Pickup/dropoff zone features helped materially, but naive route-interaction features did **not** improve future holdout performance.

## Hyperparameter study

I ran a focused XGBoost study across:
- `max_depth`
- `learning_rate`
- `n_estimators`
- `subsample`
- `colsample_bytree`
- `min_child_weight`

Main findings:
- shallow trees underfit
- deeper trees improved small-sample MAE slightly
- lower learning rate with more trees was competitive
- stronger sampling / conservative regularization did not clearly improve future holdout MAE

## Error analysis

The most important modeling insight came from error slicing, not just the headline metric.

Main findings:
- error rises sharply in the **fare tail**
- error increases for **longer trips**
- **rare routes** are much harder than common ones
- later 2025 performs worse than earlier 2025, suggesting temporal degradation
