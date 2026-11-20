# Well Log Machine Learning: Porosity (PHIE) Prediction and Net/Gross Classification

Machine learning exercise on well **A-18-00** logs, built in **Python (Jupyter)** and replicated in **KNIME 5.11.0**.

## Goals

1. **Regression:** predict effective porosity (`PHIE`) from `NPHI`, `RHOB`, `DT` and `GR`.
2. **Classification:** predict the `net_gross` flag from the same logs.

## Data

- `well_18_ml_v1.csv`: 1,104 rows, one well, 5 zones. Missing values are coded as `-9999`.
- After removing rows with missing `RHOB` or `DT`: **1,044 rows** (738 net, 306 gross).

## Repository structure

```
├── data/     well_18_ml_v1.csv
├── python/   ML_Session.ipynb, ML_Session_corregido.py
├── knime/    ML_Pozo18_KNIME.knwf
└── README.md
```

## Key finding: data leakage

`net_gross` is defined exactly as `PHIE > 0.25` (100% of rows). Using `PHIE` as an input to classify net/gross gives ~97–100% accuracy that reflects the rule, not learning. Classification here is done **without `PHIE`**.

Other fixes to the original notebook: `dropna()` only on the columns used (the original lost 440 rows and 3 of 5 zones), fixed random seeds, depth-block validation, feature scaling for SVR/KNN, and logistic regression instead of thresholded linear regression.

## Results

**PHIE regression** (Python, 5-fold depth-block CV)

| Model | R² | MAE |
|---|---|---|
| Linear | 0.828 | 0.0207 |
| Random Forest | 0.820 | 0.0201 |
| XGBoost | 0.818 | 0.0204 |

**Net/gross classification** (no PHIE; baseline "always net" = 70.7%)

| Model | Python (depth blocks) | KNIME (stratified 75/25) |
|---|---|---|
| Logistic Regression | 0.889 | 0.912 |
| KNN (k=5, scaled) | 0.898 | _TBD_ |
| Random Forest | 0.905 | _TBD_ |

Python and KNIME figures are not directly comparable (different splits).

## How to run

**Python:** `pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter`, then `python python/ML_Session_corregido.py` from the folder containing the CSV.

**KNIME:** *File → Import KNIME Workflow…* → select the `.knwf`, point the **CSV Reader** to the local CSV, and run all with **Shift + F7**.

## Limitations

- Single well: generalization to other wells is untested.
- `PHIE` is derived from `RHOB` and `NPHI`, so predicting it from them is partly circular.
- KNIME has no native SVR node; Gradient Boosted Trees replaces XGBoost.

## License

_TBD. Confirm the CSV can be published._
