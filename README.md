# MF703 BTC Prediction Project

One member prepares the shared data; three members independently train AR, Ridge, and LightGBM. Finally, `compare.py` evaluates all models consistently.

## Project structure

Each personal folder must contain **a README and runnable Notebook or Python code**. Code and result files below are examples of required deliverables and have not yet been created.

```text
MF703-Project-Ebaybarebay-English/
├── README.md
├── requirements.txt
├── config.json                      # Shared settings
├── data/
│   ├── raw/                         # Original CSV downloaded from the drive
│   └── processed/                   # Processed data
├── notebooks/
│   ├── README.md
│   ├── data/
│   │   ├── README.md                # Cleaning, features, splits, and instructions
│   │   └── prepare_data.ipynb       # Data preparation; .py is also acceptable
│   ├── ar/
│   │   ├── README.md                # Method, parameters, and data processing
│   │   └── ar.ipynb                 # Training and prediction; .py also acceptable
│   ├── ridge/
│   │   ├── README.md
│   │   └── ridge.ipynb              # Training and prediction; .py also acceptable
│   └── lightgbm/
│       ├── README.md
│       └── lightgbm.ipynb           # Training and prediction; .py also acceptable
├── results/
│   ├── README.md
│   ├── ar_metrics.csv              # RMSE and MAE for the entire test period
│   ├── ar_predictions.csv          # Individual true and predicted returns
│   ├── ridge_metrics.csv
│   ├── ridge_predictions.csv
│   ├── lightgbm_metrics.csv
│   └── lightgbm_predictions.csv
├── compare.py                       # To implement: evaluation, ranking, and plots
└── comparison/
    ├── README.md                    # Comparison date, data version, and conclusions
    ├── metrics.csv                  # Final four-row comparison table
    └── metrics.png                  # RMSE and MAE comparison charts
```

Add helper scripts within your own folder and document their execution order in your README.

## Shared rules

Predict the log return over the next fifteen minutes:

$$
y_t=\ln\left(\frac{P_{t+15}}{P_t}\right)
$$

| Item | Requirement |
|---|---|
| Data | Use the same raw and processed versions; do not overwrite shared data with personal changes |
| Price | P_t is the Close of a completed minute bar available at prediction time; the data lead verifies timestamp meaning |
| Time | Use UTC: Beijing time 2024-01-01 08:00 is recorded as 2024-01-01 00:00; do not subtract eight hours again if already UTC |
| Frequency | Predict the next fifteen minutes at minute 00, 15, 30, and 45 of each hour |
| Units | Decimal log returns, without multiplying by 100; match actual timestamps rather than shifting fifteen rows across gaps |
| Training | Use only information available at prediction time; fit scaling and other learned preprocessing on training data only |
| Parameters | Select on validation data; refit and freeze before testing, with no tuning or refitting during the test period |
| Evaluation samples | The data lead supplies common dates and true values; every model must cover all required rows |
| Individual choices | Choose features, windows, and parameters independently and document them; use random_state=42 where applicable |

## Data splits

| split | Training period | Evaluation period |
|---|---|---|
| val2021 | 2017–2020 | 2021 |
| val2022 | 2017–2021 | 2022 |
| val2023 | 2017–2022 | 2023 |
| test | 2017–2023 | From 2024 through the last valid target |

All years use UTC. Training targets must end before the next evaluation period begins. The data lead handles gaps and feature warm-up consistently and freezes the evaluation schedule.

## Model evaluation

**Rank models by RMSE over the entire test period; lower is better. MAE is supplementary.** The sum includes every test prediction, giving one score per model.

$$
\boxed{\displaystyle \mathrm{Score}(m)=\mathrm{RMSE}_m=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat y_i^{(m)}-y_i)^2}}
$$

$$
\mathrm{MAE}_m=\frac{1}{N}\sum_{i=1}^{N}\left|\hat y_i^{(m)}-y_i\right|
$$

N is the number of test predictions, y_i is the true log return, and the hatted value is model m's prediction. All models use the same test samples. The zero-return baseline always predicts 0.

| Model | Test RMSE ↓ | Test MAE ↓ | RMSE rank |
|---|---:|---:|---:|
| Zero-return baseline | Pending | Pending | Pending |
| AR | Pending | Pending | Pending |
| Ridge | Pending | Pending | Pending |
| LightGBM | Pending | Pending | Pending |

Each row summarizes one model over the entire test period. Ties may share a rank. Validation results are used to select parameters and are excluded from this final table.

## Member deliverables

| Role | Deliverables |
|---|---|
| Data lead | Code and README in `notebooks/data/`, shared data, and the common evaluation schedule |
| AR / Ridge / LightGBM | Your code, README, and the two CSV files below in `results/` |
| Integration lead, who may hold another role | `compare.py`, final model comparison table, and comparison charts |

Your README should explain data processing, features, parameters and their selection, and how to run the code.

### Ridge example: submit these two CSV files

**1. `ridge_metrics.csv`: one row summarizing the entire test period.**

| model | rmse | mae |
|---|---:|---:|
| Ridge | 0.0012 | 0.0008 |

These numbers illustrate the format only. Replace them with RMSE and MAE calculated over the entire test period.

**2. `ridge_predictions.csv`: individual predictions for verifying the scores.**

| timestamp | y_true | y_pred | split |
|---|---:|---:|---|
| 2024-01-01 00:00 | 0.0010 | 0.0008 | test |
| 2024-01-01 00:15 | -0.0005 | -0.0002 | test |
| 2024-01-01 00:30 | 0.0003 | 0.0004 | test |
| ... | ... | ... | ... |

These prediction values are also illustrative. One row per fifteen minutes; the columns mean **UTC prediction time, true log return, predicted log return, and evaluation split**. The data lead supplies the time, true value, and split. Add your predictions without deleting rows, and preserve numerical precision.

| Your model | Required files |
|---|---|
| AR | `ar_metrics.csv`, `ar_predictions.csv` |
| Ridge | `ridge_metrics.csv`, `ridge_predictions.csv` |
| LightGBM | `lightgbm_metrics.csv`, `lightgbm_predictions.csv` |

`compare.py` uses all rows marked `split=test` to verify the scores and generate the four-row comparison table and charts. Keep validation results for parameter selection. Share oversized prediction files through the team drive.

## Sharing data and code

- **Data:** Download the original CSV to `data/raw/` and processed shared data to `data/processed/`. [Team data drive](https://drive.google.com/drive/folders/1oUc_AD97JhNFmT7GM21hw91WWX0XLKNo?usp=drive_link).
- **Download code:** GitHub → Code → Download ZIP. Install `requirements.txt` using Python 3.11; add other dependencies as needed.
- **Upload code:** Accept the Collaborator invitation → open your folder → Add file → Upload files → Commit changes.
