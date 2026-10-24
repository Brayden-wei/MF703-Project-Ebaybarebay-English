# CSV submission examples

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

Final comparison: one row each for the zero-return baseline, AR, Ridge, and LightGBM; columns are model,rmse,mae,rank.
