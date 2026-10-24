# AR lead's documentation

This is a template. "To fill in" does not indicate a completed experiment. See the root README for the common target, time splits, result format, and RMSE/MAE rules.

| Item | Owner to complete |
|---|---|
| Name / GitHub username | To fill in |
| Raw and processed data versions | To fill in, e.g. raw-v1 and processed-v1 |
| Notebook / code entry point | To fill in; create the file yourself |
| Python and dependency versions | To fill in; add new dependencies to requirements.txt |
| Execution order and commands | To fill in, including paths relative to the project root |
| Experiment ID run_id | To fill in |

## Data and features

- Shared fields and features used: to fill in.
- Added features, formulas, lags, and rolling windows: to fill in.
- Missing values, anomalies, and zero-volume handling: to fill in. Do not overwrite shared data or drop required evaluation rows.
- Scaling or other transformations: specify the training interval used for fitting. Convert predictions back to unscaled log returns.
- How future information is excluded: to fill in.

## Final parameters and selection

Lag order, intercept, fitting window, and how fifteen-minute predictions are produced.

| Parameter | Final value | Selection rationale / validation range |
|---|---|---|
| To fill in | To fill in | To fill in |
| random_state, if applicable | 42 | Shared convention |

Document the three validation folds, early stopping if used, and which validation records determine parameters. Never select parameters using test labels. Refit and freeze before testing; explain any sequential updates to historical inputs.

## Outputs and findings

- One-row summary: results/ar_metrics.csv; columns model,rmse,mae; cover the entire common test period and set model to AR.
- Calculation file: results/ar_predictions.csv; columns timestamp,y_true,y_pred,split. Preserve all required predictions for compare.py to verify and recompute; this is not the final presentation table.
- compare.py produces the final table: one row per model, with test RMSE, test MAE, and RMSE rank.
- run_id, generation date, and code commit: to fill in.
- Feature or method extensions: to fill in.
- Main findings and limitations: to fill in. Official RMSE, MAE, and comparison charts are calculated consistently by compare.py.
