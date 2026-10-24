# Final comparison

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

`compare.py` checks dates and true values across the three prediction files and evaluates only test rows. Save `metrics.csv` (model,rmse,mae,rank) and `metrics.png` (one subplot per metric).

After comparison, record the test start/end dates, sample count, data version, input files, comparison date, and conclusions here. Save subsequent runs in a separate dated and numbered folder.
