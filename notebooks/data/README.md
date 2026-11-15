# Data and feature lead's documentation

See the root README for the common target, dates, and evaluation rules.

| Item | Value |
|---|---|
| Name / GitHub username | Heng Zhang / *to fill in* |
| Raw file and version | `raw-v1`: `btcusd_1-min_data.part-000…007.parquet` (or the single `btcusd_1-min_data.parquet`), 7,206,877 rows, columns `Timestamp, Open, High, Low, Close, Volume` |
| Processed version | `processed-v1`, generated 2026-11-15 |
| Code entry point | `notebooks/data/prepare_data.ipynb` (same code as `prepare_data.py`); one step, about one minute |
| Environment | Python 3.13, pandas 3.0, numpy 2.5, pyarrow 26, matplotlib 3.11. Nothing version-specific is used, so the project's Python 3.11 environment also works |
| Shared data access | Team drive → `processed-v1/` (*paste the folder link here after uploading*) |

## How to run

```bash
pip install -r requirements.txt
python notebooks/data/prepare_data.py
```

Raw files must be in `data/raw/`. Outputs are written to `data/processed/processed-v1/`,
plus a copy of `evaluation_index.csv` in `data/processed/` (the path in `config.json`).
The script stops with an error if any of its leakage or consistency checks fails.

| Output | Rows | Purpose |
|---|---:|---|
| `features_15min.parquet` | 305,009 | **The file model owners use.** One row per forecast time: shared features, `y_true`, `split` |
| `evaluation_index.csv` | 164,753 | Frozen evaluation schedule `timestamp, y_true, split` (validation and test rows only) |
| `btc_1min_clean.parquet` | 4,622,398 | Cleaned one-minute bars, for anyone building extra features |
| `qc_summary.json` | – | Quality-check counts, exclusions, baseline scores |

## Cleaning and timestamps

**Timestamp meaning.** `Timestamp` is Unix seconds in UTC and marks the **open** of each
one-minute bar (source: the Bitstamp BTC/USD one-minute dataset; this is the documented
convention of that dataset and was not independently verified against exchange data).
A bar stamped 12:14 is complete, and its `Close` known, at 12:15:00. All shared files are
therefore indexed by **availability time** = bar open + 1 minute, and `P_t` is the `Close`
of the bar stamped `t − 1 min`. No time-zone shift is applied; the data is already UTC.

**Raw file findings.**

| Check | Result |
|---|---|
| Time range | 2012-01-01 10:01 → 2025-09-14 23:57 UTC |
| Sorted / duplicate timestamps / off-minute timestamps | sorted / 0 / 0 |
| Missing values, non-positive prices, negative volume, High/Low inconsistent with Open/Close | 0 / 0 / 0 / 0 |
| Gaps | One: 1,160 missing minutes, 2025-03-15 00:01 → 19:20 |
| Zero-volume bars | 1,310,260 overall (mostly 2012–2016); 163,423 in the cleaned range |
| Extreme moves after 2017 | Kept. The largest (2017-03-10, 2019-05-17) are real high-volume events |

**Rules applied.**

1. Sort, drop duplicate timestamps and invalid rows (none were found).
2. Keep bars from 2016-12-01 onward; December 2016 only warms up the rolling features.
3. Re-index to a complete one-minute grid. Missing minutes stay NaN (`is_missing = 1`) and are never filled.
4. Zero-volume bars are kept and flagged (`is_zero_volume = 1`). The source repeats the last traded price when nothing trades, which is what was observable at the time.
5. No winsorising, scaling or outlier removal. Model owners apply their own transformations, fitted on training data only.

Rows before → after: 7,206,877 raw bars → 4,622,398 grid minutes (2016-12-01 00:00 → 2025-09-14 23:57, including the 1,160 missing).

## Target

`y_true = ln(P[t + 15 min] / P[t])`, in decimal log returns. It is computed on the complete
minute grid, so the step is fifteen minutes of elapsed time; if either price falls in the gap
the target is missing and the row is excluded. `price_t` and `price_t_plus_15` are included so
a predicted return converts to a price forecast: `price_t * exp(y_pred)`.

## Shared features

All features at time `t` use only bars completed at or before `t`. Rolling windows need a
full window, so they are missing during warm-up and after the gap. Windows are in minutes.
`r1` is the one-minute log return.

| Column(s) | Meaning and formula |
|---|---|
| `ret_1, ret_5, ret_15, ret_30, ret_60, ret_240, ret_1440` | Past log return `ln(P[t] / P[t−k])` |
| `ret_15_lag1 … ret_15_lag12` | Earlier non-overlapping fifteen-minute returns `ln(P[t−15j] / P[t−15(j+1)])`; with `ret_15` these are the AR lags 1–13 of the target series |
| `ma_gap_15, _60, _240, _1440` | `ln(P[t] / mean(P over last w minutes))` |
| `vol_15, _60, _240, _1440` | Standard deviation of `r1` over the last `w` minutes |
| `range_15`, `range_60` | `ln(max High / min Low)` over the last 15 / 60 minutes |
| `body_15` | `ln(Close / Open)` of the fifteen-minute candle ending at `t` |
| `upper_shadow_15`, `lower_shadow_15` | `ln(max High / max(Open, Close))`, `ln(min(Open, Close) / min Low)` of that candle |
| `log_volume_15` | `ln(1 + V15)`, `V15` = BTC volume over the last 15 minutes |
| `volume_chg_15` | `ln(1 + V15[t]) − ln(1 + V15[t−15])` |
| `volume_rel_240`, `volume_rel_1440` | `ln(1 + V15)` minus `ln(1 +` average fifteen-minute volume over the last 4 h / 24 h`)` |
| `zero_volume_frac_15` | Share of the last 15 one-minute bars with zero volume |
| `hour`, `day_of_week` | UTC hour 0–23; Monday = 0 |

The longest window is 1,440 minutes, so the first usable time after any break is 24 hours later.
`features_15min.parquet` contains no missing values.

## Splits and evaluation schedule

Forecast times are UTC minutes 00/15/30/45. A time is kept only when the target **and** every
shared feature are available, so all models can cover every required row.

| `split` | Period (UTC) | Rows | Excluded | Zero-return baseline RMSE | Baseline MAE |
|---|---|---:|---:|---:|---:|
| `train` | 2017-01-01 00:00 → 2020-12-31 23:45 | 140,256 | 0 | 0.004760 | 0.002712 |
| `val2021` | 2021-01-01 00:00 → 2021-12-31 23:45 | 35,040 | 0 | 0.004993 | 0.003218 |
| `val2022` | 2022-01-01 00:00 → 2022-12-31 23:45 | 35,040 | 0 | 0.003434 | 0.002117 |
| `val2023` | 2023-01-01 00:00 → 2023-12-31 23:45 | 35,040 | 0 | 0.002237 | 0.001340 |
| `test` | 2024-01-01 00:00 → 2025-09-14 23:30 | 59,633 | 175 | 0.002650 | 0.001726 |

The 175 excluded test times: 78 whose target or price falls in the 2025-03-15 gap, 96 in the
24 hours after it (feature warm-up), and the final time 2025-09-14 23:45 (no price fifteen
minutes later).

How the labels map to the folds in the root README:

| Fold | Train on `split` in | Evaluate on |
|---|---|---|
| val2021 | `train` | `val2021` |
| val2022 | `train`, `val2021` | `val2022` |
| val2023 | `train`, `val2021`, `val2022` | `val2023` |
| test | `train`, `val2021`, `val2022`, `val2023` | `test` |

The last training row of each fold (23:45 on 31 December) has a target ending exactly at
00:00, the first evaluation time, so no training target extends into an evaluation period.

## For model owners

```python
import pandas as pd

df = pd.read_parquet("data/processed/processed-v1/features_15min.parquet")
META = ["timestamp", "price_t", "price_t_plus_15", "y_true", "split"]
FEATURES = [c for c in df.columns if c not in META]      # 39 columns

train = df[df["split"].isin(["train", "val2021", "val2022", "val2023"])]   # final fit
test = df[df["split"] == "test"]
# ... fit on train[FEATURES], train["y_true"]; predict y_pred for test ...

out = test[["timestamp", "y_true", "split"]].assign(y_pred=y_pred)
out = out[["timestamp", "y_true", "y_pred", "split"]]
out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
out.to_csv("results/ridge_predictions.csv", index=False, float_format="%.12g")
```

- Never use `price_t_plus_15` or `y_true` as inputs.
- `test` must have exactly 59,633 rows matching `evaluation_index.csv`; do not drop or reorder rows, and do not change `y_true`.
- Extra features are welcome: build them from `btc_1min_clean.parquet` (index by `available_time`, use only rows at or before the forecast time) and join on `timestamp`.
- Your score must beat the baseline numbers above to add value.

## Figures

`figures/price_by_split.png`, `figures/target_distribution.png`, `figures/zero_volume_share.png` — ready for the report's data section.

## Change log

| Version | Date | Raw version | Change |
|---|---|---|---|
| processed-v1 | 2026-11-15 | raw-v1 | First release. Code commit: *fill in after uploading* |

Any change to the evaluation schedule or shared features will be released as `processed-v2`
after agreeing it with the three model leads; `processed-v1` stays available.
