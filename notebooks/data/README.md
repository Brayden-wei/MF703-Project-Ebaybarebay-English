# Data and feature lead's documentation

This is a template. Record the actual methods and settings after completing the work. See the root README for the common target, dates, and evaluation schedule.

| Item | To complete |
|---|---|
| Name / GitHub username | To fill in |
| Raw file and version | raw-v1; record source and file verification details |
| Processed version | processed-v1; record the actual published version |
| Code entry point, execution order, Python/dependency versions | To fill in |
| Shared data access | Pending upload; share private drive links within the team |

## Cleaning and timestamps

Document the original Timestamp unit, whether it represents bar opening or closing time, and its conversion to UTC data-availability time. Record sorting, duplicate timestamps, gaps, nonpositive prices, anomalies, zero volume, and missing-value handling. Report row counts and time ranges before and after processing. Do not use future information to fill past values.

## Shared features and target

For every field, record its meaning, formula, window, first available time, and warm-up/missing-value rule. Use ln(P[t+15 minutes]/P[t]); do not apply shift(-15) across gaps. Fit learned transformations separately on each training fold, not on the entire dataset.

## Shared splits and evaluation schedule

Generate evaluation_index.csv with timestamp,y_true,split at the UTC fifteen-minute times and validation/test periods in the root README. Record sample counts, start/end times, and exclusion reasons for each split; freeze the same evaluation samples for all models. Training targets must end before the next evaluation period starts.

Coordinate any common-version or evaluation-date changes with all three model leads and publish a new version while retaining the old one. Model leads must not independently alter true values or evaluation samples.

## Required deliverables

- Runnable Notebook or code in this folder and this README.
- Cleaned data, shared features, and field documentation on the drive.
- A matching evaluation_index.csv.
- Data version, generation date, code commit, and change log.
