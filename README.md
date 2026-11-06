# GROUP55_CENG313

Course project for CENG313 Introduction to Data Science, built on the Kaggle competition
[DRW - Crypto Market Prediction](https://www.kaggle.com/competitions/drw-crypto-market-prediction).

## Team

| Name | Student ID | GitHub |
|---|---|---|
| Mehmet Efe ER | 24118080012 | [zreik-blanc](https://github.com/zreik-blanc) |

## Competition

DRW asks for a prediction of short-term crypto price movement from minute-level market data.
Each row is one minute. The target `label` is an anonymized measure of the future price move.
Submissions are scored by the **Pearson correlation** between predictions and the true labels.
The competition is closed, so our scores come from late submissions.

## Dataset

| File | Rows | Content |
|---|---|---|
| `train.parquet` | 525,886 | one row per minute, 2023-03-01 to 2024-02-29; timestamp index, 5 market features, `X1`..`X780`, `label` |
| `test.parquet` | 538,150 | `ID` index, same features, timestamps masked; `label` is a placeholder |
| `sample_submission.csv` | 538,150 | `ID`, `prediction` |

The five public market features are `bid_qty`, `ask_qty`, `buy_qty`, `sell_qty` and `volume`.
The 780 `X*` columns are proprietary DRW features with no documentation.

## Approach

### Assignment 1: EDA and baselines (`assignment1_eda_baseline.ipynb`)

- Data overview, memory footprint, missing and constant columns, time coverage.
- Target behaviour: distribution, tails, autocorrelation and regime changes over time.
- Market features: distributions, correlation with the target, order-book and trade-flow imbalance.
- Anonymized features: correlation with the target, redundancy between features, drift between train and test.
- Baselines evaluated on a **chronological** holdout (last 20% of the training period) with a one-day gap before it,
  never a shuffled split, because the target is a future price move and stays autocorrelated for about an hour.

| Baseline | Description |
|---|---|
| `constant_mean` | predicts the training mean (zero-information reference) |
| `ridge` | standardized features, ridge regression |
| `lightgbm` | gradient boosted trees on all features |

### Assignment 2: proposed solution (`assignment2_solution.ipynb`)

Due in the third week of December. It will build on the findings and baselines of Assignment 1.

## Results

| Model | Holdout Pearson |
|---|---|
| `constant_mean` | 0.0000 |
| `ridge` | 0.0958 |
| `lightgbm` | 0.0536 |

Holdout is the last 20% of the training period (2023-12-18 to 2024-02-29), with the day before it dropped from the fit set.

Main findings of the EDA (details and plots in the notebook):

- The data is clean: no missing values, no constant columns, no duplicate timestamps. The minute grid is 99.8% complete
  (273 gaps, 1,154 missing minutes, the longest 35 minutes).
- `label` is heavy-tailed (kurtosis 16) and strongly autocorrelated (0.98 at 1 minute, 0.61 at 30 minutes, about 0 at 1 day),
  so a shuffled split leaks the future. All validation is chronological.
- The five market features and the imbalance ratios derived from them have almost no linear relation to the target (|Pearson| < 0.02).
- The 780 `X` features are roughly standardized, individually weak (max |Pearson| 0.09) and highly redundant (113 pairs above 0.99).
- Train and test are trivially separable (adversarial validation AUC 1.00). The separation is concentrated in a small group
  of `X` features whose level shifts between the periods (`X772` alone reaches AUC 0.99, six features exceed 0.9), and test
  volume is about 35% lower.

## How to run

**Google Colab.** Open the notebook, run the first cell. It clones this repository, asks for your
`kaggle.json` (Kaggle > Settings > API > Create New Token) and downloads the competition data.
You must have accepted the competition rules on Kaggle.

**Locally.** Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

```bash
uv sync
uv run kaggle competitions download -c drw-crypto-market-prediction -p data && unzip -o data/*.zip -d data
uv run python tests/test_src.py
uv run jupyter notebook assignment1_eda_baseline.ipynb
```

Reusable code lives in `src/`: `data.py` (loading, column groups, chronological split), `eda.py` (minute-grid
coverage, feature redundancy, adversarial validation), `models.py` (baseline registry), `utils.py` (Pearson metric,
submission writer). `tests/test_src.py` is a smoke test
on synthetic data.

## Progress log

| Date | Work |
|---|---|
| 2026-11-04 | Repository structure, uv project, `src` modules (data loading, chronological split, baseline registry, metric), smoke test |
| 2026-11-05 | EDA notebook and `src/eda.py` helpers: data quality, target behaviour, market and anonymized features, train/test drift, time patterns |
| 2026-11-06 | Baselines added to the notebook, holdout evaluation, results written up |
