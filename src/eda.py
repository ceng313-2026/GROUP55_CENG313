"""Exploratory analysis helpers used by the notebooks."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def minute_grid(index: pd.DatetimeIndex):
    """Coverage of the one-minute grid: a summary Series and the count of each gap size."""
    steps = pd.Series(index).diff().dropna()
    gaps = steps[steps > pd.Timedelta(minutes=1)]
    expected = int((index.max() - index.min()) / pd.Timedelta(minutes=1)) + 1
    summary = pd.Series({
        "expected_minutes": expected,
        "present": len(index),
        "missing": expected - len(index),
        "missing_share": (expected - len(index)) / expected,
        "gaps": len(gaps),
        "longest_gap": gaps.max(),
    })
    return summary, gaps.value_counts()


def redundant_pairs(df: pd.DataFrame, thresholds=(0.95, 0.99), n_sample=50_000, random_state=0) -> pd.Series:
    """Number of column pairs whose |correlation| exceeds each threshold, estimated on a row sample."""
    sample = df.sample(min(n_sample, len(df)), random_state=random_state)
    corr = np.nan_to_num(np.corrcoef(sample.values.T.astype(np.float64)))
    upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
    return pd.Series({f"|corr| > {t}": int((upper > t).sum()) for t in thresholds})


def adversarial_validation(train: pd.DataFrame, test: pd.DataFrame, features, n_sample=100_000, random_state=0):
    """How well can a classifier tell train rows from test rows?

    Returns the classifier's holdout AUC, its LightGBM feature importances, and the AUC
    of each raw feature used alone as a score. AUC near 0.5 means the sets look alike.
    """
    rows = pd.concat([
        train[features].sample(n_sample, random_state=random_state),
        test[features].sample(n_sample, random_state=random_state),
    ])
    is_test = np.r_[np.zeros(n_sample), np.ones(n_sample)]
    x_fit, x_val, y_fit, y_val = train_test_split(rows, is_test, test_size=0.3, random_state=random_state)
    clf = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5,
        random_state=random_state, verbose=-1,
    ).fit(x_fit, y_fit)
    auc = roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])
    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    single = pd.Series({c: roc_auc_score(is_test, rows[c]) for c in features})
    single = single.where(single >= 0.5, 1 - single).sort_values(ascending=False)
    return auc, importances, single
