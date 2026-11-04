"""Baseline models for Assignment 1.

Each entry builds a fresh scikit-learn style estimator (fit / predict), so the
notebook trains and scores every baseline with the same loop.
"""
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASELINES = {
    # Predicts the training mean. Pearson is undefined for a constant prediction,
    # so this is the "zero information" reference point.
    "constant_mean": lambda: DummyRegressor(strategy="mean"),
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        random_state=0,
        verbose=-1,
    ),
}
