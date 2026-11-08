"""Baseline models for Assignment 1.

Each entry builds a fresh scikit-learn style estimator (fit / predict), so the
notebook trains and scores every baseline with the same loop.
"""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_array


class ChunkedStandardScaler(StandardScaler):
    """StandardScaler fitted on column chunks.

    The stock fit builds float64 temporaries the size of the whole matrix (about 12 GB for
    this data), which a free Colab runtime cannot hold. Column statistics are independent,
    so fitting chunk by chunk yields identical mean_, var_ and scale_.
    """

    def __init__(self, chunk_columns=100, copy=True):
        super().__init__(copy=copy)
        self.chunk_columns = chunk_columns

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X, dtype=[np.float64, np.float32])
        parts = [StandardScaler().fit(X[:, i : i + self.chunk_columns]) for i in range(0, X.shape[1], self.chunk_columns)]
        self.mean_ = np.concatenate([p.mean_ for p in parts])
        self.var_ = np.concatenate([p.var_ for p in parts])
        self.scale_ = np.concatenate([p.scale_ for p in parts])
        self.n_samples_seen_ = parts[0].n_samples_seen_
        self.n_features_in_ = X.shape[1]
        return self


def ridge():
    """Standardized features, ridge regression solved by Cholesky in float64.

    In float32 the Gram matrix of these highly collinear columns is not numerically positive
    definite, and scikit-learn then silently falls back to a full SVD that takes minutes and
    over 10 GB. The float64 cast is a private copy, so the scaler and ridge work on it in place.
    """
    return make_pipeline(
        FunctionTransformer(np.asarray, kw_args={"dtype": np.float64}),
        ChunkedStandardScaler(copy=False),
        Ridge(alpha=1.0, solver="cholesky", copy_X=False),
    )


BASELINES = {
    # Predicts the training mean. Pearson is undefined for a constant prediction,
    # so this is the "zero information" reference point.
    "constant_mean": lambda: DummyRegressor(strategy="mean"),
    "ridge": ridge,
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
