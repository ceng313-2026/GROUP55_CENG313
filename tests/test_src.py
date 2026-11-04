"""Smoke check for src on synthetic data. Run from the repo root: python tests/test_src.py"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import MARKET_COLS, TARGET, feature_columns, time_split, x_columns  # noqa: E402
from src.models import BASELINES  # noqa: E402
from src.utils import pearson, write_submission  # noqa: E402


def make_frame(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-03-01", periods=n, freq="min")
    cols = MARKET_COLS + ["X1", "X2", "X3"]
    df = pd.DataFrame(rng.normal(size=(n, len(cols))).astype(np.float32), index=idx, columns=cols)
    signal = 0.8 * df["X1"] - 0.5 * df["bid_qty"] + rng.normal(scale=0.5, size=n)
    df[TARGET] = signal.astype(np.float32)
    return df


def main():
    df = make_frame()
    assert x_columns(df) == ["X1", "X2", "X3"]
    assert TARGET not in feature_columns(df)

    fit, val = time_split(df)
    assert fit.index.max() < val.index.min(), "holdout must come after the fit period"
    assert val.index.min() - fit.index.max() > pd.Timedelta(hours=24), "embargo gap missing"
    assert len(val) == 1000

    assert pearson([1, 2, 3], [5, 5, 5]) == 0.0
    assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12

    feats = feature_columns(df)
    for name, make in BASELINES.items():
        model = make().fit(fit[feats], fit[TARGET])
        score = pearson(val[TARGET], model.predict(val[feats]))
        expected_min = 0.0 if name == "constant_mean" else 0.5
        assert score >= expected_min, f"{name}: Pearson {score:.3f} below {expected_min}"

    with tempfile.TemporaryDirectory() as d:
        sub = write_submission(val.index[:3], [0.0, 1.0, 2.0], os.path.join(d, "sub.csv"))
    assert list(sub.columns) == ["ID", "prediction"]
    print("ok")


if __name__ == "__main__":
    main()
