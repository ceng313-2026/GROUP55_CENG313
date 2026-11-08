"""Smoke check for src on synthetic data. Run from the repo root: python tests/test_src.py"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (  # noqa: E402
    MARKET_COLS, TARGET, feature_columns, feature_matrix, load_test, load_train, time_split, x_columns,
)
from src.eda import adversarial_validation, minute_grid, redundant_pairs  # noqa: E402
from src.models import BASELINES, ChunkedStandardScaler  # noqa: E402
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

    matrix = feature_matrix(df)
    assert matrix.flags.c_contiguous and matrix.dtype == np.float32
    assert np.array_equal(matrix, df[feature_columns(df)].to_numpy())

    fit, val = time_split(df)
    assert fit.index.max() < val.index.min(), "holdout must come after the fit period"
    assert val.index.min() - fit.index.max() > pd.Timedelta(hours=24), "embargo gap missing"
    assert len(val) == 1000

    assert pearson([1, 2, 3], [5, 5, 5]) == 0.0
    assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12

    from sklearn.preprocessing import StandardScaler

    stock, chunked = StandardScaler().fit(matrix), ChunkedStandardScaler(chunk_columns=3).fit(matrix)
    for attr in ("mean_", "var_", "scale_"):
        assert np.array_equal(getattr(stock, attr), getattr(chunked, attr)), attr
    assert np.array_equal(stock.transform(matrix), chunked.transform(matrix))

    feats = feature_columns(df)
    for name, make in BASELINES.items():
        model = make().fit(fit[feats], fit[TARGET])
        score = pearson(val[TARGET], model.predict(val[feats]))
        expected_min = 0.0 if name == "constant_mean" else 0.5
        assert score >= expected_min, f"{name}: Pearson {score:.3f} below {expected_min}"

    idx = pd.date_range("2023-03-01", periods=100, freq="min").delete([10, 11, 50])
    coverage, sizes = minute_grid(idx)
    assert coverage["missing"] == 3 and coverage["gaps"] == 2
    assert coverage["longest_gap"] == pd.Timedelta(minutes=3) and sizes[pd.Timedelta(minutes=2)] == 1

    pairs = redundant_pairs(df.assign(X1_copy=df["X1"] * 2), ["X1", "X2", "X3", "X1_copy"])
    assert pairs["|corr| > 0.99"] == 1, pairs.to_dict()

    shifted = df.copy()
    shifted["X2"] = shifted["X2"] + 5
    auc, importances, single = adversarial_validation(df, shifted, feats, n_sample=500)
    assert auc > 0.9 and single.index[0] == "X2"

    with tempfile.TemporaryDirectory() as d:
        sub = write_submission(val.index[:3], [0.0, 1.0, 2.0], os.path.join(d, "sub.csv"))
        assert list(sub.columns) == ["ID", "prediction"]

        df.astype("float64").to_parquet(os.path.join(d, "train.parquet"))
        test_df = df.astype("float64").reset_index(drop=True)
        test_df[TARGET] = 0
        test_df.index = pd.RangeIndex(1, len(df) + 1, name="ID")
        test_df.to_parquet(os.path.join(d, "test.parquet"))

        train_back, test_back = load_train(d), load_test(d)
        assert train_back.equals(df) and train_back.index.equals(df.index)
        assert (test_back.dtypes == "float32").all() and test_back.index.equals(test_df.index)
        assert list(test_back.columns) == list(df.columns) and (test_back[TARGET] == 0).all()
    print("ok")


if __name__ == "__main__":
    main()
