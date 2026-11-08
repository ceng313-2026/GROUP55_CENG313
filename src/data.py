"""Loading and splitting the DRW Crypto Market Prediction data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COMPETITION = "drw-crypto-market-prediction"
TARGET = "label"
MARKET_COLS = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
# Last 20% of the training period is held out. The label is a future price move,
# so a shuffled split would let the model see the future.
VALIDATION_FRACTION = 0.2


# Columns are read in groups and cast into one preallocated float32 array. Each file is a single
# parquet row group, so reading it whole decodes ~7 GB of float64 first, more than a free Colab
# runtime has. Reading ~100 columns at a time keeps the peak near the size of the final frame.
COLUMN_GROUP = 100


def _read_index(pf: pq.ParquetFile) -> pd.Index:
    spec = json.loads(pf.schema_arrow.metadata[b"pandas"])["index_columns"][0]
    if isinstance(spec, dict):  # a RangeIndex is stored as metadata only (test: ID 1..n)
        return pd.RangeIndex(spec["start"], spec["stop"], spec["step"], name=spec["name"])
    return pd.Index(pf.read(columns=[spec]).column(0).to_pandas().values)


def _load(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    names = [n for n in pf.schema_arrow.names if not n.startswith("__index_level_")]
    values = np.empty((pf.metadata.num_rows, len(names)), dtype=np.float32, order="F")
    for start in range(0, len(names), COLUMN_GROUP):
        group = names[start : start + COLUMN_GROUP]
        block = pf.read(columns=group)
        for offset, name in enumerate(group):
            values[:, start + offset] = block.column(name).to_numpy()
        del block
    return pd.DataFrame(values, index=_read_index(pf), columns=names)


def load_train(data_dir) -> pd.DataFrame:
    df = _load(Path(data_dir) / "train.parquet")
    return df if df.index.is_monotonic_increasing else df.sort_index()


def load_test(data_dir) -> pd.DataFrame:
    return _load(Path(data_dir) / "test.parquet")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET]


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Features as a C-ordered float32 array.

    pandas stores the frame column-major; scikit-learn and LightGBM both want row-major input
    and would otherwise each make their own full copy during fit.
    """
    positions = [df.columns.get_loc(c) for c in feature_columns(df)]
    out = np.empty((len(df), len(positions)), dtype=np.float32)
    np.take(df.to_numpy(), positions, axis=1, out=out)
    return out


def x_columns(df: pd.DataFrame) -> list[str]:
    """The anonymized proprietary features X1, X2, ..."""
    return [c for c in df.columns if c.startswith("X")]


# Rows dropped between the fit and holdout periods. The label is forward-looking and its
# autocorrelation only vanishes at a lag of about one day, so rows on both sides of the
# cut would otherwise share future information.
EMBARGO_ROWS = 24 * 60


def time_split(train: pd.DataFrame, fraction: float = VALIDATION_FRACTION, embargo: int = EMBARGO_ROWS):
    """Chronological split into (fit, holdout) with an embargo gap before the holdout."""
    cut = int(len(train) * (1 - fraction))
    return train.iloc[: cut - embargo], train.iloc[cut:]
