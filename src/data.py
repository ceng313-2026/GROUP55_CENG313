"""Loading and splitting the DRW Crypto Market Prediction data."""
from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION = "drw-crypto-market-prediction"
TARGET = "label"
MARKET_COLS = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
# Last 20% of the training period is held out. The label is a future price move,
# so a shuffled split would let the model see the future.
VALIDATION_FRACTION = 0.2


def _load(path: Path) -> pd.DataFrame:
    # float32 halves memory (train is ~3.8 GB as float64), which matters on Colab.
    return pd.read_parquet(path).astype(np.float32)


def load_train(data_dir) -> pd.DataFrame:
    return _load(Path(data_dir) / "train.parquet").sort_index()


def load_test(data_dir) -> pd.DataFrame:
    return _load(Path(data_dir) / "test.parquet")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET]


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
