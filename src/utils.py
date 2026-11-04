"""Metric and submission helpers."""
import numpy as np
import pandas as pd


def pearson(y_true, y_pred) -> float:
    """Competition metric. Returns 0.0 when either side is constant (undefined correlation)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.std() == 0 or y_pred.std() == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def write_submission(ids, predictions, path) -> pd.DataFrame:
    sub = pd.DataFrame({"ID": ids, "prediction": predictions})
    sub.to_csv(path, index=False)
    return sub


def memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024**2
