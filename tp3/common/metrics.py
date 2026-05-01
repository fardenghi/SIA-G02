import numpy as np
import pandas as pd


def binary_confusion(y_true, y_pred):
    """Returns (tp, fp, fn, tn) for binary 0/1 arrays."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def precision_recall_f1(y_true, y_pred):
    tp, fp, fn, _ = binary_confusion(y_true, y_pred)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def threshold_sweep(scores, y_true, thresholds):
    """Vectorized precision/recall/F1 across many thresholds.

    scores: (N,) continuous predictions
    y_true: (N,) binary labels
    thresholds: (T,) values to sweep
    Returns three (T,) arrays: precision, recall, f1.
    """
    scores = np.asarray(scores).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()
    thresholds = np.asarray(thresholds).ravel()

    pred = (scores[None, :] >= thresholds[:, None]).astype(int)  # (T, N)
    pos = y_true == 1
    tp = (pred[:, pos] == 1).sum(axis=1)
    fp = (pred[:, ~pos] == 1).sum(axis=1)
    fn = (pred[:, pos] == 0).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return precision, recall, f1


def confusion_matrix(y_true, y_pred, n_classes):
    """Multiclass confusion matrix; rows=actual, cols=predicted."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


class MetricsTracker:
    """Per-run metrics recorder for MLP training."""

    EPOCH_COLUMNS = [
        "epoch", "loss_train", "acc_train",
        "loss_val", "acc_val", "time_s", "lr",
    ]

    def __init__(self, run_id, config_meta=None):
        self.run_id = run_id
        self.config_meta = config_meta or {}
        self._records = []

    def record(self, entry):
        row = {"run_id": self.run_id, **self.config_meta, **entry}
        self._records.append(row)

    def export_csv(self, path):
        df = pd.DataFrame(self._records)
        df.to_csv(path, index=False)
        return path

    def dataframe(self):
        return pd.DataFrame(self._records)

    @staticmethod
    def compare_runs(paths):
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
