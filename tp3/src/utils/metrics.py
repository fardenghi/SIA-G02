import pandas as pd


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
