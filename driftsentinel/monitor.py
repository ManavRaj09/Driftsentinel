"""Sliding-window monitor: pinpoints WHEN drift begins in a data stream."""
import pandas as pd
from .detector import DriftDetector


class StreamMonitor:
    def __init__(self, reference: pd.DataFrame, window: int = 500, step: int = 250):
        self.detector = DriftDetector(reference)
        self.window, self.step = window, step

    def run(self, stream: pd.DataFrame):
        """Return a DataFrame: one row per window with health score + status."""
        rows = []
        for start in range(0, len(stream) - self.window + 1, self.step):
            chunk = stream.iloc[start:start + self.window]
            res = self.detector.compare(chunk)
            worst = max(res.features, key=lambda f: f.psi)
            rows.append({"start": start, "end": start + self.window,
                         "health": res.health_score, "status": res.status,
                         "worst_feature": worst.name, "worst_psi": worst.psi})
        return pd.DataFrame(rows)

    @staticmethod
    def first_alert(timeline: pd.DataFrame, level="WARNING"):
        levels = ["WARNING", "CRITICAL"] if level == "WARNING" else ["CRITICAL"]
        bad = timeline[timeline["status"].isin(levels)]
        return None if bad.empty else int(bad.iloc[0]["start"])
