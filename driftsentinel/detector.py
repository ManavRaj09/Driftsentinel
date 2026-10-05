"""Compare a reference dataset with a current dataset, feature by feature."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from . import metrics as m

WARN_PSI, CRIT_PSI = 0.10, 0.25  # industry-standard PSI thresholds


@dataclass
class FeatureDrift:
    name: str
    kind: str
    psi: float
    p_value: float
    js: Optional[float]
    severity: str
    note: str


@dataclass
class DriftResult:
    features: list = field(default_factory=list)

    @property
    def health_score(self):
        """0-100. 100 = no drift. Each drifted feature subtracts weighted PSI."""
        if not self.features:
            return 100.0
        penalty = np.mean([min(f.psi / CRIT_PSI, 1.0) for f in self.features])
        return round(100 * (1 - penalty), 1)

    @property
    def status(self):
        sev = {f.severity for f in self.features}
        return "CRITICAL" if "CRITICAL" in sev else "WARNING" if "WARNING" in sev else "OK"

    def drifted(self):
        return [f for f in self.features if f.severity != "OK"]

    def to_frame(self):
        return pd.DataFrame([f.__dict__ for f in self.features]).sort_values("psi", ascending=False)


def _severity(psi):
    return "CRITICAL" if psi >= CRIT_PSI else "WARNING" if psi >= WARN_PSI else "OK"


def _explain(name, kind, ref, cur):
    if kind == "numeric":
        r, c = float(np.nanmean(ref)), float(np.nanmean(cur))
        pct = (c - r) / (abs(r) + 1e-9) * 100
        return f"mean {r:.2f} -> {c:.2f} ({pct:+.1f}%)"
    top_r = pd.Series(ref).value_counts(normalize=True)
    top_c = pd.Series(cur).value_counts(normalize=True)
    idx = top_r.index.union(top_c.index)
    diff = top_c.reindex(idx, fill_value=0) - top_r.reindex(idx, fill_value=0)
    k = diff.abs().idxmax()
    return f"'{k}' share {top_r.get(k, 0)*100:.0f}% -> {top_c.get(k, 0)*100:.0f}%"


class DriftDetector:
    def __init__(self, reference: pd.DataFrame, bins: int = 10):
        self.ref = reference.reset_index(drop=True)
        self.bins = bins

    def compare(self, current: pd.DataFrame) -> DriftResult:
        out = DriftResult()
        for col in self.ref.columns:
            if col not in current.columns:
                continue
            r, c = self.ref[col].dropna(), current[col].dropna()
            if len(r) == 0 or len(c) == 0:
                continue
            if pd.api.types.is_numeric_dtype(r):
                psi = m.psi_numeric(r, c, self.bins)
                _, p = m.ks_test(r, c)
                js, kind = m.js_distance(r, c), "numeric"
            else:
                psi = m.psi_categorical(r, c)
                _, p = m.chi2_test(r, c)
                js, kind = None, "categorical"
            out.features.append(FeatureDrift(
                col, kind, round(psi, 4), round(p, 6),
                None if js is None else round(js, 4),
                _severity(psi), _explain(col, kind, r, c)))
        return out
