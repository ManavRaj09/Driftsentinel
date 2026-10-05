"""Statistical drift metrics (numeric + categorical)."""
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

EPS = 1e-6


def psi_numeric(ref, cur, bins=10):
    """Population Stability Index using reference-quantile bins."""
    ref = np.asarray(ref, float)
    cur = np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        return 0.0 if np.allclose(np.mean(ref), np.mean(cur)) else 5.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur):
    ref, cur = list(ref), list(cur)
    cats = sorted(set(ref) | set(cur), key=str)
    r = np.array([ref.count(k) for k in cats], float) / len(ref)
    c = np.array([cur.count(k) for k in cats], float) / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_test(ref, cur):
    """Two-sample Kolmogorov-Smirnov -> (statistic, p_value)."""
    res = stats.ks_2samp(np.asarray(ref, float), np.asarray(cur, float))
    return float(res.statistic), float(res.pvalue)


def js_distance(ref, cur, bins=20):
    ref = np.asarray(ref, float)
    cur = np.asarray(cur, float)
    lo, hi = min(ref.min(), cur.min()), max(ref.max(), cur.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    r = np.histogram(ref, edges)[0] + EPS
    c = np.histogram(cur, edges)[0] + EPS
    return float(jensenshannon(r / r.sum(), c / c.sum(), base=2))


def chi2_test(ref, cur):
    cats = sorted(set(ref) | set(cur), key=str)
    table = np.array([[list(ref).count(k) for k in cats],
                      [list(cur).count(k) for k in cats]])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(table)
    return float(chi2), float(p)
