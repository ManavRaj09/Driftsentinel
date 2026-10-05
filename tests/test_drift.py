import numpy as np
from driftsentinel import DriftDetector, StreamMonitor
from driftsentinel.metrics import psi_numeric, psi_categorical
from driftsentinel.simulate import loan_data, drifting_stream


def test_no_drift_is_ok():
    res = DriftDetector(loan_data(3000, 0)).compare(loan_data(3000, 5))
    assert res.status == "OK" and res.health_score > 90


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)) > 0.25


def test_categorical_psi():
    assert psi_categorical(["a"] * 90 + ["b"] * 10, ["a"] * 40 + ["b"] * 60) > 0.25


def test_detector_flags_injected_drift():
    ref = loan_data(3000, 0)
    cur = loan_data(3000, 1, inflation=0.4, new_channel=0.4, rate_shock=1.0)
    res = DriftDetector(ref).compare(cur)
    names = {f.name for f in res.drifted()}
    assert {"income", "credit_score", "channel"} <= names


def test_stream_monitor_finds_start():
    ref = loan_data(3000, 0)
    mon = StreamMonitor(ref, 500, 250)
    tl = mon.run(drifting_stream(6000, 3000))
    first = mon.first_alert(tl)
    assert first is not None and 2500 <= first <= 3250
