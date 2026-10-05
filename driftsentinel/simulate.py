"""Synthetic bank-loan data with injectable real-world drift scenarios."""
import numpy as np
import pandas as pd


def loan_data(n=3000, seed=0, inflation=0.0, new_channel=0.0, rate_shock=0.0):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.8, 0.5, n) * (1 + inflation)
    age = np.clip(rng.normal(38, 10, n), 18, 75)
    credit = np.clip(rng.normal(690 - 80 * rate_shock, 60, n), 300, 850)
    loan = rng.lognormal(9.5, 0.6, n) * (1 + inflation / 2)
    probs = [1 - new_channel - 0.3, 0.3, new_channel]
    channel = rng.choice(["branch", "web", "mobile_app"], n, p=probs)
    return pd.DataFrame({"income": income.round(0), "age": age.round(0),
                         "credit_score": credit.round(0), "loan_amount": loan.round(0),
                         "channel": channel})


def drifting_stream(n=6000, drift_at=3000, seed=1):
    """Healthy data, then at `drift_at` the economy shifts."""
    a = loan_data(drift_at, seed)
    b = loan_data(n - drift_at, seed + 1, inflation=0.35, new_channel=0.4, rate_shock=1.0)
    return pd.concat([a, b], ignore_index=True)
