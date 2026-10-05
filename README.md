# DriftSentinel

**Catch ML data drift before it silently breaks your models.**

Models in production don't crash when the world changes - they just get quietly wrong.
Inflation shifts incomes, a new mobile app brings a new type of customer, interest rates move credit scores...
and a loan-approval model trained last year keeps making confident, bad decisions.

DriftSentinel compares your **training data (reference)** with **live data (current)**, tells you
**which features drifted, how badly, and when it started**, and produces a shareable HTML report.

## Features
- **PSI** (Population Stability Index) with industry thresholds: `<0.10` stable, `0.10-0.25` warning, `>0.25` critical
- **KS test**, **Jensen-Shannon distance** (numeric) and **Chi-square** (categorical)
- **Model Input Health Score** (0-100) for a one-glance status
- **Stream monitor**: sliding window finds *when* drift began
- **Plain-English explanations**: `income: mean 54,519 -> 74,086 (+35.9%)`
- **Self-contained HTML report** (no JS libraries, opens anywhere)
- **CI/CD gate**: `--fail-on CRITICAL` returns exit code 1 so a pipeline can block a bad deployment
- Built-in simulator with realistic bank-loan drift scenarios

## Quick start
```bash
git clone https://github.com/ManavRaj09/driftsentinel.git
cd driftsentinel
pip install -r requirements.txt

python -m driftsentinel demo                      # simulated scenario + HTML report
python -m driftsentinel compare train.csv live.csv --out reports/r.html --fail-on CRITICAL
python -m pytest -q                               # run tests
```

## Use as a library
```python
import pandas as pd
from driftsentinel import DriftDetector, StreamMonitor

ref = pd.read_csv("train.csv")
live = pd.read_csv("last_week.csv")

result = DriftDetector(ref).compare(live)
print(result.status, result.health_score)
print(result.to_frame())

timeline = StreamMonitor(ref, window=500, step=250).run(live)
```

## Why enterprises care
Banks, insurers, e-commerce and healthcare teams must monitor deployed models (regulators such as
RBI/SR 11-7-style model risk guidance expect it). DriftSentinel is a lightweight, dependency-light
starting point for that monitoring.

## Project structure
```
driftsentinel/
  metrics.py    PSI, KS, JS, chi-square
  detector.py   feature-by-feature comparison, severity, health score
  monitor.py    sliding-window drift timeline
  simulate.py   synthetic loan data + drift injection
  report.py     HTML report generator
  cli.py        command line interface
tests/          pytest tests
```

## Roadmap
- [ ] Concept drift (monitor prediction/label performance)
- [ ] Slack / email alerts
- [ ] Streamlit live dashboard
- [ ] Multivariate drift (PCA / domain classifier)

## License
MIT
