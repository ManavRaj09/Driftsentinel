import argparse
import os
import sys
import pandas as pd
from .detector import DriftDetector
from .monitor import StreamMonitor
from .report import build_report
from .simulate import loan_data, drifting_stream


def main(argv=None):
    p = argparse.ArgumentParser(prog="driftsentinel", description="ML data drift monitor")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="compare two CSV files")
    c.add_argument("reference")
    c.add_argument("current")
    c.add_argument("--out", default="reports/drift_report.html")
    c.add_argument("--fail-on", choices=["WARNING", "CRITICAL"], default=None,
                   help="exit code 1 if status reaches this level (for CI/CD)")
    d = sub.add_parser("demo", help="run a built-in simulated drift scenario")
    d.add_argument("--out", default="reports/demo_report.html")
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if a.cmd == "demo":
        ref = loan_data(3000, seed=0)
        stream = drifting_stream(6000, drift_at=3000)
        mon = StreamMonitor(ref, window=500, step=250)
        tl = mon.run(stream)
        res = DriftDetector(ref).compare(stream.iloc[3500:])
        build_report(res, a.out, tl, "DriftSentinel Demo: Loan Model")
        first = mon.first_alert(tl)
        print(f"Drift was injected at row 3000. First alert at window starting row {first}.")
        print(res.to_frame()[["name", "psi", "severity", "note"]].to_string(index=False))
        print(f"\nReport: {a.out}")
        return 0

    res = DriftDetector(pd.read_csv(a.reference)).compare(pd.read_csv(a.current))
    build_report(res, a.out)
    print(f"{res.status} | health {res.health_score}/100 | report: {a.out}")
    order = ["OK", "WARNING", "CRITICAL"]
    if a.fail_on and order.index(res.status) >= order.index(a.fail_on):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
