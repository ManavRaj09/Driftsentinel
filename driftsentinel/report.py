"""Self-contained HTML report (no external libraries, opens anywhere)."""
import html
from datetime import datetime

COL = {"OK": "#2e9e5b", "WARNING": "#e0a100", "CRITICAL": "#d64545"}


def _bars(result):
    rows = sorted(result.features, key=lambda f: -f.psi)
    mx = max([f.psi for f in rows] + [0.3])
    h = 34 * len(rows) + 20
    out = [f'<svg viewBox="0 0 700 {h}" width="100%">']
    for i, f in enumerate(rows):
        y = 10 + i * 34
        w = max(2, f.psi / mx * 430)
        out.append(f'<text x="0" y="{y+17}" font-size="13">{html.escape(f.name)}</text>'
                   f'<rect x="130" y="{y}" width="{w:.0f}" height="22" rx="4" fill="{COL[f.severity]}"/>'
                   f'<text x="{138+w:.0f}" y="{y+16}" font-size="12">PSI {f.psi:.3f}</text>')
    for t, c in ((0.10, "#e0a100"), (0.25, "#d64545")):
        x = 130 + t / mx * 430
        out.append(f'<line x1="{x:.0f}" x2="{x:.0f}" y1="0" y2="{h}" stroke="{c}" stroke-dasharray="4"/>')
    out.append("</svg>")
    return "".join(out)


def _timeline(tl):
    w, h = 700, 160
    n = len(tl)
    xs = [40 + i * (w - 60) / max(n - 1, 1) for i in range(n)]
    ys = [h - 20 - (v / 100) * (h - 40) for v in tl["health"]]
    pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="4" fill="{COL[s]}"/>'
                   for x, y, s in zip(xs, ys, tl["status"]))
    return (f'<svg viewBox="0 0 {w} {h}" width="100%"><polyline points="{pts}" fill="none" '
            f'stroke="#888" stroke-width="2"/>{dots}'
            f'<text x="0" y="14" font-size="11">100</text><text x="0" y="{h-20}" font-size="11">0</text></svg>')


def build_report(result, path, timeline=None, title="DriftSentinel Report"):
    rows = "".join(
        f"<tr><td>{html.escape(f.name)}</td><td>{f.kind}</td><td>{f.psi:.3f}</td>"
        f"<td>{f.p_value:.4f}</td><td style='color:{COL[f.severity]}'><b>{f.severity}</b></td>"
        f"<td>{html.escape(f.note)}</td></tr>" for f in sorted(result.features, key=lambda f: -f.psi))
    tl = ""
    if timeline is not None and len(timeline):
        tl = f"<h2>Health over time</h2>{_timeline(timeline)}"
    advice = "Retraining or investigation recommended." if result.status != "OK" else "No action needed."
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:860px;margin:30px auto;padding:0 16px;color:#222}}
.card{{border-radius:12px;padding:18px;color:#fff;background:{COL[result.status]}}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}</style></head>
<body><h1>{title}</h1><p>Generated {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="card"><h2 style="margin:0">Model input health: {result.health_score}/100 - {result.status}</h2>
<p style="margin:6px 0 0">{len(result.drifted())} of {len(result.features)} features drifted. {advice}</p></div>
<h2>Drift by feature (PSI)</h2>{_bars(result)}{tl}
<h2>Details</h2><table><tr><th>Feature</th><th>Type</th><th>PSI</th><th>p-value</th><th>Status</th><th>What changed</th></tr>{rows}</table>
<p style="font-size:12px;color:#666">PSI &lt; 0.10 stable, 0.10-0.25 moderate shift, &gt; 0.25 major shift.</p></body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
