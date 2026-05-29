"""Criterion benchmark dashboard generator.

RW-5 — Reads ``target/criterion/*/estimates.json`` files emitted by the
Rust Criterion harness and renders a single-page Plotly HTML dashboard.

Usage
-----
    python benchmarks/criterion_to_report.py [--criterion-dir DIR]
                                              [--baseline FILE]
                                              [--output report.html]

Arguments
---------
--criterion-dir DIR
    Root of Criterion output.  Defaults to ``target/criterion``.

--baseline FILE
    Optional path to a previous run's ``criterion_summary.json``.
    Benchmarks that regressed by more than 10 % are highlighted in red.

--output FILE
    Output HTML path.  Defaults to ``criterion_report.html``.

Output
------
``criterion_report.html``
    Single-page Plotly dashboard with:
    - Bar chart of median wall-times per benchmark group.
    - Regression table if ``--baseline`` is provided.

``criterion_summary.json``
    Machine-readable summary in the same directory as the output HTML,
    suitable for use as a future ``--baseline``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

try:
    from autogen_dir import autogen_path
except ImportError:
    from benchmarks.autogen_dir import autogen_path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_estimates(criterion_dir: Path) -> list[dict[str, Any]]:
    """Walk criterion_dir and collect all estimates.json records."""
    records: list[dict[str, Any]] = []
    for estimates_path in sorted(criterion_dir.rglob("estimates.json")):
        try:
            data = json.loads(estimates_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Path structure: criterion_dir / group / bench_name / base / estimates.json
        # or:             criterion_dir / bench_name / estimates.json
        rel = estimates_path.relative_to(criterion_dir)
        parts = rel.parts  # e.g. ("simplify", "add_zero", "new", "estimates.json")

        group = parts[0] if len(parts) >= 3 else "ungrouped"
        bench = parts[1] if len(parts) >= 3 else parts[0]

        median_ns = data.get("mean", {}).get("point_estimate") or data.get(
            "median", {}
        ).get("point_estimate")
        if median_ns is None:
            continue

        records.append(
            {
                "group": group,
                "bench": bench,
                "median_ns": median_ns,
                "median_us": median_ns / 1_000.0,
            }
        )
    return records


def _load_baseline(path: Path) -> dict[str, float]:
    """Load a previous criterion_summary.json → {key: median_ns}."""
    try:
        data = json.loads(path.read_text())
        return {r["key"]: r["median_ns"] for r in data}
    except (json.JSONDecodeError, OSError, KeyError):
        return {}


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

def _regression_pct(new_ns: float, old_ns: float) -> float:
    """Return % change (positive = slower)."""
    if old_ns <= 0:
        return 0.0
    return (new_ns - old_ns) / old_ns * 100.0


# ---------------------------------------------------------------------------
# HTML / Plotly report
# ---------------------------------------------------------------------------

_PLOTLY_CDN = (
    "https://cdn.plot.ly/plotly-2.27.0.min.js"
)


def _build_html(
    records: list[dict[str, Any]],
    baseline: dict[str, float],
    *,
    threshold_pct: float = 10.0,
) -> str:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r["group"], []).append(r)

    # ------------------------------------------------------------------
    # Build one Plotly bar-chart trace per group
    # ------------------------------------------------------------------
    traces_json: list[str] = []
    for group, recs in sorted(groups.items()):
        recs_sorted = sorted(recs, key=lambda r: r["bench"])
        x = [r["bench"] for r in recs_sorted]
        y = [r["median_us"] for r in recs_sorted]
        colors: list[str] = []
        for r in recs_sorted:
            key = f"{group}/{r['bench']}"
            if key in baseline:
                pct = _regression_pct(r["median_ns"], baseline[key])
                colors.append("#e74c3c" if pct > threshold_pct else "#2ecc71")
            else:
                colors.append("#3498db")

        trace = {
            "type": "bar",
            "name": group,
            "x": x,
            "y": y,
            "marker": {"color": colors},
            "hovertemplate": "%{x}<br>%{y:.2f} µs<extra></extra>",
        }
        traces_json.append(json.dumps(trace))

    layout = json.dumps(
        {
            "title": "Criterion Benchmark Dashboard — Alkahest",
            "xaxis": {"title": "Benchmark"},
            "yaxis": {"title": "Median time (µs)", "type": "log"},
            "barmode": "group",
            "legend": {"orientation": "h", "y": -0.2},
            "margin": {"b": 160},
        }
    )

    # ------------------------------------------------------------------
    # Regression table
    # ------------------------------------------------------------------
    reg_rows = ""
    if baseline:
        rows: list[tuple[str, float, float, float]] = []
        for r in records:
            key = f"{r['group']}/{r['bench']}"
            if key in baseline:
                pct = _regression_pct(r["median_ns"], baseline[key])
                rows.append((key, baseline[key] / 1_000, r["median_us"], pct))
        rows.sort(key=lambda t: t[3], reverse=True)
        for key, old_us, new_us, pct in rows:
            colour = "#e74c3c" if pct > threshold_pct else ("#2ecc71" if pct < -threshold_pct else "#ecf0f1")
            reg_rows += (
                f"<tr style='background:{colour}'>"
                f"<td>{key}</td>"
                f"<td>{old_us:.2f}</td>"
                f"<td>{new_us:.2f}</td>"
                f"<td>{pct:+.1f}%</td>"
                f"</tr>\n"
            )

    regression_table = ""
    if reg_rows:
        regression_table = f"""
<h2>Regression vs baseline (threshold {threshold_pct:.0f}%)</h2>
<table border="1" cellpadding="4" style="border-collapse:collapse;width:100%">
  <thead><tr><th>Benchmark</th><th>Baseline (µs)</th><th>Current (µs)</th><th>Δ</th></tr></thead>
  <tbody>{reg_rows}</tbody>
</table>"""

    traces_list = "[" + ",\n".join(traces_json) + "]"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Alkahest — Criterion Dashboard</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>
    body {{ font-family: sans-serif; max-width: 1400px; margin: 0 auto; padding: 1rem; }}
    h1 {{ color: #2c3e50; }}
    table {{ font-size: 0.85rem; }}
    td, th {{ padding: 4px 8px; }}
  </style>
</head>
<body>
  <h1>Alkahest — Criterion Benchmark Dashboard</h1>
  <div id="chart" style="width:100%;height:520px"></div>
  <script>
    Plotly.newPlot('chart', {traces_list}, {layout}, {{responsive: true}});
  </script>
  {regression_table}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def _write_summary(records: list[dict[str, Any]], out_path: Path) -> None:
    summary = [
        {"key": f"{r['group']}/{r['bench']}", "median_ns": r["median_ns"]}
        for r in records
    ]
    summary_path = out_path.parent / "criterion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  summary → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Plotly dashboard from Criterion benchmark results."
    )
    parser.add_argument(
        "--criterion-dir",
        default="target/criterion",
        help="Root Criterion output directory (default: target/criterion)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to a previous criterion_summary.json for regression detection",
    )
    parser.add_argument(
        "--output",
        default=str(autogen_path("criterion_report.html")),
        help="Output HTML file (default: temp-alkahest/testing/autogen/criterion_report.html)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold in %% (default: 10.0)",
    )
    args = parser.parse_args(argv)

    criterion_dir = Path(args.criterion_dir)
    if not criterion_dir.exists():
        print(
            f"ERROR: Criterion directory not found: {criterion_dir}\n"
            f"Run 'cargo bench --all' first.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading benchmarks from {criterion_dir} …")
    records = _load_estimates(criterion_dir)
    if not records:
        print("No estimates.json files found. Run 'cargo bench --all' first.")
        return 1
    print(f"  loaded {len(records)} benchmark(s) across "
          f"{len({r['group'] for r in records})} group(s)")

    baseline: dict[str, float] = {}
    if args.baseline:
        baseline_path = Path(args.baseline)
        baseline = _load_baseline(baseline_path)
        print(f"  baseline: {len(baseline)} entries from {baseline_path}")

    html = _build_html(records, baseline, threshold_pct=args.threshold)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"  report  → {out_path}")

    _write_summary(records, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
