#!/usr/bin/env python3
"""Generate (or drift-check) the certificate coverage ledger.

Runs ``tests/certificate_corpus.py`` — which drives the strict Lean corpus and
the textbook gate with the derivation entry points instrumented — and tabulates
what the Lean emitter *actually did*, per operation and shape class. Writes two
artifacts, both checked into the repo:

* ``python/alkahest/certificate_ledger.json`` — queryable from Python via
  :func:`alkahest.certificate_coverage`, and the source of truth for
  :func:`alkahest.certifiable`.
* ``docs/mdbook/src/certificate-coverage.md`` — the same table, for humans.

Usage::

    python scripts/gen_certificate_ledger.py            # regenerate
    python scripts/gen_certificate_ledger.py --check    # CI: fail on drift

``--check`` regenerates in memory and compares against the checked-in files. It
fails if they differ, so a change to the emitter that widens or narrows the
certifiable surface cannot land without the ledger — and the capability bits
derived from it — being updated in the same commit.

For that comparison to mean anything the output must be byte-reproducible, and
it is: rows are sorted, sets are sorted before serialising, and example
expressions are rendered by ``canonical_expression`` rather than ``str()``,
whose ordering of commutative operands varies between processes with the
kernel's randomly-seeded hasher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

JSON_PATH = os.path.join(REPO, "python", "alkahest", "certificate_ledger.json")
MARKDOWN_PATH = os.path.join(REPO, "docs", "mdbook", "src", "certificate-coverage.md")

#: How many example expressions to keep per row. Enough to make a row legible
#: without turning the artifact into a transcript of the corpus.
MAX_EXAMPLES = 3


def _verdict(counts: dict[str, int]) -> str:
    certified = counts.get("certified", 0)
    withheld = counts.get("withheld", 0)
    empty = counts.get("no_derivation", 0)
    if certified and not withheld and not empty:
        return "certified"
    if certified and not withheld:
        return "conditional"
    if certified:
        return "partial"
    return "withheld"


def build() -> dict:
    """Run the corpus and return the ledger document."""
    import certificate_corpus
    from alkahest._certificates import SCHEMA_VERSION

    observations = certificate_corpus.collect()

    grouped: dict[str, dict] = {}
    for observation in observations:
        row = grouped.setdefault(
            observation.shape,
            {
                "operation": observation.operation,
                "shape": observation.shape,
                "features": observation.features,
                "counts": {},
                "examples": [],
                "blocking_rules": set(),
            },
        )
        row["counts"][observation.outcome] = row["counts"].get(observation.outcome, 0) + 1
        if observation.expression not in row["examples"]:
            row["examples"].append(observation.expression)
        row["blocking_rules"].update(observation.blocking_rules)

    rows = []
    for shape in sorted(grouped):
        row = grouped[shape]
        counts = row["counts"]
        rows.append(
            {
                "operation": row["operation"],
                "shape": shape,
                "features": row["features"],
                "verdict": _verdict(counts),
                "observations": {
                    "certified": counts.get("certified", 0),
                    "withheld": counts.get("withheld", 0),
                    "no_derivation": counts.get("no_derivation", 0),
                },
                "blocking_rules": sorted(row["blocking_rules"]),
                "examples": sorted(row["examples"])[:MAX_EXAMPLES],
            }
        )
    rows.sort(key=lambda r: (r["operation"], r["shape"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/gen_certificate_ledger.py",
        "corpus": {
            "sources": [
                "tests/lean_corpus.py",
                "tests/lean_tendsto_corpus.py",
                "tests/lean_gosper_corpus.py",
                "tests/textbook_gate/",
            ],
            "observations": len(observations),
            "shape_classes": len(rows),
        },
        "rows": rows,
    }


VERDICT_LABEL = {
    "certified": "✅ certified",
    "conditional": "🟡 conditional",
    "partial": "🟠 partial",
    "withheld": "⛔ withheld",
}


def render_markdown(ledger: dict) -> str:
    """Render the ledger as the human-readable coverage page."""
    rows = ledger["rows"]
    totals: dict[str, int] = {}
    for row in rows:
        totals[row["verdict"]] = totals.get(row["verdict"], 0) + 1

    out: list[str] = []
    out.append("# Certificate coverage")
    out.append("")
    out.append(
        "<!-- GENERATED FILE — do not edit by hand.\n"
        "     Regenerate with `python scripts/gen_certificate_ledger.py`.\n"
        "     CI fails if this file drifts from a regeneration. -->"
    )
    out.append("")
    out.append(
        "`DerivedResult.certificate` returns Lean 4 source only where the emitter can "
        "prove the statement without `sorry`. This page is the map of that boundary. It "
        "is **generated by running a corpus and recording what actually emitted** — never "
        "hand-maintained — from `tests/lean_corpus.py` (the strict, CI-typechecked corpus) "
        "and `tests/textbook_gate/` (first-course calculus and algebra). "
        "`tests/lean_tendsto_corpus.py` contributes the recognised `x → +∞` "
        "`Filter.Tendsto` fragment. `tests/lean_gosper_corpus.py` contributes "
        "Gosper `Finset.sum` telescopes and the `∏ k = n!` product identity."
    )
    out.append("")
    out.append(
        f"Corpus: **{ledger['corpus']['observations']} observations** over "
        f"**{len(rows)} shape classes** — "
        + ", ".join(
            f"{totals.get(verdict, 0)} {verdict}"
            for verdict in ("certified", "conditional", "partial", "withheld")
        )
        + "."
    )
    out.append("")
    out.append("## Reading a row")
    out.append("")
    out.append(
        "A *shape class* is an operation plus a structural fingerprint of its arguments: "
        "which primitive functions occur (`funcs`), whether they are applied to the bare "
        "variable, to an integer power of it, or to something else (`fn_arg`), what kinds "
        "of exponents appear (`pow`, `pow_base`), how factors combine (`mul`), and the "
        "top-level form of the expression (`form`)."
    )
    out.append("")
    out.append("| Verdict | Meaning | `certifiable()` |")
    out.append("|---|---|---|")
    out.append("| ✅ certified | every corpus observation emitted a certificate | `True` |")
    out.append(
        "| 🟡 conditional | emits when the operation rewrites something; some "
        "observations had an empty derivation log | `False` (`class_conditional`) |"
    )
    out.append(
        "| 🟠 partial | observations disagree — the class is too coarse to separate them "
        "| `False` (`class_partial`) |"
    )
    out.append("| ⛔ withheld | no observation ever certified | `False` (`class_withheld`) |")
    out.append("")
    out.append(
        "`certifiable()` answers `True` only for `certified` classes, and in its default "
        "`verify` mode confirms by running the operation before saying so. It under-claims "
        "by construction: a shape the corpus has never reached answers `False` with reason "
        "`unknown_shape`."
    )
    out.append("")

    by_op: dict[str, list[dict]] = {}
    for row in rows:
        by_op.setdefault(row["operation"], []).append(row)

    for operation in sorted(by_op):
        op_rows = by_op[operation]
        op_totals: dict[str, int] = {}
        for row in op_rows:
            op_totals[row["verdict"]] = op_totals.get(row["verdict"], 0) + 1
        out.append(f"## `{operation}`")
        out.append("")
        out.append(
            f"{len(op_rows)} shape classes — "
            + ", ".join(f"{n} {verdict}" for verdict, n in sorted(op_totals.items()))
            + "."
        )
        out.append("")
        out.append("| Verdict | Shape | Examples | Blocked by |")
        out.append("|---|---|---|---|")
        for row in op_rows:
            shape = row["shape"].split("/", 1)[1] if "/" in row["shape"] else row["shape"]
            shape = shape.replace("/", ", ")
            examples = ", ".join(f"`{e}`" for e in row["examples"]) or "—"
            blocking = ", ".join(f"`{r}`" for r in row["blocking_rules"]) or "—"
            out.append(f"| {VERDICT_LABEL[row['verdict']]} | `{shape}` | {examples} | {blocking} |")
        out.append("")

    out.append("## Querying it")
    out.append("")
    out.append("```python")
    out.append("import alkahest as ak")
    out.append("")
    out.append("p = ak.ExprPool()")
    out.append('x = p.symbol("x")')
    out.append("")
    out.append("# Will this route give me something a referee can check?")
    out.append('answer = ak.certifiable("diff", ak.log(ak.sin(x)), x)')
    out.append("bool(answer)      # False")
    out.append("")
    out.append("# Plan across candidate routes without computing any of them.")
    out.append('ak.certifiable("diff", ak.sin(x), x, mode="ledger")')
    out.append("")
    out.append("# Fail loudly instead of degrading silently.")
    out.append("with ak.context(require_certificate=True):")
    out.append("    ak.diff(ak.log(ak.sin(x)), x)   # raises E-CERT-001")
    out.append("```")
    out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in artifacts differ from a regeneration",
    )
    args = parser.parse_args()

    ledger = build()
    json_text = json.dumps(ledger, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    markdown_text = render_markdown(ledger)

    partial = [row for row in ledger["rows"] if row["verdict"] == "partial"]

    if args.check:
        failures = []
        for path, expected in ((JSON_PATH, json_text), (MARKDOWN_PATH, markdown_text)):
            try:
                with open(path, encoding="utf-8") as handle:
                    actual = handle.read()
            except FileNotFoundError:
                failures.append(f"{os.path.relpath(path, REPO)} is missing")
                continue
            if actual != expected:
                failures.append(f"{os.path.relpath(path, REPO)} is stale")
        if failures:
            print("Certificate ledger drift detected:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            print(
                "\nThe certifiable surface changed but the ledger was not regenerated.\n"
                "Run `python scripts/gen_certificate_ledger.py` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print(
            f"Certificate ledger up to date: {ledger['corpus']['observations']} observations, "
            f"{len(ledger['rows'])} shape classes, {len(partial)} partial."
        )
        return 0

    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        handle.write(json_text)
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as handle:
        handle.write(markdown_text)

    verdicts: dict[str, int] = {}
    for row in ledger["rows"]:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    print(f"Wrote {os.path.relpath(JSON_PATH, REPO)}")
    print(f"Wrote {os.path.relpath(MARKDOWN_PATH, REPO)}")
    print(
        f"{ledger['corpus']['observations']} observations, "
        f"{len(ledger['rows'])} shape classes: "
        + ", ".join(f"{n} {verdict}" for verdict, n in sorted(verdicts.items()))
    )
    if partial:
        print(
            f"\nNOTE: {len(partial)} shape class(es) are `partial` — the feature vector "
            "cannot separate their certified and withheld observations, so "
            "`certifiable()` under-claims for them:",
        )
        for row in partial:
            print(f"  {row['shape']}  {row['observations']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
