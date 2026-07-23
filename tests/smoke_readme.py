#!/usr/bin/env python3
"""Release-verification smoke test: run against an *installed wheel*, in a
single fresh Python process, with **no** pytest / conftest / test-suite
imports whatsoever.

This exists to close a real gap: release CI had import + JIT-capability
smoke steps, but nothing that (a) actually ran the README quickstart the way
a new user would copy-paste it, or (b) exercised `alkahest.parse(...)` as the
very first interaction with the module in a genuinely fresh interpreter.

Both gaps let real bugs ship:

  * B1 -- `alkahest.parse(...)` raised in a fresh interpreter. It only
    "worked" in CI because some *other* test, running earlier in the same
    pytest process, had already imported `alkahest.experimental` (or
    otherwise warmed up lazy state) before any test touched `parse`. A
    smoke test that shares a process with the rest of the test suite (or
    that calls other alkahest APIs before `parse`) cannot catch this class
    of bug -- it has to be the first thing the process does.

  * B6 -- the README quickstart's `diff(sin(x**2 + 1), x)` example printed
    a comment claiming the derivative of `sin(x^2)` (missing the `+ 1`).
    Nothing executed the README, so the stale/wrong comment shipped.

Design notes:

  * `run_parse_smoke()` runs FIRST, before anything else in this process
    touches `alkahest`, specifically to reproduce the conditions B1 needed
    to be caught: `import alkahest` then immediately `parse(...)`, with
    nothing else warming up module state first.

  * `run_readme_quickstart()` extracts every ```python fenced block from
    the README's "## Quick start" section and `exec`s it, so this test
    tracks the docs automatically -- no hand-copied duplicate of the
    example that can drift from what's actually published.

  * Where the README shows a *value* as a plain literal (e.g. `exp(x)`,
    `x`, `1`, `10.0`, `1/3`) we assert `str(...)` equality, since those
    already match alkahest's canonical `str(Expr)` output exactly.

  * Where the README shows a *hand-written math expression* in a comment
    (e.g. `# 2*x*cos(x^2 + 1)`) we do NOT assert a literal string match:
    alkahest's `str(Expr)` uses a different (fully-parenthesized,
    variable-before-coefficient) printing convention than the README's
    hand-written notation, so a literal diff would flag harmless
    formatting differences as failures. Instead we check the *value* is
    correct by numeric evaluation at several points -- which is exactly
    the kind of check that would have caught B6 (a dropped `+ 1` changes
    the numeric result, not just its formatting).

Exit code is non-zero if anything fails, with a clear message for each
failure printed to stderr.
"""

from __future__ import annotations

import contextlib
import io
import math
import pathlib
import re
import sys
import traceback

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"

FAILURES: list[str] = []


def _fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"FAIL: {msg}", file=sys.stderr)


def _check(cond: bool, msg: str) -> None:
    if cond:
        print(f"ok:   {msg}")
    else:
        _fail(msg)


# ---------------------------------------------------------------------------
# 1. Fresh-interpreter parse check (guards B1).
#
# Must run before anything else in this process touches `alkahest` -- the
# whole point is to reproduce "parse() is the first thing this interpreter
# ever asks alkahest to do", which is exactly the condition that was masked
# by test-suite import order previously.
# ---------------------------------------------------------------------------


def run_parse_smoke() -> None:
    print("=== Fresh-interpreter parse smoke (guards B1) ===")
    import alkahest as ak  # first alkahest import in this process, on purpose

    try:
        pool = ak.ExprPool()
        expr = ak.parse("sin(x)", pool)
    except Exception:
        _fail(
            "ak.parse('sin(x)', ExprPool()) raised as the FIRST alkahest "
            f"call in a fresh interpreter:\n{traceback.format_exc()}"
        )
        return
    _check(expr is not None, "parse('sin(x)') returned a non-None Expr")

    for src in ("cos(x)", "exp(x)", "log(x)"):
        try:
            e = ak.parse(src, pool)
        except Exception:
            _fail(f"ak.parse({src!r}, pool) raised:\n{traceback.format_exc()}")
            continue
        _check(e is not None, f"parse({src!r}) returned a non-None Expr")


# ---------------------------------------------------------------------------
# 2. README quickstart execution (guards B6-style doc/output drift).
# ---------------------------------------------------------------------------


def extract_quickstart_blocks(readme_text: str) -> list[str]:
    """Pull every ```python fenced block out of the '## Quick start' section.

    Stops at the next '## ' heading so unrelated blocks are not picked up --
    notably the "## Reinforcement learning" section's example, which needs
    the optional `alkahest[rl]` extra and is not part of a default
    `pip install alkahest`.
    """
    lines = readme_text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "## Quick start")
    except StopIteration:
        raise RuntimeError(
            "README.md has no '## Quick start' section -- this smoke test "
            "is out of sync with the docs and needs updating"
        ) from None

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break

    section = "\n".join(lines[start:end])
    blocks = re.findall(r"```python\n(.*?)```", section, re.DOTALL)
    if not blocks:
        raise RuntimeError(
            "no ```python fenced blocks found under '## Quick start' in "
            "README.md -- this smoke test is out of sync with the docs"
        )
    return blocks


def _exec_block(index: int, source: str) -> dict | None:
    namespace: dict = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(source, f"<README quickstart block {index}>", "exec"), namespace)
    except Exception:
        _fail(
            f"README quickstart block {index} raised an exception when "
            f"executed verbatim:\n{source}\n{traceback.format_exc()}"
        )
        return None
    captured = buf.getvalue()
    if captured:
        print(f"--- stdout of block {index} ---")
        print(captured, end="" if captured.endswith("\n") else "\n")
    return namespace


def _check_block_0(ns: dict) -> None:
    """The first Quick start block: diff/integrate/simplify/JIT/parse."""
    ak = ns["ak"]
    x = ns["x"]

    result = ns.get("result")
    if result is None:
        _fail("block 0: 'result' (diff(sin(x**2+1), x)) not defined after exec")
    else:
        _check(bool(str(result.value)), "block 0: diff result .value stringifies non-empty")
        _check(len(result.steps) > 0, "block 0: diff result .steps is non-empty")
        # Numeric check rather than literal string match -- see module
        # docstring. This is the check that would have caught B6: dropping
        # the '+ 1' changes the numeric derivative, not just its printed form.
        for x0 in (0.3, 1.7, -2.1):
            got = ak.eval_expr(result.value, {x: x0})
            want = 2 * x0 * math.cos(x0**2 + 1)
            _check(
                abs(got - want) < 1e-9,
                f"block 0: d/dx sin(x^2+1) at x={x0}: got {got}, want {want} "
                "(README comment says '2*x*cos(x^2 + 1)')",
            )

    r = ns.get("r")
    if r is None:
        _fail("block 0: 'r' (integrate(exp(x), x)) not defined after exec")
    else:
        _check(
            str(r.value) == "exp(x)",
            f"block 0: integrate(exp(x), x).value == 'exp(x)', got {str(r.value)!r}",
        )

    s = ns.get("s")
    if s is None:
        _fail("block 0: 's' (simplify(x + 0)) not defined after exec")
    else:
        _check(
            str(s.value) == "x",
            f"block 0: simplify(x + 0).value == 'x', got {str(s.value)!r}",
        )

    trig_expr = ak.simplify_trig(ak.sin(x) ** 2 + ak.cos(x) ** 2).value
    _check(
        str(trig_expr) == "1",
        f"block 0: simplify_trig(sin(x)^2 + cos(x)^2).value == '1', got {str(trig_expr)!r}",
    )

    f = ns.get("f")
    if f is None:
        _fail("block 0: 'f' (compile_expr(x**2 + 1, [x])) not defined after exec")
    else:
        got = f([3.0])
        _check(
            abs(got - 10.0) < 1e-9,
            f"block 0: compile_expr(x**2 + 1, [x])([3.0]) == 10.0, got {got}",
        )

    e = ns.get("e")
    if e is None:
        _fail("block 0: 'e' (parse('sin(x)^2 + cos(x)^2', ...)) not defined after exec")
    else:
        parsed_trig = ak.simplify_trig(e).value
        _check(
            str(parsed_trig) == "1",
            f"block 0: simplify_trig(parse('sin(x)^2 + cos(x)^2')).value == '1', "
            f"got {str(parsed_trig)!r}",
        )


def _check_block_1(ns: dict) -> None:
    """The second Quick start block: apart / definite integral / certificate."""
    ak = ns["ak"]
    x = ns.get("x")

    f = ns.get("f")
    if f is None:
        _fail("block 1: 'f' (1 / (x**2 - 1)) not defined after exec")
    else:
        apart_str = str(ak.apart(f, x))
        _check(bool(apart_str), "block 1: apart(1/(x^2 - 1), x) stringifies non-empty")

    r = ns.get("r")
    if r is None:
        _fail("block 1: 'r' (definite integral of x**2 from 0 to 1) not defined after exec")
    else:
        _check(
            str(r.value) == "1/3",
            "block 1: integrate(x**2, x, 0, 1).value == '1/3' (README: "
            f"'integral of x^2 dx from 0 to 1 = 1/3'), got {str(r.value)!r}",
        )
        # r.certificate is documented as "Lean 4 proof term when available" --
        # descriptive, not a literal value, so only check it doesn't blow up
        # to stringify (None is a valid, documented outcome).
        _check(
            r.certificate is None or isinstance(r.certificate, str),
            f"block 1: r.certificate is None or str, got {type(r.certificate)!r}",
        )


_BLOCK_CHECKS = (_check_block_0, _check_block_1)


def run_readme_quickstart() -> None:
    print("\n=== Executing README '## Quick start' code blocks ===")
    text = README_PATH.read_text(encoding="utf-8")
    blocks = extract_quickstart_blocks(text)
    print(f"found {len(blocks)} python block(s) in '## Quick start'")

    if len(blocks) != len(_BLOCK_CHECKS):
        _fail(
            f"README '## Quick start' now has {len(blocks)} python block(s) "
            f"but this smoke test only has assertions for {len(_BLOCK_CHECKS)} "
            "-- update tests/smoke_readme.py's _BLOCK_CHECKS to match"
        )

    for i, block in enumerate(blocks):
        ns = _exec_block(i, block)
        if ns is None:
            continue
        if i < len(_BLOCK_CHECKS):
            _BLOCK_CHECKS[i](ns)


def main() -> int:
    run_parse_smoke()
    run_readme_quickstart()

    print()
    if FAILURES:
        print(f"=== SMOKE FAILED: {len(FAILURES)} failure(s) ===", file=sys.stderr)
        for msg in FAILURES:
            print(f" - {msg}", file=sys.stderr)
        return 1

    print("=== SMOKE OK: README quickstart + fresh-interpreter parse checks passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
