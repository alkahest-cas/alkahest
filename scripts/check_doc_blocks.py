#!/usr/bin/env python3
"""Run every ```python block in the user docs and report the ones that fail.

Scope: ``docs/mdbook/src/*.md`` (the docs site) and ``README.md``.

Each page runs in its own fresh interpreter. The blocks of a page run in
order in one shared namespace, the way a reader works through the page, so a
later block may use names an earlier block defined. A failing block does not
stop the page: the remaining blocks still run (and may fail in turn because
of it -- fix the first failure first).

Opting out. A block that is deliberately illustrative (needs a GPU, a
network service, a licence, or shows an error on purpose) is skipped when the
nearest non-blank line above its opening fence is::

    <!-- doc-check: skip -->

optionally followed by a reason (``<!-- doc-check: skip — needs CUDA -->``).
Use it sparingly: prefer fixing the block. A block that needs an optional
package runs when the package is importable and is skipped otherwise::

    <!-- doc-check: needs cypari2|cypari -->

(``|`` separates alternatives; any one of them suffices). A block whose whole point is to
raise should catch the error instead, so the page still shows the behaviour
and the check still runs it.

Each page gets a wall-clock timeout (``--timeout``, default 120 s) and an
address-space cap (``--mem-mb``, default 8192). ``MALLOC_ARENA_MAX`` is set
in the child so the cap does not trip on glibc's per-thread arenas.

``--arrows`` additionally reports ``# → value`` comments whose value differs
from what the line actually produces (bare expressions and single-argument
``print(...)`` calls only; whitespace is normalised). This is advisory: the
arrows are often hand-written maths, so mismatches are printed but do not
fail the run unless ``--strict-arrows`` is given.

Usage::

    python scripts/check_doc_blocks.py                 # all pages
    python scripts/check_doc_blocks.py docs/mdbook/src/solving.md
    python scripts/check_doc_blocks.py --arrows -v

Exit status is 1 if any block failed (or timed out), 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PAGES = [
    *sorted((REPO_ROOT / "docs" / "mdbook" / "src").glob("*.md")),
    REPO_ROOT / "README.md",
]

FENCE_OPEN = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,})\s*(?P<lang>[\w+-]*)")
SKIP_MARK = re.compile(r"<!--\s*doc-check:\s*skip\b")
NEEDS_MARK = re.compile(r"<!--\s*doc-check:\s*needs\s+([\w.|]+)")


def extract_blocks(path: pathlib.Path) -> list[dict]:
    """Return the page's python blocks as ``{line, code, skip}`` dicts."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        m = FENCE_OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        fence, lang, indent = m["fence"], m["lang"].lower(), len(m["indent"])
        start = i
        j = i + 1
        body: list[str] = []
        while j < len(lines) and not lines[j].strip().startswith(fence):
            body.append(lines[j][indent:] if lines[j][:indent].isspace() else lines[j])
            j += 1
        if lang in ("python", "py", "python3"):
            k = start - 1
            while k >= 0 and not lines[k].strip():
                k -= 1
            prev = lines[k] if k >= 0 else ""
            skip = bool(SKIP_MARK.search(prev))
            needs = NEEDS_MARK.search(prev)
            blocks.append(
                {
                    "line": start + 1,
                    "code": "\n".join(body) + "\n",
                    "skip": skip,
                    "needs": needs[1].split("|") if needs else [],
                }
            )
        i = j + 1
    return blocks


# The driver runs inside the child interpreter. It reads the blocks as JSON,
# executes them in one namespace, and writes one JSON result per block.
DRIVER = r"""
import ast, contextlib, io, json, re, sys, traceback

blocks = json.load(open(sys.argv[1], encoding="utf-8"))
check_arrows = sys.argv[2] == "1"
try:  # address-space cap; set here, not in preexec_fn, which is unsafe with threads
    import resource
    _cap = int(sys.argv[3]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (_cap, _cap))
except (ImportError, ValueError, OSError):
    pass
ns = {"__name__": "__main__"}
ARROW = "# →"

def _norm(s):
    return " ".join(str(s).split())

import importlib.util

def _have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False

for b in blocks:
    skip = b["skip"] or (b["needs"] and not any(_have(m) for m in b["needs"]))
    res = {"line": b["line"], "skip": bool(skip), "ok": True, "error": None, "arrows": []}
    if skip:
        print("\x00DOCCHECK " + json.dumps(res), flush=True)
        continue
    src = b["code"]
    src_lines = src.splitlines()
    try:
        tree = ast.parse(src, filename=f"<block@{b['line']}>")
        for node in tree.body:
            line_txt = src_lines[node.end_lineno - 1]
            want = None
            if check_arrows and ARROW in line_txt and isinstance(node, ast.Expr):
                want = line_txt.split(ARROW, 1)[1].strip()
            mod = ast.Module(body=[node], type_ignores=[])
            if want is None:
                exec(compile(mod, f"<block@{b['line']}>", "exec"), ns)
                continue
            call = node.value
            is_print = (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "print" and len(call.args) == 1 and not call.keywords)
            if is_print:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(compile(mod, f"<block@{b['line']}>", "exec"), ns)
                got = buf.getvalue().rstrip("\n")
                sys.stdout.write(buf.getvalue())
            else:
                val = eval(compile(ast.Expression(node.value), f"<block@{b['line']}>", "eval"), ns)
                got = str(val)
            # A trailing "   (why)" after the value is commentary, not output.
            want_v = re.split(r"\s{2,}\(", want, maxsplit=1)[0]
            if _norm(got) != _norm(want_v):
                res["arrows"].append({"line": b["line"] + node.end_lineno,
                                      "want": want, "got": got})
    except BaseException as e:  # noqa: BLE001 - report everything, incl. SystemExit
        res["ok"] = False
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        res["error"] = "".join(tb[-6:])
        # The innermost frame that is the block itself names the failing line.
        at = [f.lineno for f in traceback.extract_tb(e.__traceback__)
              if f.filename == f"<block@{b['line']}>"]
        if isinstance(e, SyntaxError) and e.filename == f"<block@{b['line']}>":
            at = [e.lineno]
        msg = (str(e).splitlines() or [""])[0]
        res["summary"] = f"{type(e).__name__}: {msg}"[:300]
        res["at"] = b["line"] + at[-1] if at else None
    sys.stdout.flush()
    print("\x00DOCCHECK " + json.dumps(res), flush=True)
"""


def run_page(path: pathlib.Path, args) -> dict:
    blocks = extract_blocks(path)
    out = {"page": path, "blocks": blocks, "results": [], "timeout": False, "crash": None}
    if not blocks:
        return out
    with tempfile.TemporaryDirectory() as td:
        bjson = pathlib.Path(td) / "blocks.json"
        bjson.write_text(json.dumps(blocks), encoding="utf-8")
        driver = pathlib.Path(td) / "driver.py"
        driver.write_text(DRIVER, encoding="utf-8")
        env = dict(os.environ)
        env.setdefault("MALLOC_ARENA_MAX", "2")
        env.setdefault("MPLBACKEND", "Agg")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(driver),
                    str(bjson),
                    "1" if args.arrows else "0",
                    str(args.mem_mb),
                ],
                cwd=td,  # scratch files a block writes land in the temp dir
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            out["timeout"] = True
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr, rc = "", None
    for line in stdout.splitlines():
        if line.startswith("\x00DOCCHECK "):
            out["results"].append(json.loads(line[len("\x00DOCCHECK ") :]))
    if not out["timeout"] and len(out["results"]) < len(blocks):
        out["crash"] = f"interpreter exited with status {rc}\n{stderr[-2000:]}"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pages", nargs="*", type=pathlib.Path, help="markdown files (default: all)")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per page")
    ap.add_argument("--mem-mb", type=int, default=8192, help="address-space cap per page")
    ap.add_argument("--arrows", action="store_true", help="report `# → value` mismatches")
    ap.add_argument("--strict-arrows", action="store_true", help="fail on arrow mismatches")
    ap.add_argument("-j", "--jobs", type=int, default=2, help="pages run concurrently")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.strict_arrows:
        args.arrows = True

    pages = [p.resolve() for p in args.pages] or DEFAULT_PAGES
    n_run = n_skip = n_fail = n_arrow = 0
    failed_pages: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results = list(ex.map(lambda p: run_page(p, args), pages))
    for page, r in zip(pages, results):
        if not r["blocks"]:
            continue
        rel = page.relative_to(REPO_ROOT) if page.is_relative_to(REPO_ROOT) else page
        page_bad = False
        for res in r["results"]:
            if res["skip"]:
                n_skip += 1
                continue
            n_run += 1
            if not res["ok"]:
                n_fail += 1
                page_bad = True
                where = res.get("at") or res["line"]
                print(f"FAIL {rel}:{where}: {res.get('summary', '?')}")
                if args.verbose:
                    print(_indent(res["error"]))
            for a in res["arrows"]:
                n_arrow += 1
                page_bad = page_bad or args.strict_arrows
                print(f"ARROW {rel}:{a['line']}: comment says {a['want']!r}, got {a['got']!r}")
        if r["timeout"]:
            page_bad = True
            n_fail += 1
            done = len(r["results"])
            line = r["blocks"][done]["line"] if done < len(r["blocks"]) else "?"
            print(f"TIMEOUT {rel}: exceeded {args.timeout:g}s in the block at line {line}")
        elif r["crash"]:
            page_bad = True
            n_fail += 1
            done = len(r["results"])
            line = r["blocks"][done]["line"] if done < len(r["blocks"]) else "?"
            print(f"CRASH {rel}: block at line {line}: {r['crash']}")
        if page_bad:
            failed_pages.append(str(rel))
        elif args.verbose:
            print(f"ok   {rel} ({len(r['blocks'])} blocks)")
    summary = f"{n_run} blocks run, {n_skip} skipped, {n_fail} failed"
    if args.arrows:
        summary += f", {n_arrow} arrow mismatches"
    print(summary)
    if failed_pages:
        print("pages with failures: " + ", ".join(failed_pages))
    return 1 if failed_pages else 0


def _indent(s: str) -> str:
    return "\n".join("    " + ln for ln in (s or "").rstrip().splitlines())


if __name__ == "__main__":
    sys.exit(main())
