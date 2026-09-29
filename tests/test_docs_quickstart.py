"""Execute the Quickstart page and check every output it claims.

`docs/mdbook/src/quickstart.md` annotates its examples with `# => value` lines.
This runs each Python block in order, in one shared namespace — the way a reader
copying down the page would — and asserts that the n-th thing printed by a block
equals the n-th `# =>` in that block.

It exists for the same reason `tests/smoke_readme.py` does. The README
quickstart once carried a comment claiming the derivative of `sin(x^2)` while
the code above it differentiated `sin(x^2 + 1)`; the comment was wrong for
months because nothing executed it. A documentation example that is only ever
read is a claim nobody checks, and this repository has found several: a
documented `.message` attribute that does not exist, a `cache.compile(...)` call
with the wrong arity, a notebook that could not run past cell 28.

Adding an example to the Quickstart therefore costs nothing extra — annotate it
with `# =>` and it is verified from then on. An example with no annotation is
still executed, so it must at least run.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "mdbook" / "src" / "quickstart.md"

#: ```python … ``` fences. `text` fences (prose maths) are deliberately skipped.
_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)

#: A claimed output: a whole-line `# => value` comment.
_EXPECT = re.compile(r"^\s*#\s*=>\s?(.*)$")


def _blocks() -> list[str]:
    assert DOC.is_file(), f"quickstart page missing: {DOC}"
    found = _BLOCK.findall(DOC.read_text(encoding="utf-8"))
    assert found, "no ```python blocks found — has the page been renamed?"
    return found


def _expected(block: str) -> list[str]:
    return [m.group(1).rstrip() for line in block.splitlines() if (m := _EXPECT.match(line))]


def test_every_quickstart_block_runs_and_prints_what_it_claims() -> None:
    """The whole page, top to bottom, in one namespace."""
    namespace: dict[str, object] = {"__name__": "__quickstart__"}
    checked = 0

    for index, block in enumerate(_blocks(), start=1):
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                exec(compile(block, f"{DOC.name}#block{index}", "exec"), namespace)
        except Exception as exc:
            pytest.fail(f"{DOC.name} block {index} raised {type(exc).__name__}: {exc}\n\n{block}")

        expected = _expected(block)
        if not expected:
            continue

        printed = buffer.getvalue().splitlines()
        assert len(printed) == len(expected), (
            f"{DOC.name} block {index}: printed {len(printed)} line(s) but claims "
            f"{len(expected)}.\nprinted={printed}\nclaimed={expected}\n\n{block}"
        )
        for printed_line, claim in zip(printed, expected, strict=True):
            assert printed_line == claim, (
                f"{DOC.name} block {index}: the page claims\n  {claim}\nbut the code prints\n"
                f"  {printed_line}"
            )
        checked += len(expected)

    assert checked >= 8, f"only {checked} claims checked; the page has lost its annotations"


def test_the_page_is_wired_into_the_book() -> None:
    """A page missing from SUMMARY.md is not published by mdBook."""
    summary = DOC.parent / "SUMMARY.md"
    assert "quickstart.md" in summary.read_text(encoding="utf-8"), (
        "quickstart.md is not linked from SUMMARY.md, so mdBook will not render it"
    )


def test_internal_links_resolve() -> None:
    """Every ./page.md link on the Quickstart points at a file that exists."""
    text = DOC.read_text(encoding="utf-8")
    missing = [
        target
        for target in re.findall(r"\]\(\./([A-Za-z0-9._-]+\.md)\)", text)
        if not (DOC.parent / target).is_file()
    ]
    assert not missing, f"quickstart links to non-existent pages: {missing}"
