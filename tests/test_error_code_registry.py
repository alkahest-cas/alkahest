"""The error-code registry is a contract, so something has to enforce it.

``scripts/check_error_codes.py`` has existed for a long time and no workflow
ran it. By the time it was next run by hand it reported **53 failures**: 51
codes that alkahest-core raises but never registered, and two PyO3 exception
classes with no Python counterpart to catch them by name.

That is the ordinary fate of a check nobody runs. Stable ``E-SUBSYSTEM-NNN``
codes are a headline guarantee of this project — an agent is told it can plan
against them — so a code that exists in the kernel and not in the registry is
an undocumented part of the public contract, and a raisable exception class
with no Python name is one a caller cannot catch.

This module puts the script on the same footing as every other gate: it runs in
``pytest tests/``, which CI already runs, so the registry cannot silently drift
again.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_error_codes.py"


@pytest.mark.skipif(not SCRIPT.is_file(), reason="checkout-only script (absent in a wheel)")
def test_error_code_registry_is_consistent():
    """Every raised code is registered, and every PyO3 class has a Python name.

    The script's own output is the failure message: it enumerates each offending
    code, which is far more useful than a bare non-zero exit.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        "scripts/check_error_codes.py failed.\n"
        "A new code must be added to alkahest_core::errors::codes::REGISTRY, and a new "
        "PyO3 Py*Error needs a matching class in python/alkahest/exceptions.py.\n\n"
        f"{proc.stdout}{proc.stderr}"
    )


def _load_checker():
    """Import the script as a module without running ``main``."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_check_error_codes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SCRIPT.is_file(), reason="checkout-only script (absent in a wheel)")
def test_the_gate_can_actually_detect_an_unregistered_code(tmp_path):
    """A gate that cannot fail is not a gate.

    Exercised against synthetic sources in ``tmp_path`` rather than by
    temporarily corrupting the real ``codes.rs``: a test that mutates tracked
    source can leave the tree dirty if it dies, and would hand a concurrent
    ``cargo build`` a broken file.
    """
    checker = _load_checker()

    registry = tmp_path / "codes.rs"
    registry.write_text(
        "pub const REGISTRY: &[ErrorSpec] = &[\n"
        '    ErrorSpec { code: "E-FAKE-001", class: "FakeError" },\n'
        "];\n",
        encoding="utf-8",
    )
    assert checker.parse_registry(registry) == {"E-FAKE-001"}

    core = tmp_path / "core"
    core.mkdir()
    (core / "fake.rs").write_text(
        "impl AlkahestError for FakeError {\n"
        "    fn code(&self) -> &'static str {\n"
        '        match self { FakeError::A => "E-FAKE-001", FakeError::B => "E-FAKE-002" }\n'
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    raised, _ = checker.collect_rust_codes(core)
    assert {"E-FAKE-001", "E-FAKE-002"} <= raised
    # E-FAKE-002 is raised and unregistered — exactly what the gate must catch.
    assert raised - checker.parse_registry(registry) == {"E-FAKE-002"}


@pytest.mark.skipif(not SCRIPT.is_file(), reason="checkout-only script (absent in a wheel)")
def test_the_gate_can_detect_one_code_claimed_by_two_impls(tmp_path):
    """The duplicate that REGISTRY's own uniqueness test could not see.

    ``E-ODE-021`` was returned by ``NumericOdeError::StepSizeTooSmall`` *and* by
    ``series_solve``'s ``SeriesError::IrregularSingular``. The registry listed it
    once, so ``no_duplicate_codes`` passed while the code meant two incompatible
    things on the wire: "the adaptive integrator gave up" and "this equation has
    no Frobenius series at this point". A caller told to branch on stable codes
    would have read the first as the second.

    Like the sibling gate above, this runs against synthetic sources rather than
    by corrupting tracked files.
    """
    checker = _load_checker()

    core = tmp_path / "core"
    core.mkdir()
    (core / "a.rs").write_text(
        "impl AlkahestError for AlphaError {\n"
        "    fn code(&self) -> &'static str {\n"
        '        match self { AlphaError::X => "E-FAKE-001" }\n'
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    (core / "b.rs").write_text(
        "impl crate::errors::AlkahestError for BetaError {\n"
        "    fn code(&self) -> &'static str {\n"
        '        match self { BetaError::Y => "E-FAKE-001" }\n'
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    owners = checker.collect_rust_code_owners(core)
    assert owners["E-FAKE-001"] == {"AlphaError", "BetaError"}
    assert "E-FAKE-001" not in checker.DELIBERATE_ALIASES


@pytest.mark.skipif(not SCRIPT.is_file(), reason="checkout-only script (absent in a wheel)")
def test_a_doc_comment_mentioning_a_code_is_not_a_claim_on_it():
    """Only an impl body counts, or the gate would fire on cross-references.

    Half the codes in this repo are named in the doc comment of some *other*
    error explaining how they differ. Counting those as owners would make the
    duplicate check unusable, and it would be turned off rather than fixed.
    """
    checker = _load_checker()
    owners = checker.collect_rust_code_owners(REPO / "alkahest-core" / "src")
    shared = {c: v for c, v in owners.items() if len(v) > 1}
    unexpected = set(shared) - set(checker.DELIBERATE_ALIASES)
    assert not unexpected, (
        f"codes claimed by two AlkahestError impls: "
        f"{ {c: sorted(shared[c]) for c in sorted(unexpected)} }"
    )


def test_every_python_exception_class_carries_a_code():
    """A wrapper with no ``.code`` is indistinguishable from a bare exception."""
    from alkahest import exceptions

    missing = []
    for name in dir(exceptions):
        cls = getattr(exceptions, name)
        if (
            isinstance(cls, type)
            and issubclass(cls, exceptions.AlkahestError)
            and cls is not exceptions.AlkahestError
        ):
            try:
                code = cls("probe").code
            except TypeError:  # pragma: no cover - constructor takes extra args
                continue
            if not code or code == "E-UNKNOWN":
                missing.append(name)
    assert not missing, f"exception classes with no stable code: {missing}"
