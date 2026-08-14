"""The property-vs-method convention for zero-argument accessors.

The rule (see CONTRIBUTING.md § "Accessors: property or method?"):

    A zero-argument, O(1), non-allocating accessor that returns a scalar or a
    flag is a **property**.  Anything that returns a collection, allocates, or
    does real work is a **method**.

Two guards live here:

``test_converted_accessors_are_properties``
    Pins the accessors converted in 3.8.0.  Runs against the *installed*
    extension, so it also catches a wheel built from stale sources.

``test_no_zero_arg_scalar_accessor_is_a_method``
    Scans ``alkahest-py/src/lib.rs`` and fails on any *new* zero-argument
    scalar/flag ``#[pymethods]`` entry that is not a ``#[getter]``.  A runtime
    version of this check is not possible: PyO3 does not expose the Rust
    return type, and calling every zero-argument method to inspect its result
    would need a live instance of every class.  The source scan is exact and
    catches the mistake where it is made.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import alkahest.alkahest as _native
import pytest

# --------------------------------------------------------------------------
# Part 1 — the accessors converted in 3.8.0 stay properties
# --------------------------------------------------------------------------

CONVERTED_ACCESSORS = [
    ("UniPoly", "degree"),
    ("UniPoly", "is_zero"),
    ("MultiPoly", "is_zero"),
    ("MultiPoly", "total_degree"),
    ("RationalFunction", "is_zero"),
    ("OdeTrajectory", "t_final"),
    ("ODE", "order"),
    ("DAE", "n_equations"),
    ("DAE", "n_variables"),
    ("HybridODE", "n_events"),
    ("Component", "n_equations"),
    ("Component", "n_ports"),
    ("ArbBall", "is_exact"),
    ("GbPoly", "is_zero"),
    ("GbPoly", "n_vars"),
    ("MultiPolyFp", "is_zero"),
    ("MultiPolyFp", "total_degree"),
]

# Accessors that were already properties before 3.8.0 and must stay that way —
# the convention is only useful if it holds in both directions.
PRE_EXISTING_PROPERTIES = [
    ("Enclosure", "lower"),
    ("Enclosure", "upper"),
    ("Enclosure", "width"),
    ("Enclosure", "subdivisions"),
    ("Matrix", "rows"),
    ("Matrix", "cols"),
    ("RegularChain", "n_vars"),
    ("RosenfeldGroebnerResult", "consistent"),
    ("ArbBall", "mid"),
    ("ArbBall", "rad"),
    ("EvaluationResult", "is_enclosure"),
]


def _descriptor(class_name: str, attr: str):
    cls = getattr(_native, class_name, None)
    if cls is None:
        pytest.skip(f"{class_name} is not in this build (feature-gated)")
    desc = inspect.getattr_static(cls, attr, None)
    assert desc is not None, f"{class_name}.{attr} does not exist"
    return desc


@pytest.mark.parametrize(("class_name", "attr"), CONVERTED_ACCESSORS)
def test_converted_accessors_are_properties(class_name: str, attr: str) -> None:
    desc = _descriptor(class_name, attr)
    assert not callable(desc), (
        f"{class_name}.{attr} is a method; the convention makes zero-argument "
        f"scalar/flag accessors properties. Note that `x.{attr}` in a boolean "
        f"or format context is a *silent* bug when it is a bound method."
    )
    assert type(desc).__name__ == "getset_descriptor"


@pytest.mark.parametrize(("class_name", "attr"), PRE_EXISTING_PROPERTIES)
def test_pre_existing_properties_stay_properties(class_name: str, attr: str) -> None:
    assert not callable(_descriptor(class_name, attr))


def test_a_converted_accessor_actually_reads_as_a_value() -> None:
    """End-to-end: the value is the scalar, not a bound method."""
    pool = _native.ExprPool()
    x = pool.symbol("x")
    p = _native.UniPoly.from_coefficients([-1, 0, 1], x)
    assert p.degree == 2
    assert p.is_zero is False
    # The failure mode this guards against: a bound method is always truthy.
    assert not p.is_zero


# --------------------------------------------------------------------------
# Part 2 — no *new* zero-argument scalar accessor may be a method
# --------------------------------------------------------------------------

_LIB_RS = Path(__file__).resolve().parent.parent / "alkahest-py" / "src" / "lib.rs"

_SCALAR_RETURN = re.compile(
    r"->\s*(?:PyResult<)?\s*"
    r"(?:usize|u8|u16|u32|u64|i8|i16|i32|i64|f32|f64|bool"
    r"|Option<\s*(?:usize|u32|u64|i64|f64|bool)\s*>)\s*>?\s*$"
)

# Zero-argument scalar-returning methods that are *correctly* methods because
# they do real work, and so are exempt.  Adding an entry here is a deliberate
# claim that the call is not O(1); say why.
REAL_WORK_EXEMPTIONS = {
    # Gaussian elimination over the symbolic entries.
    ("PyMatrix", "rank"),
    # Walks every RHS expression looking for the time variable.
    ("PyODE", "is_autonomous"),
    # Re-runs the SOS identity check in exact arithmetic.
    ("PyPositivityCertificate", "verify"),
}


def _pymethods_fns(source: str):
    """Yield ``(class, fn_name, attrs, signature)`` for every ``#[pymethods]`` fn."""
    lines = source.split("\n")
    n = len(lines)
    i = 0
    while i < n:
        if lines[i].strip() != "#[pymethods]":
            i += 1
            continue
        j = i + 1
        while j < n and not re.search(r"\bimpl\b", lines[j]):
            j += 1
        m = re.search(r"impl\s+(?:\w+\s+for\s+)?([A-Za-z0-9_]+)", lines[j])
        cls = m.group(1) if m else "?"
        depth = 0
        started = False
        k = j
        attrs: list[str] = []
        while k < n:
            line = lines[k]
            if not started:
                depth += line.count("{") - line.count("}")
                if "{" in line:
                    started = True
                k += 1
                continue
            stripped = line.strip()
            fn = re.match(r"(?:pub\s+)?fn\s+([A-Za-z0-9_]+)\s*\(", stripped)
            if fn and depth == 1:
                sig = stripped
                kk = k
                while "{" not in sig and ";" not in sig and kk + 1 < n:
                    kk += 1
                    sig += " " + lines[kk].strip()
                yield cls, fn.group(1), list(attrs), re.sub(r"\s+", " ", sig.split("{")[0])
                attrs = []
            elif stripped.startswith("#["):
                attrs.append(stripped)
            elif stripped.startswith("//") or not stripped:
                pass
            elif depth == 1:
                attrs = []
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                break
            k += 1
        i = k


@pytest.mark.skipif(not _LIB_RS.exists(), reason="running against an installed wheel")
def test_no_zero_arg_scalar_accessor_is_a_method() -> None:
    offenders = []
    for cls, name, attrs, sig in _pymethods_fns(_LIB_RS.read_text()):
        if name.startswith("__") or any("getter" in a for a in attrs):
            continue
        if any(x in a for a in attrs for x in ("staticmethod", "classmethod", "#[new]")):
            continue
        if (cls, name) in REAL_WORK_EXEMPTIONS:
            continue
        args = re.match(r".*?fn\s+\w+\s*\((.*)\)\s*(?:->.*)?$", sig)
        if not args:
            continue
        rest = re.sub(r"^\s*(?:&\s*self|slf\s*:\s*[^,]+|self)\s*,?", "", args.group(1)).strip()
        rest = re.sub(r"^\s*_?py\s*:\s*Python<[^>]*>\s*,?", "", rest).strip()
        if rest.strip(","):
            continue
        if _SCALAR_RETURN.search(sig):
            offenders.append(f"{cls}::{name} — {sig.strip()}")

    assert not offenders, (
        "These zero-argument scalar/flag accessors are exposed as methods. "
        "Add `#[getter]` (see CONTRIBUTING.md § 'Accessors: property or "
        "method?'), or add them to REAL_WORK_EXEMPTIONS with a reason:\n  " + "\n  ".join(offenders)
    )
