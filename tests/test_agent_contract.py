"""Machine-readable agent contract tests."""

import alkahest


def test_capabilities_reports_installed_build_features():
    caps = alkahest.capabilities()

    # v2: `verification` gained a generated `coverage` block and dropped the
    # never-emitted `lean_checked` status. See tests/test_certificate_ledger.py.
    assert caps["contract_version"] == 2
    assert {"groebner", "jit", "egraph", "parallel", "features", "primitives", "verification"} <= (
        caps.keys()
    )
    assert {
        "egraph",
        "groebner",
        "jit",
        "cranelift",
        "llvm_jit",
        "cranelift_jit",
        "parallel",
        "numpy",
        "cuda",
        "groebner_cuda",
    } == caps["features"].keys()
    assert caps["groebner"] is caps["features"]["groebner"]
    assert caps["egraph"] is caps["features"]["egraph"]
    assert caps["parallel"] is caps["features"]["parallel"]
    assert caps["features"]["jit"] is caps["features"]["llvm_jit"]
    assert caps["features"]["cranelift"] is caps["features"]["cranelift_jit"]
    assert caps["jit"] is (caps["features"]["llvm_jit"] or caps["features"]["cranelift_jit"])
    assert caps["jit"] is alkahest.jit_is_available()


def test_cranelift_jit_enables_session_jit_flag():
    caps = alkahest.capabilities()
    features = caps["features"]
    if features["cranelift_jit"]:
        assert caps["jit"] is True
        assert alkahest.jit_is_available()
        assert features["cranelift"] is True
        # The two backends are not mutually exclusive. The shipped wheel is
        # cranelift-only, but `--features cuda` pulls in `alkahest-core/jit`
        # and so links LLVM alongside cranelift — a real configuration, built
        # and tested on GPU hardware. Asserting `not llvm_jit` unconditionally
        # encoded "cranelift implies no LLVM", which is false there.
        if not features["cuda"]:
            assert not features["llvm_jit"]

    primitives = alkahest.capabilities()["primitives"]

    assert primitives
    assert [row["name"] for row in primitives] == sorted(row["name"] for row in primitives)
    assert {
        "name",
        "simplify",
        "diff_forward",
        "diff_reverse",
        "numeric_f64",
        "numeric_ball",
        "lower_llvm",
        "lean_theorem",
    } == primitives[0].keys()


def test_lean_theorem_bit_reflects_actual_certificate_availability():
    """`primitives[i]["lean_theorem"]` must be a *truthful* signal: true only
    for primitives whose derivative certificate actually emits (non-empty,
    no `sorry`) from `alkahest.to_lean(alkahest.diff(...))` today.

    This is deliberately narrower than "a Mathlib lemma with this name
    exists" — see the `Primitive::lean_theorem` doc comment in
    `alkahest-core/src/primitive/mod.rs`. `log` (`Real.deriv_log`, holds
    unconditionally) and `sqrt`/`tan` (explicit `x ≠ 0` / `cos x ≠ 0`
    hypothesis binders, mirroring #236's positivity-binder mechanism) are now
    certifiable at the pointwise `f(x)` shape. The hyperbolic/inverse family,
    `atan2`, and `gamma` still have no encoding, so their bit must stay
    `False` until the emitter catches up.

    The value `capabilities()` reports now comes from the generated certificate
    ledger rather than the native bit, so it cannot be hand-edited into an
    overclaim; `tests/test_certificate_ledger.py` pins the native
    `Primitive::lean_theorem` overrides to the same set. If you make a new
    primitive's certificate typecheck, flip its `lean_theorem()` override to
    `Some(...)`, regenerate the ledger
    (`python scripts/gen_certificate_ledger.py`), add it to
    `CERTIFIABLE_PRIMITIVES` below, and verify with
    `lake env lean -DwarningAsError=true <file>` in `lean/` — not by
    inspection alone.
    """
    CERTIFIABLE_PRIMITIVES = {"sin", "cos", "exp", "log", "sqrt", "tan"}

    primitives = alkahest.capabilities()["primitives"]
    claiming = {row["name"] for row in primitives if row["lean_theorem"]}
    assert claiming == CERTIFIABLE_PRIMITIVES

    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    for name in CERTIFIABLE_PRIMITIVES:
        fn = getattr(alkahest, name)
        derived = alkahest.diff(fn(x), x)
        cert = alkahest.to_lean(derived)
        assert cert.strip(), f"{name}: lean_theorem=True but to_lean() is empty"
        assert "sorry" not in cert, f"{name}: certificate contains sorry"
        assert "admit" not in cert, f"{name}: certificate contains admit"


def test_capabilities_describes_verification_boundary():
    verification = alkahest.capabilities()["verification"]

    # `lean_checked` is deliberately absent: no code path has ever produced it
    # (checking is out of process — see `checkers` below), so advertising it was
    # an overclaim of the same kind as a false `lean_theorem` bit.
    assert verification["statuses"] == [
        "certificate_available",
        "exactly_verified",
        # Emitted only by the SMT bridge, for an external `unsat` that carries
        # no checked proof. It is advertised because `smt.solve` really does
        # produce it, and it is deliberately absent from
        # `research.MACHINE_CHECKED_STATUSES`.
        "externally_asserted",
        "numerically_checked",
        "unverified",
    ]
    assert verification["artifacts"] == {"lean4_source": True, "smtlib2_script": True}
    assert verification["checkers"] == {"lean4": "external", "smt": "external"}
    assert verification["coverage"]["shape_classes"]["certified"] > 0

    # Independent implementations and solvers are reported *negatively* too: an
    # absent one appears as a falsy value rather than being omitted, so a caller
    # can never mistake "not installed" for "agreed" or "checked".
    assert "sympy" in verification["oracles"]
    assert {"z3", "cvc5"} <= set(verification["smt_solvers"])


def test_externally_asserted_is_advertised_because_it_can_be_emitted():
    """The list must track reality, not intent — pin it to a real emission."""
    verification = alkahest.capabilities()["verification"]
    assert alkahest.smt.EXTERNALLY_ASSERTED in verification["statuses"]
    # ...and it must never count as machine-checked.
    assert alkahest.smt.EXTERNALLY_ASSERTED not in alkahest.research.MACHINE_CHECKED_STATUSES


def test_derived_result_labels_emitted_lean_source_as_unchecked_evidence():
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    result = alkahest.simplify(x + pool.integer(0))

    verification = result.verification
    assert verification["status"] == "certificate_available"
    assert verification["evidence"] == "derivation_log"
    assert verification["artifact_format"] == "lean4"
    assert verification["externally_verified"] is False
    assert isinstance(verification["side_conditions"], list)
    assert isinstance(result.certificate, str)


def test_advertised_cuda_capability_matches_the_public_namespace():
    """A capability bit must not advertise an unreachable entry point.

    On a `--features cuda` build the native module defines `compile_cuda`,
    `CudaCompiledFn` and `CudaError`, but `python/alkahest/__init__.py` never
    re-exported them: `capabilities()["features"]["cuda"]` said `True` while
    `ak.compile_cuda` raised `AttributeError`, and the only route in was the
    private `alkahest.alkahest` module. Found by running the CUDA suite on real
    hardware, which is the only configuration where the two can disagree.
    """
    features = alkahest.capabilities()["features"]
    reachable = all(
        hasattr(alkahest, name) for name in ("compile_cuda", "CudaCompiledFn", "CudaError")
    )
    assert features["cuda"] == reachable, (
        f"capabilities() reports cuda={features['cuda']} but the public "
        f"namespace {'exposes' if reachable else 'does not expose'} the CUDA "
        "entry points; the contract and the namespace must agree"
    )


def test_llvm_jit_bit_tracks_what_is_linked_not_which_flag_was_named():
    """`cuda` implies `alkahest-core/jit`, so a CUDA build links LLVM.

    `alkahest-py`'s own `jit` feature can be off while the core's is on, which
    made `llvm_jit` report `False` on a build that demonstrably emits NVPTX.
    """
    features = alkahest.capabilities()["features"]
    if features["cuda"]:
        assert features["llvm_jit"], (
            "a cuda build links the LLVM backend (alkahest-core: "
            'cuda = ["jit", ...]), so llvm_jit cannot be False'
        )
        assert features["jit"]


def test_cuda_kernels_are_reachable_on_every_device_not_just_zero():
    """Device selection must be reachable from Python, not only from Rust.

    `alkahest-core` has always had `CudaCompiledFn::call_batch_on(ordinal, ..)`
    — `nvptx_gpu::nvptx_multi_device_both_3090s` drives both cards through it —
    but the binding exposed only `call_batch`, hardwired to device 0. On a
    multi-GPU host every device but the first was therefore unreachable from
    Python, which is the same shape of gap as `cuda` advertising an entry point
    the public namespace could not reach.

    Asserts agreement rather than mere reachability: the PTX is
    device-independent, so the same kernel on a different card must return the
    identical bit pattern, not merely a close one.
    """
    if not alkahest.capabilities()["features"]["cuda"]:
        import pytest

        pytest.skip("not a cuda build")

    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    fn = alkahest.compile_cuda(x * x + pool.integer(1), [x])

    assert hasattr(fn, "call_batch_on"), "no way to select a CUDA device from Python"

    pts = [0.0, 1.0, -2.5, 1e8, 1e-8]
    on_zero = fn.call_batch_on(0, [pts])
    assert on_zero == fn.call_batch([pts]), "call_batch must equal call_batch_on(0, ..)"

    # Second device only if the host has one; single-GPU CI must still pass.
    try:
        on_one = fn.call_batch_on(1, [pts])
    except alkahest.CudaError:
        return
    assert on_one == on_zero, (
        "identical PTX on a second device returned different values: "
        f"{on_one} vs {on_zero}"
    )
