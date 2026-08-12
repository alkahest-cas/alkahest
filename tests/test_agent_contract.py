"""Machine-readable agent contract tests."""

import alkahest
import pytest


def test_capabilities_reports_installed_build_features():
    caps = alkahest.capabilities()

    # v2: `verification` gained a generated `coverage` block and dropped the
    # never-emitted `lean_checked` status. See tests/test_certificate_ledger.py.
    # v3: `features` dropped `groebner_cuda` and `numpy` — see
    # `test_every_advertised_feature_has_an_entry_point` for the rule that
    # removed them and the invariant that keeps the next one out.
    assert caps["contract_version"] == 3
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
        "cuda",
    } == caps["features"].keys()
    assert "groebner_cuda" not in caps["features"]
    assert "numpy" not in caps["features"]
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


def _probe_egraph():
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    assert alkahest.simplify_egraph(x + pool.integer(0)).value is not None
    # The native module's own marker must agree with the reported bit.
    assert alkahest.alkahest.HAS_EGRAPH is True


def _probe_groebner():
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    assert hasattr(alkahest, "GroebnerBasis")
    assert alkahest.solve([x * x - pool.integer(1)], [x])


def _probe_parallel():
    # `simplify_par` is *not* a witness: it exists on every build and degrades
    # to the sequential path when the feature is off, so it cannot tell the
    # bit apart from its negation. These two methods genuinely appear and
    # disappear with `--features parallel`.
    assert hasattr(alkahest.CompiledFn, "call_batch_raw_par")
    assert hasattr(alkahest.CompiledFn, "call_batch_buffer_par")


def _probe_native_jit():
    assert alkahest.jit_is_available() is True
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    fn = alkahest.compile_expr(x * x, [x])
    assert fn([3.0]) == pytest.approx(9.0)


def _probe_cuda():
    assert hasattr(alkahest, "compile_cuda")
    assert hasattr(alkahest, "CudaCompiledFn")
    assert "compile_cuda" in alkahest.__all__


#: The whole point of a capability contract: an agent reads it once and picks
#: an operation without probing. That makes every key a promise, so every key
#: needs a named way to cash it in. `test_every_advertised_feature_has_an_entry_point`
#: below fails if a bit is added without one — which is what `groebner_cuda`
#: and `numpy` both lacked.
_FEATURE_ENTRY_POINTS = {
    "egraph": _probe_egraph,
    "groebner": _probe_groebner,
    "jit": _probe_native_jit,
    "llvm_jit": _probe_native_jit,
    "cranelift": _probe_native_jit,
    "cranelift_jit": _probe_native_jit,
    "parallel": _probe_parallel,
    "cuda": _probe_cuda,
}

#: `(owner, attribute)` pairs that exist if and only if the bit is `True`.
#: Checked in *both* directions, so this catches a bit reading `True` with the
#: entry point missing (the `cuda` bug in `d139a46`) *and* a bit reading
#: `False` on a build that really does have it.
_FEATURE_EXCLUSIVE_NAMES = {
    "cuda": (("alkahest", "compile_cuda"), ("alkahest", "CudaCompiledFn")),
    "parallel": (
        ("alkahest.CompiledFn", "call_batch_raw_par"),
        ("alkahest.CompiledFn", "call_batch_buffer_par"),
    ),
}

_EXCLUSIVE_OWNERS = {
    "alkahest": lambda: alkahest,
    "alkahest.CompiledFn": lambda: alkahest.CompiledFn,
}


def test_every_advertised_feature_has_an_entry_point():
    """Every `True` bit in `capabilities()["features"]` must be cashable.

    This is the generalisation of the two bugs that motivated contract v3, and
    would have caught both at once:

    * `groebner_cuda` was `True` on a `--features groebner-cuda` build while
      the string `groebner_cuda` appeared exactly once anywhere in
      `alkahest-py` — the capability line itself. No binding, no `*gpu*` name
      in the public or the private module, and `GroebnerBasis` exposing only
      CPU methods. Strictly worse than the `cuda` bug fixed in `d139a46`,
      which at least had a private route in.
    * `numpy` mapped to a Cargo feature gating a crate `lib.rs` never used,
      while `ak.numpy_eval` worked perfectly with the bit `False`. It meant
      nothing and correlated with nothing.

    Both were removed rather than wired up: a bit that reads `False` honestly
    beats one that reads `True` and lies, and a bit that means nothing at all
    is better gone than left to be misread. The rule this test enforces is
    that the decision has to be made *before* a key ships, because the failure
    mode of getting it wrong is a caller trusting something it should not —
    the same class of defect as a silent wrong answer.
    """
    features = alkahest.capabilities()["features"]

    assert set(_FEATURE_ENTRY_POINTS) == set(features), (
        "every capability bit needs a named entry point a caller can reach. "
        f"Undeclared bits: {sorted(set(features) - set(_FEATURE_ENTRY_POINTS))}; "
        f"stale probes: {sorted(set(_FEATURE_ENTRY_POINTS) - set(features))}. "
        "Add a probe, or drop the bit."
    )

    for name, enabled in sorted(features.items()):
        if enabled:
            _FEATURE_ENTRY_POINTS[name]()

    for name, exclusive in _FEATURE_EXCLUSIVE_NAMES.items():
        for owner_name, attr in exclusive:
            owner = _EXCLUSIVE_OWNERS[owner_name]()
            present = hasattr(owner, attr)
            assert present is features[name], (
                f"capabilities() reports {name}={features[name]} but "
                f"{owner_name}.{attr} {'exists' if present else 'does not exist'}; "
                "the contract and the namespace must agree"
            )


def test_removed_capability_bits_stay_removed():
    """`groebner_cuda` and `numpy` must not reappear without an entry point.

    Re-adding either is a real decision, not a merge accident: it means a
    Python binding now exists, and this test plus `_FEATURE_ENTRY_POINTS`
    above must both be updated to say what it is.
    """
    features = alkahest.capabilities()["features"]
    for gone in ("groebner_cuda", "numpy"):
        assert gone not in features
        assert features.get(gone, False) is False

    # The GPU Gröbner kernel is still Rust-only, by design: the crossover
    # policy in docs/symbolic-gpu-benchmarks.md says production dispatch must
    # not prefer the GPU until the benchmark harness says it wins. If that
    # changes, the binding lands first and the bit follows it — never the other
    # way round, which is the order that produced the overclaim.
    gpu_names = sorted(
        name for name in set(dir(alkahest)) | set(dir(alkahest.alkahest)) if "gpu" in name.lower()
    )
    assert not gpu_names, (
        f"a GPU entry point appeared ({gpu_names}) without a capability bit "
        "to advertise it. Add the bit and a probe in _FEATURE_ENTRY_POINTS, or "
        "keep the binding private."
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


def test_alkahest_error_catches_both_halves_of_the_hierarchy():
    """`except alkahest.AlkahestError` must catch Rust *and* Python errors.

    The Rust engines raise the native classes; the pure-Python subsystems
    (`ansatz`, `crosscheck`, `smt`, the batch helpers) raise the wrappers in
    `alkahest.exceptions`. Those two hierarchies used to be disjoint — the
    wrappers subclassed a pure-Python base that was not the native one — so the
    documented "catch anything this library raises" idiom silently missed every
    Python-layer error, including all three modules added for autoresearch
    loops. `exceptions.AlkahestError` is now a subclass of the native base,
    which makes the top-level name a true common ancestor.
    """
    from alkahest import exceptions

    pool = alkahest.ExprPool()
    x = pool.symbol("x")

    caught_native = False
    try:
        alkahest.integrate(alkahest.exp(x * x), x)
    except alkahest.AlkahestError:
        caught_native = True
    assert caught_native, "alkahest.AlkahestError missed a natively-raised error"

    caught_python = False
    try:
        raise exceptions.AnsatzError("probe")
    except alkahest.AlkahestError:
        caught_python = True
    assert caught_python, "alkahest.AlkahestError missed a Python-layer error"

    assert issubclass(exceptions.AlkahestError, alkahest.AlkahestError)


def test_python_only_errors_keep_their_keyword_constructors():
    """The overlay must not swallow the classes Python code actually raises.

    `AnsatzError`, `CrossCheckError` and `SmtError` are raised from the Python
    layer with `code=`/`remediation=` keywords; replacing them with a native
    class would break those call sites.
    """
    from alkahest import exceptions

    for cls, expected in (
        (exceptions.AnsatzError, "E-ANSATZ-"),
        (exceptions.CrossCheckError, "E-XCHECK-"),
        (exceptions.SmtError, "E-SMT-"),
    ):
        err = cls("probe")
        assert err.code.startswith(expected)
        assert isinstance(err, alkahest.AlkahestError)


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
        f"identical PTX on a second device returned different values: {on_one} vs {on_zero}"
    )
