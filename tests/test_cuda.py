"""Python-level coverage for the NVPTX / CUDA backend (``ak.compile_cuda``).

Until this file existed, ``pytest tests/ -k cuda`` selected **zero** tests: the
whole GPU surface was covered only from Rust (``alkahest-core/tests/nvptx_gpu.rs``),
so the Python binding could have been unreachable — and for three releases it was.
``capabilities()["features"]["cuda"]`` reported ``True`` on a CUDA build while
``ak.compile_cuda`` raised ``AttributeError``, because ``python/alkahest/__init__.py``
re-exported none of the three names the native module defines under that feature.

Three tiers, deliberately separated by what each one actually needs:

1. **Contract** — runs everywhere, including this non-CUDA build. Asserts the
   capability bit and the public namespace agree *in both directions*: a build
   without the feature must not expose the entry points, and a build with it must.
   This is the tier that would have caught the export gap above.
2. **Compile-only** — needs ``--features cuda`` (LLVM 15 with the NVPTX target),
   but no device: PTX emission and the ``CudaError`` refusals both happen host-side.
3. **Device** — needs a CUDA build *and* a GPU that answers. Every one of these
   compares the kernel's output against a CPU path for the same expression,
   because a GPU kernel that returns different numbers from the CPU is the
   failure that matters; "it launched" is not the property under test.

Skip discipline (mirrors ``alkahest-core/tests/nvptx_gpu.rs``): tiers 2 and 3 skip
when the feature or the device is absent — that is honest, and it is what happens
in CI, in the shipped wheel, and on any developer machine. But ``ALKAHEST_GPU_TESTS=1``
(set by ``.github/workflows/cuda_nightly.yml`` on the ``gpu-3090`` runner) *promises*
hardware, and a promise that silently degrades to a skip is how a job whose entire
purpose is to exercise a GPU reports success without touching one. With that
variable set, an unusable device is a collection error, not a skip.
"""

from __future__ import annotations

import math
import os
import warnings

import alkahest as ak
import pytest

# The two names the native module defines *under* `--features cuda`. Both must
# be reachable from the public package, or neither.
CUDA_ENTRY_POINTS = ("CudaCompiledFn", "compile_cuda")

# `CudaError` is not in that list on purpose: the native module registers it
# unconditionally, like every other exception class, so it is bound on every
# build and simply never raised without the feature. It is the type a caller
# writes `except` against, and that code must compile against the default wheel.


GPU_PROMISED = os.environ.get("ALKAHEST_GPU_TESTS") == "1"

CUDA_FEATURE = bool(ak.capabilities()["features"]["cuda"])


def _device_unavailable_reason() -> str | None:
    """Return ``None`` when a real device answered, else why it did not.

    Deliberately *not* a bare capability read: the feature flag says what was
    linked, not that a GPU is present, and ``libcuda.so.1`` is loaded lazily so
    the failure only surfaces at launch. The probe therefore compiles a trivial
    kernel and launches it.

    Only a raised exception counts as unavailable; the value that comes back is
    deliberately not inspected. A device that answers with the *wrong* number is
    a failure for the tests below to report, never grounds for declaring the
    hardware absent and skipping them.
    """
    if not CUDA_FEATURE:
        return "extension built without --features cuda"
    missing = [name for name in CUDA_ENTRY_POINTS if not hasattr(ak, name)]
    if missing:
        # test_capability_bit_and_public_namespace_agree fails loudly on this;
        # the device tiers have nothing to call, so they skip.
        return f"cuda feature reported but {', '.join(missing)} not exported"

    pool = ak.ExprPool()
    x = pool.symbol("x")
    try:
        fn = ak.compile_cuda(x + pool.integer(1), [x])
        fn.call_batch([[2.0]])
    except Exception as exc:  # any failure at all means "no usable device"
        return f"no usable CUDA device: {type(exc).__name__}: {exc}"
    return None


DEVICE_SKIP_REASON = _device_unavailable_reason()

if GPU_PROMISED and DEVICE_SKIP_REASON is not None:
    raise RuntimeError(
        "ALKAHEST_GPU_TESTS=1 promises a CUDA build running on GPU hardware, but "
        f"{DEVICE_SKIP_REASON}. Refusing to skip: a GPU job that reports success "
        "without reaching a GPU is worse than a red one."
    )

requires_cuda_build = pytest.mark.skipif(
    not CUDA_FEATURE, reason="extension built without --features cuda"
)
requires_gpu = pytest.mark.skipif(
    DEVICE_SKIP_REASON is not None,
    reason=DEVICE_SKIP_REASON or "",
)


@pytest.fixture
def pool():
    return ak.ExprPool()


def _sample_indices(n: int, k: int) -> list[int]:
    """Evenly spaced indices always including the first and the last point.

    The last point is the one an off-by-one in the grid-stride bound misses, so
    it is never sampled away.
    """
    if n <= k:
        return list(range(n))
    step = n // k
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


# ---------------------------------------------------------------------------
# 1. Contract — runs on every build, including this one
# ---------------------------------------------------------------------------


def test_capability_bit_and_public_namespace_agree():
    """``features["cuda"]`` must be true exactly when the entry points resolve.

    Both directions matter. ``True`` with nothing exported is the overclaim that
    shipped for three releases; ``False`` with the names present would mean the
    contract under-reports a feature an agent could otherwise select.
    """
    reachable = [name for name in CUDA_ENTRY_POINTS if hasattr(ak, name)]
    if CUDA_FEATURE:
        assert reachable == list(CUDA_ENTRY_POINTS), (
            "capabilities() reports cuda=True but the public namespace is missing "
            f"{sorted(set(CUDA_ENTRY_POINTS) - set(reachable))}; a capability bit "
            "must not advertise an entry point only reachable via the private "
            "alkahest.alkahest module"
        )
    else:
        assert reachable == [], (
            f"capabilities() reports cuda=False but {reachable} are exported; the "
            "contract would under-report a usable feature"
        )


def test_cuda_error_is_importable_without_the_feature():
    """``except ak.CudaError`` must be writable against any build.

    The exception class is registered unconditionally by the native module, but
    it used to be re-exported only inside the feature-gated import of
    ``compile_cuda``/``CudaCompiledFn`` — so on the shipped wheel
    ``ak.CudaError`` raised ``AttributeError`` while ``alkahest.exceptions``
    held a *different, non-identical* class of the same name. Code written to
    catch the stub would not have caught a native raise on a CUDA build.
    """
    assert issubclass(ak.CudaError, ak.AlkahestError)
    assert "CudaError" in ak.__all__
    # The re-export must be the class the native module actually raises, not
    # the pure-Python stub that shadows it by name.
    assert ak.CudaError is ak.alkahest.CudaError


def test_cuda_names_are_in_dunder_all_exactly_when_they_exist():
    """``__all__`` is the documented surface; every name in it must resolve.

    The CUDA names are appended at runtime rather than written into the literal,
    precisely because they do not exist in a default build — so the invariant to
    pin is the equivalence, in both directions.
    """
    for name in CUDA_ENTRY_POINTS:
        assert (name in ak.__all__) == hasattr(ak, name), (
            f"{name}: __all__ membership and attribute existence disagree"
        )
        if name in ak.__all__:
            assert getattr(ak, name) is not None


@requires_cuda_build
def test_public_names_are_the_native_objects_not_copies():
    """The re-export must be the native object, so ``except ak.CudaError`` catches
    what the native module raises. A shadowing pure-Python stub with the same name
    would silently fail to catch anything."""
    native = ak.alkahest
    for name in CUDA_ENTRY_POINTS:
        assert getattr(ak, name) is getattr(native, name)


def test_gpu_tests_env_var_promises_a_device():
    """``ALKAHEST_GPU_TESTS=1`` asserts hardware; it must never degrade to a skip."""
    assert not GPU_PROMISED or DEVICE_SKIP_REASON is None


# ---------------------------------------------------------------------------
# 2. Compile-only — needs `--features cuda`, does not need a device
# ---------------------------------------------------------------------------


@requires_cuda_build
def test_compile_cuda_emits_sm_86_ptx(pool):
    x = pool.symbol("x")
    y = pool.symbol("y")
    expr = ak.sin(x) * ak.cos(y) + x * x

    fn = ak.compile_cuda(expr, [x, y])

    assert fn.n_inputs == 2
    ptx = fn.ptx
    assert isinstance(ptx, str)
    assert ptx
    # The production target is Ampere; the header is what ptxas/driver reads.
    assert ".target sm_86" in ptx
    assert ".address_size 64" in ptx
    assert ".version" in ptx
    # The runtime loads this exact entry point name (nvptx.rs: load_function).
    assert "alkahest_eval" in ptx
    assert "n_inputs=2" in repr(fn)


@requires_cuda_build
def test_unbound_symbol_raises_cuda_error(pool):
    """A symbol absent from ``inputs`` has no address in the kernel: E-CUDA-002."""
    x = pool.symbol("x")
    y = pool.symbol("y")

    with pytest.raises(ak.CudaError) as excinfo:
        ak.compile_cuda(x + y, [x])  # y never passed as an input

    exc = excinfo.value
    assert exc.code == "E-CUDA-002"
    assert "unbound symbol" in str(exc)
    assert exc.remediation


@requires_cuda_build
def test_cuda_error_is_catchable_as_alkahest_error(pool):
    """``CudaError`` must sit under the common base, or ``except AlkahestError``
    around a mixed CPU/GPU pipeline would let a GPU failure escape."""
    assert issubclass(ak.CudaError, ak.AlkahestError)

    x = pool.symbol("x")
    with pytest.raises(ak.AlkahestError) as excinfo:
        ak.compile_cuda(x + pool.symbol("z"), [x])

    assert isinstance(excinfo.value, ak.CudaError)
    assert excinfo.value.code.startswith("E-CUDA-")


@requires_cuda_build
def test_function_without_nvptx_lowering_is_refused_not_approximated(pool):
    """``atan`` has no libdevice mapping in the backend. Refusing is correct;
    quietly emitting something else would be a wrong-answer bug."""
    x = pool.symbol("x")

    with pytest.raises(ak.CudaError) as excinfo:
        ak.compile_cuda(ak.atan(x), [x])

    assert excinfo.value.code == "E-CUDA-002"
    assert "atan" in str(excinfo.value)


@requires_cuda_build
def test_call_batch_rejects_wrong_column_count(pool):
    x = pool.symbol("x")
    y = pool.symbol("y")
    fn = ak.compile_cuda(x + y, [x, y])

    with pytest.raises(ValueError, match="expected 2 input columns"):
        fn.call_batch([[1.0, 2.0]])


@requires_cuda_build
def test_call_batch_rejects_ragged_columns(pool):
    x = pool.symbol("x")
    y = pool.symbol("y")
    fn = ak.compile_cuda(x + y, [x, y])

    with pytest.raises(ValueError, match="same length"):
        fn.call_batch([[1.0, 2.0, 3.0], [1.0]])


# ---------------------------------------------------------------------------
# 3. Device — needs a CUDA build and a GPU that answers
#
# Every test here cross-checks the GPU against a CPU evaluation of the *same*
# expression. Agreement is the property; a launch that returns is not.
# ---------------------------------------------------------------------------


@requires_gpu
def test_polynomial_batch_matches_the_interpreter(pool):
    """65536 points spans many blocks, so a grid-stride bound error shows up —
    which is why index ``n - 1`` is always checked."""
    x = pool.symbol("x")
    expr = x**3 + (x * pool.integer(2)) * pool.integer(-1) + pool.integer(1)

    n = 1 << 16
    xs = [i * 1e-4 for i in range(n)]

    fn = ak.compile_cuda(expr, [x])
    got = fn.call_batch([xs])

    assert len(got) == n
    for i in _sample_indices(n, 128):
        want = ak.eval_expr(expr, {x: xs[i]})
        assert math.isclose(got[i], want, rel_tol=1e-12, abs_tol=1e-12), (
            f"GPU/CPU mismatch at i={i}: gpu={got[i]!r} cpu={want!r}"
        )


@requires_gpu
def test_transcendental_batch_matches_the_interpreter(pool):
    """libdevice ``__nv_sin``/``__nv_cos`` against the host libm: not bit-identical,
    but any disagreement beyond a few ulps is a lowering bug, not rounding."""
    x = pool.symbol("x")
    y = pool.symbol("y")
    expr = ak.sin(x) * ak.cos(y) + (x * x + y * y) * pool.rational(1, 100)

    n = 4096
    xs = [(i % 617) * 1e-2 for i in range(n)]
    ys = [(i % 331) * 3e-2 for i in range(n)]

    fn = ak.compile_cuda(expr, [x, y])
    got = fn.call_batch([xs, ys])

    assert len(got) == n
    for i in _sample_indices(n, 128):
        want = ak.eval_expr(expr, {x: xs[i], y: ys[i]})
        assert math.isclose(got[i], want, rel_tol=1e-10, abs_tol=1e-10), (
            f"GPU/CPU mismatch at i={i}: gpu={got[i]!r} cpu={want!r}"
        )


@requires_gpu
def test_matches_the_jit_compiled_cpu_function(pool):
    """Cross-check against the *compiled* CPU path, not only the interpreter:
    the two backends lower the same expression independently, so agreement
    between them is evidence neither one drifted."""
    x = pool.symbol("x")
    expr = ak.exp(x * pool.rational(-1, 4)) * ak.sin(x) + ak.sqrt(x + pool.integer(1))

    n = 1024
    xs = [i * 2e-3 for i in range(n)]

    with warnings.catch_warnings():
        # A build without a CPU JIT falls back to the interpreter with a warning;
        # the comparison is still valid, so do not let it turn into an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        cpu = ak.compile_expr(expr, [x])

    gpu = ak.compile_cuda(expr, [x])
    got = gpu.call_batch([xs])

    for i in _sample_indices(n, 64):
        want = cpu([xs[i]])
        assert math.isclose(got[i], want, rel_tol=1e-10, abs_tol=1e-10), (
            f"GPU/CPU-JIT mismatch at i={i}: gpu={got[i]!r} cpu={want!r}"
        )


@requires_gpu
def test_matches_numpy_eval_over_the_whole_batch(pool):
    """The vectorised CPU path, compared element-for-element rather than sampled."""
    np = pytest.importorskip("numpy")

    x = pool.symbol("x")
    y = pool.symbol("y")
    expr = x * x + y * y * pool.integer(3)

    n = 8192
    rng = np.random.default_rng(0)
    xs = rng.standard_normal(n, dtype=np.float64)
    ys = rng.standard_normal(n, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cpu = ak.compile_expr(expr, [x, y])
    want = np.asarray(ak.numpy_eval(cpu, xs, ys), dtype=np.float64)

    gpu = ak.compile_cuda(expr, [x, y])
    got = np.asarray(gpu.call_batch([xs.tolist(), ys.tolist()]), dtype=np.float64)

    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12), (
        f"max |GPU - CPU| = {float(np.max(np.abs(got - want)))}"
    )


@requires_gpu
def test_repeated_launches_are_deterministic(pool):
    """Same kernel, same inputs, same numbers — twice. A difference would mean a
    race or an uninitialised read that ``compute-sanitizer --tool racecheck``
    might or might not have provoked on the run that produced the green tick."""
    x = pool.symbol("x")
    expr = ak.sin(x) * ak.sin(x) + ak.cos(x) * ak.cos(x)

    xs = [i * 1e-3 for i in range(4096)]
    fn = ak.compile_cuda(expr, [x])

    first = fn.call_batch([xs])
    second = fn.call_batch([xs])

    assert first == second
    # sin² + cos² = 1 pointwise; the identity is a check on the kernel itself
    # that needs no CPU reference at all.
    for i in _sample_indices(len(xs), 64):
        assert abs(first[i] - 1.0) < 1e-12


@requires_gpu
def test_single_point_batch(pool):
    """One point is the degenerate grid: a block-size assumption that only holds
    for large N breaks here."""
    x = pool.symbol("x")
    expr = x * x + pool.integer(1)

    fn = ak.compile_cuda(expr, [x])
    got = fn.call_batch([[3.0]])

    assert len(got) == 1
    assert math.isclose(got[0], 10.0, rel_tol=1e-15)
