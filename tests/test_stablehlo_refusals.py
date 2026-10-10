"""``to_stablehlo`` returns a complete module or raises a coded error — never ``''``.

W10 (report 10-9): ``to_stablehlo(erf(t), [t])`` returned ``''`` with no error,
and so did ``lambert_w``, ``max`` and every other function the emitter did not
lower. A caller writing the result to a file got an empty "module" and no
diagnostic. Now the functions StableHLO can express exactly are lowered (``max``
/ ``min``, ``sign``, ``floor`` / ``ceil`` / ``round``, ``atan2``, ``asin`` /
``acos`` / ``atan`` / ``atanh``, ``heaviside``, ``Piecewise`` …) and the rest
raise ``E-STABLEHLO-001`` naming the function.

The numeric check runs the emitted module twice over: through a tiny reference
interpreter of the ops the emitter uses (always), and through IREE when
``iree-base-compiler`` / ``iree-base-runtime`` are installed.
"""

from __future__ import annotations

import math
import re

import alkahest as ak
import pytest

pool = ak.ExprPool()
t = pool.symbol("t")
u = pool.symbol("u")

UNARY_SUPPORTED = [
    ("sin", ak.sin),
    ("cos", ak.cos),
    ("tan", ak.tan),
    ("exp", ak.exp),
    ("log", ak.log),
    ("sqrt", ak.sqrt),
    ("tanh", ak.tanh),
    ("sinh", ak.sinh),
    ("cosh", ak.cosh),
    ("abs", ak.abs),
    ("sign", ak.sign),
    ("floor", ak.floor),
    ("ceil", ak.ceil),
    ("round", ak.round),
    ("asin", ak.asin),
    ("acos", ak.acos),
    ("atan", ak.atan),
    ("atanh", ak.atanh),
]

POINTS = [-3.7, -1.0, -0.999, -0.5, -1e-9, 0.0, 1e-9, 0.25, 0.5, 0.999, 1.0, 2.5]


def _supported_cases():
    cases = [(name, f(t), [t]) for name, f in UNARY_SUPPORTED]
    cases.append(("heaviside", pool.func("heaviside", [t]), [t]))

    cases.append(("atan2", pool.func("atan2", [t, u]), [t, u]))
    cases.append(("max", pool.func("max", [t, u]), [t, u]))
    cases.append(("min", pool.func("min", [t, u]), [t, u]))
    cases.append(("sin*t", ak.sin(t) * t, [t]))
    cases.append(
        (
            "piecewise",
            ak.piecewise(
                [(pool.lt(t, pool.integer(0)), -t), (pool.lt(t, pool.integer(1)), t**2)],
                ak.sin(t),
            ),
            [t],
        )
    )
    return cases


SUPPORTED = _supported_cases()


# ---------------------------------------------------------------------------
# A reference interpreter for the emitted text
# ---------------------------------------------------------------------------

_LINE = re.compile(r"^(%v\d+) = stablehlo\.(\w+) (.*?) : (.*)$")


def _sign(x):
    if math.isnan(x) or x == 0.0:
        return x
    return math.copysign(1.0, x)


def _safe(f):
    def g(*a):
        try:
            return f(*a)
        except (ValueError, OverflowError):
            return math.nan

    return g


_UNARY = {
    "negate": lambda x: -x,
    "abs": abs,
    "sign": _sign,
    "floor": math.floor,
    "ceil": math.ceil,
    "round_nearest_afz": lambda x: math.copysign(math.floor(abs(x) + 0.5), x),
    "sine": math.sin,
    "cosine": math.cos,
    "tanh": math.tanh,
    "exponential": _safe(math.exp),
    "exponential_minus_one": _safe(math.expm1),
    "log": _safe(lambda x: -math.inf if x == 0 else math.log(x)),
    "log_plus_one": _safe(lambda x: -math.inf if x == -1 else math.log1p(x)),
    "sqrt": _safe(math.sqrt),
}


def _div(a, b):
    if b == 0:
        if a == 0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


_BINARY = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": _div,
    "power": _safe(math.pow),
    "atan2": math.atan2,
    "maximum": lambda a, b: math.nan if math.isnan(a) or math.isnan(b) else max(a, b),
    "minimum": lambda a, b: math.nan if math.isnan(a) or math.isnan(b) else min(a, b),
}

_CMP = {
    "LT": lambda a, b: a < b,
    "LE": lambda a, b: a <= b,
    "GT": lambda a, b: a > b,
    "GE": lambda a, b: a >= b,
    "EQ": lambda a, b: a == b,
    "NE": lambda a, b: a != b,
}


def _run(src: str, args: list[float]) -> float:
    """Execute an emitted module; asserts it is well formed along the way."""
    lines = [ln.strip() for ln in src.strip().splitlines()]
    assert lines[0] == "module {"
    assert re.fullmatch(r"func\.func @\w+\((.*)\) -> tensor<f64> \{", lines[1]), lines[1]
    assert lines[-2:] == ["}", "}"]
    env = {f"%arg{i}": a for i, a in enumerate(args)}
    for ln in lines[2:-2]:
        if ln.startswith("return "):
            v, ty = ln[len("return ") :].split(" : ")
            assert ty == "tensor<f64>"
            return env[v]
        m = _LINE.match(ln)
        assert m, f"not a stablehlo op: {ln}"
        lhs, op, operands, _ty = m.groups()
        assert lhs not in env, f"{lhs} redefined"
        if op == "constant":
            lit = re.fullmatch(r"dense<(.*)>", operands).group(1)
            env[lhs] = {"true": True, "false": False}.get(lit)
            if env[lhs] is None:
                assert "." in lit, lit
                env[lhs] = float(lit)
            continue
        parts = operands.split(", ")
        if op == "compare":
            env[lhs] = _CMP[parts[0]](env[parts[1]], env[parts[2]])
            continue
        vals = [env[p] for p in parts]  # KeyError = use before definition
        if op == "select":
            env[lhs] = vals[1] if vals[0] else vals[2]
        elif op == "and":
            env[lhs] = vals[0] and vals[1]
        elif op == "or":
            env[lhs] = vals[0] or vals[1]
        elif op == "not":
            env[lhs] = not vals[0]
        elif len(vals) == 1:
            env[lhs] = float(_UNARY[op](vals[0]))
        else:
            env[lhs] = float(_BINARY[op](*vals))
    raise AssertionError("no return")


def _reference(expr, inputs, args):
    try:
        return ak.eval_expr(expr, dict(zip(inputs, args)))
    except (ak.AlkahestError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _close(got, want, rel=1e-12):
    if math.isnan(want):
        return math.isnan(got)
    if math.isinf(want):
        return got == want
    return abs(got - want) <= rel * max(1.0, abs(want))


def _arg_tuples(n):
    if n == 1:
        return [(p,) for p in POINTS]
    return [(a, b) for a in (-2.0, 0.0, 1.5) for b in (-1.0, 0.0, 3.0)]


# ---------------------------------------------------------------------------
# The report's repro
# ---------------------------------------------------------------------------


def test_repro_supported_expression_still_emits_a_module():
    src = ak.to_stablehlo(ak.sin(t) * t, [t])
    assert src.startswith("module {")
    assert "func.func @alkahest_fn" in src


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("erf", lambda: ak.erf(t)),
        ("lambert_w", lambda: ak.lambert_w(t)),
        ("asinh", lambda: ak.asinh(t)),
        ("gamma", lambda: pool.func("gamma", [t])),
        ("erfc", lambda: pool.func("erfc", [t])),
        ("digamma", lambda: pool.func("digamma", [t])),
        ("acosh", lambda: pool.func("acosh", [t])),
        ("f", lambda: pool.func("f", [t])),
        ("erf", lambda: ak.sin(t) + ak.erf(t) * t),  # nested: still nothing emitted
    ],
)
def test_unsupported_function_raises_a_coded_error(name, build):
    with pytest.raises(ak.AlkahestError) as info:
        ak.to_stablehlo(build(), [t])
    assert info.value.code == "E-STABLEHLO-001"
    assert name in str(info.value)
    assert info.value.remediation


def test_other_refusals_are_coded():
    with pytest.raises(ak.AlkahestError) as info:
        ak.to_stablehlo(t + u, [t])
    assert info.value.code == "E-STABLEHLO-003"
    assert "u" in str(info.value)

    with pytest.raises(ak.AlkahestError) as info:
        ak.to_stablehlo(t, [t], fn_name="not a name")
    assert info.value.code == "E-STABLEHLO-005"

    with pytest.raises(ak.AlkahestError) as info:
        ak.to_stablehlo(t + pool.float(math.inf), [t])
    assert info.value.code == "E-STABLEHLO-004"


# ---------------------------------------------------------------------------
# Every supported function: well formed and numerically right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expr", "inputs"), SUPPORTED, ids=[c[0] for c in SUPPORTED])
def test_supported_function_emits_a_correct_module(name, expr, inputs):
    src = ak.to_stablehlo(expr, inputs)
    assert src, f"{name}: empty module"
    assert "chlo." not in src
    assert "mhlo." not in src
    checked = 0
    for args in _arg_tuples(len(inputs)):
        want = _reference(expr, inputs, args)
        if want is None:
            continue
        got = _run(src, list(args))
        assert _close(got, want), f"{name}{args}: module {got}, eval_expr {want}\n{src}"
        checked += 1
    assert checked >= 3, name


def test_ln_lowers_as_the_natural_log():
    src = ak.to_stablehlo(pool.func("ln", [t]), [t])
    assert src == ak.to_stablehlo(ak.log(t), [t])


def test_sinh_has_no_cancellation_near_zero():
    src = ak.to_stablehlo(ak.sinh(t), [t])
    for p in (1e-9, -3e-12):
        assert _close(_run(src, [p]), math.sinh(p), rel=1e-15)


# ---------------------------------------------------------------------------
# Executed by a real StableHLO compiler, when one is installed
# ---------------------------------------------------------------------------


def _iree():
    compiler = pytest.importorskip("iree.compiler")
    runtime = pytest.importorskip("iree.runtime")
    np = pytest.importorskip("numpy")
    return compiler, runtime, np


@pytest.mark.parametrize(("name", "expr", "inputs"), SUPPORTED, ids=[c[0] for c in SUPPORTED])
def test_supported_function_runs_under_iree(name, expr, inputs):
    compiler, runtime, np = _iree()
    src = ak.to_stablehlo(expr, inputs)
    vmfb = compiler.compile_str(
        src,
        target_backends=["llvm-cpu"],
        input_type="stablehlo",
        # IREE's CPU backend has no f64 transcendental runtime, so it demotes
        # f64 to f32 (its default); the comparison below is at f32 accuracy.
        extra_args=["--iree-llvmcpu-target-cpu=generic"],
    )
    config = runtime.Config("local-task")
    ctx = runtime.SystemContext(config=config)
    ctx.add_vm_module(runtime.VmModule.copy_buffer(ctx.instance, vmfb))
    fn = ctx.modules.module["alkahest_fn"]
    checked = 0
    for args in _arg_tuples(len(inputs)):
        # Evaluate the reference at the f32-rounded point, so the input
        # rounding is not counted against the module.
        args32 = [np.float32(a) for a in args]
        want = _reference(expr, inputs, [float(a) for a in args32])
        if want is None or not math.isfinite(want):
            continue
        if name == "atan2" and args == (0.0, 0.0):
            # The StableHLO spec gives atan2(±0, +0) = ±0 (IEEE); IREE's
            # llvm-cpu atan2 returns NaN there. An executor deviation, not
            # the module's: the reference interpreter above checks this point.
            continue
        got = float(np.asarray(fn(*[np.array(a, dtype=np.float32) for a in args32])))
        assert abs(got - want) <= 1e-5 * max(1.0, abs(want)), (
            f"{name}{args}: iree {got}, eval_expr {want}"
        )
        checked += 1
    assert checked >= 3, name


# ---------------------------------------------------------------------------
# Sibling: to_jax on JAX >= 0.6
# ---------------------------------------------------------------------------


def test_to_jax_builds_and_evaluates_on_current_jax():
    """``jax.core.Primitive`` is gone in JAX 0.6; ``to_jax`` raised AttributeError."""
    pytest.importorskip("jax")
    np = pytest.importorskip("numpy")
    import jax.numpy as jnp

    f = ak.to_jax(ak.sin(t) * t, [t])
    xs = jnp.array([0.0, 0.5, 2.0])
    np.testing.assert_allclose(np.asarray(f(xs)), np.sin([0.0, 0.5, 2.0]) * [0.0, 0.5, 2.0])
