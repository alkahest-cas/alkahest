"""The `taylor_model` capability flag must equal what `bound_on_box` does.

`capabilities()["primitives"][i]["numeric_ball"]` is *not* the flag that
governs the validated-bounds subsystem, and used to be the only per-function
coverage bit exposed. Ball arithmetic is pointwise; a Taylor model needs a
rule with a rigorous remainder, written per function in
`alkahest-core/src/validated/taylor.rs`. Ten primitives — `erf`, `erfc`,
`bessel_j0`, `bessel_j1`, `digamma`, `lambert_w`, `acosh`, `asinh`, `floor`,
`ceil` — have real Arb ball arithmetic and no Taylor-model rule, so
`bound_on_box` refuses them with `E-VALIDATED-001` while `numeric_ball` says
``True``. That boundary was correct at runtime and invisible beforehand.

The whole point of these tests is that the *new* flag cannot repeat that
mistake. `taylor_model` is derived by running the real evaluator (see
`alkahest-core/src/primitive/taylor_support.rs`), never from a list, and
``test_taylor_model_flag_agrees_with_bound_on_box`` re-derives it here the
only other way there is — by calling `bound_on_box` on every registered
primitive — and fails if the two ever disagree.
"""

from __future__ import annotations

import alkahest as ak
import pytest

# Small budget: these tests ask *which error* comes back, never how tight the
# bound is, so there is no reason to pay for convergence.
_OPTS = {"order": 2, "prec": 64, "tol": 1e-3, "max_subdivisions": 4}
_MAX_ARITY = 3
_UNSUPPORTED = "E-VALIDATED-001"


def _primitive_rows():
    return ak.capabilities()["primitives"]


def _bound_on_box_accepts(name: str, arity: int) -> bool:
    """Does `bound_on_box` get past dispatch for ``name`` at this arity?

    ``True`` means "not refused with ``E-VALIDATED-001``". Every other
    outcome — success, a domain violation, a non-finite enclosure — means the
    evaluator *had* a rule and something about this particular box stopped it,
    which is a different question and not what the flag claims.
    """
    pool = ak.ExprPool()
    args = [pool.symbol(f"x{i}") for i in range(arity)]
    call = pool.func(name, args)
    box = [(a, 0.25, 0.5) for a in args]
    try:
        ak.bound_on_box(call, box, **_OPTS)
    except ak.ValidatedError as exc:
        return exc.code != _UNSUPPORTED
    except Exception:
        # Any other failure still means the evaluator dispatched on the name.
        return True
    return True


@pytest.mark.parametrize("row", _primitive_rows(), ids=lambda row: row["name"])
def test_taylor_model_flag_agrees_with_bound_on_box(row):
    """The flag and the subsystem it describes, cross-checked per primitive.

    This is the guard that stops the two drifting apart. If it fails, either
    a Taylor-model rule was added/removed without the flag following (it
    cannot be — the flag is derived) or the derivation itself is asking the
    wrong question.
    """
    name = row["name"]
    observed = any(_bound_on_box_accepts(name, n) for n in range(1, _MAX_ARITY + 1))
    assert row["taylor_model"] is observed, (
        f"capabilities() says taylor_model={row['taylor_model']} for `{name}`, "
        f"but bound_on_box {'accepts' if observed else 'refuses ' + _UNSUPPORTED + ' for'} it"
    )


def test_the_flag_is_not_a_restatement_of_numeric_ball():
    """The bug this fixes: `numeric_ball` was read as the coverage flag.

    `numeric_ball` is *not* wrong for these eleven — they really do have Arb
    ball arithmetic (`alkahest-core/src/primitive/taylor_support.rs` pins that
    in Rust). It answers a different question, which is why reading it as the
    validated-bounds coverage bit cost a whole workload.

    `atanh` joined the set when the capability probe stopped testing ball
    kernels at `1.0` only: its domain is the open interval `(-1, 1)`, so it
    declined the sole probe point and lost a bit it had earned, while keeping
    `numeric_f64` because that probe already tried `0.5`.
    """
    rows = {row["name"]: row for row in _primitive_rows()}
    ball_only = {
        name for name, row in rows.items() if row["numeric_ball"] and not row["taylor_model"]
    }
    assert ball_only == {
        "acosh",
        "asinh",
        "atanh",
        "bessel_j0",
        "bessel_j1",
        "ceil",
        "digamma",
        "erf",
        "erfc",
        "floor",
        "lambert_w",
    }


def test_supported_set_is_the_elementary_fragment():
    """A pin on the boundary as it stands, so a change to it is deliberate.

    Widening this set is a feature (add the rule, then add the name here);
    narrowing it silently would be a regression an agent's plan depends on.
    """
    supported = {row["name"] for row in _primitive_rows() if row["taylor_model"]}
    assert supported == {
        "abs",
        "acos",
        "asin",
        "atan",
        "cos",
        "cosh",
        "exp",
        "log",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }


def test_registry_capabilities_and_the_table_agree():
    reg = ak.PrimitiveRegistry.default_registry()
    for row in _primitive_rows():
        assert reg.capabilities(row["name"])["taylor_model"] is row["taylor_model"]


# ---------------------------------------------------------------------------
# bounds_supported — the same question for a whole expression
# ---------------------------------------------------------------------------


def test_bounds_supported_matches_bound_on_box_on_composite_expressions():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    cases = [
        ak.sin(x) * ak.exp(x) + x * x,
        x - x,
        ak.sqrt(x + 1) / (x + 2),
        ak.bessel_j0(x),
        x * ak.erf(x),
        ak.sin(x) + ak.digamma(x),
        ak.atan2(x, y),
        ak.gamma(x),
        ak.floor(x) + ak.ceil(x),
        ak.tanh(x * y) - ak.log(x + 3),
    ]
    box = [(x, 0.25, 0.5), (y, 0.25, 0.5)]
    for expr in cases:
        answer = ak.bounds_supported(expr)
        try:
            ak.bound_on_box(expr, box, **_OPTS)
            refused = False
        except ak.ValidatedError as exc:
            refused = exc.code == _UNSUPPORTED
        assert bool(answer) is (not refused), f"{expr}: {answer.detail}"


def test_bounds_supported_names_every_blocking_function():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    answer = ak.bounds_supported(ak.bessel_j0(x) + ak.erf(x) + ak.sin(x))

    assert not answer
    assert answer.supported is False
    assert answer.functions == ["bessel_j0", "erf"]
    assert "bessel_j0" in answer.blocker
    assert _UNSUPPORTED in answer.detail


def test_bounds_supported_is_truthy_and_quiet_when_everything_is_covered():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    answer = ak.bounds_supported(ak.sin(x) ** 2 + ak.cosh(x))

    assert answer
    assert answer.blocker is None
    assert answer.functions == []
    assert "E-VALIDATED-001" in answer.detail


def test_a_domain_violation_is_not_reported_as_unsupported():
    """`log` has a rule; `[-2, -1]` is just a bad box.

    Reporting the refusal as "unsupported" would send a planner off a route
    that works everywhere else, which is the failure this predicate exists to
    prevent — in the other direction.
    """
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.log(x)

    assert ak.bounds_supported(f)
    with pytest.raises(ak.ValidatedError) as excinfo:
        ak.bound_on_box(f, [(x, -2.0, -1.0)], **_OPTS)
    assert excinfo.value.code != _UNSUPPORTED


def test_arity_is_part_of_the_question():
    """`atan2` is refused for its arity, not for being `atan2`."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    answer = ak.bounds_supported(pool.func("atan2", [x, x]))

    assert not answer
    assert "2 arguments" in answer.blocker


def test_constant_expressions_are_classified_too():
    pool = ak.ExprPool()
    assert ak.bounds_supported(ak.sin(pool.integer(2)))
    assert not ak.bounds_supported(ak.erf(pool.integer(2)))


def test_bounds_support_surface():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    for name in ("bounds_supported", "BoundsSupport"):
        assert name in ak.__all__, name

    answer = ak.bounds_supported(ak.erf(x))
    assert isinstance(answer, ak.BoundsSupport)
    assert "BoundsSupport(" in repr(answer)
    assert answer.as_dict() == {
        "supported": False,
        "blocker": answer.blocker,
        "functions": ["erf"],
        "detail": answer.detail,
    }
