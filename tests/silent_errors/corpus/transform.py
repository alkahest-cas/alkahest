"""Silent-error cases for transform.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import math
from typing import Callable

import alkahest as ak
import alkahest.experimental as ex
from contracts import Case, RefusesOr, Returns

from ._shared import PI, POOL, _int

#: Symbols the transform cases share.  `T`/`S` are the Laplace pair, `XX`/`XI`
#: the Fourier pair, `NN`/`ZZ` the Z pair; `PI` is the interned π the Fourier
#: table emits and has to be bound before anything can be evaluated.
T = POOL.symbol("t")
S = POOL.symbol("s")
XX = POOL.symbol("xspace")
XI = POOL.symbol("xi")
NN = POOL.symbol("nidx")
ZZ = POOL.symbol("z")


def _unconditional(build: Callable[[], ak.Expr], env: dict) -> Callable[[], float]:
    """Answer = a transform's value at *env*, but only if it was unconditional.

    ``experimental.transform_side_conditions()`` reports the hypotheses the call
    that just ran had to assume about a symbolic parameter.  A number returned
    *under an undischarged hypothesis* is not an unconditional answer — the
    caller was told, which is exactly the distinction this gate measures — so it
    is surfaced as a refusal rather than scored as a stated value.

    ``PI`` is bound for the Fourier table, which emits the interned π symbol.
    """

    def op() -> float:
        out = build()
        conds = ex.transform_side_conditions()
        if conds:
            raise ValueError(f"answer holds only under {conds}")
        return float(ak.eval_expr(out, {**env, PI: math.pi}))

    return op


CASES: list[Case] = [
    # -- transforms: the unilateral shift is a hypothesis, not a fact ---------
    Case(
        id="transform_laplace_step_advanced_past_the_origin",
        subsystem="transform",
        statement="L{θ(t+a)}(s) at a = 1.5 is 1/s, not e^{a s}/s",
        op=_unconditional(
            lambda: ex.laplace_transform(ex.heaviside(T + POOL.symbol("shift_a")), T, S),
            {POOL.symbol("shift_a"): 1.5, S: 2.5},
        ),
        contract=RefusesOr(0.4),
        verified_by=(
            "By hand from the definition: for a > 0, θ(t+a) ≡ 1 on the whole range "
            "[0,∞) the unilateral integral sees, so L{θ(t+a)} = ∫₀^∞ e^{−st} dt = 1/s "
            "= 0.4 at s = 2.5.  The second-shift rule e^{−a s}G(s) is the transform "
            "only for a shift a ≥ 0, and θ(t+a) is θ(t−(−a))."
        ),
        note=(
            "Was a silent error: the rule fired on the negative shift and returned "
            "e^{1.5·2.5}/2.5 = 17.0084 — a factor of 42 wrong, and growing with a — "
            "with an empty side-condition list.  The literal form θ(t+1) was already "
            "refused; only the symbolic shift slipped through.  It now passes by "
            "reporting `a ≤ 0`, which the caller can check and reject."
        ),
    ),
    Case(
        id="transform_laplace_impulse_before_the_origin",
        subsystem="transform",
        statement="L{δ(t+a)}(s) at a = 1.5 is 0 — the impulse is outside [0,∞)",
        op=_unconditional(
            lambda: ex.laplace_transform(ex.dirac_delta(T + POOL.symbol("shift_b")), T, S),
            {POOL.symbol("shift_b"): 1.5, S: 2.5},
        ),
        contract=RefusesOr(0.0),
        verified_by=(
            "∫₀^∞ δ(t+1.5)·e^{−st} dt = 0: the sifting point t = −1.5 is not in the "
            "domain of integration, so the unilateral transform of an impulse placed "
            "before the origin is identically zero for every s."
        ),
        note=(
            "Same defect as the step case, one table row over: the answer was "
            "e^{1.5·2.5} = 42.52 for a quantity that is 0.  Reported as `a ≤ 0` now."
        ),
    ),
    Case(
        id="transform_inverse_laplace_of_an_advance",
        subsystem="transform",
        statement="no causal f has L{f} = e^{+2s}/(s+1)",
        op=_unconditional(
            lambda: ex.inverse_laplace_transform(ak.exp(2 * S) / (S + 1), S, T),
            {T: 1.0},
        ),
        contract=RefusesOr(),
        verified_by=(
            "A unilateral transform is bounded as Re s → +∞ (dominated convergence "
            "on ∫₀^∞ f e^{−st}), and e^{2s}/(s+1) grows without bound there, so it is "
            "the transform of no causal function at all.  Numerically: F(1.7) = "
            "11.0978, while the transform of the shift rule's answer θ(t+2)e^{−2−t} "
            "is e^{−2}/(s+1) = 0.0501 — the two are not the same function."
        ),
        note=(
            "Was worse than a wrong answer: it came back with the side condition "
            "`−2 ∈ NonNegative`, i.e. a hypothesis that can never be discharged, "
            "attached to a value that is wrong under every reading.  A refutation is "
            "not a hypothesis and now refuses."
        ),
    ),
    # -- transforms: Fourier positivity selects the branch --------------------
    Case(
        id="transform_fourier_lorentzian_negative_amplitude",
        subsystem="transform",
        statement="F{−2/(1+4π²x²)}(ξ) = −e^{−|ξ|}, not e^{+|ξ|}",
        op=_unconditional(
            lambda: ex.fourier_transform(_int(-2) / (1 + 4 * PI**2 * XX**2), XX, XI),
            {XI: 0.3},
        ),
        contract=RefusesOr(-0.7408182206817179),
        verified_by=(
            "∫ c/(c₀+4π²x²)·e^{−2πiξx} dx = c/(2√c₀)·e^{−√c₀|ξ|} (residues at "
            "x = ±i√c₀/2π); at c = −2, c₀ = 1, ξ = 0.3 that is −e^{−0.3} = "
            "−0.7408182206817179.  mpmath quadosc on the same integrand agrees to "
            "10 significant figures."
        ),
        note=(
            "The trap is that c₀ = a² is satisfied by a and by −a alike, so reading "
            "the rate off the numerator picks a branch rather than deriving one.  "
            "alkahest returned e^{+|ξ|} = +1.3499: wrong sign, and *growing* where "
            "the true transform decays.  Input is all-literal — no parameter, "
            "nothing unknown, nothing to report — so the only honest options were "
            "the right number or a refusal."
        ),
    ),
    Case(
        id="transform_fourier_growing_two_sided_exponential",
        subsystem="transform",
        statement="e^{+3|x|} has no Fourier transform — the integral diverges",
        op=_unconditional(
            lambda: ex.fourier_transform(ak.exp(3 * ak.abs(XX)), XX, XI),
            {XI: 0.3},
        ),
        contract=RefusesOr(),
        verified_by=(
            "|e^{3|x|}·e^{−2πiξx}| = e^{3|x|} ≥ 1 for every x, so ∫_{−∞}^{∞} of it "
            "diverges to +∞ at every ξ.  There is no value to return."
        ),
        note=(
            "alkahest applied the a > 0 table row with a = −3 and returned the "
            "finite −6/(9+4π²ξ²) = −0.478 at ξ = 0.3.  The Gaussian row already "
            "rejected a literal non-positive curvature; this row did not, so the "
            "refusal was inconsistent as well as absent."
        ),
    ),
    # -- transforms: controls ------------------------------------------------
    Case(
        id="transform_control_laplace_step_delayed",
        subsystem="transform",
        statement="L{θ(t−2)}(s) = e^{−2s}/s — the nearest convergent neighbour",
        op=_unconditional(lambda: ex.laplace_transform(ex.heaviside(T - 2), T, S), {S: 2.5}),
        contract=Returns(0.002695178799634187),
        verified_by=(
            "∫₂^∞ e^{−2.5t} dt = e^{−5}/2.5 = 0.002695178799634187 (math.exp(-5)/2.5). "
            "The delay is a genuine delay, so the second-shift rule applies with no "
            "hypothesis at all."
        ),
        note=(
            "The control for the two advanced-shift traps: a gate made only of "
            "refusals is passed by a library that refuses every shifted step."
        ),
    ),
    Case(
        id="transform_control_fourier_lorentzian_positive_amplitude",
        subsystem="transform",
        statement="F{2/(1+4π²x²)}(ξ) = e^{−|ξ|}",
        op=_unconditional(
            lambda: ex.fourier_transform(_int(2) / (1 + 4 * PI**2 * XX**2), XX, XI),
            {XI: 0.3},
        ),
        contract=Returns(0.7408182206817179),
        verified_by=(
            "Same residue computation as the negative-amplitude case with c = +2: "
            "e^{−0.3} = 0.7408182206817179 (math.exp(-0.3)), cross-checked with "
            "mpmath quadosc."
        ),
        note="Control: the sign fix must not turn the whole Lorentzian row into a refusal.",
    ),
    Case(
        id="transform_control_fourier_decaying_two_sided_exponential",
        subsystem="transform",
        statement="F{e^{−3|x|}}(ξ) = 6/(9+4π²ξ²)",
        op=_unconditional(lambda: ex.fourier_transform(ak.exp(-3 * ak.abs(XX)), XX, XI), {XI: 0.3}),
        contract=Returns(0.4779712002165985),
        verified_by=(
            "∫ e^{−3|x|}e^{−2πiξx} dx = 2a/(a²+4π²ξ²) with a = 3; at ξ = 0.3 that "
            "is 6/(9 + 4π²·0.09) = 0.4779712002165985 in double precision."
        ),
        note="Control for the divergent-rate refusal.",
    ),
    Case(
        id="transform_control_fourier_gaussian_is_self_dual",
        subsystem="transform",
        statement="F{e^{−πx²}}(ξ) = e^{−πξ²} in the unitary ordinary-frequency convention",
        op=_unconditional(lambda: ex.fourier_transform(ak.exp(-PI * XX**2), XX, XI), {XI: 0.5}),
        contract=Returns(0.45593812776599624),
        verified_by=(
            "The self-dual Gaussian of the unitary ordinary-frequency convention "
            "∫f(x)e^{−2πiξx}dx; at ξ = 0.5 the value is e^{−π/4} = "
            "0.45593812776599624 (math.exp(-math.pi/4))."
        ),
        note=(
            "The convention control: a 2π slip in the kernel moves this number and "
            "nothing else in the corpus would notice."
        ),
    ),
    Case(
        id="transform_control_fourier_roundtrip_recovers_the_input",
        subsystem="transform",
        statement="F⁻¹{F{e^{−x²}}}(x) = e^{−x²}",
        op=_unconditional(
            lambda: ex.inverse_fourier_transform(
                ex.fourier_transform(ak.exp(-(XX**2)), XX, XI), XI, XX
            ),
            {XX: 0.7},
        ),
        contract=Returns(0.6126263941844161),
        verified_by=(
            "e^{−0.49} = 0.6126263941844161 (math.exp(-0.49)).  The forward and "
            "inverse kernels differ only in the sign of the exponent, so a "
            "convention mismatch between the two directions leaves a stray 2π (or "
            "1/2π) here and nowhere else."
        ),
        note="Control against a forward/inverse convention mismatch.",
    ),
    Case(
        id="transform_control_inverse_laplace_delayed_sine",
        subsystem="transform",
        statement="L⁻¹{e^{−2s}/(s²+1)}(t) = θ(t−2)·sin(t−2)",
        op=_unconditional(
            lambda: ex.inverse_laplace_transform(ak.exp(-2 * S) / (S**2 + 1), S, T),
            {T: 3.0},
        ),
        contract=Returns(0.8414709848078965),
        verified_by=(
            "The second-shift rule with a genuine delay: at t = 3 the value is "
            "sin(1) = 0.8414709848078965 (math.sin(1))."
        ),
        note="Control for `transform_inverse_laplace_of_an_advance`.",
    ),
    # -- transforms: the ℚ(params) decomposition reports its confluences ------
    Case(
        id="transform_inverse_laplace_bateman_at_the_confluent_point",
        subsystem="transform",
        statement="L⁻¹{1/((s+ka)(s+ke))} at ka = ke is t·e^{−ka t}, not 0/0",
        op=_unconditional(
            lambda: ex.inverse_laplace_transform(
                1 / ((S + POOL.symbol("ka")) * (S + POOL.symbol("ke"))), S, T
            ),
            {POOL.symbol("ka"): 1.3, POOL.symbol("ke"): 1.3, T: 1.0},
        ),
        contract=RefusesOr(0.2725317930340126),
        verified_by=(
            "At ka = ke the input is the ordinary 1/(s+ka)², whose inverse is "
            "t·e^{−ka t}; at ka = ke = 1.3, t = 1 that is e^{−1.3} = "
            "0.2725317930340126 (math.exp(-1.3)).  The generic decomposition "
            "(e^{−ka t} − e^{−ke t})/(ke − ka) is 0/0 there — a true identity of "
            "rational functions that is not an identity of values."
        ),
        note=(
            "The case the ℚ(params) partial-fraction path exists for.  It passes "
            "because `ka − ke ≠ 0` is reported, not because the arithmetic happens "
            "to blow up: a caller that reads the hypothesis knows to take the "
            "confluent branch.  The E-EVAL-009 that eval_expr would raise is a "
            "second, weaker net under it."
        ),
    ),
    Case(
        id="transform_inverse_laplace_second_order_overdamped",
        subsystem="transform",
        statement="L⁻¹{1/(s²+2ζωs+ω²)} at ζ = 2 is hyperbolic, not the printed sine",
        op=_unconditional(
            lambda: ex.inverse_laplace_transform(
                1
                / (
                    S**2
                    + 2 * POOL.symbol("zeta") * POOL.symbol("omega") * S
                    + POOL.symbol("omega") ** 2
                ),
                S,
                T,
            ),
            {POOL.symbol("zeta"): 2.0, POOL.symbol("omega"): 1.0, T: 1.0},
        ),
        contract=RefusesOr(0.2139091302602793),
        verified_by=(
            "At ζ = 2, ω = 1 the characteristic roots are real: −2 ± √3.  The "
            "inverse is (e^{r₁t} − e^{r₂t})/(r₁ − r₂), which at t = 1 is "
            "(exp(-2+3**0.5) - exp(-2-3**0.5)) / (2*3**0.5) = 0.2139091302602793.  "
            "The under-damped closed form K·e^{−ζωt}sin(ω√(1−ζ²)t)/(ω√(1−ζ²)) needs "
            "ω²(1−ζ²) > 0, which is false here."
        ),
        note=(
            "Passes because ω²(1−ζ²) > 0 is reported alongside ω ≠ 0 and ζ ≠ ±1.  "
            "Without that report the printed sine is an imaginary-argument "
            "expression handed back as if it were the real answer."
        ),
    ),
    Case(
        id="transform_inverse_z_repeated_pole_confluence",
        subsystem="transform",
        statement="Z⁻¹{z/((z−a)(z−b))} at a = b is n·a^{n−1}, not (aⁿ−bⁿ)/(a−b)",
        op=_unconditional(
            lambda: ex.inverse_z_transform(
                ZZ / ((ZZ - POOL.symbol("za")) * (ZZ - POOL.symbol("zb"))), ZZ, NN
            ),
            {POOL.symbol("za"): 0.5, POOL.symbol("zb"): 0.5, NN: 3.0},
        ),
        contract=RefusesOr(0.75),
        verified_by=(
            "At a = b the transform is z/(z−a)², whose inverse is n·a^{n−1}; at "
            "a = 0.5, n = 3 that is 3·0.25 = 0.75.  Read off the geometric-series "
            "expansion of z/(z−a)² = Σ n a^{n−1} z^{−n}, independently of any "
            "partial-fraction identity."
        ),
        note=(
            "The Z-side twin of the Bateman confluence — same ℚ(params) machinery, "
            "same reported `a − b ≠ 0`."
        ),
    ),
    Case(
        id="transform_inverse_laplace_improper_rational_refuses",
        subsystem="transform",
        statement="L⁻¹{s²/(s²+1)} is δ(t) − sin t, not an ordinary function",
        op=_unconditional(lambda: ex.inverse_laplace_transform(S**2 / (S**2 + 1), S, T), {T: 1.0}),
        contract=RefusesOr(),
        verified_by=(
            "s²/(s²+1) = 1 − 1/(s²+1), and L⁻¹{1} is the Dirac δ, a distribution.  "
            "No locally integrable function has this transform, so every finite "
            "value at t = 1 is wrong — including the −sin(1) = −0.841 that dropping "
            "the polynomial part would give."
        ),
        note=(
            "Control for the refusal the changelog claims: the polynomial part is "
            "declined rather than quietly discarded, which would have produced a "
            "clean, plausible, wrong function."
        ),
    ),
]
