"""Silent-error cases for vector.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import math

import alkahest as ak
import alkahest.experimental as ex
from contracts import Case, Raises, Returns

from ._shared import POOL

#: Cartesian `(x, y, z)`.
CX = POOL.symbol("vcx")
CY = POOL.symbol("vcy")
CZ = POOL.symbol("vcz")
#: Cylindrical `(ρ, φ, z)`.
RHO = POOL.symbol("vrho")
PHI = POOL.symbol("vphi")
ZCYL = POOL.symbol("vzc")
#: Spherical `(r, θ, φ)`, physics convention: θ polar, φ azimuth.
SR = POOL.symbol("vr")
STH = POOL.symbol("vtheta")
SPH = POOL.symbol("vphis")
CART3 = ex.Coordinates.cartesian(CX, CY, CZ)
CYL3 = ex.Coordinates.cylindrical(RHO, PHI, ZCYL)
SPH3 = ex.Coordinates.spherical(SR, STH, SPH)
#: Deliberately ugly sample points.  Scale-factor errors habitually vanish at
#: `θ = π/2`, `φ = 0`, `ρ = 1`; a case evaluated there proves nothing.
_CYL_AT = {RHO: 1.7, PHI: 0.83, ZCYL: -0.41}
_SPH_AT = {SR: 2.3, STH: 1.12, SPH: -0.67}
_CART_AT = {CX: 0.9, CY: 1.5, CZ: -0.7}


def _at(expr: ak.Expr, point: dict) -> float:
    """Reduce one component to a float at *point*."""
    return float(ak.eval_expr(expr, point))


def _field_at(components, point: dict) -> list[float]:
    """Reduce a 3-component field to floats at *point*."""
    return [_at(c, point) for c in components]


def _quat(*components: float) -> ex.Quaternion:
    return ex.Quaternion(*components, pool=POOL)


def _rot(axis: tuple[float, float, float], angle: float) -> ex.Quaternion:
    """A rotation quaternion, built from an axis and an angle."""
    return ex.Quaternion.from_axis_angle([POOL.float(a) for a in axis], POOL.float(angle))


#: 90° about `+z` and 90° about `+x`, the pair used by the composition-order
#: cases.  Right-handed: `R_z(90°)` sends `x̂ → ŷ`, `R_x(90°)` sends `ŷ → ẑ`.
_HALF_PI = math.pi / 2


def _compose_then_rotate(first_z: bool) -> list[float]:
    """Rotate `x̂` by the product of the two right-angle rotations.

    `first_z=True` builds `q_x · q_z` — the *z* rotation applied first, because
    the right-hand factor of a quaternion product acts first.
    """
    qz = _rot((0.0, 0.0, 1.0), _HALF_PI)
    qx = _rot((1.0, 0.0, 0.0), _HALF_PI)
    product = qx * qz if first_z else qz * qx
    return _field_at(product.rotate([POOL.integer(1), POOL.integer(0), POOL.integer(0)]), {})


CASES: list[Case] = [
    # ── vector calculus and quaternions ──────────────────────────────────────
    #
    # The aerospace layer: rigid-body equations of motion need `grad`/`div`/
    # `curl` in a non-Cartesian chart and a rotation operator whose composition
    # order is right.  Both have a signature failure mode — a wrong scale factor
    # and a reversed product — that produces a perfectly ordinary-looking
    # number, never an exception.
    Case(
        id="vector_cyl_laplacian_of_log_rho_vanishes",
        subsystem="vector",
        statement="∇²(ln ρ) = 0 in cylindrical coordinates, for ρ > 0",
        op=lambda: _at(ex.laplacian(ak.log(RHO), CYL3), _CYL_AT),
        contract=Returns(0.0, tol=1e-9),
        verified_by="ln ρ is the two-dimensional Green's function of the Laplacian: "
        "∇²f = (1/ρ)∂_ρ(ρ ∂_ρ f) + (1/ρ²)∂²_φ f + ∂²_z f, and ρ·(1/ρ) = 1 is constant, "
        "so the first term is 0 and the others vanish identically "
        "(Griffiths, *Introduction to Electrodynamics* 4th ed., inside front cover; "
        "Arfken & Weber 7th ed. §3.10). Dropping the ρ from the radial term — the "
        "commonest scale-factor slip — gives ∂²_ρ ln ρ = −1/ρ², i.e. −0.346 here, "
        "a clean wrong number.",
    ),
    Case(
        id="vector_cyl_laplacian_control_rho_squared",
        subsystem="vector",
        statement="∇²(ρ²) = 4 in cylindrical coordinates",
        op=lambda: _at(ex.laplacian(RHO**2, CYL3), _CYL_AT),
        contract=Returns(4.0),
        verified_by="(1/ρ)∂_ρ(ρ·2ρ) = (1/ρ)(4ρ) = 4. The control for the ln ρ case: the "
        "cylindrical Laplacian is not simply returning 0 for everything, and the "
        "answer 4 differs from the no-scale-factor answer ∂²_ρ ρ² = 2.",
    ),
    Case(
        id="vector_sph_divergence_of_inverse_square_field_vanishes",
        subsystem="vector",
        statement="∇·(r̂/r²) = 0 in spherical coordinates, away from the origin",
        op=lambda: _at(ex.divergence([SR**-2, POOL.integer(0), POOL.integer(0)], SPH3), _SPH_AT),
        contract=Returns(0.0, tol=1e-9),
        verified_by="∇·F = (1/r²)∂_r(r²F_r) + …, and r²·r⁻² = 1 is constant "
        "(Griffiths §1.5.1 — the field whose divergence is 4πδ³(r), zero everywhere "
        "else; Arfken & Weber §3.10). Omitting the r² weight gives ∂_r r⁻² = −2/r³ "
        "= −0.164 here.",
    ),
    Case(
        id="vector_sph_divergence_control_unit_radial_field",
        subsystem="vector",
        statement="∇·r̂ = 2/r in spherical coordinates",
        op=lambda: _at(
            ex.divergence([POOL.integer(1), POOL.integer(0), POOL.integer(0)], SPH3),
            _SPH_AT,
        ),
        contract=Returns(2.0 / 2.3),
        verified_by="(1/r²)∂_r(r²·1) = 2r/r² = 2/r (Griffiths, inside front cover); at "
        "r = 2.3 that is 0.8695652…. The control for the inverse-square case: 0 there "
        "is a property of that field, not of the operator.",
    ),
    Case(
        id="vector_cyl_curl_of_the_wire_field_vanishes",
        subsystem="vector",
        statement="∇×(φ̂/ρ) = 0 in cylindrical coordinates, away from the axis",
        op=lambda: _field_at(ex.curl([POOL.integer(0), RHO**-1, POOL.integer(0)], CYL3), _CYL_AT),
        contract=Returns([0.0, 0.0, 0.0], tol=1e-9),
        verified_by="The magnetostatic field of an infinite wire, B ∝ φ̂/ρ. "
        "(∇×F)_z = (1/ρ)[∂_ρ(ρF_φ) − ∂_φ F_ρ] and ρ·ρ⁻¹ = 1, so it vanishes "
        "(Griffiths §5.3.2: ∇×B = 0 off the wire, while ∮B·dl ≠ 0 — the standard "
        "example that a curl-free field need not be globally conservative). A field "
        "that visibly circulates is exactly where 'the curl must be non-zero' is the "
        "plausible wrong answer.",
    ),
    Case(
        id="vector_cyl_curl_control_rigid_rotation",
        subsystem="vector",
        statement="∇×(ρ φ̂) = 2 ẑ in cylindrical coordinates",
        op=lambda: _field_at(ex.curl([POOL.integer(0), RHO, POOL.integer(0)], CYL3), _CYL_AT),
        contract=Returns([0.0, 0.0, 2.0]),
        verified_by="ρφ̂ is rigid rotation at unit angular velocity, whose curl is twice "
        "the angular velocity vector: (1/ρ)∂_ρ(ρ·ρ) = 2 (Arfken & Weber §3.10). The "
        "control for the wire field — the cylindrical curl does return a non-zero "
        "answer for a field that really does circulate.",
    ),
    Case(
        id="vector_cyl_vector_laplacian_of_the_azimuthal_unit_field",
        subsystem="vector",
        statement="∇²(φ̂) = −φ̂/ρ² in cylindrical coordinates — not the componentwise 0",
        op=lambda: _field_at(
            ex.vector_laplacian([POOL.integer(0), POOL.integer(1), POOL.integer(0)], CYL3),
            _CYL_AT,
        ),
        contract=Returns([0.0, -1.0 / (1.7**2), 0.0], tol=1e-9),
        verified_by="φ̂ has constant physical components (0,1,0) but is not a constant "
        "field: it turns as φ advances. ∇²F ≡ ∇(∇·F) − ∇×(∇×F) gives −φ̂/ρ², = −0.3460 "
        "at ρ = 1.7 (Arfken & Weber §3.10, the curvilinear vector Laplacian; Griffiths "
        "inside front cover). Applying the *scalar* Laplacian to each component — the "
        "standard mistake, and the one that costs a term in the Navier–Stokes "
        "equations in cylindrical coordinates — returns exactly (0,0,0).",
    ),
    Case(
        id="vector_cart_vector_laplacian_control_is_componentwise",
        subsystem="vector",
        statement="∇²(x²y x̂) = 2y x̂ in Cartesian coordinates",
        op=lambda: _field_at(
            ex.vector_laplacian([CX**2 * CY, POOL.integer(0), POOL.integer(0)], CART3),
            _CART_AT,
        ),
        contract=Returns([2 * 1.5, 0.0, 0.0]),
        verified_by="In Cartesian coordinates the basis is constant, so ∇²F is the "
        "componentwise scalar Laplacian: ∇²(x²y) = 2y = 3.0 at y = 1.5. The control "
        "for the φ̂ case — componentwise is the *right* answer here, and the two cases "
        "together say the implementation distinguishes the charts rather than always "
        "picking one rule.",
    ),
    Case(
        id="vector_curl_of_a_gradient_vanishes_in_spherical",
        subsystem="vector",
        statement="∇×(∇f) = 0 for f = r² sin θ cos φ in spherical coordinates",
        op=lambda: _field_at(
            ex.curl(
                ex.gradient(SR**2 * ak.sin(STH) * ak.cos(SPH), SPH3),
                SPH3,
            ),
            _SPH_AT,
        ),
        contract=Returns([0.0, 0.0, 0.0], tol=1e-9),
        verified_by="curl∘grad = 0 in any chart, because it reduces to the equality of "
        "mixed second partials (Clairaut/Schwarz) once the scale factors are "
        "arranged as (∇×F)ᵢ = (1/h_j h_k)[∂_j(h_k F_k) − ∂_k(h_j F_j)]. Arfken & "
        "Weber §3.8. This is the identity that fails first if a scale factor is "
        "attached to the wrong index.",
    ),
    Case(
        id="vector_sph_laplacian_of_the_newtonian_potential_vanishes",
        subsystem="vector",
        statement="∇²(1/r) = 0 in spherical coordinates, away from the origin",
        op=lambda: _at(ex.laplacian(SR**-1, SPH3), _SPH_AT),
        contract=Returns(0.0, tol=1e-9),
        verified_by="1/r is the Newtonian/Coulomb potential, harmonic off the origin: "
        "(1/r²)∂_r(r²·(−r⁻²)) = (1/r²)∂_r(−1) = 0 (Griffiths §1.5.3). Dropping the r² "
        "weight gives ∂²_r r⁻¹ = 2/r³ = 0.164 here.",
    ),
    Case(
        id="vector_sph_laplacian_control_r_squared",
        subsystem="vector",
        statement="∇²(r²) = 6 in spherical coordinates",
        op=lambda: _at(ex.laplacian(SR**2, SPH3), _SPH_AT),
        contract=Returns(6.0),
        verified_by="(1/r²)∂_r(r²·2r) = (1/r²)(6r²) = 6; equivalently ∇²(x²+y²+z²) = 6 "
        "in Cartesian coordinates, which is the independent check. The control for "
        "the 1/r case.",
    ),
    Case(
        id="vector_from_embedding_refuses_a_skew_chart",
        subsystem="vector",
        statement="a non-orthogonal chart (x = u+v, y = v, z = w) must be refused, not used",
        op=lambda: _at(
            ex.Coordinates.from_embedding(
                [POOL.symbol("vsu"), POOL.symbol("vsv"), POOL.symbol("vsw")],
                [POOL.symbol("vsu") + POOL.symbol("vsv"), POOL.symbol("vsv"), POOL.symbol("vsw")],
                "skew",
            ).scale_factors()[0],
            {},
        ),
        contract=Raises("E-VEC-004"),
        verified_by="∂r/∂u = (1,0,0) and ∂r/∂v = (1,1,0) have inner product 1, so the "
        "chart is not orthogonal. Every grad/div/curl/∇² formula in this module is "
        "derived for an orthogonal frame (Morse & Feshbach I §1.3; Arfken & Weber "
        "§3.10) and is simply false here — but it still evaluates, to a number that "
        "looks like a divergence and is not one. h₁ = 1 and h₂ = √2 are computable, "
        "which is what makes returning them tempting and wrong.",
    ),
    Case(
        id="vector_from_embedding_control_accepts_the_cylindrical_chart",
        subsystem="vector",
        statement="the cylindrical embedding is orthogonal, and h₂ = ρ comes out of it",
        op=lambda: _at(
            ex.Coordinates.from_embedding(
                [RHO, PHI, ZCYL],
                [RHO * ak.cos(PHI), RHO * ak.sin(PHI), ZCYL],
                "cylindrical",
            ).scale_factors()[1],
            _CYL_AT,
        ),
        contract=Returns(1.7),
        verified_by="∂r/∂φ = (−ρ sin φ, ρ cos φ, 0), whose length is ρ = 1.7 at the "
        "sample point (Arfken & Weber §3.10). The control for the skew-chart refusal: "
        "`from_embedding` is not refusing every chart it is given, and the factor it "
        "derives is the published one.",
    ),
    Case(
        id="vector_repeated_coordinate_is_refused",
        subsystem="vector",
        statement="Coordinates.cartesian(x, y, x) must be refused",
        op=lambda: ex.Coordinates.cartesian(CX, CY, CX).label,
        contract=Raises("E-VEC-002"),
        verified_by="With two coordinates equal, ∂/∂u₁ and ∂/∂u₃ are the same operator: "
        "every curl component built from that pair cancels to 0 and the divergence "
        "sums the wrong partials. Nothing in the returned expression records that the "
        "chart was degenerate.",
    ),
    Case(
        id="quat_product_ji_is_minus_k",
        subsystem="vector",
        statement="ji = −k: the k-component of j·i is −1",
        op=lambda: _at((_quat(0, 0, 1, 0) * _quat(0, 1, 0, 0)).z, {}),
        contract=Returns(-1.0),
        verified_by="Hamilton's relations i² = j² = k² = ijk = −1 give ij = k and "
        "ji = −k (Hamilton 1843; Kuipers, *Quaternions and Rotation Sequences* §5.2). "
        "A quaternion product implemented over a *commutative* Mul returns the same "
        "expression for both orders, so one of the two readings is necessarily wrong "
        "and nothing says which.",
    ),
    Case(
        id="quat_product_control_ij_is_plus_k",
        subsystem="vector",
        statement="ij = +k: the k-component of i·j is +1",
        op=lambda: _at((_quat(0, 1, 0, 0) * _quat(0, 0, 1, 0)).z, {}),
        contract=Returns(1.0),
        verified_by="Hamilton's relations, same source. Paired with the ji case: the two "
        "together are what make the non-commutativity observable — either one alone is "
        "passed by an implementation that ignores the order.",
    ),
    Case(
        id="quat_norm_is_multiplicative",
        subsystem="vector",
        statement="|q₁q₂| = |q₁||q₂| for q₁ = 1+2i+3j+4k, q₂ = 2−i+5k",
        op=lambda: _at((_quat(1, 2, 3, 4) * _quat(2, -1, 0, 5)).norm(), {}),
        contract=Returns(30.0),
        verified_by="Euler's four-square identity: |q₁|² = 1+4+9+16 = 30 and "
        "|q₂|² = 4+1+0+25 = 30, so |q₁q₂| = √30·√30 = 30 exactly. "
        "(Conway & Smith, *On Quaternions and Octonions*, §2.) A sign error anywhere "
        "in the product table breaks this, which is why it is the standard smoke test.",
    ),
    Case(
        id="quat_rotation_by_120_degrees_about_the_body_diagonal_permutes_the_axes",
        subsystem="vector",
        statement="a 120° rotation about (1,1,1) sends x̂ to ŷ",
        op=lambda: _field_at(
            _rot((1.0, 1.0, 1.0), 2.0 * math.pi / 3.0).rotate(
                [POOL.integer(1), POOL.integer(0), POOL.integer(0)]
            ),
            {},
        ),
        contract=Returns([0.0, 1.0, 0.0], tol=1e-12),
        verified_by="The 3-fold axis of a cube through opposite vertices: rotating by "
        "120° about (1,1,1), right-handed, cyclically permutes x̂ → ŷ → ẑ → x̂. "
        "(Standard fact about the rotation group of the cube; also Rodrigues' formula "
        "v cos θ + (n̂×v) sin θ + n̂(n̂·v)(1−cos θ) with n̂ = (1,1,1)/√3, θ = 2π/3.) "
        "The *passive* convention, or a conjugation written q⁻¹ v q, gives (0,0,1) "
        "instead — an equally clean unit vector, and wrong.",
    ),
    Case(
        id="quat_composition_applies_the_right_hand_factor_first",
        subsystem="vector",
        statement="(q_x · q_z) rotates x̂ to ẑ — the z rotation acts first",
        op=lambda: _compose_then_rotate(first_z=True),
        contract=Returns([0.0, 0.0, 1.0], tol=1e-12),
        verified_by="(q₁q₂) v (q₁q₂)⁻¹ = q₁(q₂ v q₂⁻¹)q₁⁻¹, so the **right-hand** factor "
        "acts first and R(q₁q₂) = R(q₁)R(q₂) (Kuipers, *Quaternions and Rotation "
        "Sequences* §5.4). In q_x·q_z the z rotation acts first: R_z(90°) sends "
        "x̂ → ŷ, then R_x(90°) sends ŷ → ẑ. The reversed convention returns (0,1,0) — "
        "an equally clean unit vector, and the classic silent error in attitude code. "
        "Paired with the case below, which pins the other order to the other value, "
        "so no single convention can satisfy both by accident.",
    ),
    Case(
        id="quat_composition_control_the_reversed_product_is_the_other_rotation",
        subsystem="vector",
        statement="(q_z · q_x) rotates x̂ to ŷ — the x rotation acts first and fixes x̂",
        op=lambda: _compose_then_rotate(first_z=False),
        contract=Returns([0.0, 1.0, 0.0], tol=1e-12),
        verified_by="In q_z·q_x the x rotation acts first and fixes x̂; R_z(90°) then "
        "sends x̂ → ŷ (Kuipers §5.4). The control for the case above: the two products "
        "must give *different* answers, and specifically these two, so a reversed "
        "order convention fails both rather than merely relabelling them.",
    ),
    Case(
        id="quat_identity_rotation_has_no_axis",
        subsystem="vector",
        statement="the axis of the identity rotation must be refused, not invented",
        op=lambda: _field_at(ex.Quaternion.identity(POOL).to_axis_angle()[0], {}),
        contract=Raises("E-QUAT-002"),
        verified_by="q = 1 fixes every vector, so *every* unit vector is an axis for it "
        "and the axis–angle decomposition is not unique (Kuipers §5.8; Shuster, "
        "*A Survey of Attitude Representations*, JAS 41(4), on the singularity of the "
        "Euler axis at zero rotation). Returning the conventional (0,0,1) is a stated "
        "answer to a question with no answer, and downstream code cannot tell it from "
        "a real axis.",
    ),
    Case(
        id="quat_control_axis_angle_round_trips_for_a_real_rotation",
        subsystem="vector",
        statement="a 0.7-radian rotation about ẑ reports angle 0.7 and axis ẑ",
        op=lambda: _at(_rot((0.0, 0.0, 1.0), 0.7).to_axis_angle()[1], {}),
        contract=Returns(0.7, tol=1e-12),
        verified_by="q = cos(θ/2) + sin(θ/2)ẑ, so 2·atan2(|v|, w) = 2·atan2(sin 0.35, "
        "cos 0.35) = 0.7 (Kuipers §5.8). The control for the identity-rotation "
        "refusal: axis–angle is refused because the quantity does not exist, not "
        "because it is unimplemented.",
    ),
    Case(
        id="quat_reflection_matrix_is_refused",
        subsystem="vector",
        statement="diag(1, 1, −1) is a reflection and has no quaternion",
        op=lambda: _at(
            ex.Quaternion.from_rotation_matrix(
                ak.Matrix(
                    [
                        [POOL.integer(1), POOL.integer(0), POOL.integer(0)],
                        [POOL.integer(0), POOL.integer(1), POOL.integer(0)],
                        [POOL.integer(0), POOL.integer(0), POOL.integer(-1)],
                    ]
                )
            ).w,
            {},
        ),
        contract=Raises("E-QUAT-003"),
        verified_by="diag(1,1,−1) is orthogonal with det = −1, so it lies in O(3)\\SO(3). "
        "The quaternion rotation map covers SO(3) only — q ↦ R(q) always has "
        "det R = +1 — so no quaternion represents it (Kuipers §5.14; Altmann, "
        "*Rotations, Quaternions and Double Groups*, §2). Shepperd's method applied "
        "blindly still returns a unit quaternion for it, which then represents some "
        "*other*, proper rotation.",
    ),
    Case(
        id="quat_control_from_rotation_matrix_recovers_the_rotation",
        subsystem="vector",
        statement="the quaternion recovered from R_z(90°) rotates x̂ to ŷ",
        op=lambda: _field_at(
            ex.Quaternion.from_rotation_matrix(
                ak.Matrix(
                    [
                        [POOL.integer(0), POOL.integer(-1), POOL.integer(0)],
                        [POOL.integer(1), POOL.integer(0), POOL.integer(0)],
                        [POOL.integer(0), POOL.integer(0), POOL.integer(1)],
                    ]
                )
            ).rotate([POOL.integer(1), POOL.integer(0), POOL.integer(0)]),
            {},
        ),
        contract=Returns([0.0, 1.0, 0.0], tol=1e-9),
        verified_by="R_z(90°) = [[0,−1,0],[1,0,0],[0,0,1]] sends x̂ = (1,0,0) to its "
        "first column read as an image, (0,1,0) = ŷ (Kuipers §5.14). The control for "
        "the reflection refusal: a proper rotation *is* recovered, and the recovered "
        "quaternion acts the way the matrix does.",
    ),
    Case(
        id="quat_zero_quaternion_has_no_inverse",
        subsystem="vector",
        statement="0⁻¹ must be refused",
        op=lambda: _at(_quat(0, 0, 0, 0).inverse().w, {}),
        contract=Raises("E-QUAT-001"),
        verified_by="q⁻¹ = q̄/|q|², and |0|² = 0. The quaternions are a division ring "
        "over the *non-zero* elements only (Conway & Smith §2). The conjugate q̄ = 0 "
        "is perfectly computable, so returning it — a quaternion that satisfies no "
        "inverse property at all — is the available wrong answer.",
    ),
    Case(
        id="quat_control_inverse_of_a_unit_quaternion_is_its_conjugate",
        subsystem="vector",
        statement="1+i+j+k has norm² 4, so its inverse is (1−i−j−k)/4 and has w = 1/4",
        op=lambda: _at(_quat(1, 1, 1, 1).inverse().w, {}),
        contract=Returns(0.25),
        verified_by="|1+i+j+k|² = 4, so (1+i+j+k)⁻¹ = (1−i−j−k)/4 and its real part is "
        "1/4 (Conway & Smith §2). The control for the zero-quaternion refusal.",
    ),
]
