"""
alkahest.experimental — APIs that are not yet semver-stable.

Functions and classes in this module may change signature, be renamed, or be
removed between minor versions without a deprecation cycle.  Graduate to
``alkahest.*`` once the API has been exercised in production.

Matrix linear algebra (``Matrix.rref``, ``nullspace``, ``rank``, ``lu``, ``qr``,
``cholesky``, ``jordan_form``, ``matrix_exp``, …) lives on the stable
:class:`~alkahest.Matrix` type and raises :class:`~alkahest.LinearAlgebraError`
(``E-LINALG-*``) when unsupported. Agents should probe
``alkahest.capabilities()`` at session start.

Graduated to the top-level stable surface (still re-exported here for
backward compatibility):
- :class:`Assumptions`, :func:`evaluate` / :class:`EvaluationResult`
- :func:`conjugate`, :func:`re`, :func:`im`, :func:`arg`
- :func:`residue`
- :func:`lambert_w`, :func:`digamma`, :func:`bessel_j0`, :func:`bessel_j1`
- :func:`solve`, :class:`GroebnerBasis`, :class:`GbPoly` (require ``groebner``)
- :func:`to_lean`, :func:`to_stablehlo`

Remaining experimental surface:
- :func:`to_jax` — JAX primitive integration (requires JAX)
- :func:`compile_cuda` / :class:`CudaCompiledFn` — NVPTX codegen (requires
  ``cuda`` + ``jit``)

Probability and statistics:
- :class:`Distribution` and its constructors :func:`Normal`, :func:`LogNormal`,
  :func:`Uniform`, :func:`Exponential`, :func:`Gamma`, :func:`Beta`,
  :func:`Bernoulli`, :func:`Binomial`, :func:`Poisson` — a law is a *name plus
  symbolic parameters*, carrying its density, its support, and the constraints
  that make it a distribution at all. A numeric parameter outside its
  constraint raises ``E-PROB-001``; a symbolic one cannot be decided and is
  carried on :meth:`Distribution.constraints` for the caller to discharge
- :meth:`Distribution.mean`, :meth:`~Distribution.variance`,
  :meth:`~Distribution.moment`, :meth:`~Distribution.pdf`,
  :meth:`~Distribution.cdf`, :meth:`~Distribution.quantile`,
  :meth:`~Distribution.characteristic_function`, plus
  :attr:`~Distribution.kind`, :meth:`~Distribution.params`,
  :meth:`~Distribution.support` and :meth:`~Distribution.constraints`. These
  are **methods on the law**, not module-level functions — ``Normal(mu,
  sigma).cdf(x)``, not ``cdf(dist, x)`` — so there is one way to ask each
  question. Every derived quantity is **checked against numerical quadrature
  of its own defining integral before it is returned**; one the checker cannot
  confirm raises ``E-PROB-005`` rather than arriving with a caveat
- :meth:`~Distribution.characteristic_function` is **complex-valued**. Evaluate
  it with ``evaluate(phi, env, mode="complex")``; the real modes return
  ``value=None`` with ``status="unsupported"`` rather than dropping the
  imaginary part
- The generating-function family, also methods on the law:
  :meth:`~Distribution.moment_generating_function` (``M_X(t) = E[e^{tX}]``),
  :meth:`~Distribution.cumulant_generating_function` (``K = log M``),
  :meth:`~Distribution.probability_generating_function` (``G_X(z) = E[z**X]``),
  :meth:`~Distribution.cumulant`, :meth:`~Distribution.factorial_moment`,
  :meth:`~Distribution.skewness` and :meth:`~Distribution.excess_kurtosis`
- **These refuse where the quantity does not exist, which is most of the point
  of having them here rather than in a table.** ``M_X`` for a
  :func:`LogNormal` raises ``E-PROB-006``: ``E[e^{tX}]`` is ``+inf`` for every
  ``t > 0``, and a CAS that completes the square anyway returns a clean closed
  form that is the value of no integral. ``M_X(t) = lambda/(lambda - t)`` for
  an :func:`Exponential` holds *only* on ``t < lambda``, and ``(1 - theta t)``
  ``**-k`` for a :func:`Gamma` only on ``t < 1/theta`` — outside those strips
  the expressions are still finite, still plausible and still wrong, so a
  decidable argument outside the strip raises ``E-PROB-006`` and an undecidable
  one publishes the condition on :func:`prob_side_conditions`. A law whose MGF
  is entire leaves that list empty, so empty means *checked*
- ``G_X(z) = sum_k z**k P(X = k)`` is defined only for a law on the
  non-negative integers; asking a :func:`Normal` for one raises ``E-PROB-002``
  rather than returning the formal ``E[exp(X log z)]``, which is the *moment*
  generating function at ``log z`` and says nothing about any ``P(X = k)``
- :meth:`~Distribution.cumulant` raises ``E-PROB-006`` for a :func:`LogNormal`
  — ``kappa_n = K**(n)(0)`` and there is no ``K`` — while
  :meth:`~Distribution.skewness` and :meth:`~Distribution.excess_kurtosis`,
  which are defined from *central* moments, are returned there as usual. The
  kurtosis is the **excess** one: ``0`` for a normal, not ``3``
- :func:`prob_side_conditions` — the hypotheses the last :class:`Distribution`
  method or :func:`expectation` call had to *assume*. The return value is an
  ``Expr``, so there is nowhere in band to hang one; without this channel a
  conditional answer and a theorem look identical at the call site. Non-empty
  for a symbolic ``cdf`` argument (the closed form is the in-support branch),
  a symbolic ``quantile`` argument, and a payoff kink that cannot be placed
  inside the support
- :func:`expectation` — ``E[f(X)]``, reduced to integrals the existing
  integrator attempts. A divergence the convergence gate can see raises
  ``E-PROB-006`` (checked *before* the symbolic work, so the answer is "no
  value exists" rather than "the integrator declined"); an integral that does
  not close raises ``E-PROB-003`` **naming it**, never an unevaluated object
  dressed as an answer. The gate is sufficient, not complete — a divergence it
  cannot see (``E[e**X]`` under ``Exponential(1)``, whose integrand is the
  constant ``1``) arrives as ``E-PROB-003`` instead, so read ``006`` as "proved
  divergent", never ``003`` as "proved convergent". ``max(S - K, 0)`` under a
  :func:`LogNormal` derives Black–Scholes
- :func:`expectation_affine` — linearity, which needs no independence at all —
  and :func:`variance_affine_independent`, which does and says so in its name.
  There are no joint distributions, no conditioning and no covariance here: a
  product of two variates raises ``E-PROB-002`` rather than guessing at a
  joint law

Information theory over those same laws:
- :meth:`Distribution.entropy` — Shannon ``H = -sum p log p`` on a discrete
  support, **differential** ``h = -int f log f`` on a continuous one. They are
  not the same quantity and are not unified behind one formula: ``h`` is not
  the limit of ``H``, is not non-negative (``Uniform(0, 1/2).entropy()`` is
  ``log(1/2) < 0``, returned rather than clamped), and is **not invariant under
  a change of variables** — it shifts by ``E[log|dx/dy|]``, which is exactly
  the ``mu`` by which a :func:`LogNormal`'s entropy exceeds the underlying
  :func:`Normal`'s. ``base=None`` is nats, the right default for symbolic work;
  ``base=2`` is bits
- :func:`kl_divergence` — ``D(P||Q)``, **not** a distance. Not symmetric, and
  ``+inf`` whenever ``P`` charges a set ``Q`` gives probability zero. That last
  one is *gated*, because the closed form does not fail there: substituting
  ``Uniform(0,1)`` against ``Uniform(0,1/2)`` into ``log((b2-a2)/(b1-a1))``
  gives ``-0.693``, a finite negative KL divergence, which Gibbs' inequality
  forbids. Decidably non-nested supports raise ``E-PROB-006`` naming the value
  as ``+inf``; an undecidable containment is published on
  :func:`prob_side_conditions` instead of assumed
- :func:`cross_entropy` — ``H(P, Q)``, assembled as ``H(P) + D(P||Q)`` from the
  two tables and then checked against ``-E_P[log q]``, which is neither of
  them, so the identity is falsifiable rather than definitional
- :func:`mutual_information_independent` — ``0``, under an independence the
  caller asserts by choosing the name. There is no dependent case: the
  bivariate-normal ``-log(1 - rho^2)/2`` would need a joint-distribution type
  this library does not have

Calculus / ODE / transform surface:
- :func:`heaviside`, :func:`dirac_delta` — distribution primitive constructors
- :func:`dsolve` — classical symbolic ODE solver (#153); constant coefficients
  may be symbolic, in which case the returned dicts carry ``side_conditions``
  and ``notes`` naming the parameter branch that was *not* assumed
- :func:`dsolve_system` — linear constant-coefficient systems ``y' = A·y + f(t)``
  over an :class:`alkahest.ODE`, by Putzer's algorithm (defective ``A``
  included); symbolic entries are allowed and an undecided eigenvalue
  confluence is reported rather than assumed
- :func:`laplace_transform` / :func:`inverse_laplace_transform` (#152)
- :func:`fourier_transform` / :func:`inverse_fourier_transform` (#158)
- :func:`z_transform` / :func:`inverse_z_transform` (#159)
- :func:`multilimit` — two-variable limits (#156)
- :func:`asymptotic_expand` — asymptotic expansion at infinity (#161)
- :func:`puiseux_series` / :class:`PuiseuxExpansion` — **Puiseux** expansion:
  a truncated series in *fractional* powers of ``var - point``. The sibling of
  the stable :func:`alkahest.series` for the half-integer valuations a
  ``Series`` has no representation for — ``sqrt(x)``, ``sqrt(sin(x))``,
  ``x**Fraction(1,2)*sin(x)`` — each of which :func:`alkahest.series` refuses
  with ``E-SERIES-004``. Every returned expansion is **verified** before it is
  returned (residual decay at several points approaching the expansion point,
  plus an exact ``S**e == f**e`` comparison where the shape allows one); one
  that cannot be confirmed raises ``E-SERIES-006`` instead. ``log(x)``,
  ``sqrt(x)*log(x)`` (a Puiseux-*log* term, not a Puiseux series) and
  ``exp(1/x)`` refuse with ``E-SERIES-005``
- :func:`asymptotics_from_recurrence` — Poincaré–Perron growth of a P-recursive
  sequence, with the derived growth rate/exponent kept apart from the fitted
  connection constant (M5)
- :func:`series_solve` — power-series / Frobenius ODE solutions (#160)
- :class:`Fps` — lazy formal power series over ℚ (#155)

``q``-analogue creative telescoping (M4b):
- :func:`q_zeilberger` / :class:`QZeilbergerCertificate` — ``q``-Zeilberger for
  ``q``-hypergeometric sums (Gaussian binomials, ``q``-Pochhammer symbols),
  with the certificate re-checked as an exact identity in ``Q(q)(q**n)(q**k)``
  and a two-valued verdict on whether it carries over to the sum
- :func:`qbinomial`, :func:`qpochhammer` — builders for the two function heads
  the engine recognises
- :meth:`~alkahest.experimental.QZeilbergerCertificate.specialize_at_root_of_unity`
  / :class:`QRootOfUnitySpecialization` (M4) — the step from a ``Q(q)``
  identity to ``q = ζ_d``, a primitive ``d``-th root of unity: the
  ``q``-supercongruence literature. Pole and vanishing hypotheses are decided
  exactly by polynomial divisibility by ``Φ_d(q)`` over ``Q``, never
  numerically, and the verdict is three-valued (``"specializes"``,
  ``"obstructed"``, ``"unknown"``) rather than a silent specialisation
- :func:`cyclotomic_polynomial` — ``Φ_d(q)``, the modulus the root-of-unity
  arithmetic works over, exposed so a caller can redo the divisibility check
  by hand

Vector calculus and quaternions (the rigid-body layer):
- :class:`Coordinates` — an orthogonal chart, built by
  :meth:`Coordinates.cartesian`, :meth:`Coordinates.cylindrical`,
  :meth:`Coordinates.spherical`, or :meth:`Coordinates.from_embedding`, which
  derives the scale factors from a Cartesian parametrisation and **refuses**
  (``E-VEC-004``) a chart it cannot prove orthogonal. There is no constructor
  taking scale factors directly
- :func:`gradient`, :func:`divergence`, :func:`curl`, :func:`laplacian`,
  :func:`vector_laplacian`, and the algebra they are used with —
  :func:`dot`, :func:`cross`, :func:`norm`. Vector fields are three
  **physical** components, read in the local orthonormal frame, which is the
  convention ``sympy.vector`` uses and the one an engineer means by "the ρ
  component"
- :func:`vector_laplacian` is ``∇(∇·F) − ∇×(∇×F)``. That equals the
  componentwise scalar Laplacian in Cartesian coordinates *only*; applying
  :func:`laplacian` to each physical component of a cylindrical or spherical
  field is a silent error, and this entry point exists so nobody has to
- :class:`Quaternion` — Hamilton product (``i*j == k``, ``j*i == -k``),
  conjugate, norm, inverse, the active rotation ``q v q⁻¹``, and conversions
  to and from a rotation matrix and axis–angle form.
  ``(q1 * q2)`` applies ``q2`` first, so
  ``(q1 * q2).to_rotation_matrix()`` is ``R(q1) @ R(q2)``.
  :meth:`Quaternion.to_axis_angle` **refuses** (``E-QUAT-002``) for the
  identity rotation, which has no axis because every unit vector is one, and
  :meth:`Quaternion.from_rotation_matrix` refuses (``E-QUAT-003``) anything it
  cannot check is a proper rotation — including a symbolic matrix, where the
  branch selection is a comparison it cannot make

Risch–Norman (parallel Risch) heuristic integration:
- :func:`integrate_parallel_risch` / :class:`ParallelRischResult` — posits
  ``F = P/Q + Σ dⱼ·log(pⱼ)`` over the monomial basis of
  ``ℚ(x, exp …, log …)``, differentiates it, and solves one linear system
  over ℚ instead of building a differential-field tower.  Every returned
  antiderivative passes a ``d/dx F == f`` gate, and the algebraic-independence
  precondition (Bronstein 2007) is checked before the system is built.
  **A declined result is not a non-elementarity verdict** — the function
  returns a result object and never raises; see the class documentation.
  Separate from :func:`alkahest.integrate`, whose routing is unchanged.

Double-sum (Apagodu–Zeilberger) creative telescoping (M4):
- :func:`telescope2d` / :class:`Telescoping2dCertificate` — creative
  telescoping for a proper hypergeometric term ``F(n,j,k)`` with **two** bound
  indices, generalizing :func:`alkahest.zeilberger` from one summation index
  to two: finds ``a_0(n), …, a_J(n)`` and two rational certificates ``c_1,
  c_2`` with ``Σ_i a_i(n)·F(n+i,j,k) = Δ_j(c_1·F) + Δ_k(c_2·F)``, re-checked
  as an exact identity in ``Q(n,j,k)`` before being returned
- :meth:`~alkahest.experimental.Telescoping2dCertificate.boundary_status` —
  the two-dimensional boundary/corner analysis: the boundary of a rectangle
  is **four one-dimensional strip sums**, not four corner evaluations (read
  the Rust module docs, ``alkahest_cas::holonomic::telescoping2d::boundary``,
  for the derivation). Only **constant** (not ``n``-dependent) rectangles are
  supported, and only a sufficient "each strip vanishes pointwise" criterion
  is checked — real, stated restrictions, not unfinished polish
- :func:`telescope_md` / :class:`TelescopingMdCertificate` (since 3.10) — the
  general form of :func:`telescope2d` for an arbitrary number ``m >= 1`` of
  bound indices, not just two: ``telescope_md(term, n, [x_1, ..., x_m])``
  finds ``m`` rational certificates ``c_1, ..., c_m`` with
  ``Σ_i a_i(n)·F(n+i,x) = Σ_t Δ_t(c_t·F)``, and
  :meth:`~alkahest.experimental.TelescopingMdCertificate.boundary_status`
  decides the box-sum boundary (``2m`` face sums, the ``m``-dimensional
  generalization of the four-strip-sum result, **not** ``2**m`` corner
  evaluations). Same proper-hypergeometric-only, fixed-denominator,
  constant-box-only scope as ``telescope2d`` — no genuinely broader summand
  class. Raising ``m`` or ``max_cert_degree`` grows the ansatz search space
  fast (a certificate numerator is a box of
  ``(max_cert_degree + 1)**(m + 1)`` unknowns, and there are ``m`` of them),
  so this module also enforces two resource ceilings on the underlying exact
  linear solve — a single search probe is refused outright above 400
  unknowns, and the total work spent on probes at or above 150 unknowns in
  one search call is capped to 300 — so a search with no certificate in
  reach at all comes back as a fast, honest ``SearchExhausted`` (naming the
  ceiling when that, not genuine non-existence, is why) rather than running
  unboundedly long. See the Rust module docs
  (``alkahest_cas::holonomic::telescoping2d``) for the complete, honestly-
  stated scope and the exact ceiling values
- This is a genuinely scoped-down engine: proper hypergeometric summands
  only, no general Wegschaider reduction, a bounded-degree ansatz search
  rather than a minimal Gosper normal form. See the Rust module docs for the
  complete, honest limitations list

Novelty filtering (:mod:`alkahest.experimental.novelty`):
- :class:`RecurrenceClaim` — a recurrence in a normal form two presentations
  of the same fact share, plus a stable ``claim_hash`` to dedupe on
- :class:`QRecurrenceClaim` — the same for a ``q``-recurrence, whose
  coefficients live in ``Q(q, q^n)`` rather than ``Q[n]``
- :func:`check_novelty` / :class:`NoveltyVerdict` — was this claim already
  written down? Three-valued, and a negative is never reported as "novel"
- :class:`OeisCache` (offline, the tested path) and :class:`OeisWeb` (opt-in
  network) as sources

Coefficient fields for elimination (M9):
- :class:`ParametricGroebnerBasis` / :class:`ParametricGbPoly` — a Gröbner
  basis in ``Q(params)[vars]`` rather than ``Q[vars, params]``, reachable as
  ``GroebnerBasis.compute(polys, vars, params=[...])``.  The basis is generic,
  so it reports the hypersurfaces its leading coefficients assumed non-zero
  (``conditions()``) and refuses to ``specialize()`` on them instead of
  returning something that is not a basis — or, with ``specialize(pt,
  verify=True)``, re-solves at the point and refuses only if the refusal was
  really necessary
- :class:`ParametricRosenfeldGroebnerResult` — differential elimination with
  the parameters in the coefficient field, from
  ``rosenfeld_groebner(dae, params=[...])``; with ``eliminate=[...]`` it also
  reports the first prolongation round that was informative, and warns when
  more rounds were taken than that

Computational group theory (permutation groups):
- :class:`Permutation` — a bijection of ``0, …, n-1`` in images-array form,
  with :meth:`~Permutation.compose`, :meth:`~Permutation.inverse`,
  :meth:`~Permutation.pow`, :meth:`~Permutation.cycles`,
  :meth:`~Permutation.cycle_type`, :meth:`~Permutation.order` (an exact Python
  ``int``: the largest order in ``S_n`` is Landau's function, which passes 64
  bits a little after degree 180) and :meth:`~Permutation.sign`. **Points are
  0-based** — GAP, the ATLAS and the literature are 1-based, so transcribe
  generators with :meth:`Permutation.from_cycles_one_based` rather than
  subtracting one by hand. **Composition is left-to-right**: ``p * q`` applies
  ``p`` first, so ``(p * q).apply(i) == q.apply(p.apply(i))``. That is GAP's
  convention and the opposite of ``f(g(x))``; mixing the two gives a group of
  the right order whose elements are the inverses of the ones you meant
- :class:`PermutationGroup` — the subgroup of ``S_n`` generated by a list of
  permutations, with the standard families
  :meth:`~PermutationGroup.symmetric`, :meth:`~PermutationGroup.alternating`,
  :meth:`~PermutationGroup.cyclic` and :meth:`~PermutationGroup.dihedral`
  (``|D_n| = 2n``; ``n < 3`` refuses with ``E-GRP-006`` rather than returning a
  group of the wrong order under the right name)
- :meth:`~PermutationGroup.orbit`, :meth:`~PermutationGroup.orbits` and
  :meth:`~PermutationGroup.schreier_vector` — orbit computation with the
  Schreier vector exposed, at any degree
- :meth:`~PermutationGroup.order`, :meth:`~PermutationGroup.contains` and
  :meth:`~PermutationGroup.sift` — all read off a base and strong generating
  set computed by deterministic Schreier–Sims and cached. The order is exact
  and unbounded; membership is decided by sifting, never by enumeration, so it
  works in groups far too large to list. The chain itself is exposed through
  :meth:`~PermutationGroup.base`,
  :meth:`~PermutationGroup.strong_generators`,
  :meth:`~PermutationGroup.basic_orbits` and
  :meth:`~PermutationGroup.stabilizer_generators`
- :meth:`~PermutationGroup.elements` refuses above
  ``GROUP_DEFAULT_ELEMENT_CAP`` with ``E-GRP-004``. A group of order ``10**20``
  has a perfectly computable order and no listable element set, and the two
  must not be confused; ``order()`` keeps working
- Scope of :class:`PermutationGroup` itself: permutation groups only.
  Finitely presented groups, Todd–Coxeter coset enumeration and low-degree
  group cohomology are :class:`FpGroup`; matrix groups over ``GF(q)`` are
  :class:`MatGroup`; conjugacy classes and character tables are
  :class:`CharacterTable` — all three below. What is still absent is anything
  needing backtrack search: Sylow subgroups, element centralizers, subgroup
  lattices and normaliser computations. See the Rust module docs
  (``alkahest_cas::group``) for the full list

Matrix groups over GF(q):
- :class:`MatGroup` — a subgroup of ``GL(d, q)`` given by **any** list of
  invertible :class:`GfMatrix` generators, not only a classical family. The
  action is on **row** vectors, ``v -> v @ M``, so a vector is a ``1 x d``
  ``GfMatrix`` and ``M @ N`` means "apply ``M``, then ``N``" — the same
  left-to-right order as :class:`Permutation`
- :meth:`~MatGroup.order`, :meth:`~MatGroup.contains` and
  :meth:`~MatGroup.sift` read off a base and strong generating set computed by
  Schreier–Sims **on the action on vectors**, with the base always inside the
  standard basis (a matrix fixing every ``e_i`` is the identity, so the chain
  has at most ``d`` levels). The order is a computation, not a formula:
  :func:`matgroup_gl_order`, :func:`matgroup_sl_order` and
  :func:`matgroup_sp_order` are the closed forms to check it against
- Constructors :meth:`~MatGroup.general_linear`,
  :meth:`~MatGroup.special_linear`, :meth:`~MatGroup.symplectic` (the argument
  is ``n``; the matrices are ``2n x 2n``), :meth:`~MatGroup.singer_cycle` (a
  cyclic subgroup of order ``q**n - 1``) and :meth:`~MatGroup.trivial`
- :meth:`~MatGroup.vector_orbit` / :meth:`~MatGroup.vector_orbits`, and
  :meth:`~MatGroup.projective_orbit` / :meth:`~MatGroup.projective_orbits` on
  the ``(q**d - 1)/(q - 1)`` one-dimensional subspaces, returned as canonical
  representatives (first non-zero coordinate scaled to ``1``)
- :meth:`~MatGroup.permutation_action_on_vectors` (faithful, so the
  :class:`PermutationGroup` it returns has exactly ``order()`` elements) and
  :meth:`~MatGroup.permutation_action_on_projective_points`, which is
  ``PGL``/``PSL``/``PSp``: **not** faithful, its order being ``|G|`` divided by
  the number of scalars in ``G``. ``PSL(2, 7)`` is the projective action of
  ``SL(2, 7)`` on the 8 points of ``PG(1, 7)``, of order 168
- :meth:`~MatGroup.random_element` by product replacement, seeded so it is
  reproducible. Every value is a product of generators, so it is in the group
  by construction; the distribution is the standard heuristic and is not
  claimed uniform
- :meth:`~MatGroup.derived_subgroup`, :meth:`~MatGroup.centre` and
  :meth:`~MatGroup.normal_closure`. The centre avoids backtrack search: it is
  ``G`` intersected with the algebra of matrices commuting with the generators,
  and that algebra is a **nullspace** (:meth:`~MatGroup.commutant_basis`)
- Scope: the caps are ``MATGROUP_MAX_DEGREE``, ``MATGROUP_MAX_ORBIT`` (a cap on
  ``q**d``, not on ``|G|``), ``MATGROUP_MAX_SCHREIER_WORK``,
  ``MATGROUP_DEFAULT_ELEMENT_CAP`` and ``MATGROUP_MAX_FIELD_ORDER``. Each is a
  typed ``MatGroupError`` refusal; in particular an exhausted Schreier–Sims
  budget reports **no order at all**, because an incomplete chain reports a
  proper divisor of the true one. ``GU``/``SU``/``SO``/``O`` and the twisted
  types have no constructor here — build them from your own generators.
  ``PGL``/``PSL`` exist only as permutation groups, because a quotient of
  matrix groups is not a matrix group

Finitely presented groups (``alkahest_cas::fpgroup``):
- :class:`FpGroup` — a presentation ``<X | R>``, built from generator names and
  relator strings: ``FpGroup(["a", "b"], ["a^2", "b^3", "(a*b)^5"])`` is
  ``A5``. Relators are products of powers of names and parenthesised sub-words,
  and juxtaposition is multiplication, so ``"ab"`` and ``"a*b"`` agree
- :meth:`~FpGroup.order`, :meth:`~FpGroup.index`,
  :meth:`~FpGroup.coset_table` and :meth:`~FpGroup.permutation_group` — HLT
  Todd–Coxeter coset enumeration with lookahead. **These can refuse**, and the
  two refusals are different facts: ``E-FPGRP-004`` means the coset cap was
  reached and says *nothing* about whether the group is finite, while
  ``E-FPGRP-005`` is a **proof** that it is infinite. The ``(2,3,7)`` triangle
  group ``FpGroup(["a", "b"], ["a^2", "b^3", "(a*b)^7"])`` is infinite with a
  trivial abelianisation, so it lands on ``E-FPGRP-004``
- :meth:`~FpGroup.abelian_invariants` — ``G/[G, G]`` from a Smith normal form,
  and the one question here that **always terminates**
- :meth:`~FpGroup.subgroup_presentation` — Reidemeister–Schreier, unsimplified
- :meth:`~FpGroup.cohomology` — ``H^0``, ``H^1`` and ``H^2`` of a finite group
  with coefficients in a finitely generated abelian module, from the bar
  resolution. ``H^2`` classifies extensions. Capped at ``|G| <=
  FPGROUP_MAX_COHOMOLOGY_GROUP_ORDER`` and ``rank(M) * |G|**(d+1) <=
  FPGROUP_MAX_COCHAIN_DIMENSION``; an action whose matrices do not satisfy the
  relators is refused (``E-FPGRP-011``), never used

Numeric ODE integrators (Phase 16b):
- :func:`ode_integrate_rk4` — fixed-step 4th-order Runge–Kutta integrator
- :func:`ode_integrate_rk45` — adaptive Dormand–Prince RK4(5) integrator
- :class:`OdeTrajectory` — sampled trajectory returned by the integrators
"""

from __future__ import annotations

import contextlib

# Graduated symbols — re-exported from the stable top-level for callers that
# still import ``alkahest.experimental``.
from alkahest import (
    Assumptions,
    EvaluationResult,
    arg,
    bessel_j0,
    bessel_j1,
    conjugate,
    digamma,
    evaluate,
    im,
    lambert_w,
    re,
    residue,
    to_stablehlo,
)

# M4(b) — q-analogue creative telescoping.  The engine is in the kernel; the
# two term builders are sugar over `pool.func`.
from alkahest._qterm import qbinomial, qpochhammer

# M5 — recurrence -> asymptotics.  The dispatch over the three shapes a
# recurrence arrives in is Python; the mathematics is in the kernel.
from alkahest._recurrence_asymptotics import (
    RecurrenceAsymptotics,
    asymptotics_from_recurrence,
)

# Calculus / ODE / transform surface (still experimental).
from alkahest.alkahest import (
    # Caps on conjugacy-class enumeration and on the cyclotomic field a
    # character table's values live in. The first bounds |G| (every element is
    # held in memory), the second bounds phi(exp G) and not the order at all.
    CHARACTER_DEFAULT_CLASS_CAP,
    CHARACTER_DEFAULT_TABLE_CAP,
    CHARACTER_MAX_CLASS_CAP,
    CHARACTER_MAX_EXPONENT_FIELD_DEGREE,
    # Caps on the finitely-presented-group layer: Todd-Coxeter cosets, the
    # presentation rank, and the cohomology's group order / module rank /
    # cochain dimension. Above the coset cap the call raises E-FPGRP-004,
    # which means "did not complete" and NOT "the group is infinite".
    FPGROUP_DEFAULT_MAX_COSETS,
    FPGROUP_MAX_COCHAIN_DIMENSION,
    FPGROUP_MAX_COHOMOLOGY_DEGREE,
    FPGROUP_MAX_COHOMOLOGY_GROUP_ORDER,
    FPGROUP_MAX_FREE_RANK,
    FPGROUP_MAX_MODULE_RANK,
    # Caps on permutation-group element enumeration and on Schreier–Sims
    GROUP_DEFAULT_ELEMENT_CAP,
    GROUP_MAX_BSGS_DEGREE,
    GROUP_MAX_ELEMENT_CAP,
    # Caps on exact lattice enumeration (SVP/CVP/theta series)
    LATTICE_DEFAULT_ENUM_NODE_BUDGET,
    LATTICE_MAX_ENUM_RANK,
    LATTICE_MAX_THETA_NORM,
    # Caps on the matrix-group layer: degree, basic orbit size, Schreier–Sims
    # work budget, element enumeration, GF(q) enumeration and the centralizing
    # algebra the centre is read off
    MATGROUP_DEFAULT_ELEMENT_CAP,
    MATGROUP_MAX_COMMUTANT_ELEMENTS,
    MATGROUP_MAX_DEGREE,
    MATGROUP_MAX_FIELD_ORDER,
    MATGROUP_MAX_ORBIT,
    MATGROUP_MAX_SCHREIER_WORK,
    # Caps on the stabilizer layer: qubits, the exhaustive distance search,
    # and brute-force matrix-group enumeration
    STABILIZER_MAX_DISTANCE_SEARCH_DIM,
    STABILIZER_MAX_MATRIX_ENUMERATION,
    STABILIZER_MAX_QUBITS,
    # The classical arithmetic functions' work cap (E-NT-006). A *subclass*
    # of alkahest.NumberTheoryError, so `except NumberTheoryError` catches
    # it together with the domain refusals (E-NT-001 / E-NT-002).
    # A finitely generated abelian group as a free rank plus invariant
    # factors — the answer type of `FpGroup.abelian_invariants()` and of
    # `FpGroup.cohomology()`.
    AbelianInvariants,
    ArithmeticError,
    # P1 item 10 — asymptotic expansion at scale
    AsymptoticReport,
    Bernoulli,
    Beta,
    Binomial,
    # Conjugacy classes and exact character tables of permutation groups, by
    # Dixon-Schneider. Values are elements of Q(zeta_exp G), so A_4's cube
    # roots of unity and A_5's golden-ratio pair are values, not
    # approximations. Row/column orthogonality is checked before a table is
    # returned; a violation is E-CHAR-007 and nothing comes back.
    CharacterError,
    CharacterTable,
    # Classical linear codes, weight enumerators and the certified Delsarte
    # LP bound on A_q(n, d) — exact rational arithmetic throughout, and the
    # bound comes with the dual certificate that proves it.
    CodingError,
    # Riemann theta, modular and Weierstrass functions as rigorous
    # enclosures (FLINT/Arb). Every value is a `ComplexBall` carrying its
    # own error bound; `ComplexBall.value()` refuses rather than hand back
    # a midpoint with nothing behind it.
    ComplexBall,
    ConjugacyClass,
    ConjugacyClasses,
    Coordinates,
    CosetTable,
    # Binary symplectic / stabilizer codes
    CssCode,
    DelsarteBound,
    Distance,
    Distribution,
    Divisor,
    DivisorClass,
    Exponential,
    FiniteField,
    FiniteFieldError,
    # Finitely presented groups <X | R>: Todd-Coxeter enumeration, the
    # permutation representation on the cosets, the abelianisation,
    # Reidemeister-Schreier, and H^0/H^1/H^2.
    FpGroup,
    FpGroupError,
    Fps,
    FunctionField,
    FunctionFieldElement,
    FunctionFieldError,
    Gamma,
    GfMatrix,
    GfRref,
    GroupError,
    # Lattices: standard families, exact SVP/CVP, theta series, densities
    Lattice,
    LatticeGeometryError,
    LatticeVector,
    LinearCode,
    LogNormal,
    # Matrix groups over GF(q) from *arbitrary* generators: order and
    # membership from a Schreier–Sims BSGS on the action on vectors, orbits on
    # vectors and on projective points, product-replacement random elements,
    # derived subgroup, centre and normal closure. Supersedes MatrixGroup,
    # which describes GL/SL/Sp by formula and has no generators.
    MatGroup,
    MatGroupError,
    MatrixGroup,
    MatSiftResult,
    Normal,
    NumberField,
    NumberFieldElement,
    NumberFieldError,
    OdeTrajectory,
    ParallelRischResult,
    PauliOperator,
    # Computational group theory (permutation groups)
    Permutation,
    PermutationGroup,
    Place,
    Poisson,
    # Puiseux (fractional-exponent) expansion — verified before it is returned
    PuiseuxExpansion,
    QRootOfUnitySpecialization,
    # Vector calculus over orthogonal charts, and Hamilton quaternions
    Quaternion,
    QuaternionError,
    QZeilbergerCertificate,
    RiemannRochSpace,
    SiegelMatrix,
    SiegelReduction,
    SiftResult,
    StabilizerCode,
    StabilizerError,
    StabilizerGroup,
    SubgroupPresentation,
    Telescoping2dCertificate,
    TelescopingMdCertificate,
    ThetaError,
    ThetaValues,
    Uniform,
    VectorError,
    WeightEnumerator,
    Word,
    apart_side_conditions,
    arb_backend_available,
    asymptotic_expand,
    bernoulli_number,
    # P1 item 10 — asymptotic expansion at scale
    coefficient_asymptotics,
    cross,
    cross_entropy,
    curl,
    cyclotomic_polynomial,
    cyclotomic_polynomial_coeffs,
    dedekind_eta,
    delsarte_lp_bound,
    dirac_delta,
    divergence,
    divisor_sigma,
    dot,
    dsolve,
    dsolve_system,
    eisenstein_series,
    euler_maclaurin,
    euler_number,
    expectation,
    expectation_affine,
    fourier_transform,
    gradient,
    hamming_bound,
    harmonic_number,
    heaviside,
    integrate_parallel_risch,
    inverse_fourier_transform,
    inverse_laplace_transform,
    inverse_z_transform,
    is_symplectic,
    j_invariant,
    jacobi_theta,
    jacobi_theta_null,
    kl_divergence,
    krawtchouk,
    krawtchouk_poly,
    laplace_transform,
    laplacian,
    # The closed-form classical orders, so a caller can check MatGroup.order()
    # (which is computed by Schreier–Sims) against the product formula.
    matgroup_gl_order,
    matgroup_sl_order,
    matgroup_sp_order,
    modular_discriminant,
    modular_lambda,
    moebius_mu,
    multilimit,
    mutual_information_independent,
    norm,
    ode_integrate_rk4,
    ode_integrate_rk45,
    partition_number,
    prob_side_conditions,
    puiseux_series,
    q_zeilberger,
    riemann_roch,
    riemann_theta,
    riemann_theta_available,
    riemann_theta_characteristic,
    riemann_theta_squared,
    series_solve,
    siegel_is_reduced,
    siegel_reduce,
    singleton_bound,
    stirling_first,
    stirling_first_unsigned,
    stirling_second,
    sum_of_squares,
    symplectic_complement,
    symplectic_form,
    symplectic_gram_matrix,
    symplectic_gram_schmidt,
    telescope2d,
    telescope_md,
    theta_characteristic_bits,
    theta_characteristic_index,
    theta_characteristic_is_even,
    transform_side_conditions,
    variance_affine_independent,
    vector_laplacian,
    weierstrass_invariants,
    weierstrass_p,
    weierstrass_p_prime,
    weierstrass_roots,
    weierstrass_sigma,
    weierstrass_zeta,
    z_transform,
)

# M11 — novelty filtering.  Claim normalisation, a stable hash, and a lookup
# whose negative is never reported as "novel".  Pure Python by CONTRIBUTING's
# rule: HTTP, JSON, and parsing a third party's prose.
from alkahest.experimental import novelty
from alkahest.experimental.novelty import (
    NoveltyMatch,
    NoveltyVerdict,
    OeisCache,
    OeisEntry,
    OeisWeb,
    QRecurrenceClaim,
    RecurrenceClaim,
    check_novelty,
)

with contextlib.suppress(ImportError):
    from alkahest import to_lean

with contextlib.suppress(ImportError):
    from alkahest._jax import to_jax

with contextlib.suppress(ImportError):
    from alkahest import GbPoly, GroebnerBasis, solve

# M9 — Gröbner bases over the coefficient field Q(params).  Registered by the
# extension only on `groebner` builds, hence the suppressed import.
with contextlib.suppress(ImportError):
    from alkahest.alkahest import (
        ParametricGbPoly,
        ParametricGroebnerBasis,
        ParametricRosenfeldGroebnerResult,
    )

with contextlib.suppress(ImportError):
    from alkahest.alkahest import CudaCompiledFn, compile_cuda

__all__ = [
    # Conjugacy classes and character tables. CHARACTER_*_CLASS_CAP and
    # CHARACTER_DEFAULT_TABLE_CAP bound |G|, because the class partition holds
    # every element; CHARACTER_MAX_EXPONENT_FIELD_DEGREE bounds phi(exp G), the
    # degree of the one cyclotomic field every value lives in, and is a limit
    # on the exponent rather than on the order.
    "CHARACTER_DEFAULT_CLASS_CAP",
    "CHARACTER_DEFAULT_TABLE_CAP",
    "CHARACTER_MAX_CLASS_CAP",
    "CHARACTER_MAX_EXPONENT_FIELD_DEGREE",
    # Caps on finitely presented groups. Above FPGROUP_DEFAULT_MAX_COSETS a
    # coset enumeration raises E-FPGRP-004, which means "did not complete
    # within the cap" and is **not** a claim that the group is infinite; that
    # claim is E-FPGRP-005, and only the abelianisation can prove it.
    "FPGROUP_DEFAULT_MAX_COSETS",
    "FPGROUP_MAX_COCHAIN_DIMENSION",
    "FPGROUP_MAX_COHOMOLOGY_DEGREE",
    "FPGROUP_MAX_COHOMOLOGY_GROUP_ORDER",
    "FPGROUP_MAX_FREE_RANK",
    "FPGROUP_MAX_MODULE_RANK",
    # Computational group theory (permutation groups)
    "GROUP_DEFAULT_ELEMENT_CAP",
    "GROUP_MAX_BSGS_DEGREE",
    "GROUP_MAX_ELEMENT_CAP",
    # Caps on exact lattice enumeration; above them the call raises
    # LatticeError (E-LAT-008 / E-LAT-009) rather than approximating
    "LATTICE_DEFAULT_ENUM_NODE_BUDGET",
    "LATTICE_MAX_ENUM_RANK",
    "LATTICE_MAX_THETA_NORM",
    # Caps on matrix groups over GF(q). MATGROUP_MAX_ORBIT is a cap on q**d and
    # *not* on |G|, because an orbit on vectors is bounded by q**d - 1; and
    # MATGROUP_MAX_SCHREIER_WORK is a refusal (E-MATGRP-007) rather than a
    # partial chain, because a chain missing a level reports a proper divisor
    # of the true order.
    "MATGROUP_DEFAULT_ELEMENT_CAP",
    "MATGROUP_MAX_COMMUTANT_ELEMENTS",
    "MATGROUP_MAX_DEGREE",
    "MATGROUP_MAX_FIELD_ORDER",
    "MATGROUP_MAX_ORBIT",
    "MATGROUP_MAX_SCHREIER_WORK",
    # Binary symplectic / stabilizer codes: the (x | z) form over GF(2),
    # Pauli operators with a Z4 phase, stabilizer and CSS codes, and the
    # classical matrix groups GL / SL / Sp over GF(q). Distance is exhaustive
    # and capped; above the cap it refuses (E-STAB-008) rather than guessing,
    # and a bound comes back as `Distance(at_most=...)`, not as an int.
    "STABILIZER_MAX_DISTANCE_SEARCH_DIM",
    "STABILIZER_MAX_MATRIX_ENUMERATION",
    "STABILIZER_MAX_QUBITS",
    # The classical arithmetic functions' work cap (E-NT-006). A *subclass*
    # of alkahest.NumberTheoryError, so `except NumberTheoryError` catches it
    # together with the domain refusals (E-NT-001 / E-NT-002).
    # A finitely generated abelian group: free rank plus invariant factors.
    # `order()` returns None exactly when the group is infinite — a proof, not
    # a failure to decide.
    "AbelianInvariants",
    "ArithmeticError",
    "Assumptions",
    # P1 item 10 — asymptotic expansion at scale
    "AsymptoticReport",
    # Probability: a law is a name plus symbolic parameters
    "Bernoulli",
    "Beta",
    "Binomial",
    # Classical coding theory. The Delsarte LP bound is *certified*: the
    # number is read off the dual programme and comes with the non-negative
    # multipliers that prove it, all in exact rational arithmetic, so it is an
    # upper bound rather than a floating-point estimate of one. A bound that
    # came out too small would "rule out" codes that exist, so the constructor
    # refuses (E-CODE-007) rather than return one whose certificate did not
    # re-verify.
    # Conjugacy classes and the exact ordinary character table, by
    # Dixon–Schneider. Values are elements of Q(zeta_exp G) rather than floats
    # or rationals-when-convenient, so A_4's cube roots of unity and A_5's
    # golden-ratio pair come out of the same code path that gives S_4 its
    # integers. Row and column orthogonality, sum chi(1)**2 == |G| and one
    # character per class are checked as exact identities before a table is
    # returned: a table that failed them is withheld (E-CHAR-007), never
    # returned with a caveat.
    "CharacterError",
    "CharacterTable",
    "CodingError",
    # Riemann theta functions, the classical modular functions and the
    # Weierstrass family, as rigorous enclosures backed by FLINT's Arb
    # layer. Genus 1 and 2 are the tested regime; see the module docs for
    # the genus ceiling and what `Precision.AccurateTo` refuses.
    "ComplexBall",
    "ConjugacyClass",
    "ConjugacyClasses",
    # Vector calculus over orthogonal curvilinear charts
    "Coordinates",
    # A complete Todd-Coxeter coset table, and the permutation representation
    # it is. Cosets are 0-based and coset 0 is the subgroup H.
    "CosetTable",
    "CssCode",
    "CudaCompiledFn",
    "DelsarteBound",
    "Distance",
    "Distribution",
    "Divisor",
    "DivisorClass",
    "EvaluationResult",
    "Exponential",
    # Linear algebra over GF(q), q = p^k — built for linear codes. Rectangular
    # shapes are first-class and `GfMatrix.nullspace()` over GF(2) is the path
    # the rest is arranged around. A non-prime characteristic or a reducible
    # defining polynomial is refused (E-GFQ-001 / E-GFQ-004) rather than
    # approximated: the quotient would be a ring with zero divisors, in which
    # "rank" and "nullspace" are not well defined.
    "FiniteField",
    "FiniteFieldError",
    # Finitely presented groups <X | R>. `order()` is the index of the trivial
    # subgroup by Todd-Coxeter, which may refuse (E-FPGRP-004) because the word
    # problem is undecidable; `abelian_invariants()` always terminates and can
    # prove a group infinite (E-FPGRP-005). `cohomology(2, ...)` classifies
    # extensions, and is capped at small groups and modules rather than being
    # general-looking and unverifiable.
    "FpGroup",
    "FpGroupError",
    "Fps",
    "FunctionField",
    "FunctionFieldElement",
    "FunctionFieldError",
    "Gamma",
    "GbPoly",
    "GfMatrix",
    "GfRref",
    "GroebnerBasis",
    "GroupError",
    # Lattices. `Lattice.from_gram` is the general constructor — E_8 and the
    # Leech lattice have no rational basis in their natural embedding, and
    # every invariant here (determinant, dual, minimum, kissing number, theta
    # series, densities) depends on the Gram matrix alone. SVP/CVP/theta are
    # exact enumeration, capped at LATTICE_MAX_ENUM_RANK.
    "Lattice",
    # The toolkit's refusals (E-LAT-005 … E-LAT-014). A *subclass* of
    # alkahest.LatticeError, so `except LatticeError` catches the reduction
    # errors (E-LAT-001 … E-LAT-004) and these together.
    "LatticeGeometryError",
    "LatticeVector",
    "LinearCode",
    # Probability (continued)
    "LogNormal",
    # Matrix groups over GF(q) from *arbitrary* generators. The action is on
    # **row** vectors, v -> v @ M, so vectors are 1 x d GfMatrix objects.
    # `order()` is a Schreier-Sims computation, not a formula — the three
    # `matgroup_*_order` functions give the closed forms to check it against.
    "MatGroup",
    # E-MATGRP-001 … E-MATGRP-013. Wraps FiniteFieldError and GroupError, which
    # keep their own E-GFQ-* / E-GRP-* codes.
    "MatGroupError",
    "MatSiftResult",
    "MatrixGroup",
    "Normal",
    # M11 — novelty filtering
    "NoveltyMatch",
    "NoveltyVerdict",
    # Algebraic number fields Q[x]/(f), on FLINT's `nf`/`nf_elem`, including
    # the cyclotomic fields Q(zeta_n) that lattice cryptography and ZK proof
    # systems are built over. The defining polynomial is **checked** for
    # irreducibility (E-NUMF-003), because Q[x]/(f) for reducible f is a ring
    # with zero divisors in which `inverse` has no answer. Note
    # `polynomial_discriminant` is the discriminant of the defining
    # polynomial, not the field discriminant: the ring of integers is not
    # computed.
    "NumberField",
    "NumberFieldElement",
    "NumberFieldError",
    "OdeTrajectory",
    # M11 — novelty filtering
    "OeisCache",
    "OeisEntry",
    "OeisWeb",
    "ParallelRischResult",
    # M9 — coefficient fields for elimination
    "ParametricGbPoly",
    "ParametricGroebnerBasis",
    "ParametricRosenfeldGroebnerResult",
    # Risch-Norman heuristic integration result
    # Computational group theory (permutation groups)
    "PauliOperator",
    "Permutation",
    "PermutationGroup",
    "Place",
    # Probability (continued)
    "Poisson",
    # Puiseux (fractional-exponent) expansion
    "PuiseuxExpansion",
    # M11 — novelty filtering
    "QRecurrenceClaim",
    # M4 — root-of-unity specialisation
    "QRootOfUnitySpecialization",
    # M4(b) — q-analogue creative telescoping
    "QZeilbergerCertificate",
    # Hamilton quaternions and the rotation operator q v q⁻¹
    "Quaternion",
    "QuaternionError",
    # M5 — recurrence -> asymptotics
    "RecurrenceAsymptotics",
    # M11 — novelty filtering
    "RecurrenceClaim",
    "RiemannRochSpace",
    "SiegelMatrix",
    "SiegelReduction",
    # Sifting an element through a stabilizer chain
    "SiftResult",
    "StabilizerCode",
    "StabilizerError",
    "StabilizerGroup",
    # A Reidemeister-Schreier presentation of a finite-index subgroup. Not
    # simplified: no Tietze pass, so the generator count is the raw
    # [G:H]*|X| - [G:H] + 1.
    "SubgroupPresentation",
    # M4 — double-sum (Apagodu-Zeilberger) creative telescoping
    "Telescoping2dCertificate",
    "TelescopingMdCertificate",
    "ThetaError",
    "ThetaValues",
    # Probability (continued)
    "Uniform",
    "VectorError",
    "WeightEnumerator",
    # A free-group word: signed, 1-based generator indices, freely reduced on
    # construction.
    "Word",
    # Hypotheses the last `apart` on this thread rests on (ℚ(params) path).
    "apart_side_conditions",
    "arb_backend_available",
    "arg",
    "asymptotic_expand",
    # M5 — recurrence -> asymptotics
    "asymptotics_from_recurrence",
    "bernoulli_number",
    "bessel_j0",
    "bessel_j1",
    # M11 — novelty filtering
    "check_novelty",
    # P1 item 10 — asymptotic expansion at scale
    "coefficient_asymptotics",
    "compile_cuda",
    "conjugate",
    # Vector calculus
    "cross",
    # Information theory: H(P, Q) = H(P) + D(P||Q), checked against -E_P[log q]
    "cross_entropy",
    "curl",
    # M4 — root-of-unity specialisation
    "cyclotomic_polynomial",
    "cyclotomic_polynomial_coeffs",
    "dedekind_eta",
    "delsarte_lp_bound",
    "digamma",
    "dirac_delta",
    # Vector calculus
    "divergence",
    "divisor_sigma",
    "dot",
    "dsolve",
    "dsolve_system",
    "eisenstein_series",
    "euler_maclaurin",
    "euler_number",
    "evaluate",
    "expectation",
    "expectation_affine",
    "fourier_transform",
    # Vector calculus
    "gradient",
    "hamming_bound",
    "harmonic_number",
    "heaviside",
    "im",
    # Risch-Norman (parallel Risch) heuristic integrator.  Returns a result
    # object, never raises: a declined result is not a non-elementarity verdict.
    "integrate_parallel_risch",
    "inverse_fourier_transform",
    "inverse_laplace_transform",
    "inverse_z_transform",
    "is_symplectic",
    "j_invariant",
    "jacobi_theta",
    "jacobi_theta_null",
    # Information theory: not symmetric, and +inf off a nested support
    "kl_divergence",
    # Eigenvalues of the Hamming association scheme, exactly
    "krawtchouk",
    "krawtchouk_poly",
    "lambert_w",
    "laplace_transform",
    # Vector calculus
    "laplacian",
    # The classical orders in closed form, as an oracle for MatGroup.order()
    "matgroup_gl_order",
    "matgroup_sl_order",
    "matgroup_sp_order",
    "modular_discriminant",
    "modular_lambda",
    "moebius_mu",
    "multilimit",
    # Information theory: 0, under an independence the caller asserts
    "mutual_information_independent",
    "norm",
    # M11 — novelty filtering (the module itself, for `novelty.RecordedRecurrence`
    # and the status tables)
    "novelty",
    "ode_integrate_rk4",
    "ode_integrate_rk45",
    "partition_number",
    # Hypotheses the last Distribution method / expectation call rests on
    "prob_side_conditions",
    # Puiseux (fractional-exponent) expansion
    "puiseux_series",
    # M4(b) — q-analogue creative telescoping
    "q_zeilberger",
    "qbinomial",
    "qpochhammer",
    "re",
    "residue",
    "riemann_roch",
    "riemann_theta",
    "riemann_theta_available",
    "riemann_theta_characteristic",
    "riemann_theta_squared",
    "series_solve",
    "siegel_is_reduced",
    "siegel_reduce",
    "singleton_bound",
    "solve",
    "stirling_first",
    "stirling_first_unsigned",
    "stirling_second",
    "sum_of_squares",
    "symplectic_complement",
    "symplectic_form",
    "symplectic_gram_matrix",
    "symplectic_gram_schmidt",
    # M4 — double-sum (Apagodu-Zeilberger) creative telescoping
    "telescope2d",
    "telescope_md",
    "theta_characteristic_bits",
    "theta_characteristic_index",
    "theta_characteristic_is_even",
    "to_jax",
    "to_lean",
    "to_stablehlo",
    # Hypotheses the last inverse Laplace / Z transform on this thread rests on.
    "transform_side_conditions",
    "variance_affine_independent",
    # Vector calculus
    "vector_laplacian",
    "weierstrass_invariants",
    "weierstrass_p",
    "weierstrass_p_prime",
    "weierstrass_roots",
    "weierstrass_sigma",
    "weierstrass_zeta",
    "z_transform",
]
