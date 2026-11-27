# Green's functions and fundamental solutions

> **Experimental.** Everything on this page lives in `alkahest.experimental`
> (Rust: `alkahest_cas::pde::green`) and may change between minor releases.

A Green's function turns a linear boundary-value problem into an integral:
`u = ∫ G f + boundary terms`. Alkahest builds them four ways:

| function | what it builds | result |
|---|---|---|
| `fundamental_solution` | the whole-space kernel of a constant-coefficient operator | `FundamentalSolution` |
| `image_green_function` | half- and quarter-spaces, strips and slabs, the disk and ball, by images | `GreenFunction` |
| `green_function` | a `SturmLiouville` problem: the closed form by variation of parameters, or the eigenfunction series (static, heat, wave) | `GreenFunction` |
| `green_solve` | the solution of a `PDE` as `∫ G f + boundary terms` | `GreenSolution` |

**Convention.** `G` solves `L G = δ` for the operator **exactly as written**,
which is Mathematica's `GreenFunction` convention: `Δ G = δ` gives
`G = log(r)/(2π)` in the plane, with no minus sign. Evolution kernels are
causal (`L G = δ(x) δ(t)`, zero for `t < 0`), and Helmholtz kernels are
outgoing (`e^{+ikr}`). The Sturm–Liouville Green's functions are for
`𝓛u = −[(p u')' + q u]`, the sign convention of `SturmLiouville`
(`𝓛φₙ = λₙ w φₙ`).

## How an answer is checked

`verified` is `True` only when **every** check is an exact proof. A numeric
check can make Alkahest refuse an answer, but it never certifies one. Each
result carries its `checks` (dicts with `what`, `status`, `method` and
`detail`) and the theorem that turns them into a proof of `L G = δ`:

* `L G = 0` off the singular set, by substitution. Residuals with Bessel
  functions are reduced by the three-term recurrences.
* For 1-D operators, the jump formula: continuity of `G` and a jump of
  `1/a₂` in `∂G` at the source prove `L G = δ` distributionally.
* For elliptic kernels in 2 and 3 dimensions, unit flux through a small
  sphere around the source. For `K₁` and `H₁⁽¹⁾` this uses the DLMF
  leading terms.
* For the heat kernel, unit mass and the transfer function.
* For wave kernels, `∫ G dx = t`, and for Klein–Gordon and telegraph,
  the value 1 of the Riemann function on the characteristics.
* For images, the wall conditions by substitution. An image series is
  checked term by term with the pairs that cancel (`j ↔ −j`, `j ↔ 1 − j`).
* For variation of parameters, continuity, the jump `−1/p(ξ)`, both end
  conditions, symmetry and Abel's identity.

A construction that fails its own check is withheld with `E-PDE-255`.

## Fundamental solutions

```python
import alkahest as ak
import alkahest.experimental as akx

p = ak.ExprPool()
x, y = p.symbol("x"), p.symbol("y")

fs = akx.fundamental_solution("laplace", [x, y])
assert fs.verified
print(fs.kernel)        # log(sqrt(x^2 + y^2)) / (2 pi)
print(fs.identity)      # Δ G = δ(x)
print(sorted({c["what"] for c in fs.checks}))
```

The space variables stand for `x − ξ`. The operators are `"laplace"`,
`"biharmonic"`, `"helmholtz"` and `"modified_helmholtz"` (`k=`), `"heat"`
(`diffusivity=`), `"wave"` (`speed=`), `"klein_gordon"` (`speed=`, `mass=`)
and `"telegraph"` (`speed=`, `damping=`, for `u_tt + 2a u_t − c² u_xx`). All
of them are available in 1 to 3 space dimensions, except Klein–Gordon and
telegraph, which are 1-D only. For Laplace and heat, `dimension=n,
radius=r` gives the radial kernel in `n` dimensions with a symbolic `n`. A 1-D
kernel is proved by the jump formula:

```python
k = p.symbol("k", "positive")
fs1 = akx.fundamental_solution("helmholtz", [x], k=k)
assert fs1.verified
print(fs1.kernel)       # −i e^{ik|x|} / (2k)
print(fs1.identity_proof)
```

Each record also has `derivation`, `references`, `support`, `radial` (as a
function of `r`, and `t` for evolution kernels) and `side_conditions`.

## The method of images

`region` is `("half_space", axis)`, `("quarter_space", a, b)`,
`("strip", axis, width)`, `("disk", R)` or `("ball", R)`, where the axes index
`space`. `walls` gives one `"dirichlet"` or `"neumann"` per wall. For a strip,
the wall at 0 comes first. Strips and slabs come back as an infinite image
`Sum`. The disk and ball come from Kelvin inversion, and their result includes
the Poisson kernel `∂G/∂n_ξ`:

```python
xi, eta = p.symbol("xi"), p.symbol("eta")
R = p.symbol("R", "positive")
g = akx.image_green_function("laplace", [x, y], [xi, eta], ("disk", R), "dirichlet")
assert g.verified
print(g.method)          # kelvin
print(g.poisson_kernel)
```

When images do not apply, Alkahest refuses with `E-PDE-252`. Examples are the
Neumann disk (the Neumann function of the disk is not an image sum) and a
Helmholtz strip:

```python
try:
    akx.image_green_function("laplace", [x, y], [xi, eta], ("disk", p.integer(1)), "neumann")
except Exception as e:
    print(e)             # E-PDE-252 …
```

## Two-point problems and eigenfunction series

`green_function` takes a `SturmLiouville` problem (see `sturm_liouville`). With
`kind="closed"` it builds the closed form by variation of parameters. With
`"static"`, `"heat"` or `"wave"` it builds the eigenfunction series from the
problem's `Eigensystem`. Transcendental eigenvalues stay as `RootOfFamily`.
When the family is in closed form, the series is proved to be the expansion of
the closed form: `∫ G φₙ w dx = φₙ(ξ)/λₙ` for symbolic `n`.

```python
L, h = p.symbol("L", "positive"), p.symbol("h", "positive")
sl = akx.SturmLiouville(x, (0, L), ("dirichlet", ("robin", h, 1)))
closed = akx.green_function(sl, xi, "closed")
series = akx.green_function(sl, xi, "static")
assert closed.verified and series.verified
print(closed.branches)   # (G for x < ξ, G for x > ξ)
```

If `λ = 0` is an eigenvalue, no Green's function exists and `"closed"` refuses
with `E-PDE-254`. In that case the static series is the **modified** Green's
function, which is orthogonal to the zero mode:

```python
nn = akx.SturmLiouville(x, (0, L), ("neumann", "neumann"))
try:
    akx.green_function(nn, xi, "closed")
except Exception as e:
    print(e)             # E-PDE-254 …
g = akx.green_function(nn, xi, "static")
print(g.method)          # modified_eigenfunction_series
```

## Solving with a Green's function

`green_solve(pde)` returns `u = ∫ G f + boundary terms` for three classes of
problem:

* two-point problems `a₂u'' + a₁u' + a₀u = F` with Robin data at both ends,
  reduced to Sturm–Liouville form by an integrating factor;
* the heat and wave equations on the line with sources, by Duhamel's
  principle;
* Poisson's equation in the upper half-plane, by the Poisson kernel.

The integrals are closed wherever `integrate` closes them, and the result is
verified by substitution. A `GreenSolution` has the shape of `pdsolve`'s
`PdeSolution`: `kind` (`"closed_form"` or `"integral_representation"`),
`value`, `closed_part`, `terms`, the `green` function it used, `checks` and
`steps`.

```python
U = akx.Function("u")
u = U(x)
pde = (
    akx.PDE(-akx.D(u, x, 2) - p.integer(1), u)
    .on(akx.Interval(x, p.integer(0), p.integer(1)))
    .bc(ak.subs(u, {x: p.integer(0)}), 0)
    .bc(ak.subs(u, {x: p.integer(1)}), 0)
)
sol = akx.green_solve(pde)
assert sol.verified
print(sol.kind, sol.value)
```

A problem outside these classes is refused with `E-PDE-256`.

## Klein–Gordon and the telegraph equation

`transform_solve` returns Klein–Gordon and telegraph problems on the line as
light-cone integrals with Bessel kernels, not as Fourier integrals:

`u = e^{−Dt/2}(f(x − ct) + f(x + ct))/2 + ∫_{x−ct}^{x+ct} K_f f + ∫ K_g g`.

For Klein–Gordon the kernels are built from `J₀((m/c)√(c²t² − x²))`, and for
the telegraph equation from `e^{−at} I₀(…)`. Each kernel is verified to solve
the equation, and its Riemann function is verified to equal 1 on the
characteristics.

```python
t = p.symbol("t", "positive")
v = U(x, t)
kg = (
    akx.PDE(akx.D(v, t, 2) - akx.D(v, x, 2) + v, v)
    .on(akx.Line(x) * akx.HalfLine(t, 0))
    .ic(ak.subs(v, {t: p.integer(0)}), ak.exp(-(x**2)))
    .ic(ak.subs(akx.D(v, t), {t: p.integer(0)}), 0)
)
sol = akx.transform_solve(kg)
assert sol.verified
print(sol.kind)          # integral_representation
print(sol.terms[0]["kernel"])
```

## Comparison with Mathematica

The corpus `tests/pde_corpus/green_problems.json` has 59 problems from
Strauss (ch. 3 and 7), Evans (§2.2–2.4), Haberman (ch. 9 and 11) and Duffy.
`tests/pde_corpus/mathematica/green_ref.txt` holds the Mathematica 14.3
references (`GreenFunction` / `DSolveValue`, 120 s per problem), and
`tests/pde_corpus/score_m5.py` prints the table.

| category | problems | Alkahest, exactly verified | Mathematica, correct closed form |
|---|---|---|---|
| fundamental solutions | 20 | 20 | 9 (+1 wrong) |
| symbolic dimension `n` | 2 | 2 | 0 |
| images | 14 | 14 | 0 |
| two-point closed forms | 7 | 7 (one is the expected `E-PDE-254`) | 6 |
| eigenfunction series | 6 | 6 | 2 (2 not queried) |
| `green_solve` | 10 | 10 | 9 (+1 left as an integral) |
| **total** | **59** | **59** | **26** |

Where both systems return a closed form, the values agree at the sample
points. The one disagreement is the 3-D heat kernel. Mathematica returns
`e^{−r²/(4κt)} / (4πκ t^{3/2})`, which is missing a factor `(4π)^{−1/2}`, so its
mass is `√(4π)` rather than 1. `tests/test_pde_m5.py` documents this.

## Errors

`E-PDE-250` (no kernel here: Klein–Gordon or telegraph above 1-D, a symbolic
dimension for another operator), `251` (malformed input), `252` (images do not
apply), `253` (the series cannot be assembled), `254` (no Green's function
exists: `λ = 0` is an eigenvalue), `255` (a construction refuted by its own
check, withheld), `256` (`green_solve` does not recognise the problem). See
[Error handling](./errors.md).
