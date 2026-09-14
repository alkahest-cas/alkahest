# Vector calculus and quaternions

**Experimental.** Everything on this page lives in `alkahest.experimental` and may
change in a minor release — see [Stability policy](./stability.md).

```python
import alkahest as ak
from alkahest import experimental as ex
```

## Charts, not scale factors

`Coordinates` is an **orthogonal** chart. There is no constructor taking scale
factors directly — the three built-in charts carry textbook values, and anything
else has to come from an embedding whose orthogonality was *checked*:

```python
pool = ak.ExprPool()
x, y, z = (pool.symbol(n) for n in ("x", "y", "z"))
rho, phi = pool.symbol("rho"), pool.symbol("phi")
r, theta = pool.symbol("r"), pool.symbol("theta")

cart = ex.Coordinates.cartesian(x, y, z)          # h = (1, 1, 1)
cyl  = ex.Coordinates.cylindrical(rho, phi, z)    # h = (1, rho, 1)
sph  = ex.Coordinates.spherical(r, theta, phi)    # h = (1, r, r·sin θ)

sph.vars()            # (r, theta, phi)
sph.scale_factors()   # (1, r, r*sin(theta))
sph.label             # 'spherical'
```

`Coordinates.from_embedding(coordinates, embedding, label=None)` derives the
scale factors from a Cartesian parametrisation and **refuses** (`E-VEC-004`) a
chart it cannot *prove* orthogonal. That refusal is the point of the type: every
`grad`/`div`/`curl`/`∇²` formula in the module is derived for an orthogonal
frame, and on a skew chart they still evaluate — to an expression that looks like
a divergence and is not one. Undecided is a refusal, not an assumption. A scale
factor that is identically zero, or whose non-vanishing could not be established,
refuses with `E-VEC-005`, because every operator divides by it.

## Fields are physical components

Vector fields are three **physical** components, read in the local orthonormal
frame. That is the convention `sympy.vector` uses and the one an engineer means
by "the ρ component".

```python
f = x * y + z
F = [x, y, z]

ex.gradient(f, cart)          # (y, x, 1)
ex.divergence(F, cart)        # 3
ex.curl(F, cart)              # (0, 0, 0)
ex.laplacian(f, cart)         # 0
ex.vector_laplacian(F, cart)  # (0, 0, 0)

ex.dot(F, F)                  # x² + y² + z²
ex.cross(F, [pool.integer(0), pool.integer(0), pool.integer(1)])
ex.norm(F)                    # sqrt(x² + y² + z²)
```

### `vector_laplacian` is not `laplacian` three times

`ex.vector_laplacian` is `∇(∇·F) − ∇×(∇×F)`. That equals the componentwise
scalar Laplacian **in Cartesian coordinates only**. Applying `ex.laplacian` to
each physical component of a cylindrical or spherical field silently drops the
terms that come from the basis turning — `∇²(φ̂)` is `−φ̂/ρ²`, not `0`. This is a
trap with no error code, because there *is* a right answer; the entry point
exists so nobody has to derive it.

## Quaternions

`Quaternion(w, x, y, z)` is a Hamilton quaternion `w + xi + yj + zk`, built from
four `Expr`s. `i² = j² = k² = ijk = −1`, so `*` does not commute:

```python
i = ex.Quaternion(pool.integer(0), pool.integer(1), pool.integer(0), pool.integer(0))
j = ex.Quaternion(pool.integer(0), pool.integer(0), pool.integer(1), pool.integer(0))

(i * j).components()   # (0, 0, 0,  1)   ==  k
(j * i).components()   # (0, 0, 0, -1)   == -k
```

The surface is `conjugate`, `inverse`, `norm`, `norm_squared`, `normalize`,
`vector_part`, `components`, `simplify`, `rotate`, `to_rotation_matrix` /
`from_rotation_matrix`, `to_axis_angle` / `from_axis_angle`, and
`Quaternion.identity(pool)`.

`rotate(v)` is the **active** rotation `v ↦ q v q⁻¹` in a right-handed frame, and
composition follows from it: `(q1 * q2)` applies `q2` first, so
`(q1 * q2).to_rotation_matrix()` is `q1.to_rotation_matrix() @
q2.to_rotation_matrix()`.

### The two refusals

| Code | Raised by | Why a refusal rather than an answer |
|---|---|---|
| `E-QUAT-002` | `to_axis_angle` | The identity rotation has no axis: *every* unit vector is one. The conventional stand-in `(0, 0, 1)` is a stated answer to a question with no answer, and nothing downstream can tell it from a real axis |
| `E-QUAT-003` | `from_rotation_matrix` | Shepperd's method returns a perfectly ordinary unit quaternion for a reflection, a scaled matrix or a shear — representing some *other*, proper rotation. So `RᵀR = I` and `det R = +1` are checked, the recovered quaternion must reproduce the matrix, and a **symbolic** matrix refuses outright, because the branch selection is a comparison between entries and there is none to make on a symbol |

`E-QUAT-001` is the zero-or-undecided norm: `inverse`, `normalize` and
`to_axis_angle` all divide by it.

```python
try:
    ex.Quaternion.identity(pool).to_axis_angle()
except ak.experimental.QuaternionError as e:
    print(e.code)          # E-QUAT-002
```

`VectorError` and `QuaternionError` are reached as
`alkahest.experimental.VectorError` / `.QuaternionError` (they are not on the
top-level namespace), and both subclass `alkahest.AlkahestError`, so
`except ak.AlkahestError` catches them.

## See also

* [Error handling](./errors.md#vector-calculus-and-quaternions) — the full
  refusal table
* [Probability and information theory](./probability.md) — the other surface
  added in the same cycle
