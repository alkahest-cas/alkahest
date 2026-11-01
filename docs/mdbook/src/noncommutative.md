# Noncommutative algebra

Symbols created with `commutative=False` keep their order in products
(`A*B != B*A`). On top of that, Alkahest reads an expression over such symbols
as a **noncommutative polynomial** — a sum of terms `c·w` with a commutative
coefficient `c` and an ordered *word* `w` of noncommutative factors — and
provides word-rewriting systems that run to a normal form.

```python
import alkahest as ak
p = ak.ExprPool()
I = p.imaginary_unit()
X, P = (p.symbol(n, commutative=False) for n in ("X", "P"))
hbar = p.symbol("hbar")
```

## Expansion

```python
sx, sy = (p.symbol(n, commutative=False) for n in ("sx", "sy"))
ak.nc_expand((sx + sy)**2)        # sx^2 + sx*sy + sy*sx + sy^2   (order kept)
ak.nc_terms(2*X*P + P*X)          # [(2, [X, P]), (1, [P, X])]
```

Products distribute, non-negative integer powers of noncommutative factors
unroll, equal words collect and their coefficients are simplified.

## User rules inside products

`make_rule` / `simplify_with` match a product pattern **inside** a longer
product:

* noncommutative product — a *contiguous run* of factors, in order
  (`X*P` inside `X*P*P`, never inside `P*X`);
* commutative product — a *sub-multiset* of factors (`A*B → C` turns
  `A*B*D` into `C*D`).

For noncommutative input `simplify_with` re-expands after each pass and
re-applies the rules until they stop firing:

```python
rule = ak.make_rule(X*P, P*X + I*hbar)          # [X, P] = iħ
ak.simplify_with(X*P*P - P*P*X, [rule]).value   # 2*I*hbar*P
```

Noncommutative symbols are never pattern wildcards, whatever their name
(`a*ad → ad*a + 1` means those two operators).

`nc_rewrite(expr, rules, max_steps=200000)` is the strict form: literal word
rules, a guaranteed normal form, and `E-NC-001` when the rules do not
terminate within `max_steps` (e.g. `A*B → B*A` together with `B*A → A*B`).

## Commutators

```python
ak.commutator(X, P**2, [rule])      # 2*I*hbar*P
ak.anticommutator(sx, sy)           # sx*sy + sy*sx
```

## Pauli algebra

`simplify_pauli` reduces every word over the noncommutative symbols `sx`, `sy`,
`sz` to at most one σ with `σᵢ² = 1` and `σᵢσⱼ = δᵢⱼ + iεᵢⱼₖσₖ` (`i` is the
pool's imaginary unit `I`):

```python
sz = p.symbol("sz", commutative=False)
ak.simplify_pauli(sz**2).value          # 1
ak.simplify_pauli(sx*sy*sx).value       # -sy
ak.simplify_pauli((sx + sy)**2).value   # 2
```

## Ladder operators

```python
a, ad = p.symbol("a", commutative=False), p.symbol("ad", commutative=False)
ak.normal_order(a * ad**3, bosons=(a, ad))             # ad^3*a + 3*ad^2
ak.vacuum_expectation(a*a*ad*ad, bosons=(a, ad))       # 2
c, cd = p.symbol("c", commutative=False), p.symbol("cd", commutative=False)
ak.normal_order(c*cd, fermions=(c, cd))                # 1 - cd*c
ak.normal_order(c*c, fermions=(c, cd))                 # 0
```

`[a, a†] = 1` for bosons, `{c, c†} = 1` and `c² = 0` for fermions; modes are
lists of `(annihilator, creator)` pairs. Different modes commute (two
fermionic ones anticommute) and are ordered by their position in the lists.
The extra terms produced are exactly Wick's contractions. The number operator
of a mode is `ad*a`.

## Dirac traces

`gamma_trace(slots)` computes `tr(γ…γ)` in four dimensions with metric
`diag(+1,−1,−1,−1)` and `γ⁵ = iγ⁰γ¹γ²γ³`. A slot is `0`–`3` (`γ^k`), `5`
(`γ⁵`) or a commutative symbol `p` for the slashed vector `p̸`:

```python
p_, q_, k_, l_ = (p.symbol(n) for n in "pqkl")
ak.gamma_trace([p_, q_])                 # 4*dot(p, q)
ak.gamma_trace([p_, q_, k_, l_])         # 4*dot(p,q)*dot(k,l) - 4*dot(k,p)*dot(l,q) + 4*dot(l,p)*dot(k,q)
ak.gamma_trace([5, p_, q_, k_, l_])      # -4*I*levi_civita(p, q, k, l)
ak.gamma_trace([0, 1, 0, 1], pool=p)     # 4
```

`dot(p, q)` is `p·q` (`dot(k, p) = p^k` for an explicit index `k`) and
`levi_civita(a, b, c, d) = ε^{μνρσ} a_μ b_ν c_ρ d_σ` with `ε^{0123} = +1`. A
symbol may also stand for a free Lorentz index (`dot(μ, ν) = g^{μν}`). Up to 14
slots are expanded (`E-NC-006` beyond); with `γ⁵` and more than four other
slots, `γ⁵` is expanded as `iγ⁰γ¹γ²γ³`, so the answer uses components.

### d dimensions

`gamma_trace(slots, dim=d, indices=[...])` works with a symbolic dimension.
Slots are commutative labels; a label in `indices` is a Lorentz index, and an
index occurring twice is summed with `g^μ_μ = d` (`tr 1 = 4`, as in FORM and
FeynCalc; `tr(γ^μγ^ν) = 4 g^{μν}`). γ⁵ and explicit components `0`–`3` are
four-dimensional objects and are refused (`E-NC-005`).

```python
d = p.symbol("d"); mu, nu = p.symbol("mu"), p.symbol("nu")
ak.gamma_trace([mu, mu], dim=d, indices=[mu])                # 4*d
ak.gamma_trace([mu, nu, mu, nu], dim=d, indices=[mu, nu])    # 4*d*(2 - d)
ak.gamma_contract([mu, nu, mu], [mu], d)                     # [(2 - d, [nu])]
m, n, r, s = (p.symbol(i) for i in "mnrs")
t = ak.gamma_trace([m, n, r, s], dim=d)                      # free indices
ak.lorentz_contract(t*t, [m, n, r, s], d)                    # 16*d*(3*d - 2)
```

`lorentz_contract` turns any polynomial in `dot(a, b)` (a product of traces)
into its contraction: `dot(μ, μ) → d`, `dot(μ, a) dot(μ, b) → dot(a, b)`.
`gamma_contract` reduces a gamma *string* with contracted pairs using only the
Clifford relation, so the identities hold for every `d`.

## Lie algebras

`lie_algebra(generators, {(a, b, c): f_abc})` (1-based) defines
`[T_a, T_b] = i Σ f_abc T_c`, checks the Jacobi identity and compiles
Poincaré–Birkhoff–Witt ordering rules. `lie_su2` (`f = ε`, `J = σ/2`) and
`lie_su3` (Gell-Mann `f_abc`, `T = λ/2`; the tables `su3_f` and `su3_d` are
exported) are built in.

```python
su2 = ak.lie_su2(p); J = su2.generators
su2.commutator(J[0], J[1])        # I*J3
su2.casimir()                      # J1**2 + J2**2 + J3**2, commutes with every J
su3 = ak.lie_su3(p); su3.normal_order(su3.generators[7]*su3.generators[0])
```

`commutator` reduces the commutator of any noncommutative polynomial in the
generators; `casimir(inverse_metric=None)` is `Σ g^{ab} T_a T_b` (default
`Σ T_a²`, right for the orthonormal su(n) bases).

## Sequence wildcards

`make_rule(A*X*B, rhs, sequence=[X])` declares the noncommutative symbol `X` a
sequence wildcard: it matches any contiguous run of factors, possibly empty,
between `A` and `B`, and `X` in `rhs` is replaced by that run. Matching is
leftmost, and each wildcard takes the *shortest* run that lets the rest match;
a repeated wildcard must match equal runs. Such a rule fires only inside a
product of two or more noncommutative factors, applies through `simplify_with`
(`nc_rewrite` refuses it, `E-NC-003`), and gives up (does not fire) rather than
run unboundedly: the match is capped and honours the active budget.

## Refusals

| Code | Meaning |
|------|---------|
| `E-NC-001` | rewriting did not reach a normal form within `max_steps` |
| `E-NC-002` | an expansion past the term (200 000) or power (512) cap |
| `E-NC-003` | a rule whose left side is not a single word with coefficient 1 |
| `E-NC-004` | a malformed ladder-mode list (not a noncommutative symbol, or repeated) |
| `E-NC-005` | a γ slot that is not `0`–`3`, `5` or a commutative label |
| `E-NC-006` | a γ trace with more than 14 slots to expand |
| `E-NC-007` | a vacuum expectation over a factor that is not a listed ladder operator |
| `E-NC-008` | a malformed Lie-algebra table (generators, indices, inconsistent entries, failed Jacobi identity) |
| `E-NC-009` | a malformed sequence-wildcard pattern |

Not yet covered: γ⁵ in `d` dimensions (no scheme is implemented),
Lie-algebra Casimirs beyond the quadratic one, and sequence wildcards inside
commutative products.
