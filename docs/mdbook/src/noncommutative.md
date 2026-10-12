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

Not yet covered: γ-matrix algebra beyond traces (contraction identities such as
`γ^μ γ_μ = 4`, `d`-dimensional traces), general Lie-algebra commutator tables,
and pattern wildcards that bind sub-words.
