# Parsing expressions from strings

`alkahest.parse` converts a human-readable math string into an `Expr` node
using a Pratt (top-down operator precedence) recursive-descent parser.

```python
import alkahest
from alkahest import ExprPool, parse, diff, simplify

pool = ExprPool()
x = pool.symbol("x")

e = parse("x^2 + 2*x + 1", pool, {"x": x})
print(e)                    # x^2 + 2*x + 1

dr = diff(e, x)
print(dr.value)             # 2*x + 2
```

## Syntax

| Form | Meaning |
|---|---|
| `42`, `3.14`, `1.5e-3` | Integer or float literal |
| `x`, `alpha`, `x_1` | Symbol (created in `pool` on first use) |
| `a + b`, `a - b` | Addition / subtraction |
| `a * b`, `a / b` | Multiplication / division |
| `a ^ b`, `a ** b` | Exponentiation (right-associative) |
| `-a`, `+a` | Unary negation / identity |
| `(expr)` | Grouping |
| `sin(x)`, `atan2(y, x)` | Function call (one or two arguments) |

Whitespace (spaces, tabs, newlines) is ignored everywhere.

## Operator precedence

From lowest to highest:

| Level | Operators |
|---|---|
| 10 | `+` `-` (infix) |
| 20 | `*` `/` |
| 25 | Unary `-` `+` |
| 30 | `^` `**` (right-associative) |

So `-x^2` parses as `-(x^2)`, not `(-x)^2`, and `x^2^3` parses as
`x^(2^3) = x^8`.

## Supported functions

`abs`, `acos`, `asin`, `atan`, `atan2`, `ceil`, `cos`, `cosh`, `erf`,
`erfc`, `exp`, `floor`, `gamma`, `log`, `round`, `sign`, `sin`, `sinh`,
`sqrt`, `tan`, `tanh`

## The `symbols` map

By default, every new identifier is interned as a fresh `pool.symbol(name)`.
Pass a pre-built `symbols` dict to bind identifiers to existing `Expr`
objects, or to collect the symbols that were created:

```python
# Pre-bind x to an existing symbol
x = pool.symbol("x")
e = parse("sin(x)^2 + cos(x)^2", pool, {"x": x})

# Collect auto-created symbols after parsing
sym_map: dict = {}
e = parse("a*x^2 + b*x + c", pool, sym_map)
print(sym_map.keys())   # dict_keys(['a', 'x', 'b', 'c'])
```

Identifiers not in the map are created and then *added* to the map, so the
same string name always resolves to the same `Expr` within a single `parse`
call.

## Error handling

`parse` raises `ParseError` (code `E-PARSE-001`) on any lexical or syntax
error. The exception's `.span` attribute gives the `(start, end)` byte range
of the offending token, and `.remediation` provides a hint:

```python
from alkahest import ParseError

try:
    parse("sin(x) @ 2", pool, {"x": x})
except ParseError as e:
    print(e)           # unexpected character '@' at offset 7
    print(e.span)      # (7, 8)

try:
    parse("zeta(x)", pool, {"x": x})
except ParseError as e:
    print(e.remediation)  # known functions: abs, acos, asin, ...
```

## Round-trip with pretty-printing

`parse` is the inverse of `str()` for expressions built from the operators
and functions listed above:

```python
from alkahest import latex, unicode_str

e = parse("sin(x)^2 + cos(x)^2", pool, {"x": x})
print(latex(e))        # \sin\!\left(x\right)^{2} + \cos\!\left(x\right)^{2}
print(unicode_str(e))  # sin(x)² + cos(x)²
```
