# Kernel design

The expression kernel is the foundation everything else builds on. It lives in `alkahest-core/src/kernel/`.

## Hash-consed DAG

Every expression is represented as a directed acyclic graph stored in an `ExprPool`. Nodes are interned: before inserting a new node, the pool checks whether a structurally identical node already exists. If it does, the existing `ExprId` is returned instead of allocating a new node.

This gives three properties:

1. **Structural equality is a pointer comparison.** `id_a == id_b` iff the expressions are structurally identical. No tree traversal required.
2. **Automatic subexpression sharing.** If `sin(x²)` appears in ten different expressions, there is only one `sin(x²)` node in memory.
3. **Hash-based memoization is cheap.** Caching the result of a transformation keyed by `ExprId` is O(1) and correct. Hot recursive paths (simplify, differentiation, integration guards, JIT interpreter) use per-call `HashMap<ExprId, T>` memo tables so shared DAG nodes are processed once, not once per tree occurrence.

### ExprPool

`ExprPool` is the intern table. It owns all expressions in a session.

```python
pool = ExprPool()
x = pool.symbol("x")       # intern a Symbol node
n = pool.integer(42)       # intern an Integer node
```

Multiple pools are independent. An `ExprId` from one pool must not be mixed into another — the pool validates this in debug builds.

**Persistent pool (V1-14).** A pool can be serialized to disk and reopened, preserving all `ExprId`s across sessions:

```python
pool.save_to("session.alkp")
pool2 = ExprPool.load_from("session.alkp")
```

**Sharded pool.** With `--features parallel`, the intern table uses a sharded concurrent hashmap (`DashMap`), allowing multiple threads to insert expressions without contention.

## ExprData variants

Each interned node is one of:

| Variant | Description |
|---|---|
| `Symbol(name, domain)` | Named variable with a domain annotation |
| `Integer(n)` | Exact arbitrary-precision integer |
| `Rational(p, q)` | Exact rational number |
| `Add(children)` | N-ary addition |
| `Mul(children)` | N-ary multiplication |
| `Pow(base, exp)` | Exponentiation |
| `Call(primitive, args)` | Application of a registered primitive |
| `Piecewise(cases)` | Conditional expression |
| `Predicate(kind, args)` | Boolean condition (inequality, equality) |

`Add` and `Mul` are n-ary: `a + b + c` is one `Add` node with three children, not two nested `Add` nodes. Children are sorted at construction time so that commutativity is structural — `a + b` and `b + a` produce the same interned node.

## Domains

Every symbol carries a domain as part of its structural identity:

```python
x_real = pool.symbol("x", "real")
x_complex = pool.symbol("x", "complex")
# x_real and x_complex are distinct expressions — different ExprIds
```

The domain is not a global assumption; it is part of what the symbol *is*. Simplification rules can query a symbol's domain to decide whether a rewrite is valid (e.g. `sqrt(x²) → x` requires `x` to be non-negative).

Available domains: `real`, `positive`, `nonnegative`, `integer`, `complex`. The default when no domain is specified is `real`.

## ExprId and memory

`ExprId` is a 32-bit index into the pool's internal arena. It is `Copy`, `Send`, and `Sync`. Cloning an `ExprId` is free. No reference counting is needed because the pool owns all nodes; expressions are not freed until the pool is dropped.

The kernel is designed with parallelism as a first-class property. All kernel types are `Send + Sync`. The simplification and differentiation passes can run concurrently on disjoint `ExprId`s from the same pool.

## Interning cost model

Interning a new node requires:
1. Hash the `ExprData`.
2. Look up in the concurrent hash map.
3. On miss: allocate the node in the arena and insert into the map.
4. On hit: return the existing `ExprId`.

Step 4 (the common case in a running computation) is a single hash lookup plus a pointer load. The arena uses bump allocation, so step 3 is also fast.

The memory benchmark group in `alkahest-core/benches/alkahest_bench.rs` verifies that rebuilding an identical expression tree does not grow the pool.
