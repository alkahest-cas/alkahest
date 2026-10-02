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

Multiple pools are independent. An `ExprId` from one pool must not be mixed into another — the pool validates this in debug builds. From Python, an `Expr` carries its pool: equality and hashing are by (pool, id), so expressions from two pools never compare equal, and the operators, the pool constructors and every function taking more than one expression raise `PoolError` (`E-POOL-001`) when handed an expression from another pool.

### Building large sums and products

`pool.add(terms)` and `pool.mul(factors)` are the bulk constructors: one call interns one flat node, linear in the number of operands.

Accumulating with an operator is not. `Add` is flat, so `s = s + t` interns a new node one child wider than the last at every step — quadratic time, and quadratic pool memory that the pool never returns (see [ExprId and memory](#exprid-and-memory)). Builtin `sum()` is the same loop:

```python
terms = [x**i for i in range(10_000)]

s = pool.add(terms)      # ~30 ms, one node
s = sum(terms[1:], terms[0])   # ~2 s and ~400 MB of dead intermediate sums
```

The two give the same expression. `alkahest.parse` gathers each run of `+`/`-` (and of `*`) and uses the bulk constructors, so parsing a long sum is linear too.

**Persistent pool (V1-14).** A pool can be serialized to disk and reopened, preserving all `ExprId`s across sessions:

```python
pool.save_to("session.alkp")
pool2 = ExprPool.load_from("session.alkp")
```

**Pickling.** `Expr` and `ExprPool` support `pickle` (and so `copy.deepcopy`, `multiprocessing`, `concurrent.futures`, `joblib`):

- An expression pickles as **its own DAG** — the nodes reachable from it, not the pool's whole intern table — plus a reference to its pool.
- A pool pickles as a **token** naming it, not as its node table. Unpickling the token gives the live pool with that token in the current process if there is one, otherwise a new empty pool that takes the token.
- So expressions pickled together (one `pickle.dumps` of a list, tuple, dict, …) share one pool on the other side; expressions from the same pool pickled separately also share one pool when loaded into the same process; and results a `multiprocessing` worker sends back land in the **original** pool, where they combine with expressions that never left. In one process, `pickle.loads(pickle.dumps(e)) == e`.
- `pickle.loads(pickle.dumps(pool))` in a new process is an empty pool: to persist a whole pool, use `save_to` / `load_from`.

```python
import concurrent.futures, operator
with concurrent.futures.ProcessPoolExecutor() as ex:
    products = list(ex.map(operator.mul, [x, y], [y, y]))
products[0] + x          # fine: the result is back in x's pool
```

**Sharded pool.** Each node is stored once, in an append-only array; the intern index holds only ids (and hash bits) that compare through that array. With `--features parallel` the index is split into lock-striped shards, so threads interning unrelated expressions rarely contend, and two threads interning the same expression still receive the same id.

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

> **That last clause is a hard limit, not an implementation detail.** The arena is
> **append-only**: there is no `clear`, no `truncate`, no refcount and no GC, so nothing
> is ever reclaimed while the pool is alive, and the storage cannot shrink. A distinct
> expression costs roughly 115 bytes of resident memory per node, permanently. A loop
> that builds a module-scope pool once and then calls into it forever grows linearly and
> without bound, at flat per-call latency — so it OOMs with no slowdown to warn you
> first. Every `Expr`, `Matrix`, `Series` and `DerivedResult` holds a strong reference to
> its pool, so retaining one result retains the whole history.
>
> The supported pattern for unattended work is **one pool per problem**, dropped when the
> problem is done: see
> [Budgets → `ExprPool` never reclaims](./budgets.md#exprpool-never-reclaims).

The kernel is designed with parallelism as a first-class property. All kernel types are `Send + Sync`. The simplification and differentiation passes can run concurrently on disjoint `ExprId`s from the same pool.

## Interning cost model

Interning a new node requires:
1. Hash the `ExprData`.
2. Look up in the concurrent hash map.
3. On miss: allocate the node in the arena and insert into the map.
4. On hit: return the existing `ExprId`.

Step 4 (the common case in a running computation) is a single hash lookup plus a pointer load. The arena uses bump allocation, so step 3 is also fast.

The memory benchmark group in `alkahest-core/benches/alkahest_bench.rs` verifies that rebuilding an *identical* expression tree does not grow the pool. Note the scope of that guarantee: it covers repeated work, not new work. A stream of *distinct* inputs grows the pool by every node it interns, and none of it comes back — see the warning under [ExprId and memory](#exprid-and-memory).

One documented exception to "identical input does not grow the pool": `Matrix.eigenvals()` interns a fresh gensym per call, so it grows by about 1.9 KB per call even on the same matrix. Cache its result rather than recomputing.
