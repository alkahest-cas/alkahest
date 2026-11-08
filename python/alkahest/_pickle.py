"""Pickling of :class:`~alkahest.Expr` and :class:`~alkahest.ExprPool`.

An ``Expr`` is an index into one ``ExprPool``, so pickling one means deciding
which pool it lands in on the other side.  The rules:

* **An expression pickles as its own DAG** — every node reachable from it,
  and nothing else from its pool — plus a reference to its pool.  The size of
  ``pickle.dumps(e)`` is the size of ``e``, not of the pool it came from.

* **A pool pickles as a token**, a random name it is given the first time it
  is pickled.  It does *not* carry its node table: that is a cache of
  everything ever interned, and the expressions that matter carry their own
  nodes.  ``pickle.loads(pickle.dumps(pool))`` is therefore a pool with no
  expressions of its own — use :meth:`ExprPool.save_to` /
  :meth:`ExprPool.load_from` to persist a whole pool.

* **Unpickling a token yields the live pool with that token in this process,
  if there is one**, and otherwise a new, empty pool that takes the token.
  So:

  - expressions pickled *together* (in one ``pickle.dumps`` call — a list, a
    tuple, a dict, an object holding several) share one pool after loading,
    because ``pickle`` memoises the pool object and rebuilds it once;
  - expressions pickled *separately* but from the same pool also share one
    pool when loaded into the same process, because the token names the same
    pool both times;
  - a round trip through a worker process comes home to the *original* pool:
    ``multiprocessing`` sends ``e`` to a worker (a new pool there, with the
    token), the worker's results come back naming that token, and the parent
    still holds the pool that has it.  Results can be combined with the
    expressions that were never sent.

  Within one process, ``pickle.loads(pickle.dumps(e)) == e``.

Symbols keep their name, domain and commutativity; an expression that
reaches a node its new pool already holds reuses it (hash-consing), exactly
as if it had been built there.

A :class:`~alkahest.DerivedResult` pickles as its value, its whole
derivation log and the inputs its certificates are built from, as one DAG in
the same pool scheme, so ``.value``, ``.steps``, ``.derivation`` and
``verification`` are the same after a round trip.  A :class:`~alkahest.Matrix`
pickles as its entries and shape.  Both therefore go through
``multiprocessing.Pool.map``.

An :class:`~alkahest.ArbBall` pickles its midpoint and radius as exact
binary floats (``mantissa · 2**exp`` at each float's own precision), never
as decimal text, so the ball that comes back is bit-for-bit the one sent at
any precision.  An :class:`~alkahest.EvaluationResult` pickles every field
(its enclosure as such a ball), a :class:`~alkahest.RootInterval` its two
exact rational endpoints, a :class:`~alkahest.Series` its expansion as an
expression, a :class:`~alkahest.Domain` its member name, a
:class:`~alkahest.GroebnerBasis` its exact generators, order, proof status and
variables, and a :class:`~alkahest.NumberField` its defining polynomial.

The process-wide :func:`~alkahest.default_pool` (where ``alkahest.pi`` and
``alkahest.sin(2)`` live) is not given a token: it unpickles as the receiving
process's own default pool.

Loading untrusted pickles is unsafe in general (that is ``pickle``, not
alkahest); the expression bytes themselves are validated like a pool file
and a malformed payload raises :class:`~alkahest.IoError`.
"""

from __future__ import annotations

import os
import threading
import weakref

__all__: list[str] = []

_lock = threading.Lock()
# pool -> token, and token -> pool.  Both weak: pickling a pool must not keep
# it alive, and an unpickled pool lives exactly as long as its expressions.
_token_of: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_pool_of: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _token(pool) -> str:
    with _lock:
        tok = _token_of.get(pool)
        if tok is None:
            tok = os.urandom(16).hex()
            _token_of[pool] = tok
            _pool_of[tok] = pool
        return tok


def _unpickle_pool(token: str):
    from .alkahest import ExprPool

    with _lock:
        pool = _pool_of.get(token)
        if pool is None:
            pool = ExprPool()
            _token_of[pool] = token
            _pool_of[token] = pool
        return pool


def _unpickle_expr(pool, data: bytes):
    return pool._expr_from_dag(data)


def _unpickle_default_pool():
    from .alkahest import default_pool

    return default_pool()


def _reduce_pool(pool):
    from .alkahest import default_pool

    if pool is default_pool():
        # The default pool is a per-process singleton, and it unpickles as the
        # receiving process's own: ``alkahest.pi`` sent to a worker is that
        # worker's ``alkahest.pi``, and comes home as the parent's.
        return (_unpickle_default_pool, ())
    return (_unpickle_pool, (_token(pool),))


def _reduce_expr(pool, data: bytes):
    return (_unpickle_expr, (pool, data))


def _unpickle_derived(pool, data: bytes, meta):
    from .alkahest import DerivedResult

    return DerivedResult._from_pickle(pool, data, meta)


def _reduce_derived(pool, data: bytes, meta):
    return (_unpickle_derived, (pool, data, meta))


def _unpickle_matrix(pool, data: bytes, rows: int, cols: int):
    from .alkahest import Matrix

    return Matrix._from_pickle(pool, data, rows, cols)


def _reduce_matrix(pool, data: bytes, rows: int, cols: int):
    return (_unpickle_matrix, (pool, data, rows, cols))


def _unpickle_arb_ball(prec: int, mid, rad):
    from .alkahest import ArbBall

    return ArbBall._from_pickle(prec, mid, rad)


def _unpickle_evaluation_result(*fields):
    from .alkahest import EvaluationResult

    return EvaluationResult._from_pickle(*fields)


def _unpickle_root_interval(lo, hi):
    from .alkahest import RootInterval

    return RootInterval._from_pickle(lo, hi)


def _unpickle_series(expr):
    from .alkahest import Series

    return Series._from_pickle(expr)


def _unpickle_domain(name: str):
    from .alkahest import Domain

    return getattr(Domain, name)


def _unpickle_groebner(gens, order: str, proof_status: str, pool, variables):
    from .alkahest import GroebnerBasis

    return GroebnerBasis._from_pickle(gens, order, proof_status, pool, variables)


def _unpickle_number_field(coeffs, variable: str, cyclotomic_order):
    from .alkahest import NumberField

    if cyclotomic_order is not None:
        field = NumberField.cyclotomic(cyclotomic_order)
    else:
        field = NumberField(coeffs)
    return field.with_variable(variable)


def _unpickle_number_field_element(field, coeffs):
    return field.element(coeffs)
