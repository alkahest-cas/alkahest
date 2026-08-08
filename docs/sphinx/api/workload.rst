Search / workload API
=====================

.. currentmodule:: alkahest

APIs for running Alkahest inside agent math-search loops: per-call budgets,
batch fan-out that never aborts on one bad candidate, and related helpers.

Conceptual guide: `Autoresearch / agent loops <../search-plumbing.html>`_,
`Budgets <../budgets.html>`_, `Batch <../batch.html>`_.

Budgets
-------

.. class:: Budget(wall_ms=None, max_steps=None, seed=None)

   Immutable per-call resource budget.

   :param wall_ms: Optional wall-clock limit in milliseconds.
   :param max_steps: Optional cooperative step limit.
   :param seed: Optional determinism seed exposed via :func:`budget_seed`.

   Entered with ``context(budget=...)``. Trips raise
   :exc:`BudgetExceededError` (``E-BUDGET-001`` wall, ``E-BUDGET-002`` steps,
   ``E-BUDGET-003`` cancelled) from engines that check cooperatively
   (notably :func:`integrate`).

.. function:: request_cancel()

   Set the process-wide cancellation flag so cooperative checkpoints return
   ``E-BUDGET-003``.

.. function:: clear_cancel()

   Clear the cancellation flag before the next candidate.

.. function:: is_cancelled() -> bool

.. function:: is_budget_active() -> bool

.. function:: budget_seed() -> int | None

   Seed of the innermost active budget, or ``None``.

.. function:: active_budget() -> Budget | None

   The :class:`Budget` from the innermost Python ``context(budget=...)``, or
   ``None``.

.. function:: run_with_wall_fallback(fn, *args, budget=None, **kwargs)

   Python-layer wall-clock fallback for callables that cannot raise
   :exc:`BudgetExceededError` through their own return type (e.g.
   :func:`simplify`). Prefer ``context(budget=...)`` for engines that already
   honor Rust cooperative checkpoints.

Batch evaluation
----------------

.. class:: BatchItem

   One outcome from :func:`batch_map` / ``*_many``.

   .. attribute:: index
   .. attribute:: ok
   .. attribute:: value
   .. attribute:: error
   .. attribute:: elapsed_ms

   On failure, ``error`` is a dict with ``code``, ``message``, ``remediation``,
   and ``type``. ``code`` is the exception's ``E-*`` code when present,
   otherwise ``E-BATCH-001``.

.. function:: batch_map(fn, items, *, parallel=False, max_workers=None, **kwargs) -> list[BatchItem]

   Call ``fn(item, **kwargs)`` for every item. **Never raises** for a single
   bad element. Always returns results in **input order**.

.. function:: batch_map_iter(fn, items, *, parallel=False, max_workers=None, **kwargs)

   Streaming counterpart. Under ``parallel=True``, yields in **completion
   order** (each item still carries its original ``index``).

.. function:: integrate_many(exprs, var, *bounds, parallel=False, max_workers=None, **kwargs) -> list[BatchItem]

.. function:: simplify_many(exprs, *, parallel=False, max_workers=None, **kwargs) -> list[BatchItem]

.. function:: diff_many(exprs, var, *, parallel=False, max_workers=None, **kwargs) -> list[BatchItem]

   Thin :func:`batch_map` wrappers over the common derivation entry points.

Schema constants
----------------

.. data:: RESULT_SCHEMA_VERSION
.. data:: STEPS_SCHEMA_VERSION
.. data:: STEP_FIELDS
.. data:: STEP_FIELDS_COMPACT

   Version and field-name tables for :meth:`DerivedResult.to_dict`. See
   `derivation logs <../derivations.html#machine-parseable-output-to_dict--to_json>`_.
