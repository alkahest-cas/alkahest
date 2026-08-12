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
   ``E-BUDGET-003`` cancelled) from engines that check cooperatively —
   :func:`integrate` and :func:`limit`. :func:`simplify` has no error channel
   and stops early instead of raising. Gröbner bases and homotopy continuation
   do not check the budget at all.

   ``wall_ms`` is **cooperative**: the call stops at the first checkpoint
   after the deadline. The granularity is one primitive polynomial operation,
   and on a high-degree input that operation can be an uninterruptible FLINT
   call. Budgets are **thread-local**; the cancellation flag is process-wide.

.. function:: request_cancel()

   Set the process-wide cancellation flag so cooperative checkpoints return
   ``E-BUDGET-003``. Because :func:`integrate` and :func:`limit` release the
   GIL around their core call, this reaches one of them that is **already
   running** — a watchdog thread can stop a call in flight, not only one that
   has not started. No other engine releases the GIL, so none of the others
   can be cancelled mid-call.

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

   Runs ``fn`` on a worker thread with ``budget`` entered on that thread, and
   raises :exc:`BudgetExceededError` (``E-BUDGET-001``) when it overruns
   ``wall_ms``. Its purpose is to turn a callable that cannot raise through its
   own return type (e.g. :func:`simplify`, which truncates silently) into a
   coded error.

   .. warning::

      **It does not bound wall time for an uncooperative callee.** It joins its
      worker before the exception propagates, so it returns control when the
      callee returns, not at ``wall_ms``::

          run_with_wall_fallback(time.sleep, 3.0, budget=Budget(wall_ms=50))
          # raises E-BUDGET-001 after 3000 ms

      The message reports the real elapsed time for exactly this reason. Python
      cannot kill a thread, and abandoning one would leak a live thread that
      still allocates into the pool and can only be stopped through the
      process-wide cancel flag. Prefer ``context(budget=...)`` for engines that
      honor the cooperative checkpoints, and an **OS-level timeout**
      (subprocess or process watchdog) for anything else.

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

   The active budget is propagated into ``parallel=True`` workers: a Rust
   budget frame is thread-local, so ``batch_map`` snapshots the caller's budget
   and re-enters it inside each worker task. ``wall_ms`` remains a single
   sweep-wide deadline (captured at the ``batch_map`` call, not at
   ``context(budget=...)`` entry); ``max_steps`` becomes **per item**, because
   the Rust step counter is not readable from Python. Without this, a fanned-out
   sweep ran unbudgeted and reported ``E-INT-001`` — a mathematical verdict —
   where a sequential sweep reported ``E-BUDGET-001``.

   One item tripping its budget never cancels its siblings; ``batch_map`` never
   sets the process-wide cancel flag. :func:`request_cancel` does abort every
   in-flight worker, by design.

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
