"""Versioned schema documentation for :class:`alkahest.DerivedResult` output.

P1 search plumbing item 6 — agents pay for every character an operation
returns, so :meth:`DerivedResult.to_dict` / :meth:`DerivedResult.to_json`
(implemented in the native extension, ``alkahest-py/src/lib.rs``) expose a
stable, versioned envelope with a token-efficient ``mode="compact"``.

This module carries no logic of its own — the encoders live in Rust so
``to_dict``/``to_json`` never drift from ``.steps``/``.verification``. It
re-exports the two version constants from the compiled extension and
documents the field-name contract in one place so agents (and this
project's own tests/docs) have a single canonical import.

Schema versions
----------------
``RESULT_SCHEMA_VERSION`` covers the *envelope*: the set of top-level keys
(``kind``, ``schema_version``, ``steps_schema_version``, ``value``,
``verification``, ``certificate_status``, ``steps``, ``has_certificate``).
Bump it when a top-level key is added, removed, or renamed.

``STEPS_SCHEMA_VERSION`` covers one entry of ``steps``: the full-mode field
names (``rule``, ``before``, ``after``, ``side_conditions``) and the
compact-mode short-key mapping (``r``, ``s``). Bump it independently of
``RESULT_SCHEMA_VERSION`` when that shape changes.

Both start at ``1`` and are also available as ``DerivedResult.SCHEMA_VERSION``
/ ``DerivedResult.STEPS_SCHEMA_VERSION`` class attributes.

Full-mode step fields
----------------------
Each entry of ``.steps`` (and of ``to_dict()["steps"]`` in full mode) is a
dict with exactly these keys, matching the ``.steps`` getter that predates
this schema:

* ``rule`` — rewrite rule name (``str``)
* ``before`` — expression display string before the rewrite (``str``)
* ``after`` — expression display string after the rewrite (``str``)
* ``side_conditions`` — side conditions recorded for the rewrite
  (``list[str]``, possibly empty)

Compact-mode step fields
-------------------------
``to_dict(mode="compact")["steps"]`` entries use short keys and drop the
``before``/``after`` strings — usually the largest strings in a derivation,
and the single biggest token cost of a multi-step result:

* ``r`` — same as full-mode ``rule``
* ``s`` — same as full-mode ``side_conditions``, but the key is **omitted
  entirely** when the list is empty (most steps have none)

Honesty in compact mode
------------------------
Compact mode never drops or renames ``verification["status"]`` — the field
that distinguishes ``exactly_verified`` / ``numerically_checked`` /
``certificate_available`` / ``unverified`` — because that is the honesty
signal this schema exists to preserve token budget around, not obscure.
``verification["externally_verified"]`` (always ``False`` today; no
external Lean check has ever run in-process) is kept alongside it so a
compact reader cannot mistake a generated certificate for a checked one.
Compact mode also never includes Lean certificate source text in either
mode's ``certificate_status`` — use the ``.certificate`` getter for that;
``has_certificate`` plus ``certificate_status["reason"]`` is enough to know
whether one exists and why not.

Example
-------
>>> import alkahest as ak
>>> pool = ak.ExprPool()
>>> x = pool.symbol("x")
>>> dr = ak.diff(ak.sin(x), x)
>>> full = dr.to_dict()                    # mode="full" is the default
>>> compact = dr.to_dict(mode="compact")
>>> len(dr.to_json(mode="compact")) <= len(dr.to_json(mode="full"))
True
>>> full["verification"]["status"] == compact["verification"]["status"]
True

Agents in hot loops (batch derivations, autoresearch search plumbing)
should prefer ``to_dict(mode="compact")`` / ``to_json(mode="compact")``
over the full envelope or over reading ``.steps`` directly.
"""

from .alkahest import RESULT_SCHEMA_VERSION, STEPS_SCHEMA_VERSION

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "STEPS_SCHEMA_VERSION",
    "STEP_FIELDS",
    "STEP_FIELDS_COMPACT",
]

#: Field names of a full-mode step record (``.steps`` entries and
#: ``to_dict()["steps"]`` entries under ``mode="full"``).
STEP_FIELDS: tuple[str, ...] = ("rule", "before", "after", "side_conditions")

#: Short keys of a compact-mode step record
#: (``to_dict(mode="compact")["steps"]`` entries). ``s`` is omitted from a
#: given step's dict entirely when that step has no side conditions.
STEP_FIELDS_COMPACT: tuple[str, ...] = ("r", "s")
