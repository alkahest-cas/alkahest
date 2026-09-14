# Stability policy

Alkahest follows semantic versioning starting at `1.0`.

## Stable surface

The stable surface is the API Alkahest commits to maintaining without breaking changes across a major version:

- **Rust:** everything re-exported from `alkahest_cas::stable`
- **Python:** every name in `alkahest.__all__` at release time

Breaking changes to the stable surface require a major-version bump (e.g. 1.x → 2.0).

## Experimental surface

- **Rust:** `alkahest_cas::experimental::*`, plus anything not in `stable`
- **Python:** `alkahest.experimental.*`, plus anything re-exported from the native module but not in `__all__`

Experimental APIs may change in any minor release. Pin a specific point release if you depend on them.

## Deprecation policy

Removed stable symbols are kept as `#[deprecated]` shims for one full major cycle before deletion:

1. Symbol is deprecated in 1.x with `#[deprecated(since = "1.x", note = "use Y instead")]`
2. Symbol is removed in 2.0

Python deprecations emit `DeprecationWarning` from the point of deprecation.

## Enforcement

- `cargo semver-checks` — runs on every PR via `.github/workflows/alkahest-semver-check.yml`. Fails the PR if any stable Rust API breaks.
- `scripts/check_api_freeze.py` — guards against removals from `alkahest.__all__` within a major cycle.
- `CHANGELOG.md` — Keep-a-Changelog format; every release documents additions, deprecations, and (in major bumps) removals.

## Error codes

Diagnostic error codes (e.g. `E-POLY-001`) are also stable. A code that is **reachable
from a released binding** will not be renumbered or removed within a major cycle; new
codes are added by incrementing within the existing prefix.

The qualifier is load-bearing, and so far it has been used once. `series_solve` raised
an uncoded bare `ValueError` until 3.10, so its internal `E-ODE-020…025` had never
surfaced to any caller — while colliding head-on with the numeric integrators' block, where
`E-ODE-021` simultaneously meant "the adaptive step size fell below the floor" and "the
point is irregular singular". That is precisely the branch these codes exist to support, so
the series block moved to `E-ODE-040…045` when it was wired up. A code nobody could observe
is not yet part of the stable surface; one code meaning two things is worse than a
renumbering. `scripts/check_error_codes.py` now fails any PR that reintroduces a collision.

## Diagnostic codes and their stability

A code joins the stable surface at the version where it first becomes **reachable from a
released binding** — which is usually the version it is written in, but not always; see the
qualifier above. From that point it is not renumbered or removed within the major cycle.
See [Error handling](./errors.md) for the current code table.
