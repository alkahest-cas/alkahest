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

Diagnostic error codes (e.g. `E-POLY-001`) are also stable. A code introduced in 1.x will not be renumbered or removed until 2.0. New codes are added by incrementing within the existing prefix.

## Diagnostic codes and their stability

Error codes are part of the stable surface from the version they first appear. See [Error handling](./errors.md) for the current code table.
