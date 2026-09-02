//! Keeps the nightly ThreadSanitizer shard's scope honest.
//!
//! The `tsan` shard used to run the whole workspace under TSan. That is 133
//! minutes of instrumented tests of which ~89 are three single-threaded
//! semidefinite-programming searches in `real::sos` — code that cannot race
//! because it never starts a second thread. The shard now runs
//! `alkahest-cas`'s unit tests filtered through `.github/tsan-scope.txt`
//! (everything else in the workspace still runs in full; see the step comment
//! in `.github/workflows/ci.yml`).
//!
//! A filter list is only safe if it cannot silently stop covering something.
//! That is this file's job, and it is an ordinary integration test so it runs
//! on every PR rather than once a night:
//!
//! * [`tsan_scope_covers_every_module_that_can_start_a_thread`] walks the
//!   module tree from `src/lib.rs`, and fails if a module whose source spawns
//!   a thread or hands work to Rayon is not covered by a scope entry. Add
//!   `rayon::` to a new module and Tier-1 CI goes red until the scope names
//!   it.
//! * [`tsan_scope_entries_all_name_a_live_module`] fails when an entry no
//!   longer corresponds to a module — the rename case, where a filter would
//!   otherwise quietly match nothing and the shard would pass vacuously. (The
//!   workflow independently fails any filter that matches zero *tests*.)
//!
//! # What this check does not prove
//!
//! It is a syntactic scan, not a call graph. A test in an out-of-scope module
//! that reaches Rayon through a helper in a *third* module — one that names
//! neither `rayon::` nor a parallel entry point in its own source — would not
//! be flagged. The parallel surface is small and entirely public
//! (`simplify_par`, `simplify_redex`, `simplify_auto`, `call_batch_par`), so
//! that path is narrow, and [`PARALLEL_ENTRY_POINTS`] catches the direct call.
//! The residual risk is accepted deliberately; the alternative is instrumenting
//! 2 700 tests to look for races in a simplex solver.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Source spellings that put a second thread on the CPU.
///
/// Deliberately crude and deliberately over-eager: a false positive costs one
/// line in `.github/tsan-scope.txt`, a false negative costs a race that
/// nothing looks for.
const SPAWN_TOKENS: &[&str] = &[
    "thread::spawn",
    "thread::scope",
    "thread::Builder",
    "spawn_scoped",
    "rayon::",
    "par_iter",
    "par_chunks",
    "par_bridge",
    "par_extend",
    "par_sort",
];

/// Public functions that run the caller's work on a Rayon pool.
///
/// A test naming one of these executes concurrent code even if its own module
/// is otherwise sequential, so the module has to be in scope.
const PARALLEL_ENTRY_POINTS: &[&str] = &[
    "simplify_par",
    "simplify_redex",
    "simplify_auto",
    "call_batch_par",
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("alkahest-core has a parent directory")
        .to_path_buf()
}

fn scope_path() -> PathBuf {
    repo_root().join(".github").join("tsan-scope.txt")
}

/// `false` when this is a published `.crate` rather than the git checkout.
///
/// `.github/` is not part of the packaged crate, so a downstream `cargo test`
/// on a vendored `alkahest-cas` has nothing to check and should not fail. The
/// probe is the *workflow*, not the scope file: inside the repo the workflow
/// always exists, so this can never quietly disable the guard where it matters
/// — a missing or emptied `tsan-scope.txt` still fails loudly below.
fn in_source_repo() -> bool {
    repo_root()
        .join(".github")
        .join("workflows")
        .join("ci.yml")
        .is_file()
}

/// The filter lines of `.github/tsan-scope.txt`, trailing `::` trimmed.
fn scope_entries() -> Vec<String> {
    let path = scope_path();
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let entries: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.trim_end_matches("::").to_string())
        .collect();
    assert!(
        !entries.is_empty(),
        "{} declares no filters; the tsan shard would run nothing",
        path.display()
    );
    entries
}

fn is_covered(module: &str, entries: &[String]) -> bool {
    entries
        .iter()
        .any(|e| module == e || module.starts_with(&format!("{e}::")))
}

/// Drop `//` line comments and `/* */` block comments that own their line.
///
/// Doc comments are the main source of false positives here — `simplify::rules`
/// and `simplify::depth` both discuss `simplify_par` in prose without calling
/// it. Comments that trail code are left alone; a trailing comment mentioning a
/// spawn token is rare and only ever over-includes.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut in_block = false;
    for line in src.lines() {
        let trimmed = line.trim_start();
        if in_block {
            if trimmed.contains("*/") {
                in_block = false;
            }
            continue;
        }
        if trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("/*") {
            if !trimmed.contains("*/") {
                in_block = true;
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// The `mod foo;` declarations in one file, in source order.
fn child_module_names(src: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in src.lines() {
        let mut rest = line.trim();
        if let Some(r) = rest.strip_prefix("pub") {
            rest = r.trim_start();
            if rest.starts_with('(') {
                match rest.find(')') {
                    Some(i) => rest = rest[i + 1..].trim_start(),
                    None => continue,
                }
            }
        }
        let Some(rest) = rest.strip_prefix("mod ") else {
            continue;
        };
        let name = rest.trim();
        // `mod foo;` only. `mod foo {` is inline and owns no file.
        let Some(name) = name.strip_suffix(';') else {
            continue;
        };
        let name = name.trim();
        if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            names.push(name.to_string());
        }
    }
    names
}

/// Every file reachable from `src/lib.rs` by following `mod foo;`, paired with
/// its module path.
///
/// Walking the declarations rather than the directory is what keeps a stray
/// source file out of the result: `poly/groebner/f4.rs` still sits on disk and
/// still contains `par_iter`, but `poly/groebner/mod.rs` replaced it with an
/// inline re-export shim years ago and never compiles it. Demanding TSan
/// coverage for a file the build ignores would be noise.
fn module_files() -> Vec<(String, PathBuf)> {
    let src = repo_root().join("alkahest-core").join("src");
    let mut found = Vec::new();
    let mut queue = vec![(String::new(), src.join("lib.rs"))];
    let mut seen = BTreeSet::new();

    while let Some((path, file)) = queue.pop() {
        if !seen.insert(file.clone()) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&file) else {
            continue;
        };
        let stripped = strip_comments(&text);

        // Where this file's children live: `lib.rs`/`mod.rs` own their own
        // directory, `foo.rs` owns the sibling directory `foo/`.
        let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let dir = match stem {
            "lib" | "mod" => file.parent().expect("file has a parent").to_path_buf(),
            other => file
                .parent()
                .expect("file has a parent")
                .join(other)
                .to_path_buf(),
        };

        for name in child_module_names(&stripped) {
            let child_path = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}::{name}")
            };
            let flat = dir.join(format!("{name}.rs"));
            let nested = dir.join(&name).join("mod.rs");
            if flat.is_file() {
                queue.push((child_path, flat));
            } else if nested.is_file() {
                queue.push((child_path, nested));
            }
        }
        found.push((path, file));
    }
    found
}

/// Test functions in `src`, as `(name, body)`.
///
/// Brace matching from the `{` that opens the function. Good enough for a
/// formatted tree — `cargo fmt` runs in CI — and only ever fails by returning
/// too much body, which over-includes.
fn test_fn_bodies(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let bytes: Vec<&str> = src.lines().collect();
    for (i, line) in bytes.iter().enumerate() {
        if line.trim() != "#[test]" {
            continue;
        }
        // The `fn` line may be a few attributes down (`#[should_panic]`, ...).
        let Some(fn_line) = (i + 1..(i + 8).min(bytes.len()))
            .find(|&j| bytes[j].trim_start().starts_with("fn ") || bytes[j].contains(" fn "))
        else {
            continue;
        };
        let name = bytes[fn_line]
            .split("fn ")
            .nth(1)
            .and_then(|r| r.split(['(', '<']).next())
            .unwrap_or("")
            .trim()
            .to_string();
        let mut depth = 0i32;
        let mut started = false;
        let mut body = String::new();
        for l in &bytes[fn_line..] {
            for c in l.chars() {
                if c == '{' {
                    depth += 1;
                    started = true;
                } else if c == '}' {
                    depth -= 1;
                }
            }
            body.push_str(l);
            body.push('\n');
            if started && depth <= 0 {
                break;
            }
        }
        out.push((name, body));
    }
    out
}

#[test]
fn tsan_scope_covers_every_module_that_can_start_a_thread() {
    if !in_source_repo() {
        eprintln!("not a git checkout (.github/ absent) — tsan scope guard skipped");
        return;
    }
    let entries = scope_entries();
    let mut gaps: Vec<String> = Vec::new();

    for (module, file) in module_files() {
        let Ok(text) = fs::read_to_string(&file) else {
            continue;
        };
        let stripped = strip_comments(&text);
        let shown = if module.is_empty() {
            "<crate root>"
        } else {
            &module
        };

        let spawns: Vec<&str> = SPAWN_TOKENS
            .iter()
            .copied()
            .filter(|t| stripped.contains(t))
            .collect();
        if !spawns.is_empty() && !is_covered(&module, &entries) {
            gaps.push(format!(
                "  {shown}  ({}) spawns threads: {}",
                file.display(),
                spawns.join(", ")
            ));
            continue;
        }

        if is_covered(&module, &entries) {
            continue;
        }

        // A module with tests that names a Rayon-backed entry point anywhere in
        // its code. The file-level condition is the one that decides; the loop
        // below only exists to name the offending test in the message. Deciding
        // at file level means a mis-braced body in `test_fn_bodies` can cost a
        // vague message but never a missed module.
        let file_calls: Vec<&str> = PARALLEL_ENTRY_POINTS
            .iter()
            .copied()
            .filter(|t| stripped.contains(t))
            .collect();
        if file_calls.is_empty() || !stripped.contains("#[test]") {
            continue;
        }
        let mut named = false;
        for (name, body) in test_fn_bodies(&stripped) {
            let calls: Vec<&str> = PARALLEL_ENTRY_POINTS
                .iter()
                .copied()
                .filter(|t| body.contains(t))
                .collect();
            if !calls.is_empty() {
                named = true;
                gaps.push(format!(
                    "  {shown}::…::{name}  ({}) calls: {}",
                    file.display(),
                    calls.join(", ")
                ));
            }
        }
        if !named {
            gaps.push(format!(
                "  {shown}  ({}) has tests and names: {}",
                file.display(),
                file_calls.join(", ")
            ));
        }
    }

    assert!(
        gaps.is_empty(),
        "these modules can run two threads at once but the nightly \
         ThreadSanitizer shard would not run their tests:\n{}\n\n\
         Add the module path to {} (or move the test into a module already \
         listed there). The shard only instruments what that file names — a \
         module missing from it is a race nothing in CI looks for.",
        gaps.join("\n"),
        scope_path().display(),
    );
}

#[test]
fn tsan_scope_entries_all_name_a_live_module() {
    if !in_source_repo() {
        eprintln!("not a git checkout (.github/ absent) — tsan scope guard skipped");
        return;
    }
    let entries = scope_entries();
    let modules: BTreeSet<String> = module_files().into_iter().map(|(m, _)| m).collect();

    let stale: Vec<&String> = entries
        .iter()
        .filter(|e| {
            !modules
                .iter()
                .any(|m| m == *e || m.starts_with(&format!("{e}::")))
        })
        .collect();

    assert!(
        stale.is_empty(),
        "{} names modules that do not exist in alkahest-core: {:?}\n\
         A filter that matches nothing makes the tsan shard pass vacuously. \
         Delete the entry or fix the path.",
        scope_path().display(),
        stale,
    );
}
