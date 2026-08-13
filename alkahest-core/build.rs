/// Informational FLINT probe output (stderr only — avoids rustc-style `cargo:warning=` spam in CI).
fn flint_probe_note(msg: impl std::fmt::Display) {
    eprintln!("[alkahest-core/build.rs] {msg}");
}

fn main() {
    // Custom cfgs from this build script; keeps `unexpected_cfgs` quiet.
    println!("cargo::rustc-check-cfg=cfg(flint3)");
    // flint3_stride: fmpz_mat_struct uses `stride: slong` instead of `rows: **fmpz`.
    // FLINT 3.2.2 still uses `rows`; the change lands by 3.5.0. Read from the
    // header (`flint/fmpz_types.h`), never guessed, unless no header is found.
    println!("cargo::rustc-check-cfg=cfg(flint3_stride)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=FLINT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=FLINT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MSYS2_PREFIX");
    println!("cargo:rerun-if-env-changed=ALKAHEST_SKIP_FLINT_CHECK");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    // Explicit prefix override — for a FLINT built into a user-local prefix
    // (no root required).  `FLINT_LIB_DIR` is the directory containing
    // `libflint.so` / `.dylib` / `.dll.a`; `FLINT_INCLUDE_DIR` is the one
    // containing `flint/flint.h`.  Both are also consulted by the version
    // probes below, so a locally built FLINT 3 is detected as FLINT 3.
    if let Ok(dir) = std::env::var("FLINT_LIB_DIR") {
        if !dir.trim().is_empty() {
            println!("cargo:rustc-link-search=native={}", dir.trim());
        }
    }

    // On macOS with Homebrew (especially Apple Silicon / M1), libraries live
    // under /opt/homebrew rather than /usr/local.  Ask brew for the FLINT
    // prefix and emit the search path so rust-lld can find libflint.dylib.
    if cfg!(target_os = "macos") {
        if let Ok(out) = std::process::Command::new("brew")
            .args(["--prefix", "flint"])
            .output()
        {
            if out.status.success() {
                let prefix = String::from_utf8_lossy(&out.stdout).trim().to_string();
                println!("cargo:rustc-link-search=native={prefix}/lib");
            }
        }
    }

    // On Windows with the MinGW64 toolchain (MSYS2), FLINT lives under the
    // MSYS2 MinGW64 prefix.  Override with MSYS2_PREFIX if your installation
    // differs from the GitHub Actions default (C:/msys64/mingw64).
    if cfg!(target_os = "windows") {
        let prefix =
            std::env::var("MSYS2_PREFIX").unwrap_or_else(|_| "C:/msys64/mingw64".to_string());
        println!("cargo:rustc-link-search=native={prefix}/lib");
    }

    if detect_flint3() {
        println!("cargo:rustc-cfg=flint3");
    }

    if detect_flint3_stride() {
        println!("cargo:rustc-cfg=flint3_stride");
    }

    // Fail fast, with an actionable message, when no FLINT can be found.
    // Without this the build runs to completion and then dies at link time
    // with `unable to find library -lflint` — or, for a `cdylib`, links a
    // shared object full of undefined symbols that only fails at
    // `import alkahest` with `undefined symbol: nmod_poly_init`.
    check_flint_present();

    // NOTE: this link is deliberately *unconditional*, and must stay that way.
    //
    // The `flint3` Cargo feature and the `flint3` cfg emitted above select
    // *which FLINT API version* to call (FLINT 3 renamed one factorisation
    // accessor and changed `fmpz_mat_struct`'s layout).  Neither makes FLINT
    // itself optional: `crate::flint` is compiled unconditionally from
    // `lib.rs`, `UniPoly` stores a `FlintPoly`, and integer factorisation,
    // polynomial factorisation, resultants, Hermite/Smith normal forms and the
    // number-theory module all call FLINT directly.  There is no MPFR-only or
    // pure-Rust fallback for any of that, so gating this line behind
    // `cfg(flint3)` would not produce a FLINT-free build — it would only
    // convert a clear link error into a runtime `ImportError`.
    //
    // `alkahest-core/src/flint/ffi.rs` also carries `#[link(name = "flint")]`;
    // rustc de-duplicates the two, so removing either one alone changes
    // nothing.  Keep both: the attribute covers `cargo doc`/rustdoc paths that
    // do not re-run the build script's output, and this line keeps the
    // directive next to the `FLINT_LIB_DIR` search path emitted above.
    println!("cargo:rustc-link-lib=flint");
}

/// Directories to search for `libflint`.  `FLINT_LIB_DIR` wins; otherwise the
/// platform defaults that match the `rustc-link-search` lines emitted above.
fn flint_lib_dirs() -> Vec<String> {
    let mut dirs = Vec::new();
    if let Ok(d) = std::env::var("FLINT_LIB_DIR") {
        if !d.trim().is_empty() {
            dirs.push(d.trim().to_string());
        }
    }
    if cfg!(target_os = "macos") {
        if let Some(prefix) = brew_prefix("flint") {
            dirs.push(format!("{prefix}/lib"));
        }
    }
    if cfg!(target_os = "windows") {
        let msys =
            std::env::var("MSYS2_PREFIX").unwrap_or_else(|_| "C:/msys64/mingw64".to_string());
        dirs.push(format!("{msys}/lib"));
    }
    for d in [
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib",
        "/usr/local/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ] {
        dirs.push(d.to_string());
    }
    dirs
}

/// Best-effort check that a linkable FLINT exists.  Every probe is a positive
/// test: the hard error only fires when *all* of them come back empty, which
/// is the case where `-lflint` was going to fail anyway.
fn flint_is_linkable() -> bool {
    // 1. A library file in one of the search directories.
    for dir in flint_lib_dirs() {
        for name in [
            "libflint.so",
            "libflint.dylib",
            "libflint.a",
            "libflint.dll.a",
            "flint.lib",
        ] {
            if std::path::Path::new(&format!("{dir}/{name}")).exists() {
                flint_probe_note(format!("FLINT library found: {dir}/{name}"));
                return true;
            }
        }
    }

    // 2. Ask the C compiler driver: `-print-file-name` resolves against the
    //    linker's own search path and echoes the input back when it misses.
    for cc in [
        std::env::var("CC").unwrap_or_default(),
        "cc".to_string(),
        "gcc".to_string(),
    ] {
        if cc.is_empty() {
            continue;
        }
        let stem = if cfg!(target_os = "macos") {
            "libflint.dylib"
        } else {
            "libflint.so"
        };
        if let Ok(out) = std::process::Command::new(&cc)
            .arg(format!("-print-file-name={stem}"))
            .output()
        {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if p != stem && std::path::Path::new(&p).exists() {
                flint_probe_note(format!("FLINT library found via {cc}: {p}"));
                return true;
            }
        }
    }

    // 3. Headers or pkg-config metadata — a `-dev` package is installed.
    if flint_version_string().is_some() {
        return true;
    }

    // 4. The dynamic-linker cache.
    if let Ok(out) = std::process::Command::new("ldconfig").arg("-p").output() {
        if String::from_utf8_lossy(&out.stdout).contains("libflint") {
            return true;
        }
    }

    false
}

fn check_flint_present() {
    if std::env::var("ALKAHEST_SKIP_FLINT_CHECK").is_ok() {
        flint_probe_note("ALKAHEST_SKIP_FLINT_CHECK set — skipping the FLINT presence probe");
        return;
    }
    // docs.rs only runs `cargo doc`, which never links, and its build image has
    // no FLINT.  Refusing there would take down the published API docs.
    if std::env::var("DOCS_RS").is_ok() {
        flint_probe_note("DOCS_RS set — skipping the FLINT presence probe (rustdoc does not link)");
        return;
    }
    if flint_is_linkable() {
        return;
    }
    panic!(
        "\n\
         alkahest-cas requires a system FLINT (>= 2.9, >= 3.0 recommended) and none was found.\n\
         FLINT is a hard dependency, not an optional feature: polynomial and integer arithmetic\n\
         call it directly and there is no pure-Rust fallback. The `flint3` Cargo feature selects\n\
         which FLINT *version's* API to use; it does not make FLINT optional.\n\
         \n\
         Install it:\n\
         \x20 Debian / Ubuntu : sudo apt-get install libflint-dev\n\
         \x20 Fedora / RHEL   : sudo dnf install flint-devel\n\
         \x20 Arch            : sudo pacman -S flint\n\
         \x20 macOS           : brew install flint\n\
         \x20 Windows (MSYS2) : pacman -S mingw-w64-x86_64-flint\n\
         \x20 conda / mamba   : conda install -c conda-forge libflint\n\
         \n\
         No root? Build FLINT into a user-local prefix and point this build at it:\n\
         \x20 FLINT_LIB_DIR=$PREFIX/lib FLINT_INCLUDE_DIR=$PREFIX/include cargo build\n\
         (also set LD_LIBRARY_PATH=$PREFIX/lib, or DYLD_LIBRARY_PATH on macOS, at run time).\n\
         \n\
         Or skip the source build entirely: `pip install alkahest` ships prebuilt wheels with\n\
         FLINT already linked in.\n\
         \n\
         Set ALKAHEST_SKIP_FLINT_CHECK=1 to bypass this probe if you know FLINT is reachable\n\
         by some route it does not cover.\n"
    );
}

/// Parse `__FLINT_VERSION`, `__FLINT_VERSION_MINOR`, `__FLINT_VERSION_PATCHLEVEL`
/// from a FLINT header file.  Debian/Ubuntu's `libflint-dev` does not always
/// ship a `flint.pc`, so the header is more reliable than pkg-config.
fn read_version_from_flint_header(path: &str) -> Option<String> {
    let data = std::fs::read_to_string(path).ok()?;
    let mut major: Option<u32> = None;
    let mut minor: Option<u32> = None;
    let mut patch: Option<u32> = None;
    for raw in data.lines() {
        let line = raw.trim();
        if let Some(v) = line.strip_prefix("#define __FLINT_VERSION ") {
            major = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("#define __FLINT_VERSION_MINOR ") {
            minor = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("#define __FLINT_VERSION_PATCHLEVEL ") {
            patch = v.trim().parse().ok();
        }
    }
    match (major, minor, patch) {
        (Some(ma), Some(mi), Some(p)) => Some(format!("{ma}.{mi}.{p}")),
        (Some(ma), Some(mi), None) => Some(format!("{ma}.{mi}")),
        _ => None,
    }
}

fn read_version_from_pc(path: &str) -> Option<String> {
    let data = std::fs::read_to_string(path).ok()?;
    for raw in data.lines() {
        let line = raw.trim();
        if let Some(v) = line.strip_prefix("Version:") {
            return Some(v.trim().to_string());
        }
    }
    None
}

fn pkg_config_modversion(pkg_config_path: Option<&str>) -> Option<String> {
    let mut cmd = std::process::Command::new("pkg-config");
    if let Some(p) = pkg_config_path {
        cmd.env("PKG_CONFIG_PATH", p);
    }
    let out = cmd.args(["--modversion", "flint"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn brew_prefix(formula: &str) -> Option<String> {
    let out = std::process::Command::new("brew")
        .args(["--prefix", formula])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if p.is_empty() {
        return None;
    }
    Some(p)
}

fn flint_include_dirs() -> Vec<String> {
    // An explicit override always comes first, so a user-local FLINT build is
    // version-detected correctly even when a different FLINT is on the system.
    let mut dirs = Vec::new();
    if let Ok(d) = std::env::var("FLINT_INCLUDE_DIR") {
        if !d.trim().is_empty() {
            dirs.push(d.trim().to_string());
        }
    }
    if cfg!(target_os = "windows") {
        let msys =
            std::env::var("MSYS2_PREFIX").unwrap_or_else(|_| "C:/msys64/mingw64".to_string());
        dirs.push(format!("{msys}/include"));
        return dirs;
    }
    if cfg!(target_os = "macos") {
        if let Some(prefix) = brew_prefix("flint") {
            dirs.push(format!("{prefix}/include"));
            return dirs;
        }
    }
    dirs.push("/usr/include".to_string());
    dirs.push("/usr/local/include".to_string());
    dirs
}

fn flint_version_string() -> Option<String> {
    // Probe FLINT header first — `libflint-dev` on Debian/Ubuntu does not always
    // ship `flint.pc`, so the header is the most reliable source on Linux.
    for dir in flint_include_dirs() {
        if let Some(v) = read_version_from_flint_header(&format!("{dir}/flint/flint.h")) {
            return Some(v);
        }
    }

    // Fallback: try pkg-config (.pc files may or may not be present).
    // Linux: read known .pc paths before calling pkg-config, so that
    // Actions' PKG_CONFIG_PATH (pointing to Python's pkgconfig) cannot
    // shadow the distro FLINT.
    if cfg!(target_os = "linux") {
        for pc in [
            "/usr/lib/x86_64-linux-gnu/pkgconfig/flint.pc",
            "/usr/lib/pkgconfig/flint.pc",
        ] {
            if let Some(v) = read_version_from_pc(pc) {
                return Some(v);
            }
        }
    }

    if cfg!(target_os = "macos") {
        if let Some(prefix) = brew_prefix("flint") {
            let pc = format!("{prefix}/lib/pkgconfig/flint.pc");
            if let Some(v) = read_version_from_pc(&pc) {
                return Some(v);
            }
            let pcp = format!("{prefix}/lib/pkgconfig");
            if let Some(v) = pkg_config_modversion(Some(&pcp)) {
                return Some(v);
            }
        }
    }

    if let Some(v) = pkg_config_modversion(None) {
        return Some(v);
    }

    let msys = std::env::var("MSYS2_PREFIX").unwrap_or_else(|_| "C:/msys64/mingw64".to_string());
    let pcp = format!("{msys}/lib/pkgconfig");
    if let Some(v) = pkg_config_modversion(Some(&pcp)) {
        return Some(v);
    }
    let pc_path = format!("{msys}/lib/pkgconfig/flint.pc");
    read_version_from_pc(&pc_path)
}

fn flint_major_at_least_3(version: &str) -> bool {
    let Some(major_s) = version.split('.').next() else {
        return false;
    };
    let Ok(major) = major_s.parse::<u32>() else {
        return false;
    };
    major >= 3
}

/// FLINT 3 renamed `nmod_poly_factor_get_nmod_poly` → `nmod_poly_factor_get_poly`.
fn detect_flint3() -> bool {
    if let Some(ver) = flint_version_string() {
        flint_probe_note(format!("FLINT version (header/pkg-config): {ver}"));
        return flint_major_at_least_3(&ver);
    }
    // Header and pkg-config both failed (e.g., libflint-dev ships no .pc on
    // Debian/Ubuntu and the header path differs).  Fall back to symbol
    // inspection: FLINT 3 exports `nmod_poly_factor_get_poly` while FLINT 2
    // exported `nmod_poly_factor_get_nmod_poly`.
    flint_probe_note("FLINT header/pkg-config detection failed; trying nm");
    let r = detect_flint3_by_nm();
    flint_probe_note(format!("FLINT nm symbol detection → flint3={r}"));
    r
}

/// Slice out the body of the `typedef struct { … } fmpz_mat_struct;` declaration.
///
/// The struct terminator is the *name*, which trails the closing brace, so
/// locate `fmpz_mat_struct;` and walk back to the `typedef struct` that opened
/// it.  `typedef fmpz_mat_struct fmpz_mat_t[1];` mentions the name too, hence
/// matching on the `;`-terminated form only.
fn extract_fmpz_mat_struct(content: &str) -> Option<&str> {
    let end = content.find("fmpz_mat_struct;")?;
    let head = &content[..end];
    let start = head.rfind("typedef struct")?;
    Some(&head[start..])
}

/// FLINT 2.x and FLINT 3.0–3.2 use `rows: **fmpz` in `fmpz_mat_struct`; later
/// FLINT 3 releases replaced it with `stride: slong`.  Both fields are
/// pointer-sized, so a misdetection is not a size mismatch — it makes
/// `FlintMat` dereference an integer as a pointer.  Get it from the header.
///
/// The declaration lives in `flint/fmpz_types.h` on FLINT 3 (it is *not* in
/// `flint/fmpz_mat.h`, which only includes it), and in `flint/fmpz_mat.h` on
/// FLINT 2.  Both are searched, and a header that does not contain the
/// declaration at all is skipped rather than read as "no stride field" —
/// otherwise every FLINT 3 would be reported as the `rows` layout.
fn detect_flint3_stride() -> bool {
    for dir in flint_include_dirs() {
        for header in ["flint/fmpz_types.h", "flint/fmpz_mat.h"] {
            let path = format!("{dir}/{header}");
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Some(body) = extract_fmpz_mat_struct(&content) else {
                continue;
            };
            // Match a field line like "    slong stride;" inside the struct.
            let found = body.lines().any(|raw| {
                let l = raw.trim();
                l.ends_with("stride;")
                    && !l.starts_with("//")
                    && !l.starts_with('*')
                    && !l.starts_with('#')
            });
            let layout = if found { "stride" } else { "rows" };
            flint_probe_note(format!(
                "FLINT fmpz_mat_struct uses {layout} layout (from {path})"
            ));
            return found;
        }
    }
    // No header carried the declaration — fall back to a version heuristic.
    // FLINT 3.2.2 is confirmed to still use `rows`, so only assume `stride`
    // from 3.5 onward, where it is known to be present.
    if let Some(ver) = flint_version_string() {
        let parts: Vec<u32> = ver.split('.').filter_map(|s| s.parse().ok()).collect();
        if parts.len() >= 2 && (parts[0] > 3 || (parts[0] == 3 && parts[1] >= 5)) {
            flint_probe_note(format!("FLINT stride layout assumed from version {ver}"));
            return true;
        }
    }
    flint_probe_note("FLINT fmpz_mat_struct layout undetermined; assuming rows layout");
    false
}

fn detect_flint3_by_nm() -> bool {
    detect_flint3_by_nm_inner().unwrap_or(false)
}

fn detect_flint3_by_nm_inner() -> Option<bool> {
    let lib_path = locate_flint_library()?;
    let nm = std::process::Command::new("nm")
        .args(["-D", "--defined-only", &lib_path])
        .output()
        .ok()?;
    let syms = String::from_utf8_lossy(&nm.stdout);
    Some(
        syms.lines()
            .any(|l| l.split_whitespace().last() == Some("nmod_poly_factor_get_poly")),
    )
}

/// Path to a `libflint` shared object, for symbol inspection.
///
/// `FLINT_LIB_DIR` (and the other search directories) come first, so that a
/// prefix supplied *without* `FLINT_INCLUDE_DIR` — e.g. the `libflint.so`
/// unpacked from a `python-flint` wheel, which ships no headers — is still
/// version-detected instead of silently falling back to the FLINT 2 API.
fn locate_flint_library() -> Option<String> {
    for dir in flint_lib_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        // Prefer the plain `libflint.so`; accept `libflint*.so*` (auditwheel
        // rewrites it to e.g. `libflint-39a891bf.so.17`).
        let mut candidates: Vec<String> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("libflint") && (n.contains(".so") || n.contains(".dylib")))
            .collect();
        candidates.sort();
        if let Some(name) = candidates
            .iter()
            .find(|n| n.as_str() == "libflint.so" || n.as_str() == "libflint.dylib")
            .or_else(|| candidates.first())
        {
            let path = format!("{dir}/{name}");
            flint_probe_note(format!("FLINT library located for nm probe: {path}"));
            return Some(path);
        }
    }

    // ldconfig -p lists cached shared libraries with their full paths.
    let ldconfig = std::process::Command::new("ldconfig")
        .arg("-p")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&ldconfig.stdout);
    let lib_path = text
        .lines()
        .filter(|l| l.contains("libflint"))
        .filter_map(|l| l.split("=>").nth(1).map(|s| s.trim().to_string()))
        .find(|p| !p.is_empty())?;
    flint_probe_note(format!("FLINT library found by ldconfig: {lib_path}"));
    Some(lib_path)
}
