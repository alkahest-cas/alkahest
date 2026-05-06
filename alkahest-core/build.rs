fn main() {
    // Custom cfgs from this build script; keeps `unexpected_cfgs` quiet.
    println!("cargo::rustc-check-cfg=cfg(flint3)");
    // flint3_stride: fmpz_mat_struct uses `stride: slong` instead of `rows: **fmpz`.
    // Introduced sometime between FLINT 3.0.1 and 3.5.0; detected from the header.
    println!("cargo::rustc-check-cfg=cfg(flint3_stride)");
    println!("cargo:rerun-if-changed=build.rs");

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

    println!("cargo:rustc-link-lib=flint");
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
    if cfg!(target_os = "macos") {
        if let Some(prefix) = brew_prefix("flint") {
            return vec![format!("{prefix}/include")];
        }
    }
    if cfg!(target_os = "windows") {
        let msys =
            std::env::var("MSYS2_PREFIX").unwrap_or_else(|_| "C:/msys64/mingw64".to_string());
        return vec![format!("{msys}/include")];
    }
    vec![
        "/usr/include".to_string(),
        "/usr/local/include".to_string(),
    ]
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
        println!("cargo:warning=FLINT version (header/pkg-config): {ver}");
        return flint_major_at_least_3(&ver);
    }
    // Header and pkg-config both failed (e.g., libflint-dev ships no .pc on
    // Debian/Ubuntu and the header path differs).  Fall back to symbol
    // inspection: FLINT 3 exports `nmod_poly_factor_get_poly` while FLINT 2
    // exported `nmod_poly_factor_get_nmod_poly`.
    println!("cargo:warning=FLINT header/pkg-config detection failed; trying nm");
    let r = detect_flint3_by_nm();
    println!("cargo:warning=FLINT nm symbol detection → flint3={r}");
    r
}

/// FLINT 3.0.x still used `rows: **fmpz` in fmpz_mat_struct (same as FLINT 2).
/// The stride-based layout was introduced later (visible in FLINT 3.5.0).
/// Detect by looking for a `stride` field declaration in fmpz_mat.h.
fn detect_flint3_stride() -> bool {
    for dir in flint_include_dirs() {
        let path = format!("{dir}/flint/fmpz_mat.h");
        if let Ok(content) = std::fs::read_to_string(&path) {
            // Match a line like "    slong stride;" inside the struct.
            let found = content.lines().any(|raw| {
                let l = raw.trim();
                l.ends_with("stride;")
                    && !l.starts_with("//")
                    && !l.starts_with('*')
                    && !l.starts_with('#')
            });
            if found {
                println!("cargo:warning=FLINT fmpz_mat_struct uses stride layout (from {path})");
                return true;
            } else {
                println!("cargo:warning=FLINT fmpz_mat_struct uses rows layout (from {path})");
                return false;
            }
        }
    }
    // Could not find the header — fall back to version-based heuristic.
    // FLINT 3.0.x uses rows; anything 3.1+ may use stride.
    if let Some(ver) = flint_version_string() {
        let parts: Vec<u32> = ver.split('.').filter_map(|s| s.parse().ok()).collect();
        if parts.len() >= 2 && parts[0] >= 3 && parts[1] >= 1 {
            println!("cargo:warning=FLINT stride layout assumed from version {ver}");
            return true;
        }
    }
    false
}

fn detect_flint3_by_nm() -> bool {
    detect_flint3_by_nm_inner().unwrap_or(false)
}

fn detect_flint3_by_nm_inner() -> Option<bool> {
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
    println!("cargo:warning=FLINT library found by ldconfig: {lib_path}");
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
