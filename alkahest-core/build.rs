fn main() {
    // Custom cfg from this build script (FLINT 2 vs 3 API); keeps `unexpected_cfgs` quiet.
    println!("cargo::rustc-check-cfg=cfg(flint3)");
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

    println!("cargo:rustc-link-lib=flint");
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

fn flint_version_string() -> Option<String> {
    if let Some(v) = pkg_config_modversion(None) {
        return Some(v);
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

    let msys = std::env::var("MSYS2_PREFIX").unwrap_or_else(|_| "C:/msys64/mingw64".to_string());
    let pcp = format!("{msys}/lib/pkgconfig");
    if let Some(v) = pkg_config_modversion(Some(&pcp)) {
        return Some(v);
    }
    let pc_path = format!("{msys}/lib/pkgconfig/flint.pc");
    if let Some(v) = read_version_from_pc(&pc_path) {
        return Some(v);
    }

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

    None
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
    let Some(ver) = flint_version_string() else {
        return false;
    };
    flint_major_at_least_3(&ver)
}
