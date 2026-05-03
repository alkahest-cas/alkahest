fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // On macOS, ask Homebrew for the FLINT prefix so rust-lld can find
    // libflint.dylib.  When cross-compiling to x86_64 from an Apple Silicon
    // host the x86_64 Homebrew lives at /usr/local (Rosetta) rather than the
    // native ARM64 prefix /opt/homebrew, so pick the right binary.
    if cfg!(target_os = "macos") {
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let brew = if target_arch == "x86_64" {
            "/usr/local/bin/brew"
        } else {
            "brew"
        };
        if let Ok(out) = std::process::Command::new(brew)
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

    println!("cargo:rustc-link-lib=flint");
}
