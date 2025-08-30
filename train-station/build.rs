use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Build script for Train Station
///
/// # Build Process Overview
///
/// This build script handles cross-platform compilation of CUDA FFI stubs when the "cuda" feature is enabled.
/// The core library maintains ZERO external dependencies - all C++ code is statically linked.
///
/// ## Build Conventions
///
/// ### Target Matching
/// - **Convention**: Runtime target matches compilation target (no cross-runtime deployment)
/// - **Cross-compilation**: Supported via TARGET environment variable
/// - **Toolchain Detection**: Host-based compiler detection, target-based library linking
///
/// ### Static Linking Strategy
/// - **CUDA Stubs**: Always statically linked for zero runtime dependencies
/// - **C++ Standard Library**: Dynamically linked (platform standard practice)
/// - **Position Independent Code**: Enabled on Unix for future shared library compatibility
///
/// ### Platform-Specific Build Process
///
/// #### Windows
/// - **MSVC**: Uses `cl.exe` + `lib.exe` → `cuda_wrapper.lib` (static)
/// - **MinGW**: Uses `g++` + `ar` → `libcuda_wrapper.a` (static) + explicit `libstdc++` linking
/// - **Files**: `cuda_wrapper.obj` → `[lib]cuda_wrapper.{lib|a}`
///
/// #### macOS
/// - **Clang++**: Uses `clang++` + `ar` → `libcuda_wrapper.a` (static) + `libc++` linking
/// - **GCC**: Uses `g++` + `ar` → `libcuda_wrapper.a` (static) + `libc++` linking
/// - **Files**: `cuda_wrapper.o` → `libcuda_wrapper.a`
///
/// #### Linux/Unix
/// - **GCC**: Uses `g++` + `ar` → `libcuda_wrapper.a` (static) + `libstdc++` linking
/// - **Clang++**: Uses `clang++` + `ar` → `libcuda_wrapper.a` (static) + `libstdc++` linking
/// - **Files**: `cuda_wrapper.o` → `libcuda_wrapper.a`
///
/// ### Graceful Degradation
/// - Missing compilers: Skip build with warning (non-blocking)
/// - Missing archiver tools: Skip build with warning (non-blocking)
/// - Compilation failures: Skip build with detailed error output
///
/// ### Feature Flags
/// - **No "cuda" feature**: Build script exits early (zero-dependency core)
/// - **"cuda" feature enabled**: Builds CUDA FFI stubs as static library
/// - **docs.rs**: Build script exits early (documentation builds)
fn main() {
    // Skip build script entirely on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // Only build CUDA stubs when cuda feature is enabled
    let has_cuda_feature = std::env::var("CARGO_FEATURE_CUDA").is_ok();

    if !has_cuda_feature {
        // Core library builds with zero dependencies
        return;
    }

    let workspace_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // --- CUDA stub build (feature-gated) ---
    println!("cargo:warning=Building CUDA FFI stubs (no real CUDA linkage yet)");

    // Cross-platform compiler and static library detection
    let (compiler, obj_ext, lib_prefix, lib_ext, is_msvc) = if cfg!(target_os = "windows") {
        // Windows: try cl.exe first (MSVC), then g++ (MinGW)
        if Command::new("cl.exe").arg("/?").output().is_ok() {
            ("cl.exe", "obj", "", "lib", true)
        } else if Command::new("g++").arg("--version").output().is_ok() {
            ("g++", "o", "lib", "a", false)
        } else {
            println!("cargo:warning=No suitable C++ compiler found on Windows. Skipping CUDA stub build.");
            return;
        }
    } else if cfg!(target_os = "macos") {
        // macOS: prefer clang++, fallback to g++
        if Command::new("clang++").arg("--version").output().is_ok() {
            ("clang++", "o", "lib", "a", false)
        } else if Command::new("g++").arg("--version").output().is_ok() {
            ("g++", "o", "lib", "a", false)
        } else {
            println!(
                "cargo:warning=No suitable C++ compiler found on macOS. Skipping CUDA stub build."
            );
            return;
        }
    } else {
        // Linux and other Unix-like: prefer g++, fallback to clang++
        if Command::new("g++").arg("--version").output().is_ok() {
            ("g++", "o", "lib", "a", false)
        } else if Command::new("clang++").arg("--version").output().is_ok() {
            ("clang++", "o", "lib", "a", false)
        } else {
            println!("cargo:warning=No suitable C++ compiler found. Skipping CUDA stub build.");
            return;
        }
    };

    let cuda_cpp = PathBuf::from(&workspace_dir).join("src/cuda/cuda_wrapper.cpp");
    let cuda_obj = out_dir.join(format!("cuda_wrapper.{}", obj_ext));
    let cuda_lib = out_dir.join(format!("{}cuda_wrapper.{}", lib_prefix, lib_ext));

    // Compile CUDA wrapper C++ file (CPU stub)
    let compile_result = if compiler == "cl.exe" {
        // MSVC compiler
        let cuda_cpp_str = cuda_cpp.to_string_lossy();
        let cuda_obj_str = cuda_obj.to_string_lossy();
        Command::new(compiler)
            .args([
                "/std:c++17",
                "/c",
                &cuda_cpp_str,
                "/O2",
                "/DNDEBUG",
                &format!("/Fo:{}", cuda_obj_str),
            ])
            .output()
    } else {
        // GCC/Clang compiler
        let cuda_cpp_str = cuda_cpp.to_string_lossy();
        let cuda_obj_str = cuda_obj.to_string_lossy();
        let mut args = vec![
            "-std=c++17",
            "-c",
            &cuda_cpp_str,
            "-O2",
            "-DNDEBUG",
            "-o",
            &cuda_obj_str,
        ];

        // Add -fPIC for position-independent code (required for static libraries that may be linked into shared libraries)
        if !cfg!(target_os = "windows") {
            args.insert(1, "-fPIC");
        }

        Command::new(compiler).args(args).output()
    };

    let compile_output = match compile_result {
        Ok(output) => output,
        Err(e) => {
            println!("cargo:warning=Failed to execute {} for CUDA stub compilation: {}. Skipping CUDA stub build.", compiler, e);
            return;
        }
    };

    if !compile_output.status.success() {
        println!(
            "cargo:warning=CUDA stub compilation failed:\n{}\n{}. Skipping CUDA stub build.",
            String::from_utf8_lossy(&compile_output.stdout),
            String::from_utf8_lossy(&compile_output.stderr)
        );
        return;
    }

    // Create static library using appropriate archiver
    let link_result = if is_msvc {
        // MSVC lib.exe for static libraries - check if lib.exe is available
        if Command::new("lib.exe").arg("/?").output().is_err() {
            println!(
                "cargo:warning=lib.exe not found. Skipping CUDA stub static library creation."
            );
            return;
        }
        Command::new("lib.exe")
            .arg(format!("/OUT:{}", cuda_lib.display()))
            .arg(&cuda_obj)
            .output()
    } else {
        // Unix ar for static libraries - check if ar is available
        // Use a simple test that works on all ar implementations
        if Command::new("ar")
            .arg("t")
            .arg("/dev/null")
            .output()
            .is_err()
        {
            println!("cargo:warning=ar not found. Skipping CUDA stub static library creation.");
            return;
        }
        Command::new("ar")
            .arg("rcs")
            .arg(&cuda_lib)
            .arg(&cuda_obj)
            .output()
    };

    let link_output = match link_result {
        Ok(output) => output,
        Err(e) => {
            println!("cargo:warning=Failed to execute linker for CUDA stub: {}. Skipping CUDA stub build.", e);
            return;
        }
    };

    if !link_output.status.success() {
        println!(
            "cargo:warning=CUDA stub static library creation failed:\n{}\n{}. Skipping CUDA stub build.",
            String::from_utf8_lossy(&link_output.stdout),
            String::from_utf8_lossy(&link_output.stderr)
        );
        return;
    }

    // Link the static library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cuda_wrapper");

    // Link C++ standard library (required for C++ code)
    // Use TARGET environment variable for cross-compilation support
    let target = env::var("TARGET").unwrap_or_else(|_| env::var("HOST").unwrap_or_default());

    if target.contains("linux") || target.contains("android") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if target.contains("apple") || target.contains("darwin") || target.contains("ios") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target.contains("windows") && !is_msvc {
        // MinGW on Windows needs explicit libstdc++ linking
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    // Windows MSVC automatically links C++ runtime

    // Rerun if sources change
    println!("cargo:rerun-if-changed=src/cuda/cuda_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/cuda/cuda_wrapper.h");
}
