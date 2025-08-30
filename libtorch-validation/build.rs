use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Build script for libtorch-validation crate
///
/// # Build Process Overview
///
/// This build script handles cross-platform compilation of LibTorch FFI wrapper for validation and testing.
/// Unlike the core library, this crate uses DYNAMIC linking to integrate with LibTorch shared libraries.
///
/// ## Build Conventions
///
/// ### Target Matching
/// - **Convention**: Runtime target matches compilation target (no cross-runtime deployment)
/// - **Cross-compilation**: Supported via TARGET environment variable
/// - **Toolchain Detection**: Host-based compiler detection, target-based library linking
///
/// ### Dynamic Linking Strategy
/// - **LibTorch Libraries**: Always dynamically linked (`torch`, `torch_cpu`, `c10`)
/// - **FFI Wrapper**: Built as shared library for LibTorch integration
/// - **C++ Standard Library**: Dynamically linked (platform standard practice)
/// - **Runtime Path**: RPATH set on Unix for automatic library discovery
///
/// ### Platform-Specific Build Process
///
/// #### Windows
/// - **MSVC**: Uses `cl.exe` + `link.exe` → `libtorch_wrapper.dll` (shared)
/// - **MinGW**: Uses `g++` (shared linking) → `liblibtorch_wrapper.dll` (shared) + explicit `libstdc++` linking
/// - **Files**: `libtorch_wrapper.obj` → `[lib]libtorch_wrapper.dll`
/// - **LibTorch**: Links against `torch.lib`, `torch_cpu.lib`, `c10.lib`
///
/// #### macOS
/// - **Clang++**: Uses `clang++` (shared linking) → `liblibtorch_wrapper.dylib` (shared) + `libc++` linking
/// - **GCC**: Uses `g++` (shared linking) → `liblibtorch_wrapper.dylib` (shared) + `libc++` linking
/// - **Files**: `libtorch_wrapper.o` → `liblibtorch_wrapper.dylib`
/// - **LibTorch**: Links against `libtorch`, `libtorch_cpu`, `libc10`
/// - **RPATH**: Set to find LibTorch libraries at runtime
///
/// #### Linux/Unix
/// - **GCC**: Uses `g++` (shared linking) → `liblibtorch_wrapper.so` (shared) + `libstdc++` linking
/// - **Clang++**: Uses `clang++` (shared linking) → `liblibtorch_wrapper.so` (shared) + `libstdc++` linking
/// - **Files**: `libtorch_wrapper.o` → `liblibtorch_wrapper.so`
/// - **LibTorch**: Links against `libtorch`, `libtorch_cpu`, `libc10`
/// - **RPATH**: Set to find LibTorch libraries at runtime
///
/// ### LibTorch Integration
/// - **Library Path**: Expected at `./libtorch/lib/` within crate directory
/// - **Include Path**: Uses `./libtorch/include/` and `./libtorch/include/torch/csrc/api/include/`
/// - **Runtime Discovery**: RPATH (Unix) or PATH (Windows) for dynamic library loading
///
/// ### Graceful Degradation
/// - Missing LibTorch: Skip build with warning (validation tests won't run)
/// - Missing compilers: Skip build with warning (non-blocking)
/// - Missing linker tools: Skip build with warning (non-blocking)
/// - Compilation failures: Skip build with detailed error output
///
/// ### Validation Purpose
/// - **Mathematical Validation**: Compare Train Station results against LibTorch reference
/// - **Performance Benchmarking**: Measure performance against LibTorch baseline
/// - **Correctness Testing**: Ensure numerical equivalence across all operations
fn main() {
    let workspace_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Check if LibTorch is available before building FFI
    // LibTorch is now organized within the libtorch-validation crate directory
    let libtorch_path = PathBuf::from(&workspace_dir).join("libtorch");
    let libtorch_lib_path = libtorch_path.join("lib");

    if !libtorch_lib_path.exists() {
        println!(
            "cargo:warning=LibTorch not found at {}, skipping FFI build",
            libtorch_lib_path.display()
        );
        return;
    }

    println!("cargo:warning=Building LibTorch FFI for validation");

    // Set up libtorch library linking
    println!(
        "cargo:rustc-link-search=native={}/lib",
        libtorch_path.display()
    );
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    // Cross-platform compiler and shared library detection
    let (compiler, obj_ext, lib_prefix, lib_ext, is_msvc) = if cfg!(target_os = "windows") {
        // Windows: try cl.exe first (MSVC), then g++ (MinGW)
        if Command::new("cl.exe").arg("/?").output().is_ok() {
            ("cl.exe", "obj", "", "dll", true)
        } else if Command::new("g++").arg("--version").output().is_ok() {
            ("g++", "o", "lib", "dll", false)
        } else {
            println!("cargo:warning=No suitable C++ compiler found on Windows. Skipping LibTorch FFI build.");
            return;
        }
    } else if cfg!(target_os = "macos") {
        // macOS: prefer clang++, fallback to g++
        if Command::new("clang++").arg("--version").output().is_ok() {
            ("clang++", "o", "lib", "dylib", false)
        } else if Command::new("g++").arg("--version").output().is_ok() {
            ("g++", "o", "lib", "dylib", false)
        } else {
            println!("cargo:warning=No suitable C++ compiler found on macOS. Skipping LibTorch FFI build.");
            return;
        }
    } else {
        // Linux and other Unix-like: prefer g++, fallback to clang++
        if Command::new("g++").arg("--version").output().is_ok() {
            ("g++", "o", "lib", "so", false)
        } else if Command::new("clang++").arg("--version").output().is_ok() {
            ("clang++", "o", "lib", "so", false)
        } else {
            println!("cargo:warning=No suitable C++ compiler found. Skipping LibTorch FFI build.");
            return;
        }
    };

    let include_path = libtorch_path.join("include");
    let cpp_file = PathBuf::from(&workspace_dir).join("src/ffi/libtorch_wrapper.cpp");
    let obj_file = out_dir.join(format!("libtorch_wrapper.{}", obj_ext));
    let lib_file = out_dir.join(format!("{}libtorch_wrapper.{}", lib_prefix, lib_ext));

    // Compile to object file
    let compile_result = if compiler == "cl.exe" {
        // MSVC compiler
        let cpp_file_str = cpp_file.to_string_lossy();
        let obj_file_str = obj_file.to_string_lossy();
        let include_path_str = include_path.to_string_lossy();
        let torch_api_include = include_path.join("torch/csrc/api/include");
        let torch_api_include_str = torch_api_include.to_string_lossy();
        Command::new(compiler)
            .args([
                "/std:c++17",
                "/c",
                &format!("/I{}", include_path_str),
                &format!("/I{}", torch_api_include_str),
                "/O2",
                "/DNDEBUG",
                &cpp_file_str,
                &format!("/Fo:{}", obj_file_str),
            ])
            .output()
    } else {
        // GCC/Clang compiler
        let cpp_file_str = cpp_file.to_string_lossy();
        let obj_file_str = obj_file.to_string_lossy();
        let include_path_str = include_path.to_string_lossy();
        let torch_api_include = include_path.join("torch/csrc/api/include");
        let torch_api_include_str = torch_api_include.to_string_lossy();
        let include_flag = format!("-I{}", include_path_str);
        let torch_api_include_flag = format!("-I{}", torch_api_include_str);
        let mut args = vec![
            "-std=c++17",
            "-c",
            &include_flag,
            &torch_api_include_flag,
            "-O2",
            "-DNDEBUG",
            &cpp_file_str,
            "-o",
            &obj_file_str,
        ];

        // Add -fPIC for shared libraries on Unix-like systems
        if !cfg!(target_os = "windows") {
            args.insert(1, "-fPIC");
        }

        Command::new(compiler).args(args).output()
    };

    let compile_output = match compile_result {
        Ok(output) => output,
        Err(e) => {
            println!("cargo:warning=Failed to execute {} for LibTorch FFI compilation: {}. Skipping LibTorch FFI build.", compiler, e);
            return;
        }
    };

    if !compile_output.status.success() {
        println!(
            "cargo:warning=LibTorch FFI compilation failed:\n{}\n{}. Skipping LibTorch FFI build.",
            String::from_utf8_lossy(&compile_output.stdout),
            String::from_utf8_lossy(&compile_output.stderr)
        );
        return;
    }

    // Link into a shared library
    let link_result = if is_msvc {
        // MSVC linker - check if link.exe is available
        if Command::new("link.exe").arg("/?").output().is_err() {
            println!("cargo:warning=link.exe not found. Skipping LibTorch FFI build.");
            return;
        }
        let libtorch_lib_path = libtorch_path.join("lib");
        Command::new("link.exe")
            .arg("/DLL")
            .arg(&obj_file)
            .arg(format!("/LIBPATH:{}", libtorch_lib_path.display()))
            .arg("torch.lib")
            .arg("torch_cpu.lib")
            .arg("c10.lib")
            .arg(format!("/OUT:{}", lib_file.display()))
            .output()
    } else {
        // GCC/Clang linker
        let libtorch_lib_path = libtorch_path.join("lib");
        let mut cmd = Command::new(compiler);
        cmd.arg("-shared");

        // Add -fPIC for shared libraries on Unix-like systems
        if !cfg!(target_os = "windows") {
            cmd.arg("-fPIC");
            cmd.arg("-std=c++17");
        }

        cmd.arg(&obj_file)
            .arg(format!("-L{}", libtorch_lib_path.display()))
            .arg("-ltorch")
            .arg("-ltorch_cpu")
            .arg("-lc10")
            .arg("-o")
            .arg(&lib_file)
            .output()
    };

    let link_output = match link_result {
        Ok(output) => output,
        Err(e) => {
            println!("cargo:warning=Failed to execute linker for LibTorch FFI: {}. Skipping LibTorch FFI build.", e);
            return;
        }
    };

    if !link_output.status.success() {
        println!(
            "cargo:warning=LibTorch FFI shared library creation failed:\n{}\n{}. Skipping LibTorch FFI build.",
            String::from_utf8_lossy(&link_output.stdout),
            String::from_utf8_lossy(&link_output.stderr)
        );
        return;
    }

    // Link the shared library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=libtorch_wrapper");

    // Set runtime path for finding the library (Unix-only)
    if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir.display());
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}/lib",
            libtorch_path.display()
        );
    }

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

    // Tell Cargo to rerun if the wrapper files change
    println!("cargo:rerun-if-changed=src/ffi/libtorch_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/ffi/libtorch_wrapper.h");
}
