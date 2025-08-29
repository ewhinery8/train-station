use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Build script for Train Station
///
/// IMPORTANT: The core library has ZERO external dependencies.
/// CUDA FFI is compiled ONLY when the cuda feature is enabled.
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
    let cuda_cpp = PathBuf::from(&workspace_dir).join("src/cuda/cuda_wrapper.cpp");
    let cuda_obj = out_dir.join("cuda_wrapper.o");
    let cuda_so = out_dir.join("libcuda_wrapper.so");

    // Compile CUDA wrapper C++ file (CPU stub)
    let compile_output = Command::new("g++")
        .args([
            "-std=c++17",
            "-fPIC",
            "-c",
            &cuda_cpp.to_string_lossy(),
            "-O2",
            "-DNDEBUG",
            "-o",
            &cuda_obj.to_string_lossy(),
        ])
        .output()
        .expect("Failed to execute g++ for CUDA stub");
    if !compile_output.status.success() {
        panic!(
            "CUDA stub compilation failed:\n{}\n{}",
            String::from_utf8_lossy(&compile_output.stdout),
            String::from_utf8_lossy(&compile_output.stderr)
        );
    }

    // Link into a shared object
    let link_output = Command::new("g++")
        .args([
            "-shared",
            "-fPIC",
            "-std=c++17",
            &cuda_obj.to_string_lossy(),
            "-o",
            &cuda_so.to_string_lossy(),
        ])
        .output()
        .expect("Failed to execute g++ for linking CUDA stub");
    if !link_output.status.success() {
        panic!(
            "CUDA stub shared library creation failed:\n{}\n{}",
            String::from_utf8_lossy(&link_output.stdout),
            String::from_utf8_lossy(&link_output.stderr)
        );
    }

    // Link the shared library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=cuda_wrapper");

    // Rerun if sources change
    println!("cargo:rerun-if-changed=src/cuda/cuda_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/cuda/cuda_wrapper.h");
}
