use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Build script for libtorch-validation crate
///
/// This crate is specifically for LibTorch validation and testing.
/// The FFI is always built since this is the validation crate.
/// LibTorch library is expected to be in ./libtorch/ within this crate.
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

    // Compile the C++ wrapper directly using g++
    let include_path = libtorch_path.join("include");
    let cpp_file = PathBuf::from(&workspace_dir).join("src/ffi/libtorch_wrapper.cpp");
    let obj_file = out_dir.join("libtorch_wrapper.o");

    // Compile to object file
    let compile_output = Command::new("g++")
        .args([
            "-std=c++17",
            "-fPIC",
            "-c",
            &format!("-I{}", include_path.display()),
            &format!(
                "-I{}",
                include_path.join("torch/csrc/api/include").display()
            ),
            "-O2",
            "-DNDEBUG",
            &cpp_file.to_string_lossy(),
            "-o",
            &obj_file.to_string_lossy(),
        ])
        .output()
        .expect("Failed to execute g++");

    if !compile_output.status.success() {
        panic!(
            "C++ compilation failed:\n{}\n{}",
            String::from_utf8_lossy(&compile_output.stdout),
            String::from_utf8_lossy(&compile_output.stderr)
        );
    }

    // Compile directly into a shared object
    let so_file = out_dir.join("liblibtorch_wrapper.so");

    let link_output = Command::new("g++")
        .args([
            "-shared",
            "-fPIC",
            "-std=c++17",
            &obj_file.to_string_lossy(),
            &format!("-L{}", libtorch_path.join("lib").display()),
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
            "-o",
            &so_file.to_string_lossy(),
        ])
        .output()
        .expect("Failed to execute g++ for linking");

    if !link_output.status.success() {
        panic!(
            "Shared library creation failed:\n{}\n{}",
            String::from_utf8_lossy(&link_output.stdout),
            String::from_utf8_lossy(&link_output.stderr)
        );
    }

    // Link the shared library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=libtorch_wrapper");

    // Set runtime path for finding the library
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir.display());
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}/lib",
        libtorch_path.display()
    );

    // Tell Cargo to rerun if the wrapper files change
    println!("cargo:rerun-if-changed=src/ffi/libtorch_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/ffi/libtorch_wrapper.h");

    // Set LD_LIBRARY_PATH for runtime
    println!(
        "cargo:rustc-env=LD_LIBRARY_PATH={}/lib",
        libtorch_path.display()
    );
}
