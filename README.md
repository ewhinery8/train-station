# Train Station

[![CI](https://github.com/ewhinery8/train-station/actions/workflows/ci.yml/badge.svg)](https://github.com/ewhinery8/train-station/actions/workflows/ci.yml)
[![Linux](https://github.com/ewhinery8/train-station/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/ewhinery8/train-station/actions/workflows/ci-linux.yml)
[![Windows](https://github.com/ewhinery8/train-station/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/ewhinery8/train-station/actions/workflows/ci-windows.yml)
[![macOS](https://github.com/ewhinery8/train-station/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/ewhinery8/train-station/actions/workflows/ci-macos.yml)
[![Release](https://github.com/ewhinery8/train-station/actions/workflows/release.yml/badge.svg)](https://github.com/ewhinery8/train-station/actions/workflows/release.yml)
[![Architecture](https://img.shields.io/badge/arch-x86__64%20%7C%20ARM64-green)](https://github.com/ewhinery8/train-station#platform-support)
[![Cross-Compile](https://github.com/ewhinery8/train-station/actions/workflows/cross-compile.yml/badge.svg)](https://github.com/ewhinery8/train-station/actions/workflows/cross-compile.yml)
[![Crates.io](https://img.shields.io/crates/v/train-station.svg)](https://crates.io/crates/train-station)
[![Documentation](https://docs.rs/train-station/badge.svg)](https://docs.rs/train-station)
[![License](https://img.shields.io/crates/l/train-station.svg)](https://github.com/ewhinery8/train-station#license)
[![Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://blog.rust-lang.org/2023/06/01/Rust-1.70.0.html)

> A zero-dependency, PyTorch inspired maximum performance Rust machine learning library.

## Table of Contents

- [What's Train Station?](#whats-train-station)
- [Key Features](#key-features)
- [Performance Results](#performance-results)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [What's in the Box](#whats-in-the-box-will-be-expanding-this-rapidly)
- [Platform Support](#platform-support)
- [Installation](#installation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## What's Train Station?

Train Station is a from-scratch ML framework designed for lightweight production workflows (think embedded applications or edge deployments). Built with zero external dependencies and a focus on zero-cost abstractions, it provides PyTorch-like ergonomics while maintaining complete control over performance. Train Station aims to encourage first-principles thinking and ground-up research. It provides the tooling to construct industry standard architectures and layers as well as completely novel architectures and techniques.  

Due to the zero dependency nature, building across various platforms and distrubution should be a breeze. The days of having to move around various dynamically linked objects or dealing with FFI boundary support and error handling are gone with this library.

With this said, the project is in its infancy, and we plan on rapidly expanding operation support, functionality, and eventually adding CUDA support (also, please report any bugs discovered so that we can patch things up as quickly as possible). Feel free to help contribute as we expand functionality and support!

Why "Train Station"? 

Simplicity, familiarity, performance, ease of use.

## Key Features

### Performance First
- **Zero dependencies** - Just Rust and raw performance
- **SIMD everywhere** - AVX2 acceleration for all major operations
- **Cache-friendly** - Memory layouts optimized for modern CPUs
- **Proven fast** - Benchmarked against LibTorch with impressive results
- **Iterator Itegration** - Our Tensor ties into the vast Iterator ecosystem while maintaining gradtrack auto differentiation. 

### Readiness
- **Thread-safe** - Safe concurrent execution by design
- **LibTorch-tested** - Every operation validated against LibTorch
- **Ergonomic API** - Familiar patterns

## Performance Results

Base performance seems solid - with signifigant speedups compared to LibTorch (C++ PyTorch backend) on certain ops. With this said, it will be hard to match performance for all operations considering our zero-dependency design (talking BLAS, MKL which larger behemoth libraries leverage). Our aim is to out-perform the competition. As the project matures we will continue to persue increasing levels of performance. 

### Performance Benchmarks

#### Addition Operation (CPU)
![Addition Speedup Performance](https://raw.githubusercontent.com/ewhinery8/train-station/master/benches/ts_vs_libtorch_speedup_add.png)
![Addition Time Performance](https://raw.githubusercontent.com/ewhinery8/train-station/master/benches/ts_vs_libtorch_timing_add.png)

Train Station's addition operation showcases performance across tensor sizes with particularly impressive results on small to medium tensors where our zero-overhead approach shines. We are 2-3X faster on average across the benchmarks run. 

#### Subtraction Operation (CPU)
![Subtraction Speedup Performance](https://raw.githubusercontent.com/ewhinery8/train-station/master/benches/ts_vs_libtorch_speedup_sub.png)
![Subtraction Time Performance](https://raw.githubusercontent.com/ewhinery8/train-station/master/benches/ts_vs_libtorch_timing_sub.png)

Train Station's subtraction operation showcases performance across tensor sizes with particularly impressive results on small to medium tensors where our zero-overhead approach shines. We are 3-4X faster on average across the benchmarks run. 

#### Multiplication Operation (CPU)
![Multiplication Speedup Performance](https://raw.githubusercontent.com/ewhinery8/train-station/master/benches/ts_vs_libtorch_speedup_mul.png)
![Multiplication Time Performance](https://raw.githubusercontent.com/ewhinery8/train-station/master/benches/ts_vs_libtorch_timing_mul.png)

Train Station's multiplication operation showcases performance across tensor sizes with particularly impressive results on small to medium tensors where our zero-overhead approach shines. We are 2-3X faster on average across the benchmarks run. 

#### Division Operation (CPU)
![Division Speedup Performance](https://raw.githubusercontent.com/ewhinery8/train-station/master/benches/ts_vs_libtorch_speedup_div.png)
![Division Time Performance](https://raw.githubusercontent.com/ewhinery8/train-station/master/benches/ts_vs_libtorch_timing_div.png)

Train Station's division operation showcases performance across tensor sizes with particularly impressive results on small to medium tensors where our zero-overhead approach shines. We are 3-4X faster on average across the benchmarks run. 

#### Matrix Multiplication (CPU)
![Matrix Multiplication Speedup Performance](https://raw.githubusercontent.com/ewhinery8/train-station/master/benches/ts_vs_libtorch_speedup_matmul.png)
![Matrix Multiplication Time Performance](https://raw.githubusercontent.com/ewhinery8/train-station/master/benches/ts_vs_libtorch_timing_matmul.png)

Train Station's matmul operation shows increased performance at the very bottom end, but end up being slower on average than LibTorch (hard to compete with LibTorch's handcrafted matmul kernels and decades of MKL and BLAS library refinement). We are exploring ways to get this benchmark up without adding async operations. 

## Quick Start

```rust
use train_station::{Tensor, Device, Adam};

// Create tensors - familiar API for PyTorch users
let x = Tensor::randn(vec![32, 784], None);
let w = Tensor::randn(vec![784, 128], None).with_requires_grad();
let b = Tensor::zeros(vec![128]).with_requires_grad();

// Forward pass - it's that simple
let y = x.matmul(&w).add_tensor(&b).relu();

// Automatic differentiation - because we've got your backward
let loss = y.sum();
loss.backward(None);

// Optimize - Adam optimizer with all the bells and whistles
let mut optimizer = Adam::new();
optimizer.add_parameters(&[&w, &b]);
optimizer.step(&mut [&mut w, &mut b]);
```

## Architecture

Train Station is built on three core principles:

1. **Performance**: Every line of code is written with speed in mind. Unsafe Rust? Yes, but only where benchmarks justify it, and always validated for correctness.

2. **Simplicity**: No unnecessary abstractions. KISS approach applied. We try to avoid rediculously complex and unmaintainable semantics. 

3. **Correctness**: Extensive validation against LibTorch ensures mathematical equivalence. We're fast, but we also want to be accurate.

## What's in the Box? 

<mark>(This will expand rapidly in future releases)<mark/>

### Core Operations **(see docs for full list of ops)**
- **Arithmetic**: `add`, `sub`, `mul`, `div` with full broadcasting
- **Activations**: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `softmax`
- **Matrix ops**: `matmul` with optimized kernels
- **Reductions**: `sum`, `mean`, `min`, `max`, `std`, `var`, `norm`
- **Transformations**: `reshape`, `permute`, `transpose`, `squeeze`, `unsqueeze`

### Automatic Differentiation
- Dynamic computation graphs
- Gradient accumulation
- Mixed precision ready
- Memory efficient backward passes

### Optimizers
- Adam with AMSGrad variant
- Full state serialization
- Learning rate scheduling ready

### Device Management
- Transparent CPU/GPU operations
- Thread-safe context switching
- Zero-overhead abstractions

## Platform Support

Train Station is designed for maximum cross-platform compatibility with comprehensive testing across multiple operating systems and architectures.

### Supported Platforms

#### Operating Systems
- **Linux** (Ubuntu 22.04+, other distributions)
- **Windows** (Windows 10+, via Git Bash or WSL2)
- **macOS** (macOS 12+, Intel and Apple Silicon)

#### Architectures
- **x86_64** - Full native support with AVX2 SIMD optimizations
- **ARM64** - Cross-compilation support for:
  - **Apple Silicon** (M1/M2/M3 Macs)
  - **Linux ARM64** (Jetson, mobile, embedded)
- **ARMv7** - Cross-compilation support for Raspberry Pi and embedded systems
- **WebAssembly** - Future web and WASI support via `wasm32-unknown-unknown`

#### Supported Cross-Compilation Targets
- `x86_64-pc-windows-gnu` - Windows GNU (MinGW)
- `x86_64-pc-windows-msvc` - Windows MSVC (Native)
- `aarch64-apple-darwin` - Apple Silicon (M1/M2/M3)
- `x86_64-apple-darwin` - Intel Mac
- `aarch64-unknown-linux-gnu` - Linux ARM64 (Jetson/Mobile/Embedded)
- `armv7-unknown-linux-gnueabihf` - Linux ARMv7 (Raspberry Pi)
- `wasm32-unknown-unknown` - WebAssembly (Browser/WASI)
- `x86_64-unknown-linux-gnu` - Linux x86_64 (Server/Desktop)
- `x86_64-unknown-linux-musl` - Linux x86_64 (Static/Alpine)

### Cross-Platform Features

#### Zero Dependencies
- **No external libraries** - Pure Rust implementation
- **No system dependencies** - Works out of the box
- **No FFI overhead** - Direct memory management

#### SIMD Optimizations
- **x86_64 AVX2** - Vectorized operations for maximum performance
- **Scalar fallbacks** - Automatic fallback for non-SIMD hardware
- **Runtime detection** - Optimal code path selection

#### Build System
- **Standard Cargo** - No special build requirements
- **Cross-compilation** - Build for any target from any host
- **Feature flags** - Optional CUDA support via `--features cuda`

### Testing Coverage

Our CI system provides comprehensive cross-platform validation:

- **Platform-specific workflows** - Dedicated testing for each OS
- **Cross-compilation testing** - Automated validation across 9 target architectures
- **Architecture validation** - ARM64, x86_64, and WebAssembly support
- **Tool compatibility** - BSD vs GNU tool differences handled
- **Performance benchmarking** - LibTorch comparison across platforms
- **Zero-dependency validation** - Ensures no external dependencies across all targets

### Performance Characteristics

#### Platform-Specific Optimizations
- **Linux** - GNU coreutils, optimized for server deployments
- **Windows** - Git Bash compatibility, WSL2 support
- **macOS** - BSD tools, native Apple Silicon performance

#### Cross-Compilation Performance
- **ARM64 Linux** - Validated compilation for embedded/mobile
- **Apple Silicon** - Native performance on M-series processors
- **Jetson/Edge** - Optimized for edge AI deployments

## Installation

### Platform Support Statement

Train Station is **fully tested and supported** across all major platforms with comprehensive CI/CD validation:

- ✅ **Linux** (Ubuntu 22.04+, other distributions) - Baseline reference platform
- ✅ **Windows** (Windows 10+, via Git Bash or WSL2) - Full compatibility testing
- ✅ **macOS** (macOS 12+, Intel and Apple Silicon) - Native ARM64 support

**Architecture Compatibility**:
- ✅ **x86_64** - Full native support with AVX2 SIMD optimizations
- ✅ **ARM64** - Cross-compilation support for Apple Silicon, Linux ARM64, and embedded systems
- ✅ **ARMv7** - Cross-compilation support for Raspberry Pi and embedded devices
- ✅ **WebAssembly** - Future web and WASI support via `wasm32-unknown-unknown`

**Quality Assurance**: Every release is validated across all supported platforms through our multi-platform CI/CD pipeline.

 <mark>**WARNING** - 
**CUDA FEATURE UNSTABLE, ACTIVATION NOT RECCOMMENDED AT THIS TIME**<mark/>      

### Basic Installation

Train Station works out-of-the-box on all supported platforms with zero external dependencies:

```toml
[dependencies]
train-station = "0.1"

# Optional features (experimental)
train-station = { version = "0.1", features = ["cuda"] }  # CUDA support incomplete
```

**Installation Verification**:
```bash
# Verify installation works on your platform
cargo add train-station
cargo check  # Should compile successfully on all supported platforms
```

### Platform-Specific Installation

#### Linux (Recommended - Baseline Reference)

Linux provides the most straightforward installation experience:

```bash
# Standard installation - zero additional dependencies
cargo add train-station

# Verify installation
cargo check

# Optional: Development setup with build tools
sudo apt update && sudo apt install build-essential  # Ubuntu/Debian
```

**Linux Advantages**:
- ✅ **Fastest CI execution** - Primary testing platform
- ✅ **LibTorch validation available** - Full mathematical validation support
- ✅ **Native GNU tools** - Optimal script compatibility
- ✅ **Zero setup complexity** - Works immediately after Rust installation

#### Windows (WSL2 Recommended)

**Option A: WSL2 (Strongly Recommended)**
```bash
# Install WSL2 with Ubuntu (provides full Linux compatibility)
wsl --install -d Ubuntu

# Inside WSL2, follow Linux installation above
cargo add train-station
cargo check  # Full Linux compatibility
```

**Option B: Native Windows (Git Bash)**
```bash
# Requires Git for Windows (includes Git Bash)
# Download from: https://git-scm.com/download/win

# In Git Bash or PowerShell:
cargo add train-station
cargo check  # Validated through Windows CI
```

**Windows Platform Notes**:
- ✅ **Full CI validation** - Every release tested on Windows latest
- ✅ **Git Bash compatibility** - All development tools work in Git Bash
- ⚠️ **LibTorch limitations** - Use WSL2 for LibTorch validation if needed
- ✅ **Cross-compilation support** - Can cross-compile for other targets

#### macOS (Full Native Support)

macOS provides excellent native support with Apple Silicon optimization:

```bash
# Works on both Intel and Apple Silicon Macs
cargo add train-station
cargo check  # Native performance on both architectures

# Optional: Install Xcode command line tools for development
xcode-select --install
```

**macOS Platform Notes**:
- ✅ **Apple Silicon native** - Full ARM64 optimization on M1/M2/M3 Macs
- ✅ **Intel Mac support** - Complete x86_64 compatibility
- ✅ **BSD tool compatibility** - All scripts adapted for macOS differences
- ✅ **DYLD_LIBRARY_PATH support** - LibTorch validation available

### Cross-Compilation Support

Train Station supports comprehensive cross-compilation across 9 target architectures:

#### Supported Cross-Compilation Targets
```bash
# Install cross-compilation targets
rustup target add aarch64-unknown-linux-gnu    # Linux ARM64 (Jetson/Mobile/Embedded)
rustup target add aarch64-apple-darwin         # Apple Silicon (M1/M2/M3)
rustup target add armv7-unknown-linux-gnueabihf # Linux ARMv7 (Raspberry Pi)
rustup target add x86_64-pc-windows-gnu        # Windows GNU (MinGW)
rustup target add x86_64-pc-windows-msvc       # Windows MSVC (Native)
rustup target add x86_64-unknown-linux-musl    # Linux x86_64 (Static/Alpine)
rustup target add wasm32-unknown-unknown       # WebAssembly (Browser/WASI)
```

#### Cross-Compilation Examples
```bash
# Cross-compile for ARM64 Linux (Jetson, mobile, embedded)
cargo build --target aarch64-unknown-linux-gnu --lib -p train-station

# Cross-compile for Apple Silicon
cargo build --target aarch64-apple-darwin --lib -p train-station

# Cross-compile for Raspberry Pi
cargo build --target armv7-unknown-linux-gnueabihf --lib -p train-station

# Cross-compile for WebAssembly
cargo build --target wasm32-unknown-unknown --lib -p train-station
```

**Cross-Compilation Validation**: All targets are automatically validated through our cross-compilation CI workflow.

### Feature Flag Compatibility

Train Station's feature flags are designed for cross-platform compatibility:

#### Available Features
```toml
[dependencies]
# Default: Zero dependencies, maximum compatibility
train-station = "0.1"

# CUDA support (experimental, Linux/WSL2 recommended)
train-station = { version = "0.1", features = ["cuda"] }

# All features (for development)
train-station = { version = "0.1", features = ["cuda"] }
```

#### Platform-Specific Feature Support
- **Default features**: ✅ All platforms (Linux, Windows, macOS, all architectures)
- **CUDA features**: ⚠️ Experimental (Linux/WSL2 recommended, limited Windows/macOS support)

#### Feature Flag Validation
```bash
# Test default features (should work on all platforms)
cargo check --lib -p train-station

# Test with CUDA features (platform-dependent)
cargo check --lib -p train-station --features cuda

# Test all features
cargo check --lib -p train-station --all-features
```

### Development Setup

For contributors and advanced users:

```bash
# Clone the repository
git clone https://github.com/ewhinery8/train-station.git
cd train-station

# Platform-specific testing
cargo test -p train-station --lib  # Core library tests (all platforms)

# Cross-compilation validation (run on any platform)
cargo check --target aarch64-unknown-linux-gnu --lib -p train-station
cargo check --target aarch64-apple-darwin --lib -p train-station

# Platform-specific development setup
# Linux/WSL2: Ready to go
# Windows: Use WSL2 or Git Bash
# macOS: Install Xcode command line tools if needed
```

### Installation Troubleshooting

#### Common Installation Issues

**Rust Not Found**:
```bash
# Install Rust via rustup (all platforms)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env  # Linux/macOS/WSL2
# Restart terminal on Windows
```

**Compilation Errors**:
```bash
# Linux: Install build tools
sudo apt install build-essential

# Windows: Use WSL2 or ensure Git Bash is installed
# macOS: Install Xcode command line tools
xcode-select --install
```

**Cross-Compilation Issues**:
```bash
# Install missing targets
rustup target add <target-name>

# Linux: Install cross-compilation tools for ARM
sudo apt install gcc-aarch64-linux-gnu gcc-arm-linux-gnueabihf
```

### Platform Verification

Verify Train Station works correctly on your platform:

```bash
# Basic functionality test
cargo add train-station
echo 'fn main() { println!("Train Station installed successfully!"); }' > test.rs
cargo run --bin test  # Should compile and run on all platforms

# Advanced verification
cargo test -p train-station --lib  # Run core tests
cargo check --all-targets --all-features  # Verify all features compile
```

## Roadmap

### Near Term
- Rapid expansion of operation support
- Focus on building block functionality
- Mature the functionality and listen to user feedback

### On the Horizon  
- More complex objects (various layer types, more complex operations and loss functions)
- CUDA support

## Contributing

We welcome contributions that improve performance, add features, or enhance documentation! Train Station has a comprehensive development infrastructure to make contributing smooth and efficient.

### Quick Start for Contributors

1. **Follow our conventions**: We use [conventional commits](https://www.conventionalcommits.org/) and structured branch naming
2. **Use our helper scripts**: Automated tools for branch creation, changelog management, and releases
3. **Comprehensive testing**: Every change is validated against LibTorch for mathematical correctness
4. **Zero dependencies**: Keep the core library dependency-free

### Development Resources

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Complete contribution guidelines with developer tools
- **[CHANGELOG.md](CHANGELOG.md)** - Project changelog following Keep a Changelog format
- **Helper Scripts** - Located in `scripts/` directory for streamlined development
- **Automated CI/CD** - Comprehensive testing and release automation
- **Cross-Platform Testing** - Platform-specific CI workflows for Linux, Windows, and macOS

### Before Submitting PRs

- Run benchmarks to ensure no performance regressions
- Add tests for new functionality  
- Validate against LibTorch for correctness
- Follow our branch naming and commit message conventions
- **Test cross-platform compatibility** - Ensure changes work across all supported platforms

### Cross-Platform Development

Our comprehensive CI system automatically tests all contributions across:

- **Linux (Ubuntu 22.04)** - GNU tools, server-optimized testing
- **Windows (latest)** - Git Bash environment, Windows-specific validation  
- **macOS (latest)** - BSD tools, Apple Silicon and Intel testing
- **ARM64 Cross-compilation** - Validation for embedded and mobile targets

Each platform has dedicated CI workflows that you can monitor via the platform-specific badges above.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines, development setup, and available tools.

## License

Licensed under either:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Acknowledgments

Train Station's design is inspired by PyTorch's excellent API. We stand on the shoulders of giants - PyTorch pioneered the patterns that make deep learning accessible. We're just making them faster in Rust.

Special thanks to the Rust ML community for showing us what's possible.

---

*Built for speed. Validated for correctness. Build, break, fix, innovate, repeat. Keep moving forward.*