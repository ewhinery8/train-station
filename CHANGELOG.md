# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 

### Changed
- 

### Fixed
- 

### Performance
- 

### Security
- 

## [0.1.3] - 2025-08-28

### Fixed
- Skip build.rs compilation for docs.rs environment to improve documentation generation

## [0.1.2] - 2025-08-28

### Changed
- Added publishing metadata to Cargo.toml for better crates.io presentation

## [0.1.1] - 2025-08-28

### Fixed
- Updated PNG image hosting in README to use GitHub for better accessibility

## [0.1.0] - 2025-08-28

### Added
- Initial public release of train-station crate
- Zero-dependency tensor operations with SIMD optimizations
- Complete autograd system with gradient tracking and propagation
- LibTorch validation framework for mathematical correctness (validation crate only)
- CUDA acceleration support via FFI (when cuda feature enabled)
- Comprehensive serialization framework supporting binary and JSON formats
- Adam optimizer implementation with full serialization support
- Memory pool system with thread-safe global allocator
- Device management system with CPU/CUDA context switching
- Performance benchmarking system with LibTorch comparison capabilities
- Interactive visualization scripts for performance analysis
- Comprehensive examples and documentation
- MIT and Apache-2.0 dual licensing
- Contributing guidelines and code ownership documentation

### Performance
- SIMD-optimized operations using AVX2 for maximum CPU performance
- Cache-friendly memory layouts optimized for modern CPU architectures
- Zero-copy tensor views and transformations where mathematically possible
- Blocked matrix multiplication algorithms for improved cache utilization
- Memory-aligned tensor data for optimal SIMD instruction usage
