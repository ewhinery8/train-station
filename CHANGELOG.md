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

### Documentation
- 

### Maintenance
- 

### Security
- 

## [0.2.0] - 2025-09-21

### Added
- add view system, memory pool, thread-safe autograd, and broadcasting
  - Zero-copy views (reshape, transpose, slice, as_strided)
  - SIMD-aligned TensorMemoryPool with thread-local optimization
  - Thread-safe GradTrack with local/shared computation graphs
  - NumPy-compatible broadcasting for element-wise and batched ops
  - Enhanced matmul/add with SIMD acceleration
  - Add retain_grad for non-leaf tensor gradients
  - Improve function signatures for better Rust idioms
  - Enhanced Iterator system with idiomatic Rust patterns
  - Add iter_values for efficient element iteration
  - Add collect_shape for seamless tensor reconstruction from iterators
  - Add TensorCollectExt trait for iterator-based tensor construction

### Fixed
- No bug fixes

### Performance
- No performance improvements

### Documentation
- No documentation changes

### Maintenance
- No maintenance changes

### Other Changes
- No other changes

## [0.1.6] - 2025-09-01

### Added
- No new features

### Fixed
- No bug fixes

### Performance
- No performance improvements

### Documentation
- No documentation changes

### Maintenance
- add track_caller to all public API fns for improved debugging

### Other Changes
- No other changes

## [0.1.5] - 2025-08-30

### Added
- No new features

### Fixed
- No bug fixes

### Performance
- No performance improvements

### Documentation
- No documentation changes

### Maintenance
- No maintenance changes

### Other Changes
- refactor: removing build process from release workflow

## [0.1.4] - 2025-08-30

### Added
- enhance cross-platform build system with comprehensive CI/CD pipeline
  feat: enhance cross-platform build system with comprehensive CI/CD pipeline
  - Add sophisticated build scripts with static linking for train-station CUDA stubs
  - Support Windows (MSVC/MinGW), macOS (Clang/GCC), Linux (GCC/Clang) toolchains
  - Add graceful degradation for missing compilers and archiver tools
  - Fix cross-compilation support using TARGET environment variable
  - Resolve all clippy warnings
  - Update CI workflows
  - Establish target matching convention (runtime target = compilation target)
  - Implement zero runtime dependencies through static linking strategy

### Fixed
- No bug fixes

### Performance
- No performance improvements

### Documentation
- No documentation changes

### Maintenance
- No maintenance changes

### Other Changes
- No other changes

## [0.1.3] - 2025-08-28

### Fixed
- skipping build.rs for docs.rs env

## [0.1.2] - 2025-08-28

### Maintenance
- adding publishing metadata to toml

## [0.1.1] - 2025-08-28

### Documentation
- updated png hosting in README to github

## [0.1.0] - 2025-08-28

### Added
- initial commit and release v0.1.0 - ship it!
