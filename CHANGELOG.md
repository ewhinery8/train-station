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
