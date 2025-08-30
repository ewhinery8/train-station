# Contributing to Train Station

Thank you for considering contributing to Train Station! We welcome contributions that improve performance, add features, enhance documentation, or fix bugs.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Branch Naming Convention](#branch-naming-convention)
- [Commit Message Convention](#commit-message-convention)
- [Development Workflow](#development-workflow)
- [Development Setup](#development-setup)
- [Developer Tools and Scripts](#developer-tools-and-scripts)
- [Contributing Guidelines](#contributing-guidelines)
- [Performance Requirements](#performance-requirements)
- [Testing Standards](#testing-standards)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Code Style](#code-style)
- [Areas for Contribution](#areas-for-contribution)
- [Release Process](#release-process)
- [CI/CD Workflow](#cicd-workflow)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/train-station.git`
3. Add upstream remote: `git remote add upstream https://github.com/originalowner/train-station.git`
4. Create a feature branch following our naming convention (see below)

## Branch Naming Convention

We follow a structured branch naming convention to improve code organization and review processes.

### Format
```
<type>/<short-description>
```

### Branch Types
- **`feat/`** - New features (e.g., `feat/matrix-multiplication`)
- **`fix/`** - Bug fixes (e.g., `fix/memory-leak-tensors`)
- **`perf/`** - Performance improvements (e.g., `perf/simd-optimization`)
- **`docs/`** - Documentation updates (e.g., `docs/api-reference`)
- **`refactor/`** - Code refactoring (e.g., `refactor/tensor-core`)
- **`test/`** - Adding/improving tests (e.g., `test/cuda-validation`)
- **`chore/`** - Maintenance tasks (e.g., `chore/update-dependencies`)

### Guidelines
- Use lowercase with hyphens
- Be descriptive but concise
- Match branch type with commit type (e.g., `feat/` branches use `feat:` commits)

## Commit Message Convention

This project follows [Conventional Commits](https://www.conventionalcommits.org/) to enable automated changelog generation and semantic versioning.

### Format
```
<type>: <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New features or enhancements
- `fix`: Bug fixes
- `docs`: Documentation changes
- `perf`: Performance improvements (include metrics when possible)
- `refactor`: Code refactoring without functional changes
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates

### Examples
```
feat: add matrix multiplication with SIMD optimization
fix: resolve memory leak in tensor allocation
perf: optimize broadcasting operations by 40%
docs: update API documentation for autograd system
refactor: simplify gradient computation logic
test: add comprehensive validation for CUDA operations
chore: bump version to 0.1.4
```

### Performance Commits
For performance improvements, include measurable impact:
```
perf: optimize SIMD operations for 50% speed improvement
perf: reduce memory allocation overhead by 30%
perf: implement cache-friendly matrix multiplication
```

### Commit Message Setup
Configure Git to use our commit message template:
```bash
git config commit.template .gitmessage
```

## Development Workflow

### 1. Create a Feature Branch
```bash
# Ensure you're on master and up to date
git checkout master
git pull upstream master

# Create feature branch following naming convention
git checkout -b feat/tensor-broadcasting
```

### 2. Make Changes and Commit
```bash
# Make your changes
git add .
git commit -m "feat: implement tensor broadcasting for arbitrary shapes"

# Push to your fork
git push origin feat/tensor-broadcasting
```

### 3. Create Pull Request
- Go to GitHub and create a PR from your branch to `master`
- Fill out the PR template with clear description
- Ensure all CI checks pass
- Request review from maintainers

## Development Setup

### Prerequisites

- Rust 1.70+ (stable toolchain)
- For CUDA development: CUDA Toolkit 11.0+
- For validation testing: LibTorch (see libtorch-validation/README.md)

### Platform-Specific Setup

#### Linux (Recommended - Baseline Reference Platform)

Linux serves as our baseline reference platform with the most straightforward setup:

```bash
# Ubuntu/Debian - Install build essentials
sudo apt update
sudo apt install build-essential curl git

# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install required components
rustup component add rustfmt clippy

# Clone and test
git clone https://github.com/yourusername/train-station.git
cd train-station
cargo test -p train-station --lib
```

**Linux Advantages**:
- ✅ Native GNU tools (sha256sum, find -delete)
- ✅ Standard LD_LIBRARY_PATH for LibTorch
- ✅ Fastest CI execution
- ✅ Most compatible with helper scripts

#### Windows (WSL2 Recommended)

**Option A: WSL2 (Strongly Recommended)**
```bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu

# Inside WSL2, follow Linux setup above
# WSL2 provides full Linux compatibility
```

**Option B: Git Bash (Alternative)**
```bash
# Install Git for Windows (includes Git Bash)
# Download from: https://git-scm.com/download/win

# In Git Bash:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Restart Git Bash to reload PATH
rustup component add rustfmt clippy
```

**Windows Platform Notes**:
- ✅ WSL2 provides full Linux compatibility
- ⚠️ Git Bash has some tool limitations (no sha256sum fallback to file size)
- ⚠️ PowerShell/CMD not recommended for development
- ✅ All CI workflows test Windows compatibility

#### macOS (BSD Tool Considerations)

```bash
# Install Xcode command line tools
xcode-select --install

# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install required components
rustup component add rustfmt clippy

# Clone and test
git clone https://github.com/yourusername/train-station.git
cd train-station
cargo test -p train-station --lib
```

**macOS Platform Notes**:
- ✅ Native Apple Silicon (ARM64) support
- ⚠️ Uses BSD tools (shasum vs sha256sum, different find behavior)
- ✅ DYLD_LIBRARY_PATH for LibTorch (instead of LD_LIBRARY_PATH)
- ✅ All scripts adapted for BSD compatibility

### Basic Setup (All Platforms)

After platform-specific setup above:

```bash
# Clone the repository
git clone https://github.com/yourusername/train-station.git
cd train-station

# Run core tests
cargo test -p train-station --lib

# Run formatting and linting
cargo fmt --all
cargo clippy --all-targets

# Build release binaries
cargo build --all-targets --release
```

### LibTorch Validation Setup (Optional)

For mathematical validation and performance benchmarking:

#### Linux/WSL2
```bash
# Download LibTorch and extract to libtorch-validation/libtorch/
# Then set library path
export LD_LIBRARY_PATH="./libtorch-validation/libtorch/lib:$LD_LIBRARY_PATH"

# Run validation tests
cargo test -p libtorch-validation
```

#### macOS
```bash
# Download LibTorch and extract to libtorch-validation/libtorch/
# Then set library path (macOS uses DYLD_LIBRARY_PATH)
export DYLD_LIBRARY_PATH="./libtorch-validation/libtorch/lib:$DYLD_LIBRARY_PATH"

# Run validation tests
cargo test -p libtorch-validation
```

#### Windows (Git Bash)
```bash
# LibTorch validation typically not available on Windows
# Use WSL2 for LibTorch validation if needed
```

## Developer Tools and Scripts

### Supported Development Environment

**Multi-Platform Support**: Train Station supports development on all major platforms with platform-specific optimizations.

#### Platform Recommendations

**Linux (Baseline Reference)**:
- ✅ **Recommended for**: Primary development, fastest CI, LibTorch validation
- ✅ **Advantages**: Native GNU tools, fastest execution, most script compatibility
- ✅ **Best for**: Contributors doing heavy development work

**Windows (WSL2 Strongly Recommended)**:
- ✅ **WSL2**: Full Linux compatibility, recommended for all Windows development
- ⚠️ **Git Bash**: Alternative option with some limitations (no LibTorch validation)
- ❌ **PowerShell/CMD**: Not recommended for development

**macOS (Full Native Support)**:
- ✅ **Recommended for**: Apple Silicon development, native ARM64 testing
- ✅ **Advantages**: Native Apple Silicon support, BSD tool compatibility
- ⚠️ **Considerations**: Uses BSD tools (shasum, different find behavior)

#### Development Environment Features

**Cross-Platform Script Compatibility**:
- All scripts designed for bash shell environments
- Automatic tool detection (sha256sum vs shasum)
- Platform-specific fallbacks and adaptations
- Consistent behavior across all supported platforms

**Platform-Specific Optimizations**:
- **Linux**: GNU tools, LD_LIBRARY_PATH, fastest CI execution
- **Windows**: Git Bash compatibility, Windows path handling
- **macOS**: BSD tool support, DYLD_LIBRARY_PATH, Apple Silicon native

### Available Scripts

All helper scripts are located in the `scripts/` directory:

#### `scripts/create-branch.sh` - Branch Creation Helper
**Purpose**: Creates feature branches following project naming conventions

```bash
# Usage
./scripts/create-branch.sh <type> <description>

# Examples
./scripts/create-branch.sh feat tensor-broadcasting
./scripts/create-branch.sh fix memory-leak-in-autograd
./scripts/create-branch.sh perf simd-optimization
```

**What it does**:
- Validates branch type and description format
- Ensures clean working directory and master branch
- Checks for existing branches (local and remote)
- Updates master branch and creates new feature branch
- Provides next steps and commit message guidance

#### `scripts/add-changelog-entry.sh` - Changelog Management
**Purpose**: Adds entries to CHANGELOG.md in proper format

```bash
# Usage
./scripts/add-changelog-entry.sh <type> <description>

# Examples
./scripts/add-changelog-entry.sh added "tensor broadcasting support"
./scripts/add-changelog-entry.sh fixed "memory leak in gradient computation"
./scripts/add-changelog-entry.sh performance "SIMD optimization for operations"
```

**What it does**:
- Validates entry type (added, changed, fixed, performance, security)
- Locates correct section in Unreleased area
- Intelligently adds entries (replaces placeholders or inserts)
- Maintains Keep a Changelog format
- Shows commit commands for documentation updates

#### `scripts/prepare-release.sh` - Release Preparation
**Purpose**: Comprehensive release preparation and validation

```bash
# Usage (maintainers only)
./scripts/prepare-release.sh <version>

# Example
./scripts/prepare-release.sh 0.1.4
```

**What it does**:
- Validates environment (clean working directory, master branch, up-to-date)
- Updates version in Cargo.toml
- Generates changelog entries from conventional commits
- Updates CHANGELOG.md with new version section
- Runs comprehensive tests (core tests, release build)
- Provides exact commands for committing and tagging

### Development Workflow with Scripts

#### Complete Feature Development Flow
```bash
# 1. Create feature branch
./scripts/create-branch.sh feat matrix-multiplication

# 2. Make your changes
# ... code changes ...

# 3. Add changelog entry (optional, for significant changes)
./scripts/add-changelog-entry.sh added "matrix multiplication with SIMD optimization"

# 4. Commit changes
git add .
git commit -m "feat: add matrix multiplication with SIMD optimization"

# 5. Push and create PR
git push -u origin feat/matrix-multiplication
# Create PR on GitHub using the PR template
```

#### Release Process (Maintainers)
```bash
# 1. Prepare release
./scripts/prepare-release.sh 0.1.4

# 2. Review changes
git diff

# 3. Commit release
git add Cargo.toml CHANGELOG.md
git commit -m "chore: release version 0.1.4"

# 4. Create and push tag
git tag v0.1.4
git push origin master --tags

# 5. GitHub Actions handles the rest automatically
```

### Additional Resources

#### Configuration Files
- `.gitmessage` - Commit message template (use: `git config commit.template .gitmessage`)
- `.github/pull_request_template.md` - PR template for consistent submissions
- `.github/BRANCH_NAMING.md` - Detailed branch naming convention guide

#### GitHub Workflows
- `.github/workflows/ci.yml` - Continuous Integration (runs on PRs and main pushes)
- `.github/workflows/release.yml` - Release automation (triggered by version tags)

#### Documentation Files
- `CHANGELOG.md` - Project changelog following Keep a Changelog format
- `CONTRIBUTING.md` - This file (complete contribution guidelines)
- `README.md` - Project overview with badges and quick start

### Script Requirements

All scripts require:
- **Bash shell** (version 4.0+)
- **Git** (for repository operations)
- **Standard Unix tools** (sed, grep, curl, date)
- **Project root directory** (scripts must be run from workspace root)

### Troubleshooting Scripts

#### Common Issues
- **Permission denied**: Run `chmod +x scripts/*.sh` to make scripts executable
- **Command not found**: Ensure you're in the project root directory
- **Git errors**: Ensure you have proper git configuration and remote access

#### Getting Help
```bash
# All scripts show usage help without arguments
./scripts/create-branch.sh
./scripts/add-changelog-entry.sh  
./scripts/prepare-release.sh
```

## Contributing Guidelines

### Core Principles

All contributions must align with Train Station's core principles:

1. **Performance First**: Every change must be justified by performance benefits or maintain existing performance
2. **Zero Dependencies**: No external dependencies in the core train-station crate
3. **Safety**: Unsafe code must be justified by benchmarks and validated for correctness
4. **Simplicity**: Prefer simple, direct implementations over complex abstractions

### Mandatory Requirements

Before submitting any PR:

- [ ] All tests pass: `cargo test -p train-station`
- [ ] Code follows project style guidelines
- [ ] New features include comprehensive tests
- [ ] Performance-critical code includes benchmarks
- [ ] Documentation is updated for user-facing changes
- [ ] LibTorch validation tests pass (for tensor operations)

## Performance Requirements

### Benchmarking

All performance-critical contributions must include benchmarks:

```rust
#[cfg(test)]
mod benches {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bench_new_operation() {
        let tensor = Tensor::randn(vec![1000, 1000], None);
        
        let start = Instant::now();
        for _ in 0..100 {
            let _ = tensor.your_new_operation();
        }
        let duration = start.elapsed();
        
        // Document expected performance characteristics
        println!("Operation took: {:?}", duration);
    }
}
```

### SIMD Requirements

For tensor operations:

- Implement AVX2 SIMD path where beneficial
- Provide scalar fallback for all SIMD code
- Use runtime feature detection: `is_x86_feature_detected!("avx2")`
- Follow existing patterns in `/tensor/ops/` files

### Memory Efficiency

- Minimize allocations in hot paths
- Use stack allocation for small, fixed-size data
- Leverage memory pools for temporary allocations
- Document memory complexity of new operations

## Testing Standards

### Unit Tests

All new functionality requires comprehensive unit tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Test normal cases
    }

    #[test]
    fn test_edge_cases() {
        // Test empty tensors, zero-sized dimensions, etc.
    }

    #[test]
    fn test_error_conditions() {
        // Test invalid inputs, panics, etc.
    }
}
```

### LibTorch Validation

For tensor operations, mathematical validation is required:

1. Add validation test in `libtorch-validation/src/validation/tensor/ops/your_op.rs`
2. Target 1e-6 absolute or relative error tolerance
3. Test multiple tensor shapes and edge cases
4. Validate gradients for autograd operations

### Test Organization

- Co-locate tests with functionality (no separate test directories)
- Use descriptive test names
- Group related tests in the same `mod tests` block
- Document test intent with comments

## Documentation

### Code Documentation

- Document all public APIs with rustdoc comments
- Include examples for complex operations
- Document safety requirements for unsafe code
- Explain performance characteristics where relevant

### Examples

For new features, provide examples in `/examples/`:

```rust
//! Demonstrates how to use the new feature
//! 
//! This example shows basic usage patterns and common use cases.

use train_station::Tensor;

fn main() {
    // Clear, working example code
    let tensor = Tensor::new(vec![2, 3]);
    println!("Created tensor: {:?}", tensor);
}
```

### Performance Documentation

Document performance characteristics:

- Time complexity (O(n), O(n²), etc.)
- Memory usage patterns
- SIMD acceleration availability
- Hardware requirements

## Pull Request Process

### Before Submitting

1. Rebase on latest master: `git rebase upstream/master`
2. Run full test suite: `cargo test`
3. Check formatting: `cargo fmt`
4. Run Clippy: `cargo clippy`
5. Update documentation as needed

### PR Description Template

```markdown
## Summary
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Performance improvement
- [ ] Documentation update

## Performance Impact
- [ ] No performance impact
- [ ] Performance improvement (include benchmarks)
- [ ] Performance regression (justify why acceptable)

## Testing
- [ ] Unit tests added/updated
- [ ] LibTorch validation added (for tensor ops)
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new dependencies added (unless absolutely necessary)
```

### Review Process

1. Automated checks must pass
2. Code review by maintainers
3. Performance review for critical paths
4. Final approval and merge

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug.

**Reproduction Steps**
1. Step one
2. Step two
3. Expected vs actual behavior

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Rust version: [e.g., 1.70.0]
- Train Station version: [e.g., 0.1.0]
- Features enabled: [e.g., cuda]

**Additional Context**
Any other relevant information.
```

### Feature Requests

For new features:

1. Check existing issues to avoid duplicates
2. Describe the use case and motivation
3. Propose API design if applicable
4. Consider performance implications
5. Volunteer to implement if possible

## Code Style

### Rust Style

- Follow `rustfmt` default configuration
- Use `cargo clippy` and fix all warnings
- Prefer explicit types in public APIs
- Use descriptive variable names

### Project Conventions

- File organization: One major functionality per file
- Error handling: Use `assert!` for programmer errors, `Result` for recoverable errors
- Safety: Document all unsafe code with safety invariants
- Performance: Profile before optimizing, benchmark all claims

### Commit Messages

Follow conventional commits:

```
type(scope): description

- feat: new feature
- fix: bug fix
- perf: performance improvement
- docs: documentation changes
- style: formatting changes
- refactor: code restructuring
- test: adding tests
- chore: maintenance tasks
```

Examples:
- `feat(tensor): add matrix multiplication operation`
- `perf(ops): optimize addition with AVX2 SIMD`
- `fix(autograd): correct gradient computation for broadcasting`

## Areas for Contribution

### High Priority

- **Neural Network Layers**: Linear, Conv2D, BatchNorm, LayerNorm
- **Loss Functions**: MSE, CrossEntropy, BinaryCrossEntropy
- **Optimizers**: SGD, AdamW, RMSprop
- **SIMD Optimizations**: ARM NEON, additional x86 operations

### Medium Priority

- **Data Loading**: Dataset traits, DataLoader utilities
- **Serialization**: Additional formats, streaming support
- **Documentation**: Tutorials, API guides, performance guides
- **Examples**: Real-world use cases, benchmark comparisons

### Low Priority

- **Platform Support**: Additional architectures, WASM support
- **Distributed Training**: Multi-GPU, multi-node support
- **Quantization**: Int8, Int16 tensor support
- **JIT Compilation**: Runtime optimization

### Getting Help

- Open an issue for questions
- Check existing documentation
- Look at similar implementations in the codebase
- Ask in discussions for design questions

## Release Process

### For Contributors
- Follow branch naming convention (`<type>/<description>`)
- Follow conventional commit messages (`<type>: <description>`)
- No need to update CHANGELOG.md manually (automated)
- No need to worry about versioning

### For Maintainers

#### Creating a Release
1. **Prepare the release**:
   ```bash
   ./scripts/prepare-release.sh 0.1.4
   ```

2. **Review and commit**:
   ```bash
   git add Cargo.toml CHANGELOG.md
   git commit -m "chore: release version 0.1.4"
   ```

3. **Create and push tag**:
   ```bash
   git tag v0.1.4
   git push origin master --tags
   ```

4. **Automated process handles**:
   - Runs comprehensive tests (blocks release if tests fail)
   - Creates GitHub release with auto-generated release notes
   - Updates CHANGELOG.md in repository
   - Handles missing releases for existing tags

#### Manual Changelog Generation (if needed)
```bash
# Preview unreleased changes
git log --oneline --pretty=format:"%s" $(git describe --tags --abbrev=0)..HEAD

# Update CHANGELOG.md manually if needed
# Follow Keep a Changelog format
```

## CI/CD Workflow

### Multi-Platform Continuous Integration

Our CI system provides comprehensive testing across all supported platforms:

#### CI Workflow Structure
```
Push/PR → Triggers Multiple CI Workflows
├── CI Summary (ubuntu-22.04) - Quick validation
├── CI-Linux (ubuntu-22.04) - Full Linux testing
├── CI-Windows (windows-latest) - Full Windows testing
├── CI-macOS (macos-latest) - Full macOS testing
└── Cross-Compilation - 9 target architectures
```

#### Platform-Specific CI Workflows

**Linux CI (ci-linux.yml)**
- **Platform**: Ubuntu 22.04 (GNU tools baseline)
- **Shell**: bash
- **Tools**: sha256sum, find -delete, LD_LIBRARY_PATH
- **Features**: Fastest execution, LibTorch validation available
- **ARM64**: Cross-compilation for aarch64-unknown-linux-gnu

**Windows CI (ci-windows.yml)**  
- **Platform**: Windows latest (Git Bash environment)
- **Shell**: bash (for consistency)
- **Tools**: sha256sum (via Git Bash), Windows-compatible paths
- **Features**: MSVC/MinGW compatibility testing
- **ARM64**: Cross-compilation for aarch64-apple-darwin

**macOS CI (ci-macos.yml)**
- **Platform**: macOS latest (BSD tools)
- **Shell**: bash
- **Tools**: shasum -a 256, BSD find, DYLD_LIBRARY_PATH
- **Features**: Apple Silicon native support, BSD compatibility
- **ARM64**: Native aarch64-apple-darwin testing

#### What Each CI Platform Tests
- **Code formatting** (`cargo fmt --all -- --check`)
- **Linting** (`cargo clippy --all-targets -- -D warnings`)
- **Compilation** (debug and release modes with --all-features)
- **Core tests** (`cargo test -p train-station --lib`)
- **Documentation** (`cargo doc` and `cargo test --doc`)
- **Security audit** (`cargo audit`)
- **Cross-compilation** (ARM64 targets)
- **LibTorch validation** (non-blocking, platform-dependent)

#### Cross-Platform Consistency
All platforms use:
- ✅ **Same shell**: `shell: bash` for consistency
- ✅ **Same Rust installation**: Manual rustup with error handling
- ✅ **Same validation steps**: Identical test commands
- ✅ **Platform-specific adaptations**: Tool detection and fallbacks

#### CI Requirements
- ✅ **All platforms must pass** - No platform-specific exceptions
- ✅ **Code must be formatted** - Consistent across all platforms
- ✅ **No clippy warnings** - Zero warnings policy
- ✅ **Documentation must build** - Cross-platform doc compatibility
- ✅ **Cross-compilation must work** - ARM64 targets validated

### Multi-Platform Release Workflow

Triggered by pushing version tags (`v*`):

#### Release Validation Matrix
```yaml
validate:
  strategy:
    matrix:
      os: [ubuntu-22.04, windows-latest, macos-latest]
```

**Enhanced Release Process**:
1. **Multi-platform pre-release validation** - ALL platforms must pass
2. **Comprehensive testing** - Full CI suite on each platform
3. **Cross-compilation validation** - ARM64 targets on all platforms
4. **Security auditing** - Dependency vulnerability scanning
5. **Create GitHub release** - Only after all platforms validate
6. **Missing release fix-up** - Historical release management

#### Release Platform Requirements
Each platform must pass:
- ✅ **Formatting validation** - `cargo fmt --all -- --check`
- ✅ **Linting validation** - `cargo clippy --all-targets -- -D warnings`
- ✅ **Compilation validation** - Debug and release modes
- ✅ **Test validation** - `cargo test -p train-station --lib`
- ✅ **Documentation validation** - Doc building and testing
- ✅ **Cross-compilation validation** - ARM64 target compilation
- ✅ **Security validation** - `cargo audit` vulnerability scanning

#### Release Notes Generation
- **Automatic categorization** from conventional commits
- **Added** - `feat:` commits
- **Fixed** - `fix:` commits  
- **Performance** - `perf:` commits
- **Documentation** - `docs:` commits
- **Other Changes** - remaining commits

### Troubleshooting CI

#### Common Issues (All Platforms)
- **Formatting failures**: Run `cargo fmt --all` locally
- **Clippy warnings**: Run `cargo clippy --all-targets -- -D warnings` locally
- **Test failures**: Run `cargo test -p train-station --lib` locally
- **Doc failures**: Run `cargo doc --no-deps --document-private-items` locally
- **Cross-compilation failures**: Install target with `rustup target add <target>`

#### Platform-Specific Troubleshooting

##### Linux-Specific Issues
```bash
# Missing build tools
sudo apt update
sudo apt install build-essential

# LibTorch validation path issues
export LD_LIBRARY_PATH="./libtorch-validation/libtorch/lib:$LD_LIBRARY_PATH"

# Cache permission issues
rm -rf ~/.cache/cargo-ci/linux
```

**Common Linux Errors**:
- ✅ **"gcc not found"** → Install `build-essential`
- ✅ **"sha256sum not found"** → Should not occur on Linux
- ✅ **LibTorch linking errors** → Check `LD_LIBRARY_PATH`

##### Windows-Specific Issues
```bash
# WSL2 setup (recommended)
wsl --install -d Ubuntu
# Then follow Linux setup inside WSL2

# Git Bash PATH issues
# Restart Git Bash after rustup installation
source ~/.cargo/env

# Cache issues in Git Bash
rm -rf ~/.cache/cargo-ci/windows
```

**Common Windows Errors**:
- ✅ **"rustup not found"** → Restart Git Bash after installation
- ✅ **"sha256sum not found"** → Uses file size fallback in Git Bash
- ✅ **Path separator issues** → Use WSL2 for full compatibility
- ✅ **LibTorch not available** → Use WSL2 for LibTorch validation

##### macOS-Specific Issues
```bash
# Missing Xcode tools
xcode-select --install

# LibTorch validation path issues (note DYLD vs LD)
export DYLD_LIBRARY_PATH="./libtorch-validation/libtorch/lib:$DYLD_LIBRARY_PATH"

# BSD find compatibility issues
# Scripts automatically detect and adapt

# Cache cleanup issues
rm -rf ~/.cache/cargo-ci/macos
```

**Common macOS Errors**:
- ✅ **"clang not found"** → Install Xcode command line tools
- ✅ **"sha256sum not found"** → Uses `shasum -a 256` automatically
- ✅ **BSD find differences** → Scripts handle automatically
- ✅ **LibTorch DYLD issues** → Use `DYLD_LIBRARY_PATH` not `LD_LIBRARY_PATH`

#### Cross-Compilation Troubleshooting

##### ARM64 Cross-Compilation Issues
```bash
# Install missing targets
rustup target add aarch64-unknown-linux-gnu
rustup target add aarch64-apple-darwin

# Linux: Install cross-compilation tools
sudo apt install gcc-aarch64-linux-gnu

# Check target installation
rustup target list --installed
```

**Common Cross-Compilation Errors**:
- ✅ **"target not found"** → Install with `rustup target add`
- ✅ **"linker not found"** → Install platform-specific cross-tools
- ✅ **"feature not supported"** → Some features may not work on all targets

#### Local Testing (Platform-Specific)

##### Linux/WSL2 Local Testing
```bash
# Run the same checks as Linux CI
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test -p train-station --lib
cargo build --all-targets --release
cargo doc --no-deps --document-private-items

# Test cross-compilation
rustup target add aarch64-unknown-linux-gnu
cargo check --target aarch64-unknown-linux-gnu --lib -p train-station

# Test LibTorch validation (if available)
export LD_LIBRARY_PATH="./libtorch-validation/libtorch/lib:$LD_LIBRARY_PATH"
cargo test -p libtorch-validation
```

##### Windows (Git Bash) Local Testing
```bash
# Run the same checks as Windows CI
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test -p train-station --lib
cargo build --all-targets --release
cargo doc --no-deps --document-private-items

# Test cross-compilation
rustup target add aarch64-apple-darwin
cargo check --target aarch64-apple-darwin --lib -p train-station

# Note: LibTorch validation typically not available on Windows
```

##### macOS Local Testing
```bash
# Run the same checks as macOS CI
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test -p train-station --lib
cargo build --all-targets --release
cargo doc --no-deps --document-private-items

# Test cross-compilation (native ARM64 if on Apple Silicon)
rustup target add aarch64-apple-darwin
cargo check --target aarch64-apple-darwin --lib -p train-station

# Test LibTorch validation (if available)
export DYLD_LIBRARY_PATH="./libtorch-validation/libtorch/lib:$DYLD_LIBRARY_PATH"
cargo test -p libtorch-validation
```

#### CI Badge Monitoring

Monitor platform-specific CI status via README badges:
- **Linux**: [![Linux](https://github.com/ewhinery8/train-station/actions/workflows/ci-linux.yml/badge.svg)]
- **Windows**: [![Windows](https://github.com/ewhinery8/train-station/actions/workflows/ci-windows.yml/badge.svg)]  
- **macOS**: [![macOS](https://github.com/ewhinery8/train-station/actions/workflows/ci-macos.yml/badge.svg)]
- **Cross-Compile**: [![Cross-Compile](https://github.com/ewhinery8/train-station/actions/workflows/cross-compile.yml/badge.svg)]

#### Getting Help with Platform Issues

1. **Check platform-specific CI logs** - Click the relevant badge above
2. **Compare with working platforms** - See which platforms pass/fail
3. **Test locally** - Use platform-specific local testing commands above
4. **Check tool availability** - Verify required tools are installed
5. **Use recommended setup** - Follow platform-specific setup instructions

## License

By contributing to Train Station, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).

---

Thank you for contributing to Train Station! Your efforts help make high-performance ML in Rust accessible to everyone.
