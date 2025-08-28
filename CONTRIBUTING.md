# Contributing to Train Station

Thank you for considering contributing to Train Station! We welcome contributions that improve performance, add features, enhance documentation, or fix bugs.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Performance Requirements](#performance-requirements)
- [Testing Standards](#testing-standards)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Code Style](#code-style)
- [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/train-station.git`
3. Add upstream remote: `git remote add upstream https://github.com/originalowner/train-station.git`
4. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Rust 1.70+ (stable toolchain)
- For CUDA development: CUDA Toolkit 11.0+
- For validation testing: LibTorch (see libtorch-validation/README.md)

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/train-station.git
cd train-station

# Run tests
cargo test -p train-station

# Run examples
cargo run --example tensor_basics
```

### LibTorch Validation Setup (Optional)

For mathematical validation and performance benchmarking:

```bash
# Download LibTorch and extract to libtorch-validation/libtorch/
# Then set library path
export LD_LIBRARY_PATH="./libtorch-validation/libtorch/lib:$LD_LIBRARY_PATH"

# Run validation tests
cargo test -p libtorch-validation
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

1. Rebase on latest main: `git rebase upstream/main`
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

## License

By contributing to Train Station, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).

---

Thank you for contributing to Train Station! Your efforts help make high-performance ML in Rust accessible to everyone.
