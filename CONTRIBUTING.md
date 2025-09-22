# Contributing to Train Station

Clear, fast, and simple. This guide keeps contributions consistent and easy for both developers and AI agents.

## Basics
- Be respectful. We follow the Rust Code of Conduct.
- Target performance and safety. Keep the core crate zero-dependency.
- Prefer simple solutions over complex abstractions.

## Workflow
1. Fork and clone the repo.
2. Create a branch: `feat/...`, `fix/...`, `perf/...`, `docs/...`, `refactor/...`, `test/...`, or `chore/...`.
3. Commit using Conventional Commits, for example: `feat: add broadcasting for add`.
4. Push and open a pull request to `master`.

## Before opening a PR
- Tests pass: `cargo test -p train-station`.
- Format and lint: `cargo fmt --all` and `cargo clippy --all-targets`.
- Add tests for new features and fixes.
- Document public APIs with examples where appropriate.
- For performance work, include a short benchmark or timing note.

## Development setup (quick)
```bash
rustup update
rustup component add rustfmt clippy
cargo test -p train-station --lib
```

## Performance
- Optimize hot paths. Minimize allocations.
- Use SIMD where it clearly helps; always provide a scalar path.
- Detect features at runtime for x86 (for example, avx2).

## Testing
- Co-locate tests with code in `mod tests { ... }`.
- Cover normal cases, edge cases, and error paths.
- Validate broadcasting and gradients where relevant.

## Documentation
- Keep rustdoc concise and example-driven.
- Document safety for any unsafe code.
- Prefer small, runnable examples.

## Commit messages
Use Conventional Commits. Examples:
```text
feat: add matmul with batched broadcasting
fix: correct gradient for slice_view
perf: speed up add by avoiding extra allocs
docs: document iterator API with examples
```

## Issue reports and feature requests
- Describe the problem, steps to reproduce, and expected behavior.
- Include OS, Rust version, and feature flags if relevant.
- For features, explain the use case and proposed API.

## Maintainers: releases (summary)
1. Prepare version with scripts or manual edits.
2. Ensure CI is green on all platforms.
3. Tag and push. GitHub Actions publishes.

## License
By contributing, you agree your code is licensed under MIT or Apache-2.0, at your option.

Thank you for contributing to Train Station.

