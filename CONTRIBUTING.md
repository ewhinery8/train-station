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

### Expected format (for CHANGELOG rendering)

Our release scripts automatically generate changelog entries from commit messages. The format is:

**Subject line:** Conventional prefixes are stripped since section headers indicate the type
- `feat: add new feature` → `- add new feature` (in Added section)
- `feat!: breaking change` → `- breaking change` (in Added section)
- `fix(scope): fix bug` → `- fix bug` (in Fixed section)
- `perf: improve performance` → `- improve performance` (in Performance section)

**Breaking Changes section:** Shows content after `BREAKING CHANGE:` or stripped subject
- `feat!: breaking change` with `BREAKING CHANGE: API changed` → `- API changed`
- `feat!: breaking change` without `BREAKING CHANGE:` block → `- breaking change`

**Body (optional):** Bullet list of details, each line prefixed with four spaces and a dash
```text
feat: improved Tensor Iterator system semantics and collection performance

    - removed iter_values() in favor of tensor.data().iter()
    - updated semantics for iterators to better reflect std Rust
    - updated code docs to better reflect functionality
    - added training examples and basic network example
```

**Breaking changes:** Include `!` after type and `BREAKING CHANGE:` in body for automatic categorization
```text
feat!: iterator API updates

    - renamed iter_values() to data().iter()
    - updated public API signatures

BREAKING CHANGE: Iterator signatures changed; see examples and docs
```

The Breaking Changes section will show:
- `- Iterator signatures changed; see examples and docs` (text after `BREAKING CHANGE:`)
- If no `BREAKING CHANGE:` block exists, falls back to: `- iterator API updates` (subject without prefix)

**Changelog sections generated:**
- `feat` commits → **Added** section
- `fix` commits → **Fixed** section  
- `perf` commits → **Performance** section
- `docs` commits → **Documentation** section
- `chore` commits → **Maintenance** section
- Any commit with `!` → **Breaking Changes** section (in addition to type section)
- Other types → **Other Changes** section

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

