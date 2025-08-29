# Branch Naming Convention

This project follows a structured branch naming convention to improve code organization and review processes.

## Format
```
<type>/<short-description>
```

## Branch Types

### Feature Branches
- **`feat/`** - New features or enhancements
- Examples:
  - `feat/matrix-multiplication`
  - `feat/cuda-acceleration`
  - `feat/adam-optimizer`

### Bug Fix Branches  
- **`fix/`** - Bug fixes
- Examples:
  - `fix/memory-leak-tensors`
  - `fix/gradient-computation`
  - `fix/simd-alignment-issue`

### Performance Branches
- **`perf/`** - Performance improvements
- Examples:
  - `perf/simd-optimization`
  - `perf/cache-friendly-layout`
  - `perf/reduce-allocations`

### Documentation Branches
- **`docs/`** - Documentation updates
- Examples:
  - `docs/api-reference`
  - `docs/getting-started`
  - `docs/performance-guide`

### Refactoring Branches
- **`refactor/`** - Code refactoring
- Examples:
  - `refactor/tensor-core`
  - `refactor/autograd-system`
  - `refactor/memory-management`

### Testing Branches
- **`test/`** - Adding or improving tests
- Examples:
  - `test/libtorch-validation`
  - `test/cuda-operations`
  - `test/edge-cases`

### Maintenance Branches
- **`chore/`** - Maintenance tasks
- Examples:
  - `chore/update-dependencies`
  - `chore/ci-improvements`
  - `chore/cleanup-warnings`

## Naming Guidelines

### Do ✅
- Use lowercase letters
- Use hyphens to separate words
- Keep descriptions concise but descriptive
- Be specific about what the branch does

### Don't ❌
- Use spaces or underscores
- Use generic names like `fix/bug` or `feat/new-feature`
- Include issue numbers in branch names
- Use uppercase letters

## Examples

### Good Branch Names ✅
```
feat/tensor-broadcasting
fix/gradient-accumulation-bug
perf/avx2-simd-operations
docs/autograd-examples
refactor/memory-pool-system
test/cuda-validation-suite
chore/update-rust-version
```

### Bad Branch Names ❌
```
feature_branch          # Use feat/ prefix and hyphens
Fix_Bug                 # Use fix/ prefix and lowercase
new-stuff               # Too generic, no type prefix
feat/issue-123          # Don't include issue numbers
performance             # Use perf/ prefix
```

## Workflow Integration

Branches should align with commit types:
- `feat/` branches → `feat:` commits
- `fix/` branches → `fix:` commits  
- `perf/` branches → `perf:` commits
- etc.
