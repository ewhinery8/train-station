# LibTorch Validation

> Internal validation crate for Train Station - used for testing gradtrack and operations

**WARNING: This crate is for internal use only and will never be published to crates.io**

## Purpose

This crate serves as the validation backbone for Train Station, providing:

- Mathematical correctness validation against PyTorch/LibTorch
- Performance benchmarking comparisons
- Regression detection for all tensor operations
- Gradient computation verification

Every operation in Train Station must pass through this gauntlet before it's considered production-ready.

## Architecture

The validation crate mirrors Train Station's structure exactly:

```
libtorch-validation/
├── src/
│   ├── ffi/           # C++ LibTorch wrapper
│   ├── validation/    # Correctness validation
│   └── performance/   # Performance benchmarking
```

## Setup

### Prerequisites

1. Download LibTorch (CPU version recommended for validation):
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip -d libtorch-validation/
```

2. Set library path :
```bash
export LD_LIBRARY_PATH="<./libtorch-validation/libtorch/lib:$LD_LIBRARY_PATH>"
```

### Building

The validation crate builds automatically with:
```bash
cargo build -p libtorch-validation
```

## Validation Testing

### Running All Validation Tests

```bash
# ALWAYS set the library path first!
LD_LIBRARY_PATH="./libtorch-validation/libtorch/lib" cargo test -p libtorch-validation
```

### Validation Standards

Every operation must achieve:
- **Numerical accuracy**: < 1e-6 absolute error (0.00e0 ideal)
- **Shape correctness**: Exact dimension matching
- **Broadcasting**: Identical behavior to PyTorch
- **Gradient accuracy**: Backprop validates to < 1e-6 error

## Performance Benchmarking

### Running Benchmarks

```bash
# Run all performance comparisons
LD_LIBRARY_PATH="./libtorch-validation/libtorch/lib" cargo run --release -p libtorch-validation --example add_performance
```

### Benchmark Configuration (configurable in source code)

Each benchmark runs:
- 1000 iterations per test
- 10 warmup iterations
- Multiple tensor sizes (32, 64, 128, 256, 512, 1024, 2048)
- Statistical analysis (mean, std dev, throughput)

### Visualization

Generate performance comparison plots:

```bash
cd scripts/visualization
pip install -r requirements.txt

# Generate visualizations
python3 performance_scatter.py <BENCHMARK RESULT JSON PATH>
```

## Adding New Validations

### 1. Create Validation File

Follow the mirror structure in `src/validation/tensor/ops/your_op.rs`:

```rust
impl TensorValidator {
    pub fn test_your_op(&self, shape: &[usize]) -> ComparisonResult {
        // Create Train Station tensor
        let ts_input = Tensor::randn(shape.to_vec(), None);
        
        // Create LibTorch tensor with same data
        let lt_input = LibTorchTensor::from_tensor(&ts_input)?;
        
        // Perform operations
        let ts_result = ts_input.your_op();
        let lt_result = lt_input.your_op();
        
        // Validate equivalence
        self.compare_tensors(&ts_result, &lt_result, "your_op")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_your_op_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let shapes = standard_test_shapes();
        
        for shape in shapes {
            let result = validator.test_your_op(&shape);
            assert!(result.passed, "Validation failed: {}", result.details);
        }
    }
}
```

### 2. Add Gradient Validation

```rust
#[test]
fn test_your_op_gradient_validation() {
    let validator = TensorValidator::new(1e-6, 1e-8);
    
    // Test gradient computation
    let x = Tensor::randn(vec![10, 10], None).with_requires_grad();
    let result = x.your_op();
    result.backward(None);
    
    // Compare with LibTorch gradients
    let lt_x = LibTorchTensor::from_tensor(&x)?;
    lt_x.set_requires_grad(true);
    let lt_result = lt_x.your_op();
    lt_result.backward();
    
    // Validate gradient equivalence
    let x_grad = x.grad().unwrap();
    let lt_x_grad = lt_x.grad()?;
    
    let comparison = validator.compare_tensors(&x_grad, &lt_x_grad, "gradient");
    assert!(comparison.passed);
}
```

### 3. Add Performance Benchmark

Create `src/performance/tensor/ops/your_op.rs`:

```rust
pub struct YourOpPerformanceTester {
    iterations: usize,
    warmup: usize,
}

impl YourOpPerformanceTester {
    pub fn test_performance(&self, shape: &[usize]) -> PerformanceResult {
        // Benchmark Train Station
        let ts_time = self.benchmark_train_station(shape);
        
        // Benchmark LibTorch
        let lt_time = self.benchmark_libtorch(shape);
        
        PerformanceResult {
            operation: "your_op",
            shape: shape.to_vec(),
            train_station_ns: ts_time,
            libtorch_ns: lt_time,
            speedup: lt_time as f64 / ts_time as f64,
        }
    }
}
```

## Common Issues

### Missing LD_LIBRARY_PATH
```
error while loading shared libraries: libtorch.so: cannot open shared object file
```
**Solution**: Always set `LD_LIBRARY_PATH` before running tests

### Version Mismatches
```
undefined reference to `at::Tensor::add'
```
**Solution**: Ensure LibTorch version matches the one used in build.rs

## Validation Checklist

Before marking an operation as complete:

- [ ] Unit tests pass in train-station crate
- [ ] LibTorch validation tests pass
- [ ] Gradient validation tests pass
- [ ] Performance benchmarks completed
- [ ] No performance regression vs previous version
- [ ] Edge cases tested (empty tensors, broadcasting, etc.)
- [ ] Documentation includes validation results

## Notes

- This crate uses LibTorch under the hood - it's the one exception to our zero-dependency rule
- All validation code is excluded from release builds
- Performance numbers may vary based on hardware and LibTorch version
- The FFI wrapper is kept minimal to reduce maintenance burden

---

*Trust, but verify. Speed is nothing without correctness.*

*The validation must pass. This is the way.*