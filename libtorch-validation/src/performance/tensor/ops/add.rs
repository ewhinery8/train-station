//! Addition operation performance benchmarking
//!
//! This module provides performance testing for tensor addition operations
//! against LibTorch, including both tensor-tensor and tensor-scalar addition.

use super::super::super::core::{
    create_test_tensor, PerformanceConfig, PerformanceResult, PerformanceTester, TestPattern,
};
use crate::ffi::LibTorchTensor;

/// Generate deterministic test data matching `create_test_tensor` patterns
fn create_data_with_pattern(shape: &[usize], pattern: TestPattern) -> Vec<f32> {
    let size: usize = shape.iter().product();
    let mut data = vec![0.0f32; size];
    match pattern {
        TestPattern::Ones => {
            for v in &mut data {
                *v = 1.0;
            }
        }
        TestPattern::Sequential => {
            for (i, v) in data.iter_mut().enumerate() {
                *v = (i + 1) as f32;
            }
        }
        TestPattern::Random => {
            for (i, v) in data.iter_mut().enumerate() {
                *v = ((i * 37 + 17) % 100) as f32 / 100.0;
            }
        }
    }
    data
}

/// Addition operations performance tester
pub struct AddPerformanceTester {
    pub tester: PerformanceTester,
}

impl AddPerformanceTester {
    /// Create a new addition performance tester
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        AddPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new addition performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        AddPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Test tensor + tensor addition performance
    pub fn test_add_tensor(&mut self, shape: &[usize]) -> PerformanceResult {
        self.tester.benchmark_operation(
            "add_tensor",
            shape,
            || {
                let a = create_test_tensor(shape, TestPattern::Random);
                let b = create_test_tensor(shape, TestPattern::Sequential);
                a.add_tensor(&b)
            },
            || {
                let a_data = create_data_with_pattern(shape, TestPattern::Random);
                let b_data = create_data_with_pattern(shape, TestPattern::Sequential);
                let a = LibTorchTensor::from_data(&a_data, shape)
                    .expect("Failed to create LibTorch tensor A");
                let b = LibTorchTensor::from_data(&b_data, shape)
                    .expect("Failed to create LibTorch tensor B");
                a.add_tensor(&b)
            },
        )
    }

    /// Test tensor + scalar addition performance
    pub fn test_add_scalar(&mut self, shape: &[usize], scalar: f32) -> PerformanceResult {
        self.tester.benchmark_with_params(
            "add_scalar",
            shape,
            &[("scalar", &scalar.to_string())],
            || {
                let a = create_test_tensor(shape, TestPattern::Random);
                a.add_scalar(scalar)
            },
            || {
                let a_data = create_data_with_pattern(shape, TestPattern::Random);
                let a = LibTorchTensor::from_data(&a_data, shape)
                    .expect("Failed to create LibTorch tensor");
                a.add_scalar(scalar)
            },
        )
    }

    /// Run comprehensive addition performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        self.test_all_operations_with_shapes(&super::super::super::core::generate_test_shapes())
    }

    /// Run comprehensive addition performance tests with custom shapes
    pub fn test_all_operations_with_shapes(
        &mut self,
        test_shapes: &[Vec<usize>],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(60));
        println!("          TENSOR ADDITION PERFORMANCE TESTS");
        println!("{}", "=".repeat(60));

        let mut results = Vec::new();

        for shape in test_shapes {
            // Test tensor addition
            let result = self.test_add_tensor(shape);
            results.push(result);

            // Test scalar addition with a few different values
            for scalar in [1.0, 42.42, -5.0] {
                let result = self.test_add_scalar(shape, scalar);
                results.push(result);
            }
        }

        results
    }

    /// Get access to underlying performance tester
    pub fn tester(&self) -> &PerformanceTester {
        &self.tester
    }

    /// Get mutable access to underlying performance tester
    pub fn tester_mut(&mut self) -> &mut PerformanceTester {
        &mut self.tester
    }
}

impl Default for AddPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_performance_tester() {
        let mut tester = AddPerformanceTester::new();

        // Test a small operation to ensure everything works
        let result = tester.test_add_tensor(&[2, 2]);
        assert_eq!(result.operation, "add_tensor");
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.iterations, 1000);
    }

    #[test]
    fn test_add_scalar_performance() {
        let mut tester = AddPerformanceTester::new();

        let result = tester.test_add_scalar(&[4, 4], 5.0);
        assert_eq!(result.operation, "add_scalar");
        assert_eq!(result.shape, vec![4, 4]);

        // Check that scalar parameter was recorded
        assert_eq!(result.test_params.get("scalar"), Some(&"5".to_string()));
    }
}
