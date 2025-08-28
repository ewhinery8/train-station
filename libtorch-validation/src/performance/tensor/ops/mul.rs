//! Multiplication operation performance benchmarking
//!
//! This module provides performance testing for tensor multiplication operations
//! against LibTorch, including both tensor-tensor and tensor-scalar multiplication.

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

/// Multiplication operations performance tester
pub struct MulPerformanceTester {
    pub tester: PerformanceTester,
}

impl MulPerformanceTester {
    /// Create a new multiplication performance tester
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        MulPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new multiplication performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        MulPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Test tensor * tensor multiplication performance
    pub fn test_mul_tensor(&mut self, shape: &[usize]) -> PerformanceResult {
        self.tester.benchmark_operation(
            "mul_tensor",
            shape,
            || {
                let a = create_test_tensor(shape, TestPattern::Random);
                let b = create_test_tensor(shape, TestPattern::Sequential);
                a.mul_tensor(&b)
            },
            || {
                let a_data = create_data_with_pattern(shape, TestPattern::Random);
                let b_data = create_data_with_pattern(shape, TestPattern::Sequential);
                let a = LibTorchTensor::from_data(&a_data, shape)
                    .expect("Failed to create LibTorch tensor A");
                let b = LibTorchTensor::from_data(&b_data, shape)
                    .expect("Failed to create LibTorch tensor B");
                a.mul_tensor(&b)
            },
        )
    }

    /// Test tensor * scalar multiplication performance
    pub fn test_mul_scalar(&mut self, shape: &[usize], scalar: f32) -> PerformanceResult {
        self.tester.benchmark_with_params(
            "mul_scalar",
            shape,
            &[("scalar", &scalar.to_string())],
            || {
                let a = create_test_tensor(shape, TestPattern::Random);
                a.mul_scalar(scalar)
            },
            || {
                let a_data = create_data_with_pattern(shape, TestPattern::Random);
                let a = LibTorchTensor::from_data(&a_data, shape)
                    .expect("Failed to create LibTorch tensor");
                a.mul_scalar(scalar)
            },
        )
    }

    /// Run comprehensive multiplication performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        self.test_all_operations_with_shapes(&super::super::super::core::generate_test_shapes())
    }

    /// Run comprehensive multiplication performance tests with custom shapes
    pub fn test_all_operations_with_shapes(
        &mut self,
        test_shapes: &[Vec<usize>],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(60));
        println!("      TENSOR MULTIPLICATION PERFORMANCE TESTS");
        println!("{}", "=".repeat(60));

        let mut results = Vec::new();

        for shape in test_shapes {
            // Test tensor multiplication
            let result = self.test_mul_tensor(shape);
            results.push(result);

            // Test scalar multiplication with a few different values
            for scalar in [1.0, 42.42, -5.0] {
                let result = self.test_mul_scalar(shape, scalar);
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

impl Default for MulPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_performance_tester() {
        let mut tester = MulPerformanceTester::new();

        // Test a small operation to ensure everything works
        let result = tester.test_mul_tensor(&[2, 2]);
        assert_eq!(result.operation, "mul_tensor");
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.iterations, 1000);
    }

    #[test]
    fn test_mul_scalar_performance() {
        let mut tester = MulPerformanceTester::new();

        let result = tester.test_mul_scalar(&[4, 4], 5.0);
        assert_eq!(result.operation, "mul_scalar");
        assert_eq!(result.shape, vec![4, 4]);

        // Check that scalar parameter was recorded
        assert_eq!(result.test_params.get("scalar"), Some(&"5".to_string()));
    }
}
