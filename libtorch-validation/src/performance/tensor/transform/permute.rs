//! Permute operation performance benchmarking
//!
//! This module provides performance testing for tensor permute operations
//! against LibTorch, including 3D permute operations.

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

/// Permute operations performance tester
pub struct PermutePerformanceTester {
    pub tester: PerformanceTester,
}

impl PermutePerformanceTester {
    /// Create a new permute performance tester
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        PermutePerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new permute performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        PermutePerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Test 3D permute operation performance
    pub fn test_permute_3d(&mut self, shape: &[usize]) -> PerformanceResult {
        assert_eq!(shape.len(), 3, "test_permute_3d requires 3D shape");
        let shape_param = format!("{:?}", shape);

        self.tester.benchmark_with_params(
            "permute_3d",
            shape,
            &[("input_shape", &shape_param)],
            || {
                let a = create_test_tensor(shape, TestPattern::Random);
                a.permute(vec![0, 2, 1])
            },
            || {
                let data = create_data_with_pattern(shape, TestPattern::Random);
                let torch = LibTorchTensor::from_data(&data, shape)?;
                torch.permute(&[0, 2, 1])
            },
        )
    }

    /// Run comprehensive permute performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        self.test_all_operations_with_shapes(&super::super::super::core::generate_test_shapes())
    }

    /// Run comprehensive permute performance tests with custom shapes
    pub fn test_all_operations_with_shapes(
        &mut self,
        test_shapes: &[Vec<usize>],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(60));
        println!("          TENSOR PERMUTE PERFORMANCE TESTS");
        println!("{}", "=".repeat(60));

        let mut results = Vec::new();

        for shape in test_shapes {
            // Only test shapes that can be permuted (at least 3D)
            if shape.len() >= 3 {
                let result = self.test_permute_3d(shape);
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

impl Default for PermutePerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permute_performance_tester() {
        let mut tester = PermutePerformanceTester::new();

        // Test a small operation to ensure everything works
        let result = tester.test_permute_3d(&[2, 3, 4]);
        assert_eq!(result.operation, "permute_3d");
        assert_eq!(result.shape, vec![2, 3, 4]);
        assert_eq!(result.iterations, 1000);
    }
}
