//! Flatten operation performance benchmarking
//!
//! This module provides performance testing for tensor flatten operations
//! against LibTorch, including flattening tensors of various dimensions.

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

/// Flatten operations performance tester
pub struct FlattenPerformanceTester {
    pub tester: PerformanceTester,
}

impl FlattenPerformanceTester {
    /// Create a new flatten performance tester
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        FlattenPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new flatten performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        FlattenPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Test flatten operation performance
    pub fn test_flatten(&mut self, shape: &[usize]) -> PerformanceResult {
        let numel: usize = shape.iter().product();
        self.tester.benchmark_with_params(
            "flatten",
            &[numel],
            &[("original_shape", &format!("{:?}", shape))],
            || {
                let a = create_test_tensor(shape, TestPattern::Sequential);
                a.flatten()
            },
            || {
                let data = create_data_with_pattern(shape, TestPattern::Sequential);
                let torch = LibTorchTensor::from_data(&data, shape)?;
                torch.view(&[numel])
            },
        )
    }

    /// Run comprehensive flatten performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        self.test_all_operations_with_shapes(&super::super::super::core::generate_test_shapes())
    }

    /// Run comprehensive flatten performance tests with custom shapes
    pub fn test_all_operations_with_shapes(
        &mut self,
        test_shapes: &[Vec<usize>],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(60));
        println!("          TENSOR FLATTEN PERFORMANCE TESTS");
        println!("{}", "=".repeat(60));

        let mut results = Vec::new();

        for shape in test_shapes {
            // Only test shapes that can be flattened (at least 2D)
            if shape.len() >= 2 {
                let result = self.test_flatten(shape);
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

impl Default for FlattenPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_performance_tester() {
        let mut tester = FlattenPerformanceTester::new();

        // Test a small operation to ensure everything works
        let result = tester.test_flatten(&[2, 3]);
        assert_eq!(result.operation, "flatten");
        assert_eq!(result.shape, vec![6]); // Flattened to 1D
        assert_eq!(result.iterations, 1000);
    }
}
