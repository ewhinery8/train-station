//! Transpose operation performance benchmarking
//!
//! This module provides performance testing for tensor transpose operations
//! against LibTorch, including 2D transpose operations.

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

/// Transpose operations performance tester
pub struct TransposePerformanceTester {
    pub tester: PerformanceTester,
}

impl TransposePerformanceTester {
    /// Create a new transpose performance tester
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        TransposePerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new transpose performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        TransposePerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Test 2D transpose operation performance
    pub fn test_transpose_2d(&mut self, shape: &[usize]) -> PerformanceResult {
        assert_eq!(shape.len(), 2, "test_transpose_2d requires 2D shape");
        let shape_param = format!("{:?}", shape);

        self.tester.benchmark_with_params(
            "transpose_2d",
            shape,
            &[("input_shape", &shape_param)],
            || {
                let a = create_test_tensor(shape, TestPattern::Sequential);
                a.transpose(0, 1)
            },
            || {
                let data = create_data_with_pattern(shape, TestPattern::Sequential);
                let torch = LibTorchTensor::from_data(&data, shape)?;
                torch.permute(&[1, 0])
            },
        )
    }

    /// Run comprehensive transpose performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        self.test_all_operations_with_shapes(&super::super::super::core::generate_test_shapes())
    }

    /// Run comprehensive transpose performance tests with custom shapes
    pub fn test_all_operations_with_shapes(
        &mut self,
        test_shapes: &[Vec<usize>],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(60));
        println!("         TENSOR TRANSPOSE PERFORMANCE TESTS");
        println!("{}", "=".repeat(60));

        let mut results = Vec::new();

        for shape in test_shapes {
            // Only test shapes that can be transposed (at least 2D)
            if shape.len() >= 2 {
                let result = self.test_transpose_2d(shape);
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

impl Default for TransposePerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_performance_tester() {
        let mut tester = TransposePerformanceTester::new();

        // Test a small operation to ensure everything works
        let result = tester.test_transpose_2d(&[2, 3]);
        assert_eq!(result.operation, "transpose_2d");
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.iterations, 1000);
    }
}
