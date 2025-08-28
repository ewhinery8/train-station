//! View operation performance benchmarking
//!
//! This module provides performance testing for tensor view operations
//! against LibTorch, including reshaping tensors to different dimensions.

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

/// View operations performance tester
pub struct ViewPerformanceTester {
    pub tester: PerformanceTester,
}

impl ViewPerformanceTester {
    /// Create a new view performance tester
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        ViewPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new view performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        ViewPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Test 1D view to 2D operation performance
    pub fn test_view_1d_to_2d(&mut self, shape: &[usize]) -> Option<PerformanceResult> {
        assert_eq!(shape.len(), 1);
        let n = shape[0];
        if n % 2 != 0 {
            return None;
        }
        let shape_param = format!("{:?}", shape);

        Some(self.tester.benchmark_with_params(
            "view_1d_to_2d",
            &[2, n / 2],
            &[
                ("original_shape", &shape_param),
                ("input_shape", &shape_param),
            ],
            || {
                let a = create_test_tensor(shape, TestPattern::Sequential);
                a.view(vec![2, -1])
            },
            || {
                let data = create_data_with_pattern(shape, TestPattern::Sequential);
                let torch = LibTorchTensor::from_data(&data, shape)?;
                torch.view(&[2, n / 2])
            },
        ))
    }

    /// Run comprehensive view performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        self.test_all_operations_with_shapes(&super::super::super::core::generate_test_shapes())
    }

    /// Run comprehensive view performance tests with custom shapes
    pub fn test_all_operations_with_shapes(
        &mut self,
        test_shapes: &[Vec<usize>],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(60));
        println!("           TENSOR VIEW PERFORMANCE TESTS");
        println!("{}", "=".repeat(60));

        let mut results = Vec::new();

        for shape in test_shapes {
            // Only test shapes that can be viewed (at least 1D)
            if !shape.is_empty() {
                if let Some(result) = self.test_view_1d_to_2d(shape) {
                    results.push(result);
                }
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

impl Default for ViewPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_performance_tester() {
        let mut tester = ViewPerformanceTester::new();

        // Test a small operation to ensure everything works
        let result = tester.test_view_1d_to_2d(&[6]);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.operation, "view_1d_to_2d");
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.iterations, 1000);
    }

    #[test]
    fn test_view_odd_shape() {
        let mut tester = ViewPerformanceTester::new();

        // Test with odd shape (should return None)
        let result = tester.test_view_1d_to_2d(&[5]);
        assert!(result.is_none());
    }
}
