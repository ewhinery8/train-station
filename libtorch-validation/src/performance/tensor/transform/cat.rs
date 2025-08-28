//! Concatenation operation performance benchmarking
//!
//! This module provides performance testing for tensor concatenation operations
//! against LibTorch, including concatenating tensors along different dimensions.

use super::super::super::core::{
    create_test_tensor, PerformanceConfig, PerformanceResult, PerformanceTester, TestPattern,
};
use crate::ffi::LibTorchTensor;
use train_station::Tensor;

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

/// Concatenation operations performance tester
pub struct CatPerformanceTester {
    pub tester: PerformanceTester,
}

impl CatPerformanceTester {
    /// Create a new concatenation performance tester
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        CatPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new concatenation performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        CatPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Test concatenation along last dimension performance
    pub fn test_cat_last_dim(&mut self, shape: &[usize]) -> PerformanceResult {
        assert!(!shape.is_empty());
        let mut out_shape = shape.to_vec();
        let last_dim = *shape.last().unwrap();
        let last_index = out_shape.len() - 1;
        out_shape[last_index] = last_dim * 2;

        let concat_dim = last_index;
        let out_shape_vec = out_shape.clone();
        let shape_param = format!("{:?}", shape);
        self.tester.benchmark_with_params(
            "cat_last_dim",
            &out_shape_vec,
            &[
                ("concat_dim", &format!("{}", concat_dim)),
                ("input_shape", &shape_param),
            ],
            || {
                let a = create_test_tensor(shape, TestPattern::Random);
                let b = create_test_tensor(shape, TestPattern::Sequential);
                Tensor::cat(&[a, b], concat_dim)
            },
            || {
                let a_data = create_data_with_pattern(shape, TestPattern::Random);
                let b_data = create_data_with_pattern(shape, TestPattern::Sequential);
                let a = LibTorchTensor::from_data(&a_data, shape)?;
                let b = LibTorchTensor::from_data(&b_data, shape)?;
                LibTorchTensor::cat(&[a, b], concat_dim)
            },
        )
    }

    /// Run comprehensive concatenation performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        self.test_all_operations_with_shapes(&super::super::super::core::generate_test_shapes())
    }

    /// Run comprehensive concatenation performance tests with custom shapes
    pub fn test_all_operations_with_shapes(
        &mut self,
        test_shapes: &[Vec<usize>],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(60));
        println!("       TENSOR CONCATENATION PERFORMANCE TESTS");
        println!("{}", "=".repeat(60));

        let mut results = Vec::new();

        for shape in test_shapes {
            // Only test shapes that can be concatenated (at least 2D)
            if shape.len() >= 2 {
                let result = self.test_cat_last_dim(shape);
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

impl Default for CatPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cat_performance_tester() {
        let mut tester = CatPerformanceTester::new();

        // Test a small operation to ensure everything works
        let result = tester.test_cat_last_dim(&[2, 3]);
        assert_eq!(result.operation, "cat_last_dim");
        assert_eq!(result.shape, vec![2, 6]); // Concatenated along last dim
        assert_eq!(result.iterations, 1000);
    }
}
