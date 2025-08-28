//! Stack operation performance benchmarking
//!
//! This module provides performance testing for tensor stack operations
//! against LibTorch, including stacking tensors along new dimensions.

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

/// Stack operations performance tester
pub struct StackPerformanceTester {
    pub tester: PerformanceTester,
}

impl StackPerformanceTester {
    /// Create a new stack performance tester
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        StackPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new stack performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        StackPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Test stack along leading dimension performance
    pub fn test_stack_leading_dim(&mut self, shape: &[usize]) -> PerformanceResult {
        let mut out_shape = Vec::with_capacity(shape.len() + 1);
        out_shape.push(2);
        out_shape.extend_from_slice(shape);

        let shape_param = format!("{:?}", shape);
        self.tester.benchmark_with_params(
            "stack_leading_dim",
            &out_shape,
            &[("stack_dim", "0"), ("input_shape", &shape_param)],
            || {
                let a = create_test_tensor(shape, TestPattern::Random);
                let b = create_test_tensor(shape, TestPattern::Sequential);
                Tensor::stack(&[a, b], 0)
            },
            || {
                let a_data = create_data_with_pattern(shape, TestPattern::Random);
                let b_data = create_data_with_pattern(shape, TestPattern::Sequential);
                let a = LibTorchTensor::from_data(&a_data, shape).expect("torch a");
                let b = LibTorchTensor::from_data(&b_data, shape).expect("torch b");
                LibTorchTensor::stack(&[a, b], 0)
            },
        )
    }

    /// Run comprehensive stack performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        self.test_all_operations_with_shapes(&super::super::super::core::generate_test_shapes())
    }

    /// Run comprehensive stack performance tests with custom shapes
    pub fn test_all_operations_with_shapes(
        &mut self,
        test_shapes: &[Vec<usize>],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(60));
        println!("           TENSOR STACK PERFORMANCE TESTS");
        println!("{}", "=".repeat(60));

        let mut results = Vec::new();

        for shape in test_shapes {
            // Only test shapes that can be stacked (at least 1D)
            if !shape.is_empty() {
                let result = self.test_stack_leading_dim(shape);
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

impl Default for StackPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_performance_tester() {
        let mut tester = StackPerformanceTester::new();

        // Test a small operation to ensure everything works
        let result = tester.test_stack_leading_dim(&[2, 3]);
        assert_eq!(result.operation, "stack_leading_dim");
        assert_eq!(result.shape, vec![2, 2, 3]); // Stacked along leading dim
        assert_eq!(result.iterations, 1000);
    }
}
