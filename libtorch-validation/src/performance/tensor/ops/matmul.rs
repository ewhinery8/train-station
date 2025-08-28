//! Matrix multiplication operation performance benchmarking
//!
//! This module provides performance testing for tensor matrix multiplication operations
//! against LibTorch, including both 2D and higher-dimensional matrix multiplication.

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

/// Matrix multiplication operations performance tester
pub struct MatmulPerformanceTester {
    pub tester: PerformanceTester,
}

impl MatmulPerformanceTester {
    /// Create a new matrix multiplication performance tester
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        MatmulPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new matrix multiplication performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        MatmulPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Test matrix multiplication performance for compatible shapes
    pub fn test_matmul(&mut self, shape_a: &[usize], shape_b: &[usize]) -> PerformanceResult {
        self.tester.benchmark_with_params(
            "matmul",
            shape_a,
            &[("shape_b", &format!("{:?}", shape_b))],
            || {
                let a = create_test_tensor(shape_a, TestPattern::Random);
                let b = create_test_tensor(shape_b, TestPattern::Sequential);
                a.matmul(&b)
            },
            || {
                let a_data = create_data_with_pattern(shape_a, TestPattern::Random);
                let b_data = create_data_with_pattern(shape_b, TestPattern::Sequential);
                let a = LibTorchTensor::from_data(&a_data, shape_a)
                    .expect("Failed to create LibTorch tensor A");
                let b = LibTorchTensor::from_data(&b_data, shape_b)
                    .expect("Failed to create LibTorch tensor B");
                a.matmul(&b)
            },
        )
    }

    /// Run comprehensive matrix multiplication performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        // Generate standard matrix multiplication test cases
        let standard_shape_pairs = vec![
            // 2D square matrices
            (vec![32, 32], vec![32, 32]),
            (vec![64, 64], vec![64, 64]),
            (vec![128, 128], vec![128, 128]),
            // 2D non-square matrices
            (vec![2, 3], vec![3, 4]),
            (vec![4, 5], vec![5, 6]),
            (vec![10, 20], vec![20, 15]),
            // 3D batch matrices
            (vec![32, 64, 64], vec![32, 64, 64]),
            (vec![64, 128, 128], vec![64, 128, 128]),
        ];
        self.test_all_operations_with_shapes(&standard_shape_pairs)
    }

    /// Run comprehensive matrix multiplication performance tests with custom shape pairs
    pub fn test_all_operations_with_shapes(
        &mut self,
        shape_pairs: &[(Vec<usize>, Vec<usize>)],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(60));
        println!("    TENSOR MATRIX MULTIPLICATION PERFORMANCE TESTS");
        println!("{}", "=".repeat(60));

        let mut results = Vec::new();

        // Validate that all shape pairs are compatible for matrix multiplication
        for (shape_a, shape_b) in shape_pairs {
            self.validate_matmul_shapes(shape_a, shape_b);
        }

        println!(
            "Testing {} matrix multiplication shape pairs",
            shape_pairs.len()
        );

        // Run performance tests for all provided shape pairs
        for (shape_a, shape_b) in shape_pairs {
            println!("Testing matmul: {:?} * {:?}", shape_a, shape_b);
            let result = self.test_matmul(shape_a, shape_b);
            results.push(result);
        }

        results
    }

    /// Validate that two shapes are compatible for matrix multiplication
    fn validate_matmul_shapes(&self, shape_a: &[usize], shape_b: &[usize]) {
        match (shape_a.len(), shape_b.len()) {
            (2, 2) => {
                // 2D matrix multiplication: A(m×n) * B(n×p) = C(m×p)
                let cols_a = shape_a[1];
                let rows_b = shape_b[0];
                assert_eq!(
                    cols_a, rows_b,
                    "2D matrix multiplication requires cols_a == rows_b, got {} != {}",
                    cols_a, rows_b
                );
            }
            (3, 3) => {
                // 3D batch matrix multiplication: A(batch×m×n) * B(batch×n×p) = C(batch×m×p)
                let batch_a = shape_a[0];
                let batch_b = shape_b[0];
                let cols_a = shape_a[2];
                let rows_b = shape_b[1];
                assert_eq!(
                    batch_a, batch_b,
                    "3D batch matrix multiplication requires batch_a == batch_b, got {} != {}",
                    batch_a, batch_b
                );
                assert_eq!(
                    cols_a, rows_b,
                    "3D batch matrix multiplication requires cols_a == rows_b, got {} != {}",
                    cols_a, rows_b
                );
            }
            _ => {
                panic!(
                    "Matrix multiplication requires both shapes to be 2D or both 3D, got {:?} and {:?}",
                    shape_a, shape_b
                );
            }
        }
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

impl Default for MatmulPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_performance_tester() {
        let mut tester = MatmulPerformanceTester::new();

        // Test a small operation to ensure everything works
        let result = tester.test_matmul(&[2, 3], &[3, 4]);
        assert_eq!(result.operation, "matmul");
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.iterations, 1000);

        // Check that shape_b parameter was recorded
        assert_eq!(
            result.test_params.get("shape_b"),
            Some(&"[3, 4]".to_string())
        );
    }

    #[test]
    fn test_matmul_with_shape_pairs() {
        let mut tester = MatmulPerformanceTester::new();

        // Test with valid shape pairs
        let shape_pairs = vec![
            (vec![64, 64], vec![64, 64]),         // 2D square
            (vec![128, 128], vec![128, 128]),     // 2D square
            (vec![32, 64, 64], vec![32, 64, 64]), // 3D batch
        ];

        let results = tester.test_all_operations_with_shapes(&shape_pairs);

        // Should have results from all shape pairs
        assert_eq!(
            results.len(),
            shape_pairs.len(),
            "Should test all shape pairs"
        );

        // Verify that all results are from matmul operations
        for result in &results {
            assert_eq!(result.operation, "matmul");
        }
    }

    #[test]
    #[should_panic(expected = "Matrix multiplication requires both shapes to be 2D or both 3D")]
    fn test_matmul_incompatible_shapes() {
        let mut tester = MatmulPerformanceTester::new();

        // Test with incompatible shape pairs that should cause validation to fail
        let incompatible_shape_pairs = vec![
            (vec![32], vec![64]), // 1D shapes - should panic
        ];

        // This should panic during validation
        tester.test_all_operations_with_shapes(&incompatible_shape_pairs);
    }

    #[test]
    #[should_panic(expected = "2D matrix multiplication requires cols_a == rows_b")]
    fn test_matmul_dimension_mismatch() {
        let mut tester = MatmulPerformanceTester::new();

        // Test with 2D shapes that have incompatible dimensions
        let incompatible_shape_pairs = vec![
            (vec![2, 3], vec![4, 5]), // 2×3 * 4×5 - cols_a(3) != rows_b(4)
        ];

        // This should panic during validation
        tester.test_all_operations_with_shapes(&incompatible_shape_pairs);
    }
}
