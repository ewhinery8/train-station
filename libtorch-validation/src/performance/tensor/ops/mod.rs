//! Tensor operations performance benchmarking
//!
//! This module provides comprehensive performance testing for all tensor operations
//! against LibTorch, covering various shapes and scenarios to identify performance
//! characteristics and optimization opportunities.
//!
//! ## Test Coverage
//!
//! - **Dimensions**: 32, 64, 128, 256, 512, 1024, 2056
//! - **Shapes**: 1D vectors, 2D matrices, 3D tensors with realistic batch sizes
//! - **Operations**: All arithmetic, activation, and mathematical operations
//! - **Iterations**: 1000 iterations per test for statistical significance

pub mod add;
pub mod div;
pub mod matmul;
pub mod mul;
pub mod sub;

use super::super::core::{PerformanceConfig, PerformanceResult, PerformanceTester};

// Re-export individual operation testers for convenience
pub use add::AddPerformanceTester;
pub use div::DivPerformanceTester;
pub use matmul::MatmulPerformanceTester;
pub use mul::MulPerformanceTester;
pub use sub::SubPerformanceTester;

/// Create compatible matrix multiplication shape pairs from test shapes
fn create_matmul_shape_pairs(test_shapes: &[Vec<usize>]) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut shape_pairs = Vec::new();

    // Process 2D shapes for matrix multiplication
    let two_d_shapes: Vec<_> = test_shapes
        .iter()
        .filter(|shape| shape.len() == 2)
        .collect();

    // Create square matrix multiplication pairs (A: n×n, B: n×n)
    for shape_a in &two_d_shapes {
        let cols_a = shape_a[1];

        // Find compatible shapes for matrix multiplication
        for shape_b in &two_d_shapes {
            let rows_b = shape_b[0];

            // Matrix multiplication: A(m×n) * B(n×p) = C(m×p)
            // Check if cols_a == rows_b (inner dimensions must match)
            if cols_a == rows_b {
                shape_pairs.push((shape_a.to_vec(), shape_b.to_vec()));
            }
        }
    }

    // Process 3D shapes for batch matrix multiplication
    let three_d_shapes: Vec<_> = test_shapes
        .iter()
        .filter(|shape| shape.len() == 3)
        .collect();

    // Create batch matrix multiplication pairs (A: batch×m×n, B: batch×n×p)
    for shape_a in &three_d_shapes {
        let (batch_a, cols_a) = (shape_a[0], shape_a[2]);

        for shape_b in &three_d_shapes {
            let (batch_b, rows_b) = (shape_b[0], shape_b[1]);

            // Batch matrix multiplication: A(batch×m×n) * B(batch×n×p) = C(batch×m×p)
            // Check if batch sizes match and inner dimensions are compatible
            if batch_a == batch_b && cols_a == rows_b {
                shape_pairs.push((shape_a.to_vec(), shape_b.to_vec()));
            }
        }
    }

    // Add some additional non-square test cases for comprehensive coverage
    shape_pairs.push((vec![2usize, 3], vec![3usize, 4])); // 2×3 * 3×4 = 2×4
    shape_pairs.push((vec![4usize, 5], vec![5usize, 6])); // 4×5 * 5×6 = 4×6
    shape_pairs.push((vec![10usize, 20], vec![20usize, 15])); // 10×20 * 20×15 = 10×15
    shape_pairs.push((vec![1usize, 100], vec![100usize, 1])); // 1×100 * 100×1 = 1×1 (dot product)
    shape_pairs.push((vec![100usize, 1], vec![1usize, 100])); // 100×1 * 1×100 = 100×100 (outer product)
    shape_pairs.push((vec![2usize, 3, 4], vec![2usize, 4, 5])); // batch×3×4 * batch×4×5 = batch×3×5
    shape_pairs.push((vec![3usize, 4, 5], vec![3usize, 5, 6])); // batch×4×5 * batch×5×6 = batch×4×6

    // Remove duplicates while preserving order
    shape_pairs.sort();
    shape_pairs.dedup();

    shape_pairs
}

/// Comprehensive performance tester for all tensor operations
pub struct OpPerformanceTester {
    pub tester: PerformanceTester,
}

impl OpPerformanceTester {
    /// Create a new ops performance tester with default configuration
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        OpPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new ops performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        OpPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Run all tensor operation performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        self.test_all_operations_with_shapes(&super::super::core::generate_test_shapes())
    }

    /// Run all tensor operation performance tests with custom shapes
    pub fn test_all_operations_with_shapes(
        &mut self,
        test_shapes: &[Vec<usize>],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(70));
        println!("    COMPREHENSIVE TENSOR OPERATIONS PERFORMANCE BENCHMARK");
        println!("{}", "=".repeat(70));
        println!(
            "Testing with {} iterations per test",
            self.tester.config.iterations
        );
        println!("{}", "=".repeat(70));

        let mut all_results = Vec::new();

        // Test all basic operations using individual testers
        let mut add_tester = AddPerformanceTester::with_config(self.tester.config.clone());
        all_results.extend(add_tester.test_all_operations_with_shapes(test_shapes));

        let mut sub_tester = SubPerformanceTester::with_config(self.tester.config.clone());
        all_results.extend(sub_tester.test_all_operations_with_shapes(test_shapes));

        let mut mul_tester = MulPerformanceTester::with_config(self.tester.config.clone());
        all_results.extend(mul_tester.test_all_operations_with_shapes(test_shapes));

        let mut div_tester = DivPerformanceTester::with_config(self.tester.config.clone());
        all_results.extend(div_tester.test_all_operations_with_shapes(test_shapes));

        let mut matmul_tester = MatmulPerformanceTester::with_config(self.tester.config.clone());
        // Create shape pairs for matmul from test shapes
        let matmul_shape_pairs = create_matmul_shape_pairs(test_shapes);
        all_results.extend(matmul_tester.test_all_operations_with_shapes(&matmul_shape_pairs));

        // Add all results to our tester for summary and saving
        self.tester.add_results(all_results.clone());

        // Save results using serialization system
        if let Err(e) = self.tester.save_results("tensor_ops_performance.json") {
            eprintln!("Failed to save results: {}", e);
        }

        // Print comprehensive summary
        self.tester.print_summary();

        all_results
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

impl Default for OpPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ops_performance_tester() {
        let tester = OpPerformanceTester::new();

        // Test that we can create individual operation testers
        let _add_tester = AddPerformanceTester::new();
        let _sub_tester = SubPerformanceTester::new();
        let _mul_tester = MulPerformanceTester::new();
        let _div_tester = DivPerformanceTester::new();
        let _matmul_tester = MatmulPerformanceTester::new();

        // Test that the main tester can be created
        assert_eq!(tester.tester().config.iterations, 1000);
    }
}
