//! Transform operations performance benchmarking
//!
//! This module provides performance testing for tensor transformation operations
//! against LibTorch, including reshape, permute, transpose, view, cat, and stack.
//!
//! ## Module Structure
//!
//! - `flatten`: Flatten operations (tensor to 1D)
//! - `transpose`: Transpose operations (2D tensor transposition)
//! - `permute`: Permute operations (3D tensor permutation)
//! - `view`: View operations (reshaping without copying)
//! - `cat`: Concatenation operations (joining tensors)
//! - `stack`: Stack operations (stacking along new dimensions)

pub mod cat;
pub mod flatten;
pub mod permute;
pub mod stack;
pub mod transpose;
pub mod view;

use super::super::core::{PerformanceConfig, PerformanceResult, PerformanceTester};

// Re-export individual transform testers for convenience
pub use cat::CatPerformanceTester;
pub use flatten::FlattenPerformanceTester;
pub use permute::PermutePerformanceTester;
pub use stack::StackPerformanceTester;
pub use transpose::TransposePerformanceTester;
pub use view::ViewPerformanceTester;

/// Comprehensive transform operations performance tester
pub struct TransformPerformanceTester {
    pub tester: PerformanceTester,
}

impl TransformPerformanceTester {
    /// Create a new transform performance tester with default configuration
    pub fn new() -> Self {
        let config = PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        };

        TransformPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Create a new transform performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        TransformPerformanceTester {
            tester: PerformanceTester::with_config(config),
        }
    }

    /// Run all transform operation performance tests
    pub fn test_all_operations(&mut self) -> Vec<PerformanceResult> {
        self.test_all_operations_with_shapes(&super::super::core::generate_test_shapes())
    }

    /// Run all transform operation performance tests with custom shapes
    pub fn test_all_operations_with_shapes(
        &mut self,
        test_shapes: &[Vec<usize>],
    ) -> Vec<PerformanceResult> {
        println!("\n{}", "=".repeat(70));
        println!("    COMPREHENSIVE TENSOR TRANSFORM PERFORMANCE BENCHMARK");
        println!("{}", "=".repeat(70));
        println!(
            "Testing with {} iterations per test",
            self.tester.config.iterations
        );
        println!("{}", "=".repeat(70));

        let mut all_results = Vec::new();

        // Test all transform operations using individual testers
        let mut cat_tester = CatPerformanceTester::with_config(self.tester.config.clone());
        all_results.extend(cat_tester.test_all_operations_with_shapes(test_shapes));

        let mut flatten_tester = FlattenPerformanceTester::with_config(self.tester.config.clone());
        all_results.extend(flatten_tester.test_all_operations_with_shapes(test_shapes));

        let mut transpose_tester =
            TransposePerformanceTester::with_config(self.tester.config.clone());
        all_results.extend(transpose_tester.test_all_operations_with_shapes(test_shapes));

        let mut permute_tester = PermutePerformanceTester::with_config(self.tester.config.clone());
        all_results.extend(permute_tester.test_all_operations_with_shapes(test_shapes));

        let mut view_tester = ViewPerformanceTester::with_config(self.tester.config.clone());
        all_results.extend(view_tester.test_all_operations_with_shapes(test_shapes));

        let mut stack_tester = StackPerformanceTester::with_config(self.tester.config.clone());
        all_results.extend(stack_tester.test_all_operations_with_shapes(test_shapes));

        // Add all results to our tester for summary and saving
        self.tester.add_results(all_results.clone());

        // Save results using serialization system
        if let Err(e) = self
            .tester
            .save_results("tensor_transform_performance.json")
        {
            eprintln!("Failed to save transform results: {}", e);
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

impl Default for TransformPerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_performance_tester() {
        let tester = TransformPerformanceTester::new();

        // Test that we can create individual transform testers
        let _flatten_tester = FlattenPerformanceTester::new();
        let _transpose_tester = TransposePerformanceTester::new();
        let _permute_tester = PermutePerformanceTester::new();
        let _view_tester = ViewPerformanceTester::new();
        let _cat_tester = CatPerformanceTester::new();
        let _stack_tester = StackPerformanceTester::new();

        // Test that the main tester can be created
        assert_eq!(tester.tester().config.iterations, 1000);
    }
}
