//! Performance benchmarking module for LibTorch comparison
//!
//! This module provides comprehensive performance comparison functionality against
//! LibTorch, helping to measure and analyze the performance characteristics of
//! tensor operations across different shapes and scenarios.
//!
//! ## Structure
//!
//! The performance module mirrors the validation structure to ensure consistency:
//! - `core`: Core performance testing framework and utilities
//! - `tensor/ops`: Individual tensor operation performance tests (one file per operation)
//! - Future: `tensor/transform`, `tensor/indexing`, etc.
//!
//! ## Usage
//!
//! ```rust,ignore
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use libtorch_validation::performance::{
//!     PerformanceTester, PerformanceConfig,
//!     tensor::ops::{AddPerformanceTester, SubPerformanceTester}
//! };
//!
//! // Test individual operations
//! let mut add_tester = AddPerformanceTester::new();
//! let results = add_tester.test_all_operations();
//! add_tester.tester().save_results("add_performance.json")?;
//!
//! // Test all operations
//! let config = PerformanceConfig {
//!     iterations: 1000,
//!     verbose: true,
//!     ..Default::default()
//! };
//!
//! let mut tester = PerformanceTester::with_config(config);
//! tester.benchmark_operation(
//!     "add_tensor",
//!     &[512, 512],
//!     || our_add_implementation(),
//!     || libtorch_add_implementation(),
//! );
//!
//! tester.save_results("performance_results.json")?;
//! tester.print_summary();
//! # Ok(())
//! # }
//! ```

pub mod core;
pub mod tensor;

pub use core::{
    create_test_tensor, generate_test_shapes, generate_test_shapes_with_config, PerformanceConfig,
    PerformanceResult, PerformanceTester, TestPattern, BATCH_SIZES, TEST_DIMS,
};

// Re-export tensor operation testers for convenience
pub use tensor::ops::{
    AddPerformanceTester, DivPerformanceTester, MatmulPerformanceTester, MulPerformanceTester,
    OpPerformanceTester, SubPerformanceTester,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_module_exports() {
        // Test that all core exports are available
        let _config = PerformanceConfig::default();
        let _tester = PerformanceTester::new();
        let _shapes = generate_test_shapes();
        let _tensor = create_test_tensor(&[2, 2], TestPattern::Ones);

        // Test that operation testers can be created
        let _add_tester = AddPerformanceTester::new();
        let _sub_tester = SubPerformanceTester::new();
        let _mul_tester = MulPerformanceTester::new();
        let _div_tester = DivPerformanceTester::new();
        let _matmul_tester = MatmulPerformanceTester::new();
    }
}
