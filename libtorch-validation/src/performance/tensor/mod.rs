//! Tensor performance testing module
//!
//! Provides comprehensive performance benchmarking for all tensor operations,
//! transforms, indexing, and reduction operations against LibTorch reference
//! implementation.
//!
//! ## Module Structure
//!
//! - `ops`: Basic tensor operations (add, sub, mul, div, matmul)
//! - `transform`: Shape and layout transformations
//! - `indexing`: Tensor indexing and selection operations  
//! - `reductions`: Reduction and aggregation operations
//!
//! ## Usage
//!
//! ```rust,ignore
//! use libtorch_validation::performance::tensor::{
//!     ops::OpPerformanceTester,
//!     // Future modules:
//!     // transform::TransformPerformanceTester,
//!     // indexing::IndexingPerformanceTester,
//!     // reductions::ReductionPerformanceTester,
//! };
//!
//! // Test all tensor operations
//! let mut ops_tester = OpPerformanceTester::new();
//! let results = ops_tester.test_all_operations();
//! ```

pub mod ops;
pub mod transform;

// Re-export for convenience
pub use ops::OpPerformanceTester;
pub use transform::TransformPerformanceTester;
