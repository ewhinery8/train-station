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

pub mod ops;
pub mod transform;

// Re-export for convenience
pub use ops::OpPerformanceTester;
pub use transform::TransformPerformanceTester;
