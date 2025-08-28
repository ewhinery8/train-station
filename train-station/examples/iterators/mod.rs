//! Iterator Examples for Train Station
//!
//! This module contains comprehensive examples demonstrating tensor iterator functionality
//! and element-wise processing patterns in Train Station:
//! - Basic element iteration and transformation
//! - Advanced iterator patterns and chaining
//! - Performance optimization and memory-efficient processing
//!
//! These examples showcase the powerful iterator API that provides seamless integration
//! with Rust's standard library while maintaining full tensor operation capabilities.
//!
//! # Learning Objectives
//!
//! - Master tensor element iteration with full gradient tracking
//! - Learn advanced iterator patterns for data processing
//! - Understand performance optimization techniques
//! - Implement memory-efficient element-wise operations
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge and iterator concepts
//! - Understanding of tensor basics (see getting_started/tensor_basics.rs)
//! - Familiarity with functional programming patterns
//! - Knowledge of gradient tracking and autograd
//!
//! # Key Features Demonstrated
//!
//! - **Standard Library Integration**: Full compatibility with Rust's iterator traits
//! - **Gradient Tracking**: Automatic gradient propagation through element operations
//! - **Performance Optimization**: Zero-copy views with shared memory
//! - **Memory Efficiency**: Adaptive processing for large tensors
//! - **SIMD Compatibility**: Leveraging existing optimized tensor operations
//!
//! # Usage
//!
//! ```bash
//! cargo run --example element_iteration
//! cargo run --example advanced_patterns
//! cargo run --example performance_optimization
//! ```
//!
//! # Architecture Overview
//!
//! The iterator system provides:
//! - **TensorElementIterator**: Core iterator implementation with all standard traits
//! - **Element Views**: True tensor views of shape [1] for each element
//! - **Gradient Functions**: Integration with existing autograd system
//! - **Collection**: Optimized reconstruction from element views
//!
//! # Performance Characteristics
//!
//! - **View Creation**: O(1) per element with true zero-copy views
//! - **Memory Overhead**: ~64 bytes per view tensor (no data copying)
//! - **SIMD Operations**: Full utilization of existing optimizations
//! - **Gradient Tracking**: True gradient flow with element-level accumulation

pub mod advanced_patterns;
pub mod element_iteration;
pub mod performance_optimization;

pub use advanced_patterns::*;
pub use element_iteration::*;
pub use performance_optimization::*;
