//! Tensor initialization operations
//!
//! This module provides comprehensive tensor initialization methods for creating tensors
//! with various initial values and patterns. The initialization methods are designed
//! to be efficient, flexible, and follow PyTorch conventions for familiarity.
//!
//! The initialization module supports creating tensors with specific values, random
//! distributions, and from existing data sources. All methods are optimized for
//! performance with SIMD operations where applicable and provide full gradtrack support.
//!
//! # Key Features
//!
//! - **Basic Initialization**: Create tensors with zeros, ones, or filled values
//! - **Random Initialization**: Generate tensors with various random distributions
//! - **Data-Based Initialization**: Create tensors from existing data sources
//! - **Performance Optimized**: SIMD-optimized operations for maximum speed
//! - **GradTrack Support**: Full gradient tracking for all initialization methods
//! - **PyTorch Compatibility**: Follows PyTorch conventions for method names and behavior
//!
//! # Performance Characteristics
//!
//! - **Basic Initialization**: O(n) with SIMD-optimized memory operations
//! - **Random Initialization**: O(n) with efficient random number generation
//! - **Data-Based Initialization**: O(n) with optimized memory copying
//! - **Memory Efficiency**: Minimal overhead for tensor creation
//! - **SIMD Optimization**: Vectorized operations for large tensors
//!
//! # Organization
//!
//! The initialization module is organized into specialized submodules:
//!
//! - **`basic`**: Basic initialization methods including zeros, ones, and fill operations
//! - **`random`**: Random number generation with various distributions (normal, uniform, etc.)
//! - **`data`**: Data-based initialization from slices, vectors, and other data sources
//!
//! # Examples
//!
//! ## Basic Tensor Creation
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create tensors with different initialization patterns
//! let zeros = Tensor::zeros(vec![2, 3]);
//! let ones = Tensor::ones(vec![2, 3]);
//! let mut filled = Tensor::new(vec![2, 3]);
//! filled.fill(5.0);
//!
//! assert_eq!(zeros.shape().dims, vec![2, 3]);
//! assert_eq!(ones.shape().dims, vec![2, 3]);
//! assert_eq!(filled.shape().dims, vec![2, 3]);
//!
//! // Verify initialization values
//! assert_eq!(zeros.get(&[0, 0]), 0.0);
//! assert_eq!(ones.get(&[0, 0]), 1.0);
//! assert_eq!(filled.get(&[0, 0]), 5.0);
//! ```
//!
//! ## Random Tensor Generation
//!
//! ```
//! use train_station::Tensor;
//!
//! // Generate random tensors with normal distribution
//! let normal = Tensor::randn(vec![2, 3], Some(42));
//!
//! assert_eq!(normal.shape().dims, vec![2, 3]);
//!
//! // Random values should be different from zeros/ones
//! assert!(normal.get(&[0, 0]) != 0.0);
//! ```
//!
//! ## Data-Based Initialization
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create tensors from existing data
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let tensor = Tensor::from_slice(&data, vec![2, 3]).unwrap();
//!
//! assert_eq!(tensor.shape().dims, vec![2, 3]);
//! assert_eq!(tensor.get(&[0, 0]), 1.0);
//! assert_eq!(tensor.get(&[1, 2]), 6.0);
//! ```
//!
//! ## Initialization with Gradient Tracking
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create tensors with gradient tracking enabled
//! let zeros = Tensor::zeros(vec![2, 2]).with_requires_grad();
//! let ones = Tensor::ones(vec![2, 2]).with_requires_grad();
//!
//! // Perform operations and compute gradients
//! let mut result = zeros.add_tensor(&ones);
//! result.backward(None);
//!
//! // Verify gradients are computed
//! let grad = zeros.grad_by_value().expect("gradient missing");
//! assert_eq!(grad.shape().dims, vec![2, 2]);
//! ```
//!
//! # Design Principles
//!
//! - **PyTorch Compatibility**: Method names and behavior follow PyTorch conventions
//! - **Performance First**: Optimized implementations with SIMD acceleration
//! - **Memory Safety**: Safe operations with proper bounds checking
//! - **GradTrack Integration**: Seamless integration with the gradient tracking system
//! - **Type Safety**: Strong typing for all initialization parameters
//! - **Zero-Copy Operations**: Efficient data-based initialization when possible

pub mod basic;
pub mod data;
pub mod random;

// Re-export commonly used initialization methods for convenience
// These methods are implemented on Tensor, so they're available through Tensor::
