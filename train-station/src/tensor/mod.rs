//! Tensor module for high-performance multi-dimensional data structures
//!
//! This module provides the foundational building blocks for tensor operations,
//! organized into specialized submodules for maximum performance and maintainability.
//! The tensor system is designed for zero-cost abstractions with SIMD optimization
//! and comprehensive automatic differentiation support.
//!
//! # Organization
//!
//! The tensor module is organized into specialized submodules:
//! - **`core`**: Main `Tensor` struct with memory management and operator overloading
//! - **`shape`**: Dimension management, stride calculation, and broadcasting
//! - **`ops`**: Mathematical operations (add, sub, mul, div, matmul) with SIMD optimization
//! - **`transform`**: Shape transformations (reshape, permute, transpose, cat, stack)
//! - **`indexing`**: Tensor indexing and selection operations (select, gather, masked_fill)
//! - **`reductions`**: Reduction operations (sum, mean, min, max, std, var)
//! - **`init`**: Tensor initialization methods (zeros, ones, randn, from_slice)
//!
//! # Key Features
//!
//! - **Zero-Cost Abstractions**: Minimal overhead for tensor operations
//! - **SIMD Optimization**: AVX2 optimizations for x86_64 architectures
//! - **Memory Efficiency**: Optimized alignment and layout strategies
//! - **GradTrack Integration**: Built-in gradient tracking and computation
//! - **Operator Overloading**: Natural mathematical expressions (+, -, *, /, +=, -=, *=, /=)
//! - **Thread Safety**: Send + Sync implementation for concurrent usage
//! - **Device Support**: CPU and future CUDA device placement
//! - **View Tensors**: Zero-copy tensor views with shared memory
//!
//! # Performance Characteristics
//!
//! - **Memory Overhead**: ~64 bytes per tensor (excluding data)
//! - **SIMD Alignment**: 32-byte alignment for AVX2 operations
//! - **Cache Optimization**: Cache-line alignment for large tensors
//! - **View Efficiency**: Zero-copy views with shared memory management
//! - **Operator Performance**: Zero-cost operator overloading for mathematical expressions
//! - **Thread Safety**: Lock-free operations with atomic ID generation
//!
//! # Examples
//!
//! ## Basic Tensor Operations
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create tensors with different configurations
//! let tensor = Tensor::new(vec![2, 3, 4]);
//! let tensor_with_grad = Tensor::ones(vec![10, 10]).with_requires_grad();
//!
//! // Access tensor properties
//! assert_eq!(tensor.size(), 24);
//! assert_eq!(tensor.shape().dims, vec![2, 3, 4]);
//! assert!(tensor.is_contiguous());
//! ```
//!
//! ## Operator Overloading
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create tensors for operations
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//!
//! // Tensor operations with operators
//! let result = a.clone() + b.clone();                    // Tensor addition
//! let result = a.clone() * b.clone();                    // Element-wise multiplication
//! let result = a.clone() - b.clone();                    // Tensor subtraction
//! let result = a.clone() / b.clone();                    // Element-wise division
//!
//! // Scalar operations
//! let result = a.clone() + 5.0;                          // Tensor + scalar
//! let result = 5.0 + a.clone();                          // Scalar + tensor
//! let result = a.clone() * 3.0;                          // Tensor * scalar
//! let result = 3.0 * a.clone();                          // Scalar * tensor
//!
//! // Compound expressions
//! let result = (a.clone() + b.clone()) * 2.0 - 1.0;      // Complex mathematical expressions
//!
//! // Assignment operators
//! let mut c = a.clone();
//! c += b.clone();                                        // In-place addition
//! c *= 2.0;                                              // In-place scalar multiplication
//!
//! // Negation
//! let result = -a;                                       // Negate all elements
//! ```
//!
//! ## Automatic Differentiation
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create tensors with gradient tracking
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap().with_requires_grad();
//! let y = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]).unwrap().with_requires_grad();
//!
//! // Perform operations (gradients are automatically tracked)
//! let z = x.clone() * y.clone() + 2.0;
//! let mut loss = z.sum();
//!
//! // Compute gradients
//! loss.backward(None);
//!
//! // Access gradients (gradients are computed and stored)
//! // Note: Gradient availability depends on the computation graph
//! let x_grad = x.grad();
//! let y_grad = y.grad();
//! ```
//!
//! # Thread Safety
//!
//! All tensor operations are thread-safe and implement `Send + Sync`. Tensors can be
//! safely shared between threads for concurrent read access. Write operations should
//! be synchronized externally if multiple threads need to modify the same tensor.
//!
//! # Design Principles
//!
//! - **Performance First**: Every design decision optimized for speed
//! - **Memory Safety**: RAII patterns with justified unsafe usage
//! - **Zero Dependencies**: Only standard library dependencies
//! - **SIMD Ready**: Optimized for vectorized operations
//! - **Future Proof**: Foundation for advanced ML operations
//! - **Natural API**: Operator overloading for intuitive mathematical expressions
//! - **Modular Organization**: Specialized submodules for maintainability
//! - **Comprehensive Testing**: 100% coverage with FFI mathematical validation

pub(crate) mod core;
pub(crate) mod indexing;
pub(crate) mod init;
pub(crate) mod iterator;
pub(crate) mod ops;
pub(crate) mod reductions;
pub(crate) mod transform;

pub(crate) use core::MemoryLayout;
pub use core::{Shape, Tensor};
