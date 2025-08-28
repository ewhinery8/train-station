//! Tensor transformation operations
//!
//! This module provides shape transformation operations for tensors that modify
//! the layout and dimensionality of tensor data while preserving the underlying
//! values. These operations are essential for preparing tensors for various
//! machine learning operations and data processing tasks.
//!
//! # Operations
//!
//! ## View Operations (Zero-Copy)
//! * `view()` - Reshape tensor as a view when possible (zero-copy)
//! * `reshape()` - Change tensor dimensions, copying if necessary
//! * `transpose()` - Swap two dimensions (zero-copy view)
//! * `permute()` - Reorder dimensions according to permutation (zero-copy view)
//! * `squeeze()` - Remove dimensions of size 1 (zero-copy view)
//! * `unsqueeze()` - Add dimension of size 1 at specified position (zero-copy view)
//! * `flatten()` - Convert to 1D tensor (zero-copy view when possible)
//!
//! ## Copy Operations (When Necessary)
//! * `contiguous()` - Ensure tensor is in contiguous memory layout
//! * `cat()` - Concatenate tensors along specified dimension
//! * `stack()` - Stack tensors along new dimension
//! * `split()` - Split tensor into multiple tensors along dimension
//!
//! # Memory Efficiency
//!
//! The module prioritizes memory efficiency by using views (zero-copy operations)
//! whenever mathematically possible. Operations only create copies when the
//! desired layout cannot be achieved through view transformations.
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create a 2x3 tensor
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//!
//! // Transpose to 3x2 (zero-copy view)
//! let transposed = tensor.transpose(0, 1);
//! assert_eq!(transposed.shape().dims, vec![3, 2]);
//! assert!(!transposed.is_contiguous()); // View is not contiguous
//!
//! // Reshape to 1D (zero-copy when possible)
//! let flattened = tensor.flatten();
//! assert_eq!(flattened.shape().dims, vec![6]);
//! ```
//!
//! ```
//! use train_station::Tensor;
//!
//! // Concatenate tensors along dimension 0
//! let t1 = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]).unwrap();
//! let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]).unwrap();
//! let concatenated = Tensor::cat(&[t1.clone(), t2.clone()], 0);
//! assert_eq!(concatenated.shape().dims, vec![2, 2]);
//! ```
//!
//! ```
//! use train_station::Tensor;
//!
//! // Stack tensors along new dimension
//! let t1 = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
//! let t2 = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
//! let stacked = Tensor::stack(&[t1.clone(), t2.clone()], 0);
//! assert_eq!(stacked.shape().dims, vec![2, 2]);
//! ```
//!
//! # Performance Characteristics
//!
//! - **View Operations**: O(1) time complexity, zero memory allocation
//! - **Copy Operations**: O(n) time complexity, requires memory allocation
//! - **Contiguous Operations**: Automatically optimized for SIMD operations
//! - **Stride-Aware**: All operations work correctly with non-contiguous tensors
//!
//! # Gradient Tracking
//!
//! All transformation operations support automatic gradient tracking through
//! the GradTrack system when `requires_grad` is enabled. View operations
//! maintain gradient connectivity, while copy operations create new gradient
//! computation graphs as needed.
//!
//! # Thread Safety
//!
//! All transformation operations are thread-safe and can be used concurrently
//! across multiple threads. View operations share underlying data safely,
//! while copy operations create independent tensor instances.

pub mod cat;
pub mod contiguous;
pub mod flatten;
pub mod permute;
pub mod reshape;
pub mod split;
pub mod squeeze;
pub mod stack;
pub mod transpose;
pub mod unsqueeze;
pub mod view;
