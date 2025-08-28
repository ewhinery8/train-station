//! Reduction operations for tensors
//!
//! Provides comprehensive tensor reduction operations following PyTorch conventions with
//! GradTrack support and optimized implementations for various reduction patterns.
//!
//! # Key Features
//!
//! - **Statistical Reductions**: `sum()`, `mean()`, `std()`, `var()` - Statistical aggregation operations
//! - **Extremal Reductions**: `min()`, `max()` - Minimum and maximum value operations
//! - **Index Reductions**: `argmin()`, `argmax()` - Index-based extremal operations
//! - **Norm Reductions**: `norm()` - Vector and matrix norm computations
//! - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
//! - **Dimension Reduction**: Support for reducing along specific dimensions
//! - **Performance Optimization**: SIMD-optimized implementations where applicable
//!
//! # Mathematical Properties
//!
//! The reduction operations have the following properties:
//! - **Sum**: `sum(x)` - Sum of all elements or along specified dimensions
//! - **Mean**: `mean(x)` - Arithmetic mean of elements or along specified dimensions
//! - **Variance**: `var(x)` - Sample variance with optional unbiased estimation
//! - **Standard Deviation**: `std(x)` - Square root of variance
//! - **Minimum/Maximum**: `min(x)`, `max(x)` - Extremal values with optional dimension reduction
//! - **Argmin/Argmax**: `argmin(x)`, `argmax(x)` - Indices of extremal values
//! - **Norm**: `norm(x)` - Vector/matrix norms (L1, L2, Frobenius)
//!
//! # Performance Characteristics
//!
//! - **SIMD Optimization**: Vectorized implementations for statistical reductions
//! - **Memory Efficiency**: Optimized memory access patterns for large tensors
//! - **Dimension Handling**: Efficient reduction along arbitrary dimensions
//! - **GradTrack Optimization**: Efficient automatic differentiation with gradient computation
//! - **Numerical Stability**: Robust implementations for edge cases and extreme values
//!
//! # Examples
//!
//! ## Basic Reductions
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//!
//! // Sum all elements
//! let total = tensor.sum();
//! assert_eq!(total.value(), 10.0);
//!
//! // Mean of all elements
//! let average = tensor.mean();
//! assert_eq!(average.value(), 2.5);
//!
//! // Maximum value
//! let max_val = tensor.max();
//! assert_eq!(max_val.value(), 4.0);
//! ```
//!
//! ## Dimension Reductions
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//!
//! // Sum along first dimension (rows)
//! let row_sums = tensor.sum_dims(&[0], false);
//! assert_eq!(row_sums.shape().dims, vec![2]);
//! assert_eq!(row_sums.get(&[0]), 4.0); // 1.0 + 3.0
//! assert_eq!(row_sums.get(&[1]), 6.0); // 2.0 + 4.0
//!
//! // Mean along second dimension (columns)
//! let col_means = tensor.mean_dims(&[1], false);
//! assert_eq!(col_means.shape().dims, vec![2]);
//! assert_eq!(col_means.get(&[0]), 1.5); // (1.0 + 2.0) / 2
//! assert_eq!(col_means.get(&[1]), 3.5); // (3.0 + 4.0) / 2
//! ```
//!
//! ## Statistical Operations
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
//!
//! // Variance
//! let variance = tensor.var();
//! assert!((variance.value() - 1.25).abs() < 1e-6);
//!
//! // Standard deviation
//! let std_dev = tensor.std();
//! assert!((std_dev.value() - 1.118034).abs() < 1e-6);
//!
//! // Norm
//! let norm_val = tensor.norm();
//! assert!((norm_val.value() - 5.477226).abs() < 1e-6);
//! ```

pub mod argmax;
pub mod argmin;
pub mod max;
pub mod mean;
pub mod min;
pub mod norm;
pub mod std;
pub mod sum;
pub mod var;
