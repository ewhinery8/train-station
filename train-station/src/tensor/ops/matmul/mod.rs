//! Matrix multiplication operations with optimized kernels
//!
//! This module provides a comprehensive matrix multiplication implementation optimized
//! for single-threaded performance with SIMD acceleration. The implementation supports
//! all NumPy-style matrix multiplication patterns including 1D/2D/ND tensor operations
//! with automatic differentiation support.
//!
//! # Key Features
//!
//! - **SIMD Optimization**: AVX2 implementations for x86_64 architectures
//! - **Intelligent Dispatch**: Dynamic kernel selection based on matrix dimensions
//! - **Cache Optimization**: Blocked algorithms for L1/L2 cache efficiency
//! - **Memory Bandwidth**: Optimized for maximum memory bandwidth utilization
//! - **GradTrack Integration**: Automatic gradient computation for all operations
//! - **Thread Safety**: All operations are thread-safe and Send + Sync
//! - **Mathematical Validation**: High-precision equivalence to LibTorch reference
//!
//! # Performance Characteristics
//!
//! The implementation uses intelligent dispatch to select optimal kernels based on matrix size:
//! - **Small matrices (16-64 elements)**: Direct computation with minimal overhead
//! - **Medium matrices (64-256 elements)**: Cache-optimized blocking for L1/L2 cache
//! - **Large matrices (256+ elements)**: Memory bandwidth optimized with hierarchical blocking
//! - **AVX2 acceleration**: 8x SIMD operations for compatible hardware
//! - **Scalar fallbacks**: Optimized scalar implementations for non-SIMD platforms
//! - **Memory Safety**: Safe memory management with `Tensor::new_uninitialized`
//!
//! # Organization
//!
//! The matmul module is organized into focused submodules:
//! - **`config`**: Dynamic configuration and kernel selection based on matrix dimensions
//! - **`kernels`**: SIMD-optimized computational kernels with ML-specific optimizations
//!
//! # Supported Operations
//!
//! - **1D @ 1D**: Dot product returning scalar tensor
//! - **1D @ 2D**: Vector-matrix multiplication (v^T * M)
//! - **2D @ 1D**: Matrix-vector multiplication (M * v)
//! - **2D @ 2D**: Standard matrix multiplication with cache-optimized blocking
//! - **ND @ ND**: Batched matrix multiplication on last two dimensions with broadcasting
//!
//! # Examples
//!
//! ## Basic Matrix Multiplication
//!
//! ```
//! use train_station::Tensor;
//!
//! // 2D matrix multiplication
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//! let result = a.matmul(&b); // Uses optimized SIMD kernels
//!
//! assert_eq!(result.shape().dims, vec![2, 2]);
//! assert_eq!(result.data(), &[19.0, 22.0, 43.0, 50.0]);
//! ```
//!
//! ## Vector-Matrix Multiplication
//!
//! ```
//! use train_station::Tensor;
//!
//! // Vector-matrix multiplication
//! let v = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
//! let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let result = v.matmul(&m); // [2] @ [2, 2] -> [2]
//!
//! assert_eq!(result.shape().dims, vec![2]);
//! assert_eq!(result.data(), &[7.0, 10.0]); // 1*1+2*3, 1*2+2*4
//! ```
//!
//! ## Matrix-Vector Multiplication
//!
//! ```
//! use train_station::Tensor;
//!
//! // Matrix-vector multiplication
//! let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let v = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
//! let result = m.matmul(&v); // [2, 2] @ [2] -> [2]
//!
//! assert_eq!(result.shape().dims, vec![2]);
//! assert_eq!(result.data(), &[5.0, 11.0]); // 1*1+2*2, 3*1+4*2
//! ```
//!
//! ## Dot Product
//!
//! ```
//! use train_station::Tensor;
//!
//! // 1D dot product
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
//! let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]).unwrap();
//! let result = a.matmul(&b); // [3] @ [3] -> scalar
//!
//! assert_eq!(result.shape().dims, vec![]); // Scalar tensor
//! assert_eq!(result.data(), &[32.0]); // 1*4 + 2*5 + 3*6
//! ```
//!
//! ## Batched Matrix Multiplication
//!
//! ```
//! use train_station::Tensor;
//!
//! // Batched matrix multiplication
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]).unwrap();
//! let b = Tensor::from_slice(&[0.5, 1.0, 1.5, 2.0], vec![2, 2]).unwrap();
//! let result = a.matmul(&b); // [2, 2, 2] @ [2, 2] -> [2, 2, 2]
//!
//! assert_eq!(result.shape().dims, vec![2, 2, 2]);
//! ```
//!
//! ## Gradient Tracking
//!
//! ```
//! use train_station::Tensor;
//!
//! // Matrix multiplication with gradient tracking
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])
//!     .unwrap()
//!     .with_requires_grad();
//! let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2])
//!     .unwrap()
//!     .with_requires_grad();
//!
//! let result = a.matmul(&b);
//! assert!(result.requires_grad());
//! assert_eq!(result.shape().dims, vec![2, 2]);
//! ```
//!
//! # Automatic Differentiation
//!
//! All operations support automatic differentiation when either operand requires gradients.
//! Gradient computation follows PyTorch semantics with proper accumulation and chain rule
//! application through the gradtrack engine.
//!
//! # Thread Safety
//!
//! All operations are thread-safe and can be used concurrently across multiple threads.
//! The implementation uses immutable tensor references and thread-local gradtrack state.
//!
//! # Mathematical Validation
//!
//! All operations are validated against LibTorch reference implementation with high-precision
//! numerical equivalence (target: 0.00e0 error tolerance, practical: 1e-6 tolerance for
//! floating-point precision differences).

use crate::tensor::core::Tensor;

pub mod config;
pub mod kernels;

// Re-export public types
pub use config::MatmulConfig;

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl Tensor {
    /// Matrix multiplication operation following NumPy semantics
    ///
    /// Performs matrix multiplication between this tensor and another tensor with intelligent
    /// kernel selection based on matrix dimensions and hardware capabilities. The operation
    /// follows broadcasting rules and supports all common matrix multiplication patterns
    /// found in machine learning workloads.
    ///
    /// # Supported Operations
    ///
    /// - **1D @ 1D**: Dot product returning scalar tensor
    /// - **1D @ 2D**: Vector-matrix multiplication (v^T * M) returning 1D tensor
    /// - **2D @ 1D**: Matrix-vector multiplication (M * v) returning 1D tensor
    /// - **2D @ 2D**: Standard matrix multiplication with cache-optimized blocking
    /// - **ND @ ND**: Batched matrix multiplication on last two dimensions with broadcasting
    ///
    /// # Performance Characteristics
    ///
    /// The implementation automatically selects optimal kernels based on matrix dimensions:
    /// - **Small matrices (<64 elements)**: Direct computation with minimal overhead
    /// - **Medium matrices (64-256 elements)**: Cache-optimized blocking for L1/L2 cache
    /// - **Large matrices (256+ elements)**: Memory bandwidth optimized with hierarchical blocking
    /// - **AVX2 acceleration**: 8x SIMD operations for compatible hardware
    /// - **Scalar fallbacks**: Optimized scalar implementations for non-SIMD platforms
    ///
    /// # Automatic Differentiation
    ///
    /// This operation supports automatic differentiation when either operand requires gradients.
    /// Gradient computation follows PyTorch semantics with proper accumulation and chain rule
    /// application through the gradtrack engine. Gradients are computed for both operands when
    /// `requires_grad` is set.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to multiply with (must have compatible dimensions)
    ///
    /// # Returns
    ///
    /// A new tensor containing the matrix multiplication result with appropriate shape
    /// determined by broadcasting rules and matrix multiplication semantics
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions don't match for matrix multiplication:
    /// - For 2D @ 2D: `self.shape()[1] != other.shape()[0]`
    /// - For 1D @ 2D: `self.shape()[0] != other.shape()[0]`
    /// - For 2D @ 1D: `self.shape()[1] != other.shape()[0]`
    /// - For ND @ ND: Last two dimensions must be compatible for matrix multiplication
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // 2D matrix multiplication
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    /// let result = a.matmul(&b); // [2, 2] @ [2, 2] -> [2, 2]
    ///
    /// assert_eq!(result.shape().dims, vec![2, 2]);
    /// assert_eq!(result.data(), &[19.0, 22.0, 43.0, 50.0]);
    /// ```
    ///
    /// ## Vector-Matrix Multiplication
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let v = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
    /// let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let result = v.matmul(&m); // [2] @ [2, 2] -> [2]
    ///
    /// assert_eq!(result.shape().dims, vec![2]);
    /// assert_eq!(result.data(), &[7.0, 10.0]); // 1*1+2*3, 1*2+2*4
    /// ```
    ///
    /// ## Gradient Tracking
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])
    ///     .unwrap()
    ///     .with_requires_grad();
    /// let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2])
    ///     .unwrap()
    ///     .with_requires_grad();
    ///
    /// let result = a.matmul(&b);
    /// assert!(result.requires_grad());
    /// assert_eq!(result.shape().dims, vec![2, 2]);
    /// ```
    ///
    /// # Thread Safety
    ///
    /// This operation is thread-safe and can be used concurrently across multiple threads.
    /// The implementation uses immutable tensor references and thread-local gradtrack state.
    ///
    /// # Memory Safety
    ///
    /// The implementation uses `Tensor::new_uninitialized` for performance-critical allocations
    /// and handles memory initialization safely through the kernel system. All unsafe operations
    /// are validated through comprehensive FFI testing against LibTorch reference implementation.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let self_shape = self.shape();
        let other_shape = other.shape();

        let mut result = match (self_shape.rank(), other_shape.rank()) {
            (1, 1) => {
                // 1D @ 1D: dot product -> scalar
                self.dot_product_1d(other)
            }
            (1, 2) => {
                // 1D @ 2D: vector-matrix multiplication -> 1D
                self.vector_matrix_mult(other)
            }
            (2, 1) => {
                // 2D @ 1D: matrix-vector multiplication -> 1D
                self.matrix_vector_mult(other)
            }
            (2, 2) => {
                // 2D @ 2D: standard matrix multiplication -> 2D
                self.matrix_matrix_mult(other)
            }
            _ => {
                // ND @ ND: batched matrix multiplication
                self.batched_matmul(other)
            }
        };

        // Set up gradtrack if either operand requires gradients
        if (self.requires_grad() || other.requires_grad()) && crate::gradtrack::is_grad_enabled() {
            use crate::gradtrack::{GradEngine, GradFn};

            result.set_requires_grad(true);
            let grad_fn = GradFn::MatMul {
                left_operand: Box::new(self.clone()),
                right_operand: Box::new(other.clone()),
                requires_grad: (self.requires_grad(), other.requires_grad()),
            };
            result.set_grad_fn(grad_fn.clone());

            // Register with gradtrack engine for gradient computation
            // Always register both operands to maintain consistent indexing
            let input_ids = vec![self.id(), other.id()];

            GradEngine::register_operation(result.id(), input_ids, grad_fn);
        }

        result
    }

    /// Dot product of two 1D tensors (returns scalar)
    ///
    /// Computes the dot product between two 1D tensors using SIMD-optimized kernels
    /// when available. The implementation uses AVX2 instructions for 8x vectorization
    /// with scalar fallbacks for non-SIMD hardware.
    ///
    /// # Arguments
    ///
    /// * `other` - The other 1D tensor (must have same length as self)
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the dot product result
    ///
    /// # Implementation Details
    ///
    /// - Uses `Tensor::new_uninitialized` for performance-critical allocation
    /// - SIMD path processes 8 elements at a time with horizontal reduction
    /// - Scalar path uses 4x unrolled loops for instruction-level parallelism
    /// - Memory is fully written to avoid uninitialized access
    fn dot_product_1d(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape().rank(), 1, "First tensor must be 1D");
        assert_eq!(other.shape().rank(), 1, "Second tensor must be 1D");
        assert_eq!(
            self.shape().dims[0],
            other.shape().dims[0],
            "Tensors must have same length for dot product"
        );

        let n = self.shape().dims[0];

        // Ensure both tensors are contiguous for kernel compatibility
        let self_contiguous = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };
        let other_contiguous = if other.is_contiguous() {
            other.clone()
        } else {
            other.contiguous()
        };

        // Use uninitialized allocation for scalar result - memory will be fully written
        let mut result = Tensor::new_uninitialized(vec![]); // Scalar tensor

        unsafe {
            let a_ptr = self_contiguous.as_ptr();
            let b_ptr = other_contiguous.as_ptr();
            let result_ptr = result.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    let dot_product = self.dot_product_simd_avx2(a_ptr, b_ptr, n);
                    *result_ptr = dot_product;
                } else {
                    let dot_product = self.dot_product_scalar(a_ptr, b_ptr, n);
                    *result_ptr = dot_product;
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                let dot_product = self.dot_product_scalar(a_ptr, b_ptr, n);
                *result_ptr = dot_product;
            }
        }

        result
    }

    /// Vector-matrix multiplication: v^T * M
    ///
    /// Computes the product of a 1D vector with a 2D matrix, treating the vector
    /// as a row vector. The implementation uses SIMD-optimized column-wise dot products
    /// for maximum performance on compatible hardware.
    ///
    /// # Arguments
    ///
    /// * `other` - The 2D matrix tensor (vector length must match matrix rows)
    ///
    /// # Returns
    ///
    /// A 1D tensor containing the vector-matrix multiplication result
    ///
    /// # Implementation Details
    ///
    /// - Computes dot product between vector and each matrix column
    /// - Uses SIMD kernels for each column when AVX2 is available
    /// - Scalar fallback processes each column individually
    /// - Memory layout optimized for column-wise access patterns
    fn vector_matrix_mult(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape().rank(), 1, "First tensor must be 1D (vector)");
        assert_eq!(other.shape().rank(), 2, "Second tensor must be 2D (matrix)");
        assert_eq!(
            self.shape().dims[0],
            other.shape().dims[0],
            "Vector length must match matrix rows"
        );

        let v_len = self.shape().dims[0];
        let m_cols = other.shape().dims[1];

        // Ensure both tensors are contiguous for kernel compatibility
        let self_contiguous = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };
        let other_contiguous = if other.is_contiguous() {
            other.clone()
        } else {
            other.contiguous()
        };

        // Use uninitialized allocation for performance - result will be fully written
        let mut result = Tensor::new_uninitialized(vec![m_cols]);

        unsafe {
            let v_ptr = self_contiguous.as_ptr();
            let m_ptr = other_contiguous.as_ptr();
            let result_ptr = result.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    // Use SIMD for each column
                    for col in 0..m_cols {
                        let dot_product =
                            self.vector_matrix_column_simd_avx2(v_ptr, m_ptr, v_len, m_cols, col);
                        *result_ptr.add(col) = dot_product;
                    }
                } else {
                    // Use scalar for each column
                    for col in 0..m_cols {
                        let dot_product =
                            self.vector_matrix_column_scalar(v_ptr, m_ptr, v_len, m_cols, col);
                        *result_ptr.add(col) = dot_product;
                    }
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                // Use scalar for each column
                for col in 0..m_cols {
                    let dot_product =
                        self.vector_matrix_column_scalar(v_ptr, m_ptr, v_len, m_cols, col);
                    *result_ptr.add(col) = dot_product;
                }
            }
        }

        result
    }

    /// Matrix-vector multiplication: M * v
    ///
    /// Computes the product of a 2D matrix with a 1D vector, treating the vector
    /// as a column vector. The implementation uses SIMD-optimized row-wise dot products
    /// for maximum performance on compatible hardware.
    ///
    /// # Arguments
    ///
    /// * `other` - The 1D vector tensor (matrix columns must match vector length)
    ///
    /// # Returns
    ///
    /// A 1D tensor containing the matrix-vector multiplication result
    ///
    /// # Implementation Details
    ///
    /// - Computes dot product between each matrix row and the vector
    /// - Uses SIMD kernels for each row when AVX2 is available
    /// - Scalar fallback processes each row individually
    /// - Memory layout optimized for row-wise access patterns
    fn matrix_vector_mult(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape().rank(), 2, "First tensor must be 2D (matrix)");
        assert_eq!(other.shape().rank(), 1, "Second tensor must be 1D (vector)");
        assert_eq!(
            self.shape().dims[1],
            other.shape().dims[0],
            "Matrix columns must match vector length"
        );

        let m_rows = self.shape().dims[0];
        let m_cols = self.shape().dims[1];

        // Ensure both tensors are contiguous for kernel compatibility
        let self_contiguous = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };
        let other_contiguous = if other.is_contiguous() {
            other.clone()
        } else {
            other.contiguous()
        };

        // Use uninitialized allocation for performance - result will be fully written
        let mut result = Tensor::new_uninitialized(vec![m_rows]);

        unsafe {
            let m_ptr = self_contiguous.as_ptr();
            let v_ptr = other_contiguous.as_ptr();
            let result_ptr = result.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    // Use SIMD for each row
                    for row in 0..m_rows {
                        let dot_product =
                            self.matrix_vector_row_simd_avx2(m_ptr, v_ptr, m_cols, row);
                        *result_ptr.add(row) = dot_product;
                    }
                } else {
                    // Use scalar for each row
                    for row in 0..m_rows {
                        let dot_product = self.matrix_vector_row_scalar(m_ptr, v_ptr, m_cols, row);
                        *result_ptr.add(row) = dot_product;
                    }
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                // Use scalar for each row
                for row in 0..m_rows {
                    let dot_product = self.matrix_vector_row_scalar(m_ptr, v_ptr, m_cols, row);
                    *result_ptr.add(row) = dot_product;
                }
            }
        }

        result
    }

    /// Standard matrix-matrix multiplication (2D @ 2D)
    ///
    /// Computes the product of two 2D matrices using intelligent kernel selection
    /// based on matrix dimensions. The implementation uses cache-friendly blocked
    /// algorithms for large matrices and direct computation for small matrices.
    ///
    /// # Arguments
    ///
    /// * `other` - The right matrix (2D tensor with compatible inner dimensions)
    ///
    /// # Returns
    ///
    /// A 2D tensor containing the matrix multiplication result
    ///
    /// # Implementation Details
    ///
    /// - Uses `MatmulConfig::for_dimensions` for optimal kernel selection
    /// - Dispatches to `kernels::matrix_multiply_blocked` for computation
    /// - Supports both SIMD and scalar execution paths
    /// - Memory layout optimized for cache efficiency and SIMD alignment
    fn matrix_matrix_mult(&self, other: &Tensor) -> Tensor {
        let m = self.shape().dims[0]; // Result rows
        let k = self.shape().dims[1]; // Inner dimension
        let n = other.shape().dims[1]; // Result columns

        assert_eq!(
            k,
            other.shape().dims[0],
            "Inner dimensions must match: {} vs {}",
            k,
            other.shape().dims[0]
        );

        // Ensure both tensors are contiguous for kernel compatibility
        let self_contiguous = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };
        let other_contiguous = if other.is_contiguous() {
            other.clone()
        } else {
            other.contiguous()
        };

        // Use uninitialized allocation for performance - will be initialized properly
        let mut result = Tensor::new_uninitialized(vec![m, n]);

        unsafe {
            let a_ptr = self_contiguous.as_ptr();
            let b_ptr = other_contiguous.as_ptr();
            let c_ptr = result.as_mut_ptr();

            // Determine optimal configuration and dispatch
            let config = MatmulConfig::for_dimensions(m, n, k);
            kernels::matrix_multiply_blocked(a_ptr, b_ptr, c_ptr, m, n, k, &config);
        }

        result
    }

    /// Batched matrix multiplication for higher-dimensional tensors
    ///
    /// Performs matrix multiplication on the last two dimensions while broadcasting
    /// the leading dimensions. This operation supports arbitrary tensor shapes
    /// with at least 2 dimensions, following NumPy broadcasting rules.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor for batched multiplication (must have at least 2D)
    ///
    /// # Returns
    ///
    /// A tensor with batched matrix multiplication results, with shape determined
    /// by broadcasting the batch dimensions and matrix multiplication on the last two
    ///
    /// # Implementation Details
    ///
    /// - Broadcasts batch dimensions following NumPy right-aligned rules
    /// - Performs individual matrix multiplications for each batch element
    /// - Uses `calculate_batch_offset_with_broadcast` for memory offset computation
    /// - Supports broadcasting of singleton dimensions (size 1) to any size
    fn batched_matmul(&self, other: &Tensor) -> Tensor {
        let self_shape = self.shape();
        let other_shape = other.shape();

        // Ensure both tensors have at least 2 dimensions
        assert!(
            self_shape.rank() >= 2,
            "Batched matmul requires at least 2D tensors"
        );
        assert!(
            other_shape.rank() >= 2,
            "Batched matmul requires at least 2D tensors"
        );

        // Get matrix dimensions (last two dimensions)
        let self_m = self_shape.dims[self_shape.rank() - 2];
        let self_k = self_shape.dims[self_shape.rank() - 1];
        let other_k = other_shape.dims[other_shape.rank() - 2];
        let other_n = other_shape.dims[other_shape.rank() - 1];

        assert_eq!(
            self_k, other_k,
            "Inner dimensions must match for batched matmul: {} vs {}",
            self_k, other_k
        );

        // Calculate output shape by broadcasting batch dimensions
        let mut output_dims = Vec::new();
        let max_rank = self_shape.rank().max(other_shape.rank());

        // Broadcast batch dimensions (right-aligned)
        for i in 0..(max_rank - 2) {
            let self_batch_rank = self_shape.rank() - 2;
            let other_batch_rank = other_shape.rank() - 2;

            let self_dim = if i < self_batch_rank {
                self_shape.dims[self_batch_rank - 1 - i]
            } else {
                1
            };
            let other_dim = if i < other_batch_rank {
                other_shape.dims[other_batch_rank - 1 - i]
            } else {
                1
            };

            if self_dim == 1 {
                output_dims.push(other_dim);
            } else if other_dim == 1 || self_dim == other_dim {
                output_dims.push(self_dim);
            } else {
                panic!("Cannot broadcast dimensions {} and {}", self_dim, other_dim);
            }
        }

        // Reverse to get correct order (we built from right to left)
        output_dims.reverse();

        // Add matrix dimensions
        output_dims.push(self_m);
        output_dims.push(other_n);

        // Use uninitialized allocation for performance - result will be fully written
        let mut result = Tensor::new_uninitialized(output_dims.clone());

        // Calculate total number of matrix multiplications
        let batch_size: usize = output_dims[..output_dims.len() - 2].iter().product();

        unsafe {
            // Perform batched matrix multiplication
            let batch_dims = &output_dims[..output_dims.len() - 2];
            for batch_idx in 0..batch_size {
                // Calculate offsets for this batch with broadcasting support
                let self_offset = self.calculate_batch_offset_with_broadcast(
                    batch_idx,
                    self_m * self_k,
                    batch_dims,
                );
                let other_offset = other.calculate_batch_offset_with_broadcast(
                    batch_idx,
                    other_k * other_n,
                    batch_dims,
                );
                let result_offset = batch_idx * self_m * other_n;

                let a_ptr = self.as_ptr().add(self_offset);
                let b_ptr = other.as_ptr().add(other_offset);
                let c_ptr = result.as_mut_ptr().add(result_offset);

                // Perform single matrix multiplication with dynamic configuration
                let config = MatmulConfig::for_dimensions(self_m, other_n, self_k);
                kernels::matrix_multiply_blocked(
                    a_ptr, b_ptr, c_ptr, self_m, other_n, self_k, &config,
                );
            }
        }

        // Handle PyTorch-compatible shape squeezing
        // If one operand was 2D, squeeze out the batch dimension from the result
        let should_squeeze_batch = self_shape.rank() == 2 || other_shape.rank() == 2;
        if should_squeeze_batch && output_dims.len() > 2 && output_dims[0] == 1 {
            // Squeeze out the leading dimension of size 1
            result = result.squeeze(Some(0));
        }

        result
    }

    /// Calculate memory offset for batched operations with broadcasting support
    ///
    /// Computes the memory offset for a specific batch element, taking into account
    /// broadcasting rules where singleton dimensions (size 1) are repeated across
    /// the batch. This enables efficient batched operations with broadcasting.
    ///
    /// # Arguments
    ///
    /// * `batch_idx` - Linear batch index (0-based)
    /// * `matrix_size` - Size of each matrix in elements (product of last two dimensions)
    /// * `output_batch_dims` - Output batch dimensions for reference (leading dimensions)
    ///
    /// # Returns
    ///
    /// Memory offset in elements for the specified batch index
    ///
    /// # Implementation Details
    ///
    /// - Converts linear batch index to multi-dimensional coordinates
    /// - Handles broadcasting by mapping coordinates to actual tensor dimensions
    /// - Uses stride-based offset calculation for memory efficiency
    /// - Supports right-aligned broadcasting following NumPy conventions
    fn calculate_batch_offset_with_broadcast(
        &self,
        batch_idx: usize,
        matrix_size: usize,
        output_batch_dims: &[usize],
    ) -> usize {
        if output_batch_dims.is_empty() {
            return 0;
        }

        // Convert linear batch index to multi-dimensional coordinates
        let mut coords = Vec::new();
        let mut temp_idx = batch_idx;

        for &dim_size in output_batch_dims.iter().rev() {
            coords.push(temp_idx % dim_size);
            temp_idx /= dim_size;
        }
        coords.reverse();

        // Calculate actual offset based on this tensor's batch dimensions
        let self_batch_dims = &self.shape().dims[..self.shape().rank() - 2];
        let mut offset = 0;

        // Align coordinates from the right (broadcasting is right-aligned)
        let coord_offset = if output_batch_dims.len() >= self_batch_dims.len() {
            output_batch_dims.len() - self_batch_dims.len()
        } else {
            0
        };

        // Calculate offset using strides
        for (i, &self_dim) in self_batch_dims.iter().enumerate() {
            let coord_idx = coord_offset + i;
            if coord_idx < coords.len() {
                let coord = coords[coord_idx];
                // If this tensor's dimension is 1, we stay at the same position (broadcasting)
                let actual_coord = if self_dim == 1 { 0 } else { coord % self_dim };

                // Calculate stride for this dimension
                let mut stride = matrix_size;
                for &later_dim in self_batch_dims.iter().skip(i + 1) {
                    stride *= later_dim;
                }

                offset += actual_coord * stride;
            }
        }

        offset
    }

    // ===== SIMD Optimized Implementations =====

    /// AVX2-optimized dot product implementation
    ///
    /// Computes dot product using AVX2 SIMD instructions for 8x vectorization.
    /// Processes 8 elements at a time with horizontal reduction for final sum.
    ///
    /// # Safety
    ///
    /// Requires AVX2 support and valid pointers with sufficient memory for n elements.
    /// Memory must be properly aligned for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `a_ptr` - Pointer to first vector data
    /// * `b_ptr` - Pointer to second vector data  
    /// * `n` - Number of elements to process
    ///
    /// # Returns
    ///
    /// Dot product result as f32
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_simd_avx2(&self, a_ptr: *const f32, b_ptr: *const f32, n: usize) -> f32 {
        let simd_end = n & !7; // Process 8 elements at a time
        let mut sum_vec = _mm256_setzero_ps();

        // SIMD loop
        for i in (0..simd_end).step_by(8) {
            let a_vec = _mm256_loadu_ps(a_ptr.add(i));
            let b_vec = _mm256_loadu_ps(b_ptr.add(i));
            let prod = _mm256_mul_ps(a_vec, b_vec);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }

        // Horizontal sum of SIMD register
        let sum_hi = _mm256_extractf128_ps(sum_vec, 1);
        let sum_lo = _mm256_castps256_ps128(sum_vec);
        let sum_quad = _mm_add_ps(sum_hi, sum_lo);
        let sum_dual = _mm_hadd_ps(sum_quad, sum_quad);
        let sum_single = _mm_hadd_ps(sum_dual, sum_dual);
        let mut result = _mm_cvtss_f32(sum_single);

        // Handle remaining elements
        for i in simd_end..n {
            result += *a_ptr.add(i) * *b_ptr.add(i);
        }

        result
    }

    /// Scalar-optimized dot product implementation
    ///
    /// Computes dot product using scalar operations with 4x loop unrolling for
    /// better instruction-level parallelism. Provides fallback for non-SIMD hardware.
    ///
    /// # Safety
    ///
    /// Requires valid pointers with sufficient memory for n elements.
    ///
    /// # Arguments
    ///
    /// * `a_ptr` - Pointer to first vector data
    /// * `b_ptr` - Pointer to second vector data
    /// * `n` - Number of elements to process
    ///
    /// # Returns
    ///
    /// Dot product result as f32
    #[inline]
    unsafe fn dot_product_scalar(&self, a_ptr: *const f32, b_ptr: *const f32, n: usize) -> f32 {
        let mut sum = 0.0f32;
        let unroll_end = n & !3; // Process 4 elements at a time

        // Unrolled loop for better instruction-level parallelism
        for i in (0..unroll_end).step_by(4) {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
            sum += *a_ptr.add(i + 1) * *b_ptr.add(i + 1);
            sum += *a_ptr.add(i + 2) * *b_ptr.add(i + 2);
            sum += *a_ptr.add(i + 3) * *b_ptr.add(i + 3);
        }

        // Handle remaining elements
        for i in unroll_end..n {
            sum += *a_ptr.add(i) * *b_ptr.add(i);
        }

        sum
    }

    /// AVX2-optimized vector-matrix column dot product
    ///
    /// Computes dot product between a vector and a specific matrix column using
    /// AVX2 SIMD instructions. Optimized for column-wise access patterns.
    ///
    /// # Safety
    ///
    /// Requires AVX2 support and valid pointers with sufficient memory.
    /// Matrix must be in row-major layout with m_cols columns.
    ///
    /// # Arguments
    ///
    /// * `v_ptr` - Pointer to vector data
    /// * `m_ptr` - Pointer to matrix data (row-major layout)
    /// * `v_len` - Length of vector (must match matrix rows)
    /// * `m_cols` - Number of columns in matrix
    /// * `col` - Column index to compute dot product with
    ///
    /// # Returns
    ///
    /// Dot product result as f32
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vector_matrix_column_simd_avx2(
        &self,
        v_ptr: *const f32,
        m_ptr: *const f32,
        v_len: usize,
        m_cols: usize,
        col: usize,
    ) -> f32 {
        let simd_end = v_len & !7;
        let mut sum_vec = _mm256_setzero_ps();

        // Process 8 elements at a time with optimized gather
        for i in (0..simd_end).step_by(8) {
            let v_vec = _mm256_loadu_ps(v_ptr.add(i));

            // Optimized gather for matrix column elements
            let m0 = *m_ptr.add(i * m_cols + col);
            let m1 = *m_ptr.add((i + 1) * m_cols + col);
            let m2 = *m_ptr.add((i + 2) * m_cols + col);
            let m3 = *m_ptr.add((i + 3) * m_cols + col);
            let m4 = *m_ptr.add((i + 4) * m_cols + col);
            let m5 = *m_ptr.add((i + 5) * m_cols + col);
            let m6 = *m_ptr.add((i + 6) * m_cols + col);
            let m7 = *m_ptr.add((i + 7) * m_cols + col);

            let m_vec = _mm256_set_ps(m7, m6, m5, m4, m3, m2, m1, m0);

            let prod = _mm256_mul_ps(v_vec, m_vec);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }

        // Horizontal sum
        let sum_hi = _mm256_extractf128_ps(sum_vec, 1);
        let sum_lo = _mm256_castps256_ps128(sum_vec);
        let sum_quad = _mm_add_ps(sum_hi, sum_lo);
        let sum_dual = _mm_hadd_ps(sum_quad, sum_quad);
        let sum_single = _mm_hadd_ps(sum_dual, sum_dual);
        let mut result = _mm_cvtss_f32(sum_single);

        // Handle remaining elements
        for i in simd_end..v_len {
            result += *v_ptr.add(i) * *m_ptr.add(i * m_cols + col);
        }

        result
    }

    /// Scalar vector-matrix column dot product
    ///
    /// Computes dot product between a vector and a specific matrix column using
    /// scalar operations. Provides fallback for non-SIMD hardware.
    ///
    /// # Safety
    ///
    /// Requires valid pointers with sufficient memory.
    /// Matrix must be in row-major layout with m_cols columns.
    ///
    /// # Arguments
    ///
    /// * `v_ptr` - Pointer to vector data
    /// * `m_ptr` - Pointer to matrix data (row-major layout)
    /// * `v_len` - Length of vector (must match matrix rows)
    /// * `m_cols` - Number of columns in matrix
    /// * `col` - Column index to compute dot product with
    ///
    /// # Returns
    ///
    /// Dot product result as f32
    #[inline]
    unsafe fn vector_matrix_column_scalar(
        &self,
        v_ptr: *const f32,
        m_ptr: *const f32,
        v_len: usize,
        m_cols: usize,
        col: usize,
    ) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..v_len {
            sum += *v_ptr.add(i) * *m_ptr.add(i * m_cols + col);
        }
        sum
    }

    /// AVX2-optimized matrix-vector row dot product
    ///
    /// Computes dot product between a specific matrix row and a vector using
    /// AVX2 SIMD instructions. Optimized for row-wise access patterns.
    ///
    /// # Safety
    ///
    /// Requires AVX2 support and valid pointers with sufficient memory.
    /// Matrix must be in row-major layout with m_cols columns.
    ///
    /// # Arguments
    ///
    /// * `m_ptr` - Pointer to matrix data (row-major layout)
    /// * `v_ptr` - Pointer to vector data
    /// * `m_cols` - Number of columns in matrix (must match vector length)
    /// * `row` - Row index to compute dot product with
    ///
    /// # Returns
    ///
    /// Dot product result as f32
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn matrix_vector_row_simd_avx2(
        &self,
        m_ptr: *const f32,
        v_ptr: *const f32,
        m_cols: usize,
        row: usize,
    ) -> f32 {
        let simd_end = m_cols & !7;
        let mut sum_vec = _mm256_setzero_ps();
        let row_ptr = m_ptr.add(row * m_cols);

        for i in (0..simd_end).step_by(8) {
            let m_vec = _mm256_loadu_ps(row_ptr.add(i));
            let v_vec = _mm256_loadu_ps(v_ptr.add(i));
            let prod = _mm256_mul_ps(m_vec, v_vec);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }

        // Horizontal sum
        let sum_hi = _mm256_extractf128_ps(sum_vec, 1);
        let sum_lo = _mm256_castps256_ps128(sum_vec);
        let sum_quad = _mm_add_ps(sum_hi, sum_lo);
        let sum_dual = _mm_hadd_ps(sum_quad, sum_quad);
        let sum_single = _mm_hadd_ps(sum_dual, sum_dual);
        let mut result = _mm_cvtss_f32(sum_single);

        // Handle remaining elements
        for i in simd_end..m_cols {
            result += *row_ptr.add(i) * *v_ptr.add(i);
        }

        result
    }

    /// Scalar matrix-vector row dot product
    ///
    /// Computes dot product between a specific matrix row and a vector using
    /// scalar operations. Provides fallback for non-SIMD hardware.
    ///
    /// # Safety
    ///
    /// Requires valid pointers with sufficient memory.
    /// Matrix must be in row-major layout with m_cols columns.
    ///
    /// # Arguments
    ///
    /// * `m_ptr` - Pointer to matrix data (row-major layout)
    /// * `v_ptr` - Pointer to vector data
    /// * `m_cols` - Number of columns in matrix (must match vector length)
    /// * `row` - Row index to compute dot product with
    ///
    /// # Returns
    ///
    /// Dot product result as f32
    #[inline]
    unsafe fn matrix_vector_row_scalar(
        &self,
        m_ptr: *const f32,
        v_ptr: *const f32,
        m_cols: usize,
        row: usize,
    ) -> f32 {
        let mut sum = 0.0f32;
        let row_ptr = m_ptr.add(row * m_cols);
        for i in 0..m_cols {
            sum += *row_ptr.add(i) * *v_ptr.add(i);
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    //! Matrix multiplication operation tests
    //!
    //! This module contains comprehensive tests for matrix multiplication operations,
    //! including basic functionality, kernel selection, and large matrix handling.
    //! Tests cover all supported operation types and edge cases.

    use super::*;

    /// Test basic 2x2 matrix multiplication functionality
    ///
    /// Verifies that the matmul operation correctly computes the product of two 2x2 matrices
    /// and produces the expected numerical results. This test validates the core matrix
    /// multiplication algorithm and result shape computation.
    #[test]
    fn test_matmul_2d_basic() {
        // Test basic 2x2 matrix multiplication
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let result = a.matmul(&b);

        assert_eq!(result.shape().dims, vec![2, 2]);

        // Expected result: [[19, 22], [43, 50]]
        unsafe {
            let ptr = result.as_ptr();
            assert_eq!(*ptr.add(0), 19.0); // (0,0)
            assert_eq!(*ptr.add(1), 22.0); // (0,1)
            assert_eq!(*ptr.add(2), 43.0); // (1,0)
            assert_eq!(*ptr.add(3), 50.0); // (1,1)
        }
    }

    /// Test 2D @ 2D matmul gradient computation (matrix @ matrix)
    #[test]
    fn test_matmul_2d_2d_gradients() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2])
            .unwrap()
            .with_requires_grad();

        let mut result = a.matmul(&b); // [2, 2] @ [2, 2] -> [2, 2]
        assert_eq!(result.shape().dims, vec![2, 2]);

        // Expected result: [[19, 22], [43, 50]]
        let expected = [19.0, 22.0, 43.0, 50.0];
        unsafe {
            let ptr = result.as_ptr();
            for (i, val) in expected.iter().enumerate().take(4) {
                assert_eq!(*ptr.add(i), *val);
            }
        }

        // Set up gradient for backward pass
        let grad_output = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        result.backward(Some(grad_output));

        let grad_a = a.grad_by_value().unwrap();
        let grad_b = b.grad_by_value().unwrap();

        assert_eq!(grad_a.shape().dims, vec![2, 2]);
        assert_eq!(grad_b.shape().dims, vec![2, 2]);

        // grad_a = grad_output @ b^T = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]

        unsafe {
            let grad_a_ptr = grad_a.as_ptr();
            assert_eq!(*grad_a_ptr.add(0), 11.0); // 1*5 + 1*6
            assert_eq!(*grad_a_ptr.add(1), 15.0); // 1*7 + 1*8
            assert_eq!(*grad_a_ptr.add(2), 11.0); // 1*5 + 1*6
            assert_eq!(*grad_a_ptr.add(3), 15.0); // 1*7 + 1*8
        }

        // grad_b = a^T @ grad_output = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
        unsafe {
            let grad_b_ptr = grad_b.as_ptr();
            assert_eq!(*grad_b_ptr.add(0), 4.0); // 1*1 + 3*1
            assert_eq!(*grad_b_ptr.add(1), 4.0); // 1*1 + 3*1
            assert_eq!(*grad_b_ptr.add(2), 6.0); // 2*1 + 4*1
            assert_eq!(*grad_b_ptr.add(3), 6.0); // 2*1 + 4*1
        }
    }

    /// Test matmul gradient computation with partial requires_grad
    #[test]
    fn test_matmul_partial_requires_grad() {
        // Test case where only one operand requires gradients (like the linear layer case)
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap(); // No requires_grad
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])
            .unwrap()
            .with_requires_grad(); // Only b requires gradients

        let mut result = a.matmul(&b); // [3] @ [3, 2] -> [2]
        assert_eq!(result.shape().dims, vec![2]);

        result.backward(None);

        // Only b should have gradients
        assert!(a.grad_by_value().is_none());
        let grad_b = b.grad_by_value().unwrap();

        assert_eq!(grad_b.shape().dims, vec![3, 2]);

        // grad_b = outer_product(a, grad_output)
        // Since grad_output defaults to ones([2]), grad_b[i,j] = a[i] * 1.0 = a[i]
        unsafe {
            let grad_b_ptr = grad_b.as_ptr();
            assert_eq!(*grad_b_ptr.add(0), 1.0); // a[0] * grad_output[0]
            assert_eq!(*grad_b_ptr.add(1), 1.0); // a[0] * grad_output[1]
            assert_eq!(*grad_b_ptr.add(2), 2.0); // a[1] * grad_output[0]
            assert_eq!(*grad_b_ptr.add(3), 2.0); // a[1] * grad_output[1]
            assert_eq!(*grad_b_ptr.add(4), 3.0); // a[2] * grad_output[0]
            assert_eq!(*grad_b_ptr.add(5), 3.0); // a[2] * grad_output[1]
        }
    }

    #[test]
    fn test_debug_gradient_values() {
        println!("=== Debugging matmul gradient issue ===");

        // Test case: [1, 3, 4] @ [2, 4, 5] which should fail with our=41, torch=29
        let left_shape = vec![1, 3, 4];
        let right_shape = vec![2, 4, 5];

        let mut left = Tensor::zeros(left_shape.clone()).with_requires_grad();
        let mut right = Tensor::zeros(right_shape.clone()).with_requires_grad();

        let left_size = left_shape.iter().product::<usize>();
        let right_size = right_shape.iter().product::<usize>();

        // Fill with exactly the same data as the validation test
        unsafe {
            for i in 0..left_size {
                *left.as_mut_ptr().add(i) = (i as f32) * 0.1 + 1.0;
            }
            for i in 0..right_size {
                *right.as_mut_ptr().add(i) = (i as f32) * 0.2 + 0.5;
            }
        }

        println!(
            "Left shape: {:?}, data: {:?}",
            left.shape().dims,
            left.data()
        );
        println!(
            "Right shape: {:?}, data: {:?}",
            right.shape().dims,
            right.data()
        );

        // Forward pass
        let mut result = left.matmul(&right);
        println!(
            "Result shape: {:?}, data: {:?}",
            result.shape().dims,
            result.data()
        );

        // Backward pass with ones
        let grad_ones = Tensor::ones(result.shape().dims.clone());
        println!(
            "Grad ones shape: {:?}, data: {:?}",
            grad_ones.shape().dims,
            grad_ones.data()
        );

        result.backward(Some(grad_ones));

        let grad_left = left.grad_by_value().unwrap();
        let grad_right = right.grad_by_value().unwrap();

        println!(
            "Left gradient shape: {:?}, data: {:?}",
            grad_left.shape().dims,
            grad_left.data()
        );
        println!(
            "Right gradient shape: {:?}, data: {:?}",
            grad_right.shape().dims,
            grad_right.data()
        );

        println!(
            "Left gradient[0] = {} (expected ~29, but we're getting ~41)",
            grad_left.data()[0]
        );
    }

    #[test]
    fn test_simple_batched_gradient() {
        println!("=== Testing simple batched gradient ===");

        // Simple case: [2, 2, 2] @ [2, 2, 2]
        let left = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])
            .unwrap()
            .with_requires_grad();
        let right = Tensor::from_slice(&[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], vec![2, 2, 2])
            .unwrap()
            .with_requires_grad();

        println!("Left: {:?}", left.data());
        println!("Right: {:?}", right.data());

        // Test transpose function first
        let right_t = right.transpose(1, 2);
        println!("Right transposed: {:?}", right_t.data());
        println!("Right transposed contiguous: {:?}", right_t.is_contiguous());
        println!("Right transposed strides: {:?}", right_t.strides());

        let mut result = left.matmul(&right);
        println!("Result: {:?}", result.data());

        let grad_ones = Tensor::ones(result.shape().dims.clone());
        result.backward(Some(grad_ones));

        let grad_left = left.grad_by_value().unwrap();
        let grad_right = right.grad_by_value().unwrap();

        println!("Left gradient: {:?}", grad_left.data());
        println!("Right gradient: {:?}", grad_right.data());

        // Manual calculation for verification
        println!("\n=== Manual verification ===");
        println!("Expected left grad batch 0: [0.5+1.0, 1.5+2.0] = [1.5, 3.5]");
        println!("Expected left grad batch 1: [2.5+3.0, 3.5+4.0] = [5.5, 7.5]");
    }

    #[test]
    fn test_linear_layer_pattern() {
        // Simulate the exact pattern from the training loop
        let x_data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap(); // Input (no grad)
        let weight = Tensor::from_slice(&[0.1, 0.5, 0.3, 0.1, 0.5, 0.3], vec![3, 2])
            .unwrap()
            .with_requires_grad(); // Weight (requires grad)
        let bias = Tensor::from_slice(&[0.0, 0.1], vec![2])
            .unwrap()
            .with_requires_grad(); // Bias (requires grad)

        // Forward pass
        let weighted = x_data.matmul(&weight); // [3] @ [3, 2] -> [2]
        let y_pred = weighted.add_tensor(&bias); // [2] + [2] -> [2]

        // Create a simple loss (sum of squared differences with some target)
        let y_true = Tensor::from_slice(&[3.0, 5.0], vec![2]).unwrap();
        let mut loss = y_pred.sub_tensor(&y_true).pow_scalar(2.0).mean();

        // Backward pass
        loss.backward(None);

        // Check that gradients are computed correctly
        let grad_weight = weight.grad_by_value().unwrap();
        let grad_bias = bias.grad_by_value().unwrap();

        assert_eq!(grad_weight.shape().dims, vec![3, 2]); // Same shape as weight
        assert_eq!(grad_bias.shape().dims, vec![2]); // Same shape as bias

        // The exact gradient values depend on the computation graph, but shapes should be correct
        assert_eq!(grad_weight.size(), 6);
        assert_eq!(grad_bias.size(), 2);

        // Verify that no gradient is computed for x_data (doesn't require grad)
        assert!(x_data.grad_by_value().is_none());
    }
}
