//! Matrix multiplication operations with optimized kernels
//!
//! This module provides a comprehensive matrix multiplication implementation optimized
//! for single-threaded performance with SIMD acceleration. The implementation supports
//! all NumPy-style matrix multiplication patterns including 1D/2D/ND tensor operations
//! with automatic differentiation support.
//!
//! # Key Features
//!
//! - **SIMD Optimization**: AVX512/AVX2/SSE2 implementations with runtime dispatch
//! - **Intelligent Dispatch**: Cached kernel selection based on matrix dimensions and alignment
//! - **Cache Optimization**: Blocked algorithms with panel packing for L1/L2/L3 cache efficiency
//! - **Memory Bandwidth**: Optimized for maximum memory bandwidth utilization
//! - **GradTrack Integration**: Automatic gradient computation for all operations
//! - **Thread Safety**: All operations are thread-safe and Send + Sync
//! - **Mathematical Validation**: High-precision equivalence to PyTorch reference
//!
//! # Performance Characteristics
//!
//! The implementation uses intelligent dispatch to select optimal kernels based on matrix size:
//! - **Small matrices (≤1K elements)**: Direct computation with minimal overhead
//! - **Medium matrices (1K-64K elements)**: Cache-optimized blocking for L1/L2 cache
//! - **Large matrices (≥64K elements)**: Memory bandwidth optimized with hierarchical blocking
//! - **AVX512 acceleration**: 16x SIMD operations for compatible hardware
//! - **AVX2 acceleration**: 8x SIMD operations for compatible hardware
//! - **SSE2 acceleration**: 4x SIMD operations for compatible hardware
//! - **Scalar fallbacks**: Optimized scalar implementations for non-SIMD platforms
//! - **Memory Safety**: Safe memory management with proper alignment
//!
//! # Architecture
//!
//! The matmul system uses a cached kernel dispatch architecture:
//! - **`MatMulKernels`**: Cached function pointers for optimal performance
//! - **Static Dispatch**: Runtime SIMD detection with compile-time optimization
//! - **Operation Types**: Specialized kernels for each matmul operation pattern
//! - **Size Thresholds**: Intelligent kernel selection based on matrix dimensions
//! - **Alignment Detection**: Optimized paths for aligned vs unaligned data
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
//! assert_eq!(result.shape().dims(), vec![2, 2]);
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
//! assert_eq!(result.shape().dims(), vec![2]);
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
//! assert_eq!(result.shape().dims(), vec![2]);
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
//! assert_eq!(result.shape().dims(), vec![]); // Scalar tensor
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
//! assert_eq!(result.shape().dims(), vec![2, 2, 2]);
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
//! assert_eq!(result.shape().dims(), vec![2, 2]);
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

use crate::Tensor;
#[cfg(target_arch = "x86_64")]
pub mod avx2_kernels;
#[cfg(target_arch = "x86_64")]
pub mod avx512_kernels;
pub mod classification_and_validation;
pub mod dispatch;
#[cfg(target_arch = "x86_64")]
pub mod pack_n_cache;
pub mod scalar_kernels;
#[cfg(target_arch = "x86_64")]
pub mod sse_kernels;
use dispatch::*;

impl Tensor {
    /// Matrix multiplication with intelligent kernel dispatch
    ///
    /// Performs matrix multiplication using optimized SIMD kernels selected based on:
    /// - Runtime SIMD capability (AVX512/AVX2/SSE2/Scalar)
    /// - Matrix operation type (1D@1D, 1D@2D, 2D@1D, 2D@2D, ND@ND)
    /// - Matrix size classification (Small/Medium/Large)
    /// - Memory alignment characteristics
    ///
    /// # Arguments
    /// * `other` - Right-hand side tensor for multiplication
    ///
    /// # Returns
    /// Result tensor with appropriate shape based on operation type
    ///
    /// # Panics
    /// Panics if tensor shapes are incompatible for matrix multiplication
    ///
    /// # Examples
    /// ```
    /// use train_station::Tensor;
    ///
    /// // 2D matrix multiplication
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    /// let result = a.matmul(&b);
    /// assert_eq!(result.shape().dims(), vec![2, 2]);
    ///
    /// // 1D dot product
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]).unwrap();
    /// let result = a.matmul(&b);
    /// assert_eq!(result.shape().dims(), vec![]); // Scalar
    /// ```
    #[track_caller]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Get cached kernels for dispatch
        let kernels = MatMulKernels::get_cached_kernels();

        // Classify operation type based on shapes
        let left_shape = self.shape().dims();
        let right_shape = other.shape().dims();
        let op_type = MatMulKernels::classify_operation(left_shape, right_shape);

        // Validate shapes and compute result shape
        let result_shape = Self::validate_and_compute_matmul_shape(left_shape, right_shape);

        // Get tensor pointers directly - scalar kernels handle strides
        let left_ptr = unsafe { self.as_ptr() };
        let right_ptr = unsafe { other.as_ptr() };

        // Create result tensor
        let mut result = Tensor::new(result_shape.clone());

        unsafe {
            let result_ptr = result.as_mut_ptr();

            // Dispatch to appropriate kernel based on operation type
            match op_type {
                MatMulOpType::Dot1D1D => {
                    // 1D @ 1D: Dot product returning scalar
                    let size = left_shape[0];
                    if self.is_contiguous() && other.is_contiguous() {
                        let dot_result = kernels.dispatch_dot_1d(left_ptr, right_ptr, size);
                        *result_ptr = dot_result;
                    } else {
                        let left_stride = self.strides()[0];
                        let right_stride = other.strides()[0];
                        let dot_result = kernels.dispatch_dot_1d_strided(
                            left_ptr,
                            right_ptr,
                            size,
                            left_stride,
                            right_stride,
                        );
                        *result_ptr = dot_result;
                    }
                }
                MatMulOpType::Vec1D2D => {
                    // 1D @ 2D: Vector-matrix multiplication (v^T * M)
                    let k = left_shape[0];
                    let n = right_shape[1];
                    if self.is_contiguous() && other.is_contiguous() {
                        kernels.dispatch_vec_mat(left_ptr, right_ptr, result_ptr, k, n);
                    } else {
                        let left_stride = self.strides()[0];
                        let right_strides = other.strides();
                        let right_row_stride = right_strides[0];
                        let right_col_stride = right_strides[1];
                        let result_stride = 1; // Result is always contiguous
                        kernels.dispatch_vec_mat_strided(
                            left_ptr,
                            right_ptr,
                            result_ptr,
                            k,
                            n,
                            left_stride,
                            right_row_stride,
                            right_col_stride,
                            result_stride,
                        );
                    }
                }
                MatMulOpType::Mat2D1D => {
                    // 2D @ 1D: Matrix-vector multiplication (M * v)
                    let m = left_shape[0];
                    let k = left_shape[1];
                    if self.is_contiguous() && other.is_contiguous() {
                        kernels.dispatch_mat_vec(left_ptr, right_ptr, result_ptr, m, k);
                    } else {
                        let left_strides = self.strides();
                        let left_row_stride = left_strides[0];
                        let left_col_stride = left_strides[1];
                        let right_stride = other.strides()[0];
                        let result_stride = 1; // Result is always contiguous
                        kernels.dispatch_mat_vec_strided(
                            left_ptr,
                            right_ptr,
                            result_ptr,
                            m,
                            k,
                            left_row_stride,
                            left_col_stride,
                            right_stride,
                            result_stride,
                        );
                    }
                }
                MatMulOpType::Mat2D2D => {
                    // 2D @ 2D: Standard matrix multiplication
                    let m = left_shape[0];
                    let k = left_shape[1];
                    let n = right_shape[1];
                    if self.is_contiguous() && other.is_contiguous() {
                        kernels.dispatch_mat_mat(left_ptr, right_ptr, result_ptr, m, k, n);
                    } else {
                        let left_strides = self.strides();
                        let left_row_stride = left_strides[0];
                        let left_col_stride = left_strides[1];
                        let right_strides = other.strides();
                        let right_row_stride = right_strides[0];
                        let right_col_stride = right_strides[1];
                        let result_strides = result.strides();
                        let result_row_stride = result_strides[0];
                        let result_col_stride = result_strides[1];
                        kernels.dispatch_mat_mat_strided(
                            left_ptr,
                            right_ptr,
                            result_ptr,
                            m,
                            k,
                            n,
                            left_row_stride,
                            left_col_stride,
                            right_row_stride,
                            right_col_stride,
                            result_row_stride,
                            result_col_stride,
                        );
                    }
                }
                MatMulOpType::BatchedND => {
                    // ND @ ND: Batched matrix multiplication on last two dimensions
                    Self::dispatch_batched_matmul_with_ptrs_strided(
                        left_ptr,
                        right_ptr,
                        self,
                        other,
                        &mut result,
                        kernels,
                    );
                }
            }
        }

        // Set up gradient tracking if needed
        if (self.requires_grad() || other.requires_grad()) && crate::gradtrack::is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = crate::gradtrack::grad_fn::GradFn::MatMul {
                left_operand: Box::new(self.clone()),
                right_operand: Box::new(other.clone()),
                requires_grad: (self.requires_grad(), other.requires_grad()),
            };
            result.set_grad_fn(grad_fn.clone());

            // Register operation with gradtrack engine
            // Always register both operand IDs - the gradient function will handle which ones need gradients
            let input_ids = vec![self.id(), other.id()];
            crate::gradtrack::engine::GradEngine::register_operation(
                result.id(),
                input_ids,
                grad_fn,
            );
        }

        result
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

        assert_eq!(result.shape().dims(), vec![2, 2]);

        // Expected result: [[19, 22], [43, 50]]
        unsafe {
            let ptr = result.as_ptr();
            assert_eq!(*ptr.add(0), 19.0); // (0,0)
            assert_eq!(*ptr.add(1), 22.0); // (0,1)
            assert_eq!(*ptr.add(2), 43.0); // (1,0)
            assert_eq!(*ptr.add(3), 50.0); // (1,1)
        }
    }

    #[test]
    fn test_tall_skinny_vs_wide_correctness() {
        // Tall-skinny: [128, 8] @ [8, 16]
        let mut a_ts = Tensor::new(vec![128, 8]);
        let mut b_ts = Tensor::new(vec![8, 16]);
        for (i, v) in a_ts.data_mut().iter_mut().enumerate() {
            *v = (i as f32 * 0.01).sin();
        }
        for (i, v) in b_ts.data_mut().iter_mut().enumerate() {
            *v = (i as f32 * 0.02).cos();
        }
        let r_ts = a_ts.matmul(&b_ts);
        assert_eq!(r_ts.shape().dims(), vec![128, 16]);

        // Very-wide: [16, 8] @ [8, 256]
        let mut a_w = Tensor::new(vec![16, 8]);
        let mut b_w = Tensor::new(vec![8, 256]);
        for (i, v) in a_w.data_mut().iter_mut().enumerate() {
            *v = (i as f32 * 0.03).sin();
        }
        for (i, v) in b_w.data_mut().iter_mut().enumerate() {
            *v = (i as f32 * 0.04).cos();
        }
        let r_w = a_w.matmul(&b_w);
        assert_eq!(r_w.shape().dims(), vec![16, 256]);
    }

    /// Fast path: [1, K] @ [K, N] → [1, N]
    #[test]
    fn test_matmul_row_vector_times_matrix() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let b = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, // col-major? Our tensors are row-major [K,N]
                5.0, 6.0, 7.0, 8.0,
            ],
            vec![4, 2],
        )
        .unwrap();
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), vec![1, 2]);
        unsafe {
            let p = result.as_ptr();
            // [1,4] * [4,2]
            // col0 = 1*1 + 2*3 + 3*5 + 4*7 = 1 + 6 + 15 + 28 = 50
            // col1 = 1*2 + 2*4 + 3*6 + 4*8 = 2 + 8 + 18 + 32 = 60
            assert_eq!(*p.add(0), 50.0);
            assert_eq!(*p.add(1), 60.0);
        }
    }

    /// Fast path: [K] @ [K, N] → [N]
    #[test]
    fn test_matmul_1d_rowvec_times_matrix() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let r = a.matmul(&b);
        assert_eq!(r.shape().dims(), vec![2]);
        unsafe {
            let p = r.as_ptr();
            // [1,2,3] @ [[1,2],[3,4],[5,6]]
            // = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
            assert_eq!(*p.add(0), 22.0);
            assert_eq!(*p.add(1), 28.0);
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
        assert_eq!(result.shape().dims(), vec![2, 2]);

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

        let grad_a = a.grad_owned().unwrap();
        let grad_b = b.grad_owned().unwrap();

        assert_eq!(grad_a.shape().dims(), vec![2, 2]);
        assert_eq!(grad_b.shape().dims(), vec![2, 2]);

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
        assert_eq!(result.shape().dims(), vec![2]);

        result.backward(None);

        // Only b should have gradients
        assert!(a.grad_owned().is_none());
        let grad_b = b.grad_owned().unwrap();

        assert_eq!(grad_b.shape().dims(), vec![3, 2]);

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
            left.shape().dims(),
            left.data()
        );
        println!(
            "Right shape: {:?}, data: {:?}",
            right.shape().dims(),
            right.data()
        );

        // Forward pass
        let mut result = left.matmul(&right);
        println!(
            "Result shape: {:?}, data: {:?}",
            result.shape().dims(),
            result.data()
        );

        // Backward pass with ones
        let grad_ones = Tensor::ones(result.shape().dims().to_vec());
        println!(
            "Grad ones shape: {:?}, data: {:?}",
            grad_ones.shape().dims(),
            grad_ones.data()
        );

        result.backward(Some(grad_ones));

        let grad_left = left.grad_owned().unwrap();
        let grad_right = right.grad_owned().unwrap();

        println!(
            "Left gradient shape: {:?}, data: {:?}",
            grad_left.shape().dims(),
            grad_left.data()
        );
        println!(
            "Right gradient shape: {:?}, data: {:?}",
            grad_right.shape().dims(),
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

        let grad_ones = Tensor::ones(result.shape().dims().to_vec());
        result.backward(Some(grad_ones));

        let grad_left = left.grad_owned().unwrap();
        let grad_right = right.grad_owned().unwrap();

        println!("Left gradient: {:?}", grad_left.data());
        println!("Right gradient: {:?}", grad_right.data());

        // Manual calculation for verification
        println!("\n=== Manual verification ===");
        println!("Expected left grad batch 0: [0.5+1.0, 1.5+2.0] = [1.5, 3.5]");
        println!("Expected left grad batch 1: [2.5+3.0, 3.5+4.0] = [5.5, 7.5]");
    }

    #[test]
    fn test_alignment_checks_prevent_misaligned_access() {
        use crate::tensor::core::memory::{detect_runtime_simd, simd_alignment_bytes};

        // Create tensors that may not be aligned
        let a = Tensor::new(vec![4]);
        let b = Tensor::new(vec![4]);

        let kernels = MatMulKernels::get_cached_kernels();
        let simd_alignment = simd_alignment_bytes(detect_runtime_simd());

        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            // Check if pointers are aligned
            let a_aligned = (a_ptr as usize).is_multiple_of(simd_alignment);
            let b_aligned = (b_ptr as usize).is_multiple_of(simd_alignment);

            // The alignment check should correctly identify alignment status
            let alignment_check = kernels.check_alignment_for_simd(
                a_ptr,
                b_ptr,
                std::ptr::null_mut(),
                simd_alignment,
            );

            // The check should match our manual calculation
            assert_eq!(alignment_check, a_aligned && b_aligned);
        }
    }

    #[test]
    fn test_avx512_specific_alignment_validation() {
        use crate::tensor::core::memory::SimdLevel;

        let kernels = MatMulKernels::get_cached_kernels();

        // Test AVX512-specific 64-byte alignment requirements
        let test_ptr = 0x1004 as *const f32; // 4-byte aligned but not 64-byte aligned
        let avx512_aligned_ptr = 0x1040 as *const f32; // 64-byte aligned

        // 64-byte alignment check should fail for 4-byte aligned pointer
        let not_avx512_aligned =
            kernels.check_alignment_for_simd(test_ptr, test_ptr, std::ptr::null_mut(), 64);
        assert!(
            !not_avx512_aligned,
            "4-byte aligned pointer should not be considered 64-byte aligned"
        );

        // 64-byte alignment check should pass for properly aligned pointer
        let is_avx512_aligned = kernels.check_alignment_for_simd(
            avx512_aligned_ptr,
            avx512_aligned_ptr,
            std::ptr::null_mut(),
            64,
        );
        assert!(
            is_avx512_aligned,
            "64-byte aligned pointer should pass AVX512 alignment check"
        );

        // Verify that the SIMD level correctly maps to alignment requirements
        match kernels.simd_level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx512 => {
                // Should require 64-byte alignment
                assert_eq!(kernels.alignment, 64, "AVX512 should use 64-byte alignment");
            }
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => {
                // Should require 32-byte alignment
                assert_eq!(kernels.alignment, 32, "AVX2 should use 32-byte alignment");
            }
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Sse2 => {
                // Should require 16-byte alignment
                assert_eq!(kernels.alignment, 16, "SSE2 should use 16-byte alignment");
            }
            SimdLevel::Scalar => {
                // Should require at least 4-byte alignment
                assert!(
                    kernels.alignment >= 4,
                    "Scalar should use at least 4-byte alignment"
                );
            }
        }
    }

    #[test]
    fn test_comprehensive_alignment_management() {
        use crate::tensor::core::memory::{detect_runtime_simd, simd_alignment_bytes};

        // Test that alignment checks work correctly throughout the entire pipeline
        let kernels = MatMulKernels::get_cached_kernels();
        let simd_alignment = simd_alignment_bytes(detect_runtime_simd());

        // Create tensors - new tensors should be properly aligned
        let a = Tensor::new(vec![16]);
        let b = Tensor::new(vec![16]);

        // Verify that new tensors are aligned
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            let a_aligned = (a_ptr as usize).is_multiple_of(simd_alignment);
            let b_aligned = (b_ptr as usize).is_multiple_of(simd_alignment);

            // New tensors should be aligned
            assert!(a_aligned, "New tensor 'a' should be SIMD-aligned");
            assert!(b_aligned, "New tensor 'b' should be SIMD-aligned");

            // The helper function should correctly identify alignment (using a dummy c_ptr for testing)
            let dummy_c_ptr = a_ptr as *mut f32; // Use a_ptr as dummy since we're just testing alignment
            let alignment_check = kernels.check_actual_alignment(a_ptr, b_ptr, dummy_c_ptr);
            // Now that aligned kernels are enabled, the alignment check should be true here
            assert!(
                alignment_check,
                "Alignment check should pass for aligned pointers"
            );
        }

        // Test that contiguous() preserves alignment
        let a_transposed = a.transpose(0, 0); // Identity transpose, should still be contiguous
        let a_contiguous = a_transposed.contiguous();

        unsafe {
            let a_cont_ptr = a_contiguous.as_ptr();
            let a_cont_aligned = (a_cont_ptr as usize).is_multiple_of(simd_alignment);
            assert!(
                a_cont_aligned,
                "Contiguous tensor should maintain SIMD alignment"
            );
        }

        // Test actual matmul operations use correct kernels
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), vec![]); // Dot product result is scalar

        // Test that the system correctly handles both aligned and unaligned cases
        // by ensuring no crashes occur and results are computed correctly
        let a_2d = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b_2d = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let result_2d = a_2d.matmul(&b_2d);

        assert_eq!(result_2d.shape().dims(), vec![2, 2]);
        // Verify computation correctness: [1,2; 3,4] @ [5,6; 7,8] = [19,22; 43,50]
        assert!((result_2d.get(&[0, 0]) - 19.0).abs() < 1e-6);
        assert!((result_2d.get(&[0, 1]) - 22.0).abs() < 1e-6);
        assert!((result_2d.get(&[1, 0]) - 43.0).abs() < 1e-6);
        assert!((result_2d.get(&[1, 1]) - 50.0).abs() < 1e-6);
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
        let grad_bias = bias.grad_owned().unwrap();
        let grad_weight = weight.grad_owned().unwrap();

        assert_eq!(grad_weight.shape().dims(), vec![3, 2]); // Same shape as weight
        assert_eq!(grad_bias.shape().dims(), vec![2]); // Same shape as bias

        // The exact gradient values depend on the computation graph, but shapes should be correct
        assert_eq!(grad_weight.size(), 6);
        assert_eq!(grad_bias.size(), 2);

        // Verify that no gradient is computed for x_data (doesn't require grad)
        assert!(x_data.grad_owned().is_none());
    }

    #[test]
    fn test_debug_large_matmul_gradient() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        println!("=== Debug Large MatMul Gradient Issue ===");

        // Test progressively larger sizes to find where the issue starts
        for &size in &[4, 8, 16, 32, 64] {
            println!("\n--- Testing size {}x{} ---", size, size);

            let left = Tensor::from_slice(
                &(0..size * size)
                    .map(|i| (i as f32) * 0.1 + 1.0)
                    .collect::<Vec<_>>(),
                vec![size, size],
            )
            .unwrap()
            .with_requires_grad();

            let right = Tensor::from_slice(
                &(0..size * size)
                    .map(|i| (i as f32) * 0.2 + 0.5)
                    .collect::<Vec<_>>(),
                vec![size, size],
            )
            .unwrap()
            .with_requires_grad();

            let mut result = left.matmul(&right);

            // Backward with ones
            result.backward(None);

            let grad_left = left.grad_owned().unwrap();
            let _grad_right = right.grad_owned().unwrap();

            // Manual verification
            // For C = A @ B, dC/dA = grad_output @ B^T, dC/dB = A^T @ grad_output
            let right_t = right.transpose(0, 1);
            let grad_ones = Tensor::ones(vec![size, size]);
            let expected_grad_left = grad_ones.matmul(&right_t);

            // Check if our gradient matches expected
            let mut max_diff = 0.0f32;
            let mut max_diff_idx = 0;
            for i in 0..grad_left.size() {
                let our_val = unsafe { *grad_left.as_ptr().add(i) };
                let expected_val = unsafe { *expected_grad_left.as_ptr().add(i) };
                let diff = (our_val - expected_val).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_idx = i;
                }
            }

            println!(
                "Max gradient diff: {} at index {} (size {}x{})",
                max_diff, max_diff_idx, size, size
            );

            if max_diff > 1e-4 {
                println!("PROBLEM DETECTED at size {}x{}", size, size);
                let our_val = unsafe { *grad_left.as_ptr().add(max_diff_idx) };
                let expected_val = unsafe { *expected_grad_left.as_ptr().add(max_diff_idx) };
                println!(
                    "  our_val={}, expected_val={}, diff={}",
                    our_val, expected_val, max_diff
                );

                // Check if the issue is in transpose or contiguous
                println!("  right.is_contiguous(): {}", right.is_contiguous());
                println!("  right_t.is_contiguous(): {}", right_t.is_contiguous());

                break;
            }

            clear_gradients();
        }
    }

    #[test]
    fn test_debug_transpose_contiguous_issue() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        println!("=== Debug Transpose/Contiguous Issue ===");

        let size = 8;
        let right = Tensor::from_slice(
            &(0..size * size)
                .map(|i| (i as f32) * 0.2 + 0.5)
                .collect::<Vec<_>>(),
            vec![size, size],
        )
        .unwrap();

        println!("Original right tensor:");
        println!("  is_contiguous: {}", right.is_contiguous());
        println!("  strides: {:?}", right.strides());

        let right_t = right.transpose(0, 1);
        println!("Transposed right tensor:");
        println!("  is_contiguous: {}", right_t.is_contiguous());
        println!("  strides: {:?}", right_t.strides());

        let right_t_contiguous = if right_t.is_contiguous() {
            right_t.clone()
        } else {
            right_t.contiguous()
        };
        println!("Contiguous transposed right tensor:");
        println!("  is_contiguous: {}", right_t_contiguous.is_contiguous());
        println!("  strides: {:?}", right_t_contiguous.strides());

        // Compare the data to make sure contiguous() works correctly
        println!("Original data (first 8): {:?}", &right.data()[0..8]);
        println!(
            "Transposed data (first 8): {:?}",
            &right_t_contiguous.data()[0..8]
        );

        // Manual transpose verification
        for i in 0..4 {
            for j in 0..4 {
                let orig_val = right.get(&[i, j]);
                let trans_val = right_t_contiguous.get(&[j, i]);
                if (orig_val - trans_val).abs() > 1e-6 {
                    println!(
                        "Transpose error at ({},{}): orig={}, trans={}",
                        i, j, orig_val, trans_val
                    );
                }
            }
        }
    }

    #[test]
    fn test_debug_scalar_kernel_accuracy() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        println!("=== Debug Scalar Kernel Accuracy ===");

        // Test the scalar kernel directly with known values
        let size = 3;
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b_data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // Identity matrix
        let mut c_data = vec![0.0; 9];

        unsafe {
            crate::tensor::ops::matmul::scalar_kernels::matmul_scalar_2d_2d(
                a_data.as_ptr(),
                b_data.as_ptr(),
                c_data.as_mut_ptr(),
                size,
                size,
                size,
                size,
                1, // a strides
                size,
                1, // b strides
                size,
                1, // c strides
            );
        }

        println!("A @ I = {:?}", c_data);
        println!("Expected: {:?}", a_data);

        // Should be identical since A @ I = A
        for (i, (&expected, &actual)) in a_data.iter().zip(c_data.iter()).enumerate() {
            if (expected - actual).abs() > 1e-6 {
                println!(
                    "Scalar kernel error at index {}: expected={}, actual={}",
                    i, expected, actual
                );
            }
        }

        // Now test a more complex case
        let a_data2 = [1.0, 2.0, 3.0, 4.0];
        let b_data2 = [2.0, 0.0, 0.0, 2.0];
        let mut c_data2 = vec![0.0; 4];

        unsafe {
            crate::tensor::ops::matmul::scalar_kernels::matmul_scalar_2d_2d(
                a_data2.as_ptr(),
                b_data2.as_ptr(),
                c_data2.as_mut_ptr(),
                2,
                2,
                2,
                2,
                1, // a strides
                2,
                1, // b strides
                2,
                1, // c strides
            );
        }

        println!("[[1,2],[3,4]] @ [[2,0],[0,2]] = {:?}", c_data2);
        println!("Expected: [2.0, 4.0, 6.0, 8.0]");
    }

    #[test]
    fn test_minimal_noncontiguous_gradient() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        println!("=== Minimal Non-contiguous Gradient Test ===");

        // Create simple tensors
        let left = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .unwrap()
            .with_requires_grad();

        let right = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2])
            .unwrap()
            .with_requires_grad();

        println!("Left requires_grad: {}", left.requires_grad());
        println!("Right requires_grad: {}", right.requires_grad());

        // Make non-contiguous
        let left_nc = left.transpose(0, 1).transpose(0, 1);
        let right_nc = right.transpose(0, 1).transpose(0, 1);

        println!("Left NC requires_grad: {}", left_nc.requires_grad());
        println!("Right NC requires_grad: {}", right_nc.requires_grad());
        println!("Left NC is_contiguous: {}", left_nc.is_contiguous());
        println!("Right NC is_contiguous: {}", right_nc.is_contiguous());

        // Matmul
        let mut result = left_nc.matmul(&right_nc);
        println!("Result requires_grad: {}", result.requires_grad());

        // Backward
        result.backward(None);

        // Check gradients
        println!(
            "Left NC gradient exists: {}",
            left_nc.grad_owned().is_some()
        );
        println!(
            "Right NC gradient exists: {}",
            right_nc.grad_owned().is_some()
        );

        // Also check original tensors
        println!(
            "Original left gradient exists: {}",
            left.grad_owned().is_some()
        );
        println!(
            "Original right gradient exists: {}",
            right.grad_owned().is_some()
        );
    }

    #[test]
    fn test_debug_transpose_gradient_tracking() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        println!("=== Debug Transpose Gradient Tracking ===");

        // Create a tensor with requires_grad
        let original = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();

        println!("Original requires_grad: {}", original.requires_grad());

        // Single transpose
        let t1 = original.transpose(0, 1);
        println!(
            "After first transpose requires_grad: {}",
            t1.requires_grad()
        );

        // Double transpose (should be identity)
        let t2 = t1.transpose(0, 1);
        println!(
            "After second transpose requires_grad: {}",
            t2.requires_grad()
        );

        // Test the pattern from the failing test
        let left = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
        )
        .unwrap()
        .with_requires_grad();

        println!("Left original requires_grad: {}", left.requires_grad());

        // Make non-contiguous like in the test
        let left_nc = left.transpose(1, 2).transpose(1, 2);
        println!(
            "Left non-contiguous requires_grad: {}",
            left_nc.requires_grad()
        );
        println!(
            "Left non-contiguous is_contiguous: {}",
            left_nc.is_contiguous()
        );

        // Test matmul
        let right = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 3, 2],
        )
        .unwrap()
        .with_requires_grad();

        let right_nc = right.transpose(1, 2).transpose(1, 2);
        println!(
            "Right non-contiguous requires_grad: {}",
            right_nc.requires_grad()
        );
        println!(
            "Right non-contiguous is_contiguous: {}",
            right_nc.is_contiguous()
        );

        let mut result = left_nc.matmul(&right_nc);
        println!("Result requires_grad: {}", result.requires_grad());

        // Backward
        result.backward(None);

        // Check gradients
        println!("Left gradient exists: {}", left_nc.grad_owned().is_some());
        println!("Right gradient exists: {}", right_nc.grad_owned().is_some());

        if let Some(grad_left) = left_nc.grad_owned() {
            println!("Left gradient shape: {:?}", grad_left.shape().dims());
        }
        if let Some(grad_right) = right_nc.grad_owned() {
            println!("Right gradient shape: {:?}", grad_right.shape().dims());
        }
    }

    #[test]
    fn test_debug_4d_gradient_issue() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        println!("=== Debug 4D Gradient Issue ===");

        // Test case: 4D: 2x3x4x5 @ 2x3x5x6 gradients (failing case)
        let left_shape = vec![2, 3, 4, 5];
        let right_shape = vec![2, 3, 5, 6];

        let left_size: usize = left_shape.iter().product();
        let right_size: usize = right_shape.iter().product();

        let left_data: Vec<f32> = (0..left_size).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let right_data: Vec<f32> = (0..right_size).map(|i| (i as f32) * 0.2 + 0.5).collect();

        let left = Tensor::from_slice(&left_data, left_shape.clone())
            .unwrap()
            .with_requires_grad();
        let right = Tensor::from_slice(&right_data, right_shape.clone())
            .unwrap()
            .with_requires_grad();

        println!("Left shape: {:?}", left.shape().dims());
        println!("Right shape: {:?}", right.shape().dims());

        // Check if forward pass is correct first
        let result_no_grad = crate::gradtrack::with_no_grad(|| left.matmul(&right));
        println!("Forward result shape: {:?}", result_no_grad.shape().dims());
        println!("Forward result[0] = {}", result_no_grad.data()[0]);

        // Forward pass with gradients
        let mut result = left.matmul(&right);
        println!("Result shape: {:?}", result.shape().dims());

        // Check if forward results match
        let forward_diff = (result.data()[0] - result_no_grad.data()[0]).abs();
        println!("Forward pass difference: {}", forward_diff);

        // Backward pass
        result.backward(None);

        let grad_left = left.grad_owned().unwrap();

        println!("Left gradient shape: {:?}", grad_left.shape().dims());
        println!("Left gradient[40] = {}", grad_left.data()[40]);
        println!("Expected: ~78, Got: {}", grad_left.data()[40]);

        // Check if gradient is all zeros (which would indicate a major issue)
        let grad_sum: f32 = grad_left.data().iter().sum();
        println!("Gradient sum: {} (should be non-zero)", grad_sum);

        if grad_sum.abs() < 1e-6 {
            println!("ERROR: Gradient is essentially zero - major computation issue!");
        }

        // Manual gradient computation for verification
        println!("\n=== Manual Verification ===");

        // For C = A @ B, grad_A = grad_output @ B.T
        let grad_ones = Tensor::ones(result.shape().dims().to_vec());
        println!("Grad ones shape: {:?}", grad_ones.shape().dims());

        // Transpose the last two dimensions of right
        let right_rank = right.shape().dims().len();
        let right_t = right.transpose(right_rank - 2, right_rank - 1);
        println!("Right transposed shape: {:?}", right_t.shape().dims());

        let manual_grad_left =
            crate::gradtrack::with_no_grad(|| grad_ones.matmul(&right_t.contiguous()));
        println!(
            "Manual grad left shape: {:?}",
            manual_grad_left.shape().dims()
        );
        println!("Manual grad left[40] = {}", manual_grad_left.data()[40]);

        // Check if they match
        let diff = (grad_left.data()[40] - manual_grad_left.data()[40]).abs();
        println!("Difference at element 40: {}", diff);

        if diff > 1e-3 {
            println!("MAJOR GRADIENT ERROR DETECTED!");
            println!(
                "Expected (manual): {}, Got (automatic): {}",
                manual_grad_left.data()[40],
                grad_left.data()[40]
            );

            // Check if the issue is in the gradient computation or the reduce function
            println!("\n=== Investigating Root Cause ===");
            println!(
                "Manual gradient sum: {}",
                manual_grad_left.data().iter().sum::<f32>()
            );
            println!(
                "Automatic gradient sum: {}",
                grad_left.data().iter().sum::<f32>()
            );
        } else {
            println!("Gradients match!");
        }
    }

    #[test]
    fn test_debug_matmul_with_known_values() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        println!("=== Debug MatMul with Known Values ===");

        // Use simple values that should produce exact results
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]) // Identity
            .unwrap()
            .with_requires_grad();

        println!("A: {:?}", a.data());
        println!("B (identity): {:?}", b.data());

        let mut result = a.matmul(&b);
        println!("A @ B: {:?}", result.data());
        println!("Expected: {:?}", a.data()); // Should be same as A since B is identity

        // Backward pass
        result.backward(None);

        let grad_a = a.grad_owned().unwrap();
        let grad_b = b.grad_owned().unwrap();

        println!("grad_A: {:?}", grad_a.data());
        println!("grad_B: {:?}", grad_b.data());

        // For A @ B where B is identity:
        // grad_A = grad_output @ B^T = grad_output @ I = grad_output = ones([2,2])
        // grad_B = A^T @ grad_output = A^T @ ones([2,2])

        let expected_grad_a = vec![1.0, 1.0, 1.0, 1.0];
        let expected_grad_b = vec![4.0, 6.0, 4.0, 6.0]; // A^T @ ones = [1+3, 2+4, 1+3, 2+4] = [4,6,4,6]

        println!("Expected grad_A: {:?}", expected_grad_a);
        println!("Expected grad_B: {:?}", expected_grad_b);

        for (i, (&expected, &actual)) in
            expected_grad_a.iter().zip(grad_a.data().iter()).enumerate()
        {
            if (expected - actual).abs() > 1e-5 {
                println!(
                    "grad_A error at index {}: expected={}, actual={}, diff={}",
                    i,
                    expected,
                    actual,
                    (expected - actual).abs()
                );
            }
        }

        for (i, (&expected, &actual)) in
            expected_grad_b.iter().zip(grad_b.data().iter()).enumerate()
        {
            if (expected - actual).abs() > 1e-5 {
                println!(
                    "grad_B error at index {}: expected={}, actual={}, diff={}",
                    i,
                    expected,
                    actual,
                    (expected - actual).abs()
                );
            }
        }
    }

    /// New: 1D @ ND batched vector broadcasting forward and backward
    #[test]
    fn test_matmul_1d_nd_broadcast_forward_backward() {
        // [K] @ [B, K, N] -> [B, N]
        let k = 4;
        let b = 2;
        let n = 3;
        let left_data: Vec<f32> = (0..k).map(|i| i as f32 + 1.0).collect(); // [1,2,3,4]
        let right_data: Vec<f32> = (0..b * k * n).map(|i| (i as f32) * 0.1 + 0.5).collect();

        let left = Tensor::from_slice(&left_data, vec![k])
            .unwrap()
            .with_requires_grad();
        let right = Tensor::from_slice(&right_data, vec![b, k, n])
            .unwrap()
            .with_requires_grad();

        let mut out = left.matmul(&right);
        assert_eq!(out.shape().dims(), vec![b, n]);

        // Backward with ones
        out.backward(None);
        let grad_left = left.grad_owned().unwrap();
        let grad_right = right.grad_owned().unwrap();

        assert_eq!(grad_left.shape().dims(), vec![k]);
        assert_eq!(grad_right.shape().dims(), vec![b, k, n]);
    }

    /// New: ND @ 1D batched matrix-vector forward and backward
    #[test]
    fn test_matmul_nd_1d_broadcast_forward_backward() {
        // [B, M, K] @ [K] -> [B, M]
        let b = 3;
        let m = 2;
        let k = 5;
        let left_data: Vec<f32> = (0..b * m * k).map(|i| (i as f32) * 0.05 + 1.0).collect();
        let right_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.2 + 0.5).collect();

        let left = Tensor::from_slice(&left_data, vec![b, m, k])
            .unwrap()
            .with_requires_grad();
        let right = Tensor::from_slice(&right_data, vec![k])
            .unwrap()
            .with_requires_grad();

        let mut out = left.matmul(&right);
        assert_eq!(out.shape().dims(), vec![b, m]);

        // Backward with ones
        out.backward(None);
        let grad_left = left.grad_owned().unwrap();
        let grad_right = right.grad_owned().unwrap();

        assert_eq!(grad_left.shape().dims(), vec![b, m, k]);
        assert_eq!(grad_right.shape().dims(), vec![k]);
    }

    #[test]
    fn test_batched_mat_vec_forward_and_shapes() {
        // [B, M, K] @ [B, K] -> [B, M]
        let b = 3usize;
        let m = 5usize;
        let k = 7usize;
        let left_data: Vec<f32> = (0..b * m * k).map(|i| (i as f32) * 0.01 + 1.0).collect();
        let right_data: Vec<f32> = (0..b * k).map(|i| (i as f32) * 0.02 + 0.5).collect();
        let left = Tensor::from_slice(&left_data, vec![b, m, k]).unwrap();
        let right = Tensor::from_slice(&right_data, vec![b, k]).unwrap();
        let out = left.matmul(&right);
        assert_eq!(out.shape().dims(), vec![b, m]);
    }

    #[test]
    fn test_batched_mat_vec_numeric_small() {
        // Small numeric check: [2, 2, 3] @ [2, 3]
        let left = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, // b0 r0
                4.0, 5.0, 6.0, // b0 r1
                1.0, 1.0, 1.0, // b1 r0
                2.0, 2.0, 2.0, // b1 r1
            ],
            vec![2, 2, 3],
        )
        .unwrap();
        let right = Tensor::from_slice(&[1.0, 1.0, 1.0, 2.0, 3.0, 4.0], vec![2, 3]).unwrap();
        let out = left.matmul(&right); // [2,2]
        assert_eq!(out.shape().dims(), vec![2, 2]);
        unsafe {
            // batch 0: [2,3] @ [3] = [6, 15]
            assert!((*out.as_ptr() - 6.0).abs() < 1e-6);
            assert!((*out.as_ptr().add(1) - 15.0).abs() < 1e-6);
            // batch 1: [2,3] @ [3] = [9, 18]
            assert!((*out.as_ptr().add(2) - 9.0).abs() < 1e-6);
            assert!((*out.as_ptr().add(3) - 18.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batched_vector_cases_grad_shapes() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // ND @ vec
        let left = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2])
            .unwrap()
            .with_requires_grad();
        let right = Tensor::from_slice(&[1.0, 1.0], vec![1, 2])
            .unwrap()
            .with_requires_grad();
        let mut out = left.matmul(&right);
        assert_eq!(out.shape().dims(), vec![1, 2]);
        out.backward(None);
        let gl = left.grad_owned().unwrap();
        let gr = right.grad_owned().unwrap();
        assert_eq!(gl.shape().dims(), vec![1, 2, 2]);
        assert_eq!(gr.shape().dims(), vec![1, 2]);

        clear_gradients();
        // vec @ ND
        let left2 = Tensor::from_slice(&[1.0, 1.0], vec![2])
            .unwrap()
            .with_requires_grad();
        let right2 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .unwrap()
            .with_requires_grad();
        let mut out2 = left2.matmul(&right2);
        assert_eq!(out2.shape().dims(), vec![2]);
        out2.backward(None);
        let gl2 = left2.grad_owned().unwrap();
        let gr2 = right2.grad_owned().unwrap();
        assert_eq!(gl2.shape().dims(), vec![2]);
        assert_eq!(gr2.shape().dims(), vec![2, 2]);
    }

    #[test]
    fn test_broadcast_matmul_gradient_complex_case() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        println!("=== Testing complex broadcast case: [1,2,2,1,4,5] @ [2,1,5,6] ===");

        // The exact failing case from the issue
        let left_shape = vec![1, 2, 2, 1, 4, 5];
        let right_shape = vec![2, 1, 5, 6];
        let expected_result_shape = vec![1, 2, 2, 1, 4, 6];

        println!("Left shape: {:?}", left_shape);
        println!("Right shape: {:?}", right_shape);
        println!("Expected result shape: {:?}", expected_result_shape);

        // Create tensors with some data
        let left_size: usize = left_shape.iter().product();
        let right_size: usize = right_shape.iter().product();

        let left_data: Vec<f32> = (0..left_size).map(|i| (i as f32) * 0.01 + 1.0).collect();
        let right_data: Vec<f32> = (0..right_size).map(|i| (i as f32) * 0.02 + 0.5).collect();

        let left = Tensor::from_slice(&left_data, left_shape.clone())
            .unwrap()
            .with_requires_grad();
        let right = Tensor::from_slice(&right_data, right_shape.clone())
            .unwrap()
            .with_requires_grad();

        // Forward pass
        let mut result = left.matmul(&right);
        println!("Actual result shape: {:?}", result.shape().dims());

        // Verify forward result shape
        assert_eq!(
            result.shape().dims(),
            expected_result_shape,
            "Forward result shape mismatch"
        );

        // Backward pass
        result.backward(None);

        // Check gradient shapes
        let grad_left = left.grad_owned().unwrap();
        let grad_right = right.grad_owned().unwrap();

        println!("Left gradient shape: {:?}", grad_left.shape().dims());
        println!("Right gradient shape: {:?}", grad_right.shape().dims());

        // Gradients should have the same shape as the original tensors
        assert_eq!(
            grad_left.shape().dims(),
            left_shape,
            "Left gradient shape mismatch"
        );
        assert_eq!(
            grad_right.shape().dims(),
            right_shape,
            "Right gradient shape mismatch"
        );

        // Check that gradients are not all zeros (sanity check)
        let left_grad_sum: f32 = grad_left.data().iter().sum();
        let right_grad_sum: f32 = grad_right.data().iter().sum();

        println!("Left gradient sum: {}", left_grad_sum);
        println!("Right gradient sum: {}", right_grad_sum);

        assert!(
            left_grad_sum.abs() > 1e-6,
            "Left gradient should not be zero"
        );
        assert!(
            right_grad_sum.abs() > 1e-6,
            "Right gradient should not be zero"
        );

        println!("✓ Complex broadcast matmul gradient test passed!");
    }

    #[test]
    fn test_grad_left_vec_at_bkn_manual() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();
        let k = 8usize;
        let b = 3usize;
        let n = 5usize;

        // left vector [K]
        let left = Tensor::from_slice(
            &(0..k).map(|i| i as f32 * 0.1 + 1.0).collect::<Vec<_>>(),
            vec![k],
        )
        .unwrap()
        .with_requires_grad();

        // right [B,K,N]
        let right = Tensor::from_slice(
            &(0..b * k * n)
                .map(|i| i as f32 * 0.2 + 0.5)
                .collect::<Vec<_>>(),
            vec![b, k, n],
        )
        .unwrap()
        .with_requires_grad();

        let mut out = left.matmul(&right); // [B,N]
                                           // Backward with ones
        out.backward(None);

        // Compute manual expected grad for left: sum_{b,n} right[b,k,n]
        let grad_left = left.grad_owned().unwrap();
        assert_eq!(grad_left.shape().dims(), vec![k]);
        let rp = unsafe { right.as_ptr() };
        for kk in 0..k {
            let mut sum = 0.0f32;
            for bb in 0..b {
                for nn in 0..n {
                    let idx = bb * (k * n) + kk * n + nn;
                    unsafe { sum += *rp.add(idx) };
                }
            }
            let got = unsafe { *grad_left.as_ptr().add(kk) };
            assert!(
                (got - sum).abs() < 1e-4,
                "k={} got={} expected={}",
                kk,
                got,
                sum
            );
        }
    }
}
