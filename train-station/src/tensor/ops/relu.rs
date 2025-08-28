//! ReLU activation function
//!
//! Provides the Rectified Linear Unit activation function following PyTorch conventions with
//! comprehensive automatic differentiation support and SIMD-optimized computation.
//!
//! # Key Features
//!
//! - **ReLU Activation**: `relu()` - Computes max(0, x) for each element (PyTorch `relu()` equivalent)
//! - **Automatic Differentiation**: Full gradtrack support with efficient gradient computation
//! - **SIMD Optimization**: AVX2-optimized implementation for maximum performance
//! - **Cache Optimization**: Memory access patterns optimized for modern CPUs
//! - **Mathematical Accuracy**: High-precision activation computation
//! - **Branch Prediction**: Optimized conditional logic for modern CPUs
//!
//! # Mathematical Properties
//!
//! The ReLU activation function has the following properties:
//! - **Definition**: f(x) = max(0, x)
//! - **Range**: [0, ∞) - outputs are always non-negative
//! - **Monotonicity**: Strictly increasing for x > 0
//! - **Continuity**: Continuous everywhere, differentiable everywhere except at x = 0
//! - **Gradient**: f'(x) = 1 if x > 0, f'(x) = 0 if x ≤ 0
//! - **Sparsity**: Produces sparse activations (many zeros) for negative inputs
//!
//! # Performance Characteristics
//!
//! - **SIMD Optimization**: AVX2-optimized with 32-element blocks and 4x unrolling
//! - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
//! - **Cache-friendly Access**: Linear memory access patterns
//! - **Branch Prediction**: Optimized conditional logic for modern CPUs
//! - **Gradient Optimization**: Efficient gradtrack with NoGradTrack support

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl Tensor {
    /// Element-wise ReLU (Rectified Linear Unit) activation.
    ///
    /// Applies ReLU to each element: `output[i] = max(0, self[i])`
    ///
    /// # Returns
    /// A new tensor with ReLU applied to each element
    ///
    /// # Examples
    ///
    /// ## Basic ReLU Activation
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[-1.0, 0.0, 2.5], vec![3]).unwrap();
    /// let b = a.relu();
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 0.0); // max(0, -1.0) = 0.0
    /// assert_eq!(b.get(&[1]), 0.0); // max(0, 0.0) = 0.0
    /// assert_eq!(b.get(&[2]), 2.5); // max(0, 2.5) = 2.5
    /// ```
    ///
    /// ## Mixed Positive and Negative Values
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[-5.0, -0.1, 0.0, 0.1, 5.0], vec![5]).unwrap();
    /// let b = a.relu();
    /// assert_eq!(b.shape().dims, vec![5]);
    /// assert_eq!(b.get(&[0]), 0.0); // max(0, -5.0) = 0.0
    /// assert_eq!(b.get(&[1]), 0.0); // max(0, -0.1) = 0.0
    /// assert_eq!(b.get(&[2]), 0.0); // max(0, 0.0) = 0.0
    /// assert_eq!(b.get(&[3]), 0.1); // max(0, 0.1) = 0.1
    /// assert_eq!(b.get(&[4]), 5.0); // max(0, 5.0) = 5.0
    /// ```
    pub fn relu(&self) -> Tensor {
        let mut out = self.relu_optimized();

        if self.requires_grad() && is_grad_enabled() {
            out.set_requires_grad_internal(true);
            let grad_fn = GradFn::Relu {
                saved_input: Box::new(self.clone()),
            };
            out.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(out.id(), vec![self.id()], grad_fn);
        }

        out
    }

    /// Internal optimized ReLU operation
    ///
    /// Performs element-wise ReLU activation using SIMD optimization when available
    /// and falling back to optimized scalar computation. This is the core implementation
    /// used by `relu()`.
    ///
    /// # Returns
    ///
    /// A new tensor containing the ReLU activation of each element
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks when available
    /// - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Branch Prediction**: Optimized conditional logic for modern CPUs
    /// - **Mathematical Accuracy**: High-precision activation computation
    /// - **Zero-sized Handling**: Fast return for empty tensors
    ///
    /// # Implementation Details
    ///
    /// Automatically selects between SIMD and scalar implementations based on hardware
    /// capabilities. SIMD implementation uses AVX2 vector max operations for optimal
    /// performance. Scalar implementation uses 4x unrolling for better instruction-level
    /// parallelism.
    #[inline]
    pub(crate) fn relu_optimized(&self) -> Tensor {
        let mut output = Tensor::new(self.shape().dims.clone());

        if self.size() == 0 {
            return output;
        }

        unsafe {
            let src = self.as_ptr();
            let dst = output.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    self.relu_simd_avx2_optimized(src, dst);
                    return output;
                }
            }

            // Scalar fallback
            self.relu_scalar_optimized(src, dst);
        }

        output
    }

    /// AVX2-optimized ReLU implementation
    ///
    /// Performs element-wise ReLU activation using AVX2 SIMD instructions for maximum
    /// performance on x86_64 architectures with AVX2 support.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    ///
    /// # Safety
    ///
    /// Requires valid pointers with sufficient memory for the tensor size.
    /// All pointers must point to valid tensor data. Requires AVX2 support.
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Processing**: 32 elements per iteration with 4x unrolling
    /// - **Memory Access**: Linear access patterns for cache efficiency
    /// - **Vector Operations**: Uses AVX2 max instructions for ReLU computation
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Hardware Requirements**: Requires x86_64 with AVX2 support
    ///
    /// # Implementation Details
    ///
    /// Uses AVX2 vector max operations to compute max(0, x) efficiently.
    /// Implements 4x unrolling for optimal instruction throughput and cache utilization.
    /// Processes remaining elements with scalar operations for complete coverage.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn relu_simd_avx2_optimized(&self, src: *const f32, dst: *mut f32) {
        let size = self.size();
        let zero_vec = _mm256_setzero_ps();
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let mut offset = 0;

        // Unrolled SIMD loop for maximum throughput
        for _ in 0..simd_count {
            // Process 4 AVX2 vectors (32 elements) per iteration
            let src_vec1 = _mm256_loadu_ps(src.add(offset));
            let relu_vec1 = _mm256_max_ps(src_vec1, zero_vec);
            _mm256_storeu_ps(dst.add(offset), relu_vec1);

            let src_vec2 = _mm256_loadu_ps(src.add(offset + 8));
            let relu_vec2 = _mm256_max_ps(src_vec2, zero_vec);
            _mm256_storeu_ps(dst.add(offset + 8), relu_vec2);

            let src_vec3 = _mm256_loadu_ps(src.add(offset + 16));
            let relu_vec3 = _mm256_max_ps(src_vec3, zero_vec);
            _mm256_storeu_ps(dst.add(offset + 16), relu_vec3);

            let src_vec4 = _mm256_loadu_ps(src.add(offset + 24));
            let relu_vec4 = _mm256_max_ps(src_vec4, zero_vec);
            _mm256_storeu_ps(dst.add(offset + 24), relu_vec4);

            offset += 32;
        }

        // Handle remaining 8-element blocks
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            let src_vec = _mm256_loadu_ps(src.add(offset));
            let relu_vec = _mm256_max_ps(src_vec, zero_vec);
            _mm256_storeu_ps(dst.add(offset), relu_vec);
            offset += 8;
        }

        // Handle remaining elements with scalar fallback
        for i in offset..size {
            let v = *src.add(i);
            *dst.add(i) = if v > 0.0 { v } else { 0.0 };
        }
    }

    /// Optimized scalar ReLU fallback
    ///
    /// Performs element-wise ReLU activation using optimized scalar operations with
    /// 4x unrolling for better instruction-level parallelism and cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    ///
    /// # Safety
    ///
    /// Requires valid pointers with sufficient memory for the tensor size.
    /// All pointers must point to valid tensor data.
    ///
    /// # Performance Characteristics
    ///
    /// - **Unrolling**: 4x unrolling for instruction-level parallelism
    /// - **Memory Access**: Linear access patterns for cache efficiency
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Cache Optimization**: Optimized for modern CPU cache hierarchies
    /// - **Mathematical Accuracy**: High-precision scalar ReLU computation
    ///
    /// # Implementation Details
    ///
    /// Uses 4x unrolled scalar operations for optimal performance on non-SIMD hardware.
    /// Processes elements in groups of 4 to improve instruction-level parallelism
    /// and reduce loop overhead.
    #[inline]
    unsafe fn relu_scalar_optimized(&self, src: *const f32, dst: *mut f32) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar loop for better performance
        for _ in 0..unroll_count {
            let v1 = *src.add(offset);
            let v2 = *src.add(offset + 1);
            let v3 = *src.add(offset + 2);
            let v4 = *src.add(offset + 3);

            *dst.add(offset) = if v1 > 0.0 { v1 } else { 0.0 };
            *dst.add(offset + 1) = if v2 > 0.0 { v2 } else { 0.0 };
            *dst.add(offset + 2) = if v3 > 0.0 { v3 } else { 0.0 };
            *dst.add(offset + 3) = if v4 > 0.0 { v4 } else { 0.0 };

            offset += 4;
        }

        // Handle remaining elements
        for i in offset..size {
            let v = *src.add(i);
            *dst.add(i) = if v > 0.0 { v } else { 0.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward_basic() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 2.5], vec![3]).unwrap();
        let y = x.relu();
        unsafe {
            assert_eq!(*y.as_ptr(), 0.0);
            assert_eq!(*y.as_ptr().add(1), 0.0);
            assert_eq!(*y.as_ptr().add(2), 2.5);
        }
    }
}
