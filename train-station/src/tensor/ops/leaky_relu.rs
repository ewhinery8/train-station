//! Leaky ReLU activation operations for tensors
//!
//! Provides the Leaky ReLU activation function following PyTorch conventions with
//! comprehensive automatic differentiation support and SIMD-optimized computation.
//!
//! # Key Features
//!
//! - **Leaky ReLU Activation**: `leaky_relu(negative_slope)` - Computes max(0, x) + negative_slope * min(0, x) (PyTorch `leaky_relu()` equivalent)
//! - **Automatic Differentiation**: Full gradtrack support with efficient gradient computation
//! - **SIMD Optimization**: AVX2-optimized implementation for maximum performance
//! - **Scalar Fallback**: Optimized scalar implementation for non-SIMD hardware
//! - **Cache Optimization**: Memory access patterns optimized for modern CPUs
//! - **Mathematical Accuracy**: High-precision activation computation
//!
//! # Mathematical Properties
//!
//! The Leaky ReLU function f(x) = max(0, x) + negative_slope * min(0, x) has the following properties:
//! - For x > 0: f(x) = x (identity function)
//! - For x ≤ 0: f(x) = negative_slope * x (small negative gradient)
//! - Gradient: f'(x) = 1 for x > 0, f'(x) = negative_slope for x ≤ 0
//! - Continuous at x = 0: f(0) = 0
//! - Monotonic: f'(x) > 0 for all x (when negative_slope > 0)
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
    /// Element-wise Leaky ReLU activation.
    ///
    /// Applies Leaky ReLU to each element: `output[i] = max(0, x) + negative_slope * min(0, x)`
    ///
    /// Unlike standard ReLU, allows a small gradient when the unit is not active.
    ///
    /// # Arguments
    /// * `negative_slope` - Slope for negative values (typically small, e.g., 0.01 or 0.1)
    ///
    /// # Returns
    /// A new tensor with Leaky ReLU applied to each element
    ///
    /// # Examples
    ///
    /// ## Basic Leaky ReLU
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0], vec![4]).unwrap();
    /// let b = a.leaky_relu(0.1);
    /// assert_eq!(b.shape().dims, vec![4]);
    /// assert!((b.get(&[0]) - (-0.2)).abs() < 1e-6); // -2.0 * 0.1 = -0.2
    /// assert!((b.get(&[1]) - (-0.1)).abs() < 1e-6); // -1.0 * 0.1 = -0.1
    /// assert_eq!(b.get(&[2]), 0.0); // max(0, 0) = 0
    /// assert_eq!(b.get(&[3]), 1.0); // max(0, 1) = 1
    /// ```
    ///
    /// ## Different Negative Slopes
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[-1.0, 0.0, 1.0], vec![3]).unwrap();
    /// let b = a.leaky_relu(0.01); // Smaller negative slope
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert!((b.get(&[0]) - (-0.01)).abs() < 1e-6); // -1.0 * 0.01 = -0.01
    /// assert_eq!(b.get(&[1]), 0.0); // max(0, 0) = 0
    /// assert_eq!(b.get(&[2]), 1.0); // max(0, 1) = 1
    /// ```
    pub fn leaky_relu(&self, negative_slope: f32) -> Tensor {
        let mut out = self.leaky_relu_optimized(negative_slope);

        if self.requires_grad() && is_grad_enabled() {
            out.set_requires_grad_internal(true);
            let grad_fn = GradFn::LeakyRelu {
                negative_slope,
                saved_input: Box::new(self.clone()),
            };
            out.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(out.id(), vec![self.id()], grad_fn);
        }

        out
    }

    /// Internal optimized Leaky ReLU operation
    ///
    /// Performs element-wise Leaky ReLU computation using SIMD optimization when available
    /// and falling back to optimized scalar computation. This is the core implementation
    /// used by `leaky_relu()`.
    ///
    /// # Arguments
    ///
    /// * `negative_slope` - Slope for negative values (typically small, e.g., 0.01 or 0.1)
    ///
    /// # Returns
    ///
    /// A new tensor containing the Leaky ReLU activation of each element
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks when available
    /// - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Branch Prediction**: Optimized conditional logic for modern CPUs
    /// - **Zero-sized Handling**: Fast return for empty tensors
    ///
    /// # Implementation Details
    ///
    /// Automatically selects between SIMD and scalar implementations based on hardware
    /// capabilities. SIMD implementation processes 32 elements per iteration with 4x
    /// unrolling for maximum throughput.
    #[inline]
    pub(crate) fn leaky_relu_optimized(&self, negative_slope: f32) -> Tensor {
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
                    self.leaky_relu_simd_avx2_optimized(src, dst, negative_slope);
                    return output;
                }
            }

            // Scalar fallback
            self.leaky_relu_scalar_optimized(src, dst, negative_slope);
        }

        output
    }

    /// AVX2-optimized Leaky ReLU implementation
    ///
    /// Performs element-wise Leaky ReLU using AVX2 SIMD instructions for maximum
    /// performance on x86_64 architectures with AVX2 support.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `negative_slope` - Slope for negative values
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
    /// - **Branch Prediction**: Optimized conditional logic using SIMD masks
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Hardware Requirements**: Requires x86_64 with AVX2 support
    ///
    /// # Implementation Details
    ///
    /// Uses AVX2 vector instructions to process 8 elements simultaneously.
    /// Implements 4x unrolling for optimal instruction throughput and cache utilization.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn leaky_relu_simd_avx2_optimized(
        &self,
        src: *const f32,
        dst: *mut f32,
        negative_slope: f32,
    ) {
        let size = self.size();
        let zero_vec = _mm256_setzero_ps();
        let slope_vec = _mm256_set1_ps(negative_slope);
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let mut offset = 0;

        // Unrolled SIMD loop for maximum throughput
        for _ in 0..simd_count {
            // Process 4 AVX2 vectors (32 elements) per iteration
            self.leaky_relu_simd_block(src, dst, offset, zero_vec, slope_vec);
            self.leaky_relu_simd_block(src, dst, offset + 8, zero_vec, slope_vec);
            self.leaky_relu_simd_block(src, dst, offset + 16, zero_vec, slope_vec);
            self.leaky_relu_simd_block(src, dst, offset + 24, zero_vec, slope_vec);
            offset += 32;
        }

        // Handle remaining 8-element blocks
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            self.leaky_relu_simd_block(src, dst, offset, zero_vec, slope_vec);
            offset += 8;
        }

        // Handle remaining elements with scalar fallback
        for i in offset..size {
            let x = *src.add(i);
            *dst.add(i) = if x > 0.0 { x } else { negative_slope * x };
        }
    }

    /// AVX2 SIMD block processing for Leaky ReLU
    ///
    /// Processes a single 8-element block using AVX2 vector instructions.
    /// This is a helper function for the main SIMD implementation.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `offset` - Offset into the tensor data
    /// * `zero_vec` - AVX2 vector containing zeros
    /// * `slope_vec` - AVX2 vector containing the negative slope value
    ///
    /// # Safety
    ///
    /// Requires valid pointers with sufficient memory for 8 elements starting at offset.
    /// All pointers must point to valid tensor data. Requires AVX2 support.
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Processing**: Processes 8 elements in a single vector operation
    /// - **Vector Operations**: Uses AVX2 comparison, multiplication, and blending
    /// - **Branch-free**: No conditional branches in the SIMD path
    /// - **Memory Access**: Single load and store operation per block
    ///
    /// # Implementation Details
    ///
    /// Uses AVX2 vector comparison to create a mask for positive values,
    /// then blends between the original values and scaled negative values
    /// based on the comparison result.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn leaky_relu_simd_block(
        &self,
        src: *const f32,
        dst: *mut f32,
        offset: usize,
        zero_vec: __m256,
        slope_vec: __m256,
    ) {
        let src_vec = _mm256_loadu_ps(src.add(offset));

        // Create mask for positive values
        let pos_mask = _mm256_cmp_ps(src_vec, zero_vec, _CMP_GT_OQ);

        // Compute negative part: negative_slope * x
        let neg_part = _mm256_mul_ps(src_vec, slope_vec);

        // Blend: use src_vec where positive, neg_part where negative
        let result = _mm256_blendv_ps(neg_part, src_vec, pos_mask);

        _mm256_storeu_ps(dst.add(offset), result);
    }

    /// Optimized scalar Leaky ReLU fallback
    ///
    /// Performs element-wise Leaky ReLU using optimized scalar operations with
    /// 4x unrolling for better instruction-level parallelism and cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `negative_slope` - Slope for negative values
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
    /// - **Branch Prediction**: Optimized conditional logic for modern CPUs
    /// - **Mathematical Accuracy**: High-precision scalar computation
    ///
    /// # Implementation Details
    ///
    /// Uses 4x unrolled scalar operations for optimal performance on non-SIMD hardware.
    /// Processes elements in groups of 4 to improve instruction-level parallelism
    /// and reduce loop overhead.
    #[inline]
    unsafe fn leaky_relu_scalar_optimized(
        &self,
        src: *const f32,
        dst: *mut f32,
        negative_slope: f32,
    ) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar loop for better performance
        for _ in 0..unroll_count {
            let x1 = *src.add(offset);
            let x2 = *src.add(offset + 1);
            let x3 = *src.add(offset + 2);
            let x4 = *src.add(offset + 3);

            *dst.add(offset) = if x1 > 0.0 { x1 } else { negative_slope * x1 };
            *dst.add(offset + 1) = if x2 > 0.0 { x2 } else { negative_slope * x2 };
            *dst.add(offset + 2) = if x3 > 0.0 { x3 } else { negative_slope * x3 };
            *dst.add(offset + 3) = if x4 > 0.0 { x4 } else { negative_slope * x4 };

            offset += 4;
        }

        // Handle remaining elements
        for i in offset..size {
            let x = *src.add(i);
            *dst.add(i) = if x > 0.0 { x } else { negative_slope * x };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaky_relu_forward_basic() {
        let x = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.5], vec![4]).unwrap();
        let y = x.leaky_relu(0.1);
        unsafe {
            assert!((*y.as_ptr() + 0.2).abs() < 1e-6);
            assert!((*y.as_ptr().add(1) + 0.1).abs() < 1e-6);
            assert!((*y.as_ptr().add(2) - 0.0).abs() < 1e-6);
            assert!((*y.as_ptr().add(3) - 1.5).abs() < 1e-6);
        }
    }
}
