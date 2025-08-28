//! Square root operation for tensors
//!
//! Provides element-wise square root following PyTorch conventions with
//! comprehensive GradTrack support and SIMD-optimized computation.
//!
//! # Key Features
//!
//! - **Square Root**: `sqrt()` - Computes square root for each element (PyTorch `sqrt()` equivalent)
//! - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
//! - **SIMD Optimization**: AVX2-optimized implementation for maximum performance
//! - **Mathematical Accuracy**: High-precision square root computation
//! - **Domain Validation**: Handles negative values appropriately
//! - **Performance Optimization**: 4x unrolled SIMD operations with scalar fallback
//!
//! # Mathematical Properties
//!
//! The square root function has the following properties:
//! - **Definition**: f(x) = √x
//! - **Domain**: [0, ∞) - defined for non-negative real numbers
//! - **Range**: [0, ∞) - outputs are always non-negative
//! - **Monotonicity**: Strictly increasing function
//! - **Continuity**: Continuous on its domain
//! - **Gradient**: f'(x) = 0.5 / √x for x > 0
//! - **Special Cases**: f(0) = 0, f(1) = 1
//!
//! # Performance Characteristics
//!
//! - **SIMD Optimization**: AVX2-optimized with 32-element blocks and 4x unrolling
//! - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
//! - **Cache-friendly Access**: Linear memory access patterns
//! - **Mathematical Accuracy**: High-precision square root computation
//! - **GradTrack Optimization**: Efficient automatic differentiation with NoGradTrack support

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl Tensor {
    /// Element-wise square root
    ///
    /// Computes the square root for each element: `output[i] = sqrt(self[i])`
    ///
    /// Uses SIMD optimization when available for maximum performance, with automatic
    /// fallback to optimized scalar computation for non-SIMD hardware.
    ///
    /// # Returns
    ///
    /// A new tensor with the square root of each element
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks and 4x unrolling
    /// - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Mathematical Accuracy**: High-precision square root computation
    /// - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
    ///
    /// # Implementation Details
    ///
    /// Automatically selects between SIMD and scalar implementations based on hardware
    /// capabilities. SIMD implementation uses AVX2 vector square root operations for optimal
    /// performance. Scalar implementation uses 4x unrolling for better instruction-level
    /// parallelism.
    ///
    /// # Examples
    ///
    /// ## Basic Square Root
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 4.0, 9.0], vec![3]).unwrap();
    /// let b = a.sqrt();
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 1.0); // sqrt(1.0) = 1.0
    /// assert_eq!(b.get(&[1]), 2.0); // sqrt(4.0) = 2.0
    /// assert_eq!(b.get(&[2]), 3.0); // sqrt(9.0) = 3.0
    /// ```
    ///
    /// ## Zero and Special Values
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[0.0, 1.0, 16.0], vec![3]).unwrap();
    /// let b = a.sqrt();
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 0.0); // sqrt(0.0) = 0.0
    /// assert_eq!(b.get(&[1]), 1.0); // sqrt(1.0) = 1.0
    /// assert_eq!(b.get(&[2]), 4.0); // sqrt(16.0) = 4.0
    /// ```
    ///
    /// # Note
    /// Results are undefined for negative values (may produce NaN)
    #[inline]
    pub fn sqrt(&self) -> Tensor {
        let mut result = self.sqrt_optimized();
        if self.requires_grad() && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Sqrt {
                saved_output: Box::new(result.clone()),
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
        }
        result
    }
    /// Internal optimized square root operation
    ///
    /// Performs element-wise square root using SIMD optimization when available
    /// and falling back to optimized scalar computation. This is the core implementation
    /// used by `sqrt()`.
    ///
    /// # Returns
    ///
    /// A new tensor containing the square root of each element
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks when available
    /// - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Mathematical Accuracy**: High-precision square root computation
    /// - **Zero-sized Handling**: Fast return for empty tensors
    ///
    /// # Implementation Details
    ///
    /// Automatically selects between SIMD and scalar implementations based on hardware
    /// capabilities. SIMD implementation uses AVX2 vector square root operations for optimal
    /// performance. Scalar implementation uses 4x unrolling for better instruction-level
    /// parallelism.
    #[inline]
    pub(crate) fn sqrt_optimized(&self) -> Tensor {
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
                    self.sqrt_simd_avx2_optimized(src, dst);
                    return output;
                }
            }

            // Scalar fallback
            self.sqrt_scalar_optimized(src, dst);
        }

        output
    }

    /// AVX2-optimized square root implementation
    ///
    /// Performs element-wise square root using AVX2 SIMD instructions for maximum
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
    /// - **Vector Operations**: Uses AVX2 sqrt instructions for square root computation
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Hardware Requirements**: Requires x86_64 with AVX2 support
    ///
    /// # Implementation Details
    ///
    /// Uses AVX2 vector square root operations to compute sqrt(x) efficiently.
    /// Implements 4x unrolling for optimal instruction throughput and cache utilization.
    /// Processes remaining elements with scalar operations for complete coverage.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sqrt_simd_avx2_optimized(&self, src: *const f32, dst: *mut f32) {
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let mut offset = 0;

        // Unrolled SIMD loop for maximum throughput
        for _ in 0..simd_count {
            // Process 4 AVX2 vectors (32 elements) per iteration
            let src_vec1 = _mm256_loadu_ps(src.add(offset));
            let sqrt_vec1 = _mm256_sqrt_ps(src_vec1);
            _mm256_storeu_ps(dst.add(offset), sqrt_vec1);

            let src_vec2 = _mm256_loadu_ps(src.add(offset + 8));
            let sqrt_vec2 = _mm256_sqrt_ps(src_vec2);
            _mm256_storeu_ps(dst.add(offset + 8), sqrt_vec2);

            let src_vec3 = _mm256_loadu_ps(src.add(offset + 16));
            let sqrt_vec3 = _mm256_sqrt_ps(src_vec3);
            _mm256_storeu_ps(dst.add(offset + 16), sqrt_vec3);

            let src_vec4 = _mm256_loadu_ps(src.add(offset + 24));
            let sqrt_vec4 = _mm256_sqrt_ps(src_vec4);
            _mm256_storeu_ps(dst.add(offset + 24), sqrt_vec4);

            offset += 32;
        }

        // Handle remaining 8-element blocks
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            let src_vec = _mm256_loadu_ps(src.add(offset));
            let sqrt_vec = _mm256_sqrt_ps(src_vec);
            _mm256_storeu_ps(dst.add(offset), sqrt_vec);
            offset += 8;
        }

        // Handle remaining elements with scalar fallback
        for i in offset..size {
            *dst.add(i) = (*src.add(i)).sqrt();
        }
    }

    /// Optimized scalar square root fallback
    ///
    /// Performs element-wise square root using optimized scalar operations with
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
    /// - **Mathematical Accuracy**: High-precision scalar square root computation
    ///
    /// # Implementation Details
    ///
    /// Uses 4x unrolled scalar operations for optimal performance on non-SIMD hardware.
    /// Processes elements in groups of 4 to improve instruction-level parallelism
    /// and reduce loop overhead.
    #[inline]
    unsafe fn sqrt_scalar_optimized(&self, src: *const f32, dst: *mut f32) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar loop for better performance
        for _ in 0..unroll_count {
            *dst.add(offset) = (*src.add(offset)).sqrt();
            *dst.add(offset + 1) = (*src.add(offset + 1)).sqrt();
            *dst.add(offset + 2) = (*src.add(offset + 2)).sqrt();
            *dst.add(offset + 3) = (*src.add(offset + 3)).sqrt();
            offset += 4;
        }

        // Handle remaining elements
        for i in offset..size {
            *dst.add(i) = (*src.add(i)).sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_basic() {
        let x = Tensor::from_slice(&[0.0, 1.0, 4.0, 9.0], vec![2, 2]).unwrap();
        let y = x.sqrt_optimized();
        unsafe {
            let yd = std::slice::from_raw_parts(y.as_ptr(), y.size());
            assert!((yd[0] - 0.0).abs() < 1e-6);
            assert!((yd[1] - 1.0).abs() < 1e-6);
            assert!((yd[2] - 2.0).abs() < 1e-6);
            assert!((yd[3] - 3.0).abs() < 1e-6);
        }
    }
}
