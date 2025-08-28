//! Power operations for tensors
//!
//! Provides element-wise power functions following PyTorch conventions with
//! comprehensive automatic differentiation support and SIMD-optimized computation.
//!
//! # Key Features
//!
//! - **Scalar Power**: `pow_scalar(exponent)` - Raises each element to a scalar power (PyTorch `pow(tensor, scalar)` equivalent)
//! - **Tensor Power**: `pow_tensor(exponent)` - Element-wise power with tensor exponents (PyTorch `pow(tensor, tensor)` equivalent)
//! - **Automatic Differentiation**: Full gradtrack support with efficient gradient computation
//! - **SIMD Optimization**: AVX2-optimized implementation for common cases (x^2, x^0.5)
//! - **Smart Dispatch**: Optimized paths for common exponents (2.0, 0.5) with scalar fallback for others
//! - **Cache Optimization**: Memory access patterns optimized for modern CPUs
//! - **Mathematical Accuracy**: High-precision power computation
//!
//! # Mathematical Properties
//!
//! The power operations have the following properties:
//! - **Power Laws**: (x^a)^b = x^(a*b), x^a * x^b = x^(a+b)
//! - **Special Cases**: x^0 = 1, x^1 = x, x^2 = x*x, x^0.5 = sqrt(x)
//! - **Domain**: x^a is defined for x > 0 when a is not an integer
//! - **Gradient**: d/dx(x^a) = a * x^(a-1) for scalar power
//! - **Gradient**: d/dx(x^y) = y * x^(y-1), d/dy(x^y) = x^y * ln(x) for tensor power
//!
//! # Performance Characteristics
//!
//! - **SIMD Optimization**: AVX2-optimized for x^2 and x^0.5 with 32-element blocks and 4x unrolling
//! - **Smart Dispatch**: Fast paths for common exponents (2.0, 0.5) with scalar fallback for others
//! - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware and general exponents
//! - **Cache-friendly Access**: Linear memory access patterns
//! - **Branch Prediction**: Optimized conditional logic for modern CPUs
//! - **Gradient Optimization**: Efficient gradtrack with NoGradTrack support

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl Tensor {
    /// Raises each element to a scalar power.
    ///
    /// Computes element-wise power: `output[i] = self[i]^exponent`
    ///
    /// # Arguments
    /// * `exponent` - The scalar exponent to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element raised to the given power
    ///
    /// # Examples
    ///
    /// ## Basic Scalar Power
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = a.pow_scalar(2.0);
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 1.0); // 1.0^2 = 1.0
    /// assert_eq!(b.get(&[1]), 4.0); // 2.0^2 = 4.0
    /// assert_eq!(b.get(&[2]), 9.0); // 3.0^2 = 9.0
    /// ```
    ///
    /// ## Square Root (Power 0.5)
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 4.0, 9.0], vec![3]).unwrap();
    /// let b = a.pow_scalar(0.5);
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 1.0); // sqrt(1.0) = 1.0
    /// assert_eq!(b.get(&[1]), 2.0); // sqrt(4.0) = 2.0
    /// assert_eq!(b.get(&[2]), 3.0); // sqrt(9.0) = 3.0
    /// ```
    pub fn pow_scalar(&self, exponent: f32) -> Tensor {
        let mut out = self.pow_scalar_optimized(exponent);

        if self.requires_grad() && is_grad_enabled() {
            out.set_requires_grad_internal(true);
            let grad_fn = GradFn::PowScalar {
                exponent,
                saved_input: Box::new(self.clone()),
            };
            out.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(out.id(), vec![self.id()], grad_fn);
        }

        out
    }

    /// Internal optimized scalar power operation
    ///
    /// Performs element-wise scalar power computation using smart dispatch for common
    /// exponents and optimized scalar computation. This is the core implementation
    /// used by `pow_scalar()`.
    ///
    /// # Arguments
    ///
    /// * `exponent` - The scalar exponent to raise each element to
    ///
    /// # Returns
    ///
    /// A new tensor containing each element raised to the given power
    ///
    /// # Performance Characteristics
    ///
    /// - **Smart Dispatch**: Fast paths for common exponents (2.0, 0.5) with SIMD optimization
    /// - **SIMD Optimization**: AVX2-optimized for x^2 and x^0.5 when available
    /// - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware and general exponents
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Mathematical Accuracy**: High-precision power computation
    /// - **Zero-sized Handling**: Fast return for empty tensors
    ///
    /// # Implementation Details
    ///
    /// Uses smart dispatch to optimize common cases:
    /// - `exponent == 2.0`: Uses SIMD multiplication for x^2
    /// - `exponent == 0.5`: Uses SIMD square root for x^0.5
    /// - Other exponents: Uses scalar `powf()` for accuracy
    #[inline]
    pub(crate) fn pow_scalar_optimized(&self, exponent: f32) -> Tensor {
        let mut output = Tensor::new(self.shape().dims.clone());

        if self.size() == 0 {
            return output;
        }

        unsafe {
            let src = self.as_ptr();
            let dst = output.as_mut_ptr();

            // Handle common cases with SIMD optimizations
            if exponent == 2.0 {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") {
                        self.pow_square_simd_avx2_optimized(src, dst);
                        return output;
                    }
                }
                self.pow_square_scalar_optimized(src, dst);
            } else if exponent == 0.5 {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") {
                        self.pow_sqrt_simd_avx2_optimized(src, dst);
                        return output;
                    }
                }
                self.pow_sqrt_scalar_optimized(src, dst);
            } else {
                // General case - use scalar fallback for accuracy
                self.pow_general_scalar_optimized(src, dst, exponent);
            }
        }

        output
    }

    /// AVX2-optimized square implementation (x^2)
    ///
    /// Performs element-wise squaring using AVX2 SIMD instructions for maximum
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
    /// - **Vector Operations**: Uses AVX2 multiplication instructions for x^2
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Hardware Requirements**: Requires x86_64 with AVX2 support
    ///
    /// # Implementation Details
    ///
    /// Uses AVX2 vector multiplication to compute x^2 efficiently.
    /// Implements 4x unrolling for optimal instruction throughput and cache utilization.
    /// Processes remaining elements with scalar operations for complete coverage.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pow_square_simd_avx2_optimized(&self, src: *const f32, dst: *mut f32) {
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let mut offset = 0;

        // Unrolled SIMD loop for x^2
        for _ in 0..simd_count {
            // Process 4 AVX2 vectors (32 elements) per iteration
            let src_vec1 = _mm256_loadu_ps(src.add(offset));
            let square_vec1 = _mm256_mul_ps(src_vec1, src_vec1);
            _mm256_storeu_ps(dst.add(offset), square_vec1);

            let src_vec2 = _mm256_loadu_ps(src.add(offset + 8));
            let square_vec2 = _mm256_mul_ps(src_vec2, src_vec2);
            _mm256_storeu_ps(dst.add(offset + 8), square_vec2);

            let src_vec3 = _mm256_loadu_ps(src.add(offset + 16));
            let square_vec3 = _mm256_mul_ps(src_vec3, src_vec3);
            _mm256_storeu_ps(dst.add(offset + 16), square_vec3);

            let src_vec4 = _mm256_loadu_ps(src.add(offset + 24));
            let square_vec4 = _mm256_mul_ps(src_vec4, src_vec4);
            _mm256_storeu_ps(dst.add(offset + 24), square_vec4);

            offset += 32;
        }

        // Handle remaining 8-element blocks
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            let src_vec = _mm256_loadu_ps(src.add(offset));
            let square_vec = _mm256_mul_ps(src_vec, src_vec);
            _mm256_storeu_ps(dst.add(offset), square_vec);
            offset += 8;
        }

        // Handle remaining elements
        for i in offset..size {
            let v = *src.add(i);
            *dst.add(i) = v * v;
        }
    }

    /// AVX2-optimized square root implementation (x^0.5)
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
    /// - **Vector Operations**: Uses AVX2 square root instructions for x^0.5
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Hardware Requirements**: Requires x86_64 with AVX2 support
    ///
    /// # Implementation Details
    ///
    /// Uses AVX2 vector square root instructions to compute x^0.5 efficiently.
    /// Implements 4x unrolling for optimal instruction throughput and cache utilization.
    /// Processes remaining elements with scalar operations for complete coverage.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pow_sqrt_simd_avx2_optimized(&self, src: *const f32, dst: *mut f32) {
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let mut offset = 0;

        // Unrolled SIMD loop for x^0.5 (sqrt)
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

        // Handle remaining elements
        for i in offset..size {
            *dst.add(i) = (*src.add(i)).sqrt();
        }
    }

    /// Optimized scalar square fallback (x^2)
    ///
    /// Performs element-wise squaring using optimized scalar operations with
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
    /// - **Mathematical Accuracy**: High-precision scalar multiplication
    ///
    /// # Implementation Details
    ///
    /// Uses 4x unrolled scalar multiplication for optimal performance on non-SIMD hardware.
    /// Processes elements in groups of 4 to improve instruction-level parallelism
    /// and reduce loop overhead.
    #[inline]
    unsafe fn pow_square_scalar_optimized(&self, src: *const f32, dst: *mut f32) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar loop for x^2
        for _ in 0..unroll_count {
            let v1 = *src.add(offset);
            let v2 = *src.add(offset + 1);
            let v3 = *src.add(offset + 2);
            let v4 = *src.add(offset + 3);

            *dst.add(offset) = v1 * v1;
            *dst.add(offset + 1) = v2 * v2;
            *dst.add(offset + 2) = v3 * v3;
            *dst.add(offset + 3) = v4 * v4;

            offset += 4;
        }

        // Handle remaining elements
        for i in offset..size {
            let v = *src.add(i);
            *dst.add(i) = v * v;
        }
    }

    /// Optimized scalar square root fallback (x^0.5)
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
    /// - **Mathematical Accuracy**: High-precision scalar square root
    ///
    /// # Implementation Details
    ///
    /// Uses 4x unrolled scalar square root for optimal performance on non-SIMD hardware.
    /// Processes elements in groups of 4 to improve instruction-level parallelism
    /// and reduce loop overhead.
    #[inline]
    unsafe fn pow_sqrt_scalar_optimized(&self, src: *const f32, dst: *mut f32) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar loop for x^0.5
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

    /// Optimized scalar general power fallback (x^exponent)
    ///
    /// Performs element-wise power computation using optimized scalar operations with
    /// 4x unrolling for better instruction-level parallelism and cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `exponent` - The scalar exponent to raise each element to
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
    /// - **Mathematical Accuracy**: High-precision scalar power computation
    ///
    /// # Implementation Details
    ///
    /// Uses 4x unrolled scalar power for optimal performance on non-SIMD hardware.
    /// Processes elements in groups of 4 to improve instruction-level parallelism
    /// and reduce loop overhead. Uses `powf()` for general exponent support.
    #[inline]
    unsafe fn pow_general_scalar_optimized(&self, src: *const f32, dst: *mut f32, exponent: f32) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar loop for general exponent
        for _ in 0..unroll_count {
            *dst.add(offset) = (*src.add(offset)).powf(exponent);
            *dst.add(offset + 1) = (*src.add(offset + 1)).powf(exponent);
            *dst.add(offset + 2) = (*src.add(offset + 2)).powf(exponent);
            *dst.add(offset + 3) = (*src.add(offset + 3)).powf(exponent);
            offset += 4;
        }

        // Handle remaining elements
        for i in offset..size {
            *dst.add(i) = (*src.add(i)).powf(exponent);
        }
    }

    /// Element-wise power with tensor exponents.
    ///
    /// Computes element-wise power: `output[i] = self[i]^exponent[i]`
    ///
    /// # Arguments
    /// * `exponent` - Tensor of exponents, must have the same shape as self
    ///
    /// # Returns
    /// A new tensor with each element raised to the corresponding power
    ///
    /// # Examples
    ///
    /// ## Basic Tensor Power
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let base = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]).unwrap();
    /// let exp = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let result = base.pow_tensor(&exp);
    /// assert_eq!(result.shape().dims, vec![3]);
    /// assert_eq!(result.get(&[0]), 2.0); // 2.0^1.0 = 2.0
    /// assert_eq!(result.get(&[1]), 9.0); // 3.0^2.0 = 9.0
    /// assert_eq!(result.get(&[2]), 64.0); // 4.0^3.0 = 64.0
    /// ```
    ///
    /// ## Mixed Exponents
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let base = Tensor::from_slice(&[4.0, 9.0, 16.0], vec![3]).unwrap();
    /// let exp = Tensor::from_slice(&[0.5, 1.0, 2.0], vec![3]).unwrap();
    /// let result = base.pow_tensor(&exp);
    /// assert_eq!(result.shape().dims, vec![3]);
    /// assert_eq!(result.get(&[0]), 2.0); // sqrt(4.0) = 2.0
    /// assert_eq!(result.get(&[1]), 9.0); // 9.0^1.0 = 9.0
    /// assert_eq!(result.get(&[2]), 256.0); // 16.0^2.0 = 256.0
    /// ```
    ///
    /// # Panics
    /// Panics if tensor shapes don't match
    pub fn pow_tensor(&self, exponent: &Tensor) -> Tensor {
        assert_eq!(
            self.shape().dims,
            exponent.shape().dims,
            "pow_tensor requires identical shapes"
        );
        let mut out = Tensor::new(self.shape().dims.clone());
        unsafe {
            let x = self.as_ptr();
            let a = exponent.as_ptr();
            let y = out.as_mut_ptr();
            let n = out.size();
            for i in 0..n {
                *y.add(i) = (*x.add(i)).powf(*a.add(i));
            }
        }

        if (self.requires_grad() || exponent.requires_grad()) && is_grad_enabled() {
            let mut result = out.clone();
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::PowTensor {
                saved_base: Box::new(self.clone()),
                saved_exponent: Box::new(exponent.clone()),
            };
            result.set_grad_fn(grad_fn.clone());
            let parents = vec![self.id(), exponent.id()];
            GradEngine::register_operation(result.id(), parents, grad_fn);
            return result;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_scalar_forward() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = x.pow_scalar(2.0);
        assert_eq!(y.shape().dims, vec![4]);
        unsafe {
            assert_eq!(*y.as_ptr(), 1.0);
            assert_eq!(*y.as_ptr().add(1), 4.0);
            assert_eq!(*y.as_ptr().add(2), 9.0);
            assert_eq!(*y.as_ptr().add(3), 16.0);
        }
    }
}
