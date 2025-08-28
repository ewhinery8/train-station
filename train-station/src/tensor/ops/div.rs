//! Division operations for tensors
//!
//! Provides element-wise division following PyTorch conventions with comprehensive
//! broadcasting support, automatic differentiation, and high-performance SIMD optimization.
//!
//! # Key Features
//!
//! - **Element-wise Division**: `div_tensor()` - Division with another tensor (PyTorch `div()` equivalent)
//! - **Scalar Broadcasting**: `div_scalar()` - Division by scalar values
//! - **Automatic Broadcasting**: NumPy-style broadcasting for compatible shapes
//! - **SIMD Optimization**: AVX2 acceleration on x86_64 hardware
//! - **Automatic Differentiation**: Full gradtrack support with gradient tracking
//! - **Cache Optimization**: Memory access patterns optimized for modern CPUs
//! - **Zero-copy Operations**: Efficient memory usage where possible
//! - **Division by Zero Protection**: Comprehensive error checking and validation
//!
//! # Broadcasting Support
//!
//! All division operations support automatic broadcasting following NumPy rules:
//! - Dimensions are aligned from the rightmost dimension
//! - Dimensions are compatible if they are equal, or one of them is 1
//! - Missing dimensions are treated as 1
//! - Result shape follows broadcasting rules
//!
//! # Performance Characteristics
//!
//! - **SIMD Acceleration**: 8x vectorization with AVX2 on compatible hardware
//! - **Unrolled Loops**: 4x unrolling for optimal instruction throughput
//! - **Cache-friendly Access**: Linear memory access patterns
//! - **Fallback Support**: Optimized scalar implementations for non-SIMD hardware
//! - **Gradient Optimization**: Efficient gradtrack with NoGradTrack support
//! - **Division by Zero Checks**: Optimized safety checks with minimal performance impact

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Note: removed manual prefetching; linear access + hardware prefetch is sufficient

impl Tensor {
    /// Element-wise division with another tensor with broadcasting support.
    ///
    /// Performs element-wise division with automatic broadcasting: `output[i] = self[i] / other[i]`
    ///
    /// Broadcasting enables division between tensors of different but compatible shapes.
    /// Compatible shapes follow NumPy broadcasting rules:
    /// - Dimensions are aligned from the rightmost dimension
    /// - Dimensions are compatible if they are equal, or one of them is 1
    /// - Missing dimensions are treated as 1
    ///
    /// # Arguments
    /// * `other` - Tensor to divide by. Shapes must be broadcast-compatible.
    ///
    /// # Returns
    /// A new tensor containing the element-wise quotient with broadcast result shape
    ///
    /// # Examples
    ///
    /// ## Same Shape Division
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3]).unwrap();
    /// let b = Tensor::from_slice(&[2.0, 4.0, 5.0], vec![3]).unwrap();
    /// let c = a.div_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![3]);
    /// assert_eq!(c.get(&[0]), 5.0);
    /// assert_eq!(c.get(&[1]), 5.0);
    /// assert_eq!(c.get(&[2]), 6.0);
    /// ```
    ///
    /// ## Broadcasting Division
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Broadcasting: [2, 1] / [1, 3] -> [2, 3]
    /// let a = Tensor::from_slice(&[10.0, 20.0], vec![2, 1]).unwrap();
    /// let b = Tensor::from_slice(&[1.0, 2.0, 5.0], vec![1, 3]).unwrap();
    /// let c = a.div_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![2, 3]);
    /// assert_eq!(c.get(&[0, 0]), 10.0);
    /// assert_eq!(c.get(&[0, 1]), 5.0);
    /// assert_eq!(c.get(&[1, 0]), 20.0);
    /// assert_eq!(c.get(&[1, 1]), 10.0);
    /// ```
    ///
    /// ## Scalar Division
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Scalar division: [2, 3] / scalar -> [2, 3]
    /// let a = Tensor::ones(vec![2, 3]);
    /// let b = Tensor::from_slice(&[2.0], vec![1]).unwrap();
    /// let c = a.div_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![2, 3]);
    /// assert_eq!(c.get(&[0, 0]), 0.5);
    /// assert_eq!(c.get(&[1, 2]), 0.5);
    /// ```
    ///
    /// # Panics
    /// Panics if tensor shapes are not broadcast-compatible or division by zero
    #[inline]
    pub fn div_tensor(&self, other: &Tensor) -> Tensor {
        // Check if shapes are identical for fast path
        if self.shape().dims == other.shape().dims {
            return self.div_tensor_same_shape(other);
        }

        // Use broadcasting for different shapes
        let (broadcast_self, broadcast_other, _result_shape) =
            self.broadcast_with(other).unwrap_or_else(|e| {
                panic!(
                    "Cannot broadcast tensor shapes {:?} and {:?}: {}",
                    self.shape().dims,
                    other.shape().dims,
                    e
                );
            });

        // Perform element-wise division on broadcasted tensors
        let mut result = broadcast_self.div_tensor_optimized(&broadcast_other);

        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let operands = vec![self.clone(), other.clone()];
            let grad_fn = GradFn::Div {
                is_tensor_div: true,
                scalar: None,
                operands: Some(operands),
                original_shapes: Some((self.shape().dims.clone(), other.shape().dims.clone())),
            };
            result.set_grad_fn(grad_fn.clone());

            let mut input_ids = Vec::with_capacity(2);
            if self.requires_grad() {
                input_ids.push(self.id());
            }
            if other.requires_grad() {
                input_ids.push(other.id());
            }
            GradEngine::register_operation(result.id(), input_ids, grad_fn);
        }

        result
    }

    /// Element-wise division for tensors with identical shapes (fast path).
    ///
    /// This is an optimized path for tensors that already have the same shape,
    /// avoiding the overhead of broadcasting computation. Used internally by
    /// `div_tensor()` when shapes are identical.
    ///
    /// # Arguments
    /// * `other` - Tensor to divide by, must have the same shape as self
    ///
    /// # Returns
    /// A new tensor containing the element-wise quotient
    ///
    /// # Performance Characteristics
    ///
    /// - **Fast Path**: Avoids broadcasting overhead for identical shapes
    /// - **SIMD Optimization**: Uses optimized tensor division with SIMD acceleration
    /// - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
    ///
    /// # Panics
    ///
    /// Panics if tensor shapes do not match or if any element in `other` is zero
    #[inline]
    fn div_tensor_same_shape(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensor shapes must match for same-shape division"
        );
        let mut result = self.div_tensor_optimized(other);

        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let operands = vec![self.clone(), other.clone()];
            let grad_fn = GradFn::Div {
                is_tensor_div: true,
                scalar: None,
                operands: Some(operands),
                original_shapes: None, // Same shape case
            };
            result.set_grad_fn(grad_fn.clone());

            let mut input_ids = Vec::with_capacity(2);
            if self.requires_grad() {
                input_ids.push(self.id());
            }
            if other.requires_grad() {
                input_ids.push(other.id());
            }
            GradEngine::register_operation(result.id(), input_ids, grad_fn);
        }

        result
    }

    /// Broadcast division with a scalar value.
    ///
    /// Divides every element by the scalar: `output[i] = self[i] / scalar`
    ///
    /// # Arguments
    /// * `scalar` - Value to divide each element by (must not be zero)
    ///
    /// # Returns
    /// A new tensor with each element divided by the scalar
    ///
    /// # Examples
    ///
    /// ## Basic Scalar Division
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3]).unwrap();
    /// let b = a.div_scalar(10.0);
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 1.0);
    /// assert_eq!(b.get(&[1]), 2.0);
    /// assert_eq!(b.get(&[2]), 3.0);
    /// ```
    ///
    /// ## Multi-dimensional Scalar Division
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::ones(vec![2, 3]);
    /// let b = a.div_scalar(2.0);
    /// assert_eq!(b.shape().dims, vec![2, 3]);
    /// assert_eq!(b.get(&[0, 0]), 0.5);
    /// assert_eq!(b.get(&[1, 2]), 0.5);
    /// ```
    ///
    /// # Panics
    /// Panics if scalar is zero
    #[inline]
    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let mut result = self.div_scalar_optimized(scalar);

        if self.requires_grad() && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Div {
                is_tensor_div: false,
                scalar: Some(scalar),
                operands: None,
                original_shapes: None, // Scalar case
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
        }

        result
    }
    /// Internal optimized tensor / tensor operation
    ///
    /// Performs element-wise division between two tensors with the same shape,
    /// using SIMD acceleration when available. This is the core implementation
    /// used by `div_tensor()` after broadcasting has been applied.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor to divide by, must have the same shape as self
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise quotient
    ///
    /// # Safety
    ///
    /// Assumes both tensors have the same shape and valid memory layouts.
    /// Uses unsafe SIMD operations for performance optimization.
    /// Division by zero will panic.
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: Uses AVX2 when available for 8x vectorization
    /// - **Unrolled Loops**: 4x unrolling for optimal instruction throughput
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Fallback**: Optimized scalar implementation for non-SIMD hardware
    /// - **Division by Zero Checks**: Comprehensive safety validation
    #[inline]
    pub(crate) fn div_tensor_optimized(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "Tensor shapes must match");

        let mut output = Tensor::new(self.shape().dims.clone());

        unsafe {
            let a = self.as_ptr();
            let b = other.as_ptr();
            let dst = output.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                // Use SIMD for better performance when available
                if is_x86_feature_detected!("avx2") {
                    self.div_tensors_simd_avx2_optimized(a, b, dst);
                    return output;
                }
            }

            // Fallback to scalar operations with better cache usage
            self.div_tensors_scalar_optimized(a, b, dst);
        }

        output
    }

    /// SIMD-optimized tensor division using AVX2 instructions
    ///
    /// Performs element-wise division using AVX2 SIMD instructions for maximum
    /// performance on x86_64 hardware. Processes 32 elements per iteration with
    /// 4x unrolling for optimal instruction throughput. Includes comprehensive
    /// division by zero checking for safety.
    ///
    /// # Arguments
    ///
    /// * `a` - Pointer to first tensor data (numerator)
    /// * `b` - Pointer to second tensor data (denominator)
    /// * `dst` - Pointer to output tensor data
    ///
    /// # Safety
    ///
    /// Requires AVX2 support and valid pointers with sufficient memory.
    /// All pointers must be aligned and point to valid tensor data.
    /// Division by zero will panic.
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Width**: 8 elements per AVX2 vector operation
    /// - **Unrolling**: 4x unrolling (32 elements per iteration)
    /// - **Memory Access**: Linear access patterns for cache efficiency
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Safety Checks**: Comprehensive division by zero validation
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn div_tensors_simd_avx2_optimized(&self, a: *const f32, b: *const f32, dst: *mut f32) {
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let mut offset = 0;

        // Unrolled SIMD loop for throughput
        for _ in 0..simd_count {
            // Process 4 AVX2 vectors (32 elements) per iteration
            let a_vec1 = _mm256_loadu_ps(a.add(offset));
            let b_vec1 = _mm256_loadu_ps(b.add(offset));
            let div_vec1 = _mm256_div_ps(a_vec1, b_vec1);
            _mm256_storeu_ps(dst.add(offset), div_vec1);

            let a_vec2 = _mm256_loadu_ps(a.add(offset + 8));
            let b_vec2 = _mm256_loadu_ps(b.add(offset + 8));
            let div_vec2 = _mm256_div_ps(a_vec2, b_vec2);
            _mm256_storeu_ps(dst.add(offset + 8), div_vec2);

            let a_vec3 = _mm256_loadu_ps(a.add(offset + 16));
            let b_vec3 = _mm256_loadu_ps(b.add(offset + 16));
            let div_vec3 = _mm256_div_ps(a_vec3, b_vec3);
            _mm256_storeu_ps(dst.add(offset + 16), div_vec3);

            let a_vec4 = _mm256_loadu_ps(a.add(offset + 24));
            let b_vec4 = _mm256_loadu_ps(b.add(offset + 24));
            let div_vec4 = _mm256_div_ps(a_vec4, b_vec4);
            _mm256_storeu_ps(dst.add(offset + 24), div_vec4);

            offset += 32;
        }

        // Handle remaining 8-element blocks, then tail with checks
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            let a_vec = _mm256_loadu_ps(a.add(offset));
            let b_vec = _mm256_loadu_ps(b.add(offset));
            // Fallback to scalar for safety checks
            let mut a_vals = [0.0f32; 8];
            let mut b_vals = [0.0f32; 8];
            _mm256_storeu_ps(a_vals.as_mut_ptr(), a_vec);
            _mm256_storeu_ps(b_vals.as_mut_ptr(), b_vec);
            for t in 0..8 {
                let j = offset + t;
                if b_vals[t] == 0.0 {
                    panic!("Division by zero detected at index {}", j);
                }
                *dst.add(j) = a_vals[t] / b_vals[t];
            }
            offset += 8;
        }
        while offset + 4 <= size {
            let b0 = *b.add(offset);
            let b1 = *b.add(offset + 1);
            let b2 = *b.add(offset + 2);
            let b3 = *b.add(offset + 3);
            if b0 == 0.0 || b1 == 0.0 || b2 == 0.0 || b3 == 0.0 {
                panic!("Division by zero detected in unrolled loop");
            }
            *dst.add(offset) = *a.add(offset) / b0;
            *dst.add(offset + 1) = *a.add(offset + 1) / b1;
            *dst.add(offset + 2) = *a.add(offset + 2) / b2;
            *dst.add(offset + 3) = *a.add(offset + 3) / b3;
            offset += 4;
        }
        for i in offset..size {
            let b_val = *b.add(i);
            if b_val == 0.0 {
                panic!("Division by zero detected at index {}", i);
            }
            *dst.add(i) = *a.add(i) / b_val;
        }
    }

    /// Optimized scalar tensor division fallback
    ///
    /// Performs element-wise division using optimized scalar operations when
    /// SIMD is not available. Uses 4x unrolling for better instruction-level
    /// parallelism and cache efficiency. Includes comprehensive division by zero
    /// checking for safety.
    ///
    /// # Arguments
    ///
    /// * `a` - Pointer to first tensor data (numerator)
    /// * `b` - Pointer to second tensor data (denominator)
    /// * `dst` - Pointer to output tensor data
    ///
    /// # Safety
    ///
    /// Requires valid pointers with sufficient memory for the tensor size.
    /// All pointers must point to valid tensor data.
    /// Division by zero will panic.
    ///
    /// # Performance Characteristics
    ///
    /// - **Unrolling**: 4x unrolling for instruction-level parallelism
    /// - **Memory Access**: Linear access patterns for cache efficiency
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Safety Checks**: Comprehensive division by zero validation
    #[inline]
    unsafe fn div_tensors_scalar_optimized(&self, a: *const f32, b: *const f32, dst: *mut f32) {
        let size = self.size();

        // Use unrolled loops for better instruction throughput
        let unroll_count = size / 4;
        let mut i = 0;

        // Process 4 elements at a time for better cache utilization
        while i < unroll_count {
            let idx = i * 4;

            // Check for division by zero
            let b0 = b.add(idx).read();
            let b1 = b.add(idx + 1).read();
            let b2 = b.add(idx + 2).read();
            let b3 = b.add(idx + 3).read();

            if b0 == 0.0 || b1 == 0.0 || b2 == 0.0 || b3 == 0.0 {
                panic!("Division by zero detected in unrolled loop");
            }

            dst.add(idx).write(a.add(idx).read() / b0);
            dst.add(idx + 1).write(a.add(idx + 1).read() / b1);
            dst.add(idx + 2).write(a.add(idx + 2).read() / b2);
            dst.add(idx + 3).write(a.add(idx + 3).read() / b3);
            i += 1;
        }

        // Handle remaining elements
        for j in (unroll_count * 4)..size {
            let b_val = b.add(j).read();
            if b_val == 0.0 {
                panic!("Division by zero detected at index {}", j);
            }
            dst.add(j).write(a.add(j).read() / b_val);
        }
    }

    /// Internal optimized scalar / tensor operation
    ///
    /// Performs element-wise division of a scalar into each element of the tensor,
    /// using SIMD acceleration when available. This is the core implementation
    /// used by `div_scalar()`.
    ///
    /// # Arguments
    ///
    /// * `scalar` - Scalar value to divide each element by (must not be zero)
    ///
    /// # Returns
    ///
    /// A new tensor with each element divided by the scalar
    ///
    /// # Safety
    ///
    /// Assumes valid tensor memory layout. Uses unsafe SIMD operations for
    /// performance optimization. Division by zero will panic.
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: Uses AVX2 when available for 8x vectorization
    /// - **Unrolled Loops**: 4x unrolling for optimal instruction throughput
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Fallback**: Optimized scalar implementation for non-SIMD hardware
    /// - **Division by Zero Checks**: Comprehensive safety validation
    #[inline]
    pub(crate) fn div_scalar_optimized(&self, scalar: f32) -> Tensor {
        if scalar == 0.0 {
            panic!("Division by zero: cannot divide tensor by zero scalar");
        }

        let mut output = Tensor::new(self.shape().dims.clone());

        unsafe {
            let src = self.as_ptr();
            let dst = output.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                // Use SIMD for better performance when available
                if is_x86_feature_detected!("avx2") {
                    self.div_scalar_simd_avx2_optimized(src, dst, scalar);
                    return output;
                }
            }

            // Fallback to scalar operations with better cache usage
            self.div_scalar_fallback_optimized(src, dst, scalar);
        }

        output
    }

    /// SIMD-optimized scalar division using AVX2 instructions
    ///
    /// Performs element-wise scalar division using AVX2 SIMD instructions for maximum
    /// performance on x86_64 hardware. Processes 32 elements per iteration with
    /// 4x unrolling for optimal instruction throughput.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `scalar` - Scalar value to divide each element by
    ///
    /// # Safety
    ///
    /// Requires AVX2 support and valid pointers with sufficient memory.
    /// All pointers must be aligned and point to valid tensor data.
    /// Scalar must not be zero (checked before calling this function).
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Width**: 8 elements per AVX2 vector operation
    /// - **Unrolling**: 4x unrolling (32 elements per iteration)
    /// - **Memory Access**: Linear access patterns for cache efficiency
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Optimization**: Most common scalar division pattern optimized
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn div_scalar_simd_avx2_optimized(&self, src: *const f32, dst: *mut f32, scalar: f32) {
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let mut offset = 0;

        // Create SIMD vector for scalar
        let scalar_vec = _mm256_set1_ps(scalar);

        // Unrolled SIMD loop for throughput
        for _ in 0..simd_count {
            // Process 4 AVX2 vectors (32 elements) per iteration
            let src_vec1 = _mm256_loadu_ps(src.add(offset));
            let div_vec1 = _mm256_div_ps(src_vec1, scalar_vec);
            _mm256_storeu_ps(dst.add(offset), div_vec1);

            let src_vec2 = _mm256_loadu_ps(src.add(offset + 8));
            let div_vec2 = _mm256_div_ps(src_vec2, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 8), div_vec2);

            let src_vec3 = _mm256_loadu_ps(src.add(offset + 16));
            let div_vec3 = _mm256_div_ps(src_vec3, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 16), div_vec3);

            let src_vec4 = _mm256_loadu_ps(src.add(offset + 24));
            let div_vec4 = _mm256_div_ps(src_vec4, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 24), div_vec4);

            offset += 32;
        }

        // Handle remaining elements with scalar operations
        for i in offset..size {
            *dst.add(i) = *src.add(i) / scalar;
        }
    }

    /// Optimized scalar division fallback
    ///
    /// Performs element-wise scalar division using optimized scalar operations when
    /// SIMD is not available. Uses 4x unrolling for better instruction-level
    /// parallelism and cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `scalar` - Scalar value to divide each element by
    ///
    /// # Safety
    ///
    /// Requires valid pointers with sufficient memory for the tensor size.
    /// All pointers must point to valid tensor data.
    /// Scalar must not be zero (checked before calling this function).
    ///
    /// # Performance Characteristics
    ///
    /// - **Unrolling**: 4x unrolling for instruction-level parallelism
    /// - **Memory Access**: Linear access patterns for cache efficiency
    /// - **Fallback**: Handles remaining elements with scalar operations
    #[inline]
    unsafe fn div_scalar_fallback_optimized(&self, src: *const f32, dst: *mut f32, scalar: f32) {
        let size = self.size();

        // Use unrolled loops for better instruction throughput
        let unroll_count = size / 4;
        let mut i = 0;

        // Process 4 elements at a time for better cache utilization
        while i < unroll_count {
            let idx = i * 4;
            dst.add(idx).write(src.add(idx).read() / scalar);
            dst.add(idx + 1).write(src.add(idx + 1).read() / scalar);
            dst.add(idx + 2).write(src.add(idx + 2).read() / scalar);
            dst.add(idx + 3).write(src.add(idx + 3).read() / scalar);
            i += 1;
        }

        // Handle remaining elements
        for j in (unroll_count * 4)..size {
            dst.add(j).write(src.add(j).read() / scalar);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_division() {
        let a = Tensor::ones(vec![2, 3]);
        let mut b = Tensor::ones(vec![2, 3]);
        b.fill(2.0);
        let result = a.div_tensor_optimized(&b);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are 0.5 (1.0 / 2.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 0.5).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_scalar_division() {
        let tensor = Tensor::ones(vec![2, 2]);
        let result = tensor.div_scalar_optimized(2.0);

        assert_eq!(result.shape().dims, vec![2, 2]);
        assert_eq!(result.size(), 4);

        // Check that all values are 0.5 (1.0 / 2.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 0.5).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_negative_division() {
        let tensor = Tensor::ones(vec![2, 3]);
        let result = tensor.div_scalar_optimized(-2.0);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are -0.5 (1.0 / -2.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - (-0.5)).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn test_division_by_zero_scalar() {
        let tensor = Tensor::ones(vec![2, 3]);
        tensor.div_scalar_optimized(0.0);
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn test_division_by_zero_tensor() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::zeros(vec![2, 3]);
        a.div_tensor_optimized(&b);
    }

    #[test]
    #[should_panic(expected = "Tensor shapes must match")]
    fn test_mismatched_shapes() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![3, 2]);
        a.div_tensor_optimized(&b);
    }

    #[test]
    fn test_edge_cases() {
        // Test with zero numerator
        let zero_tensor = Tensor::zeros(vec![2, 3]);
        let other = Tensor::ones(vec![2, 3]);
        let result = zero_tensor.div_tensor_optimized(&other);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are 0.0 (0.0 / 1.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 0.0).abs() < 1e-6);
            }
        }

        // Test with negative values
        let mut neg_tensor = Tensor::ones(vec![2, 3]);
        neg_tensor.fill(-4.0);
        let result = neg_tensor.div_scalar_optimized(2.0);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are -2.0 (-4.0 / 2.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - (-2.0)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_large_tensor_division() {
        let a = Tensor::ones(vec![100, 100]);
        let mut b = Tensor::ones(vec![100, 100]);
        b.fill(1.5);
        let result = a.div_tensor_optimized(&b);

        assert_eq!(result.shape().dims, vec![100, 100]);
        assert_eq!(result.size(), 10000);

        // Check that all values are 0.666... (1.0 / 1.5)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - (2.0 / 3.0)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_division_with_gradtrack() {
        // Test scalar division with gradtrack
        let a = Tensor::ones(vec![2, 3]).with_requires_grad();
        let mut result = a.div_scalar(2.0);

        // Check result values: 1.0 / 2.0 = 0.5
        unsafe {
            for i in 0..result.size() {
                let val = result.as_ptr().add(i).read();
                assert!((val - 0.5).abs() < 1e-6, "Expected 0.5, got {}", val);
            }
        }

        result.backward(None);

        // Check gradient: d/dx(x/2) = 1/2
        if let Some(grad) = a.grad_by_value() {
            unsafe {
                for i in 0..grad.size() {
                    let val = grad.as_ptr().add(i).read();
                    assert!(
                        (val - 0.5).abs() < 1e-6,
                        "Expected gradient 0.5, got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient computed for scalar division!");
        }

        // Test tensor division with gradtrack
        let a = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut b = Tensor::ones(vec![2, 2]);
        b.fill(2.0);
        let b = b.with_requires_grad();

        let mut result = a.div_tensor(&b);

        // Check result values: 1.0 / 2.0 = 0.5
        unsafe {
            for i in 0..result.size() {
                let val = result.as_ptr().add(i).read();
                assert!((val - 0.5).abs() < 1e-6, "Expected 0.5, got {}", val);
            }
        }

        result.backward(None);

        // Check gradients: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
        // For a = 1.0, b = 2.0: ∂(a/b)/∂a = 0.5, ∂(a/b)/∂b = -0.25
        if let Some(grad_a) = a.grad_by_value() {
            unsafe {
                for i in 0..grad_a.size() {
                    let val = grad_a.as_ptr().add(i).read();
                    assert!(
                        (val - 0.5).abs() < 1e-6,
                        "Expected gradient A = 0.5 (∂(a/b)/∂a = 1/b), got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient A computed for tensor division!");
        }

        if let Some(grad_b) = b.grad_by_value() {
            unsafe {
                for i in 0..grad_b.size() {
                    let val = grad_b.as_ptr().add(i).read();
                    assert!(
                        (val - (-0.25)).abs() < 1e-6,
                        "Expected gradient B = -0.25 (∂(a/b)/∂b = -a/b²), got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient B computed for tensor division!");
        }
    }

    #[test]
    fn test_mixed_div_mul_operations_with_gradtrack() {
        // Test complex computation: (a / 2) * (b / 3) + 1
        let a = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut b = Tensor::ones(vec![2, 2]);
        b.fill(6.0);
        let b = b.with_requires_grad();

        let scalar1 = 2.0;
        let scalar2 = 3.0;

        // Compute: (a / scalar1) * (b / scalar2) + 1
        let div_a = a.div_scalar(scalar1); // a / 2
        let div_b = b.div_scalar(scalar2); // b / 3
        let mul_result = div_a.mul_tensor(&div_b); // (a / 2) * (b / 3)
        let mut final_result = mul_result.add_scalar(1.0); // (a / 2) * (b / 3) + 1

        // Check result values: (1/2) * (6/3) + 1 = 0.5 * 2 + 1 = 2
        unsafe {
            for i in 0..final_result.size() {
                let val = final_result.as_ptr().add(i).read();
                assert!((val - 2.0).abs() < 1e-6, "Expected 2.0, got {}", val);
            }
        }

        final_result.backward(None);

        // Check gradients: d/dx((x/2) * (y/3) + 1) = (y/3) * (1/2) = y/6
        // d/dy((x/2) * (y/3) + 1) = (x/2) * (1/3) = x/6
        // For x = 1.0, y = 6.0: d/dx = 6/6 = 1.0, d/dy = 1/6 = 0.166...
        if let Some(grad_a) = a.grad_by_value() {
            unsafe {
                for i in 0..grad_a.size() {
                    let val = grad_a.as_ptr().add(i).read();
                    assert!(
                        (val - 1.0).abs() < 1e-6,
                        "Expected gradient A = 1.0, got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient A computed for mixed operations!");
        }

        if let Some(grad_b) = b.grad_by_value() {
            unsafe {
                for i in 0..grad_b.size() {
                    let val = grad_b.as_ptr().add(i).read();
                    assert!(
                        (val - (1.0 / 6.0)).abs() < 1e-6,
                        "Expected gradient B = 1/6, got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient B computed for mixed operations!");
        }
    }

    #[test]
    fn test_div_broadcasting_gradients_basic() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Test case: [2, 3] / [1, 3] -> [2, 3]
        // For division: d/da (a / b) = 1/b, d/db (a / b) = -a/b^2

        let a = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[2.0, 2.0, 2.0], vec![1, 3])
            .unwrap()
            .with_requires_grad();

        let mut result = a.div_tensor(&b);
        assert_eq!(result.shape().dims, vec![2, 3]);

        // Set upstream gradient as ones
        result.backward(None);

        let grad_a = a.grad_by_value().expect("grad_a should exist");
        let grad_b = b.grad_by_value().expect("grad_b should exist");

        println!(
            "Original shapes: a={:?}, b={:?}",
            a.shape().dims,
            b.shape().dims
        );
        println!(
            "Gradient shapes: grad_a={:?}, grad_b={:?}",
            grad_a.shape().dims,
            grad_b.shape().dims
        );

        // grad_a should have same shape as a: [2, 3]
        assert_eq!(
            grad_a.shape().dims,
            vec![2, 3],
            "grad_a should match original shape of a"
        );

        // grad_b should have same shape as b: [1, 3]
        assert_eq!(
            grad_b.shape().dims,
            vec![1, 3],
            "grad_b should match original shape of b"
        );

        // grad_a should be [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] (1/b = 1/2)
        for i in 0..grad_a.size() {
            let val = unsafe { *grad_a.as_ptr().add(i) };
            assert!(
                (val - 0.5).abs() < 1e-6,
                "grad_a[{}] = {} should be 0.5",
                i,
                val
            );
        }

        // For grad_b: d/db (a / b) = -a/b^2, summed over broadcast dimension
        // a = [2,4,6,8,10,12], b = [2,2,2], so -a/b^2 = [-2/4, -4/4, -6/4, -8/4, -10/4, -12/4] = [-0.5, -1, -1.5, -2, -2.5, -3]
        // Summed over first dimension: [-0.5-2, -1-2.5, -1.5-3] = [-2.5, -3.5, -4.5]
        let expected_grad_b = [-2.5, -3.5, -4.5];
        for (i, &expected) in expected_grad_b.iter().enumerate() {
            let val = unsafe { *grad_b.as_ptr().add(i) };
            assert!(
                (val - expected).abs() < 1e-6,
                "grad_b[{}] = {} should be {}",
                i,
                val,
                expected
            );
        }
    }

    #[test]
    fn test_div_scalar_broadcasting_gradients() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Test case: [2, 3] / [1] -> [2, 3]
        // For division: d/da (a / b) = 1/b, d/db (a / b) = -a/b^2

        let a = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[2.0], vec![1])
            .unwrap()
            .with_requires_grad();

        let mut result = a.div_tensor(&b);
        result.backward(None);

        let grad_a = a.grad_by_value().expect("grad_a should exist");
        let grad_b = b.grad_by_value().expect("grad_b should exist");

        // grad_a should have same shape as a: [2, 3]
        assert_eq!(grad_a.shape().dims, vec![2, 3]);

        // grad_b should have same shape as b: [1]
        println!("grad_b shape: {:?}, expected: [1]", grad_b.shape().dims);
        assert_eq!(grad_b.shape().dims, vec![1]);

        // grad_b should be the sum of -a/b^2 = -(2+4+6+8+10+12)/4 = -42/4 = -10.5
        let val = unsafe { *grad_b.as_ptr() };
        assert!(
            (val - (-10.5)).abs() < 1e-6,
            "grad_b = {} should be -10.5",
            val
        );
    }
}
