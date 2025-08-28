//! Subtraction operations for tensors
//!
//! Provides element-wise subtraction operations following PyTorch conventions with
//! comprehensive GradTrack support and SIMD-optimized computation.
//!
//! # Key Features
//!
//! - **Tensor Subtraction**: `sub_tensor()` - Element-wise subtraction with broadcasting support
//! - **Scalar Subtraction**: `sub_scalar()` - Subtraction of scalar from tensor
//! - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
//! - **SIMD Optimization**: AVX2-optimized implementation for maximum performance
//! - **Broadcasting Support**: NumPy-style broadcasting for compatible shapes
//! - **Performance Optimization**: 4x unrolled SIMD operations with scalar fallback
//!
//! # Mathematical Properties
//!
//! The subtraction operations have the following properties:
//! - **Tensor-Tensor**: `output[i] = a[i] - b[i]` with broadcasting
//! - **Tensor-Scalar**: `output[i] = a[i] - scalar` for all elements
//! - **Commutativity**: Subtraction is not commutative (a - b ≠ b - a)
//! - **Associativity**: Subtraction is not associative ((a - b) - c ≠ a - (b - c))
//! - **Gradient**: For tensor-tensor: ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
//! - **Broadcasting**: Follows NumPy broadcasting rules for shape compatibility
//!
//! # Performance Characteristics
//!
//! - **SIMD Optimization**: AVX2-optimized with 32-element blocks and 4x unrolling
//! - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
//! - **Cache-friendly Access**: Linear memory access patterns
//! - **Broadcasting Overhead**: Minimal overhead for compatible shapes
//! - **GradTrack Optimization**: Efficient automatic differentiation with NoGradTrack support

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Note: removed manual prefetching; linear access + hardware prefetch is sufficient

impl Tensor {
    /// Element-wise subtraction with another tensor with broadcasting support
    ///
    /// Performs element-wise subtraction with automatic broadcasting: `output[i] = self[i] - other[i]`
    ///
    /// Broadcasting enables subtraction between tensors of different but compatible shapes.
    /// Compatible shapes follow NumPy broadcasting rules:
    /// - Dimensions are aligned from the rightmost dimension
    /// - Dimensions are compatible if they are equal, or one of them is 1
    /// - Missing dimensions are treated as 1
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor to subtract. Shapes must be broadcast-compatible.
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise difference with broadcast result shape
    ///
    /// # Performance Characteristics
    ///
    /// - **Fast Path**: Optimized for identical shapes to avoid broadcasting overhead
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks and 4x unrolling
    /// - **Broadcasting**: Efficient broadcasting for compatible shapes
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
    ///
    /// # Implementation Details
    ///
    /// Uses a fast path for identical shapes to avoid broadcasting overhead.
    /// For different shapes, performs broadcasting followed by optimized element-wise subtraction.
    /// Automatically selects between SIMD and scalar implementations based on hardware capabilities.
    ///
    /// # Examples
    ///
    /// ## Same Shape Subtraction
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[5.0, 7.0, 9.0], vec![3]).unwrap();
    /// let b = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let c = a.sub_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![3]);
    /// assert_eq!(c.get(&[0]), 4.0); // 5.0 - 1.0
    /// assert_eq!(c.get(&[1]), 5.0); // 7.0 - 2.0
    /// assert_eq!(c.get(&[2]), 6.0); // 9.0 - 3.0
    /// ```
    ///
    /// ## Broadcasting Subtraction
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[5.0, 10.0], vec![2, 1]).unwrap();
    /// let b = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    /// let c = a.sub_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![2, 3]);
    /// // Result: [[4.0, 3.0, 2.0], [9.0, 8.0, 7.0]]
    /// assert_eq!(c.get(&[0, 0]), 4.0); // 5.0 - 1.0
    /// assert_eq!(c.get(&[0, 1]), 3.0); // 5.0 - 2.0
    /// assert_eq!(c.get(&[1, 0]), 9.0); // 10.0 - 1.0
    /// ```
    ///
    /// ## Scalar Subtraction
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::ones(vec![2, 3]);
    /// let b = Tensor::from_slice(&[0.5], vec![1]).unwrap();
    /// let c = a.sub_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![2, 3]);
    /// assert_eq!(c.get(&[0, 0]), 0.5); // 1.0 - 0.5
    /// ```
    ///
    /// # Panics
    /// Panics if tensor shapes are not broadcast-compatible
    #[inline]
    pub fn sub_tensor(&self, other: &Tensor) -> Tensor {
        // Check if shapes are identical for fast path
        if self.shape().dims == other.shape().dims {
            return self.sub_tensor_same_shape(other);
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

        // Perform element-wise subtraction on broadcasted tensors
        let mut result = broadcast_self.sub_tensor_optimized(&broadcast_other);

        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Sub {
                is_tensor_sub: true,
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

    /// Element-wise subtraction for tensors with identical shapes (fast path)
    ///
    /// This is an optimized path for tensors that already have the same shape,
    /// avoiding the overhead of broadcasting computation.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor to subtract, must have the same shape as self
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise difference
    ///
    /// # Performance Characteristics
    ///
    /// - **Fast Path**: Avoids broadcasting overhead for identical shapes
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks and 4x unrolling
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
    ///
    /// # Implementation Details
    ///
    /// This method is used internally by `sub_tensor()` when tensors have identical shapes.
    /// It bypasses the broadcasting logic and directly calls the optimized subtraction implementation.
    #[inline]
    fn sub_tensor_same_shape(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensor shapes must match for same-shape subtraction"
        );
        let mut result = self.sub_tensor_optimized(other);

        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Sub {
                is_tensor_sub: true,
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

    /// Element-wise subtraction of a scalar from this tensor
    ///
    /// Performs element-wise subtraction of a scalar value: `output[i] = self[i] - scalar`
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to subtract from each element
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar subtracted from each element
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks and 4x unrolling
    /// - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Mathematical Accuracy**: High-precision subtraction computation
    /// - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
    ///
    /// # Examples
    ///
    /// ## Basic Scalar Subtraction
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[5.0, 7.0, 9.0], vec![3]).unwrap();
    /// let b = a.sub_scalar(2.0);
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 3.0); // 5.0 - 2.0
    /// assert_eq!(b.get(&[1]), 5.0); // 7.0 - 2.0
    /// assert_eq!(b.get(&[2]), 7.0); // 9.0 - 2.0
    /// ```
    ///
    /// ## Negative Scalar Subtraction
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = a.sub_scalar(-2.0); // Subtracting negative = adding
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 3.0); // 1.0 - (-2.0) = 3.0
    /// assert_eq!(b.get(&[1]), 4.0); // 2.0 - (-2.0) = 4.0
    /// assert_eq!(b.get(&[2]), 5.0); // 3.0 - (-2.0) = 5.0
    /// ```
    #[inline]
    pub fn sub_scalar(&self, scalar: f32) -> Tensor {
        let mut result = self.sub_scalar_optimized(scalar);

        if self.requires_grad() && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Sub {
                is_tensor_sub: false,
                original_shapes: None, // Scalar case
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
        }

        result
    }
    /// Optimized tensor subtraction using SIMD when available
    ///
    /// Performs element-wise subtraction between tensors with identical shapes using
    /// SIMD optimization when available and falling back to optimized scalar computation.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to subtract from this tensor
    ///
    /// # Returns
    ///
    /// A new tensor with the result of the subtraction (self - other)
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks when available
    /// - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Mathematical Accuracy**: High-precision subtraction computation
    /// - **Zero-sized Handling**: Fast return for empty tensors
    ///
    /// # Implementation Details
    ///
    /// Automatically selects between SIMD and scalar implementations based on hardware
    /// capabilities. SIMD implementation uses AVX2 vector subtraction operations for optimal
    /// performance. Scalar implementation uses 4x unrolling for better instruction-level
    /// parallelism.
    ///
    /// # Safety
    ///
    /// This operation assumes the tensors have the same shape.
    #[inline]
    pub(crate) fn sub_tensor_optimized(&self, other: &Tensor) -> Tensor {
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
                    self.sub_tensors_simd_avx2_optimized(a, b, dst);
                    return output;
                }
            }

            // Fallback to scalar operations with better cache usage
            self.sub_tensors_scalar_optimized(a, b, dst);
        }

        output
    }

    /// AVX2-optimized tensor subtraction implementation
    ///
    /// Performs element-wise subtraction using AVX2 SIMD instructions for maximum
    /// performance on x86_64 architectures with AVX2 support.
    ///
    /// # Arguments
    ///
    /// * `a` - Pointer to first tensor data
    /// * `b` - Pointer to second tensor data
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
    /// - **Vector Operations**: Uses AVX2 subtraction instructions for computation
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Hardware Requirements**: Requires x86_64 with AVX2 support
    ///
    /// # Implementation Details
    ///
    /// Uses AVX2 vector subtraction operations to compute a - b efficiently.
    /// Implements 4x unrolling for optimal instruction throughput and cache utilization.
    /// Processes remaining elements with scalar operations for complete coverage.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_tensors_simd_avx2_optimized(&self, a: *const f32, b: *const f32, dst: *mut f32) {
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let mut offset = 0;

        // Unrolled SIMD loop for throughput
        for _ in 0..simd_count {
            // Process 4 AVX2 vectors (32 elements) per iteration
            let a_vec1 = _mm256_loadu_ps(a.add(offset));
            let b_vec1 = _mm256_loadu_ps(b.add(offset));
            let sub_vec1 = _mm256_sub_ps(a_vec1, b_vec1);
            _mm256_storeu_ps(dst.add(offset), sub_vec1);

            let a_vec2 = _mm256_loadu_ps(a.add(offset + 8));
            let b_vec2 = _mm256_loadu_ps(b.add(offset + 8));
            let sub_vec2 = _mm256_sub_ps(a_vec2, b_vec2);
            _mm256_storeu_ps(dst.add(offset + 8), sub_vec2);

            let a_vec3 = _mm256_loadu_ps(a.add(offset + 16));
            let b_vec3 = _mm256_loadu_ps(b.add(offset + 16));
            let sub_vec3 = _mm256_sub_ps(a_vec3, b_vec3);
            _mm256_storeu_ps(dst.add(offset + 16), sub_vec3);

            let a_vec4 = _mm256_loadu_ps(a.add(offset + 24));
            let b_vec4 = _mm256_loadu_ps(b.add(offset + 24));
            let sub_vec4 = _mm256_sub_ps(a_vec4, b_vec4);
            _mm256_storeu_ps(dst.add(offset + 24), sub_vec4);

            offset += 32;
        }

        // Handle remaining 8-element blocks
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            let a_vec = _mm256_loadu_ps(a.add(offset));
            let b_vec = _mm256_loadu_ps(b.add(offset));
            let sub_vec = _mm256_sub_ps(a_vec, b_vec);
            _mm256_storeu_ps(dst.add(offset), sub_vec);
            offset += 8;
        }

        // Handle final elements with unrolled loop
        let remaining = size - offset;
        let unroll_count = remaining / 4;
        for _ in 0..unroll_count {
            *dst.add(offset) = *a.add(offset) - *b.add(offset);
            *dst.add(offset + 1) = *a.add(offset + 1) - *b.add(offset + 1);
            *dst.add(offset + 2) = *a.add(offset + 2) - *b.add(offset + 2);
            *dst.add(offset + 3) = *a.add(offset + 3) - *b.add(offset + 3);
            offset += 4;
        }

        for i in offset..size {
            *dst.add(i) = *a.add(i) - *b.add(i);
        }
    }

    /// Optimized scalar tensor subtraction fallback
    ///
    /// Performs element-wise subtraction using optimized scalar operations with
    /// 4x unrolling for better instruction-level parallelism and cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `a` - Pointer to first tensor data
    /// * `b` - Pointer to second tensor data
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
    /// - **Mathematical Accuracy**: High-precision scalar subtraction computation
    ///
    /// # Implementation Details
    ///
    /// Uses 4x unrolled scalar operations for optimal performance on non-SIMD hardware.
    /// Processes elements in groups of 4 to improve instruction-level parallelism
    /// and reduce loop overhead.
    #[inline]
    unsafe fn sub_tensors_scalar_optimized(&self, a: *const f32, b: *const f32, dst: *mut f32) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar loop for better performance
        for _ in 0..unroll_count {
            *dst.add(offset) = *a.add(offset) - *b.add(offset);
            *dst.add(offset + 1) = *a.add(offset + 1) - *b.add(offset + 1);
            *dst.add(offset + 2) = *a.add(offset + 2) - *b.add(offset + 2);
            *dst.add(offset + 3) = *a.add(offset + 3) - *b.add(offset + 3);
            offset += 4;
        }

        // Handle remaining elements
        for i in offset..size {
            *dst.add(i) = *a.add(i) - *b.add(i);
        }
    }

    /// Internal optimized scalar subtraction operation
    ///
    /// Performs element-wise subtraction of a scalar from tensor using SIMD optimization
    /// when available and falling back to optimized scalar computation.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to subtract from each element
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar subtracted from each element
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks when available
    /// - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Mathematical Accuracy**: High-precision subtraction computation
    /// - **Zero-sized Handling**: Fast return for empty tensors
    ///
    /// # Implementation Details
    ///
    /// Automatically selects between SIMD and scalar implementations based on hardware
    /// capabilities. SIMD implementation uses AVX2 vector subtraction operations for optimal
    /// performance. Scalar implementation uses 4x unrolling for better instruction-level
    /// parallelism.
    #[inline]
    pub(crate) fn sub_scalar_optimized(&self, scalar: f32) -> Tensor {
        let mut output = Tensor::new(self.shape().dims.clone());

        unsafe {
            let src = self.as_ptr();
            let dst = output.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                // Use SIMD for better performance when available
                if is_x86_feature_detected!("avx2") {
                    self.sub_scalar_simd_avx2_optimized(src, dst, scalar);
                    return output;
                }
            }

            // Fallback to optimized scalar operations
            self.sub_scalar_fallback_optimized(src, dst, scalar);
        }

        output
    }

    /// AVX2-optimized scalar subtraction implementation
    ///
    /// Performs element-wise subtraction of a scalar using AVX2 SIMD instructions for maximum
    /// performance on x86_64 architectures with AVX2 support.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `scalar` - The scalar value to subtract from each element
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
    /// - **Vector Operations**: Uses AVX2 subtraction instructions for computation
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Hardware Requirements**: Requires x86_64 with AVX2 support
    ///
    /// # Implementation Details
    ///
    /// Uses AVX2 vector subtraction operations to compute src - scalar efficiently.
    /// Implements 4x unrolling for optimal instruction throughput and cache utilization.
    /// Processes remaining elements with scalar operations for complete coverage.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_scalar_simd_avx2_optimized(&self, src: *const f32, dst: *mut f32, scalar: f32) {
        let scalar_vec = _mm256_set1_ps(scalar);
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration
        let mut offset = 0;

        // Unrolled SIMD loop for better instruction throughput
        for _ in 0..simd_count {
            let src_vec1 = _mm256_loadu_ps(src.add(offset));
            let sub_vec1 = _mm256_sub_ps(src_vec1, scalar_vec);
            _mm256_storeu_ps(dst.add(offset), sub_vec1);

            let src_vec2 = _mm256_loadu_ps(src.add(offset + 8));
            let sub_vec2 = _mm256_sub_ps(src_vec2, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 8), sub_vec2);

            let src_vec3 = _mm256_loadu_ps(src.add(offset + 16));
            let sub_vec3 = _mm256_sub_ps(src_vec3, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 16), sub_vec3);

            let src_vec4 = _mm256_loadu_ps(src.add(offset + 24));
            let sub_vec4 = _mm256_sub_ps(src_vec4, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 24), sub_vec4);

            offset += 32;
        }

        // Handle remaining 8-element blocks
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            let src_vec = _mm256_loadu_ps(src.add(offset));
            let sub_vec = _mm256_sub_ps(src_vec, scalar_vec);
            _mm256_storeu_ps(dst.add(offset), sub_vec);
            offset += 8;
        }

        // Handle final elements
        for i in offset..size {
            *dst.add(i) = *src.add(i) - scalar;
        }
    }

    /// Optimized scalar subtraction fallback
    ///
    /// Performs element-wise subtraction of a scalar using optimized scalar operations with
    /// 4x unrolling for better instruction-level parallelism and cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `scalar` - The scalar value to subtract from each element
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
    /// - **Mathematical Accuracy**: High-precision scalar subtraction computation
    ///
    /// # Implementation Details
    ///
    /// Uses 4x unrolled scalar operations for optimal performance on non-SIMD hardware.
    /// Processes elements in groups of 4 to improve instruction-level parallelism
    /// and reduce loop overhead.
    #[inline]
    unsafe fn sub_scalar_fallback_optimized(&self, src: *const f32, dst: *mut f32, scalar: f32) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar operations
        for _ in 0..unroll_count {
            *dst.add(offset) = *src.add(offset) - scalar;
            *dst.add(offset + 1) = *src.add(offset + 1) - scalar;
            *dst.add(offset + 2) = *src.add(offset + 2) - scalar;
            *dst.add(offset + 3) = *src.add(offset + 3) - scalar;
            offset += 4;
        }

        for i in offset..size {
            *dst.add(i) = *src.add(i) - scalar;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_subtraction() {
        let mut a = Tensor::ones(vec![2, 3]);
        a.fill(5.0); // Create a tensor with all 5.0s
        let mut b = Tensor::ones(vec![2, 3]);
        b.fill(2.0); // Create a tensor with all 2.0s
        let result = a.sub_tensor_optimized(&b);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are 3.0 (5.0 - 2.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 3.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_scalar_subtraction() {
        let mut tensor = Tensor::ones(vec![2, 2]);
        tensor.fill(10.0); // Create a tensor with all 10.0s
        let result = tensor.sub_scalar_optimized(3.0);

        assert_eq!(result.shape().dims, vec![2, 2]);
        assert_eq!(result.size(), 4);

        // Check that all values are 7.0 (10.0 - 3.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 7.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_negative_subtraction() {
        let mut a = Tensor::ones(vec![2, 2]);
        a.fill(2.0); // Create a tensor with all 2.0s
        let mut b = Tensor::ones(vec![2, 2]);
        b.fill(5.0); // Create a tensor with all 5.0s
        let result = a.sub_tensor_optimized(&b);

        // Check that all values are -3.0 (2.0 - 5.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - (-3.0)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_scalar_negative_subtraction() {
        let mut tensor = Tensor::ones(vec![2, 2]);
        tensor.fill(3.0); // Create a tensor with all 3.0s
        let result = tensor.sub_scalar_optimized(8.0);

        // Check that all values are -5.0 (3.0 - 8.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - (-5.0)).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Tensor shapes must match")]
    fn test_mismatched_shapes() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![3, 2]);
        a.sub_tensor_optimized(&b);
    }

    #[test]
    fn test_edge_cases() {
        // Test zero subtraction
        let a = Tensor::ones(vec![3]);
        let b = Tensor::zeros(vec![3]);
        let result = a.sub_tensor_optimized(&b);

        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 1.0).abs() < 1e-6);
            }
        }

        // Test self subtraction
        let mut tensor = Tensor::ones(vec![3]);
        tensor.fill(5.0);
        let result = tensor.sub_tensor_optimized(&tensor);

        unsafe {
            for i in 0..result.size() {
                assert!(result.as_ptr().add(i).read().abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_large_tensor_subtraction() {
        let mut a = Tensor::ones(vec![100, 100]);
        a.fill(10.0);
        let mut b = Tensor::ones(vec![100, 100]);
        b.fill(3.0);
        let result = a.sub_tensor_optimized(&b);

        assert_eq!(result.size(), 10000);

        // Check some values are 7.0 (10.0 - 3.0)
        unsafe {
            for i in (0..result.size()).step_by(1000) {
                assert!((result.as_ptr().add(i).read() - 7.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_negate_inplace() {
        let mut tensor = Tensor::ones(vec![2, 2]);
        tensor.fill(5.0);

        // Check initial values
        unsafe {
            for i in 0..tensor.size() {
                let val = tensor.as_ptr().add(i).read();
                assert!((val - 5.0).abs() < 1e-6, "Expected 5.0, got {}", val);
            }
        }

        tensor.negate_inplace();

        // Check negated values
        unsafe {
            for i in 0..tensor.size() {
                let val = tensor.as_ptr().add(i).read();
                assert!(
                    (val - (-5.0)).abs() < 1e-6,
                    "Expected -5.0 after negation, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_subtraction_with_gradtrack() {
        // Test scalar subtraction with gradtrack
        let a = Tensor::ones(vec![2, 3]).with_requires_grad();
        let mut result = a.sub_scalar(5.0);

        // Check result values: 1.0 - 5.0 = -4.0
        unsafe {
            for i in 0..result.size() {
                let val = result.as_ptr().add(i).read();
                assert!((val - (-4.0)).abs() < 1e-6, "Expected -4.0, got {}", val);
            }
        }

        result.backward(None);

        // Check gradient: d/dx(x - c) = 1
        if let Some(grad) = a.grad_by_value() {
            unsafe {
                for i in 0..grad.size() {
                    let val = grad.as_ptr().add(i).read();
                    assert!(
                        (val - 1.0).abs() < 1e-6,
                        "Expected gradient 1.0, got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient computed for scalar subtraction!");
        }

        // Test tensor subtraction with gradtrack
        let a = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut b = Tensor::ones(vec![2, 2]);
        b.fill(3.0);
        let b = b.with_requires_grad();

        let mut result = a.sub_tensor(&b);

        // Check result values: 1.0 - 3.0 = -2.0
        unsafe {
            for i in 0..result.size() {
                let val = result.as_ptr().add(i).read();
                assert!((val - (-2.0)).abs() < 1e-6, "Expected -2.0, got {}", val);
            }
        }

        result.backward(None);

        // Check gradients: d/dx(x - y) = 1, d/dy(x - y) = -1
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
            panic!("No gradient A computed for tensor subtraction!");
        }

        if let Some(grad_b) = b.grad_by_value() {
            unsafe {
                for i in 0..grad_b.size() {
                    let val = grad_b.as_ptr().add(i).read();
                    println!("Debug: grad_b[{}] = {}", i, val);
                    assert!(
                        (val - (-1.0)).abs() < 1e-6,
                        "Expected gradient B = -1.0, got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient B computed for tensor subtraction!");
        }
    }

    #[test]
    fn test_mixed_add_sub_operations_with_gradtrack() {
        // Test complex computation graph: (a + scalar1) - b + c - scalar2
        let a = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut b = Tensor::ones(vec![2, 2]);
        b.fill(2.0);
        let b = b.with_requires_grad();
        let mut c = Tensor::ones(vec![2, 2]);
        c.fill(3.0);
        let c = c.with_requires_grad();

        let scalar1 = 5.0;
        let scalar2 = 1.0;

        // Complex computation: (a + scalar1) - b + c - scalar2
        // Expected: (1 + 5) - 2 + 3 - 1 = 6
        let step1 = a.add_scalar(scalar1); // a + 5 = 6
        let step2 = step1.sub_tensor(&b); // 6 - 2 = 4
        let step3 = step2.add_tensor(&c); // 4 + 3 = 7
        let mut result = step3.sub_scalar(scalar2); // 7 - 1 = 6

        // Check result values
        unsafe {
            for i in 0..result.size() {
                let val = result.as_ptr().add(i).read();
                assert!((val - 6.0).abs() < 1e-6, "Expected 6.0, got {}", val);
            }
        }

        result.backward(None);

        // Check gradients
        // For computation: f = (a + 5) - b + c - 1
        // df/da = 1, df/db = -1, df/dc = 1

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
                        (val - (-1.0)).abs() < 1e-6,
                        "Expected gradient B = -1.0, got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient B computed for mixed operations!");
        }

        if let Some(grad_c) = c.grad_by_value() {
            unsafe {
                for i in 0..grad_c.size() {
                    let val = grad_c.as_ptr().add(i).read();
                    assert!(
                        (val - 1.0).abs() < 1e-6,
                        "Expected gradient C = 1.0, got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient C computed for mixed operations!");
        }

        println!("Mixed add/sub operations with gradtrack test passed!");
        println!("✓ Complex computation graph: (a + 5) - b + c - 1 = 6");
        println!("✓ Gradients: da/df = 1, db/df = -1, dc/df = 1");
    }

    #[test]
    fn test_sub_broadcasting_gradients_basic() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Test case: [2, 3] - [1, 3] -> [2, 3]
        // grad_a should be [2, 3], grad_b should be [1, 3] (summed over broadcast dim)
        // For subtraction: d/da (a - b) = 1, d/db (a - b) = -1

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[0.1, 0.2, 0.3], vec![1, 3])
            .unwrap()
            .with_requires_grad();

        let mut result = a.sub_tensor(&b);
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
        // This requires summing over the broadcasted dimension
        assert_eq!(
            grad_b.shape().dims,
            vec![1, 3],
            "grad_b should match original shape of b"
        );

        // All gradients should be 1.0 for grad_a (d/da (a - b) = 1)
        for i in 0..grad_a.size() {
            let val = unsafe { *grad_a.as_ptr().add(i) };
            assert!(
                (val - 1.0).abs() < 1e-6,
                "grad_a[{}] = {} should be 1.0",
                i,
                val
            );
        }

        // grad_b should be [-2.0, -2.0, -2.0] (sum over broadcast dim, then negated)
        let expected_grad_b = [-2.0, -2.0, -2.0]; // -1 * 2 rows = -2
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
    fn test_sub_scalar_broadcasting_gradients() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Test case: [2, 3] - [1] -> [2, 3]
        // grad_a should be [2, 3], grad_b should be [1] (summed over all dims, then negated)

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[0.5], vec![1])
            .unwrap()
            .with_requires_grad();

        let mut result = a.sub_tensor(&b);
        result.backward(None);

        let grad_a = a.grad_by_value().expect("grad_a should exist");
        let grad_b = b.grad_by_value().expect("grad_b should exist");

        // grad_a should have same shape as a: [2, 3]
        assert_eq!(grad_a.shape().dims, vec![2, 3]);

        // grad_b should have same shape as b: [1] and sum to -6.0
        println!("grad_b shape: {:?}, expected: [1]", grad_b.shape().dims);
        assert_eq!(grad_b.shape().dims, vec![1]);

        // grad_b should be -6.0 (sum over all 6 elements, then negated)
        let val = unsafe { *grad_b.as_ptr() };
        assert!(
            (val - (-6.0)).abs() < 1e-6,
            "grad_b = {} should be -6.0",
            val
        );
    }
}
