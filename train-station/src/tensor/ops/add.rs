//! Addition operations for tensors
//!
//! Provides element-wise addition following PyTorch conventions with comprehensive
//! broadcasting support, automatic differentiation, and high-performance SIMD optimization.
//!
//! # Key Features
//!
//! - **Element-wise Addition**: `add_tensor()` - Addition with another tensor (PyTorch `add()` equivalent)
//! - **Scalar Broadcasting**: `add_scalar()` - Addition with scalar values
//! - **Automatic Broadcasting**: NumPy-style broadcasting for compatible shapes
//! - **SIMD Optimization**: AVX2 acceleration on x86_64 hardware
//! - **Automatic Differentiation**: Full gradtrack support with gradient tracking
//! - **Cache Optimization**: Memory access patterns optimized for modern CPUs
//! - **Zero-copy Operations**: Efficient memory usage where possible
//!
//! # Broadcasting Support
//!
//! All addition operations support automatic broadcasting following NumPy rules:
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

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// (Removed manual prefetching: simplifies hot path; modern CPUs prefetch effectively for linear access)

impl Tensor {
    /// Element-wise addition with another tensor with broadcasting support.
    ///
    /// Performs element-wise addition with automatic broadcasting: `output[i] = self[i] + other[i]`
    ///
    /// Broadcasting enables addition between tensors of different but compatible shapes.
    /// Compatible shapes follow NumPy broadcasting rules:
    /// - Dimensions are aligned from the rightmost dimension
    /// - Dimensions are compatible if they are equal, or one of them is 1
    /// - Missing dimensions are treated as 1
    ///
    /// # Arguments
    /// * `other` - Tensor to add. Shapes must be broadcast-compatible.
    ///
    /// # Returns
    /// A new tensor containing the element-wise sum with broadcast result shape
    ///
    /// # Examples
    ///
    /// ## Same Shape Addition
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]).unwrap();
    /// let c = a.add_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![3]);
    /// assert_eq!(c.get(&[0]), 5.0);
    /// assert_eq!(c.get(&[1]), 7.0);
    /// assert_eq!(c.get(&[2]), 9.0);
    /// ```
    ///
    /// ## Broadcasting Addition
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Broadcasting: [2, 1] + [1, 3] -> [2, 3]
    /// let a = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]).unwrap();
    /// let b = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![1, 3]).unwrap();
    /// let c = a.add_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![2, 3]);
    /// assert_eq!(c.get(&[0, 0]), 11.0);
    /// assert_eq!(c.get(&[0, 1]), 21.0);
    /// assert_eq!(c.get(&[1, 0]), 12.0);
    /// assert_eq!(c.get(&[1, 1]), 22.0);
    /// ```
    ///
    /// ## Scalar Broadcasting
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Scalar broadcasting: [2, 3] + scalar -> [2, 3]
    /// let a = Tensor::ones(vec![2, 3]);
    /// let b = Tensor::from_slice(&[5.0], vec![1]).unwrap();
    /// let c = a.add_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![2, 3]);
    /// assert_eq!(c.get(&[0, 0]), 6.0);
    /// assert_eq!(c.get(&[1, 2]), 6.0);
    /// ```
    ///
    /// # Panics
    /// Panics if tensor shapes are not broadcast-compatible
    #[inline]
    pub fn add_tensor(&self, other: &Tensor) -> Tensor {
        // Check if shapes are identical for fast path
        if self.shape().dims == other.shape().dims {
            return self.add_tensor_same_shape(other);
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

        // Perform element-wise addition on broadcasted tensors
        let mut result = broadcast_self.add_tensor_optimized(&broadcast_other);

        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Add {
                is_tensor_add: true,
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

    /// Element-wise addition for tensors with identical shapes (fast path).
    ///
    /// This is an optimized path for tensors that already have the same shape,
    /// avoiding the overhead of broadcasting computation. Used internally by
    /// `add_tensor()` when shapes are identical.
    ///
    /// # Arguments
    /// * `other` - Tensor to add, must have the same shape as self
    ///
    /// # Returns
    /// A new tensor containing the element-wise sum
    ///
    /// # Performance Characteristics
    ///
    /// - **Fast Path**: Avoids broadcasting overhead for identical shapes
    /// - **SIMD Optimization**: Uses optimized tensor addition with SIMD acceleration
    /// - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
    ///
    /// # Panics
    ///
    /// Panics if tensor shapes do not match
    #[inline]
    fn add_tensor_same_shape(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensor shapes must match for same-shape addition"
        );
        let mut result = self.add_tensor_optimized(other);

        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Add {
                is_tensor_add: true,
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

    /// Broadcast addition with a scalar value.
    ///
    /// Adds the scalar to every element: `output[i] = self[i] + scalar`
    ///
    /// # Arguments
    /// * `scalar` - Value to add to each element
    ///
    /// # Returns
    /// A new tensor with the scalar added to each element
    ///
    /// # Examples
    ///
    /// ## Basic Scalar Addition
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = a.add_scalar(10.0);
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 11.0);
    /// assert_eq!(b.get(&[1]), 12.0);
    /// assert_eq!(b.get(&[2]), 13.0);
    /// ```
    ///
    /// ## Multi-dimensional Scalar Addition
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::ones(vec![2, 3]);
    /// let b = a.add_scalar(5.0);
    /// assert_eq!(b.shape().dims, vec![2, 3]);
    /// assert_eq!(b.get(&[0, 0]), 6.0);
    /// assert_eq!(b.get(&[1, 2]), 6.0);
    /// ```
    #[inline]
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let mut result = self.add_scalar_optimized(scalar);

        if self.requires_grad() && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Add {
                is_tensor_add: false,
                original_shapes: None, // Scalar case
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
        }

        result
    }
    /// Internal optimized tensor + tensor operation
    ///
    /// Performs element-wise addition between two tensors with the same shape,
    /// using SIMD acceleration when available. This is the core implementation
    /// used by `add_tensor()` after broadcasting has been applied.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor to add, must have the same shape as self
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise sum
    ///
    /// # Safety
    ///
    /// Assumes both tensors have the same shape and valid memory layouts.
    /// Uses unsafe SIMD operations for performance optimization.
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: Uses AVX2 when available for 8x vectorization
    /// - **Unrolled Loops**: 4x unrolling for optimal instruction throughput
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Fallback**: Optimized scalar implementation for non-SIMD hardware
    #[inline]
    pub(crate) fn add_tensor_optimized(&self, other: &Tensor) -> Tensor {
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
                    self.add_tensors_simd_avx2_optimized(a, b, dst);
                    return output;
                }
            }

            // Fallback to scalar operations with better cache usage
            self.add_tensors_scalar_optimized(a, b, dst);
        }

        output
    }

    /// SIMD-optimized tensor addition using AVX2 instructions
    ///
    /// Performs element-wise addition using AVX2 SIMD instructions for maximum
    /// performance on x86_64 hardware. Processes 32 elements per iteration with
    /// 4x unrolling for optimal instruction throughput.
    ///
    /// # Arguments
    ///
    /// * `a` - Pointer to first tensor data
    /// * `b` - Pointer to second tensor data
    /// * `dst` - Pointer to output tensor data
    ///
    /// # Safety
    ///
    /// Requires AVX2 support and valid pointers with sufficient memory.
    /// All pointers must be aligned and point to valid tensor data.
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Width**: 8 elements per AVX2 vector operation
    /// - **Unrolling**: 4x unrolling (32 elements per iteration)
    /// - **Memory Access**: Linear access patterns for cache efficiency
    /// - **Fallback**: Handles remaining elements with scalar operations
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_tensors_simd_avx2_optimized(&self, a: *const f32, b: *const f32, dst: *mut f32) {
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let mut offset = 0;

        // Unrolled SIMD loop for throughput
        for _ in 0..simd_count {
            // Process 4 AVX2 vectors (32 elements) per iteration
            let a_vec1 = _mm256_loadu_ps(a.add(offset));
            let b_vec1 = _mm256_loadu_ps(b.add(offset));
            let sum_vec1 = _mm256_add_ps(a_vec1, b_vec1);
            _mm256_storeu_ps(dst.add(offset), sum_vec1);

            let a_vec2 = _mm256_loadu_ps(a.add(offset + 8));
            let b_vec2 = _mm256_loadu_ps(b.add(offset + 8));
            let sum_vec2 = _mm256_add_ps(a_vec2, b_vec2);
            _mm256_storeu_ps(dst.add(offset + 8), sum_vec2);

            let a_vec3 = _mm256_loadu_ps(a.add(offset + 16));
            let b_vec3 = _mm256_loadu_ps(b.add(offset + 16));
            let sum_vec3 = _mm256_add_ps(a_vec3, b_vec3);
            _mm256_storeu_ps(dst.add(offset + 16), sum_vec3);

            let a_vec4 = _mm256_loadu_ps(a.add(offset + 24));
            let b_vec4 = _mm256_loadu_ps(b.add(offset + 24));
            let sum_vec4 = _mm256_add_ps(a_vec4, b_vec4);
            _mm256_storeu_ps(dst.add(offset + 24), sum_vec4);

            offset += 32;
        }

        // Handle remaining elements in blocks of 8 then tail
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            let a_vec = _mm256_loadu_ps(a.add(offset));
            let b_vec = _mm256_loadu_ps(b.add(offset));
            let sum_vec = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(dst.add(offset), sum_vec);
            offset += 8;
        }
        while offset + 4 <= size {
            *dst.add(offset) = *a.add(offset) + *b.add(offset);
            *dst.add(offset + 1) = *a.add(offset + 1) + *b.add(offset + 1);
            *dst.add(offset + 2) = *a.add(offset + 2) + *b.add(offset + 2);
            *dst.add(offset + 3) = *a.add(offset + 3) + *b.add(offset + 3);
            offset += 4;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    /// Optimized scalar tensor addition fallback
    ///
    /// Performs element-wise addition using optimized scalar operations when
    /// SIMD is not available. Uses 4x unrolling for better instruction-level
    /// parallelism and cache efficiency.
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
    #[inline]
    unsafe fn add_tensors_scalar_optimized(&self, a: *const f32, b: *const f32, dst: *mut f32) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar loop for better performance
        for _ in 0..unroll_count {
            *dst.add(offset) = *a.add(offset) + *b.add(offset);
            *dst.add(offset + 1) = *a.add(offset + 1) + *b.add(offset + 1);
            *dst.add(offset + 2) = *a.add(offset + 2) + *b.add(offset + 2);
            *dst.add(offset + 3) = *a.add(offset + 3) + *b.add(offset + 3);
            offset += 4;
        }

        // Handle remaining elements
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    /// Internal optimized scalar + tensor operation
    ///
    /// Performs element-wise addition of a scalar to each element of the tensor,
    /// using SIMD acceleration when available. This is the core implementation
    /// used by `add_scalar()`.
    ///
    /// # Arguments
    ///
    /// * `scalar` - Scalar value to add to each element
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar added to each element
    ///
    /// # Safety
    ///
    /// Assumes valid tensor memory layout. Uses unsafe SIMD operations for
    /// performance optimization.
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: Uses AVX2 when available for 8x vectorization
    /// - **Unrolled Loops**: 4x unrolling for optimal instruction throughput
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Fallback**: Optimized scalar implementation for non-SIMD hardware
    #[inline]
    pub(crate) fn add_scalar_optimized(&self, scalar: f32) -> Tensor {
        let mut output = Tensor::new(self.shape().dims.clone());

        unsafe {
            let src = self.as_ptr();
            let dst = output.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                // Use SIMD for better performance when available
                if is_x86_feature_detected!("avx2") {
                    self.add_scalar_simd_avx2_optimized(src, dst, scalar);
                    return output;
                }
            }

            // Fallback to optimized scalar operations
            self.add_scalar_fallback_optimized(src, dst, scalar);
        }

        output
    }

    /// SIMD-optimized scalar addition using AVX2 instructions
    ///
    /// Performs element-wise scalar addition using AVX2 SIMD instructions for maximum
    /// performance on x86_64 hardware. Processes 32 elements per iteration with
    /// 4x unrolling for optimal instruction throughput.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `scalar` - Scalar value to add to each element
    ///
    /// # Safety
    ///
    /// Requires AVX2 support and valid pointers with sufficient memory.
    /// All pointers must be aligned and point to valid tensor data.
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Width**: 8 elements per AVX2 vector operation
    /// - **Unrolling**: 4x unrolling (32 elements per iteration)
    /// - **Memory Access**: Linear access patterns for cache efficiency
    /// - **Fallback**: Handles remaining elements with scalar operations
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_scalar_simd_avx2_optimized(&self, src: *const f32, dst: *mut f32, scalar: f32) {
        let scalar_vec = _mm256_set1_ps(scalar);
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration
        let mut offset = 0;

        // Unrolled SIMD loop for instruction throughput
        for _ in 0..simd_count {
            let src_vec1 = _mm256_loadu_ps(src.add(offset));
            let sum_vec1 = _mm256_add_ps(src_vec1, scalar_vec);
            _mm256_storeu_ps(dst.add(offset), sum_vec1);

            let src_vec2 = _mm256_loadu_ps(src.add(offset + 8));
            let sum_vec2 = _mm256_add_ps(src_vec2, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 8), sum_vec2);

            let src_vec3 = _mm256_loadu_ps(src.add(offset + 16));
            let sum_vec3 = _mm256_add_ps(src_vec3, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 16), sum_vec3);

            let src_vec4 = _mm256_loadu_ps(src.add(offset + 24));
            let sum_vec4 = _mm256_add_ps(src_vec4, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 24), sum_vec4);

            offset += 32;
        }

        // Handle remaining 8-element blocks
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            let src_vec = _mm256_loadu_ps(src.add(offset));
            let sum_vec = _mm256_add_ps(src_vec, scalar_vec);
            _mm256_storeu_ps(dst.add(offset), sum_vec);
            offset += 8;
        }

        // Handle final elements
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    /// Optimized scalar addition fallback
    ///
    /// Performs element-wise scalar addition using optimized scalar operations when
    /// SIMD is not available. Uses 4x unrolling for better instruction-level
    /// parallelism and cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `scalar` - Scalar value to add to each element
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
    #[inline]
    unsafe fn add_scalar_fallback_optimized(&self, src: *const f32, dst: *mut f32, scalar: f32) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar operations with while for clarity
        for _ in 0..unroll_count {
            *dst.add(offset) = *src.add(offset) + scalar;
            *dst.add(offset + 1) = *src.add(offset + 1) + scalar;
            *dst.add(offset + 2) = *src.add(offset + 2) + scalar;
            *dst.add(offset + 3) = *src.add(offset + 3) + scalar;
            offset += 4;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_addition() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![2, 3]);
        let result = a.add_tensor_optimized(&b);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are 2.0 (1.0 + 1.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 2.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_scalar_addition() {
        let tensor = Tensor::ones(vec![2, 2]);
        let result = tensor.add_scalar_optimized(5.0);

        assert_eq!(result.shape().dims, vec![2, 2]);
        assert_eq!(result.size(), 4);

        // Check that all values are 6.0 (1.0 + 5.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 6.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Tensor shapes must match")]
    fn test_mismatched_shapes() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![3, 2]);
        a.add_tensor_optimized(&b);
    }

    #[test]
    fn test_add_with_no_grad_guard() {
        use crate::gradtrack::{is_grad_enabled, NoGradTrack};

        // Create tensors with requires_grad enabled
        let a = Tensor::ones(vec![2, 2]).with_requires_grad();
        let b = Tensor::ones(vec![2, 2]).with_requires_grad();

        // Verify gradients are enabled by default
        assert!(is_grad_enabled());

        // Normal addition with gradients
        let c1 = a.add_tensor(&b);
        assert!(
            c1.requires_grad(),
            "Result should require gradients normally"
        );

        // Addition with NoGradTrack - gradients should be disabled
        {
            let _guard = NoGradTrack::new();
            assert!(
                !is_grad_enabled(),
                "Gradients should be disabled within guard"
            );

            let c2 = a.add_tensor(&b);
            assert!(
                !c2.requires_grad(),
                "Result should not require gradients within NoGradTrack"
            );

            // Test scalar addition as well
            let c3 = a.add_scalar(5.0);
            assert!(
                !c3.requires_grad(),
                "Scalar addition result should not require gradients within NoGradTrack"
            );
        }

        // Gradients should be restored after guard goes out of scope
        assert!(
            is_grad_enabled(),
            "Gradients should be restored after guard"
        );

        let c4 = a.add_tensor(&b);
        assert!(
            c4.requires_grad(),
            "Result should require gradients after guard is dropped"
        );
    }

    #[test]
    fn test_add_nested_no_grad_guards() {
        use crate::gradtrack::{is_grad_enabled, NoGradTrack};

        let a = Tensor::ones(vec![2, 2]).with_requires_grad();
        let b = Tensor::ones(vec![2, 2]).with_requires_grad();

        assert!(is_grad_enabled());

        {
            let _guard1 = NoGradTrack::new();
            assert!(!is_grad_enabled());

            let c1 = a.add_tensor(&b);
            assert!(!c1.requires_grad());

            {
                let _guard2 = NoGradTrack::new();
                assert!(!is_grad_enabled());

                let c2 = a.add_tensor(&b);
                assert!(!c2.requires_grad());
            }

            // Still disabled after inner guard drops
            assert!(!is_grad_enabled());
            let c3 = a.add_tensor(&b);
            assert!(!c3.requires_grad());
        }

        // Restored after all guards drop
        assert!(is_grad_enabled());
        let c4 = a.add_tensor(&b);
        assert!(c4.requires_grad());
    }

    #[test]
    fn test_add_with_mixed_requires_grad() {
        use crate::gradtrack::NoGradTrack;

        let a = Tensor::ones(vec![2, 2]).with_requires_grad(); // requires_grad = true
        let b = Tensor::ones(vec![2, 2]); // requires_grad = false

        // Without NoGradTrack, result should require gradients if any input does
        let c1 = a.add_tensor(&b);
        assert!(c1.requires_grad());

        let c2 = b.add_tensor(&a);
        assert!(c2.requires_grad());

        // With NoGradTrack, result should not require gradients regardless
        {
            let _guard = NoGradTrack::new();

            let c3 = a.add_tensor(&b);
            assert!(!c3.requires_grad());

            let c4 = b.add_tensor(&a);
            assert!(!c4.requires_grad());
        }
    }

    #[test]
    fn test_add_performance_no_overhead() {
        use crate::gradtrack::NoGradTrack;
        use std::time::Instant;

        let size = 1000; // Smaller size for test stability
        let a = Tensor::ones(vec![size]).with_requires_grad();
        let b = Tensor::ones(vec![size]);

        // Time normal addition (with potential grad overhead)
        let start = Instant::now();
        for _ in 0..10 {
            let _ = a.add_tensor(&b);
        }
        let normal_duration = start.elapsed();

        // Time addition with NoGradTrack (should be faster)
        let start = Instant::now();
        {
            let _guard = NoGradTrack::new();
            for _ in 0..10 {
                let _ = a.add_tensor(&b);
            }
        }
        let no_grad_duration = start.elapsed();

        // NoGradTrack should provide performance benefit by skipping gradtrack setup
        // Allow generous variance for timing inconsistencies in tests
        println!(
            "Normal: {:?}, NoGrad: {:?}",
            normal_duration, no_grad_duration
        );

        // The key verification is that NoGradTrack doesn't add overhead
        assert!(
            no_grad_duration <= normal_duration * 3,
            "NoGradTrack should not add significant overhead"
        );
    }

    #[test]
    fn test_broadcasting_gradients_basic() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Test case: [2, 3] + [1, 3] -> [2, 3]
        // grad_a should be [2, 3], grad_b should be [1, 3] (summed over broadcast dim)

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[0.1, 0.2, 0.3], vec![1, 3])
            .unwrap()
            .with_requires_grad();

        let mut result = a.add_tensor(&b);
        assert_eq!(result.shape().dims, vec![2, 3]);

        // Set upstream gradient as ones
        result.backward(None);

        // Check gradients
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

        // All gradients should be 1.0 for grad_a
        for i in 0..grad_a.size() {
            let val = unsafe { *grad_a.as_ptr().add(i) };
            assert!(
                (val - 1.0).abs() < 1e-6,
                "grad_a[{}] = {} should be 1.0",
                i,
                val
            );
        }

        // grad_b should be [2.0, 2.0, 2.0] (sum over broadcast dim)
        let expected_grad_b = [2.0, 2.0, 2.0];
        for (i, val) in expected_grad_b.iter().enumerate().take(grad_b.size()) {
            let actual = unsafe { *grad_b.as_ptr().add(i) };
            assert!(
                (actual - val).abs() < 1e-6,
                "grad_b[{}] = {} should be {}",
                i,
                actual,
                val
            );
        }
    }

    #[test]
    fn test_scalar_broadcasting_gradients() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Test case: [2, 3] + [1] -> [2, 3]
        // grad_a should be [2, 3], grad_b should be [1] (summed over all dims)

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[0.5], vec![1])
            .unwrap()
            .with_requires_grad();

        let mut result = a.add_tensor(&b);
        result.backward(None);

        let grad_a = a.grad_by_value().expect("grad_a should exist");
        let grad_b = b.grad_by_value().expect("grad_b should exist");

        // grad_a should have same shape as a: [2, 3]
        assert_eq!(grad_a.shape().dims, vec![2, 3]);

        // grad_b should have same shape as b: [1] and sum to 6.0
        println!("grad_b shape: {:?}, expected: [1]", grad_b.shape().dims);
        assert_eq!(grad_b.shape().dims, vec![1]);

        // grad_b should be 6.0 (sum over all 6 elements)
        let val = unsafe { *grad_b.as_ptr() };
        assert!((val - 6.0).abs() < 1e-6, "grad_b = {} should be 6.0", val);
    }

    #[test]
    fn test_linear_layer_bias_broadcasting() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Simulate linear layer bias broadcasting
        // input: [2, 3], weight: [3, 4], bias: [4]
        // matmul result: [2, 4], bias broadcast: [4] -> [2, 4]

        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let weight = Tensor::from_slice(
            &(1..=12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            vec![3, 4],
        )
        .unwrap()
        .with_requires_grad();
        let bias = Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], vec![4])
            .unwrap()
            .with_requires_grad();

        // Forward pass: input @ weight + bias
        let matmul_result = input.matmul(&weight);
        println!("Matmul result shape: {:?}", matmul_result.shape().dims);
        println!("Bias shape: {:?}", bias.shape().dims);

        let linear_output = matmul_result.add_tensor(&bias);
        println!("Linear output shape: {:?}", linear_output.shape().dims);

        // Sum all outputs as loss
        let mut loss = linear_output.sum();
        loss.backward(None);

        // Check bias gradient
        let bias_grad = bias.grad_by_value().expect("bias gradient should exist");
        println!("Bias gradient shape: {:?}", bias_grad.shape().dims);
        assert_eq!(
            bias_grad.shape().dims,
            vec![4],
            "bias gradient should match bias shape"
        );

        // Bias gradient should be [2.0, 2.0, 2.0, 2.0] (sum over batch dimension)
        for i in 0..4 {
            let val = unsafe { *bias_grad.as_ptr().add(i) };
            assert!(
                (val - 2.0).abs() < 1e-6,
                "bias_grad[{}] = {} should be 2.0",
                i,
                val
            );
        }

        println!("Linear layer bias broadcasting test passed!");
    }
}
