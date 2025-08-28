//! Multiplication operations for tensors
//!
//! Provides element-wise multiplication functions following PyTorch conventions with
//! comprehensive automatic differentiation support and SIMD-optimized computation.
//!
//! # Key Features
//!
//! - **Element-wise Multiplication**: `mul_tensor()` - Element-wise multiplication with another tensor (PyTorch `mul()` equivalent)
//! - **Scalar Multiplication**: `mul_scalar()` - Broadcast multiplication with a scalar value
//! - **Automatic Differentiation**: Full gradtrack support with efficient gradient computation
//! - **SIMD Optimization**: AVX2-optimized implementation for maximum performance
//! - **Broadcasting Support**: NumPy-style broadcasting for compatible shapes
//! - **Cache Optimization**: Memory access patterns optimized for modern CPUs
//! - **Mathematical Accuracy**: High-precision multiplication computation
//!
//! # Mathematical Properties
//!
//! The multiplication operations have the following properties:
//! - **Commutative**: a * b = b * a
//! - **Associative**: (a * b) * c = a * (b * c)
//! - **Distributive**: a * (b + c) = a * b + a * c
//! - **Identity**: a * 1 = a
//! - **Zero**: a * 0 = 0
//! - **Gradient**: d/dx(a * b) = b, d/dy(a * b) = a
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

// Note: removed manual prefetching; linear access + hardware prefetch is sufficient

impl Tensor {
    /// Element-wise multiplication with another tensor with broadcasting support.
    ///
    /// Performs element-wise multiplication with automatic broadcasting: `output[i] = self[i] * other[i]`
    ///
    /// Broadcasting enables multiplication between tensors of different but compatible shapes.
    /// Compatible shapes follow NumPy broadcasting rules:
    /// - Dimensions are aligned from the rightmost dimension
    /// - Dimensions are compatible if they are equal, or one of them is 1
    /// - Missing dimensions are treated as 1
    ///
    /// # Arguments
    /// * `other` - Tensor to multiply. Shapes must be broadcast-compatible.
    ///
    /// # Returns
    /// A new tensor containing the element-wise product with broadcast result shape
    ///
    /// # Examples
    ///
    /// ## Same Shape Multiplication
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]).unwrap();
    /// let b = Tensor::from_slice(&[5.0, 6.0, 7.0], vec![3]).unwrap();
    /// let c = a.mul_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![3]);
    /// assert_eq!(c.get(&[0]), 10.0); // 2.0 * 5.0
    /// assert_eq!(c.get(&[1]), 18.0); // 3.0 * 6.0
    /// assert_eq!(c.get(&[2]), 28.0); // 4.0 * 7.0
    /// ```
    ///
    /// ## Broadcasting Multiplication
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[2.0, 3.0], vec![2, 1]).unwrap();
    /// let b = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![1, 3]).unwrap();
    /// let c = a.mul_tensor(&b);
    /// assert_eq!(c.shape().dims, vec![2, 3]);
    /// // Result: [[20.0, 40.0, 60.0], [30.0, 60.0, 90.0]]
    /// assert_eq!(c.get(&[0, 0]), 20.0); // 2.0 * 10.0
    /// assert_eq!(c.get(&[0, 1]), 40.0); // 2.0 * 20.0
    /// assert_eq!(c.get(&[1, 0]), 30.0); // 3.0 * 10.0
    /// ```
    ///
    /// # Panics
    /// Panics if tensor shapes are not broadcast-compatible
    #[inline]
    pub fn mul_tensor(&self, other: &Tensor) -> Tensor {
        // Check if shapes are identical for fast path
        if self.shape().dims == other.shape().dims {
            return self.mul_tensor_same_shape(other);
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

        // Perform element-wise multiplication on broadcasted tensors
        let mut result = broadcast_self.mul_tensor_optimized(&broadcast_other);

        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let operands = vec![self.clone(), other.clone()];
            let grad_fn = GradFn::Mul {
                is_tensor_mul: true,
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

    /// Element-wise multiplication for tensors with identical shapes (fast path)
    ///
    /// This is an optimized path for tensors that already have the same shape,
    /// avoiding the overhead of broadcasting computation. This method provides
    /// better performance when tensors have matching dimensions.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor to multiply, must have the same shape as self
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise product
    ///
    /// # Performance Characteristics
    ///
    /// - **Fast Path**: Avoids broadcasting overhead for identical shapes
    /// - **SIMD Optimization**: Uses optimized multiplication when available
    /// - **Memory Efficiency**: Direct element-wise computation without shape conversion
    /// - **Gradient Tracking**: Full gradtrack support with efficient gradient computation
    ///
    /// # Panics
    ///
    /// Panics if tensor shapes do not match
    ///
    /// # Implementation Details
    ///
    /// This method is called internally by `mul_tensor()` when shapes are identical.
    /// It provides a performance optimization by skipping the broadcasting logic
    /// and directly calling the optimized multiplication implementation.
    #[inline]
    fn mul_tensor_same_shape(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensor shapes must match for same-shape multiplication"
        );
        let mut result = self.mul_tensor_optimized(other);

        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let operands = vec![self.clone(), other.clone()];
            let grad_fn = GradFn::Mul {
                is_tensor_mul: true,
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

    /// Broadcast multiplication with a scalar value.
    ///
    /// Multiplies every element by the scalar: `output[i] = self[i] * scalar`
    ///
    /// # Arguments
    /// * `scalar` - Value to multiply with each element
    ///
    /// # Returns
    /// A new tensor with each element multiplied by the scalar
    ///
    /// # Examples
    ///
    /// ## Basic Scalar Multiplication
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = a.mul_scalar(10.0);
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 10.0); // 1.0 * 10.0
    /// assert_eq!(b.get(&[1]), 20.0); // 2.0 * 10.0
    /// assert_eq!(b.get(&[2]), 30.0); // 3.0 * 10.0
    /// ```
    ///
    /// ## Negative Scalar Multiplication
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = a.mul_scalar(-2.0);
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), -2.0); // 1.0 * -2.0
    /// assert_eq!(b.get(&[1]), -4.0); // 2.0 * -2.0
    /// assert_eq!(b.get(&[2]), -6.0); // 3.0 * -2.0
    /// ```
    #[inline]
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let mut result = self.mul_scalar_optimized(scalar);

        if self.requires_grad() && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Mul {
                is_tensor_mul: false,
                scalar: Some(scalar),
                operands: None,
                original_shapes: None, // Scalar case
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
        }

        result
    }
    /// Optimized tensor multiplication using SIMD when available
    ///
    /// Performs element-wise multiplication using SIMD optimization when available
    /// and falling back to optimized scalar computation. This is the core implementation
    /// used by `mul_tensor()`.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to multiply with
    ///
    /// # Returns
    ///
    /// A new tensor with the result of the multiplication
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks when available
    /// - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Memory Safety**: Ensures contiguous memory layout for correctness
    /// - **Zero-sized Handling**: Fast return for empty tensors
    ///
    /// # Safety
    ///
    /// This operation assumes the tensors have the same shape. The method ensures
    /// contiguous memory layout for both input tensors to guarantee correctness
    /// with view tensors.
    ///
    /// # Implementation Details
    ///
    /// Automatically selects between SIMD and scalar implementations based on hardware
    /// capabilities. Ensures both input tensors are contiguous for memory safety.
    /// SIMD implementation processes 32 elements per iteration with 4x unrolling.
    #[inline]
    pub(crate) fn mul_tensor_optimized(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "Tensor shapes must match");
        // Ensure contiguous sources for correctness with view tensors
        let a_src = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };
        let b_src = if other.is_contiguous() {
            other.clone()
        } else {
            other.contiguous()
        };

        let mut output = Tensor::new(self.shape().dims.clone());

        unsafe {
            let a = a_src.as_ptr();
            let b = b_src.as_ptr();
            let dst = output.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                // Use SIMD for better performance when available
                if is_x86_feature_detected!("avx2") {
                    self.mul_tensors_simd_avx2_optimized(a, b, dst);
                    return output;
                }
            }

            // Fallback to scalar operations with better cache usage
            self.mul_tensors_scalar_optimized(a, b, dst);
        }

        output
    }

    /// AVX2-optimized tensor multiplication implementation
    ///
    /// Performs element-wise multiplication using AVX2 SIMD instructions for maximum
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
    /// - **Vector Operations**: Uses AVX2 multiplication instructions
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Hardware Requirements**: Requires x86_64 with AVX2 support
    ///
    /// # Implementation Details
    ///
    /// Uses AVX2 vector instructions to process 8 elements simultaneously.
    /// Implements 4x unrolling for optimal instruction throughput and cache utilization.
    /// Processes remaining elements with scalar operations for complete coverage.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_tensors_simd_avx2_optimized(&self, a: *const f32, b: *const f32, dst: *mut f32) {
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let mut offset = 0;

        // Unrolled SIMD loop for throughput
        for _ in 0..simd_count {
            // Process 4 AVX2 vectors (32 elements) per iteration
            let a_vec1 = _mm256_loadu_ps(a.add(offset));
            let b_vec1 = _mm256_loadu_ps(b.add(offset));
            let mul_vec1 = _mm256_mul_ps(a_vec1, b_vec1);
            _mm256_storeu_ps(dst.add(offset), mul_vec1);

            let a_vec2 = _mm256_loadu_ps(a.add(offset + 8));
            let b_vec2 = _mm256_loadu_ps(b.add(offset + 8));
            let mul_vec2 = _mm256_mul_ps(a_vec2, b_vec2);
            _mm256_storeu_ps(dst.add(offset + 8), mul_vec2);

            let a_vec3 = _mm256_loadu_ps(a.add(offset + 16));
            let b_vec3 = _mm256_loadu_ps(b.add(offset + 16));
            let mul_vec3 = _mm256_mul_ps(a_vec3, b_vec3);
            _mm256_storeu_ps(dst.add(offset + 16), mul_vec3);

            let a_vec4 = _mm256_loadu_ps(a.add(offset + 24));
            let b_vec4 = _mm256_loadu_ps(b.add(offset + 24));
            let mul_vec4 = _mm256_mul_ps(a_vec4, b_vec4);
            _mm256_storeu_ps(dst.add(offset + 24), mul_vec4);

            offset += 32;
        }

        // Handle remaining 8-element blocks, then tail
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            let a_vec = _mm256_loadu_ps(a.add(offset));
            let b_vec = _mm256_loadu_ps(b.add(offset));
            let mul_vec = _mm256_mul_ps(a_vec, b_vec);
            _mm256_storeu_ps(dst.add(offset), mul_vec);
            offset += 8;
        }
        while offset + 4 <= size {
            *dst.add(offset) = *a.add(offset) * *b.add(offset);
            *dst.add(offset + 1) = *a.add(offset + 1) * *b.add(offset + 1);
            *dst.add(offset + 2) = *a.add(offset + 2) * *b.add(offset + 2);
            *dst.add(offset + 3) = *a.add(offset + 3) * *b.add(offset + 3);
            offset += 4;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) * *b.add(i);
        }
    }

    /// Optimized scalar tensor multiplication fallback
    ///
    /// Performs element-wise multiplication using optimized scalar operations with
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
    /// - **Mathematical Accuracy**: High-precision scalar multiplication
    ///
    /// # Implementation Details
    ///
    /// Uses 4x unrolled scalar operations for optimal performance on non-SIMD hardware.
    /// Processes elements in groups of 4 to improve instruction-level parallelism
    /// and reduce loop overhead.
    #[inline]
    unsafe fn mul_tensors_scalar_optimized(&self, a: *const f32, b: *const f32, dst: *mut f32) {
        let size = self.size();

        // Use unrolled loops for better instruction throughput
        let unroll_count = size / 4;
        let mut i = 0;

        // Process 4 elements at a time for better cache utilization
        while i < unroll_count {
            let idx = i * 4;
            dst.add(idx).write(a.add(idx).read() * b.add(idx).read());
            dst.add(idx + 1)
                .write(a.add(idx + 1).read() * b.add(idx + 1).read());
            dst.add(idx + 2)
                .write(a.add(idx + 2).read() * b.add(idx + 2).read());
            dst.add(idx + 3)
                .write(a.add(idx + 3).read() * b.add(idx + 3).read());
            i += 1;
        }

        // Handle remaining elements
        for j in (unroll_count * 4)..size {
            dst.add(j).write(a.add(j).read() * b.add(j).read());
        }
    }

    /// Optimized scalar multiplication using SIMD when available
    ///
    /// Performs element-wise scalar multiplication using SIMD optimization when available
    /// and falling back to optimized scalar computation. This is the core implementation
    /// used by `mul_scalar()`.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to multiply by
    ///
    /// # Returns
    ///
    /// A new tensor with the result of the multiplication
    ///
    /// # Performance Characteristics
    ///
    /// - **SIMD Optimization**: AVX2-optimized with 32-element blocks when available
    /// - **Scalar Fallback**: 4x unrolled scalar implementation for non-SIMD hardware
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Memory Safety**: Ensures contiguous memory layout for correctness
    /// - **Zero-sized Handling**: Fast return for empty tensors
    ///
    /// # Implementation Details
    ///
    /// Automatically selects between SIMD and scalar implementations based on hardware
    /// capabilities. Ensures the input tensor is contiguous for memory safety.
    /// SIMD implementation processes 32 elements per iteration with 4x unrolling.
    #[inline]
    pub(crate) fn mul_scalar_optimized(&self, scalar: f32) -> Tensor {
        // Ensure contiguous source for correctness with view tensors
        let src_self = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };
        let mut output = Tensor::new(self.shape().dims.clone());

        unsafe {
            let src = src_self.as_ptr();
            let dst = output.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                // Use SIMD for better performance when available
                if is_x86_feature_detected!("avx2") {
                    self.mul_scalar_simd_avx2_optimized(src, dst, scalar);
                    return output;
                }
            }

            // Fallback to scalar operations with better cache usage
            self.mul_scalar_fallback_optimized(src, dst, scalar);
        }

        output
    }

    /// AVX2-optimized scalar multiplication implementation
    ///
    /// Performs element-wise scalar multiplication using AVX2 SIMD instructions for maximum
    /// performance on x86_64 architectures with AVX2 support.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `scalar` - Scalar value to multiply by
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
    /// - **Vector Operations**: Uses AVX2 multiplication instructions
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Hardware Requirements**: Requires x86_64 with AVX2 support
    ///
    /// # Implementation Details
    ///
    /// Uses AVX2 vector instructions to process 8 elements simultaneously.
    /// Implements 4x unrolling for optimal instruction throughput and cache utilization.
    /// Processes remaining elements with scalar operations for complete coverage.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_scalar_simd_avx2_optimized(&self, src: *const f32, dst: *mut f32, scalar: f32) {
        let size = self.size();
        let simd_count = size / 32; // Process 32 elements per iteration (4x unroll)
        let scalar_vec = _mm256_set1_ps(scalar);
        let mut offset = 0;

        // Unrolled SIMD loop for throughput
        for _ in 0..simd_count {
            // Process 4 AVX2 vectors (32 elements) per iteration
            let src_vec1 = _mm256_loadu_ps(src.add(offset));
            let mul_vec1 = _mm256_mul_ps(src_vec1, scalar_vec);
            _mm256_storeu_ps(dst.add(offset), mul_vec1);

            let src_vec2 = _mm256_loadu_ps(src.add(offset + 8));
            let mul_vec2 = _mm256_mul_ps(src_vec2, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 8), mul_vec2);

            let src_vec3 = _mm256_loadu_ps(src.add(offset + 16));
            let mul_vec3 = _mm256_mul_ps(src_vec3, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 16), mul_vec3);

            let src_vec4 = _mm256_loadu_ps(src.add(offset + 24));
            let mul_vec4 = _mm256_mul_ps(src_vec4, scalar_vec);
            _mm256_storeu_ps(dst.add(offset + 24), mul_vec4);

            offset += 32;
        }

        // Handle remaining 8-element blocks, then tail
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            let src_vec = _mm256_loadu_ps(src.add(offset));
            let mul_vec = _mm256_mul_ps(src_vec, scalar_vec);
            _mm256_storeu_ps(dst.add(offset), mul_vec);
            offset += 8;
        }
        while offset + 4 <= size {
            *dst.add(offset) = *src.add(offset) * scalar;
            *dst.add(offset + 1) = *src.add(offset + 1) * scalar;
            *dst.add(offset + 2) = *src.add(offset + 2) * scalar;
            *dst.add(offset + 3) = *src.add(offset + 3) * scalar;
            offset += 4;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) * scalar;
        }
    }

    /// Optimized scalar multiplication fallback
    ///
    /// Performs element-wise scalar multiplication using optimized scalar operations with
    /// 4x unrolling for better instruction-level parallelism and cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    /// * `scalar` - Scalar value to multiply by
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
    /// Uses 4x unrolled scalar operations for optimal performance on non-SIMD hardware.
    /// Processes elements in groups of 4 to improve instruction-level parallelism
    /// and reduce loop overhead.
    #[inline]
    unsafe fn mul_scalar_fallback_optimized(&self, src: *const f32, dst: *mut f32, scalar: f32) {
        let size = self.size();

        // Use unrolled loops for better instruction throughput
        let unroll_count = size / 4;
        let mut i = 0;

        // Process 4 elements at a time for better cache utilization
        while i < unroll_count {
            let idx = i * 4;
            dst.add(idx).write(src.add(idx).read() * scalar);
            dst.add(idx + 1).write(src.add(idx + 1).read() * scalar);
            dst.add(idx + 2).write(src.add(idx + 2).read() * scalar);
            dst.add(idx + 3).write(src.add(idx + 3).read() * scalar);
            i += 1;
        }

        // Handle remaining elements
        for j in (unroll_count * 4)..size {
            dst.add(j).write(src.add(j).read() * scalar);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_multiplication() {
        let a = Tensor::ones(vec![2, 3]);
        let mut b = Tensor::ones(vec![2, 3]);
        b.fill(2.0);
        let result = a.mul_tensor_optimized(&b);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are 2.0 (1.0 * 2.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 2.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_scalar_multiplication() {
        let tensor = Tensor::ones(vec![2, 2]);
        let result = tensor.mul_scalar_optimized(3.0);

        assert_eq!(result.shape().dims, vec![2, 2]);
        assert_eq!(result.size(), 4);

        // Check that all values are 3.0 (1.0 * 3.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 3.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_zero_multiplication() {
        let tensor = Tensor::ones(vec![2, 3]);
        let result = tensor.mul_scalar_optimized(0.0);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are 0.0 (1.0 * 0.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 0.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_negative_multiplication() {
        let tensor = Tensor::ones(vec![2, 3]);
        let result = tensor.mul_scalar_optimized(-2.0);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are -2.0 (1.0 * -2.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - (-2.0)).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Tensor shapes must match")]
    fn test_mismatched_shapes() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![3, 2]);
        a.mul_tensor_optimized(&b);
    }

    #[test]
    fn test_edge_cases() {
        // Test with zero tensor
        let zero_tensor = Tensor::zeros(vec![2, 3]);
        let other = Tensor::ones(vec![2, 3]);
        let result = zero_tensor.mul_tensor_optimized(&other);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are 0.0 (0.0 * 1.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 0.0).abs() < 1e-6);
            }
        }

        // Test with negative values
        let mut neg_tensor = Tensor::ones(vec![2, 3]);
        neg_tensor.fill(-1.0);
        let result = neg_tensor.mul_scalar_optimized(2.0);

        assert_eq!(result.shape().dims, vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are -2.0 (-1.0 * 2.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - (-2.0)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_large_tensor_multiplication() {
        let a = Tensor::ones(vec![100, 100]);
        let mut b = Tensor::ones(vec![100, 100]);
        b.fill(1.5);
        let result = a.mul_tensor_optimized(&b);

        assert_eq!(result.shape().dims, vec![100, 100]);
        assert_eq!(result.size(), 10000);

        // Check that all values are 1.5 (1.0 * 1.5)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 1.5).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_multiplication_with_gradtrack() {
        // Test scalar multiplication with gradtrack
        let a = Tensor::ones(vec![2, 3]).with_requires_grad();
        let mut result = a.mul_scalar(3.0);

        // Check result values: 1.0 * 3.0 = 3.0
        unsafe {
            for i in 0..result.size() {
                let val = result.as_ptr().add(i).read();
                assert!((val - 3.0).abs() < 1e-6, "Expected 3.0, got {}", val);
            }
        }

        result.backward(None);

        // Check gradient: d/dx(3x) = 3
        if let Some(grad) = a.grad_by_value() {
            unsafe {
                for i in 0..grad.size() {
                    let val = grad.as_ptr().add(i).read();
                    assert!(
                        (val - 3.0).abs() < 1e-6,
                        "Expected gradient 3.0, got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient computed for scalar multiplication!");
        }

        // Test tensor multiplication with gradtrack
        let a = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut b = Tensor::ones(vec![2, 2]);
        b.fill(3.0);
        let b = b.with_requires_grad();

        let mut result = a.mul_tensor(&b);

        // Check result values: 1.0 * 3.0 = 3.0
        unsafe {
            for i in 0..result.size() {
                let val = result.as_ptr().add(i).read();
                assert!((val - 3.0).abs() < 1e-6, "Expected 3.0, got {}", val);
            }
        }

        result.backward(None);

        // Check gradients: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
        // For a = 1.0, b = 3.0: ∂(a*b)/∂a = 3.0, ∂(a*b)/∂b = 1.0
        if let Some(grad_a) = a.grad_by_value() {
            unsafe {
                for i in 0..grad_a.size() {
                    let val = grad_a.as_ptr().add(i).read();
                    assert!(
                        (val - 3.0).abs() < 1e-6,
                        "Expected gradient A = 3.0 (∂(a*b)/∂a = b), got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient A computed for tensor multiplication!");
        }

        if let Some(grad_b) = b.grad_by_value() {
            unsafe {
                for i in 0..grad_b.size() {
                    let val = grad_b.as_ptr().add(i).read();
                    assert!(
                        (val - 1.0).abs() < 1e-6,
                        "Expected gradient B = 1.0 (∂(a*b)/∂b = a), got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient B computed for tensor multiplication!");
        }
    }

    #[test]
    fn test_mixed_mul_add_operations_with_gradtrack() {
        // Test complex computation: (a * 2) + (b * 3) - 1
        let a = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut b = Tensor::ones(vec![2, 2]);
        b.fill(3.0);
        let b = b.with_requires_grad();

        let scalar1 = 2.0;
        let scalar2 = 3.0;

        // Compute: (a * scalar1) + (b * scalar2) - 1
        let mul_a = a.mul_scalar(scalar1); // a * 2
        let mul_b = b.mul_scalar(scalar2); // b * 3
        let add_result = mul_a.add_tensor(&mul_b); // (a * 2) + (b * 3)
        let mut final_result = add_result.sub_scalar(1.0); // (a * 2) + (b * 3) - 1

        // Check result values: (1*2) + (3*3) - 1 = 2 + 9 - 1 = 10
        unsafe {
            for i in 0..final_result.size() {
                let val = final_result.as_ptr().add(i).read();
                assert!((val - 10.0).abs() < 1e-6, "Expected 10.0, got {}", val);
            }
        }

        final_result.backward(None);

        // Check gradients: d/dx((2x + 3y - 1)) = 2, d/dy((2x + 3y - 1)) = 3
        if let Some(grad_a) = a.grad_by_value() {
            unsafe {
                for i in 0..grad_a.size() {
                    let val = grad_a.as_ptr().add(i).read();
                    assert!(
                        (val - 2.0).abs() < 1e-6,
                        "Expected gradient A = 2.0, got {}",
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
                        (val - 3.0).abs() < 1e-6,
                        "Expected gradient B = 3.0, got {}",
                        val
                    );
                }
            }
        } else {
            panic!("No gradient B computed for mixed operations!");
        }
    }

    #[test]
    fn test_mul_broadcasting_gradients() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Test case: [2, 3] * [1, 3] -> [2, 3]
        // For multiplication: d/da (a * b) = b, d/db (a * b) = a
        // So grad_a = grad_output * b, grad_b = grad_output * a (then reduced)

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![1, 3])
            .unwrap()
            .with_requires_grad();

        let mut result = a.mul_tensor(&b);
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
        assert_eq!(
            grad_b.shape().dims,
            vec![1, 3],
            "grad_b should match original shape of b"
        );

        // For multiplication gradients:
        // grad_a = grad_output * b = 1.0 * [2.0, 3.0, 4.0] broadcasted to [2, 3]
        let expected_grad_a = [2.0, 3.0, 4.0, 2.0, 3.0, 4.0];
        for (i, val) in expected_grad_a.iter().enumerate().take(grad_a.size()) {
            let actual = unsafe { *grad_a.as_ptr().add(i) };
            assert!(
                (actual - val).abs() < 1e-6,
                "grad_a[{}] = {} should be {}",
                i,
                actual,
                val
            );
        }

        // grad_b = grad_output * a summed over broadcasted dimension
        // a = [1,2,3,4,5,6] -> grad_b should be [1+4, 2+5, 3+6] = [5, 7, 9]
        let expected_grad_b = [5.0, 7.0, 9.0];
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
}
