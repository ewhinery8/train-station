//! Tensor concatenation operations
//!
//! This module provides tensor concatenation functionality that joins multiple
//! tensors along a specified dimension. Concatenation is a fundamental tensor
//! transformation operation used in machine learning for combining data from
//! multiple sources, building batch operations, and creating complex tensor
//! structures.
//!
//! # Operations
//!
//! * `cat()` - Concatenate multiple tensors along a specified dimension
//!
//! # Performance Characteristics
//!
//! * **SIMD Optimized**: Uses AVX2 instructions for large block copies when available
//! * **Memory Efficient**: Minimizes temporary allocations by reusing contiguous data
//! * **Stride Aware**: Handles non-contiguous tensors efficiently with materialization
//! * **Gradient Tracking**: Full GradTrack support for automatic differentiation
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Concatenate 1D tensors
//! let a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
//! let b = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
//! let result = Tensor::cat(&[a, b], 0);
//! assert_eq!(result.shape().dims, vec![4]);
//!
//! // Concatenate 2D tensors along different dimensions
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let y = Tensor::from_slice(&[5.0, 6.0], vec![2, 1]).unwrap();
//! let result = Tensor::cat(&[x, y], 1);
//! assert_eq!(result.shape().dims, vec![2, 3]);
//! ```

use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl Tensor {
    /// Concatenate tensors along a given dimension
    ///
    /// Joins multiple tensors along the specified dimension, creating a new tensor
    /// with the combined data. All input tensors must have the same rank and
    /// matching dimensions except for the concatenation dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensors to concatenate (must not be empty)
    /// * `dim` - Dimension along which to concatenate (must be < tensor rank)
    ///
    /// # Returns
    ///
    /// A new tensor containing the concatenated data with shape where the
    /// concatenation dimension is the sum of all input tensor sizes along that dimension.
    ///
    /// # Panics
    ///
    /// * If `tensors` is empty
    /// * If `dim` is out of bounds for the tensor rank
    /// * If tensors have different ranks
    /// * If tensors have mismatched dimensions (except along concatenation dimension)
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Concatenate 1D tensors
    /// let a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
    /// let b = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
    /// let result = Tensor::cat(&[a, b], 0);
    /// assert_eq!(result.shape().dims, vec![4]);
    /// assert_eq!(result.get(&[0]), 1.0);
    /// assert_eq!(result.get(&[1]), 2.0);
    /// assert_eq!(result.get(&[2]), 3.0);
    /// assert_eq!(result.get(&[3]), 4.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Concatenate 2D tensors along dimension 1
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let b = Tensor::from_slice(&[5.0, 6.0], vec![2, 1]).unwrap();
    /// let result = Tensor::cat(&[a, b], 1);
    /// assert_eq!(result.shape().dims, vec![2, 3]);
    /// assert_eq!(result.get(&[0, 0]), 1.0);
    /// assert_eq!(result.get(&[0, 1]), 2.0);
    /// assert_eq!(result.get(&[0, 2]), 5.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Concatenate with gradient tracking
    /// let mut a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
    /// let mut b = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
    /// a.set_requires_grad(true);
    /// b.set_requires_grad(true);
    ///
    /// let result = Tensor::cat(&[a, b], 0);
    /// assert!(result.requires_grad());
    /// ```
    pub fn cat(tensors: &[Tensor], dim: usize) -> Tensor {
        assert!(!tensors.is_empty(), "cat requires at least one tensor");

        let rank = tensors[0].shape().rank();
        assert!(
            dim < rank,
            "concat dim {} out of bounds for rank {}",
            dim,
            rank
        );

        // Validate shapes and compute output dims
        let base_shape = tensors[0].shape().dims.clone();
        for t in tensors.iter() {
            assert_eq!(t.shape().rank(), rank, "All tensors must have same rank");
            for (i, (&a, &b)) in base_shape.iter().zip(t.shape().dims.iter()).enumerate() {
                if i != dim {
                    assert_eq!(
                        a, b,
                        "All dims except concat dim must match (dim {}: {} vs {})",
                        i, a, b
                    );
                }
            }
        }

        let mut out_dims = base_shape.clone();
        let mut concat_len = 0usize;
        for t in tensors.iter() {
            concat_len += t.shape().dims[dim];
        }
        out_dims[dim] = concat_len;

        let mut output = Tensor::new(out_dims.clone());

        // Calculate block sizes for contiguous copy
        let inner: usize = out_dims[dim + 1..].iter().product();
        let outer: usize = out_dims[..dim].iter().product();

        // Prepare source buffers once to avoid per-iteration cloning/copying
        // Each entry holds a pointer to contiguous data and the length along `dim`
        struct SourceInfo {
            base_ptr: *const f32,
            len_along_dim: usize,
        }

        let mut temp_contiguous: Vec<Tensor> = Vec::new();
        let mut sources: Vec<SourceInfo> = Vec::with_capacity(tensors.len());
        for t in tensors.iter() {
            let len_d = t.shape().dims[dim];
            if len_d == 0 {
                // Skip empty tensors; keep alignment in running count during copy
                sources.push(SourceInfo {
                    base_ptr: std::ptr::null(),
                    len_along_dim: 0,
                });
                continue;
            }
            if t.is_contiguous() {
                let base_ptr = unsafe { t.as_ptr() };
                sources.push(SourceInfo {
                    base_ptr,
                    len_along_dim: len_d,
                });
            } else {
                // Materialize once and keep it alive in `temp_contiguous`
                let cont = t.contiguous();
                let base_ptr = unsafe { cont.as_ptr() };
                temp_contiguous.push(cont);
                sources.push(SourceInfo {
                    base_ptr,
                    len_along_dim: len_d,
                });
            }
        }

        unsafe {
            let dst_ptr = output.as_mut_ptr();
            for outer_idx in 0..outer {
                let mut running = 0usize;
                for src in &sources {
                    let len_d = src.len_along_dim;
                    if len_d == 0 {
                        continue;
                    }
                    let copy_elems = len_d * inner;

                    // Source base offset for this outer index
                    let src_base = outer_idx * (len_d * inner);
                    let src_ptr = src.base_ptr.add(src_base);

                    // Destination base offset
                    let dst_base = outer_idx * (concat_len * inner) + running * inner;
                    let dst_cur = dst_ptr.add(dst_base);

                    optimized_block_copy(src_ptr, dst_cur, copy_elems);
                    running += len_d;
                }
            }
        }

        // GradTrack setup if any input requires_grad
        let any_requires = tensors.iter().any(|t| t.requires_grad());
        if any_requires {
            output.set_requires_grad(true);
            let mut input_ids = Vec::with_capacity(tensors.len());
            let mut grad_input_sizes = Vec::new();
            let mut grad_input_shapes = Vec::new();
            for t in tensors.iter() {
                if t.requires_grad() {
                    input_ids.push(t.id());
                    grad_input_sizes.push(t.shape().dims[dim]);
                    grad_input_shapes.push(t.shape().dims.clone());
                }
            }
            let grad_fn = GradFn::Cat {
                dim,
                input_sizes: grad_input_sizes,
                input_shapes: grad_input_shapes,
            };
            output.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(output.id(), input_ids, grad_fn);
        }

        output
    }
}

/// Optimized block copy with SIMD acceleration for large blocks
///
/// Performs efficient memory copying with automatic SIMD optimization when
/// available. Uses AVX2 instructions for large blocks and falls back to
/// unrolled scalar operations for smaller blocks or when SIMD is not available.
///
/// # Arguments
///
/// * `src` - Source pointer to copy from
/// * `dst` - Destination pointer to copy to
/// * `count` - Number of f32 elements to copy
///
/// # Safety
///
/// The caller must ensure:
/// * `src` points to valid memory with at least `count` f32 elements
/// * `dst` points to valid writable memory with at least `count` f32 elements
/// * The source and destination regions do not overlap
/// * The pointers are properly aligned for the target architecture
///
/// # Performance
///
/// * **Large blocks (≥64 elements)**: Uses AVX2 SIMD instructions when available
/// * **Medium blocks (32-63 elements)**: Uses unrolled scalar operations
/// * **Small blocks (<32 elements)**: Uses standard library copy
#[inline]
unsafe fn optimized_block_copy(src: *const f32, dst: *mut f32, count: usize) {
    if count == 0 {
        return;
    }

    // For small blocks, use standard copy
    if count <= 32 {
        std::ptr::copy_nonoverlapping(src, dst, count);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && count >= 64 {
            simd_block_copy_avx2(src, dst, count);
            return;
        }
    }

    // Fallback to optimized scalar copy with unrolling
    scalar_block_copy_unrolled(src, dst, count);
}

/// SIMD-optimized block copy using AVX2 instructions
///
/// Performs high-performance memory copying using AVX2 vector instructions.
/// Processes 32 elements per iteration using 4 AVX2 vectors, with additional
/// optimizations for remaining elements.
///
/// # Arguments
///
/// * `src` - Source pointer to copy from
/// * `dst` - Destination pointer to copy to
/// * `count` - Number of f32 elements to copy
///
/// # Safety
///
/// The caller must ensure:
/// * AVX2 instructions are available on the target CPU
/// * `src` points to valid memory with at least `count` f32 elements
/// * `dst` points to valid writable memory with at least `count` f32 elements
/// * The source and destination regions do not overlap
/// * Pointers are properly aligned for AVX2 operations
///
/// # Performance
///
/// * **Main loop**: Processes 32 elements per iteration (4 AVX2 vectors)
/// * **Remaining blocks**: Processes 8 elements per iteration for partial blocks
/// * **Final elements**: Uses standard copy for remaining elements
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn simd_block_copy_avx2(src: *const f32, dst: *mut f32, count: usize) {
    let simd_count = count / 32; // Process 32 elements per iteration (4x AVX2 vectors)
    let mut offset = 0;

    // Unrolled SIMD loop for maximum throughput
    for _ in 0..simd_count {
        // Process 4 AVX2 vectors (32 elements) per iteration
        let vec1 = _mm256_loadu_ps(src.add(offset));
        let vec2 = _mm256_loadu_ps(src.add(offset + 8));
        let vec3 = _mm256_loadu_ps(src.add(offset + 16));
        let vec4 = _mm256_loadu_ps(src.add(offset + 24));

        _mm256_storeu_ps(dst.add(offset), vec1);
        _mm256_storeu_ps(dst.add(offset + 8), vec2);
        _mm256_storeu_ps(dst.add(offset + 16), vec3);
        _mm256_storeu_ps(dst.add(offset + 24), vec4);

        offset += 32;
    }

    // Handle remaining elements with 8-element SIMD blocks
    let remaining_full_blocks = (count - offset) / 8;
    for _ in 0..remaining_full_blocks {
        let vec = _mm256_loadu_ps(src.add(offset));
        _mm256_storeu_ps(dst.add(offset), vec);
        offset += 8;
    }

    // Handle final elements
    if offset < count {
        std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), count - offset);
    }
}

/// Unrolled scalar block copy for optimal performance
///
/// Performs memory copying using unrolled scalar operations for better
/// instruction-level parallelism and reduced loop overhead. Processes
/// 8 elements per iteration in the main loop.
///
/// # Arguments
///
/// * `src` - Source pointer to copy from
/// * `dst` - Destination pointer to copy to
/// * `count` - Number of f32 elements to copy
///
/// # Safety
///
/// The caller must ensure:
/// * `src` points to valid memory with at least `count` f32 elements
/// * `dst` points to valid writable memory with at least `count` f32 elements
/// * The source and destination regions do not overlap
///
/// # Performance
///
/// * **Main loop**: Processes 8 elements per iteration with manual unrolling
/// * **Remaining elements**: Uses standard library copy for final elements
/// * **Optimization**: Reduces loop overhead and improves instruction pipelining
#[inline]
unsafe fn scalar_block_copy_unrolled(src: *const f32, dst: *mut f32, count: usize) {
    let unroll_factor = 8;
    let unroll_count = count / unroll_factor;
    let mut offset = 0;

    // Unrolled scalar copy for better performance
    for _ in 0..unroll_count {
        *dst.add(offset) = *src.add(offset);
        *dst.add(offset + 1) = *src.add(offset + 1);
        *dst.add(offset + 2) = *src.add(offset + 2);
        *dst.add(offset + 3) = *src.add(offset + 3);
        *dst.add(offset + 4) = *src.add(offset + 4);
        *dst.add(offset + 5) = *src.add(offset + 5);
        *dst.add(offset + 6) = *src.add(offset + 6);
        *dst.add(offset + 7) = *src.add(offset + 7);
        offset += unroll_factor;
    }

    // Handle remaining elements
    if offset < count {
        std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), count - offset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cat_1d() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[3.0], vec![1]).unwrap();
        let y = Tensor::cat(&[a, b], 0);
        assert_eq!(y.shape().dims, vec![3]);
        assert_eq!(y.get(&[0]), 1.0);
        assert_eq!(y.get(&[2]), 3.0);
    }

    #[test]
    fn test_cat_2d_dim1() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0, 6.0], vec![2, 1]).unwrap();
        let y = Tensor::cat(&[a, b], 1);
        assert_eq!(y.shape().dims, vec![2, 3]);
        assert_eq!(y.get(&[0, 2]), 5.0);
        assert_eq!(y.get(&[1, 2]), 6.0);
    }

    #[test]
    #[should_panic]
    fn test_cat_mismatch() {
        let a = Tensor::new(vec![2, 2]);
        let b = Tensor::new(vec![3, 1]);
        let _ = Tensor::cat(&[a, b], 1);
    }
}
