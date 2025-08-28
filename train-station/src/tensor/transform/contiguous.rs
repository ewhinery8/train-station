//! Contiguous tensor transformation operation
//!
//! This module provides functionality to create contiguous copies of tensors,
//! ensuring that tensor data is stored in a linear, cache-friendly memory layout.
//! Contiguous tensors are essential for optimal performance in many operations,
//! particularly SIMD-optimized computations and operations that require
//! sequential memory access patterns.
//!
//! # Memory Layout
//!
//! A tensor is considered contiguous when its elements are stored in memory
//! in row-major order without gaps. Non-contiguous tensors can arise from
//! operations like transpose, permute, or slice views that change the
//! memory layout without copying data.
//!
//! # Performance Characteristics
//!
//! - **Already Contiguous**: O(1) time, returns a clone
//! - **Small Tensors (≤64 elements)**: Simple copy with coordinate conversion
//! - **Medium Tensors (65-1023 elements)**: Unrolled copy for better performance
//! - **Large Tensors (≥1024 elements)**: Blocked copy with cache optimization
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create a contiguous tensor
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! assert!(tensor.is_contiguous());
//!
//! // Create a non-contiguous tensor through transpose
//! let transposed = tensor.transpose(0, 1);
//! assert!(!transposed.is_contiguous());
//!
//! // Make it contiguous again
//! let contiguous = transposed.contiguous();
//! assert!(contiguous.is_contiguous());
//! assert_eq!(contiguous.shape().dims, vec![2, 2]);
//! ```
//!
//! ```
//! use train_station::Tensor;
//!
//! // Contiguous preserves gradient tracking
//! let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! tensor.set_requires_grad(true);
//!
//! let transposed = tensor.transpose(0, 1);
//! let contiguous = transposed.contiguous();
//! assert!(contiguous.requires_grad());
//! ```
//!
//! # Gradient Tracking
//!
//! The contiguous operation supports automatic gradient tracking through
//! the GradTrack system. When `requires_grad` is enabled, the operation
//! registers a gradient function that ensures proper gradient flow during
//! backward passes.

use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::Tensor;

impl Tensor {
    /// Creates a contiguous copy of the tensor
    ///
    /// This operation ensures that the tensor data is stored in a linear,
    /// cache-friendly memory layout. If the tensor is already contiguous,
    /// this operation returns a clone. For non-contiguous tensors, it
    /// creates a new tensor with the same data but in contiguous memory layout.
    ///
    /// The operation uses different optimization strategies based on tensor size:
    /// - Small tensors (≤64 elements): Simple coordinate-based copy
    /// - Medium tensors (65-1023 elements): Unrolled copy for better performance
    /// - Large tensors (≥1024 elements): Blocked copy with cache optimization
    ///
    /// # Returns
    ///
    /// A new tensor with contiguous memory layout containing the same data
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Already contiguous tensor
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let contiguous = tensor.contiguous();
    /// assert!(contiguous.is_contiguous());
    /// assert_eq!(contiguous.shape().dims, vec![2, 2]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Non-contiguous tensor from transpose
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let transposed = tensor.transpose(0, 1);
    /// assert!(!transposed.is_contiguous());
    ///
    /// let contiguous = transposed.contiguous();
    /// assert!(contiguous.is_contiguous());
    /// assert_eq!(contiguous.get(&[0, 0]), 1.0);
    /// assert_eq!(contiguous.get(&[0, 1]), 3.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Preserves gradient tracking
    /// let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// tensor.set_requires_grad(true);
    ///
    /// let contiguous = tensor.contiguous();
    /// assert!(contiguous.requires_grad());
    /// ```
    ///
    /// # Performance
    ///
    /// - **Already contiguous**: O(1) time complexity, returns a clone
    /// - **Non-contiguous**: O(n) time complexity with size-dependent optimizations
    /// - **Memory usage**: Creates a new tensor with the same size as the original
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            let mut cloned = self.clone();
            // Ensure gradient requirements are preserved
            if self.requires_grad() {
                cloned.set_requires_grad(true);
                // Register gradient function even for already-contiguous tensors
                let grad_fn = GradFn::Contiguous {
                    input_shape: self.shape().dims.clone(),
                };
                cloned.set_grad_fn(grad_fn.clone());
                GradEngine::register_operation(cloned.id(), vec![self.id()], grad_fn);
            }
            return cloned;
        }

        // Create new contiguous tensor and copy via optimized methods
        let mut result = Tensor::new(self.shape().dims.clone());

        unsafe {
            self.copy_to_contiguous_optimized(&mut result);
        }

        // Preserve gradient requirements and register gradient function
        if self.requires_grad() {
            result.set_requires_grad(true);
            let grad_fn = GradFn::Contiguous {
                input_shape: self.shape().dims.clone(),
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
        }

        result
    }

    /// Internal optimized contiguous copy operation
    ///
    /// This function dispatches to the appropriate copy strategy based on
    /// tensor size and rank for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `result` - The destination tensor to copy data into
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// * `result` has the same shape as `self`
    /// * `result` is properly allocated and initialized
    /// * Both tensors are valid and not moved during the operation
    #[inline]
    unsafe fn copy_to_contiguous_optimized(&self, result: &mut Tensor) {
        let size = self.size();
        let rank = self.shape().rank();
        let _src_ptr = self.as_ptr();
        let _dst_ptr = result.as_mut_ptr();

        // For simple 1D tensors or very small tensors, use simple copy
        if rank <= 1 || size <= 64 {
            self.copy_to_contiguous_simple(result, rank);
            return;
        }

        // For larger multi-dimensional tensors, use optimized stride-aware copy
        if size >= 1024 {
            self.copy_to_contiguous_large(result, rank);
        } else {
            self.copy_to_contiguous_medium(result, rank);
        }
    }

    /// Simple copy for small tensors or 1D tensors
    ///
    /// This function performs a straightforward coordinate-based copy
    /// suitable for small tensors where the overhead of more complex
    /// optimizations would not be beneficial.
    ///
    /// # Arguments
    ///
    /// * `result` - The destination tensor
    /// * `rank` - The rank of the tensor
    ///
    /// # Safety
    ///
    /// The caller must ensure both tensors are valid and properly allocated.
    #[inline]
    unsafe fn copy_to_contiguous_simple(&self, result: &mut Tensor, rank: usize) {
        let size = self.size();
        let src_ptr = self.as_ptr();
        let dst_ptr = result.as_mut_ptr();

        for dst_idx in 0..size {
            // Compute destination coordinates under contiguous strides
            let mut coords = vec![0usize; rank];
            let mut tmp = dst_idx;
            for i in (0..rank).rev() {
                let dim_size = self.shape().dims[i];
                coords[i] = tmp % dim_size;
                tmp /= dim_size;
            }
            let src_off = self.shape().offset(&coords);
            *dst_ptr.add(dst_idx) = *src_ptr.add(src_off);
        }
    }

    /// Optimized copy for medium-sized tensors with unrolling
    ///
    /// This function uses loop unrolling to improve performance for
    /// medium-sized tensors by reducing loop overhead and improving
    /// instruction-level parallelism.
    ///
    /// # Arguments
    ///
    /// * `result` - The destination tensor
    /// * `rank` - The rank of the tensor
    ///
    /// # Safety
    ///
    /// The caller must ensure both tensors are valid and properly allocated.
    #[inline]
    unsafe fn copy_to_contiguous_medium(&self, result: &mut Tensor, rank: usize) {
        let size = self.size();
        let src_ptr = self.as_ptr();
        let dst_ptr = result.as_mut_ptr();
        let unroll_factor = 4;
        let unroll_count = size / unroll_factor;
        let mut dst_idx = 0;

        // Unrolled loop for better performance
        for _ in 0..unroll_count {
            for unroll_i in 0..unroll_factor {
                let coords = self.linear_to_coords(dst_idx + unroll_i, rank);
                let src_off = self.shape().offset(&coords);
                *dst_ptr.add(dst_idx + unroll_i) = *src_ptr.add(src_off);
            }
            dst_idx += unroll_factor;
        }

        // Handle remaining elements
        for i in dst_idx..size {
            let coords = self.linear_to_coords(i, rank);
            let src_off = self.shape().offset(&coords);
            *dst_ptr.add(i) = *src_ptr.add(src_off);
        }
    }

    /// Cache-optimized copy for large tensors with blocking
    ///
    /// This function uses blocking to improve cache locality for large tensors.
    /// It processes the tensor in blocks to maximize cache hit rates and
    /// combines blocking with loop unrolling for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `result` - The destination tensor
    /// * `rank` - The rank of the tensor
    ///
    /// # Safety
    ///
    /// The caller must ensure both tensors are valid and properly allocated.
    #[inline]
    unsafe fn copy_to_contiguous_large(&self, result: &mut Tensor, rank: usize) {
        let size = self.size();
        let src_ptr = self.as_ptr();
        let dst_ptr = result.as_mut_ptr();

        // Use blocking to improve cache locality
        let block_size = 1024; // Process 1024 elements per block
        let num_blocks = (size + block_size - 1) / block_size;

        for block in 0..num_blocks {
            let start_idx = block * block_size;
            let end_idx = (start_idx + block_size).min(size);
            let block_len = end_idx - start_idx;

            // Process block with unrolling
            let unroll_factor = 4;
            let unroll_count = block_len / unroll_factor;
            let mut local_idx = 0;

            for _ in 0..unroll_count {
                for unroll_i in 0..unroll_factor {
                    let dst_idx = start_idx + local_idx + unroll_i;
                    let coords = self.linear_to_coords(dst_idx, rank);
                    let src_off = self.shape().offset(&coords);
                    *dst_ptr.add(dst_idx) = *src_ptr.add(src_off);
                }
                local_idx += unroll_factor;
            }

            // Handle remaining elements in this block
            for i in local_idx..block_len {
                let dst_idx = start_idx + i;
                let coords = self.linear_to_coords(dst_idx, rank);
                let src_off = self.shape().offset(&coords);
                *dst_ptr.add(dst_idx) = *src_ptr.add(src_off);
            }
        }
    }

    /// Helper function to convert linear index to coordinates
    ///
    /// Converts a linear (flat) index into multi-dimensional coordinates
    /// based on the tensor's shape. This is used for coordinate-based
    /// memory access in non-contiguous tensors.
    ///
    /// # Arguments
    ///
    /// * `idx` - The linear index to convert
    /// * `rank` - The rank of the tensor
    ///
    /// # Returns
    ///
    /// A vector of coordinates representing the multi-dimensional position
    #[inline]
    fn linear_to_coords(&self, mut idx: usize, rank: usize) -> Vec<usize> {
        let mut coords = vec![0usize; rank];
        for i in (0..rank).rev() {
            let dim_size = self.shape().dims[i];
            coords[i] = idx % dim_size;
            idx /= dim_size;
        }
        coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_copy() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // Test that contiguous() returns a proper copy
        let contiguous = tensor.contiguous();
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.shape().dims, tensor.shape().dims);

        // Verify data is preserved
        assert_eq!(contiguous.get(&[0, 0]), 1.0);
        assert_eq!(contiguous.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_contiguous_already_contiguous() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // For already contiguous tensors, should return a clone
        let contiguous = tensor.contiguous();
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.shape().dims, tensor.shape().dims);
        assert_eq!(contiguous.size(), tensor.size());
    }

    #[test]
    fn test_contiguous_preserves_gradient_tracking() {
        let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        tensor.set_requires_grad(true);

        let contiguous = tensor.contiguous();
        assert!(contiguous.requires_grad());
    }

    #[test]
    fn test_contiguous_gradient_flow() {
        // Test that gradients flow correctly through contiguous operation
        let mut x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        x.set_requires_grad(true);

        // Create a non-contiguous tensor through transpose
        let x_transposed = x.transpose(0, 1);
        assert!(!x_transposed.is_contiguous());

        // Make it contiguous
        let x_contiguous = x_transposed.contiguous();
        assert!(x_contiguous.is_contiguous());
        assert!(x_contiguous.requires_grad());

        // Do a simple operation and backward
        let mut result = x_contiguous.sum();
        result.backward(None);

        // Check that the original tensor received gradients
        let grad = x.grad_by_value().expect("Gradient should exist");
        assert_eq!(grad.shape().dims, vec![2, 2]);

        // All gradients should be 1.0 since sum operation
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(grad.get(&[i, j]), 1.0);
            }
        }
    }

    #[test]
    fn test_contiguous_1d() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let contiguous = tensor.contiguous();

        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.shape().dims, vec![3]);

        // Verify data preservation
        for i in 0..3 {
            assert_eq!(contiguous.get(&[i]), (i + 1) as f32);
        }
    }

    #[test]
    fn test_contiguous_3d() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();
        let contiguous = tensor.contiguous();

        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.shape().dims, vec![2, 3, 4]);
        assert_eq!(contiguous.size(), 24);
    }
}
