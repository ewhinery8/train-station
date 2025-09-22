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
//! assert_eq!(contiguous.shape().dims(), vec![2, 2]);
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
use crate::tensor::iterator::collect::optimized_copy;
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
    /// assert_eq!(contiguous.shape().dims(), vec![2, 2]);
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
    #[track_caller]
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            let mut cloned = self.clone();
            // Ensure gradient requirements are preserved
            if self.requires_grad() {
                cloned.set_requires_grad(true);
                // Register gradient function even for already-contiguous tensors
                let grad_fn = GradFn::Contiguous {
                    input_shape: self.shape().dims().to_vec(),
                };
                cloned.set_grad_fn(grad_fn.clone());
                GradEngine::register_operation(cloned.id(), vec![self.id()], grad_fn);
            }
            return cloned;
        }

        // Create new contiguous tensor and copy via optimized methods
        let mut result = Tensor::new(self.shape().dims().to_vec());

        unsafe {
            self.copy_to_contiguous_optimized(&mut result);
        }

        // Preserve gradient requirements and register gradient function
        if self.requires_grad() {
            result.set_requires_grad(true);
            let grad_fn = GradFn::Contiguous {
                input_shape: self.shape().dims().to_vec(),
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

        if size == 0 {
            return;
        }

        // Fast path: if the last dimension is contiguous in the source view,
        // copy row-by-row using SIMD-optimized contiguous copies.
        if rank >= 1 && self.stride(rank - 1) == 1 {
            let dims = self.shape().dims();
            let row_len = dims[rank - 1];

            // Number of outer rows to copy (all dims except the last)
            let outer: usize = if rank == 1 {
                1
            } else {
                dims[..rank - 1].iter().product()
            };

            let src_base = self.as_ptr();
            let dst_base = result.as_mut_ptr();
            let strides = self.strides();

            // Coordinate vector for outer dimensions (exclude last)
            let mut coords = vec![0usize; rank];

            for outer_idx in 0..outer {
                // Compute multi-index over dims[0..rank-1) in row-major order
                if rank > 1 {
                    let mut tmp = outer_idx;
                    for i in (0..rank - 1).rev() {
                        let d = dims[i];
                        coords[i] = if d == 0 { 0 } else { tmp % d };
                        if d != 0 {
                            tmp /= d;
                        }
                    }
                }
                coords[rank - 1] = 0; // start of the contiguous row

                // Compute source offset via strides
                let mut src_off = 0usize;
                for i in 0..rank {
                    src_off += coords[i] * strides[i];
                }

                // Destination offset: linear index over outer dims times row_len
                let mut dst_row_index = 0usize;
                if rank > 1 {
                    for i in 0..rank - 1 {
                        dst_row_index = dst_row_index * dims[i] + coords[i];
                    }
                }
                let dst_off = dst_row_index * row_len;

                // Copy the entire row as a contiguous block
                optimized_copy(src_base.add(src_off), dst_base.add(dst_off), row_len);
            }
            return;
        }

        // Fallback: general coordinate-based copy (works for any strided view)
        self.copy_to_contiguous_simple(result, rank);
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
                let dim_size = self.shape().dims()[i];
                coords[i] = tmp % dim_size;
                tmp /= dim_size;
            }
            let src_off = self.shape().offset(&coords);
            *dst_ptr.add(dst_idx) = *src_ptr.add(src_off);
        }
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
        assert_eq!(contiguous.shape().dims(), tensor.shape().dims());

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
        assert_eq!(contiguous.shape().dims(), tensor.shape().dims());
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
        let grad = x.grad_owned().expect("Gradient should exist");
        assert_eq!(grad.shape().dims(), vec![2, 2]);

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
        assert_eq!(contiguous.shape().dims(), vec![3]);

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
        assert_eq!(contiguous.shape().dims(), vec![2, 3, 4]);
        assert_eq!(contiguous.size(), 24);
    }
}
