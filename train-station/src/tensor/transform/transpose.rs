//! Tensor transpose operations
//!
//! This module provides tensor transpose functionality that swaps dimensions
//! of tensors, effectively changing the memory access pattern and logical
//! arrangement of data. Transposition is a fundamental tensor transformation
//! operation used in machine learning for matrix operations, preparing data
//! for specific layer types, and implementing complex tensor manipulations.
//!
//! # Operations
//!
//! * `transpose()` - Swap two specified dimensions of a tensor
//! * `t()` - Matrix transpose (swap last two dimensions)
//!
//! # Performance Characteristics
//!
//! * **Zero-Copy Operations**: Returns a view when possible using stride manipulation
//! * **Memory Efficient**: Reuses existing tensor data through view operations
//! * **Cache Optimized**: Uses optimized copying when view operations are not possible
//! * **Gradient Tracking**: Full GradTrack support for automatic differentiation
//! * **Shape Transformation**: Changes dimension order while preserving total elements
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Basic 2D transpose
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//! let transposed = tensor.transpose(0, 1);
//! assert_eq!(transposed.shape().dims, vec![3, 2]);
//! ```
//!
//! ```
//! use train_station::Tensor;
//!
//! // Matrix transpose convenience method
//! let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let transposed = matrix.t();
//! assert_eq!(transposed.shape().dims, vec![2, 2]);
//! ```
//!
//! # Gradient Tracking
//!
//! The transpose operations support automatic gradient tracking through
//! the GradTrack system. When `requires_grad` is enabled, the operations
//! register gradient functions that apply the inverse transpose during
//! backward passes.

use crate::tensor::core::Tensor;

impl Tensor {
    /// Transpose two dimensions of the tensor
    ///
    /// Swaps two specified dimensions of the tensor, modifying the shape
    /// and memory access pattern. When possible, this operation returns
    /// a zero-copy view using stride manipulation. For complex cases or
    /// non-contiguous tensors, data is copied to ensure correct transposition.
    ///
    /// The transpose operation is its own inverse - applying transpose
    /// twice with the same dimensions returns the original tensor.
    ///
    /// # Arguments
    ///
    /// * `dim0` - First dimension to swap (must be < tensor rank)
    /// * `dim1` - Second dimension to swap (must be < tensor rank)
    ///
    /// # Returns
    ///
    /// A new tensor with the specified dimensions transposed. The total
    /// number of elements remains unchanged.
    ///
    /// # Panics
    ///
    /// * If `dim0` is out of bounds for the tensor rank
    /// * If `dim1` is out of bounds for the tensor rank
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Basic 2D transpose
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let transposed = tensor.transpose(0, 1);
    /// assert_eq!(transposed.shape().dims, vec![3, 2]);
    /// assert_eq!(transposed.get(&[0, 0]), 1.0);
    /// assert_eq!(transposed.get(&[0, 1]), 4.0);
    /// assert_eq!(transposed.get(&[1, 0]), 2.0);
    /// assert_eq!(transposed.get(&[1, 1]), 5.0);
    /// assert_eq!(transposed.get(&[2, 0]), 3.0);
    /// assert_eq!(transposed.get(&[2, 1]), 6.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // 3D tensor transpose
    /// let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    /// let tensor = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();
    /// let transposed = tensor.transpose(0, 1);
    /// assert_eq!(transposed.shape().dims, vec![3, 2, 4]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Transpose with gradient tracking
    /// let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// tensor.set_requires_grad(true);
    ///
    /// let transposed = tensor.transpose(0, 1);
    /// assert!(transposed.requires_grad());
    /// assert_eq!(transposed.shape().dims, vec![2, 2]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Transpose same dimension (no change)
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let result = tensor.transpose(1, 1);
    /// assert_eq!(result.shape().dims, tensor.shape().dims);
    /// assert_eq!(result.get(&[0, 0]), tensor.get(&[0, 0]));
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Transpose is its own inverse
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let transposed = tensor.transpose(0, 1);
    /// let double_transposed = transposed.transpose(0, 1);
    /// assert_eq!(double_transposed.shape().dims, tensor.shape().dims);
    /// assert_eq!(double_transposed.get(&[0, 0]), tensor.get(&[0, 0]));
    /// ```
    ///
    /// # Performance
    ///
    /// - **Contiguous tensors**: O(1) time complexity, returns a view
    /// - **Non-contiguous tensors**: O(n) time complexity with data copying
    /// - **Memory usage**: No additional allocation for view operations
    /// - **Gradient tracking**: Preserves gradient requirements and tracking
    ///
    /// # Relationship to Other Operations
    ///
    /// This operation is related to other tensor transformations:
    /// - `t()` - Convenience method for matrix transpose (last two dimensions)
    /// - `permute()` - More general dimension reordering operation
    /// - `reshape()` - Changes shape without changing dimension order
    ///
    /// # Memory Layout
    ///
    /// For contiguous tensors, transpose returns a view with modified strides,
    /// making the tensor non-contiguous. For non-contiguous tensors or complex
    /// cases, data is copied to ensure correct transposition.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        assert!(
            dim0 < self.shape().rank(),
            "dim0 {} out of bounds for tensor with rank {}",
            dim0,
            self.shape().rank()
        );
        assert!(
            dim1 < self.shape().rank(),
            "dim1 {} out of bounds for tensor with rank {}",
            dim1,
            self.shape().rank()
        );

        // If same dimension, return a clone
        if dim0 == dim1 {
            return self.clone();
        }

        // Create new dimensions and strides by swapping
        let mut new_dims = self.shape().dims.clone();
        let mut new_strides = self.strides().to_vec();

        new_dims.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        // Create a view-based transpose when possible (creates non-contiguous tensor)
        let mut result = if self.is_contiguous() && self.can_transpose_as_view(dim0, dim1) {
            let new_shape = crate::tensor::Shape::as_view(new_dims, new_strides);
            self.create_view_with_shape(new_shape)
        } else {
            // Fallback to copy for complex cases
            self.transpose_with_copy(new_dims, new_strides, dim0, dim1)
        };

        // GradTrack: register transpose for backward (transpose is its own inverse)
        if self.requires_grad() {
            result.set_requires_grad(true);
            let grad_fn = crate::gradtrack::grad_fn::GradFn::Transpose {
                dim0,
                dim1,
                input_shape: self.shape().dims.clone(),
            };
            result.set_grad_fn(grad_fn.clone());
            crate::gradtrack::engine::GradEngine::register_operation(
                result.id(),
                vec![self.id()],
                grad_fn,
            );
        }

        result
    }

    /// Matrix transpose (transpose last two dimensions)
    ///
    /// Convenience method for the common case of matrix transposition.
    /// For 2D tensors, this performs a standard matrix transpose.
    /// For higher-dimensional tensors, this transposes the last two
    /// dimensions, treating the tensor as a batch of matrices.
    ///
    /// This method is equivalent to `transpose(rank-2, rank-1)` where
    /// `rank` is the number of dimensions in the tensor.
    ///
    /// # Returns
    ///
    /// A new tensor with the last two dimensions transposed
    ///
    /// # Panics
    ///
    /// * If the tensor has less than 2 dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // 2D matrix transpose
    /// let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let transposed = matrix.t();
    /// assert_eq!(transposed.shape().dims, vec![2, 2]);
    /// assert_eq!(transposed.get(&[0, 0]), 1.0);
    /// assert_eq!(transposed.get(&[0, 1]), 3.0);
    /// assert_eq!(transposed.get(&[1, 0]), 2.0);
    /// assert_eq!(transposed.get(&[1, 1]), 4.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // 3D tensor (batch of matrices)
    /// let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    /// let tensor = Tensor::from_slice(&data, vec![2, 2, 3]).unwrap();
    /// let transposed = tensor.t();
    /// assert_eq!(transposed.shape().dims, vec![2, 3, 2]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Matrix transpose with gradient tracking
    /// let mut matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// matrix.set_requires_grad(true);
    ///
    /// let transposed = matrix.t();
    /// assert!(transposed.requires_grad());
    /// assert_eq!(transposed.shape().dims, vec![2, 2]);
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: Same as `transpose()` - O(1) for views, O(n) for copies
    /// - **Memory Usage**: Same as `transpose()` - no allocation for views
    /// - **Gradient Tracking**: Preserves gradient requirements and tracking
    ///
    /// # Relationship to Other Operations
    ///
    /// This operation is equivalent to:
    /// ```rust
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 3, 4]);
    /// let rank = tensor.shape().rank();
    /// let transposed1 = tensor.t();
    /// let transposed2 = tensor.transpose(rank - 2, rank - 1);
    /// // transposed1 and transposed2 are identical
    /// ```
    pub fn t(&self) -> Tensor {
        assert!(
            self.shape().rank() >= 2,
            "Matrix transpose requires at least 2 dimensions, got {}",
            self.shape().rank()
        );
        let rank = self.shape().rank();
        self.transpose(rank - 2, rank - 1)
    }

    /// Check if transpose can be done as a zero-copy view operation
    ///
    /// Determines whether the transpose operation can be performed as a
    /// zero-copy view by manipulating strides rather than copying data.
    /// This is possible for contiguous tensors when swapping different dimensions.
    ///
    /// # Arguments
    ///
    /// * `dim0` - First dimension to swap
    /// * `dim1` - Second dimension to swap
    ///
    /// # Returns
    ///
    /// `true` if the transpose can be done as a view (zero-copy), `false` otherwise
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Simple boolean checks
    /// - **Memory Usage**: No allocation
    ///
    /// # Examples
    ///
    /// This method is used internally by the `transpose()` function to
    /// determine the optimal implementation strategy (view vs copy).
    fn can_transpose_as_view(&self, dim0: usize, dim1: usize) -> bool {
        // For contiguous tensors, we can always create a view with different strides
        // This is safe because we're not modifying the underlying data, just the access pattern
        self.is_contiguous() && (dim0 != dim1)
    }

    /// Transpose with data copying when view operation is not possible
    ///
    /// Performs transpose by copying data to a new tensor when a view-based
    /// transpose is not possible or optimal. This method ensures correct
    /// transposition for all tensor types and memory layouts.
    ///
    /// # Arguments
    ///
    /// * `new_dims` - The new dimensions after transposition
    /// * `_new_strides` - The new strides after transposition (unused in copy implementation)
    /// * `dim0` - First dimension that was swapped
    /// * `dim1` - Second dimension that was swapped
    ///
    /// # Returns
    ///
    /// A new tensor with copied and transposed data
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(n) where n is the number of elements
    /// - **Memory Usage**: Allocates new tensor with same total size
    /// - **Data Integrity**: Ensures correct transposition for all cases
    ///
    /// # Examples
    ///
    /// This method is called internally by `transpose()` when view-based
    /// transposition is not possible, such as for non-contiguous tensors
    /// or complex memory layouts.
    fn transpose_with_copy(
        &self,
        new_dims: Vec<usize>,
        _new_strides: Vec<usize>,
        dim0: usize,
        dim1: usize,
    ) -> Tensor {
        let mut result = Tensor::new(new_dims.clone());

        // Use stride-aware copying that correctly handles arbitrary dimension swaps
        unsafe {
            self.transpose_copy_stride_aware(&mut result, dim0, dim1);
        }

        // Preserve gradient tracking requirement
        if self.requires_grad() {
            result.set_requires_grad(true);
        }

        result
    }

    /// Stride-aware transpose copy that correctly handles arbitrary dimension swaps
    ///
    /// Performs efficient transpose copying using coordinate mapping and
    /// stride calculations. This method correctly handles transposition
    /// of any two dimensions in tensors of arbitrary rank and shape.
    ///
    /// # Arguments
    ///
    /// * `result` - Output tensor to write transposed data
    /// * `dim0` - First dimension that was swapped
    /// * `dim1` - Second dimension that was swapped
    ///
    /// # Safety
    ///
    /// This function uses unsafe pointer arithmetic for performance.
    /// The caller must ensure:
    /// * `result` tensor has the correct size and shape
    /// * `result` tensor is properly allocated and accessible
    /// * `dim0` and `dim1` are valid dimension indices
    /// * Source tensor data is valid and accessible
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(n) where n is the number of elements
    /// - **Memory Access**: Optimized for cache-friendly access patterns
    /// - **Coordinate Mapping**: Efficient conversion between linear and multi-dimensional indices
    /// - **Bounds Checking**: Debug assertions for safety in debug builds
    ///
    /// # Examples
    ///
    /// This method is used internally by `transpose_with_copy()` to perform
    /// the actual data copying with correct coordinate mapping for arbitrary
    /// dimension swaps.
    unsafe fn transpose_copy_stride_aware(&self, result: &mut Tensor, dim0: usize, dim1: usize) {
        let src_ptr = self.as_ptr();
        let dst_ptr = result.as_mut_ptr();

        // Iterate through all elements of the result tensor
        for dst_idx in 0..result.size() {
            // Convert linear index to multi-dimensional coordinates for result
            let mut dst_coords = Vec::new();
            let mut temp_idx = dst_idx;

            for &dim_size in result.shape().dims.iter().rev() {
                dst_coords.push(temp_idx % dim_size);
                temp_idx /= dim_size;
            }
            dst_coords.reverse();

            // Map result coordinates to source coordinates (reverse the transpose)
            let mut src_coords = dst_coords.clone();
            src_coords.swap(dim0, dim1);

            // Calculate source offset using strides
            let src_offset = self.shape().offset(&src_coords);

            // Bounds check to prevent buffer overruns
            debug_assert!(
                src_offset < self.size(),
                "Source offset {} out of bounds for tensor size {}",
                src_offset,
                self.size()
            );
            debug_assert!(
                dst_idx < result.size(),
                "Destination index {} out of bounds for result size {}",
                dst_idx,
                result.size()
            );

            // Copy element
            *dst_ptr.add(dst_idx) = *src_ptr.add(src_offset);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d_basic() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .expect("Failed to create tensor");
        let transposed = tensor.transpose(0, 1);

        assert_eq!(transposed.shape().dims, vec![3, 2]);

        // Verify data layout: original [2,3] -> transposed [3,2]
        // Original: [[1,2,3], [4,5,6]]
        // Transposed: [[1,4], [2,5], [3,6]]
        assert_eq!(transposed.get(&[0, 0]), 1.0);
        assert_eq!(transposed.get(&[0, 1]), 4.0);
        assert_eq!(transposed.get(&[1, 0]), 2.0);
        assert_eq!(transposed.get(&[1, 1]), 5.0);
        assert_eq!(transposed.get(&[2, 0]), 3.0);
        assert_eq!(transposed.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix =
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("Failed to create tensor");
        let transposed = matrix.t();

        assert_eq!(transposed.shape().dims, vec![2, 2]);

        // Original: [[1,2], [3,4]]
        // Transposed: [[1,3], [2,4]]
        assert_eq!(transposed.get(&[0, 0]), 1.0);
        assert_eq!(transposed.get(&[0, 1]), 3.0);
        assert_eq!(transposed.get(&[1, 0]), 2.0);
        assert_eq!(transposed.get(&[1, 1]), 4.0);
    }

    #[test]
    fn test_transpose_3d() {
        let tensor = Tensor::new(vec![2, 3, 4]);
        let transposed = tensor.transpose(0, 2);

        // Shape changes from [2,3,4] to [4,3,2]
        assert_eq!(transposed.shape().dims, vec![4, 3, 2]);
    }

    #[test]
    fn test_transpose_same_dimension() {
        let tensor =
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("Failed to create tensor");
        let result = tensor.transpose(1, 1);

        // Should be identical to original
        assert_eq!(result.shape().dims, tensor.shape().dims);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(result.get(&[i, j]), tensor.get(&[i, j]));
            }
        }
    }

    #[test]
    fn test_transpose_preserves_gradient_requirement() {
        let mut tensor = Tensor::new(vec![2, 3]);
        tensor.set_requires_grad(true);
        let transposed = tensor.transpose(0, 1);

        assert!(transposed.requires_grad());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_transpose_invalid_dimension() {
        let tensor = Tensor::new(vec![2, 3]);
        tensor.transpose(0, 3); // Should panic: dim 3 out of bounds
    }

    #[test]
    #[should_panic(expected = "Matrix transpose requires at least 2 dimensions")]
    fn test_matrix_transpose_1d() {
        let tensor = Tensor::new(vec![5]);
        tensor.t(); // Should panic: 1D tensor
    }

    #[test]
    fn test_transpose_large_tensor() {
        // Test with larger tensor to exercise cache-optimized path
        let tensor = Tensor::new(vec![32, 32]); // 1024 elements
        let transposed = tensor.transpose(0, 1);

        assert_eq!(transposed.shape().dims, vec![32, 32]);
    }

    #[test]
    fn test_transpose_memory_layout() {
        let tensor = Tensor::new(vec![3, 4]);
        assert!(tensor.is_contiguous());

        let transposed = tensor.transpose(0, 1);
        // After transpose, the result should still be valid but may not be contiguous
        // depending on implementation (view vs copy)
        assert_eq!(transposed.shape().dims, vec![4, 3]);
    }

    #[test]
    fn test_transpose_first_dimensions_3d() {
        // Test the critical bug fix: transpose dimensions other than last two
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();

        // Transpose the first two dimensions (not last two)
        let transposed = tensor.transpose(0, 1);

        // Shape should change from [2,3,4] to [3,2,4]
        assert_eq!(transposed.shape().dims, vec![3, 2, 4]);

        // Verify data is correctly transposed
        // Original: tensor[d0][d1][d2] where d0=2, d1=3, d2=4
        // After transpose(0,1): tensor[d1][d0][d2] where d1=3, d0=2, d2=4

        assert_eq!(transposed.get(&[0, 0, 0]), 0.0); // Maps to original [0,0,0]
        assert_eq!(transposed.get(&[0, 1, 0]), 12.0); // Maps to original [1,0,0]
        assert_eq!(transposed.get(&[1, 0, 0]), 4.0); // Maps to original [0,1,0]
        assert_eq!(transposed.get(&[1, 1, 0]), 16.0); // Maps to original [1,1,0]
        assert_eq!(transposed.get(&[2, 0, 0]), 8.0); // Maps to original [0,2,0]
        assert_eq!(transposed.get(&[2, 1, 0]), 20.0); // Maps to original [1,2,0]
    }
}
