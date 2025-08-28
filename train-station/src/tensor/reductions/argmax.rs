//! Argmax reduction operations for tensors
//!
//! This module provides argmax operations that find the indices of maximum values
//! in tensors. These operations are non-differentiable and never require gradients.
//!
//! # Operations
//!
//! * `argmax()` - Find the index of the maximum value across all elements
//! * `argmax_dim()` - Find the indices of maximum values along a specific dimension
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 5.0, 3.0, 2.0], vec![4]).unwrap();
//! let max_idx = tensor.argmax();
//! assert_eq!(max_idx.get(&[0]), 1.0); // Index 1 has the maximum value 5.0
//! ```

use crate::tensor::core::Tensor;

impl Tensor {
    /// Returns the index of the maximum value across all elements in the tensor
    ///
    /// This operation finds the flat index (0-based) of the element with the highest value.
    /// If multiple elements have the same maximum value, the index of the first occurrence
    /// is returned. The output is a scalar tensor with shape \[1\] containing the index as a float.
    ///
    /// This operation is non-differentiable and the output never requires gradients.
    ///
    /// # Returns
    ///
    /// A tensor with shape \[1\] containing the flat index of the maximum value
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // 1D tensor
    /// let tensor = Tensor::from_slice(&[1.0, 5.0, 3.0, 2.0], vec![4]).unwrap();
    /// let max_idx = tensor.argmax();
    /// assert_eq!(max_idx.shape().dims, vec![1]);
    /// assert_eq!(max_idx.get(&[0]), 1.0); // Index 1 has value 5.0
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // 2D tensor
    /// let tensor = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0, 0.0, 5.0], vec![2, 3]).unwrap();
    /// let max_idx = tensor.argmax();
    /// assert_eq!(max_idx.get(&[0]), 5.0); // Flat index 5 has value 5.0
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Tied values return first occurrence
    /// let tensor = Tensor::from_slice(&[3.0, 5.0, 5.0, 2.0], vec![4]).unwrap();
    /// let max_idx = tensor.argmax();
    /// assert_eq!(max_idx.get(&[0]), 1.0); // First occurrence of 5.0 at index 1
    /// ```
    pub fn argmax(&self) -> Tensor {
        let mut out = Tensor::new(vec![1]);
        if self.size() == 0 {
            out.fill(0.0);
            return out;
        }

        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0usize;

        if self.is_contiguous() {
            // Fast path for contiguous tensors
            unsafe {
                let src = self.as_ptr();
                for i in 0..self.size() {
                    let v = *src.add(i);
                    if v > best_val {
                        best_val = v;
                        best_idx = i;
                    }
                }
            }
        } else {
            // Stride-aware path for non-contiguous tensors
            let dims = self.shape().dims.clone();
            for flat_idx in 0..self.size() {
                // Convert flat index to multi-dimensional coordinates
                let mut coords = vec![0; dims.len()];
                let mut tmp = flat_idx;
                for k in (0..dims.len()).rev() {
                    coords[k] = tmp % dims[k];
                    tmp /= dims[k];
                }

                // Get value using stride-aware offset
                let offset = self.shape().offset(&coords);
                let v = unsafe { *self.as_ptr().add(offset) };
                if v > best_val {
                    best_val = v;
                    best_idx = flat_idx;
                }
            }
        }

        unsafe {
            *out.as_mut_ptr() = best_idx as f32;
        }
        out
    }

    /// Returns the indices of maximum values along a specified dimension
    ///
    /// This operation finds the indices of maximum values along the specified dimension.
    /// For each slice along the dimension, it returns the index of the maximum value.
    /// If multiple elements have the same maximum value, the index of the first occurrence
    /// is returned.
    ///
    /// The output shape depends on the `keepdim` parameter:
    /// * If `keepdim` is `true`, the reduced dimension is kept with size 1
    /// * If `keepdim` is `false`, the reduced dimension is removed
    ///
    /// This operation is non-differentiable and the output never requires gradients.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to find argmax indices (0-based)
    /// * `keepdim` - Whether to keep the reduced dimension with size 1
    ///
    /// # Returns
    ///
    /// A tensor containing the indices of maximum values along the specified dimension
    ///
    /// # Panics
    ///
    /// Panics if `dim` is out of bounds for the tensor's rank or if the dimension size is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // 2D tensor: [[1.0, 3.0, 2.0],
    /// //             [4.0, 0.0, 5.0]]
    /// let tensor = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0, 0.0, 5.0], vec![2, 3]).unwrap();
    ///
    /// // argmax along columns (dim=1)
    /// let col_max_idx = tensor.argmax_dim(1, false);
    /// assert_eq!(col_max_idx.shape().dims, vec![2]);
    /// assert_eq!(col_max_idx.get(&[0]), 1.0); // Row 0: max at index 1 (value 3.0)
    /// assert_eq!(col_max_idx.get(&[1]), 2.0); // Row 1: max at index 2 (value 5.0)
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // argmax along rows (dim=0) with keepdim
    /// let tensor = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0, 0.0, 5.0], vec![2, 3]).unwrap();
    /// let row_max_idx = tensor.argmax_dim(0, true);
    /// assert_eq!(row_max_idx.shape().dims, vec![1, 3]);
    /// assert_eq!(row_max_idx.get(&[0, 0]), 1.0); // Col 0: max at index 1 (value 4.0)
    /// assert_eq!(row_max_idx.get(&[0, 1]), 0.0); // Col 1: max at index 0 (value 3.0)
    /// assert_eq!(row_max_idx.get(&[0, 2]), 1.0); // Col 2: max at index 1 (value 5.0)
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // 1D tensor edge case
    /// let tensor = Tensor::from_slice(&[5.0, 1.0, 8.0, 3.0], vec![4]).unwrap();
    /// let max_idx = tensor.argmax_dim(0, false);
    /// assert_eq!(max_idx.shape().dims, vec![1]); // Special case: becomes [1] not []
    /// assert_eq!(max_idx.get(&[0]), 2.0); // Index 2 has maximum value 8.0
    /// ```
    pub fn argmax_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        let rank = self.shape().rank();
        assert!(
            dim < rank,
            "argmax_dim dim {} out of bounds for rank {}",
            dim,
            rank
        );

        let in_dims = self.shape().dims.clone();
        let reduce_size = in_dims[dim];
        assert!(reduce_size > 0, "cannot argmax over empty dimension");

        // Build output shape
        let mut out_dims = in_dims.clone();
        if keepdim {
            out_dims[dim] = 1;
        } else {
            out_dims.remove(dim);
        }
        if out_dims.is_empty() {
            out_dims.push(1);
        }

        let mut out = Tensor::zeros(out_dims.clone());

        // Use stride-aware approach to handle non-contiguous tensors correctly
        let out_size = out.size();

        unsafe {
            let dst = out.as_mut_ptr();

            // Iterate over all output positions
            for out_idx in 0..out_size {
                // Convert flat output index to multi-dimensional coordinates
                let mut out_coords = vec![0; out_dims.len()];
                let mut tmp = out_idx;
                for k in (0..out_dims.len()).rev() {
                    out_coords[k] = tmp % out_dims[k];
                    tmp /= out_dims[k];
                }

                // Convert output coordinates to input coordinates
                let mut in_coords = vec![0; rank];
                if keepdim {
                    // When keepdim=true, output coords map directly to input coords
                    for k in 0..rank {
                        if k == dim {
                            in_coords[k] = 0; // Will be set in the loop below
                        } else {
                            in_coords[k] = out_coords[k];
                        }
                    }
                } else {
                    // When keepdim=false, we need to insert the missing dimension
                    let mut out_coord_idx = 0;
                    for (k, in_coord) in in_coords.iter_mut().enumerate().take(rank) {
                        if k == dim {
                            *in_coord = 0; // Will be set in the loop below
                        } else {
                            *in_coord = out_coords[out_coord_idx];
                            out_coord_idx += 1;
                        }
                    }
                }

                // Find the argmax along the specified dimension
                let mut best_val = f32::NEG_INFINITY;
                let mut best_j = 0usize;

                for j in 0..reduce_size {
                    in_coords[dim] = j;
                    let in_offset = self.shape().offset(&in_coords);
                    let v = *self.as_ptr().add(in_offset);
                    if v > best_val {
                        best_val = v;
                        best_j = j;
                    }
                }

                *dst.add(out_idx) = best_j as f32;
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ====== LEVEL 1: Basic functionality tests for contiguous tensors ======

    #[test]
    fn test_argmax_level1_basic_1d() {
        let x = Tensor::from_slice(&[3.0, -2.0, 5.0, -1.0], vec![4]).unwrap();
        let idx = x.argmax();

        // Check output shape
        assert_eq!(idx.shape().dims, vec![1]);
        assert_eq!(idx.size(), 1);

        // Check result
        assert_eq!(idx.get(&[0]), 2.0); // index 2 has value 5.0
    }

    #[test]
    fn test_argmax_level1_basic_1d_edge_cases() {
        // Single element
        let x = Tensor::from_slice(&[42.0], vec![1]).unwrap();
        let idx = x.argmax();
        assert_eq!(idx.get(&[0]), 0.0);

        // All same values (should return first occurrence)
        let x = Tensor::from_slice(&[3.0, 3.0, 3.0], vec![3]).unwrap();
        let idx = x.argmax();
        assert_eq!(idx.get(&[0]), 0.0);

        // Negative values
        let x = Tensor::from_slice(&[-5.0, -2.0, -8.0, -1.0], vec![4]).unwrap();
        let idx = x.argmax();
        assert_eq!(idx.get(&[0]), 3.0); // index 3 has value -1.0
    }

    #[test]
    fn test_argmax_level1_basic_2d_contiguous() {
        // Test argmax over all elements for 2D tensor
        // Data: [[1.0, 3.0, 2.0],
        //        [4.0, 0.0, 5.0]]
        let x = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0, 0.0, 5.0], vec![2, 3]).unwrap();
        let idx = x.argmax();

        assert_eq!(idx.shape().dims, vec![1]);
        assert_eq!(idx.get(&[0]), 5.0); // flat index 5 has value 5.0
    }

    #[test]
    fn test_argmax_level1_dim_2d_basic() {
        // Test argmax_dim for simple 2D case
        // Data: [[1.0, 3.0, 2.0],
        //        [4.0, 0.0, 5.0]]
        let x = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0, 0.0, 5.0], vec![2, 3]).unwrap();

        // argmax along dim=1 (along columns within each row)
        let idx1_keepdim = x.argmax_dim(1, true);
        assert_eq!(idx1_keepdim.shape().dims, vec![2, 1]);
        assert_eq!(idx1_keepdim.get(&[0, 0]), 1.0); // row 0: max at index 1 (value 3.0)
        assert_eq!(idx1_keepdim.get(&[1, 0]), 2.0); // row 1: max at index 2 (value 5.0)

        let idx1_no_keepdim = x.argmax_dim(1, false);
        assert_eq!(idx1_no_keepdim.shape().dims, vec![2]);
        assert_eq!(idx1_no_keepdim.get(&[0]), 1.0);
        assert_eq!(idx1_no_keepdim.get(&[1]), 2.0);

        // argmax along dim=0 (along rows within each column)
        let idx0_keepdim = x.argmax_dim(0, true);
        assert_eq!(idx0_keepdim.shape().dims, vec![1, 3]);
        assert_eq!(idx0_keepdim.get(&[0, 0]), 1.0); // col 0: max at index 1 (value 4.0)
        assert_eq!(idx0_keepdim.get(&[0, 1]), 0.0); // col 1: max at index 0 (value 3.0)
        assert_eq!(idx0_keepdim.get(&[0, 2]), 1.0); // col 2: max at index 1 (value 5.0)

        let idx0_no_keepdim = x.argmax_dim(0, false);
        assert_eq!(idx0_no_keepdim.shape().dims, vec![3]);
        assert_eq!(idx0_no_keepdim.get(&[0]), 1.0);
        assert_eq!(idx0_no_keepdim.get(&[1]), 0.0);
        assert_eq!(idx0_no_keepdim.get(&[2]), 1.0);
    }

    #[test]
    fn test_argmax_level1_3d_basic() {
        // Test 3D tensor: shape [2, 2, 2]
        // Data: [[[1.0, 2.0], [3.0, 4.0]],
        //        [[5.0, 6.0], [7.0, 8.0]]]
        let x =
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]).unwrap();

        // Global argmax
        let idx = x.argmax();
        assert_eq!(idx.get(&[0]), 7.0); // flat index 7 has value 8.0

        // argmax along dim=2 (innermost dimension)
        let idx2 = x.argmax_dim(2, false);
        assert_eq!(idx2.shape().dims, vec![2, 2]);
        assert_eq!(idx2.get(&[0, 0]), 1.0); // [1.0, 2.0] -> max at index 1
        assert_eq!(idx2.get(&[0, 1]), 1.0); // [3.0, 4.0] -> max at index 1
        assert_eq!(idx2.get(&[1, 0]), 1.0); // [5.0, 6.0] -> max at index 1
        assert_eq!(idx2.get(&[1, 1]), 1.0); // [7.0, 8.0] -> max at index 1
    }

    // ====== LEVEL 2: Non-contiguous tensors (views, permuted) ======

    #[test]
    fn test_argmax_level2_transpose_view() {
        // Create a 2x3 tensor and transpose it to get a non-contiguous view
        let x = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0, 0.0, 5.0], vec![2, 3]).unwrap();
        // Original: [[1.0, 3.0, 2.0],
        //            [4.0, 0.0, 5.0]]

        let x_t = x.transpose(0, 1);
        // Transposed: [[1.0, 4.0],
        //              [3.0, 0.0],
        //              [2.0, 5.0]]
        assert_eq!(x_t.shape().dims, vec![3, 2]);
        assert!(!x_t.is_contiguous()); // Should be a view

        // Test global argmax on transposed view
        let idx = x_t.argmax();
        assert_eq!(idx.get(&[0]), 5.0); // flat index 5 still points to value 5.0

        // Test argmax along dim=0 of transposed tensor
        let idx0 = x_t.argmax_dim(0, false);
        assert_eq!(idx0.shape().dims, vec![2]);
        assert_eq!(idx0.get(&[0]), 1.0); // col 0: [1.0, 3.0, 2.0] -> max 3.0 at index 1
        assert_eq!(idx0.get(&[1]), 2.0); // col 1: [4.0, 0.0, 5.0] -> max 5.0 at index 2

        // Test argmax along dim=1 of transposed tensor
        let idx1 = x_t.argmax_dim(1, false);
        assert_eq!(idx1.shape().dims, vec![3]);
        assert_eq!(idx1.get(&[0]), 1.0); // row 0: [1.0, 4.0] -> max 4.0 at index 1
        assert_eq!(idx1.get(&[1]), 0.0); // row 1: [3.0, 0.0] -> max 3.0 at index 0
        assert_eq!(idx1.get(&[2]), 1.0); // row 2: [2.0, 5.0] -> max 5.0 at index 1
    }

    #[test]
    fn test_argmax_level2_slice_view() {
        // Create a 3x4 tensor and take a slice
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let x = Tensor::from_slice(&data, vec![3, 4]).unwrap();
        // [[1, 2, 3, 4],
        //  [5, 6, 7, 8],
        //  [9, 10, 11, 12]]

        // Select middle row (creates a view)
        let middle_row = x.select(0, 1);
        // [5, 6, 7, 8]
        assert_eq!(middle_row.shape().dims, vec![4]);

        let idx = middle_row.argmax();
        assert_eq!(idx.get(&[0]), 3.0); // index 3 has value 8.0

        // Test argmax_dim on 1D slice (should work the same as global argmax)
        let idx_dim = middle_row.argmax_dim(0, false);
        assert_eq!(idx_dim.shape().dims, vec![1]);
        assert_eq!(idx_dim.get(&[0]), 3.0);
    }

    #[test]
    fn test_argmax_level2_permuted_3d() {
        // Test 3D tensor with permuted dimensions
        let data = (0..24).map(|i| i as f32).collect::<Vec<_>>();
        let x = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();
        // Shape [2, 3, 4] with values 0 to 23

        // Permute to [4, 2, 3] (swap dims 0 and 2)
        let x_perm = x.permute(vec![2, 1, 0]);
        assert_eq!(x_perm.shape().dims, vec![4, 3, 2]);
        assert!(!x_perm.is_contiguous());

        // Global argmax should still find the maximum value (23)
        let idx = x_perm.argmax();
        assert_eq!(idx.get(&[0]), 23.0); // The max value is still 23

        // Test argmax along each dimension of permuted tensor
        let idx0 = x_perm.argmax_dim(0, false); // [3, 2]
        assert_eq!(idx0.shape().dims, vec![3, 2]);

        let idx1 = x_perm.argmax_dim(1, false); // [4, 2]
        assert_eq!(idx1.shape().dims, vec![4, 2]);

        let idx2 = x_perm.argmax_dim(2, false); // [4, 3]
        assert_eq!(idx2.shape().dims, vec![4, 3]);
    }

    #[test]
    fn test_argmax_level2_nested_views() {
        // Test nested transformations (transpose then select)
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let x = Tensor::from_slice(&data, vec![4, 3]).unwrap();

        // First transpose, then select a row
        let x_t = x.transpose(0, 1); // [3, 4]
        let row = x_t.select(0, 1); // Select second row: [2, 5, 8, 11]
        assert_eq!(row.shape().dims, vec![4]);

        let idx = row.argmax();
        assert_eq!(idx.get(&[0]), 3.0); // index 3 has value 11.0
    }

    // ====== LEVEL 3: Complex multi-dimensional cases and edge scenarios ======

    #[test]
    fn test_argmax_level3_4d_tensor() {
        // Test 4D tensor with various reduction dimensions
        let data = (0..120).map(|i| i as f32).collect::<Vec<_>>();
        let x = Tensor::from_slice(&data, vec![2, 3, 4, 5]).unwrap();
        // Shape [2, 3, 4, 5] with values 0 to 119

        // Global argmax
        let idx = x.argmax();
        assert_eq!(idx.get(&[0]), 119.0); // Maximum value 119.0 at flat index 119

        // Test argmax along each dimension
        let idx0_keepdim = x.argmax_dim(0, true);
        assert_eq!(idx0_keepdim.shape().dims, vec![1, 3, 4, 5]);

        let idx0_no_keepdim = x.argmax_dim(0, false);
        assert_eq!(idx0_no_keepdim.shape().dims, vec![3, 4, 5]);

        let idx1_keepdim = x.argmax_dim(1, true);
        assert_eq!(idx1_keepdim.shape().dims, vec![2, 1, 4, 5]);

        let idx1_no_keepdim = x.argmax_dim(1, false);
        assert_eq!(idx1_no_keepdim.shape().dims, vec![2, 4, 5]);

        let idx2_keepdim = x.argmax_dim(2, true);
        assert_eq!(idx2_keepdim.shape().dims, vec![2, 3, 1, 5]);

        let idx2_no_keepdim = x.argmax_dim(2, false);
        assert_eq!(idx2_no_keepdim.shape().dims, vec![2, 3, 5]);

        let idx3_keepdim = x.argmax_dim(3, true);
        assert_eq!(idx3_keepdim.shape().dims, vec![2, 3, 4, 1]);

        let idx3_no_keepdim = x.argmax_dim(3, false);
        assert_eq!(idx3_no_keepdim.shape().dims, vec![2, 3, 4]);

        // Check some specific values for the innermost dimension (dim=3)
        // For each [i, j, k, :] slice, argmax should be 4 (index of max in size-5 dimension)
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(idx3_no_keepdim.get(&[i, j, k]), 4.0);
                    assert_eq!(idx3_keepdim.get(&[i, j, k, 0]), 4.0);
                }
            }
        }
    }

    #[test]
    fn test_argmax_level3_edge_cases_keepdim() {
        // Test edge case: 1D tensor with keepdim
        let x1d = Tensor::from_slice(&[5.0, 1.0, 8.0, 3.0], vec![4]).unwrap();

        let idx_keepdim = x1d.argmax_dim(0, true);
        assert_eq!(idx_keepdim.shape().dims, vec![1]);
        assert_eq!(idx_keepdim.get(&[0]), 2.0);

        let idx_no_keepdim = x1d.argmax_dim(0, false);
        assert_eq!(idx_no_keepdim.shape().dims, vec![1]); // Special case: becomes [1] not []
        assert_eq!(idx_no_keepdim.get(&[0]), 2.0);

        // Test edge case: dimension of size 1
        let x_size_1 = Tensor::from_slice(&[42.0], vec![1]).unwrap();

        let idx = x_size_1.argmax_dim(0, true);
        assert_eq!(idx.shape().dims, vec![1]);
        assert_eq!(idx.get(&[0]), 0.0);

        let idx = x_size_1.argmax_dim(0, false);
        assert_eq!(idx.shape().dims, vec![1]);
        assert_eq!(idx.get(&[0]), 0.0);
    }

    #[test]
    fn test_argmax_level3_ties_handling() {
        // Test that tied values return the first occurrence (PyTorch behavior)
        let x = Tensor::from_slice(&[3.0, 5.0, 5.0, 2.0, 5.0], vec![5]).unwrap();

        let idx = x.argmax();
        assert_eq!(idx.get(&[0]), 1.0); // First occurrence of max value 5.0

        // Test with 2D ties
        let x2d = Tensor::from_slice(&[3.0, 5.0, 5.0, 2.0, 1.0, 5.0], vec![3, 2]).unwrap();

        // argmax along dim=0 (columns)
        let idx0 = x2d.argmax_dim(0, false);
        assert_eq!(idx0.shape().dims, vec![2]);
        assert_eq!(idx0.get(&[0]), 1.0); // col 0: [3, 5, 1] -> first 5 at index 1
        assert_eq!(idx0.get(&[1]), 0.0); // col 1: [5, 2, 5] -> first 5 at index 0

        // argmax along dim=1 (rows)
        let idx1 = x2d.argmax_dim(1, false);
        assert_eq!(idx1.shape().dims, vec![3]);
        assert_eq!(idx1.get(&[0]), 1.0); // row 0: [3, 5] -> max at index 1
        assert_eq!(idx1.get(&[1]), 0.0); // row 1: [5, 2] -> max at index 0
        assert_eq!(idx1.get(&[2]), 1.0); // row 2: [1, 5] -> max at index 1
    }

    #[test]
    fn test_argmax_level3_extreme_values() {
        // Test with extreme floating point values
        let x = Tensor::from_slice(
            &[f32::NEG_INFINITY, -1e10, 0.0, 1e10, f32::INFINITY, f32::NAN],
            vec![6],
        )
        .unwrap();

        let idx = x.argmax();
        // NaN comparison behavior: NaN is not > any value, so INFINITY should win
        assert_eq!(idx.get(&[0]), 4.0); // f32::INFINITY at index 4

        // Test negative values only
        let x_neg = Tensor::from_slice(&[-10.0, -5.0, -15.0, -1.0], vec![4]).unwrap();
        let idx = x_neg.argmax();
        assert_eq!(idx.get(&[0]), 3.0); // -1.0 is the maximum at index 3
    }

    #[test]
    fn test_argmax_level3_large_dimensions() {
        // Test with one very large dimension
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| (size - 1 - i) as f32).collect(); // Decreasing values
        let x = Tensor::from_slice(&data, vec![size]).unwrap();

        let idx = x.argmax();
        assert_eq!(idx.get(&[0]), 0.0); // First element has max value (size-1)

        // Test with multiple dimensions where one is large
        let data2: Vec<f32> = (0..(10 * 100)).map(|i| i as f32).collect();
        let x2 = Tensor::from_slice(&data2, vec![10, 100]).unwrap();

        let idx = x2.argmax();
        assert_eq!(idx.get(&[0]), 999.0); // Last element has max value

        // Test argmax along the large dimension
        let idx_dim1 = x2.argmax_dim(1, false);
        assert_eq!(idx_dim1.shape().dims, vec![10]);
        // Each row's max should be at index 99 (last column)
        for i in 0..10 {
            assert_eq!(idx_dim1.get(&[i]), 99.0);
        }
    }

    #[test]
    fn test_argmax_level3_consistency_with_pytorch_behavior() {
        // Test specific patterns that should match PyTorch exactly

        // Pattern 1: 3D tensor, reduce middle dimension
        let x = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, // [0, 0, :]
                5.0, 6.0, 7.0, 8.0, // [0, 1, :]
                9.0, 8.0, 7.0, 6.0, // [1, 0, :]
                5.0, 4.0, 3.0, 2.0, // [1, 1, :]
            ],
            vec![2, 2, 4],
        )
        .unwrap();

        // Reduce along dim=1 (middle dimension)
        let idx = x.argmax_dim(1, true);
        assert_eq!(idx.shape().dims, vec![2, 1, 4]);

        // For [0, :, j] where j=0,1,2,3: values are [1,5], [2,6], [3,7], [4,8]
        // Max indices should be [1,1,1,1] (second slice wins)
        assert_eq!(idx.get(&[0, 0, 0]), 1.0);
        assert_eq!(idx.get(&[0, 0, 1]), 1.0);
        assert_eq!(idx.get(&[0, 0, 2]), 1.0);
        assert_eq!(idx.get(&[0, 0, 3]), 1.0);

        // For [1, :, j] where j=0,1,2,3: values are [9,5], [8,4], [7,3], [6,2]
        // Max indices should be [0,0,0,0] (first slice wins)
        assert_eq!(idx.get(&[1, 0, 0]), 0.0);
        assert_eq!(idx.get(&[1, 0, 1]), 0.0);
        assert_eq!(idx.get(&[1, 0, 2]), 0.0);
        assert_eq!(idx.get(&[1, 0, 3]), 0.0);
    }
}
