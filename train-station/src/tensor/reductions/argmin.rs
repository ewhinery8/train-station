use crate::tensor::core::Tensor;

impl Tensor {
    /// Returns the index of the minimum value in the tensor
    ///
    /// This method finds the flat index of the minimum value across all elements
    /// in the tensor. The result is a scalar tensor containing the index as a
    /// floating-point value. This operation is non-differentiable and the output
    /// never requires gradient tracking.
    ///
    /// # Returns
    ///
    /// A tensor with shape `[1]` containing the flat index of the minimum value
    /// as a `f32`. If the input tensor is empty, returns `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[3.0, -2.0, 5.0, -1.0], vec![4]).unwrap();
    /// let min_index = tensor.argmin();
    /// assert_eq!(min_index.get(&[0]), 1.0); // -2.0 is at index 1
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Empty tensor case
    /// let empty_tensor = Tensor::new(vec![0]);
    /// let min_index = empty_tensor.argmin();
    /// assert_eq!(min_index.get(&[0]), 0.0);
    /// ```
    pub fn argmin(&self) -> Tensor {
        let mut out = Tensor::new(vec![1]);
        if self.size() == 0 {
            out.fill(0.0);
            return out;
        }

        let mut best_val = f32::INFINITY;
        let mut best_idx = 0usize;

        if self.is_contiguous() {
            // Fast path for contiguous tensors
            unsafe {
                let src = self.as_ptr();
                for i in 0..self.size() {
                    let v = *src.add(i);
                    if v < best_val {
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
                if v < best_val {
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

    /// Returns the indices of minimum values along a specified dimension
    ///
    /// This method finds the indices of minimum values along the specified dimension.
    /// The result contains the indices where the minimum values occur in that dimension.
    /// This operation is non-differentiable and the output never requires gradient tracking.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to find minimum indices (0-based)
    /// * `keepdim` - Whether to keep the reduced dimension in the output shape
    ///   - If `true`, the reduced dimension is kept with size 1
    ///   - If `false`, the reduced dimension is removed from the output shape
    ///
    /// # Returns
    ///
    /// A tensor containing the indices of minimum values along the specified dimension.
    /// The output shape depends on `keepdim`:
    /// - If `keepdim` is `true`, the reduced dimension has size 1
    /// - If `keepdim` is `false`, the reduced dimension is removed
    ///
    /// # Panics
    ///
    /// * If `dim` is out of bounds for the tensor's rank
    /// * If the dimension to reduce has size 0
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[3.0, -2.0, 5.0, -1.0, 0.0, -3.0], vec![2, 3]).unwrap();
    ///
    /// // Find minimum indices along dimension 1 (columns), keeping the dimension
    /// let indices = tensor.argmin_dim(1, true);
    /// assert_eq!(indices.shape().dims, vec![2, 1]);
    /// assert_eq!(indices.get(&[0, 0]), 1.0); // -2.0 is at index 1 in first row
    /// assert_eq!(indices.get(&[1, 0]), 2.0); // -3.0 is at index 2 in second row
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[3.0, -2.0, 5.0, -1.0, 0.0, -3.0], vec![2, 3]).unwrap();
    ///
    /// // Find minimum indices along dimension 1 (columns), removing the dimension
    /// let indices = tensor.argmin_dim(1, false);
    /// assert_eq!(indices.shape().dims, vec![2]);
    /// assert_eq!(indices.get(&[0]), 1.0); // -2.0 is at index 1 in first row
    /// assert_eq!(indices.get(&[1]), 2.0); // -3.0 is at index 2 in second row
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    ///
    /// // Find minimum index in a 1D tensor
    /// let index = tensor.argmin_dim(0, false);
    /// assert_eq!(index.shape().dims, vec![1]);
    /// assert_eq!(index.get(&[0]), 0.0); // 1.0 is at index 0
    /// ```
    pub fn argmin_dim(&self, dim: usize, keepdim: bool) -> Tensor {
        let rank = self.shape().rank();
        assert!(
            dim < rank,
            "argmin_dim dim {} out of bounds for rank {}",
            dim,
            rank
        );

        let in_dims = self.shape().dims.clone();
        let reduce_size = in_dims[dim];
        assert!(reduce_size > 0, "cannot argmin over empty dimension");

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
                    for (k, &out_coord) in out_coords.iter().enumerate() {
                        if k == dim {
                            in_coords[k] = 0; // Will be set in the loop below
                        } else {
                            in_coords[k] = out_coord;
                        }
                    }
                } else {
                    // When keepdim=false, we need to insert the missing dimension
                    let mut out_coord_idx = 0;
                    for (k, in_coord) in in_coords.iter_mut().enumerate() {
                        if k == dim {
                            *in_coord = 0; // Will be set in the loop below
                        } else {
                            *in_coord = out_coords[out_coord_idx];
                            out_coord_idx += 1;
                        }
                    }
                }

                // Find the argmin along the specified dimension
                let mut best_val = f32::INFINITY;
                let mut best_j = 0usize;

                for j in 0..reduce_size {
                    in_coords[dim] = j;
                    let in_offset = self.shape().offset(&in_coords);
                    let v = *self.as_ptr().add(in_offset);
                    if v < best_val {
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

    // Level 1 Tests: Basic functionality with simple contiguous tensors
    #[test]
    fn test_argmin_level1_basic_1d() {
        // Simple 1D case
        let x = Tensor::from_slice(&[3.0, -2.0, 5.0, -1.0], vec![4]).unwrap();
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 1.0); // -2.0 is at index 1
        assert_eq!(idx.shape().dims, vec![1]);
    }

    #[test]
    fn test_argmin_level1_basic_1d_edge_cases() {
        // Single element
        let x = Tensor::from_slice(&[42.0], vec![1]).unwrap();
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 0.0);

        // All same values - should return first occurrence
        let x = Tensor::from_slice(&[5.0, 5.0, 5.0], vec![3]).unwrap();
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 0.0);

        // Negative values
        let x = Tensor::from_slice(&[-1.0, -5.0, -2.0], vec![3]).unwrap();
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 1.0); // -5.0 is at index 1
    }

    #[test]
    fn test_argmin_level1_basic_2d_contiguous() {
        // Simple 2D case - whole tensor argmin
        let x = Tensor::from_slice(&[3.0, -2.0, 5.0, -1.0, 0.0, -3.0], vec![2, 3]).unwrap();
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 5.0); // -3.0 is at flat index 5
        assert_eq!(idx.shape().dims, vec![1]);
    }

    #[test]
    fn test_argmin_level1_dim_2d_basic() {
        // Test argmin_dim with 2D tensor
        let x = Tensor::from_slice(&[3.0, -2.0, 5.0, -1.0, 0.0, -3.0], vec![2, 3]).unwrap();
        // Tensor looks like:
        // [[3.0, -2.0, 5.0],
        //  [-1.0, 0.0, -3.0]]

        // Along dimension 1 (columns), keepdim=true
        let idx1 = x.argmin_dim(1, true);
        assert_eq!(idx1.shape().dims, vec![2, 1]);
        assert_eq!(idx1.get(&[0, 0]), 1.0); // Row 0: -2.0 is at column index 1
        assert_eq!(idx1.get(&[1, 0]), 2.0); // Row 1: -3.0 is at column index 2

        // Along dimension 1 (columns), keepdim=false
        let idx1_no_keep = x.argmin_dim(1, false);
        assert_eq!(idx1_no_keep.shape().dims, vec![2]);
        assert_eq!(idx1_no_keep.get(&[0]), 1.0);
        assert_eq!(idx1_no_keep.get(&[1]), 2.0);

        // Along dimension 0 (rows), keepdim=true
        let idx0 = x.argmin_dim(0, true);
        assert_eq!(idx0.shape().dims, vec![1, 3]);
        assert_eq!(idx0.get(&[0, 0]), 1.0); // Column 0: -1.0 is at row index 1
        assert_eq!(idx0.get(&[0, 1]), 0.0); // Column 1: -2.0 is at row index 0
        assert_eq!(idx0.get(&[0, 2]), 1.0); // Column 2: -3.0 is at row index 1
    }

    #[test]
    fn test_argmin_level1_3d_basic() {
        // Test with 3D tensor
        let data = vec![
            1.0, -2.0, // [0,0,:] = [1.0, -2.0]
            3.0, 4.0, // [0,1,:] = [3.0, 4.0]
            -5.0, 6.0, // [1,0,:] = [-5.0, 6.0]
            7.0, -8.0, // [1,1,:] = [7.0, -8.0]
        ];
        let x = Tensor::from_slice(&data, vec![2, 2, 2]).unwrap();

        // Whole tensor argmin
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 7.0); // -8.0 is at flat index 7

        // Along dimension 2 (innermost), keepdim=false
        let idx2 = x.argmin_dim(2, false);
        assert_eq!(idx2.shape().dims, vec![2, 2]);
        assert_eq!(idx2.get(&[0, 0]), 1.0); // [1.0, -2.0] -> min at index 1
        assert_eq!(idx2.get(&[0, 1]), 0.0); // [3.0, 4.0] -> min at index 0
        assert_eq!(idx2.get(&[1, 0]), 0.0); // [-5.0, 6.0] -> min at index 0
        assert_eq!(idx2.get(&[1, 1]), 1.0); // [7.0, -8.0] -> min at index 1
    }

    // Level 2 Tests: Complex shapes, higher dimensions, and edge cases
    #[test]
    fn test_argmin_level2_large_tensors() {
        // Test with larger tensors
        let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.1 - 50.0).collect();
        // Values from -50.0 to 49.9, minimum at index 0
        let x = Tensor::from_slice(&data, vec![1000]).unwrap();
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 0.0);

        // Reshape to 2D
        let x_2d = Tensor::from_slice(&data, vec![25, 40]).unwrap();
        let idx_2d = x_2d.argmin();
        assert_eq!(idx_2d.get(&[0]), 0.0);

        // Test along different dimensions
        let idx_dim0 = x_2d.argmin_dim(0, false);
        assert_eq!(idx_dim0.shape().dims, vec![40]);
        assert_eq!(idx_dim0.get(&[0]), 0.0); // Column 0: minimum at row 0

        let idx_dim1 = x_2d.argmin_dim(1, false);
        assert_eq!(idx_dim1.shape().dims, vec![25]);
        assert_eq!(idx_dim1.get(&[0]), 0.0); // Row 0: minimum at column 0
    }

    #[test]
    fn test_argmin_level2_4d_tensor() {
        // Test with 4D tensor [2, 3, 4, 5] = 120 elements
        let data: Vec<f32> = (0..120).map(|i| 120.0 - i as f32).collect();
        // Values from 120.0 down to 1.0, minimum at last index
        let x = Tensor::from_slice(&data, vec![2, 3, 4, 5]).unwrap();

        // Global argmin
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 119.0); // minimum value 1.0 is at index 119

        // Test argmin along dimension 3 (innermost)
        let idx3 = x.argmin_dim(3, false);
        assert_eq!(idx3.shape().dims, vec![2, 3, 4]);
        // Each slice along dim 3 has values decreasing, so min is always at index 4
        assert_eq!(idx3.get(&[0, 0, 0]), 4.0);
        assert_eq!(idx3.get(&[1, 2, 3]), 4.0);

        // Test argmin along dimension 0 (outermost)
        let idx0 = x.argmin_dim(0, false);
        assert_eq!(idx0.shape().dims, vec![3, 4, 5]);
        // For each position, the minimum is in the second batch (index 1)
        assert_eq!(idx0.get(&[0, 0, 0]), 1.0);
        assert_eq!(idx0.get(&[2, 3, 4]), 1.0);
    }

    #[test]
    fn test_argmin_level2_special_values() {
        // Test with special floating point values
        let data = vec![
            f32::NAN,       // 0
            f32::INFINITY,  // 1
            -f32::INFINITY, // 2 <- this should be minimum
            0.0,            // 3
            -0.0,           // 4
            1.0,            // 5
        ];
        let x = Tensor::from_slice(&data, vec![6]).unwrap();
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 2.0); // -infinity at index 2

        // Test with all NaN
        let nan_data = vec![f32::NAN, f32::NAN, f32::NAN];
        let x_nan = Tensor::from_slice(&nan_data, vec![3]).unwrap();
        let idx_nan = x_nan.argmin();
        // With all NaN, should return first index
        assert_eq!(idx_nan.get(&[0]), 0.0);

        // Test with mix of normal values and NaN
        let mixed_data = vec![1.0, f32::NAN, -5.0, f32::NAN, 3.0];
        let x_mixed = Tensor::from_slice(&mixed_data, vec![5]).unwrap();
        let idx_mixed = x_mixed.argmin();
        assert_eq!(idx_mixed.get(&[0]), 2.0); // -5.0 at index 2
    }

    #[test]
    fn test_argmin_level2_ties() {
        // Test behavior with tied minimum values (should return first occurrence)
        let data = vec![3.0, -2.0, 5.0, -2.0, 0.0, -2.0]; // -2.0 appears at indices 1, 3, 5
        let x = Tensor::from_slice(&data, vec![6]).unwrap();
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 1.0); // First occurrence of -2.0

        // Test with 2D tensor and ties
        let x_2d = Tensor::from_slice(&data, vec![2, 3]).unwrap();
        // [[3.0, -2.0, 5.0],
        //  [-2.0, 0.0, -2.0]]

        let idx_dim0 = x_2d.argmin_dim(0, false);
        assert_eq!(idx_dim0.shape().dims, vec![3]);
        assert_eq!(idx_dim0.get(&[0]), 1.0); // Column 0: min(-2.0 vs 3.0) -> row 1
        assert_eq!(idx_dim0.get(&[1]), 0.0); // Column 1: min(-2.0 vs 0.0) -> row 0
        assert_eq!(idx_dim0.get(&[2]), 1.0); // Column 2: min(5.0 vs -2.0) -> row 1

        let idx_dim1 = x_2d.argmin_dim(1, false);
        assert_eq!(idx_dim1.shape().dims, vec![2]);
        assert_eq!(idx_dim1.get(&[0]), 1.0); // Row 0: min of [3.0, -2.0, 5.0] -> col 1
        assert_eq!(idx_dim1.get(&[1]), 0.0); // Row 1: min of [-2.0, 0.0, -2.0] -> col 0 (first)
    }

    #[test]
    fn test_argmin_level2_broadcasting_dims() {
        // Test with dimensions of size 1 (singleton dimensions)
        let data = vec![5.0, -3.0, 7.0, 1.0, -8.0, 2.0];
        let x = Tensor::from_slice(&data, vec![1, 6, 1]).unwrap();

        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 4.0); // -8.0 at flat index 4

        // Test argmin along different dimensions
        let idx_dim0 = x.argmin_dim(0, false);
        assert_eq!(idx_dim0.shape().dims, vec![6, 1]);

        let idx_dim1 = x.argmin_dim(1, false);
        assert_eq!(idx_dim1.shape().dims, vec![1, 1]);
        assert_eq!(idx_dim1.get(&[0, 0]), 4.0); // -8.0 at position 4 along dim 1

        let idx_dim2 = x.argmin_dim(2, false);
        assert_eq!(idx_dim2.shape().dims, vec![1, 6]);
    }

    #[test]
    fn test_argmin_level2_complex_3d() {
        // Complex 3D case with multiple batch dimensions
        let data = vec![
            // Batch 0, Channel 0: [[1, 2], [3, 4]]
            1.0, 2.0, 3.0, 4.0, // Batch 0, Channel 1: [[5, 6], [7, 8]]
            5.0, 6.0, 7.0, 8.0, // Batch 0, Channel 2: [[-1, 0], [9, 10]]
            -1.0, 0.0, 9.0, 10.0, // Batch 1, Channel 0: [[11, 12], [13, 14]]
            11.0, 12.0, 13.0, 14.0, // Batch 1, Channel 1: [[15, 16], [17, 18]]
            15.0, 16.0, 17.0, 18.0, // Batch 1, Channel 2: [[19, 20], [21, -5]]
            19.0, 20.0, 21.0, -5.0,
        ];
        let x = Tensor::from_slice(&data, vec![2, 3, 2, 2]).unwrap();

        // Global minimum
        let idx = x.argmin();
        assert_eq!(idx.get(&[0]), 23.0); // -5.0 is at flat index 23

        // Argmin along dimension 1 (channels)
        let idx_dim1 = x.argmin_dim(1, false);
        assert_eq!(idx_dim1.shape().dims, vec![2, 2, 2]);
        // At position [0,0,0]: min(1.0, 5.0, -1.0) = -1.0 at channel 2
        assert_eq!(idx_dim1.get(&[0, 0, 0]), 2.0);
        // At position [1,1,1]: min(14.0, 18.0, -5.0) = -5.0 at channel 2
        assert_eq!(idx_dim1.get(&[1, 1, 1]), 2.0);
    }

    // Level 3 Tests: Non-contiguous tensors, views, and strided memory layouts
    #[test]
    fn test_argmin_level3_transpose_view() {
        // Create a 2x3 tensor and transpose it to get a non-contiguous view
        let x = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0, 0.0, -5.0], vec![2, 3]).unwrap();
        // Original: [[1.0, 3.0, 2.0],
        //            [4.0, 0.0, -5.0]]

        let x_t = x.transpose(0, 1);
        // Transposed: [[1.0, 4.0],
        //              [3.0, 0.0],
        //              [2.0, -5.0]]
        assert_eq!(x_t.shape().dims, vec![3, 2]);
        assert!(!x_t.is_contiguous()); // Should be a view

        // Test global argmin on transposed view
        let idx = x_t.argmin();
        assert_eq!(idx.get(&[0]), 5.0); // flat index 5 still points to value -5.0

        // Test argmin along dim=0 of transposed tensor
        let idx0 = x_t.argmin_dim(0, false);
        assert_eq!(idx0.shape().dims, vec![2]);
        assert_eq!(idx0.get(&[0]), 0.0); // col 0: [1.0, 3.0, 2.0] -> min 1.0 at index 0
        assert_eq!(idx0.get(&[1]), 2.0); // col 1: [4.0, 0.0, -5.0] -> min -5.0 at index 2

        // Test argmin along dim=1 of transposed tensor
        let idx1 = x_t.argmin_dim(1, false);
        assert_eq!(idx1.shape().dims, vec![3]);
        assert_eq!(idx1.get(&[0]), 0.0); // row 0: [1.0, 4.0] -> min 1.0 at index 0
        assert_eq!(idx1.get(&[1]), 1.0); // row 1: [3.0, 0.0] -> min 0.0 at index 1
        assert_eq!(idx1.get(&[2]), 1.0); // row 2: [2.0, -5.0] -> min -5.0 at index 1
    }

    #[test]
    fn test_argmin_level3_slice_view() {
        // Create a 3x4 tensor and take a slice
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, -6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let x = Tensor::from_slice(&data, vec![3, 4]).unwrap();
        // [[1, 2, 3, 4],
        //  [5, -6, 7, 8],
        //  [9, 10, 11, 12]]

        // Select middle row (creates a view)
        let middle_row = x.select(0, 1);
        // [5, -6, 7, 8]
        assert_eq!(middle_row.shape().dims, vec![4]);

        let idx = middle_row.argmin();
        assert_eq!(idx.get(&[0]), 1.0); // index 1 has value -6.0

        // Test argmin_dim on 1D slice (should work the same as global argmin)
        let idx_dim = middle_row.argmin_dim(0, false);
        assert_eq!(idx_dim.shape().dims, vec![1]);
        assert_eq!(idx_dim.get(&[0]), 1.0);

        // Test with column slice
        let second_col = x.select(1, 1);
        // [2, -6, 10]
        assert_eq!(second_col.shape().dims, vec![3]);
        let idx_col = second_col.argmin();
        assert_eq!(idx_col.get(&[0]), 1.0); // -6.0 at index 1
    }

    #[test]
    fn test_argmin_level3_permuted_3d() {
        // Test 3D tensor with permuted dimensions
        let data = (0..24).map(|i| 24.0 - i as f32).collect::<Vec<_>>();
        let x = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();
        // Shape [2, 3, 4] with values 24.0 down to 1.0
        // Minimum value 1.0 is at the last position

        // Permute to [4, 2, 3] (swap dims 0 and 2)
        let x_perm = x.permute(vec![2, 1, 0]);
        assert_eq!(x_perm.shape().dims, vec![4, 3, 2]);
        assert!(!x_perm.is_contiguous());

        // Global argmin should still find the minimum value (1.0)
        let idx = x_perm.argmin();
        assert_eq!(idx.get(&[0]), 23.0); // The min value 1.0 is still at flat index 23

        // Test argmin along each dimension of permuted tensor
        let idx0 = x_perm.argmin_dim(0, false); // [3, 2]
        assert_eq!(idx0.shape().dims, vec![3, 2]);

        let idx1 = x_perm.argmin_dim(1, false); // [4, 2]
        assert_eq!(idx1.shape().dims, vec![4, 2]);

        let idx2 = x_perm.argmin_dim(2, false); // [4, 3]
        assert_eq!(idx2.shape().dims, vec![4, 3]);

        // Verify some specific values
        // Since values decrease from 24.0 to 1.0, the permuted tensor should have
        // minimum values at the later positions in the original ordering
    }

    #[test]
    fn test_argmin_level3_nested_views() {
        // Test nested transformations (transpose then select)
        let data = vec![
            1.0, 2.0, -3.0, 4.0, 5.0, 6.0, 7.0, -8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let x = Tensor::from_slice(&data, vec![4, 3]).unwrap();

        // First transpose, then select a row
        let x_t = x.transpose(0, 1); // [3, 4]
        let row = x_t.select(0, 1); // Select second row: [2, 5, -8, 11]
        assert_eq!(row.shape().dims, vec![4]);

        let idx = row.argmin();
        assert_eq!(idx.get(&[0]), 2.0); // index 2 has value -8.0
    }

    #[test]
    fn test_argmin_level3_strided_memory() {
        // Test with highly strided memory patterns
        let data: Vec<f32> = (0..60).map(|i| i as f32 - 30.0).collect();
        let x = Tensor::from_slice(&data, vec![3, 4, 5]).unwrap();
        // Values from -30.0 to 29.0

        // Create complex views that result in non-contiguous memory
        let x_perm = x.permute(vec![2, 0, 1]); // [5, 3, 4]
        assert!(!x_perm.is_contiguous());

        // Test global argmin
        let idx = x_perm.argmin();
        assert_eq!(idx.get(&[0]), 0.0); // -30.0 is at index 0

        // Test dimension-wise argmin on permuted tensor
        let idx0 = x_perm.argmin_dim(0, false);
        assert_eq!(idx0.shape().dims, vec![3, 4]);

        let idx1 = x_perm.argmin_dim(1, false);
        assert_eq!(idx1.shape().dims, vec![5, 4]);

        let idx2 = x_perm.argmin_dim(2, false);
        assert_eq!(idx2.shape().dims, vec![5, 3]);
    }

    #[test]
    fn test_argmin_level3_multiple_transformations() {
        // Test with multiple chained transformations
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, -24.0,
        ];
        let x = Tensor::from_slice(&data, vec![4, 6]).unwrap();

        // Chain multiple transformations
        let x_t = x.transpose(0, 1); // [6, 4]
        let x_subset = x_t.select(0, 5); // Select last row: [6, 12, 18, -24]

        // Note: select might create contiguous tensors in some cases, so we don't assert non-contiguous
        assert_eq!(x_subset.shape().dims, vec![4]);

        let idx = x_subset.argmin();
        assert_eq!(idx.get(&[0]), 3.0); // -24.0 at index 3

        // Test on a slice of the transposed tensor
        let partial_col = x_t.select(1, 2); // Select third column: [15, 16, 17, 18, 19, 20]
        let idx_partial = partial_col.argmin();
        assert_eq!(idx_partial.get(&[0]), 0.0); // 15.0 at index 0

        // Test argmin on the non-contiguous transposed tensor
        assert!(!x_t.is_contiguous());
        let idx_trans = x_t.argmin();
        assert_eq!(idx_trans.get(&[0]), 23.0); // -24.0 is still at flat index 23
    }

    #[test]
    fn test_argmin_level3_view_consistency() {
        // Test that argmin results are consistent between original and view
        let data = vec![
            5.0, -2.0, 8.0, 1.0, // row 0: min -2.0 at col 1
            3.0, 9.0, -4.0, 7.0, // row 1: min -4.0 at col 2
            6.0, 0.0, 2.0, -1.0, // row 2: min -1.0 at col 3
        ];
        let x = Tensor::from_slice(&data, vec![3, 4]).unwrap();
        // Global minimum is -4.0 at flat index 6

        // Test argmin on original tensor
        let idx_orig = x.argmin();
        assert_eq!(idx_orig.get(&[0]), 6.0); // -4.0 at index 6

        // Create a view by transposing and test consistency
        let x_t = x.transpose(0, 1);
        // Transposed tensor:
        // [[5.0, 3.0, 6.0],     // col 0 of original -> row 0: min 3.0 at index 1
        //  [-2.0, 9.0, 0.0],    // col 1 of original -> row 1: min -2.0 at index 0
        //  [8.0, -4.0, 2.0],    // col 2 of original -> row 2: min -4.0 at index 1
        //  [1.0, 7.0, -1.0]]    // col 3 of original -> row 3: min -1.0 at index 2

        let idx_view = x_t.argmin();
        // The minimum value is still -4.0, but its flat index in the view may differ
        // Let's just check that both find the minimum value correctly

        // Extract actual minimum values to verify they're the same
        let min_val_orig = unsafe {
            let flat_idx = idx_orig.get(&[0]) as usize;
            *x.as_ptr().add(flat_idx)
        };
        let min_val_view = unsafe {
            let flat_idx = idx_view.get(&[0]) as usize;
            let dims = x_t.shape().dims.clone();
            let mut coords = vec![0; dims.len()];
            let mut tmp = flat_idx;
            for k in (0..dims.len()).rev() {
                coords[k] = tmp % dims[k];
                tmp /= dims[k];
            }
            let offset = x_t.shape().offset(&coords);
            *x_t.as_ptr().add(offset)
        };

        assert_eq!(min_val_orig, -4.0);
        assert_eq!(min_val_view, -4.0);

        // Test simpler consistency: argmin along specific dimensions
        let idx_dim0_orig = x.argmin_dim(0, false); // argmin along rows -> [4] (min of each column)
        let idx_dim1_trans = x_t.argmin_dim(1, false); // argmin along columns -> [4] (min of each row)

        // These should give the same results since we're reducing along corresponding dims
        assert_eq!(idx_dim0_orig.shape().dims, vec![4]);
        assert_eq!(idx_dim1_trans.shape().dims, vec![4]);

        // Original columns vs transposed rows should match
        assert_eq!(idx_dim0_orig.get(&[0]), 1.0); // col 0: min(5,3,6) = 3 at row 1
        assert_eq!(idx_dim0_orig.get(&[1]), 0.0); // col 1: min(-2,9,0) = -2 at row 0
        assert_eq!(idx_dim0_orig.get(&[2]), 1.0); // col 2: min(8,-4,2) = -4 at row 1
        assert_eq!(idx_dim0_orig.get(&[3]), 2.0); // col 3: min(1,7,-1) = -1 at row 2

        assert_eq!(idx_dim1_trans.get(&[0]), 1.0); // corresponds to col 0
        assert_eq!(idx_dim1_trans.get(&[1]), 0.0); // corresponds to col 1
        assert_eq!(idx_dim1_trans.get(&[2]), 1.0); // corresponds to col 2
        assert_eq!(idx_dim1_trans.get(&[3]), 2.0); // corresponds to col 3
    }

    // Keep the old basic tests for compatibility
    #[test]
    fn test_argmin_basic() {
        let x = Tensor::from_slice(&[3.0, -2.0, 5.0, -1.0], vec![4]).unwrap();
        let idx = x.argmin();
        unsafe {
            assert_eq!(*idx.as_ptr(), 1.0);
        }
    }

    #[test]
    fn test_argmin_dim() {
        let x = Tensor::from_slice(&[3.0, -2.0, 5.0, -1.0, 0.0, -3.0], vec![2, 3]).unwrap();
        let idx0 = x.argmin_dim(1, true);
        assert_eq!(idx0.shape().dims, vec![2, 1]);
        assert_eq!(idx0.get(&[0, 0]), 1.0);
        assert_eq!(idx0.get(&[1, 0]), 2.0);
    }
}
