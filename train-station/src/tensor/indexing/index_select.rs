use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Select elements along a dimension using a list of indices
    ///
    /// This operation extracts elements from the input tensor along a specified dimension
    /// using the provided indices. The output tensor has the same shape as the input
    /// except along the specified dimension, where the size becomes the length of the
    /// indices array.
    ///
    /// The index_select operation is commonly used for extracting specific rows, columns,
    /// or slices from tensors, and is particularly useful in machine learning for
    /// operations like embedding lookups and attention mechanisms.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to select elements (must be < tensor rank)
    /// * `indices` - Array of indices specifying which elements to select along `dim`
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape as the input except along `dim`, where the
    /// size is `indices.len()`
    ///
    /// # Examples
    ///
    /// ## Basic Index Selection
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create a 2x3 tensor: [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    /// let tensor = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]).unwrap();
    ///
    /// // Select columns 2 and 0 from dimension 1
    /// let result = tensor.index_select(1, &[2, 0]);
    ///
    /// // Result shape is [2, 2] (same as input except dim 1 is now 2)
    /// assert_eq!(result.shape().dims, vec![2, 2]);
    ///
    /// // Row 0: selected columns [2, 0] -> [2.0, 0.0]
    /// assert_eq!(result.get(&[0, 0]), 2.0);
    /// assert_eq!(result.get(&[0, 1]), 0.0);
    ///
    /// // Row 1: selected columns [2, 0] -> [5.0, 3.0]
    /// assert_eq!(result.get(&[1, 0]), 5.0);
    /// assert_eq!(result.get(&[1, 1]), 3.0);
    /// ```
    ///
    /// ## Index Selection with Gradient Tracking
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap()
    ///     .with_requires_grad();
    ///
    /// // Select specific elements with gradient tracking enabled
    /// let mut result = tensor.index_select(1, &[1, 2]);
    /// result.backward(None);
    ///
    /// // Verify gradients are computed correctly
    /// let grad = tensor.grad_by_value().expect("gradient missing");
    /// assert_eq!(grad.shape().dims, vec![2, 3]);
    /// ```
    ///
    /// ## Selecting Rows from a Matrix
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create a 3x2 matrix
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
    ///
    /// // Select rows 2 and 0 (dimension 0)
    /// let result = tensor.index_select(0, &[2, 0]);
    ///
    /// // Result shape is [2, 2]
    /// assert_eq!(result.shape().dims, vec![2, 2]);
    ///
    /// // Selected rows: row 2 [5.0, 6.0], row 0 [1.0, 2.0]
    /// assert_eq!(result.get(&[0, 0]), 5.0); // First row of result (was row 2)
    /// assert_eq!(result.get(&[0, 1]), 6.0);
    /// assert_eq!(result.get(&[1, 0]), 1.0); // Second row of result (was row 0)
    /// assert_eq!(result.get(&[1, 1]), 2.0);
    /// ```
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(n) where n is the number of elements in the output tensor
    /// - **Memory Usage**: Creates a new tensor with size equal to the output shape
    /// - **Optimization**: Uses precomputed strides for efficient memory access
    /// - **GradTrack Overhead**: Minimal overhead when gradient tracking is enabled
    /// - **Memory Layout**: Output tensor is always contiguous for optimal performance
    ///
    /// # Implementation Details
    ///
    /// The index_select operation works by:
    /// 1. Validating the dimension and index bounds
    /// 2. Computing the output shape (same as input except along `dim`)
    /// 3. Creating a new contiguous output tensor
    /// 4. Iterating through all positions in the output tensor using nested loops:
    ///    - Outer loop: iterate over dimensions before `dim`
    ///    - Middle loop: iterate over the selected indices
    ///    - Inner loop: iterate over dimensions after `dim`
    /// 5. Computing source offsets using the input tensor's strides
    /// 6. Copying values from input to output tensor
    /// 7. Registering the operation for gradient computation if needed
    ///
    /// # Safety
    ///
    /// This function performs comprehensive bounds checking to ensure:
    /// - The specified dimension is within the tensor's rank
    /// - All indices are within bounds for the specified dimension
    /// - Memory access is safe through proper offset calculations
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - `dim` is greater than or equal to the tensor's rank
    /// - Any index in `indices` is out of bounds for the specified dimension
    ///
    /// # Thread Safety
    ///
    /// This function is thread-safe and can be called concurrently on different tensors.
    /// The operation does not modify the input tensor and creates a new output tensor.
    pub fn index_select(&self, dim: usize, indices: &[usize]) -> Tensor {
        let rank = self.shape().rank();
        assert!(
            dim < rank,
            "index_select dim {} out of bounds for rank {}",
            dim,
            rank
        );
        for &idx in indices {
            assert!(
                idx < self.shape().dims[dim],
                "index {} out of bounds for dimension {} (size {})",
                idx,
                dim,
                self.shape().dims[dim]
            );
        }

        // Output shape is same as input except along dim -> indices.len()
        let mut out_dims = self.shape().dims.clone();
        out_dims[dim] = indices.len();
        let mut output = Tensor::new(out_dims.clone());

        // Precompute strides for fast offset computation
        let in_strides = self.strides().to_vec();
        let out_inner: usize = out_dims[dim + 1..].iter().product();
        let out_outer: usize = out_dims[..dim].iter().product();

        unsafe {
            let dst_ptr = output.as_mut_ptr();
            for outer_idx in 0..out_outer {
                // Decode outer_idx into coordinates for axes < dim
                let mut coords = vec![0usize; rank];
                if dim > 0 {
                    let mut tmp = outer_idx;
                    for i in (0..dim).rev() {
                        let s = self.shape().dims[i];
                        coords[i] = tmp % s;
                        tmp /= s;
                    }
                }

                for (j, &sel) in indices.iter().enumerate() {
                    coords[dim] = sel;
                    // Iterate over inner block
                    for inner_idx in 0..out_inner {
                        // Decode inner_idx into coordinates for axes > dim
                        let mut tmp = inner_idx;
                        for (i, c) in coords.iter_mut().enumerate().take(rank).skip(dim + 1) {
                            let s = self.shape().dims[i];
                            *c = tmp % s;
                            tmp /= s;
                        }

                        // Compute input offset via strides
                        let mut src_off = 0usize;
                        for i in 0..rank {
                            src_off += coords[i] * in_strides[i];
                        }

                        // Destination offset within output tensor (contiguous)
                        let out_block =
                            outer_idx * (indices.len() * out_inner) + j * out_inner + inner_idx;
                        *dst_ptr.add(out_block) = *self.as_ptr().add(src_off);
                    }
                }
            }
        }

        // GradTrack registration
        if self.requires_grad() {
            output.set_requires_grad(true);
            let grad_fn = GradFn::IndexSelect {
                dim,
                indices: indices.to_vec(),
                input_shape: self.shape().dims.clone(),
            };
            output.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(output.id(), vec![self.id()], grad_fn);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_select_basic() {
        let x =
            Tensor::from_slice(&(0..6).map(|i| i as f32).collect::<Vec<_>>(), vec![2, 3]).unwrap();
        let y = x.index_select(1, &[2, 0]);
        assert_eq!(y.shape().dims, vec![2, 2]);
        assert_eq!(y.get(&[0, 0]), 2.0);
        assert_eq!(y.get(&[0, 1]), 0.0);
        assert_eq!(y.get(&[1, 0]), 5.0);
        assert_eq!(y.get(&[1, 1]), 3.0);
    }
}
