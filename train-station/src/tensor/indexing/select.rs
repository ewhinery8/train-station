use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Select a slice along a given dimension at a specific index
    ///
    /// This operation extracts a slice from the input tensor by fixing a specific dimension
    /// at a given index. The result is a tensor with one fewer dimension than the input,
    /// containing the selected slice.
    ///
    /// The select operation returns a view (zero-copy) when the base offset is zero,
    /// otherwise it creates a contiguous copy to ensure correctness. This operation is
    /// commonly used for extracting specific rows, columns, or slices from tensors.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to select (must be < tensor rank)
    /// * `index` - The index within the specified dimension to select (must be < dim size)
    ///
    /// # Returns
    ///
    /// A tensor with the selected slice. The result has the same shape as the input
    /// except with the specified dimension removed.
    ///
    /// # Examples
    ///
    /// ## Basic Row Selection
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create a 2x3 tensor: [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    /// let tensor = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]).unwrap();
    ///
    /// // Select row 1 (dimension 0, index 1)
    /// let result = tensor.select(0, 1);
    ///
    /// // Result shape is [3] (dimension 0 removed)
    /// assert_eq!(result.shape().dims, vec![3]);
    /// assert_eq!(result.get(&[0]), 3.0);  // First element of row 1
    /// assert_eq!(result.get(&[1]), 4.0);  // Second element of row 1
    /// assert_eq!(result.get(&[2]), 5.0);  // Third element of row 1
    /// ```
    ///
    /// ## Column Selection
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create a 2x3 tensor: [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    /// let tensor = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]).unwrap();
    ///
    /// // Select column 1 (dimension 1, index 1)
    /// let result = tensor.select(1, 1);
    ///
    /// // Result shape is [2] (dimension 1 removed)
    /// assert_eq!(result.shape().dims, vec![2]);
    /// assert_eq!(result.get(&[0]), 1.0);  // Column 1, row 0
    /// assert_eq!(result.get(&[1]), 4.0);  // Column 1, row 1
    /// ```
    ///
    /// ## Select with Gradient Tracking
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap()
    ///     .with_requires_grad();
    ///
    /// // Select row 1 with gradient tracking enabled
    /// let mut result = tensor.select(0, 1);
    /// result.backward(None);
    ///
    /// // Verify gradients are computed correctly
    /// let grad = tensor.grad_by_value().expect("gradient missing");
    /// assert_eq!(grad.shape().dims, vec![2, 2]);
    /// // Only row 1 receives gradients
    /// assert_eq!(grad.get(&[0, 0]), 0.0);  // Row 0: no gradient
    /// assert_eq!(grad.get(&[0, 1]), 0.0);  // Row 0: no gradient
    /// assert_eq!(grad.get(&[1, 0]), 1.0);  // Row 1: gradient flows
    /// assert_eq!(grad.get(&[1, 1]), 1.0);  // Row 1: gradient flows
    /// ```
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(n) where n is the number of elements in the selected slice
    /// - **Memory Usage**: Zero-copy view when base offset is zero, otherwise creates a copy
    /// - **Optimization**: Uses efficient stride-based access for non-contiguous tensors
    /// - **GradTrack Overhead**: Minimal overhead when gradient tracking is enabled
    /// - **Memory Layout**: Result is contiguous when a copy is made, view otherwise
    ///
    /// # Implementation Details
    ///
    /// The select operation works by:
    /// 1. Validating the dimension and index bounds
    /// 2. Computing the new shape by removing the selected dimension
    /// 3. Computing the new strides by removing the selected dimension's stride
    /// 4. Calculating the base offset for the selected slice
    /// 5. If base offset is zero: creating a view with adjusted shape/strides
    /// 6. If base offset is non-zero: creating a contiguous copy of the slice
    /// 7. Registering the operation for gradient computation if needed
    ///
    /// # Safety
    ///
    /// This function performs comprehensive bounds checking to ensure:
    /// - The tensor has non-zero rank
    /// - The specified dimension is within the tensor's rank
    /// - The index is within bounds for the specified dimension
    /// - Memory access is safe through proper offset calculations
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - The tensor has zero rank
    /// - `dim` is greater than or equal to the tensor's rank
    /// - `index` is greater than or equal to the size of the specified dimension
    ///
    /// # Thread Safety
    ///
    /// This function is thread-safe and can be called concurrently on different tensors.
    /// The operation does not modify the input tensor and creates either a view or a new tensor.
    ///
    /// # View vs Copy Behavior
    ///
    /// - **View (zero-copy)**: When the base offset is zero, returns a view that shares
    ///   the same memory as the input tensor with adjusted shape and strides
    /// - **Copy**: When the base offset is non-zero, creates a contiguous copy to ensure
    ///   correctness across all operations
    ///
    /// # GradTrack Behavior
    ///
    /// When gradient tracking is enabled:
    /// - Gradients are scattered back to the selected slice in the input tensor
    /// - Other positions in the input tensor receive zero gradients
    /// - This behavior ensures correct gradient flow for the selected elements
    pub fn select(&self, dim: usize, index: usize) -> Tensor {
        let rank = self.shape().rank();
        assert!(rank > 0, "select requires non-zero rank");
        assert!(
            dim < rank,
            "select dim {} out of bounds for rank {}",
            dim,
            rank
        );
        let dim_size = self.shape().dims[dim];
        assert!(
            index < dim_size,
            "select index {} out of bounds for dimension {} (size {})",
            index,
            dim,
            dim_size
        );

        // Build new dims/strides removing the selected dimension
        let mut new_dims = Vec::with_capacity(rank - 1);
        let mut new_strides = Vec::with_capacity(rank - 1);
        for i in 0..rank {
            if i == dim {
                continue;
            }
            new_dims.push(self.shape().dims[i]);
            new_strides.push(self.strides()[i]);
        }

        // Base pointer shift by index * stride(dim)
        let base_offset = index * self.stride(dim);

        // Create a view with the same data pointer offset by base_offset
        // We simulate pointer offset by using memory_offset in access; to preserve zero-copy
        // semantics, we create a view over the same allocation and adjust shape/strides.
        let view_shape = crate::tensor::Shape::as_view(new_dims, new_strides);
        let mut result = self.create_view_with_shape(view_shape);

        // To account for base offset, rebase the `result` by materializing a small view window
        // via contiguous() if base_offset != 0 for non-view correctness. For simplicity and
        // correctness across all ops, create a contiguous copy if the base offset is non-zero.
        if base_offset != 0 {
            // Materialize contiguous slice
            let mut contiguous = Tensor::new(result.shape().dims.clone());
            // Copy elements from self using stride-aware reads starting at base_offset
            let numel = contiguous.size();
            let rank2 = result.shape().rank();
            let mut coords = vec![0usize; rank2];
            for lin in 0..numel {
                // Decode coords in result space
                let mut tmp = lin;
                for i in (0..rank2).rev() {
                    let s = result.shape().dims[i];
                    coords[i] = if s == 0 { 0 } else { tmp % s };
                    if s != 0 {
                        tmp /= s;
                    }
                }
                // Map to source coords inserting fixed index at dim
                let mut src_coords = Vec::with_capacity(rank);
                for i in 0..rank {
                    if i == dim {
                        src_coords.push(index);
                    } else {
                        let j = if i < dim { i } else { i - 1 };
                        src_coords.push(coords[j]);
                    }
                }
                let src_off = self.shape().offset(&src_coords);
                unsafe {
                    *contiguous.as_mut_ptr().add(lin) = *self.as_ptr().add(src_off);
                }
            }
            result = contiguous;
        }

        // GradTrack registration: backward scatters grad_output into zeros at the selected slice
        if self.requires_grad() {
            result.set_requires_grad(true);
            let grad_fn = GradFn::Select {
                dim,
                index,
                input_shape: self.shape().dims.clone(),
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_basic() {
        let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]).unwrap();
        let s = x.select(0, 1);
        assert_eq!(s.shape().dims, vec![3]);
        assert_eq!(s.get(&[0]), 3.0);
        assert_eq!(s.get(&[2]), 5.0);
    }

    #[test]
    fn test_select_grad() {
        let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], vec![2, 2])
            .unwrap()
            .with_requires_grad();
        let mut s = x.select(0, 1);
        s.backward(None);
        let gx = x.grad_by_value().expect("grad missing");
        // Only row 1 receives ones
        assert_eq!(gx.get(&[0, 0]), 0.0);
        assert_eq!(gx.get(&[0, 1]), 0.0);
        assert_eq!(gx.get(&[1, 0]), 1.0);
        assert_eq!(gx.get(&[1, 1]), 1.0);
    }
}
