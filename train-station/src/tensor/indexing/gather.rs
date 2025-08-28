use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Gather values along a dimension using a tensor of indices
    ///
    /// This operation extracts elements from the input tensor based on indices provided
    /// along a specified dimension. The output tensor has the same shape as the index
    /// tensor, with each element taken from the input tensor at the corresponding
    /// position with the index value substituted for the specified dimension.
    ///
    /// The gather operation is commonly used in machine learning for operations like
    /// embedding lookups, attention mechanisms, and advanced indexing patterns.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to gather values (must be < tensor rank)
    /// * `indices` - Flattened indices buffer containing the positions to gather from
    /// * `index_shape` - Shape of the indices tensor and output tensor
    ///
    /// # Returns
    ///
    /// A new tensor with shape `index_shape` containing the gathered values
    ///
    /// # Examples
    ///
    /// ## Basic Gather Operation
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create a 2x3 tensor: [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]
    /// let tensor = Tensor::from_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], vec![2, 3]).unwrap();
    ///
    /// // Gather along dimension 1 (columns) with indices [2, 0, 1, 1]
    /// let indices = [2, 0, 1, 1];
    /// let index_shape = [2, 2];
    /// let result = tensor.gather(1, &indices, &index_shape);
    ///
    /// // Result shape is [2, 2]
    /// assert_eq!(result.shape().dims, vec![2, 2]);
    ///
    /// // Row 0: indices [2, 0] -> [0.2, 0.0]
    /// assert!((result.get(&[0, 0]) - 0.2).abs() < 1e-6);
    /// assert!((result.get(&[0, 1]) - 0.0).abs() < 1e-6);
    ///
    /// // Row 1: indices [1, 1] -> [0.4, 0.4]
    /// assert!((result.get(&[1, 0]) - 0.4).abs() < 1e-6);
    /// assert!((result.get(&[1, 1]) - 0.4).abs() < 1e-6);
    /// ```
    ///
    /// ## Gather with Gradient Tracking
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], vec![2, 3]).unwrap()
    ///     .with_requires_grad();
    ///
    /// let indices = [1, 1, 0, 2];
    /// let index_shape = [2, 2];
    /// let mut result = tensor.gather(1, &indices, &index_shape);
    ///
    /// // Compute gradients
    /// result.backward(None);
    /// let grad = tensor.grad_by_value().expect("gradient missing");
    ///
    /// // Verify gradient accumulation for repeated indices
    /// assert!((grad.get(&[0, 1]) - 2.0).abs() < 1e-6); // Index 1 used twice in row 0
    /// ```
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(n) where n is the number of elements in the output
    /// - **Memory Usage**: Creates a new tensor with the same size as the index tensor
    /// - **Optimization**: Uses precomputed strides for efficient memory access
    /// - **GradTrack Overhead**: Minimal overhead when gradient tracking is enabled
    ///
    /// # Implementation Details
    ///
    /// The gather operation works by:
    /// 1. Validating input dimensions and index bounds
    /// 2. Creating an output tensor with the specified index shape
    /// 3. Iterating through all positions in the output tensor
    /// 4. Computing source offsets using the input tensor's strides
    /// 5. Copying values from the input tensor to the output tensor
    /// 6. Registering the operation for gradient computation if needed
    ///
    /// # Safety
    ///
    /// This function performs bounds checking to ensure:
    /// - The specified dimension is within the tensor's rank
    /// - All indices are within bounds for the specified dimension
    /// - The index shape is compatible with the input tensor shape
    /// - The indices buffer length matches the product of index shape dimensions
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - `dim` is greater than or equal to the tensor's rank
    /// - Any index in `indices` is out of bounds for the specified dimension
    /// - The `index_shape` rank doesn't match the input tensor's rank
    /// - The `index_shape` dimensions don't match the input tensor (except along `dim`)
    /// - The `indices` length doesn't equal the product of `index_shape` dimensions
    pub fn gather(&self, dim: usize, indices: &[usize], index_shape: &[usize]) -> Tensor {
        let rank = self.shape().rank();
        assert!(
            dim < rank,
            "gather dim {} out of bounds for rank {}",
            dim,
            rank
        );

        // Validate index_shape compatibility: same rank and all dims equal except along dim
        assert_eq!(
            index_shape.len(),
            rank,
            "index_shape rank mismatch: {} vs {}",
            index_shape.len(),
            rank
        );
        for (i, &s) in index_shape.iter().enumerate().take(rank) {
            if i != dim {
                assert_eq!(
                    s,
                    self.shape().dims[i],
                    "index_shape mismatch at dim {}: {} vs {}",
                    i,
                    s,
                    self.shape().dims[i]
                );
            }
        }

        let index_numel: usize = index_shape.iter().product();
        assert_eq!(
            indices.len(),
            index_numel,
            "indices length {} must equal product of index_shape {}",
            indices.len(),
            index_numel
        );

        // Validate indices range along dim
        let dim_size = self.shape().dims[dim];
        for &idx in indices.iter() {
            assert!(
                idx < dim_size,
                "gather index {} out of bounds for dim {} (size {})",
                idx,
                dim,
                dim_size
            );
        }

        // Output shape equals index_shape
        let mut output = Tensor::new(index_shape.to_vec());

        // Precompute input strides for fast offset computation
        let in_strides = self.strides().to_vec();

        // Iterate over all positions in output/index tensor
        let rank = index_shape.len();
        let mut coords = vec![0usize; rank];
        for (lin, &idx) in indices.iter().enumerate().take(index_numel) {
            // Decode linear index to multi-dimensional coords
            let mut tmp = lin;
            for i in (0..rank).rev() {
                let s = index_shape[i];
                coords[i] = tmp % s;
                tmp /= s;
            }

            let mut src_off = 0usize;
            for i in 0..rank {
                let c = if i == dim { idx } else { coords[i] };
                src_off += c * in_strides[i];
            }

            unsafe {
                *output.as_mut_ptr().add(lin) = *self.as_ptr().add(src_off);
            }
        }

        // GradTrack registration
        if self.requires_grad() {
            output.set_requires_grad(true);
            let grad_fn = GradFn::Gather {
                dim,
                indices: indices.to_vec(),
                input_shape: self.shape().dims.clone(),
                index_shape: index_shape.to_vec(),
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
    fn test_gather_basic() {
        // x shape [2,3]: [[0.0, 0.1, 0.2],[0.3,0.4,0.5]]
        let x = Tensor::from_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], vec![2, 3]).unwrap();
        let out = x.gather(1, &[2, 0, 1, 1], &[2, 2]);
        assert_eq!(out.shape().dims, vec![2, 2]);
        // Row 0 gathered indices [2,0] -> [0.2, 0.0]
        assert!((out.get(&[0, 0]) - 0.2).abs() < 1e-6);
        assert!((out.get(&[0, 1]) - 0.0).abs() < 1e-6);
        // Row 1 gathered indices [1,1] -> [0.4, 0.4]
        assert!((out.get(&[1, 0]) - 0.4).abs() < 1e-6);
        assert!((out.get(&[1, 1]) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_gather_gradients_accumulate() {
        // x shape [2,3], gather along dim=1 with repeated indices to test accumulation
        let x = Tensor::from_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let mut y = x.gather(1, &[1, 1, 0, 2], &[2, 2]);
        // Upstream gradient defaults to ones in our engine
        y.backward(None);
        let gx = x.grad_by_value().expect("grad missing");
        // Expected grad counts per input element:
        // For row 0: indices [1,1] -> input[0,1] gets +2
        // For row 1: indices [0,2] -> input[1,0] gets +1, input[1,2] gets +1
        assert_eq!(gx.shape().dims, vec![2, 3]);
        // Row 0
        assert!((gx.get(&[0, 0]) - 0.0).abs() < 1e-6);
        assert!((gx.get(&[0, 1]) - 2.0).abs() < 1e-6);
        assert!((gx.get(&[0, 2]) - 0.0).abs() < 1e-6);
        // Row 1
        assert!((gx.get(&[1, 0]) - 1.0).abs() < 1e-6);
        assert!((gx.get(&[1, 1]) - 0.0).abs() < 1e-6);
        assert!((gx.get(&[1, 2]) - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic]
    fn test_gather_invalid_dim() {
        let x = Tensor::zeros(vec![2, 3]);
        let _ = x.gather(2, &[0, 0], &[2, 1]);
    }

    #[test]
    #[should_panic]
    fn test_gather_index_shape_mismatch() {
        let x = Tensor::zeros(vec![2, 3]);
        // index_shape rank mismatch
        let _ = x.gather(1, &[0, 0], &[2]);
    }
}
