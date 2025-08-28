use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Fill masked elements with a specified value
    ///
    /// This operation returns a copy of the input tensor where elements are replaced
    /// by the specified value wherever the corresponding boolean mask is true.
    /// Elements where the mask is false retain their original values from the input tensor.
    ///
    /// The masked_fill operation is commonly used in machine learning for operations
    /// like masking attention weights, zeroing out specific elements, and implementing
    /// dropout-like functionality.
    ///
    /// # Arguments
    ///
    /// * `mask` - Boolean array with the same length as the number of tensor elements
    /// * `value` - The value to fill masked positions with
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape as the input, where masked elements are
    /// replaced by `value` and unmasked elements retain their original values
    ///
    /// # Examples
    ///
    /// ## Basic Masked Fill
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create a 2x3 tensor: [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    /// let tensor = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]).unwrap();
    ///
    /// // Create a mask: [false, true, false, true, false, true]
    /// let mask = [false, true, false, true, false, true];
    /// let result = tensor.masked_fill(&mask, -1.0);
    ///
    /// // Result: [[0.0, -1.0, 2.0], [-1.0, 4.0, -1.0]]
    /// assert_eq!(result.shape().dims, vec![2, 3]);
    /// assert_eq!(result.get(&[0, 0]), 0.0);   // Unmasked
    /// assert_eq!(result.get(&[0, 1]), -1.0);  // Masked
    /// assert_eq!(result.get(&[0, 2]), 2.0);   // Unmasked
    /// assert_eq!(result.get(&[1, 0]), -1.0);  // Masked
    /// assert_eq!(result.get(&[1, 1]), 4.0);   // Unmasked
    /// assert_eq!(result.get(&[1, 2]), -1.0);  // Masked
    /// ```
    ///
    /// ## Masked Fill with Gradient Tracking
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], vec![2, 3]).unwrap()
    ///     .with_requires_grad();
    ///
    /// // Create a mask with some true values
    /// let mask = [false, true, false, true, false, false];
    /// let mut result = tensor.masked_fill(&mask, 5.0);
    ///
    /// // Compute gradients
    /// result.backward(None);
    /// let grad = tensor.grad_by_value().expect("gradient missing");
    ///
    /// // Gradients should be zero where mask is true, 1 elsewhere
    /// assert_eq!(grad.shape().dims, vec![2, 3]);
    /// assert!((grad.get(&[0, 0]) - 1.0).abs() < 1e-6);   // Unmasked: gradient flows
    /// assert!((grad.get(&[0, 1]) - 0.0).abs() < 1e-6);   // Masked: no gradient
    /// assert!((grad.get(&[0, 2]) - 1.0).abs() < 1e-6);   // Unmasked: gradient flows
    /// ```
    ///
    /// ## Zeroing Out Specific Elements
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create a tensor with some values
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    ///
    /// // Create a mask to zero out every other element
    /// let mask = [true, false, true, false, true, false];
    /// let result = tensor.masked_fill(&mask, 0.0);
    ///
    /// // Result: [[0.0, 2.0, 0.0], [4.0, 0.0, 6.0]]
    /// assert_eq!(result.get(&[0, 0]), 0.0);  // Zeroed
    /// assert_eq!(result.get(&[0, 1]), 2.0);  // Kept
    /// assert_eq!(result.get(&[0, 2]), 0.0);  // Zeroed
    /// assert_eq!(result.get(&[1, 0]), 4.0);  // Kept
    /// assert_eq!(result.get(&[1, 1]), 0.0);  // Zeroed
    /// assert_eq!(result.get(&[1, 2]), 6.0);  // Kept
    /// ```
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(n) where n is the number of elements in the tensor
    /// - **Memory Usage**: Creates a new tensor with the same size as the input
    /// - **Optimization**: Uses efficient stride-based iteration for non-contiguous tensors
    /// - **GradTrack Overhead**: Minimal overhead when gradient tracking is enabled
    /// - **Memory Layout**: Output tensor is always contiguous for optimal performance
    ///
    /// # Implementation Details
    ///
    /// The masked_fill operation works by:
    /// 1. Validating that the mask length equals the number of tensor elements
    /// 2. Creating a new contiguous output tensor with the same shape
    /// 3. Iterating through all elements in logical order
    /// 4. For each element, checking the corresponding mask value:
    ///    - If mask is true: use the fill value
    ///    - If mask is false: copy the original value from input tensor
    /// 5. Computing source offsets using the input tensor's shape for non-contiguous tensors
    /// 6. Registering the operation for gradient computation if needed
    ///
    /// # Safety
    ///
    /// This function performs bounds checking to ensure:
    /// - The mask length equals the number of tensor elements
    /// - Memory access is safe through proper offset calculations
    /// - The operation handles both contiguous and non-contiguous tensors correctly
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - The mask length does not equal the number of tensor elements
    ///
    /// # Thread Safety
    ///
    /// This function is thread-safe and can be called concurrently on different tensors.
    /// The operation does not modify the input tensor and creates a new output tensor.
    ///
    /// # GradTrack Behavior
    ///
    /// When gradient tracking is enabled:
    /// - Gradients do not flow through masked positions (they are zeroed)
    /// - Gradients flow normally through unmasked positions
    /// - This behavior is useful for implementing operations like dropout
    pub fn masked_fill(&self, mask: &[bool], value: f32) -> Tensor {
        let numel = self.size();
        assert_eq!(
            mask.len(),
            numel,
            "mask length {} must equal tensor elements {}",
            mask.len(),
            numel
        );

        // Output is a contiguous copy with applied mask
        let mut output = Tensor::new(self.shape().dims.clone());

        // Iterate in logical order using strides if needed
        let rank = self.shape().rank();
        let mut coords = vec![0usize; rank];
        for (lin, &m) in mask.iter().enumerate().take(numel) {
            // Decode logical coords for mask mapping
            let mut tmp = lin;
            for i in (0..rank).rev() {
                let s = self.shape().dims[i];
                coords[i] = if s == 0 { 0 } else { tmp % s };
                tmp /= s;
            }
            let src_off = self.shape().offset(&coords);
            unsafe {
                *output.as_mut_ptr().add(lin) = if m {
                    value
                } else {
                    *self.as_ptr().add(src_off)
                };
            }
        }

        if self.requires_grad() {
            output.set_requires_grad(true);
            let grad_fn = GradFn::MaskedFill {
                mask: mask.to_vec(),
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
    fn test_masked_fill_basic() {
        let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]).unwrap();
        let mask = vec![false, true, false, true, false, true];
        let y = x.masked_fill(&mask, -1.0);
        assert_eq!(y.shape().dims, vec![2, 3]);
        assert_eq!(y.get(&[0, 0]), 0.0);
        assert_eq!(y.get(&[0, 1]), -1.0);
        assert_eq!(y.get(&[1, 0]), -1.0);
    }

    #[test]
    fn test_masked_fill_gradients() {
        let x = Tensor::from_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let mask = vec![false, true, false, true, false, false];
        let mut y = x.masked_fill(&mask, 5.0);
        y.backward(None);
        let gx = x.grad_by_value().expect("grad missing");
        // Grad should be zero where mask is true, 1 elsewhere (from upstream ones)
        for (i, &m) in mask.iter().enumerate().take(6) {
            let expected = if m { 0.0 } else { 1.0 };
            assert!((gx.get(&[i / 3, i % 3]) - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_masked_fill_coordinate_decoding_bug_fix() {
        // This test specifically targets the coordinate decoding bug that was fixed
        // Create a 3D tensor to test multi-dimensional coordinate mapping
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // [0, 0, :]
            5.0, 6.0, 7.0, 8.0, // [0, 1, :]
            9.0, 10.0, 11.0, 12.0, // [1, 0, :]
            13.0, 14.0, 15.0, 16.0, // [1, 1, :]
        ];
        let tensor = Tensor::from_slice(&data, vec![2, 2, 4]).unwrap();

        // Create a mask that selects specific positions
        // Position 0: [0,0,0] -> 1.0, Position 5: [0,1,1] -> 6.0, Position 10: [1,0,2] -> 11.0
        let mut mask = vec![false; 16];
        mask[0] = true; // [0,0,0]
        mask[5] = true; // [0,1,1]
        mask[10] = true; // [1,0,2]

        let result = tensor.masked_fill(&mask, 99.0);

        // Verify the correct positions were masked
        assert_eq!(result.get(&[0, 0, 0]), 99.0); // position 0, was 1.0
        assert_eq!(result.get(&[0, 1, 1]), 99.0); // position 5, was 6.0
        assert_eq!(result.get(&[1, 0, 2]), 99.0); // position 10, was 11.0

        // Verify unmasked positions remain unchanged
        assert_eq!(result.get(&[0, 0, 1]), 2.0); // position 1, unchanged
        assert_eq!(result.get(&[0, 1, 0]), 5.0); // position 4, unchanged
        assert_eq!(result.get(&[1, 1, 3]), 16.0); // position 15, unchanged
    }
}
