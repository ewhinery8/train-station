use crate::tensor::core::Tensor;

/// Apply gradient for slice view operation
///
/// Slice view gradient accumulation works by placing the gradient from the slice
/// back into the corresponding positions in the source tensor gradient.
///
/// For contiguous slices (step=1), gradients are placed consecutively.
/// For strided slices (step>1), gradients are placed at stepped intervals.
pub(crate) fn apply_slice_view(
    start: usize,
    step: usize,
    length: usize,
    input_shape: &[usize],
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    // Create gradient tensor for the input with same shape, initialized to zeros
    let mut grad_input = Tensor::zeros(input_shape.to_vec());

    // Accumulate gradients from the slice back to the source positions
    let grad_data = grad_output.data();

    unsafe {
        let input_grad_ptr = grad_input.as_mut_ptr();

        // For each element in the slice gradient, place it at the correct position
        // in the input gradient tensor
        for i in 0..length {
            let slice_idx = i;
            let input_idx = start + i * step;

            // Bounds check to prevent out-of-bounds access
            if input_idx < input_shape.iter().product() {
                *input_grad_ptr.add(input_idx) += grad_data[slice_idx];
            }
        }
    }

    vec![Some(grad_input)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_slice_gradient() {
        // Test contiguous slice gradient (step=1)
        let input_shape = vec![6];
        let start = 1;
        let step = 1;
        let length = 3;

        // Create gradient from slice (ones)
        let grad_output = Tensor::ones(vec![length]);

        // Apply slice gradient
        let grads = apply_slice_view(start, step, length, &input_shape, &grad_output);

        assert_eq!(grads.len(), 1);
        let grad_input = grads[0].as_ref().unwrap();

        // Check that gradient is placed correctly
        assert_eq!(grad_input.shape().dims, vec![6]);
        assert_eq!(grad_input.get(&[0]), 0.0); // Before slice
        assert_eq!(grad_input.get(&[1]), 1.0); // Slice start
        assert_eq!(grad_input.get(&[2]), 1.0); // Slice middle
        assert_eq!(grad_input.get(&[3]), 1.0); // Slice end
        assert_eq!(grad_input.get(&[4]), 0.0); // After slice
        assert_eq!(grad_input.get(&[5]), 0.0); // After slice
    }

    #[test]
    fn test_strided_slice_gradient() {
        // Test strided slice gradient (step=2)
        let input_shape = vec![8];
        let start = 1;
        let step = 2;
        let length = 3;

        // Create gradient from slice (twos)
        let mut grad_output = Tensor::zeros(vec![length]);
        grad_output.data_mut().fill(2.0);

        // Apply slice gradient
        let grads = apply_slice_view(start, step, length, &input_shape, &grad_output);

        assert_eq!(grads.len(), 1);
        let grad_input = grads[0].as_ref().unwrap();

        // Check that gradient is placed at strided positions
        assert_eq!(grad_input.shape().dims, vec![8]);
        assert_eq!(grad_input.get(&[0]), 0.0); // Not in slice
        assert_eq!(grad_input.get(&[1]), 2.0); // start + 0*step = 1
        assert_eq!(grad_input.get(&[2]), 0.0); // Not in slice
        assert_eq!(grad_input.get(&[3]), 2.0); // start + 1*step = 3
        assert_eq!(grad_input.get(&[4]), 0.0); // Not in slice
        assert_eq!(grad_input.get(&[5]), 2.0); // start + 2*step = 5
        assert_eq!(grad_input.get(&[6]), 0.0); // Not in slice
        assert_eq!(grad_input.get(&[7]), 0.0); // Not in slice
    }

    #[test]
    fn test_slice_gradient_accumulation() {
        // Test that gradients accumulate correctly when called multiple times
        let input_shape = vec![4];
        let start = 0;
        let step = 1;
        let length = 3;

        // Create gradient from slice
        let grad_output = Tensor::ones(vec![length]);

        // Apply gradient twice to test accumulation
        let grads1 = apply_slice_view(start, step, length, &input_shape, &grad_output);
        let grads2 = apply_slice_view(start, step, length, &input_shape, &grad_output);

        // Each should have gradient 1.0 at slice positions
        let grad1 = grads1[0].as_ref().unwrap();
        let grad2 = grads2[0].as_ref().unwrap();

        assert_eq!(grad1.get(&[0]), 1.0);
        assert_eq!(grad1.get(&[1]), 1.0);
        assert_eq!(grad1.get(&[2]), 1.0);
        assert_eq!(grad1.get(&[3]), 0.0);

        assert_eq!(grad2.get(&[0]), 1.0);
        assert_eq!(grad2.get(&[1]), 1.0);
        assert_eq!(grad2.get(&[2]), 1.0);
        assert_eq!(grad2.get(&[3]), 0.0);
    }
}
