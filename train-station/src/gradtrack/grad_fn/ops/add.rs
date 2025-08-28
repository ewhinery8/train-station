use super::super::utils::reduce_gradient_to_shape;
use crate::tensor::core::Tensor;

pub(crate) fn apply_add(
    is_tensor_add: bool,
    original_shapes: Option<&(Vec<usize>, Vec<usize>)>,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    if !is_tensor_add {
        // Scalar addition: only first input gets gradient
        return vec![Some(grad_output.clone())];
    }

    match original_shapes {
        Some((shape_a, shape_b)) => {
            // Broadcasting case: reduce gradients to original shapes
            let grad_a = reduce_gradient_to_shape(grad_output, shape_a);
            let grad_b = reduce_gradient_to_shape(grad_output, shape_b);
            vec![Some(grad_a), Some(grad_b)]
        }
        None => {
            // Same shape case: gradients are identical to output
            let grad_a = Some(grad_output.clone());
            let grad_b = Some(grad_output.clone());
            vec![grad_a, grad_b]
        }
    }
}
