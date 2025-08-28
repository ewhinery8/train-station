use super::super::utils::reduce_gradient_to_shape;
use crate::tensor::core::Tensor;

pub(crate) fn apply_sub(
    is_tensor_sub: bool,
    original_shapes: Option<&(Vec<usize>, Vec<usize>)>,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    if !is_tensor_sub {
        // Scalar subtraction: only first input gets gradient
        return vec![Some(grad_output.clone())];
    }

    match original_shapes {
        Some((shape_a, shape_b)) => {
            // Broadcasting case: reduce gradients to original shapes
            let grad_a = reduce_gradient_to_shape(grad_output, shape_a);
            let grad_b = reduce_gradient_to_shape(grad_output, shape_b);

            // For subtraction: d/da (a - b) = 1, d/db (a - b) = -1
            let mut neg_grad_b = grad_b;
            neg_grad_b.negate_inplace();

            vec![Some(grad_a), Some(neg_grad_b)]
        }
        None => {
            // Same shape case: no broadcasting needed
            let mut neg_grad = grad_output.clone();
            neg_grad.negate_inplace();
            vec![Some(grad_output.clone()), Some(neg_grad)]
        }
    }
}
