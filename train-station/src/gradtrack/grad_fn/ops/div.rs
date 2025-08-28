use super::super::utils::reduce_gradient_to_shape;
use crate::tensor::core::Tensor;

pub(crate) fn apply_div(
    is_tensor_div: bool,
    scalar: Option<f32>,
    operands: Option<&Vec<Tensor>>,
    original_shapes: Option<&(Vec<usize>, Vec<usize>)>,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    if !is_tensor_div {
        // Scalar division: only first input gets gradient
        if let Some(s) = scalar {
            let scaled_grad = grad_output.div_scalar_optimized(s);
            return vec![Some(scaled_grad)];
        } else {
            return vec![Some(grad_output.clone())];
        }
    }

    // Tensor division: d/da (a / b) = 1/b * grad_output, d/db (a / b) = -a/b^2 * grad_output
    if let Some(ops) = operands {
        if ops.len() == 2 {
            let grad_a_full = grad_output.div_tensor(&ops[1]);
            let neg_grad = grad_output.mul_tensor(&ops[0]);
            let b_squared = ops[1].mul_tensor(&ops[1]);
            let grad_b_full = neg_grad.div_tensor(&b_squared);
            let mut final_grad_b_full = grad_b_full;
            final_grad_b_full.negate_inplace();

            match original_shapes {
                Some((shape_a, shape_b)) => {
                    // Broadcasting case: reduce gradients to original shapes
                    let grad_a = reduce_gradient_to_shape(&grad_a_full, shape_a);
                    let grad_b = reduce_gradient_to_shape(&final_grad_b_full, shape_b);
                    vec![Some(grad_a), Some(grad_b)]
                }
                None => {
                    // Same shape case
                    vec![Some(grad_a_full), Some(final_grad_b_full)]
                }
            }
        } else {
            vec![Some(grad_output.clone()), Some(grad_output.clone())]
        }
    } else {
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}
