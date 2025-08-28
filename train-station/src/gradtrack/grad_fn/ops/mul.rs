use super::super::utils::reduce_gradient_to_shape;
use crate::tensor::core::Tensor;

pub(crate) fn apply_mul(
    is_tensor_mul: bool,
    scalar: Option<f32>,
    operands: Option<&Vec<Tensor>>,
    original_shapes: Option<&(Vec<usize>, Vec<usize>)>,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    if !is_tensor_mul {
        // Scalar multiplication: only first input gets gradient scaled by scalar
        if let Some(s) = scalar {
            let scaled_grad = grad_output.mul_scalar_optimized(s);
            return vec![Some(scaled_grad)];
        } else {
            return vec![Some(grad_output.clone())];
        }
    }

    // Tensor multiplication: d/da (a * b) = b * grad_output, d/db (a * b) = a * grad_output
    if let Some(ops) = operands {
        if ops.len() == 2 {
            match original_shapes {
                Some((shape_a, shape_b)) => {
                    // Broadcasting case: need to broadcast operands to gradient shape first
                    let (broadcast_a, broadcast_b, _) = ops[0].broadcast_with(&ops[1]).unwrap();
                    let grad_a_full = grad_output.mul_tensor_optimized(&broadcast_b);
                    let grad_b_full = grad_output.mul_tensor_optimized(&broadcast_a);

                    // Then reduce gradients to original shapes
                    let grad_a = reduce_gradient_to_shape(&grad_a_full, shape_a);
                    let grad_b = reduce_gradient_to_shape(&grad_b_full, shape_b);
                    vec![Some(grad_a), Some(grad_b)]
                }
                None => {
                    // Same shape case: operands already have correct shape
                    let grad_a_full = grad_output.mul_tensor_optimized(&ops[1]);
                    let grad_b_full = grad_output.mul_tensor_optimized(&ops[0]);
                    vec![Some(grad_a_full), Some(grad_b_full)]
                }
            }
        } else {
            vec![Some(grad_output.clone()), Some(grad_output.clone())]
        }
    } else {
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}
