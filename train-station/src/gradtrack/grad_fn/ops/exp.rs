use crate::tensor::core::Tensor;

pub(crate) fn apply_exp(saved_output: &Tensor, grad_output: &Tensor) -> Vec<Option<Tensor>> {
    let grad_input = grad_output.mul_tensor_optimized(saved_output);
    vec![Some(grad_input)]
}
