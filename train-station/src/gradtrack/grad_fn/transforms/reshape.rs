use crate::tensor::core::Tensor;

pub(crate) fn apply_reshape(original_shape: &[usize], grad_output: &Tensor) -> Vec<Option<Tensor>> {
    let grad_input = grad_output.reshape(original_shape.iter().map(|&d| d as i32).collect());
    vec![Some(grad_input)]
}
