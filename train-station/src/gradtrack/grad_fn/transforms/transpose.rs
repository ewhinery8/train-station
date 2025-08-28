use crate::tensor::core::Tensor;

pub(crate) fn apply_transpose(
    dim0: usize,
    dim1: usize,
    input_shape: &[usize],
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    // To compute gradient for transpose, we need to apply the same transpose operation
    // to the gradient output, since transpose is its own inverse for two dimensions
    let grad_input = grad_output.transpose(dim0, dim1);
    assert_eq!(grad_input.shape().dims, input_shape.to_vec());
    vec![Some(grad_input)]
}
