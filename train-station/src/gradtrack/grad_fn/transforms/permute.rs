use crate::tensor::core::Tensor;

pub(crate) fn apply_permute(
    dims: &[usize],
    input_shape: &[usize],
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let rank = dims.len();
    let mut inv = vec![0usize; rank];
    for (i, &d) in dims.iter().enumerate() {
        inv[d] = i;
    }

    let grad_input = grad_output.permute(inv);
    assert_eq!(grad_input.shape().dims, input_shape.to_vec());
    vec![Some(grad_input)]
}
