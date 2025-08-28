use crate::tensor::core::Tensor;

pub(crate) fn apply_masked_fill(
    mask: &[bool],
    input_shape: &[usize],
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    let numel = grad_input.size();
    for (i, &m) in mask.iter().enumerate().take(numel) {
        let go = unsafe { *grad_output.as_ptr().add(i) };
        let v = if m { 0.0 } else { go };
        unsafe {
            *grad_input.as_mut_ptr().add(i) = v;
        }
    }
    vec![Some(grad_input)]
}
