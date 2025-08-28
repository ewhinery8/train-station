use crate::tensor::core::Tensor;

/// Backward for mean reduction over all elements: dL/dx = (1/numel) * ones_like(x) * dL/dy
pub(crate) fn apply_reduce_mean(
    input_shape: &[usize],
    numel: usize,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let mut grad_input = Tensor::ones(input_shape.to_vec());
    let scale_base = if numel > 0 {
        1.0f32 / (numel as f32)
    } else {
        0.0
    };
    unsafe {
        let dst = grad_input.as_mut_ptr();
        if grad_output.size() == 1 {
            let upstream = *grad_output.as_ptr();
            let scale = scale_base * upstream;
            for i in 0..grad_input.size() {
                *dst.add(i) = scale;
            }
        } else {
            for i in 0..grad_input.size() {
                *dst.add(i) = scale_base;
            }
        }
    }
    vec![Some(grad_input)]
}

/// Backward for mean over dims: like sum but scaled by 1/prod(reduced sizes)
pub(crate) fn apply_reduce_mean_dims(
    dims: &[usize],
    input_shape: &[usize],
    keepdim: bool,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let mut grad = super::sum::apply_reduce_sum_dims(dims, input_shape, keepdim, grad_output)
        .remove(0)
        .unwrap();
    let factor: usize = dims.iter().map(|&d| input_shape[d]).product();
    let scale = if factor > 0 {
        1.0f32 / (factor as f32)
    } else {
        0.0
    };
    unsafe {
        let ptr = grad.as_mut_ptr();
        for i in 0..grad.size() {
            *ptr.add(i) *= scale;
        }
    }
    vec![Some(grad)]
}
