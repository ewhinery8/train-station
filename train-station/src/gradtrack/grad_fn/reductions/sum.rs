use crate::tensor::core::Tensor;

/// Backward for sum reduction over all elements: dL/dx = ones_like(x) * dL/dy_scalar
pub(crate) fn apply_reduce_sum(input_shape: &[usize], grad_output: &Tensor) -> Vec<Option<Tensor>> {
    let mut grad_input = Tensor::ones(input_shape.to_vec());
    if grad_output.size() == 1 {
        unsafe {
            let scale = *grad_output.as_ptr();
            let dst = grad_input.as_mut_ptr();
            for i in 0..grad_input.size() {
                *dst.add(i) = scale;
            }
        }
    }
    vec![Some(grad_input)]
}

/// Backward for sum over dims: expands grad_output over reduced dims and broadcasts to input
pub(crate) fn apply_reduce_sum_dims(
    dims: &[usize],
    input_shape: &[usize],
    keepdim: bool,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let rank = input_shape.len();
    let mut grad_input = Tensor::zeros(input_shape.to_vec());

    let mut coords = vec![0usize; rank];
    for lin in 0..grad_input.size() {
        let mut tmp = lin;
        for i in (0..rank).rev() {
            let s = input_shape[i];
            coords[i] = if s == 0 { 0 } else { tmp % s };
            if s != 0 {
                tmp /= s;
            }
        }
        let mut go_coords: Vec<usize> = Vec::with_capacity(rank);
        for (i, &c) in coords.iter().enumerate().take(rank) {
            if dims.contains(&i) {
                if keepdim {
                    go_coords.push(0);
                }
            } else {
                go_coords.push(c);
            }
        }
        let go_off = if go_coords.is_empty() {
            0
        } else {
            grad_output.shape().offset(&go_coords)
        };
        unsafe {
            *grad_input.as_mut_ptr().add(lin) = *grad_output.as_ptr().add(go_off);
        }
    }
    vec![Some(grad_input)]
}
