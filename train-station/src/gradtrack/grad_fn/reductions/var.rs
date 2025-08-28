use crate::tensor::core::Tensor;

// Backward for population variance var = mean((x - mu)^2)
// dvar/dx_i = 2*(x_i - mu) / N
pub(crate) fn apply_reduce_var(
    input_shape: &[usize],
    saved_mean: &Tensor,
    saved_input: &Tensor,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let n = input_shape.iter().product::<usize>();
    let inv_n2 = if n > 0 { 2.0f32 / (n as f32) } else { 0.0 };
    let upstream = if grad_output.size() == 1 {
        unsafe { *grad_output.as_ptr() }
    } else {
        1.0
    };
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    unsafe {
        let mu = *saved_mean.as_ptr();
        let xptr = saved_input.as_ptr();
        let gptr = grad_input.as_mut_ptr();
        let scale = upstream * inv_n2;
        for i in 0..n {
            *gptr.add(i) = (*xptr.add(i) - mu) * scale;
        }
    }
    vec![Some(grad_input)]
}

pub(crate) fn apply_reduce_var_dims(
    dims: &[usize],
    keepdim: bool,
    input_shape: &[usize],
    saved_mean: &Tensor,
    saved_input: &Tensor,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let rank = input_shape.len();
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    let mut reduced = dims.to_vec();
    reduced.sort_unstable();
    reduced.dedup();
    let n_reduced: usize = reduced.iter().map(|&d| input_shape[d]).product();
    let inv_n2 = if n_reduced > 0 {
        2.0f32 / (n_reduced as f32)
    } else {
        0.0
    };
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
        let mut out_coords: Vec<usize> = Vec::new();
        for (i, &c) in coords.iter().enumerate().take(rank) {
            if reduced.contains(&i) {
                if keepdim {
                    out_coords.push(0);
                }
            } else {
                out_coords.push(c);
            }
        }
        let out_off = if out_coords.is_empty() {
            0
        } else {
            saved_mean.shape().offset(&out_coords)
        };
        let mu = unsafe { *saved_mean.as_ptr().add(out_off) };
        let upstream = if grad_output.size() == 1 {
            unsafe { *grad_output.as_ptr() }
        } else {
            unsafe { *grad_output.as_ptr().add(out_off) }
        };
        let x = unsafe { *saved_input.as_ptr().add(lin) };
        let g = (x - mu) * (upstream * inv_n2);
        unsafe {
            *grad_input.as_mut_ptr().add(lin) = g;
        }
    }
    vec![Some(grad_input)]
}
