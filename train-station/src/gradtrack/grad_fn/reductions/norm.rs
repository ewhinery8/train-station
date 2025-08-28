use crate::tensor::core::Tensor;

// Backward for L2 norm over all elements: n = sqrt(sum x^2), dn/dx_i = x_i / n (n>0)
pub(crate) fn apply_reduce_norm(
    input_shape: &[usize],
    saved_norm: &Tensor,
    saved_input: &Tensor,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let n = input_shape.iter().product::<usize>();
    let norm = unsafe { *saved_norm.as_ptr() };
    let upstream = if grad_output.size() == 1 {
        unsafe { *grad_output.as_ptr() }
    } else {
        1.0
    };
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    if norm > 0.0 {
        let scale = upstream / norm;
        unsafe {
            let x = saved_input.as_ptr();
            let g = grad_input.as_mut_ptr();
            for i in 0..n {
                *g.add(i) = *x.add(i) * scale;
            }
        }
    }
    vec![Some(grad_input)]
}

// Backward for L2 norm over dims: dn/dx = x / n (broadcasted per-slice)
pub(crate) fn apply_reduce_norm_dims(
    dims: &[usize],
    keepdim: bool,
    input_shape: &[usize],
    saved_norm: &Tensor,
    saved_input: &Tensor,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let rank = input_shape.len();
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    let mut reduced = dims.to_vec();
    reduced.sort_unstable();
    reduced.dedup();
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
            saved_norm.shape().offset(&out_coords)
        };
        let nval = unsafe { *saved_norm.as_ptr().add(out_off) };
        let upstream = if grad_output.size() == 1 {
            unsafe { *grad_output.as_ptr() }
        } else {
            unsafe { *grad_output.as_ptr().add(out_off) }
        };
        let x = unsafe { *saved_input.as_ptr().add(lin) };
        let g = if nval > 0.0 {
            x * (upstream / nval)
        } else {
            0.0
        };
        unsafe {
            *grad_input.as_mut_ptr().add(lin) = g;
        }
    }
    vec![Some(grad_input)]
}
