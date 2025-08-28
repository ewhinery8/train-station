use crate::tensor::core::Tensor;

/// Backward for max over all elements: split upstream grad equally among maxima
pub(crate) fn apply_reduce_max(
    input_shape: &[usize],
    saved_output: &Tensor,
    saved_input: &Tensor,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    let max_val = unsafe { *saved_output.as_ptr() };
    let mut count = 0usize;
    unsafe {
        for i in 0..saved_input.size() {
            if *saved_input.as_ptr().add(i) == max_val {
                count += 1;
            }
        }
    }
    let upstream = if grad_output.size() == 1 {
        unsafe { *grad_output.as_ptr() }
    } else {
        1.0
    };
    let per = if count > 0 {
        upstream / (count as f32)
    } else {
        0.0
    };
    unsafe {
        let dst = grad_input.as_mut_ptr();
        for i in 0..grad_input.size() {
            if *saved_input.as_ptr().add(i) == max_val {
                *dst.add(i) = per;
            }
        }
    }
    vec![Some(grad_input)]
}

/// Backward for max over dims: split upstream per-slice grad equally among slice maxima
pub(crate) fn apply_reduce_max_dims(
    dims: &[usize],
    keepdim: bool,
    input_shape: &[usize],
    saved_output: &Tensor,
    saved_input: &Tensor,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let rank = input_shape.len();
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    let mut reduced = dims.to_vec();
    reduced.sort_unstable();
    reduced.dedup();
    let mut coords = vec![0usize; rank];
    let mut max_counts = Tensor::zeros(saved_output.shape().dims.clone());
    // Count per output position
    for lin in 0..saved_input.size() {
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
            saved_output.shape().offset(&out_coords)
        };
        let out_val = unsafe { *saved_output.as_ptr().add(out_off) };
        let x = unsafe { *saved_input.as_ptr().add(lin) };
        if x == out_val {
            unsafe {
                *max_counts.as_mut_ptr().add(out_off) += 1.0;
            }
        }
    }
    // Distribute gradient
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
            saved_output.shape().offset(&out_coords)
        };
        let out_val = unsafe { *saved_output.as_ptr().add(out_off) };
        let x = unsafe { *saved_input.as_ptr().add(lin) };
        if x == out_val {
            let count = unsafe { *max_counts.as_ptr().add(out_off) };
            let upstream = if grad_output.size() == 1 {
                unsafe { *grad_output.as_ptr() }
            } else {
                unsafe { *grad_output.as_ptr().add(out_off) }
            };
            let per = if count > 0.0 { upstream / count } else { 0.0 };
            unsafe {
                *grad_input.as_mut_ptr().add(lin) = per;
            }
        }
    }
    vec![Some(grad_input)]
}
