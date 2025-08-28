use crate::tensor::core::Tensor;

/// Backward for min over all elements: split upstream grad equally among minima
pub(crate) fn apply_reduce_min(
    input_shape: &[usize],
    saved_output: &Tensor,
    saved_input: &Tensor,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    let min_val = unsafe { *saved_output.as_ptr() };
    let mut count = 0usize;
    unsafe {
        for i in 0..saved_input.size() {
            if *saved_input.as_ptr().add(i) == min_val {
                count += 1;
            }
        }
    }
    let upstream = if grad_output.size() == 1 {
        unsafe { *grad_output.as_ptr() }
    } else {
        1.0
    };
    let per_min = if count > 0 {
        upstream / (count as f32)
    } else {
        0.0
    };
    unsafe {
        let dst = grad_input.as_mut_ptr();
        for i in 0..grad_input.size() {
            if *saved_input.as_ptr().add(i) == min_val {
                *dst.add(i) = per_min;
            }
        }
    }
    vec![Some(grad_input)]
}

/// Backward for min over dims: split upstream per-slice grad equally among slice minima
pub(crate) fn apply_reduce_min_dims(
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
    let mut min_counts = Tensor::zeros(saved_output.shape().dims.to_vec());
    // First pass: count
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
                *min_counts.as_mut_ptr().add(out_off) += 1.0;
            }
        }
    }
    // Second pass: distribute
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
            let count = unsafe { *min_counts.as_ptr().add(out_off) };
            let upstream = if grad_output.size() == 1 {
                unsafe { *grad_output.as_ptr() }
            } else {
                unsafe { *grad_output.as_ptr().add(out_off) }
            };
            let per_min = if count > 0.0 { upstream / count } else { 0.0 };
            unsafe {
                *grad_input.as_mut_ptr().add(lin) = per_min;
            }
        }
    }
    vec![Some(grad_input)]
}
