use crate::tensor::core::Tensor;

pub(crate) fn apply_index_select(
    dim: usize,
    indices: &[usize],
    input_shape: &[usize],
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    let rank = input_shape.len();
    let out_dims = grad_output.shape().dims.to_vec();
    let inner: usize = out_dims[dim + 1..].iter().product();
    let outer: usize = out_dims[..dim].iter().product();

    unsafe {
        let go_ptr = grad_output.as_ptr();
        for outer_idx in 0..outer {
            let mut coords = vec![0usize; rank];
            if dim > 0 {
                let mut tmp = outer_idx;
                for i in (0..dim).rev() {
                    let s = input_shape[i];
                    coords[i] = tmp % s;
                    tmp /= s;
                }
            }
            for (j, &sel) in indices.iter().enumerate() {
                coords[dim] = sel;
                for inner_idx in 0..inner {
                    let mut tmp = inner_idx;
                    for i in (dim + 1)..rank {
                        let s = input_shape[i];
                        coords[i] = tmp % s;
                        tmp /= s;
                    }
                    let in_off = grad_input.shape().offset(&coords);
                    let out_off = outer_idx * (indices.len() * inner) + j * inner + inner_idx;
                    let add_val = *go_ptr.add(out_off);
                    let dst_ptr = grad_input.as_mut_ptr();
                    *dst_ptr.add(in_off) += add_val;
                }
            }
        }
    }
    vec![Some(grad_input)]
}
