use crate::tensor::core::Tensor;

pub(crate) fn apply_cat(
    dim: usize,
    input_sizes: &[usize],
    input_shapes: &[Vec<usize>],
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let out_dims = grad_output.shape().dims.to_vec();
    let inner: usize = out_dims[dim + 1..].iter().product();
    let outer: usize = out_dims[..dim].iter().product();

    let mut prefix = Vec::with_capacity(input_sizes.len());
    let mut acc = 0usize;
    for &len_d in input_sizes.iter() {
        prefix.push(acc);
        acc += len_d;
    }

    let mut grads: Vec<Option<Tensor>> = Vec::with_capacity(input_sizes.len());
    for (i, shape) in input_shapes.iter().enumerate() {
        let len_d = input_sizes[i];
        let mut grad_i = Tensor::new(shape.to_vec());
        if grad_output.size() > 0 && len_d > 0 {
            unsafe {
                let go_ptr = grad_output.as_ptr();
                let dst_ptr = grad_i.as_mut_ptr();
                for outer_idx in 0..outer {
                    let src_base = outer_idx * (out_dims[dim] * inner) + prefix[i] * inner;
                    let dst_base = outer_idx * (len_d * inner);
                    std::ptr::copy_nonoverlapping(
                        go_ptr.add(src_base),
                        dst_ptr.add(dst_base),
                        len_d * inner,
                    );
                }
            }
        }
        grads.push(Some(grad_i));
    }
    grads
}
