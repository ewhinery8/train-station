use crate::tensor::core::Tensor;

pub(crate) fn apply_gather(
    dim: usize,
    indices: &[usize],
    input_shape: &[usize],
    index_shape: &[usize],
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    let rank = index_shape.len();
    let numel: usize = index_shape.iter().product();
    let in_strides = grad_input.strides().to_vec();
    let mut coords = vec![0usize; rank];
    for (lin, &idx) in indices.iter().enumerate().take(numel) {
        let mut tmp = lin;
        for i in (0..rank).rev() {
            let s = index_shape[i];
            coords[i] = tmp % s;
            tmp /= s;
        }
        let mut dst_off = 0usize;
        for i in 0..rank {
            let c = if i == dim { idx } else { coords[i] };
            dst_off += c * in_strides[i];
        }
        unsafe {
            *grad_input.as_mut_ptr().add(dst_off) += *grad_output.as_ptr().add(lin);
        }
    }
    vec![Some(grad_input)]
}
