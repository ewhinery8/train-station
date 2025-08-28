use crate::tensor::core::Tensor;

pub(crate) fn apply_select(
    dim: usize,
    index: usize,
    input_shape: &[usize],
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let rank = input_shape.len();
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    let out_numel = grad_output.size();
    let in_strides = grad_input.strides().to_vec();
    let mut coords = vec![0usize; rank - 1];
    for lin in 0..out_numel {
        let mut tmp = lin;
        for i in (0..(rank - 1)).rev() {
            let s = grad_output.shape().dims[i];
            coords[i] = if s == 0 { 0 } else { tmp % s };
            if s != 0 {
                tmp /= s;
            }
        }
        let mut dst_off = 0usize;
        let mut j = 0usize;
        for (i, &is) in in_strides.iter().enumerate().take(rank) {
            let c = if i == dim {
                index
            } else {
                let v = coords[j];
                j += 1;
                v
            };
            dst_off += c * is;
        }
        unsafe {
            *grad_input.as_mut_ptr().add(dst_off) += *grad_output.as_ptr().add(lin);
        }
    }
    vec![Some(grad_input)]
}
