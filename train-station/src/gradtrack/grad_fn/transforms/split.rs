use crate::tensor::core::Tensor;

pub(crate) fn apply_split(
    dim: usize,
    start: usize,
    length: usize,
    input_shape: &[usize],
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let rank = input_shape.len();
    let mut grad_input = Tensor::zeros(input_shape.to_vec());
    let in_strides = grad_input.strides().to_vec();

    let out_numel = grad_output.size();
    if out_numel == 0 {
        return vec![Some(grad_input)];
    }

    let mut coords = vec![0usize; rank];
    for lin in 0..out_numel {
        let mut tmp = lin;
        for i in (0..rank).rev() {
            let s = if i == dim {
                length
            } else {
                grad_output.shape().dims[i]
            };
            coords[i] = if s == 0 { 0 } else { tmp % s };
            if s != 0 {
                tmp /= s;
            }
        }
        let mut dst_off = 0usize;
        for i in 0..rank {
            let c = if i == dim {
                start + coords[i]
            } else {
                coords[i]
            };
            dst_off += c * in_strides[i];
        }
        unsafe {
            *grad_input.as_mut_ptr().add(dst_off) += *grad_output.as_ptr().add(lin);
        }
    }
    vec![Some(grad_input)]
}
