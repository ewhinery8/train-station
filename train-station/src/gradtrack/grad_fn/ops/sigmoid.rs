use crate::tensor::core::Tensor;

// Backward for sigmoid: d/dx sigmoid(x) = y * (1 - y), using saved_output y
pub(crate) fn apply_sigmoid(saved_output: &Tensor, grad_output: &Tensor) -> Vec<Option<Tensor>> {
    let mut grad_input = Tensor::zeros(saved_output.shape().dims.clone());
    unsafe {
        let y = saved_output.as_ptr();
        let go = grad_output.as_ptr();
        let gi = grad_input.as_mut_ptr();
        let n = grad_input.size();
        for i in 0..n {
            let s = *y.add(i);
            *gi.add(i) = *go.add(i) * s * (1.0 - s);
        }
    }
    vec![Some(grad_input)]
}
