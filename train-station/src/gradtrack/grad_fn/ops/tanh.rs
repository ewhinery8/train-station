use crate::tensor::core::Tensor;

// Backward for tanh: d/dx tanh(x) = 1 - tanh(x)^2; using saved_output
pub(crate) fn apply_tanh(saved_output: &Tensor, grad_output: &Tensor) -> Vec<Option<Tensor>> {
    let mut grad_input = Tensor::zeros(saved_output.shape().dims.clone());
    unsafe {
        let y = saved_output.as_ptr();
        let go = grad_output.as_ptr();
        let gi = grad_input.as_mut_ptr();
        let n = grad_input.size();
        for i in 0..n {
            let t = *y.add(i);
            *gi.add(i) = *go.add(i) * (1.0 - t * t);
        }
    }
    vec![Some(grad_input)]
}
