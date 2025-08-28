use crate::tensor::core::Tensor;

// Backward for LeakyReLU: dL/dx = dL/dy if x>0 else negative_slope * dL/dy
pub(crate) fn apply_leaky_relu(
    negative_slope: f32,
    saved_input: &Tensor,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let mut grad_input = Tensor::zeros(saved_input.shape().dims.clone());
    unsafe {
        let x = saved_input.as_ptr();
        let go = grad_output.as_ptr();
        let gi = grad_input.as_mut_ptr();
        let n = grad_input.size();
        for i in 0..n {
            let v = *x.add(i);
            *gi.add(i) = if v > 0.0 {
                *go.add(i)
            } else {
                negative_slope * *go.add(i)
            };
        }
    }
    vec![Some(grad_input)]
}
