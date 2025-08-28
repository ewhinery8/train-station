use crate::tensor::core::Tensor;

pub(crate) fn apply_sqrt(saved_output: &Tensor, grad_output: &Tensor) -> Vec<Option<Tensor>> {
    let half = 0.5f32;
    let inv = {
        let mut out = Tensor::new(saved_output.shape().dims.clone());
        unsafe {
            let s = saved_output.as_ptr();
            let d = out.as_mut_ptr();
            let n = saved_output.size();
            for i in 0..n {
                *d.add(i) = 1.0f32 / *s.add(i);
            }
        }
        out
    };
    let scaled = inv.mul_scalar_optimized(half);
    let grad_input = grad_output.mul_tensor_optimized(&scaled);
    vec![Some(grad_input)]
}
