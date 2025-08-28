use crate::tensor::core::Tensor;

pub(crate) fn apply_log(saved_input: &Tensor, grad_output: &Tensor) -> Vec<Option<Tensor>> {
    let mut inv = Tensor::new(saved_input.shape().dims.clone());
    unsafe {
        let src = saved_input.as_ptr();
        let dst = inv.as_mut_ptr();
        let size = saved_input.size();
        for i in 0..size {
            let x = *src.add(i);
            debug_assert!(x > 0.0);
            *dst.add(i) = 1.0 / x;
        }
    }
    let grad_input = grad_output.mul_tensor_optimized(&inv);
    vec![Some(grad_input)]
}
