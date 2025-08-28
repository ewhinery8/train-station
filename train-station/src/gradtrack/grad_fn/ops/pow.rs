use crate::tensor::core::Tensor;

// y = x^a
// For scalar exponent a: dy/dx = a * x^(a-1)
pub(crate) fn apply_pow_scalar(
    exponent: f32,
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
            let base = *x.add(i);
            let local = if exponent == 0.0 {
                0.0
            } else {
                exponent * base.powf(exponent - 1.0)
            };
            *gi.add(i) = *go.add(i) * local;
        }
    }
    vec![Some(grad_input)]
}

// For tensor exponent a: dy/dx = a * x^(a-1); dy/da = y * ln(x)
pub(crate) fn apply_pow_tensor(
    saved_base: &Tensor,
    saved_exponent: &Tensor,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let mut grad_base = Tensor::zeros(saved_base.shape().dims.clone());
    let mut grad_exp = Tensor::zeros(saved_exponent.shape().dims.clone());
    unsafe {
        let x = saved_base.as_ptr();
        let a = saved_exponent.as_ptr();
        let go = grad_output.as_ptr();
        let gb = grad_base.as_mut_ptr();
        let ge = grad_exp.as_mut_ptr();
        let n = grad_base.size();
        for i in 0..n {
            let base = *x.add(i);
            let exp = *a.add(i);
            let y = base.powf(exp);
            let d_dx = if exp == 0.0 {
                0.0
            } else {
                exp * base.powf(exp - 1.0)
            };
            let d_da = if base > 0.0 { y * base.ln() } else { 0.0 };
            let upstream = *go.add(i);
            *gb.add(i) = upstream * d_dx;
            *ge.add(i) = upstream * d_da;
        }
    }
    vec![Some(grad_base), Some(grad_exp)]
}
