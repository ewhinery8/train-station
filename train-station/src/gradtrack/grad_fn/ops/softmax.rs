use crate::tensor::core::Tensor;

// Backward for softmax along dim: if y = softmax(x), dL/dx = y * (dL/dy - sum(dL/dy * y) along dim)
pub(crate) fn apply_softmax(
    dim: usize,
    saved_output: &Tensor,
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let dims = saved_output.shape().dims.clone();
    let reduce = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let outer: usize = dims[..dim].iter().product();
    let mut grad_input = Tensor::zeros(dims.clone());
    unsafe {
        let y = saved_output.as_ptr();
        let go = grad_output.as_ptr();
        let gi = grad_input.as_mut_ptr();
        for o in 0..outer {
            for i in 0..inner {
                // sum_j (go_j * y_j)
                let mut dot = 0.0f32;
                for j in 0..reduce {
                    let off = o * (reduce * inner) + j * inner + i;
                    dot += *go.add(off) * *y.add(off);
                }
                // gi_k = y_k * (go_k - dot)
                for j in 0..reduce {
                    let off = o * (reduce * inner) + j * inner + i;
                    *gi.add(off) = *y.add(off) * (*go.add(off) - dot);
                }
            }
        }
    }
    vec![Some(grad_input)]
}
