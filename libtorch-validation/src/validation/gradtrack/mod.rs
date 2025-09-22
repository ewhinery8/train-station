//! GradTrack cross-thread graph promotion/merge validation
//!
//! This module validates implicit promotion from Local→Shared and group merges
//! for the GradTrack engine by constructing multi-threaded forward graphs and
//! comparing forward/backward results to single-threaded LibTorch equivalents.

use std::sync::Arc;

use crate::validation::{ComparisonResult, TensorValidator};
use train_station::tensor::with_no_mem_pool;
use train_station::Tensor;

fn to_libtorch_tensor(t: &Tensor) -> Result<crate::ffi::LibTorchTensor, String> {
    crate::ffi::LibTorchTensor::from_data(t.data(), t.shape().dims())
}

fn compare_forward(
    validator: &TensorValidator,
    a: &Tensor,
    b: &crate::ffi::LibTorchTensor,
) -> ComparisonResult {
    validator.compare_tensors(a, b)
}

fn compare_backward(
    validator: &TensorValidator,
    wrt: &Tensor,
    torch_wrt: &crate::ffi::LibTorchTensor,
) -> ComparisonResult {
    let our_grad = wrt
        .grad_owned()
        .unwrap_or_else(|| Tensor::zeros(wrt.shape().dims().to_vec()));
    match torch_wrt.grad() {
        Some(g) => validator.compare_tensors(&our_grad, &g),
        None => ComparisonResult::failure("missing torch grad".into()),
    }
}

fn simple_linear(a: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    // (N,D) @ (D,M) + (M)
    let z = a.matmul(w);
    z.add_tensor(b)
}

fn scalar_loss(t: &Tensor) -> Tensor {
    // mean for stable scalar loss
    t.mean()
}

/// Shared utility to prepare LibTorch baseline for linear + mean
fn libtorch_linear_mean(
    a: &Tensor,
    w: &Tensor,
    b: &Tensor,
) -> Result<
    (
        crate::ffi::LibTorchTensor,
        crate::ffi::LibTorchTensor,
        crate::ffi::LibTorchTensor,
        crate::ffi::LibTorchTensor, // z (pre-mean)
        crate::ffi::LibTorchTensor, // loss (mean)
    ),
    String,
> {
    let ta = to_libtorch_tensor(a)?.requires_grad_(true)?;
    let tw = to_libtorch_tensor(w)?.requires_grad_(true)?;
    let tb = to_libtorch_tensor(b)?.requires_grad_(true)?;
    let z = ta.matmul(&tw)?;
    let z = z.add_tensor(&tb)?;
    let loss = z.mean()?;
    Ok((ta, tw, tb, z, loss))
}

/// Build A,W,B possibly across threads to trigger promotions/merges
fn build_cross_thread_linear(
    n: usize,
    d: usize,
    m: usize,
    pattern: usize,
) -> (Arc<Tensor>, Arc<Tensor>, Arc<Tensor>, Tensor) {
    with_no_mem_pool(|| {
        let a = Arc::new(Tensor::ones(vec![n, d]).with_requires_grad());
        let w = Arc::new(Tensor::ones(vec![d, m]).with_requires_grad());
        let b = Arc::new(Tensor::zeros(vec![m]).with_requires_grad());

        let out = match pattern {
            0 => simple_linear(&a, &w, &b),
            1 => {
                let a_cl = a.clone();
                let w_cl = w.clone();
                let handle = std::thread::spawn(move || with_no_mem_pool(|| a_cl.matmul(&w_cl)));
                let z = handle.join().unwrap();
                z.add_tensor(&b)
            }
            2 => {
                let a_cl = a.clone();
                let w_cl = w.clone();
                let h1 = std::thread::spawn(move || with_no_mem_pool(|| a_cl.matmul(&w_cl)));
                let z = h1.join().unwrap();
                z.add_tensor(&b)
            }
            3 => {
                let a_cl = a.clone();
                let w_cl = w.clone();
                let b_cl = b.clone();
                let h1 = std::thread::spawn(move || with_no_mem_pool(|| a_cl.matmul(&w_cl)));
                let h2 = std::thread::spawn(move || with_no_mem_pool(|| b_cl.add_scalar(0.0)));
                let z = h1.join().unwrap();
                let b2 = h2.join().unwrap();
                z.add_tensor(&b2)
            }
            4 => {
                let a_cl = a.clone();
                let w_cl = w.clone();
                let h1 = std::thread::spawn(move || with_no_mem_pool(|| a_cl.matmul(&w_cl)));
                let z = h1.join().unwrap();
                z.add_tensor(&b)
            }
            5 => {
                let a1 = a.clone();
                let a2 = a.clone();
                let w1 = w.clone();
                let w2 = w.clone();
                let h1 = std::thread::spawn(move || with_no_mem_pool(|| a1.matmul(&w1)));
                let h2 = std::thread::spawn(move || with_no_mem_pool(|| a2.matmul(&w2)));
                let z1 = h1.join().unwrap();
                let z2 = h2.join().unwrap();
                z1.add_tensor(&z2).add_tensor(&b)
            }
            _ => simple_linear(&a, &w, &b),
        };
        (a, w, b, out)
    })
}

fn run_linear_case(pattern: usize) -> ComparisonResult {
    let (n, d, m) = (16, 8, 4);
    let (a, w, b, out) = build_cross_thread_linear(n, d, m, pattern);
    let mut loss = scalar_loss(&out);

    // LibTorch baseline (match pattern semantics)
    let (ta, tw, tb, torch_z, tloss) = if pattern == 5 {
        // Two matmuls summed, then add bias, then mean
        let ta = match to_libtorch_tensor(&a).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch ta failed: {}", e)),
        };
        let tw = match to_libtorch_tensor(&w).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch tw failed: {}", e)),
        };
        let tb = match to_libtorch_tensor(&b).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch tb failed: {}", e)),
        };
        let z1 = match ta.matmul(&tw) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let z2 = match ta.matmul(&tw) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let zsum = match z1.add_tensor(&z2) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let z = match zsum.add_tensor(&tb) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let loss = match z.mean() {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        (ta, tw, tb, z, loss)
    } else {
        match libtorch_linear_mean(&a, &w, &b) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(format!("libtorch setup failed: {}", e)),
        }
    };

    // Compare forward using the exact z used for loss (before backward to avoid any accidental mutation)
    let validator = TensorValidator::default();
    let fwd = compare_forward(&validator, &out, &torch_z);
    if !fwd.passed {
        return fwd;
    }

    // Compare grads
    // Backward both
    let _ = tloss.backward_scalar();
    loss.backward(None);
    let ga = compare_backward(&validator, &a, &ta);
    if !ga.passed {
        return ga;
    }
    let gw = compare_backward(&validator, &w, &tw);
    if !gw.passed {
        return gw;
    }
    let gb = compare_backward(&validator, &b, &tb);
    if !gb.passed {
        return gb;
    }

    ComparisonResult::success()
}

fn run_broadcast_row_bias_cross_thread() -> ComparisonResult {
    with_no_mem_pool(|| {
        let (n, d, m) = (16, 8, 4);
        let a = Arc::new(Tensor::ones(vec![n, d]).with_requires_grad());
        let w = Arc::new(Tensor::ones(vec![d, m]).with_requires_grad());
        // Row bias with explicit leading 1-dim to force broadcasting along batch axis
        let b_row = Arc::new(Tensor::ones(vec![1, m]).with_requires_grad());

        // Cross-thread compute: matmul on one thread, bias materialization on another
        let a1 = a.clone();
        let w1 = w.clone();
        let h1 = std::thread::spawn(move || with_no_mem_pool(|| a1.matmul(&w1)));
        let b1 = b_row.clone();
        let h2 = std::thread::spawn(move || with_no_mem_pool(|| b1.add_scalar(0.0)));
        let z = h1.join().unwrap();
        let bcast_bias = h2.join().unwrap();
        let out = z.add_tensor(&bcast_bias);
        let mut loss = out.mean();

        // LibTorch sequential baseline
        let ta = match to_libtorch_tensor(&a).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch ta failed: {}", e)),
        };
        let tw = match to_libtorch_tensor(&w).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch tw failed: {}", e)),
        };
        let tb = match to_libtorch_tensor(&b_row).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch tb failed: {}", e)),
        };
        let torch_z = match ta.matmul(&tw).and_then(|z| z.add_tensor(&tb)) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let tloss = match torch_z.mean() {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };

        let validator = TensorValidator::default();
        let fwd = compare_forward(&validator, &out, &torch_z);
        if !fwd.passed {
            return fwd;
        }

        let _ = tloss.backward_scalar();
        loss.backward(None);

        for (t, tt) in [(&a, &ta), (&w, &tw), (&b_row, &tb)] {
            let r = compare_backward(&validator, t, tt);
            if !r.passed {
                return r;
            }
        }
        ComparisonResult::success()
    })
}

fn run_broadcast_scalar_param_cross_thread() -> ComparisonResult {
    with_no_mem_pool(|| {
        let (n, d, m) = (16, 8, 4);
        let a = Arc::new(Tensor::ones(vec![n, d]).with_requires_grad());
        let w = Arc::new(Tensor::ones(vec![d, m]).with_requires_grad());
        // Trainable scalar parameter as shape [1] to broadcast to [n,m]
        let s = Arc::new(Tensor::ones(vec![1]).with_requires_grad());

        let a1 = a.clone();
        let w1 = w.clone();
        let h1 = std::thread::spawn(move || with_no_mem_pool(|| a1.matmul(&w1)));
        let s1 = s.clone();
        let h2 = std::thread::spawn(move || with_no_mem_pool(|| s1.add_scalar(0.0)));
        let z = h1.join().unwrap();
        let sb = h2.join().unwrap();
        let out = z.add_tensor(&sb);
        let mut loss = out.mean();

        // LibTorch sequential baseline
        let ta = match to_libtorch_tensor(&a).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch ta failed: {}", e)),
        };
        let tw = match to_libtorch_tensor(&w).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch tw failed: {}", e)),
        };
        let ts = match to_libtorch_tensor(&s).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch ts failed: {}", e)),
        };
        let torch_z = match ta.matmul(&tw).and_then(|z| z.add_tensor(&ts)) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let tloss = match torch_z.mean() {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };

        let validator = TensorValidator::default();
        let fwd = compare_forward(&validator, &out, &torch_z);
        if !fwd.passed {
            return fwd;
        }

        let _ = tloss.backward_scalar();
        loss.backward(None);

        for (t, tt) in [(&a, &ta), (&w, &tw), (&s, &ts)] {
            let r = compare_backward(&validator, t, tt);
            if !r.passed {
                return r;
            }
        }
        ComparisonResult::success()
    })
}

fn run_three_matmul_merge_cross_thread() -> ComparisonResult {
    with_no_mem_pool(|| {
        let (n, d, m) = (16, 8, 4);
        let x = Arc::new(Tensor::ones(vec![n, d]).with_requires_grad());
        let w1 = Arc::new(Tensor::ones(vec![d, m]).with_requires_grad());
        let w2 = Arc::new(Tensor::ones(vec![d, m]).with_requires_grad());
        let w3 = Arc::new(Tensor::ones(vec![d, m]).with_requires_grad());
        let b = Arc::new(Tensor::zeros(vec![m]).with_requires_grad());

        let x1 = x.clone();
        let w1c = w1.clone();
        let h1 = std::thread::spawn(move || with_no_mem_pool(|| x1.matmul(&w1c)));
        let x2 = x.clone();
        let w2c = w2.clone();
        let h2 = std::thread::spawn(move || with_no_mem_pool(|| x2.matmul(&w2c)));
        let x3 = x.clone();
        let w3c = w3.clone();
        let h3 = std::thread::spawn(move || with_no_mem_pool(|| x3.matmul(&w3c)));

        let z1 = h1.join().unwrap();
        let z2 = h2.join().unwrap();
        let z3 = h3.join().unwrap();
        let out = z1.add_tensor(&z2).add_tensor(&z3).add_tensor(&b);
        let mut loss = out.mean();

        // LibTorch sequential baseline
        let tx = match to_libtorch_tensor(&x).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch tx failed: {}", e)),
        };
        let tw1 = match to_libtorch_tensor(&w1).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch tw1 failed: {}", e)),
        };
        let tw2 = match to_libtorch_tensor(&w2).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch tw2 failed: {}", e)),
        };
        let tw3 = match to_libtorch_tensor(&w3).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch tw3 failed: {}", e)),
        };
        let tb = match to_libtorch_tensor(&b).and_then(|t| t.requires_grad_(true)) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("libtorch tb failed: {}", e)),
        };
        let zt1 = match tx.matmul(&tw1) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let zt2 = match tx.matmul(&tw2) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let zt3 = match tx.matmul(&tw3) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let zsum = match zt1.add_tensor(&zt2) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let zsum = match zsum.add_tensor(&zt3) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_z = match zsum.add_tensor(&tb) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let tloss = match torch_z.mean() {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };

        let validator = TensorValidator::default();
        let fwd = compare_forward(&validator, &out, &torch_z);
        if !fwd.passed {
            return fwd;
        }

        let _ = tloss.backward_scalar();
        loss.backward(None);

        // Compare grads for x, w1, w2, w3, b
        for (t, tt) in [(&x, &tx), (&w1, &tw1), (&w2, &tw2), (&w3, &tw3), (&b, &tb)] {
            let r = compare_backward(&validator, t, tt);
            if !r.passed {
                return r;
            }
        }
        ComparisonResult::success()
    })
}

/// Public entrypoint to run all graph promotion/merge validations
pub fn run_gradtrack_promotion_merge_validations() -> Vec<(String, ComparisonResult)> {
    vec![
        ("promotion_first_remote_use".into(), run_linear_case(1)),
        ("merge_local_local_same_thread".into(), run_linear_case(0)),
        ("merge_local_local_cross_thread".into(), run_linear_case(3)),
        ("merge_shared_local_into_shared".into(), run_linear_case(4)),
        ("merge_shared_shared".into(), run_linear_case(5)),
        (
            "broadcast_row_bias_cross_thread".into(),
            run_broadcast_row_bias_cross_thread(),
        ),
        (
            "broadcast_scalar_param_cross_thread".into(),
            run_broadcast_scalar_param_cross_thread(),
        ),
        (
            "three_matmul_merge_cross_thread".into(),
            run_three_matmul_merge_cross_thread(),
        ),
        (
            "retain_grad_across_promotions".into(),
            test_retain_grad_across_promotions(),
        ),
        (
            "complex_dag_multithread".into(),
            test_complex_dag_multithread(),
        ),
    ]
}

fn test_retain_grad_across_promotions() -> ComparisonResult {
    let validator = TensorValidator::default();
    let x = Arc::new(Tensor::ones(vec![8, 8]).with_requires_grad());
    let w1 = Arc::new(Tensor::ones(vec![8, 8]).with_requires_grad());
    let w2 = Arc::new(Tensor::ones(vec![8, 8]).with_requires_grad());

    let mut x1 = {
        let xc = x.clone();
        let w1c = w1.clone();
        std::thread::spawn(move || with_no_mem_pool(|| xc.matmul(&w1c)))
            .join()
            .unwrap()
    };
    x1.retain_grad_(true);
    let y = x1.add_tensor(&{
        let xc = x.clone();
        let w2c = w2.clone();
        std::thread::spawn(move || with_no_mem_pool(|| xc.matmul(&w2c)))
            .join()
            .unwrap()
    });
    let mut loss_tensor = y.mean();

    let (tx, tw1, tw2, tloss) = {
        let tx_res = to_libtorch_tensor(&x).and_then(|t| t.requires_grad_(true));
        let tw1_res = to_libtorch_tensor(&w1).and_then(|t| t.requires_grad_(true));
        let tw2_res = to_libtorch_tensor(&w2).and_then(|t| t.requires_grad_(true));
        let (tx, tw1, tw2) = match (tx_res, tw1_res, tw2_res) {
            (Ok(tx), Ok(tw1), Ok(tw2)) => (tx, tw1, tw2),
            _ => return ComparisonResult::failure("libtorch setup".to_string()),
        };
        let z1 = match tx.matmul(&tw1) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let z2 = match tx.matmul(&tw2) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let z = match z1.add_tensor(&z2) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let tloss = match z.mean() {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        (tx, tw1, tw2, tloss)
    };

    let _ = tloss.backward_scalar();
    loss_tensor.backward(None);

    let cx = compare_backward(&validator, &x, &tx);
    if !cx.passed {
        return cx;
    }
    let cw1 = compare_backward(&validator, &w1, &tw1);
    if !cw1.passed {
        return cw1;
    }
    let cw2 = compare_backward(&validator, &w2, &tw2);
    if !cw2.passed {
        return cw2;
    }
    if x1.grad_owned().is_none() {
        return ComparisonResult::failure("x1 retain_grad missing".into());
    }
    ComparisonResult::success()
}

fn test_complex_dag_multithread() -> ComparisonResult {
    let a = Arc::new(Tensor::ones(vec![8, 8]).with_requires_grad());
    let b = Arc::new(Tensor::ones(vec![8, 8]).with_requires_grad());
    let c = Arc::new(Tensor::ones(vec![8, 8]).with_requires_grad());
    let d = Arc::new(Tensor::ones(vec![8, 8]).with_requires_grad());

    let h1 = {
        let a = a.clone();
        let b = b.clone();
        std::thread::spawn(move || with_no_mem_pool(|| a.matmul(&b)))
    };
    let h2 = {
        let c = c.clone();
        let d = d.clone();
        std::thread::spawn(move || with_no_mem_pool(|| c.matmul(&d)))
    };
    let z1 = h1.join().unwrap();
    let z2 = h2.join().unwrap();
    let out = z1.add_tensor(&z2).add_scalar(1.0).log().relu();
    let mut loss_tensor = out.mean();

    let (ta, tb, tc, td, tloss) = {
        let ta_res = to_libtorch_tensor(&a).and_then(|t| t.requires_grad_(true));
        let tb_res = to_libtorch_tensor(&b).and_then(|t| t.requires_grad_(true));
        let tc_res = to_libtorch_tensor(&c).and_then(|t| t.requires_grad_(true));
        let td_res = to_libtorch_tensor(&d).and_then(|t| t.requires_grad_(true));
        let (ta, tb, tc, td) = match (ta_res, tb_res, tc_res, td_res) {
            (Ok(ta), Ok(tb), Ok(tc), Ok(td)) => (ta, tb, tc, td),
            _ => return ComparisonResult::failure("libtorch setup".into()),
        };
        let z1 = match ta.matmul(&tb) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let z2 = match tc.matmul(&td) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let z = match z1.add_tensor(&z2) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let z = match z.add_scalar(1.0) {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let z = match z.log() {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let z = match z.relu() {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        let tloss = match z.mean() {
            Ok(v) => v,
            Err(e) => return ComparisonResult::failure(e),
        };
        (ta, tb, tc, td, tloss)
    };

    let _ = tloss.backward_scalar();
    loss_tensor.backward(None);

    let validator = TensorValidator::default();
    let fwd = match to_libtorch_tensor(&out) {
        Ok(torch_out) => compare_forward(&validator, &out, &torch_out),
        Err(e) => ComparisonResult::failure(format!("torch out conv failed: {}", e)),
    };
    if !fwd.passed {
        return fwd;
    }

    for (t, tt) in [(&a, &ta), (&b, &tb), (&c, &tc), (&d, &td)].iter() {
        let r = compare_backward(&validator, t, tt);
        if !r.passed {
            return r;
        }
    }
    ComparisonResult::success()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradtrack_promotions_and_merges_suite() {
        let results = run_gradtrack_promotion_merge_validations();
        for (name, r) in results {
            assert!(r.passed, "{} failed: {}", name, r.details);
        }
    }
}
