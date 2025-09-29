//! Supervised regression with a small Feed-Forward Network (Train Station public API)
//!
//! - Continuous regression on a simple y = 2x1 - 3x2 + 0.5 noisy dataset
//! - Reuses `feedforward_network.rs` building block
//! - Proper parameter linking, zero_grad, backward, step, and graph clearing
//! - Gradient clipping and helpful console logging (loss, RMSE, R^2)
//!
//! Run:
//!   cargo run --release --example supervised_regression

use train_station::{
    gradtrack::clear_all_graphs_known,
    optimizers::{Adam, Optimizer},
    Tensor,
};

#[allow(clippy::duplicate_mod)]
#[path = "../neural_networks/feedforward_network.rs"]
mod feedforward_network;
use feedforward_network::{FeedForwardConfig, FeedForwardNetwork};

fn clip_gradients(parameters: &mut [&mut Tensor], max_norm: f32, eps: f32) {
    let mut total_sq = 0.0f32;
    for p in parameters.iter() {
        if let Some(g) = p.grad_owned() {
            for &v in g.data() {
                total_sq += v * v;
            }
        }
    }
    let norm = total_sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / (norm + eps);
        for p in parameters.iter_mut() {
            if let Some(g) = p.grad_owned() {
                p.set_grad(g.mul_scalar(scale));
            }
        }
    }
}

fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.sub_tensor(target).pow_scalar(2.0).mean()
}

fn rmse(pred: &Tensor, target: &Tensor) -> f32 {
    mse(pred, target).sqrt().value()
}

fn r2_score(pred: &Tensor, target: &Tensor) -> f32 {
    // R^2 = 1 - SS_res / SS_tot
    let y = target;
    let y_mean = y.mean();
    let ss_res = pred.sub_tensor(y).pow_scalar(2.0).sum();
    let ss_tot = y.sub_tensor(&y_mean).pow_scalar(2.0).sum();
    let ss_res_v = ss_res.value();
    let ss_tot_v = ss_tot.value().max(1e-12); // avoid divide by zero
    1.0 - (ss_res_v / ss_tot_v)
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Supervised Regression Example (MSE) ===");

    // Generate simple synthetic data: y = 2*x1 - 3*x2 + 0.5 + noise
    let n = 1024usize;
    let mut xs: Vec<f32> = Vec::with_capacity(n * 2);
    let mut ys: Vec<f32> = Vec::with_capacity(n);
    // Simple LCG RNG for reproducibility
    let mut state: u64 = 123456789;
    let mut rand_f32 = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 16) as f32 / (u32::MAX as f32)
    };
    for _ in 0..n {
        let x1 = rand_f32() * 2.0 - 1.0;
        let x2 = rand_f32() * 2.0 - 1.0;
        let noise = (rand_f32() * 2.0 - 1.0) * 0.05;
        let y = 2.0 * x1 - 3.0 * x2 + 0.5 + noise;
        xs.push(x1);
        xs.push(x2);
        ys.push(y);
    }

    // Normalize targets to [-1, 1] (max-abs scaling) for reasonable loss magnitudes
    let mut max_abs = 0.0f32;
    for &v in &ys {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs < 1e-8 {
        max_abs = 1.0;
    }
    for v in ys.iter_mut() {
        *v /= max_abs;
    }

    // Normalize inputs per-feature to [-1, 1] (min-max scaling)
    let mut min1 = f32::INFINITY;
    let mut max1 = f32::NEG_INFINITY;
    let mut min2 = f32::INFINITY;
    let mut max2 = f32::NEG_INFINITY;
    for i in (0..xs.len()).step_by(2) {
        let a = xs[i];
        let b = xs[i + 1];
        if a < min1 {
            min1 = a;
        }
        if a > max1 {
            max1 = a;
        }
        if b < min2 {
            min2 = b;
        }
        if b > max2 {
            max2 = b;
        }
    }
    let rng1 = (max1 - min1).max(1e-8);
    let rng2 = (max2 - min2).max(1e-8);
    for i in (0..xs.len()).step_by(2) {
        let a = xs[i];
        let b = xs[i + 1];
        xs[i] = 2.0 * (a - min1) / rng1 - 1.0;
        xs[i + 1] = 2.0 * (b - min2) / rng2 - 1.0;
    }

    // Train/Val split (80/20)
    let n_train = (n as f32 * 0.8) as usize;
    let x_train = Tensor::from_slice(&xs[..n_train * 2], vec![n_train, 2]).unwrap();
    let y_train = Tensor::from_slice(&ys[..n_train], vec![n_train, 1]).unwrap();
    let x_val = Tensor::from_slice(&xs[n_train * 2..], vec![n - n_train, 2]).unwrap();
    let y_val = Tensor::from_slice(&ys[n_train..], vec![n - n_train, 1]).unwrap();

    // Model config: 2 -> 64 -> 64 -> 1
    let cfg = FeedForwardConfig {
        input_size: 2,
        hidden_sizes: vec![64, 64],
        output_size: 1,
        use_bias: true,
    };
    let mut net = FeedForwardNetwork::new(cfg, Some(2025));

    // Optimizer and parameter linking
    let mut opt = Adam::with_learning_rate(1e-3);
    for p in net.parameters() {
        opt.add_parameter(p);
    }

    let epochs = 400usize;
    let max_grad_norm = 1.0f32;
    let mut best_val_rmse = f32::INFINITY;
    let mut best_val_r2 = -f32::INFINITY;

    for e in 0..epochs {
        // Zero grads
        {
            let mut params = net.parameters();
            opt.zero_grad(&mut params);
        }

        // Forward
        let pred = net.forward(&x_train);
        let mut loss = mse(&pred, &y_train);
        loss.backward(None);

        // Step
        {
            let params = net.parameters();
            let mut with_grads: Vec<&mut Tensor> = Vec::new();
            for p in params {
                if p.grad_owned().is_some() {
                    with_grads.push(p);
                }
            }
            if !with_grads.is_empty() {
                clip_gradients(&mut with_grads, max_grad_norm, 1e-6);
                opt.step(&mut with_grads);
                opt.zero_grad(&mut with_grads);
            }
        }

        // Metrics
        let train_rmse = rmse(&pred, &y_train);
        let train_r2 = r2_score(&pred, &y_train);
        let val_pred = net.forward(&x_val);
        let val_rmse = rmse(&val_pred, &y_val);
        let val_r2 = r2_score(&val_pred, &y_val);
        if val_rmse < best_val_rmse {
            best_val_rmse = val_rmse;
        }
        if val_r2 > best_val_r2 {
            best_val_r2 = val_r2;
        }

        if e % 20 == 0 || e + 1 == epochs {
            // Clamp displayed R^2 to avoid huge negative prints on early epochs
            let train_r2_disp = train_r2.max(-10.0);
            let val_r2_disp = val_r2.max(-10.0);
            println!(
                "epoch {:4} | train_rmse={:.4} r2={:.3} | val_rmse={:.4} r2={:.3} | best_val_rmse={:.4} best_val_r2={:.3}",
                e, train_rmse, train_r2_disp, val_rmse, val_r2_disp, best_val_rmse, best_val_r2
            );
        }

        clear_all_graphs_known();
    }

    // Quick sanity predictions on small samples
    let sample = Tensor::from_slice(&[0.5, -0.25, -0.8, 0.3], vec![2, 2]).unwrap();
    let sample_pred = net.forward(&sample);
    println!("samples pred: {:?}", sample_pred.data());

    println!("=== Supervised regression finished ===");
    Ok(())
}
