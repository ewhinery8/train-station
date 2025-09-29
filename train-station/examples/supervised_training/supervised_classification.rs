//! Supervised classification with Feed-Forward Network (softmax + cross-entropy)
//!
//! - Multi-class toy dataset (3 classes) generated synthetically
//! - Reuses `feedforward_network.rs` building block
//! - Stable cross-entropy over logits (no softmax in loss path)
//! - Input normalization to [-1, 1], parameter linking, clipping, graph clearing
//! - Logs loss and accuracy
//!
//! Run:
//!   cargo run --release --example supervised_classification

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

// Cross-entropy over logits: CE = -mean(log_softmax(logits)[range, labels])
fn cross_entropy_logits(
    logits: &Tensor,
    labels: &[usize],
    batch: usize,
    _num_classes: usize,
) -> Tensor {
    // log_softmax = logits - logsumexp(logits, dim=1)
    let max_logits = logits.max_dims(&[1], true);
    let shifted = logits.sub_tensor(&max_logits);
    let exp = shifted.exp();
    let sum_exp = exp.sum_dims(&[1], true);
    let log_sum_exp = sum_exp.log();
    let log_softmax = shifted.sub_tensor(&log_sum_exp);
    let ll = log_softmax.gather(1, labels, &[batch, 1]); // selected log-probs
    ll.mul_scalar(-1.0).mean()
}

fn accuracy_from_logits(
    logits: &Tensor,
    labels: &[usize],
    batch: usize,
    num_classes: usize,
) -> f32 {
    let row = logits.data();
    let mut correct = 0usize;
    for (i, &label) in labels.iter().enumerate().take(batch) {
        let base = i * num_classes;
        let mut best_j = 0usize;
        let mut best_v = row[base];
        for j in 1..num_classes {
            let v = row[base + j];
            if v > best_v {
                best_v = v;
                best_j = j;
            }
        }
        if best_j == label {
            correct += 1;
        }
    }
    correct as f32 / batch as f32
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Supervised Classification Example (Cross-Entropy) ===");

    // Synthetic 2D inputs, 3 classes with linear-ish separations
    let n = 1200usize;
    let classes = 3usize;
    let mut xs: Vec<f32> = Vec::with_capacity(n * 2);
    let mut ys: Vec<usize> = Vec::with_capacity(n);

    // Simple RNG
    let mut state: u64 = 424242;
    let mut rand_f32 = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 16) as f32 / (u32::MAX as f32)
    };

    for _ in 0..n {
        let x1 = rand_f32() * 4.0 - 2.0;
        let x2 = rand_f32() * 4.0 - 2.0;
        // Class by quadrant-ish rule with noise
        let mut c = if x1 + 0.5 * x2 > 0.5 {
            0
        } else if x1 - x2 < -0.5 {
            1
        } else {
            2
        };
        if rand_f32() < 0.05 {
            c = (c + 1) % classes;
        }
        xs.push(x1);
        xs.push(x2);
        ys.push(c);
    }

    // Normalize inputs per-feature to [-1, 1]
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
    let y_train = ys[..n_train].to_vec();
    let x_val = Tensor::from_slice(&xs[n_train * 2..], vec![n - n_train, 2]).unwrap();
    let y_val = ys[n_train..].to_vec();

    // Model: 2 -> 64 -> 64 -> 3 (logits)
    let cfg = FeedForwardConfig {
        input_size: 2,
        hidden_sizes: vec![64, 64],
        output_size: classes,
        use_bias: true,
    };
    let mut net = FeedForwardNetwork::new(cfg, Some(303));

    // Optimizer
    let mut opt = Adam::with_learning_rate(1e-3);
    for p in net.parameters() {
        opt.add_parameter(p);
    }

    let epochs = 300usize;
    let max_grad_norm = 1.0f32;
    let mut best_val_acc = 0.0f32;
    let mut best_val_loss = f32::INFINITY;

    for e in 0..epochs {
        // Zero grads
        {
            let mut params = net.parameters();
            opt.zero_grad(&mut params);
        }

        // Forward logits
        let logits = net.forward(&x_train);
        let mut loss = cross_entropy_logits(&logits, &y_train, n_train, classes);
        loss.backward(None);

        // Step clipped
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
        let train_acc = accuracy_from_logits(&logits, &y_train, n_train, classes);
        let val_logits = net.forward(&x_val);
        let val_loss = cross_entropy_logits(&val_logits, &y_val, n - n_train, classes).value();
        let val_acc = accuracy_from_logits(&val_logits, &y_val, n - n_train, classes);
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
        }
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
        }

        if e % 10 == 0 || e + 1 == epochs {
            println!(
                "epoch {:4} | loss={:.4} acc={:.3} | val_loss={:.4} val_acc={:.3} | best_val_acc={:.3}",
                e, loss.value(), train_acc, val_loss, val_acc, best_val_acc
            );
        }

        clear_all_graphs_known();
    }

    // Quick sample preds via softmax
    let samples = Tensor::from_slice(&[-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], vec![3, 2]).unwrap();
    let sm = net.forward(&samples).softmax(1);
    println!("sample class probs: {:?}", sm.data());

    println!("=== Supervised classification finished ===");
    Ok(())
}
