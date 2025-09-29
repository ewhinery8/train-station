//! Supervised learning with a small Feed-Forward Network (Train Station public API)
//!
//! - Binary classification on a simple XOR-like dataset
//! - Reuses `feedforward_network.rs` building block (no duplication)
//! - Proper parameter linking, zero_grad, backward, step, and graph clearing
//! - Gradient clipping and helpful console logging
//!
//! Run:
//!   cargo run --release --example supervised_ffn

use train_station::{
    gradtrack::clear_all_graphs_known,
    optimizers::{Adam, Optimizer},
    Tensor,
};

#[allow(clippy::duplicate_mod)]
#[path = "../neural_networks/feedforward_network.rs"]
mod feedforward_network;
use feedforward_network::{FeedForwardConfig, FeedForwardNetwork};

// Small helper: global-norm gradient clipping
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

fn accuracy(pred: &Tensor, targets: &Tensor) -> f32 {
    // pred: [B,1] with sigmoid; threshold at 0.5
    let p = pred.data();
    let t = targets.data();
    let mut correct = 0usize;
    for i in 0..p.len() {
        let yhat = if p[i] >= 0.5 { 1.0 } else { 0.0 };
        if (yhat - t[i]).abs() < 1e-6 {
            correct += 1;
        }
    }
    correct as f32 / (p.len() as f32)
}

// Numerically stable BCE with logits:
// L = mean( relu(z) - z*y + log(1 + exp(-|z|)) )
fn bce_with_logits(logits: &Tensor, targets: &Tensor) -> Tensor {
    let relu_z = logits.relu();
    let zy = logits.mul_tensor(targets);
    // |z| = relu(z) + relu(-z)
    let abs_z = relu_z.add_tensor(&logits.mul_scalar(-1.0).relu());
    let log_term = abs_z.mul_scalar(-1.0).exp().add_scalar(1.0).log();
    relu_z.sub_tensor(&zy).add_tensor(&log_term).mean()
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Supervised FFN Example (XOR) ===");

    // Dataset: XOR (repeat to form a small batch)
    let inputs: Vec<f32> = vec![
        0.0, 0.0, // -> 0
        0.0, 1.0, // -> 1
        1.0, 0.0, // -> 1
        1.0, 1.0, // -> 0
    ];
    let targets: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    // Repeat the base patterns to stabilize training
    let repeats = 64usize; // effective batch = 4 * repeats = 256
    let mut xs = Vec::with_capacity(repeats * inputs.len());
    let mut ys = Vec::with_capacity(repeats * targets.len());
    for _ in 0..repeats {
        xs.extend_from_slice(&inputs);
        ys.extend_from_slice(&targets);
    }

    let batch = xs.len() / 2; // two features
    let x_t = Tensor::from_slice(&xs, vec![batch, 2]).unwrap();
    let y_t = Tensor::from_slice(&ys, vec![batch, 1]).unwrap();

    // Model config: 2 -> 32 -> 32 -> 1, final sigmoid via loss path
    let cfg = FeedForwardConfig {
        input_size: 2,
        hidden_sizes: vec![32, 32],
        output_size: 1,
        use_bias: true,
    };
    let mut net = FeedForwardNetwork::new(cfg, Some(777));

    // Optimizer and parameter linking
    let mut opt = Adam::with_learning_rate(1e-3);
    for p in net.parameters() {
        opt.add_parameter(p);
    }

    let epochs = 1000usize;
    let max_grad_norm = 1.0f32;
    let mut best_loss = f32::INFINITY;
    let mut best_acc = 0.0f32;

    for e in 0..epochs {
        // Zero grads each iteration
        {
            let mut params = net.parameters();
            opt.zero_grad(&mut params);
        }

        // Forward -> logits; use numerically stable BCE-with-logits for loss
        let logits = net.forward(&x_t);
        let mut loss = bce_with_logits(&logits, &y_t);
        loss.backward(None);

        // Step only params with grads
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

        // Metrics (use sigmoid only for reporting accuracy)
        let preds = logits.sigmoid();
        let acc = accuracy(&preds, &y_t);
        if loss.value() < best_loss {
            best_loss = loss.value();
        }
        if acc > best_acc {
            best_acc = acc;
        }
        if e % 10 == 0 || e + 1 == epochs {
            println!(
                "epoch {:4} | loss={:.5} acc={:.3} | best_loss={:.5} best_acc={:.3}",
                e,
                loss.value(),
                acc,
                best_loss,
                best_acc
            );
        }

        // Clear graphs to avoid stale accumulation across epochs
        clear_all_graphs_known();
    }

    // Quick sanity check predictions
    let test = Tensor::from_slice(&inputs, vec![4, 2]).unwrap();
    let out = net.forward(&test).sigmoid();
    println!("predictions (approx): {:?}", out.data());

    println!("=== Supervised training finished ===");
    Ok(())
}
