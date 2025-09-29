//! DQN (Deep Q-Network) - Minimal example using Train Station public API
//!
//! - Discrete `YardEnv` (3 actions: -1, 0, +1)
//! - Experience Replay + Double DQN targets + target network hard updates
//! - Gradient clipping, zero_grad, clear_all_graphs between steps
//! - Reuses `basic_linear_layer.rs` for a small MLP
//!
//! Run:
//!   cargo run --release --example dqn

use train_station::{
    gradtrack::{clear_all_graphs_known, NoGradTrack},
    optimizers::{Adam, Optimizer},
    Tensor,
};

// Reuse simple LinearLayer to build tiny MLP
#[allow(clippy::duplicate_mod)]
#[path = "../neural_networks/basic_linear_layer.rs"]
mod basic_linear_layer;
use basic_linear_layer::LinearLayer;

// -------------------------------
// Utilities
// -------------------------------

// Simple LCG RNG (no external deps)
struct SmallRng {
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 16) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }
    fn sample_index(&mut self, upper_exclusive: usize) -> usize {
        (self.next_u32() as usize) % upper_exclusive.max(1)
    }
}

// -------------------------------
// Tiny MLP builder on LinearLayer
// -------------------------------

struct Mlp {
    layers: Vec<LinearLayer>,
}

impl Mlp {
    fn new(sizes: &[usize], seed: Option<u64>) -> Self {
        assert!(sizes.len() >= 2);
        let mut layers = Vec::new();
        let mut s = seed;
        for w in sizes.windows(2) {
            layers.push(LinearLayer::new(w[0], w[1], s));
            s = s.map(|v| v + 1);
        }
        Self { layers }
    }

    fn forward(&self, input: &Tensor, final_activation: Option<fn(&Tensor) -> Tensor>) -> Tensor {
        let mut current: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let out = if i == 0 {
                layer.forward(input)
            } else {
                layer.forward(current.as_ref().unwrap())
            };
            let is_last = i + 1 == self.layers.len();
            let out = if !is_last {
                out.relu()
            } else if let Some(act) = final_activation {
                act(&out)
            } else {
                out
            };
            current = Some(out);
        }
        current.expect("MLP has at least one layer")
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for l in &mut self.layers {
            params.extend(l.parameters());
        }
        params
    }

    fn set_requires_grad_all(&mut self, enable: bool) {
        for l in &mut self.layers {
            l.weight.set_requires_grad(enable);
            l.bias.set_requires_grad(enable);
        }
    }

    // In-place copy (preserve tensor IDs and optimizer links on targets)
    fn copy_from(&mut self, other: &Self) {
        for (t, s) in self.layers.iter_mut().zip(other.layers.iter()) {
            {
                let src = s.weight.data();
                let dst = t.weight.data_mut();
                dst.copy_from_slice(src);
            }
            {
                let src = s.bias.data();
                let dst = t.bias.data_mut();
                dst.copy_from_slice(src);
            }
            t.weight.set_requires_grad(false);
            t.bias.set_requires_grad(false);
        }
    }
}

// -------------------------------
// Q-Network (state -> Q-values over actions)
// -------------------------------

struct QNet {
    net: Mlp,
}

impl QNet {
    fn new(state_dim: usize, action_dim: usize, seed: Option<u64>) -> Self {
        let net = Mlp::new(&[state_dim, 64, 64, action_dim], seed);
        Self { net }
    }
    fn forward(&self, state: &Tensor) -> Tensor {
        self.net.forward(state, None)
    }
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.net.parameters()
    }
    fn set_requires_grad_all(&mut self, enable: bool) {
        self.net.set_requires_grad_all(enable);
    }
}

// -------------------------------
// Discrete YardEnv (3 actions: -1, 0, +1)
// -------------------------------

struct YardEnv {
    pos: f32,
    vel: f32,
    steps: usize,
    max_steps: usize,
    rng: SmallRng,
}

impl YardEnv {
    const ACTIONS: [f32; 3] = [-1.0, 0.0, 1.0];

    fn new(seed: u64) -> Self {
        let mut env = Self {
            pos: 0.0,
            vel: 0.0,
            steps: 0,
            max_steps: 200,
            rng: SmallRng::new(seed),
        };
        env.reset();
        env
    }

    fn reset(&mut self) -> Tensor {
        self.pos = self.rng.uniform(-0.5, 0.5);
        self.vel = self.rng.uniform(-0.1, 0.1);
        self.steps = 0;
        self.state_tensor()
    }

    fn state_tensor(&self) -> Tensor {
        Tensor::from_slice(&[self.pos, self.vel, 0.0], vec![1, 3]).unwrap()
    }

    fn step(&mut self, action_index: usize) -> (Tensor, f32, bool) {
        let a = Self::ACTIONS[action_index.min(2)];
        self.vel += 0.1 * a - 0.01 * self.pos;
        self.pos += self.vel;
        self.steps += 1;
        let reward = -(self.pos * self.pos) - 0.05 * (a * a);
        let done = self.pos.abs() > 3.0 || self.steps >= self.max_steps;
        (self.state_tensor(), reward, done)
    }
}

// -------------------------------
// Replay Buffer
// -------------------------------

struct ReplayBuffer {
    capacity: usize,
    size: usize,
    pos: usize,
    state_dim: usize,
    states: Vec<f32>,
    actions: Vec<usize>,
    rewards: Vec<f32>,
    dones: Vec<f32>,
    next_states: Vec<f32>,
}

impl ReplayBuffer {
    fn new(capacity: usize, state_dim: usize) -> Self {
        Self {
            capacity,
            size: 0,
            pos: 0,
            state_dim,
            states: vec![0.0; capacity * state_dim],
            actions: vec![0usize; capacity],
            rewards: vec![0.0; capacity],
            dones: vec![0.0; capacity],
            next_states: vec![0.0; capacity * state_dim],
        }
    }

    fn push(&mut self, s: &[f32], a_idx: usize, r: f32, d: f32, s2: &[f32]) {
        let i = self.pos;
        let so = i * self.state_dim;
        self.states[so..so + self.state_dim].copy_from_slice(s);
        self.actions[i] = a_idx;
        self.rewards[i] = r;
        self.dones[i] = d;
        self.next_states[so..so + self.state_dim].copy_from_slice(s2);
        self.pos = (self.pos + 1) % self.capacity;
        self.size = self.size.saturating_add(1).min(self.capacity);
    }

    fn can_sample(&self, batch_size: usize) -> bool {
        self.size >= batch_size
    }

    fn sample(
        &self,
        batch_size: usize,
        rng: &mut SmallRng,
    ) -> (Tensor, Vec<usize>, Tensor, Tensor, Tensor) {
        let mut s_vec = Vec::with_capacity(batch_size * self.state_dim);
        let mut a_idx = Vec::with_capacity(batch_size);
        let mut r_vec = Vec::with_capacity(batch_size);
        let mut d_vec = Vec::with_capacity(batch_size);
        let mut s2_vec = Vec::with_capacity(batch_size * self.state_dim);
        for _ in 0..batch_size {
            let idx = rng.sample_index(self.size);
            let so = idx * self.state_dim;
            s_vec.extend_from_slice(&self.states[so..so + self.state_dim]);
            a_idx.push(self.actions[idx]);
            r_vec.push(self.rewards[idx]);
            d_vec.push(self.dones[idx]);
            s2_vec.extend_from_slice(&self.next_states[so..so + self.state_dim]);
        }
        let s = Tensor::from_slice(&s_vec, vec![batch_size, self.state_dim]).unwrap();
        let r = Tensor::from_slice(&r_vec, vec![batch_size, 1]).unwrap();
        let d = Tensor::from_slice(&d_vec, vec![batch_size, 1]).unwrap();
        let s2 = Tensor::from_slice(&s2_vec, vec![batch_size, self.state_dim]).unwrap();
        (s, a_idx, r, d, s2)
    }
}

// -------------------------------
// Helpers
// -------------------------------

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

fn grad_global_norm(parameters: &mut [&mut Tensor]) -> f32 {
    let mut total_sq = 0.0f32;
    for p in parameters.iter_mut() {
        if let Some(g) = p.grad_owned() {
            for &v in g.data() {
                total_sq += v * v;
            }
        }
    }
    total_sq.sqrt()
}

fn params_l2_norm(parameters: &mut [&mut Tensor]) -> f32 {
    let _ng = NoGradTrack::new();
    let mut total_sq = 0.0f32;
    for p in parameters.iter_mut() {
        for &v in p.data() {
            total_sq += v * v;
        }
    }
    total_sq.sqrt()
}

// Pseudo-Huber loss: sqrt(1 + diff^2) - 1 (smooth, robust)
fn pseudo_huber_mean(diff: &Tensor) -> Tensor {
    diff.pow_scalar(2.0)
        .add_scalar(1.0)
        .sqrt()
        .sub_scalar(1.0)
        .mean()
}

// -------------------------------
// Main
// -------------------------------

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DQN Example (YardEnv discrete) ===");

    // Dims
    let state_dim = 3usize;
    let action_dim = 3usize;

    // Hparams
    let gamma = 0.99f32;
    let batch_size = 64usize;
    let start_steps = 200usize;
    let target_update_interval = 200usize; // hard update cadence
    let max_grad_norm = 1.0f32;
    let mut epsilon = 1.0f32;
    let eps_min = 0.05f32;
    let eps_decay_steps = 2_000usize; // linear decay
    let total_steps = std::env::var("DQN_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3000usize);

    // Models
    let mut q_net = QNet::new(state_dim, action_dim, Some(7));
    let mut q_targ = QNet::new(state_dim, action_dim, Some(8));
    q_targ.net.copy_from(&q_net.net);
    q_targ.set_requires_grad_all(false);

    // Optimizer
    let mut q_opt = Adam::with_learning_rate(3e-4);
    for p in q_net.parameters() {
        q_opt.add_parameter(p);
    }

    // Replay + env
    let mut rb = ReplayBuffer::new(100_000, state_dim);
    let mut env = YardEnv::new(12345);
    let mut rng = SmallRng::new(999_111);

    // Metrics
    let mut state = env.reset();
    let mut episode_return = 0.0f32;
    let mut episode = 0usize;
    let mut ema_return: Option<f32> = None;
    let ema_alpha = 0.05f32;
    let mut best_return = f32::NEG_INFINITY;

    for t in 0..total_steps {
        // Epsilon-greedy action
        let action_index = if t < start_steps || rng.next_f32() < epsilon {
            rng.sample_index(action_dim)
        } else {
            let _ng = NoGradTrack::new();
            let q_vals = q_net.forward(&state);
            let row = q_vals.data();
            let mut best_i = 0usize;
            let mut best_v = row[0];
            for (i, &r) in row.iter().enumerate().take(action_dim).skip(1) {
                if r > best_v {
                    best_v = r;
                    best_i = i;
                }
            }
            best_i
        };

        // Env step
        let (next_state, reward, done) = env.step(action_index);
        episode_return += reward;

        // Store
        let s_slice = state.data().to_vec();
        let s2_slice = next_state.data().to_vec();
        rb.push(
            &s_slice,
            action_index,
            reward,
            if done { 1.0 } else { 0.0 },
            &s2_slice,
        );

        // Reset on done
        state = if done {
            let st = env.reset();
            ema_return = Some(match ema_return {
                None => episode_return,
                Some(prev) => prev * (1.0 - ema_alpha) + ema_alpha * episode_return,
            });
            if episode_return > best_return {
                best_return = episode_return;
            }
            println!(
                "step {:5} | episode {:4} return={:.3} ema={:.3} best={:.3} | rb_size={}",
                t,
                episode,
                episode_return,
                ema_return.unwrap_or(episode_return),
                best_return,
                rb.size
            );
            episode_return = 0.0;
            episode += 1;
            st
        } else {
            next_state
        };

        // Epsilon linear decay
        if t < eps_decay_steps {
            epsilon = (1.0 - (t as f32) / (eps_decay_steps as f32)) * (1.0 - eps_min) + eps_min;
        }

        // Train
        if rb.can_sample(batch_size) {
            let (s, a_idx, r, d, s2) = rb.sample(batch_size, &mut rng);

            // Double DQN target: a* = argmax_a Q_online(s2,a); y = r + (1-d)*gamma*Q_target(s2, a*)
            let target_q = {
                let _ng = NoGradTrack::new();
                let q_online_s2 = q_net.forward(&s2);
                // argmax per row (manual on CPU)
                let row_stride = action_dim;
                let qd = q_online_s2.data();
                let mut next_actions: Vec<usize> = Vec::with_capacity(batch_size);
                for i in 0..batch_size {
                    let base = i * row_stride;
                    let mut bi = 0usize;
                    let mut bv = qd[base];
                    for j in 1..action_dim {
                        let v = qd[base + j];
                        if v > bv {
                            bv = v;
                            bi = j;
                        }
                    }
                    next_actions.push(bi);
                }
                let q_targ_s2 = q_targ.forward(&s2);
                let q_targ_g = q_targ_s2.gather(1, &next_actions, &[batch_size, 1]);
                let not_done = Tensor::ones(vec![batch_size, 1]).sub_tensor(&d);
                r.add_tensor(&not_done.mul_scalar(gamma).mul_tensor(&q_targ_g))
            };

            // Q(s,a) for current actions
            // Zero grads first
            {
                let mut params = q_net.parameters();
                q_opt.zero_grad(&mut params);
            }

            let q_all = q_net.forward(&s);
            let q_sa = q_all.gather(1, &a_idx, &[batch_size, 1]);
            let diff = q_sa.sub_tensor(&target_q);
            let mut loss = pseudo_huber_mean(&diff);
            loss.backward(None);

            // Step (filter only params with grads)
            {
                let params = q_net.parameters();
                let mut with_grads: Vec<&mut Tensor> = Vec::new();
                for p in params {
                    if p.grad_owned().is_some() {
                        with_grads.push(p);
                    }
                }
                if !with_grads.is_empty() {
                    let gn = grad_global_norm(&mut with_grads);
                    clip_gradients(&mut with_grads, max_grad_norm, 1e-6);
                    q_opt.step(&mut with_grads);
                    q_opt.zero_grad(&mut with_grads);
                    if t % 100 == 0 {
                        let mut pn = q_net.parameters();
                        let pn_l2 = params_l2_norm(&mut pn);
                        let q_mean = q_all.mean().value();
                        println!(
                            "t={:5} | loss={:.4} | q_mean={:.3} | grad_norm={:.3} | param_norm={:.3} | eps={:.3}",
                            t, loss.value(), q_mean, gn, pn_l2, epsilon
                        );
                    }
                }
            }

            // Target hard update
            if t % target_update_interval == 0 {
                q_targ.net.copy_from(&q_net.net);
            }

            // Clear graphs
            clear_all_graphs_known();
        }
    }

    println!("=== DQN training finished ===");
    Ok(())
}
