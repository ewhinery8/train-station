//! PPO (Proximal Policy Optimization) - Discrete actions example using Train Station public API
//!
//! - Discrete `YardEnv` (3 actions: -1, 0, +1)
//! - Actor outputs logits over actions (softmax for probabilities), Critic outputs value
//! - Trajectory collection, GAE advantages, PPO clipped surrogate objective
//! - Gradient clipping, zero_grad, clear_all_graphs between updates
//! - Reuses `basic_linear_layer.rs`; no unsafe code
//!
//! Run:
//!   cargo run --release --example ppo_discrete

use train_station::{
    gradtrack::{clear_all_graphs_known, NoGradTrack},
    optimizers::{Adam, Optimizer},
    Tensor,
};

#[allow(clippy::duplicate_mod)]
#[path = "../neural_networks/basic_linear_layer.rs"]
mod basic_linear_layer;
use basic_linear_layer::LinearLayer;

// -------------------------------
// Small RNG
// -------------------------------

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
}

// -------------------------------
// MLP
// -------------------------------

struct Mlp {
    layers: Vec<LinearLayer>,
}
impl Mlp {
    fn new(sizes: &[usize], seed: Option<u64>) -> Self {
        let mut layers = Vec::new();
        let mut s = seed;
        for w in sizes.windows(2) {
            layers.push(LinearLayer::new(w[0], w[1], s));
            s = s.map(|v| v + 1);
        }
        Self { layers }
    }
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut current: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let out = if i == 0 {
                layer.forward(input)
            } else {
                layer.forward(current.as_ref().unwrap())
            };
            let is_last = i + 1 == self.layers.len();
            let out = if !is_last { out.relu() } else { out };
            current = Some(out);
        }
        current.expect("MLP has at least one layer")
    }
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.layers
            .iter_mut()
            .flat_map(|l| l.parameters())
            .collect()
    }
}

// -------------------------------
// Actor (logits) + Critic
// -------------------------------

struct Actor {
    net: Mlp,
}
impl Actor {
    fn new(state_dim: usize, action_dim: usize, seed: Option<u64>) -> Self {
        Self {
            net: Mlp::new(&[state_dim, 64, 64, action_dim], seed),
        }
    }
    fn forward(&self, state: &Tensor) -> Tensor {
        self.net.forward(state)
    } // logits [B, A]
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.net.parameters()
    }
}

struct Critic {
    net: Mlp,
}
impl Critic {
    fn new(state_dim: usize, seed: Option<u64>) -> Self {
        Self {
            net: Mlp::new(&[state_dim, 64, 64, 1], seed),
        }
    }
    fn forward(&self, state: &Tensor) -> Tensor {
        self.net.forward(state)
    }
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.net.parameters()
    }
}

// -------------------------------
// Discrete YardEnv (3 actions -> -1, 0, +1)
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
        let mut e = Self {
            pos: 0.0,
            vel: 0.0,
            steps: 0,
            max_steps: 200,
            rng: SmallRng::new(seed),
        };
        e.reset();
        e
    }
    fn reset(&mut self) -> Tensor {
        self.pos = (self.rng.next_f32() * 1.0) - 0.5;
        self.vel = (self.rng.next_f32() * 0.2) - 0.1;
        self.steps = 0;
        self.state_tensor()
    }
    fn state_tensor(&self) -> Tensor {
        Tensor::from_slice(&[self.pos, self.vel, 0.0], vec![1, 3]).unwrap()
    }
    fn step(&mut self, action_idx: usize) -> (Tensor, f32, bool) {
        let a = Self::ACTIONS[action_idx.min(2)];
        self.vel += 0.1 * a - 0.01 * self.pos;
        self.pos += self.vel;
        self.steps += 1;
        let reward = -(self.pos * self.pos) - 0.05 * (a * a);
        let done = self.pos.abs() > 3.0 || self.steps >= self.max_steps;
        (self.state_tensor(), reward, done)
    }
}

// -------------------------------
// Rollout storage
// -------------------------------

struct RolloutBatch {
    states: Vec<f32>,
    actions: Vec<usize>,
    old_logps: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<f32>,
    values: Vec<f32>,
    next_states: Vec<f32>,
    _state_dim: usize,
}
impl RolloutBatch {
    fn new(cap: usize, sd: usize) -> Self {
        Self {
            states: Vec::with_capacity(cap * sd),
            actions: Vec::with_capacity(cap),
            old_logps: Vec::with_capacity(cap),
            rewards: Vec::with_capacity(cap),
            dones: Vec::with_capacity(cap),
            values: Vec::with_capacity(cap),
            next_states: Vec::with_capacity(cap * sd),
            _state_dim: sd,
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, s: &[f32], a: usize, lp: f32, r: f32, d: f32, v: f32, s2: &[f32]) {
        self.states.extend_from_slice(s);
        self.actions.push(a);
        self.old_logps.push(lp);
        self.rewards.push(r);
        self.dones.push(d);
        self.values.push(v);
        self.next_states.extend_from_slice(s2);
    }
    fn len(&self) -> usize {
        self.actions.len()
    }
}

// -------------------------------
// Helpers
// -------------------------------

#[allow(clippy::too_many_arguments)]
fn compute_gae(
    returns_out: &mut [f32],
    adv_out: &mut [f32],
    rewards: &[f32],
    dones: &[f32],
    values: &[f32],
    next_values: &[f32],
    gamma: f32,
    lam: f32,
) {
    let n = rewards.len();
    let mut gae = 0.0f32;
    for t in (0..n).rev() {
        let not_done = 1.0 - dones[t];
        let delta = rewards[t] + gamma * next_values[t] * not_done - values[t];
        gae = delta + gamma * lam * not_done * gae;
        adv_out[t] = gae;
        returns_out[t] = gae + values[t];
    }
}

fn normalize_in_place(x: &mut [f32], eps: f32) {
    let n = x.len() as f32;
    if n <= 1.0 {
        return;
    }
    let mean = x.iter().copied().sum::<f32>() / n;
    let var = x
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let std = (var + eps).sqrt();
    for v in x.iter_mut() {
        *v = (*v - mean) / std;
    }
}

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

// log-softmax for selected actions: given logits [B,A] and actions Vec<usize> -> log_prob [B,1]
fn log_prob_actions(
    logits: &Tensor,
    actions: &[usize],
    batch: usize,
    _action_dim: usize,
) -> Tensor {
    let max_logits = logits.max_dims(&[1], true); // [B,1]
    let shifted = logits.sub_tensor(&max_logits);
    let exp = shifted.exp();
    let sum_exp = exp.sum_dims(&[1], true); // [B,1]
    let log_sum_exp = sum_exp.log(); // [B,1]
    let log_softmax = shifted.sub_tensor(&log_sum_exp); // [B,A]
                                                        // gather selected action log-probs
    log_softmax.gather(1, actions, &[batch, 1])
}

// probability ratio = exp(new_logp - old_logp)
fn ratio_from_logps(new_logp: &Tensor, old_logp: &Tensor) -> Tensor {
    new_logp.sub_tensor(old_logp).exp()
}

// Clamp ratio to [1-clip, 1+clip] using ReLU-based clamp (no custom ops)
fn clamp_ratio(ratio: &Tensor, clip_eps: f32) -> Tensor {
    let b = ratio.shape().dims()[0];
    let low = Tensor::from_slice(&vec![1.0 - clip_eps; b], vec![b, 1]).unwrap();
    let high = Tensor::from_slice(&vec![1.0 + clip_eps; b], vec![b, 1]).unwrap();
    let ge_low = ratio.sub_tensor(&low).relu().add_tensor(&low);
    high.sub_tensor(&ge_low.sub_tensor(&high).relu())
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

// -------------------------------
// Main
// -------------------------------

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PPO Discrete Example (YardEnv) ===");

    let state_dim = 3usize;
    let action_dim = 3usize;
    let total_steps = std::env::var("PPOD_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3500usize);
    let horizon = 128usize;
    let epochs = 4usize;
    let mini_batch_size = 64usize;
    let gamma = 0.99f32;
    let lam = 0.95f32;
    let clip_eps = 0.2f32;
    let vf_coef = 0.5f32;
    let ent_coef = 0.0f32;
    let max_grad_norm = 1.0f32;

    let mut actor = Actor::new(state_dim, action_dim, Some(111));
    let mut critic = Critic::new(state_dim, Some(222));
    let mut actor_opt = Adam::with_learning_rate(3e-4);
    for p in actor.parameters() {
        actor_opt.add_parameter(p);
    }
    let mut critic_opt = Adam::with_learning_rate(3e-4);
    for p in critic.parameters() {
        critic_opt.add_parameter(p);
    }

    let mut env = YardEnv::new(1234);
    let mut rng = SmallRng::new(98765);
    let mut state = env.reset();
    let mut episode_return = 0.0f32;
    let mut episode = 0usize;
    let mut ema_return: Option<f32> = None;
    let ema_alpha = 0.05f32;
    let mut best_return = f32::NEG_INFINITY;

    let mut t = 0usize;
    while t < total_steps {
        let mut batch = RolloutBatch::new(horizon, state_dim);
        for _ in 0..horizon {
            // Actor logits and categorical sampling
            let logits = actor.forward(&state); // [1, A]
            let probs = logits.softmax(1); // [1, A]
                                           // sample action from probs (CPU sampling)
            let p = probs.data();
            let (p0, p1, _p2) = (p[0], p[1], p[2]);
            let u = rng.next_f32();
            let a_idx = if u < p0 {
                0
            } else if u < p0 + p1 {
                1
            } else {
                2
            };

            let old_logp = {
                let _ng = NoGradTrack::new();
                let lp = log_prob_actions(&logits, &[a_idx], 1, action_dim);
                lp.data()[0]
            };

            // Step env
            let (next_state, reward, done) = env.step(a_idx);
            episode_return += reward;

            // Critic value
            let value_t = critic.forward(&state);
            let value_v = value_t.data()[0];

            batch.push(
                state.data(),
                a_idx,
                old_logp,
                reward,
                if done { 1.0 } else { 0.0 },
                value_v,
                next_state.data(),
            );

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
                    "step {:5} | episode {:4} return={:.3} ema={:.3} best={:.3}",
                    t,
                    episode,
                    episode_return,
                    ema_return.unwrap_or(episode_return),
                    best_return
                );
                episode_return = 0.0;
                episode += 1;
                st
            } else {
                next_state
            };

            t += 1;
            if t >= total_steps {
                break;
            }
        }

        // Bootstrap values for GAE
        let next_values: Vec<f32> = {
            let mut out = Vec::with_capacity(batch.len());
            for i in 0..batch.len() {
                let s2 = &batch.next_states[i * state_dim..(i + 1) * state_dim];
                let s2_t = Tensor::from_slice(s2, vec![1, state_dim]).unwrap();
                out.push(critic.forward(&s2_t).data()[0]);
            }
            out
        };

        let mut returns = vec![0.0f32; batch.len()];
        let mut adv = vec![0.0f32; batch.len()];
        compute_gae(
            &mut returns,
            &mut adv,
            &batch.rewards,
            &batch.dones,
            &batch.values,
            &next_values,
            gamma,
            lam,
        );
        normalize_in_place(&mut adv, 1e-8);

        // Tensors for training
        let states_t = Tensor::from_slice(&batch.states, vec![batch.len(), state_dim]).unwrap();
        let actions_vec = batch.actions.clone();
        let old_logp_t = Tensor::from_slice(&batch.old_logps, vec![batch.len(), 1]).unwrap();
        let returns_t = Tensor::from_slice(&returns, vec![batch.len(), 1]).unwrap();
        let adv_t = Tensor::from_slice(&adv, vec![batch.len(), 1]).unwrap();

        // PPO epochs
        let num_minibatches = batch.len().div_ceil(mini_batch_size);
        for e in 0..epochs {
            for mb in 0..num_minibatches {
                let start = mb * mini_batch_size;
                let end = (start + mini_batch_size).min(batch.len());
                if start >= end {
                    break;
                }

                // Views
                let s_mb = states_t
                    .slice_view(start * state_dim, 1, (end - start) * state_dim)
                    .reshape(vec![(end - start) as i32, state_dim as i32]);
                let oldlp_mb = old_logp_t
                    .slice_view(start, 1, end - start)
                    .reshape(vec![(end - start) as i32, 1]);
                let ret_mb = returns_t
                    .slice_view(start, 1, end - start)
                    .reshape(vec![(end - start) as i32, 1]);
                let adv_mb = adv_t
                    .slice_view(start, 1, end - start)
                    .reshape(vec![(end - start) as i32, 1]);
                let a_slice = &actions_vec[start..end];

                // Zero grads
                {
                    let mut ps = actor.parameters();
                    actor_opt.zero_grad(&mut ps);
                }
                {
                    let mut ps = critic.parameters();
                    critic_opt.zero_grad(&mut ps);
                }

                // Forward
                let logits_mb = actor.forward(&s_mb); // [B,A]
                let new_logp_mb = log_prob_actions(&logits_mb, a_slice, end - start, action_dim); // [B,1]
                let ratio = ratio_from_logps(&new_logp_mb, &oldlp_mb);
                let ratio_clipped = clamp_ratio(&ratio, clip_eps);
                let pg1 = ratio.mul_tensor(&adv_mb);
                let pg2 = ratio_clipped.mul_tensor(&adv_mb);
                // min(pg1, pg2) = pg2 - relu(pg2 - pg1)
                let actor_min = pg2.sub_tensor(&pg2.sub_tensor(&pg1).relu());
                let actor_loss = actor_min.mul_scalar(-1.0).mean();

                let v_pred = critic.forward(&s_mb);
                let v_loss = v_pred
                    .sub_tensor(&ret_mb)
                    .pow_scalar(2.0)
                    .mean()
                    .mul_scalar(vf_coef);

                // Entropy bonus from logits (categorical entropy) ≈ -sum p*logp
                let probs_mb = logits_mb.softmax(1);
                let logp_all = probs_mb.add_scalar(1e-8).log();
                let ent = probs_mb
                    .mul_tensor(&logp_all)
                    .sum_dims(&[1], true)
                    .mul_scalar(-1.0)
                    .mean()
                    .mul_scalar(ent_coef);

                let mut loss = actor_loss.add_tensor(&v_loss).sub_tensor(&ent);
                loss.backward(None);

                // Step actor
                {
                    let params = actor.parameters();
                    let mut with_grads: Vec<&mut Tensor> = Vec::new();
                    for p in params {
                        if p.grad_owned().is_some() {
                            with_grads.push(p);
                        }
                    }
                    if !with_grads.is_empty() {
                        let _ = grad_global_norm(&mut with_grads);
                        clip_gradients(&mut with_grads, max_grad_norm, 1e-6);
                        actor_opt.step(&mut with_grads);
                        actor_opt.zero_grad(&mut with_grads);
                    }
                }

                // Step critic
                {
                    let params = critic.parameters();
                    let mut with_grads: Vec<&mut Tensor> = Vec::new();
                    for p in params {
                        if p.grad_owned().is_some() {
                            with_grads.push(p);
                        }
                    }
                    if !with_grads.is_empty() {
                        let _ = grad_global_norm(&mut with_grads);
                        clip_gradients(&mut with_grads, max_grad_norm, 1e-6);
                        critic_opt.step(&mut with_grads);
                        critic_opt.zero_grad(&mut with_grads);
                    }
                }

                if e == 0 && mb == 0 {
                    println!(
                        "update@t={} | actor_loss={:.4} v_loss={:.4}",
                        t,
                        actor_loss.value(),
                        v_loss.value()
                    );
                }

                clear_all_graphs_known();
            }
        }
    }

    println!("=== PPO discrete training finished ===");
    Ok(())
}
