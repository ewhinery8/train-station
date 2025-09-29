//! PPO (Proximal Policy Optimization) - Continuous actions example using Train Station public API
//!
//! - Continuous `YardEnv` (action in [-1, 1])
//! - Actor (Gaussian policy: mean from MLP, learnable log_std) + Critic (value function)
//! - Trajectory collection, GAE advantages, PPO clipped surrogate objective
//! - Gradient clipping, zero_grad, clear_all_graphs between updates
//! - Reuses `basic_linear_layer.rs` for small MLPs; no unsafe code
//!
//! Run:
//!   cargo run --release --example ppo_continuous

use train_station::{
    gradtrack::clear_all_graphs_known,
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
    fn normal(&mut self) -> f32 {
        // Box-Muller
        let u1 = self.next_f32().clamp(1e-7, 1.0 - 1e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
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
// Actor: mean = MLP(state); log_std is a learnable parameter tensor
// -------------------------------

struct Actor {
    net: Mlp,
    log_std: Tensor, // shape [action_dim]
}
impl Actor {
    fn new(state_dim: usize, action_dim: usize, seed: Option<u64>) -> Self {
        let net = Mlp::new(&[state_dim, 64, 64, action_dim], seed);
        let log_std = Tensor::from_slice(&vec![0.0; action_dim], vec![action_dim])
            .unwrap()
            .with_requires_grad();
        Self { net, log_std }
    }
    fn forward(&self, state: &Tensor) -> (Tensor, Tensor) {
        // Returns (mean [B, A], log_std [A])
        let mean = self.net.forward(state);
        (
            mean,
            self.log_std
                .view(vec![1, self.log_std.shape().dims()[0] as i32]),
        )
    }
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut ps = self.net.parameters();
        ps.push(&mut self.log_std);
        ps
    }
}

// -------------------------------
// Critic: value function V(s)
// -------------------------------

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
// Continuous YardEnv (same dynamics as TD3 env)
// -------------------------------

struct YardEnv {
    pos: f32,
    vel: f32,
    steps: usize,
    max_steps: usize,
    rng: SmallRng,
}
impl YardEnv {
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
    fn step(&mut self, action_value: f32) -> (Tensor, f32, bool) {
        let a = action_value.clamp(-1.0, 1.0);
        self.vel += 0.1 * a - 0.01 * self.pos;
        self.pos += self.vel;
        self.steps += 1;
        let reward = -(self.pos * self.pos) - 0.1 * (a * a);
        let done = self.pos.abs() > 3.0 || self.steps >= self.max_steps;
        (self.state_tensor(), reward, done)
    }
}

// -------------------------------
// Trajectory storage
// -------------------------------

struct RolloutBatch {
    states: Vec<f32>,
    actions: Vec<f32>,
    log_probs: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<f32>,
    values: Vec<f32>,
    next_states: Vec<f32>,
    _state_dim: usize,
}
impl RolloutBatch {
    fn new(capacity: usize, state_dim: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity * state_dim),
            actions: Vec::with_capacity(capacity),
            log_probs: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            next_states: Vec::with_capacity(capacity * state_dim),
            _state_dim: state_dim,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, s: &[f32], a: f32, lp: f32, r: f32, d: f32, v: f32, s2: &[f32]) {
        self.states.extend_from_slice(s);
        self.actions.push(a);
        self.log_probs.push(lp);
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
// Math helpers
// -------------------------------

fn gaussian_log_prob(action: &Tensor, mean: &Tensor, log_std: &Tensor) -> Tensor {
    // All tensors shaped [B, A] (log_std is broadcastable)
    let std = log_std.exp();
    let var = std.pow_scalar(2.0);
    let log_scale = log_std;
    let diff = action.sub_tensor(mean);
    let log_prob = diff
        .pow_scalar(2.0)
        .div_tensor(&var)
        .add_scalar(std::f32::consts::LN_2 + std::f32::consts::PI)
        .add_tensor(&log_scale.mul_scalar(2.0))
        .mul_scalar(0.5)
        .mul_scalar(-1.0);
    // Sum across action dim (dim=1) -> [B,1]
    log_prob.sum_dims(&[1], true)
}

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
    println!("=== PPO Continuous Example (YardEnv) ===");

    let state_dim = 3usize;
    let action_dim = 1usize;

    // Hparams
    let total_steps = std::env::var("PPO_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4000usize);
    let horizon = 128usize; // rollout length per update
    let epochs = 4usize; // PPO epochs per update
    let mini_batch_size = 64usize; // minibatch from horizon
    let gamma = 0.99f32;
    let lam = 0.95f32; // GAE lambda
    let clip_eps = 0.2f32;
    let vf_coef = 0.5f32;
    let ent_coef = 0.0f32;
    let max_grad_norm = 1.0f32;

    // Models
    let mut actor = Actor::new(state_dim, action_dim, Some(101));
    let mut critic = Critic::new(state_dim, Some(202));

    // Opts
    let mut actor_opt = Adam::with_learning_rate(3e-4);
    for p in actor.parameters() {
        actor_opt.add_parameter(p);
    }
    let mut critic_opt = Adam::with_learning_rate(3e-4);
    for p in critic.parameters() {
        critic_opt.add_parameter(p);
    }

    // Env and RNG
    let mut env = YardEnv::new(42);
    let mut rng = SmallRng::new(999);
    let mut state = env.reset();

    // Metrics
    let mut episode_return = 0.0f32;
    let mut episode = 0usize;
    let mut ema_return: Option<f32> = None;
    let ema_alpha = 0.05f32;
    let mut best_return = f32::NEG_INFINITY;

    let mut t = 0usize;
    while t < total_steps {
        // Collect a rollout
        let mut batch = RolloutBatch::new(horizon, state_dim);
        for _ in 0..horizon {
            // Policy forward (detached sampling to not blow graph; we use stored log_probs)
            let (mean, log_std_row) = actor.forward(&state);
            let mean_v = mean.data()[0];
            let log_std_v = log_std_row.data()[0];
            let std_v = log_std_v.exp();
            let noise = rng.normal();
            let action_v = (mean_v + std_v * noise).clamp(-1.0, 1.0);

            // Build action tensor [1, A] for log_prob calculation with autograd
            let action_t = Tensor::from_slice(&[action_v], vec![1, action_dim]).unwrap();
            let log_prob_t = gaussian_log_prob(&action_t, &mean, &log_std_row);
            let log_prob_v = log_prob_t.data()[0];

            // Step env
            let (next_state, reward, done) = env.step(action_v);
            episode_return += reward;

            // Value
            let value_t = critic.forward(&state);
            let value_v = value_t.data()[0];

            // Push
            batch.push(
                state.data(),
                action_v,
                log_prob_v,
                reward,
                if done { 1.0 } else { 0.0 },
                value_v,
                next_state.data(),
            );

            // Reset
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

        // Bootstrap next values for GAE
        let next_values: Vec<f32> = {
            let mut out = Vec::with_capacity(batch.len());
            for i in 0..batch.len() {
                let s2 = &batch.next_states[i * state_dim..(i + 1) * state_dim];
                let s2_t = Tensor::from_slice(s2, vec![1, state_dim]).unwrap();
                let v2 = critic.forward(&s2_t).data()[0];
                out.push(v2);
            }
            out
        };

        // Compute returns and advantages
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

        // Prepare tensors for training
        let states_t = Tensor::from_slice(&batch.states, vec![batch.len(), state_dim]).unwrap();
        let actions_t = Tensor::from_slice(&batch.actions, vec![batch.len(), action_dim]).unwrap();
        let old_logp_t = Tensor::from_slice(&batch.log_probs, vec![batch.len(), 1]).unwrap();
        let returns_t = Tensor::from_slice(&returns, vec![batch.len(), 1]).unwrap();
        let adv_t = Tensor::from_slice(&adv, vec![batch.len(), 1]).unwrap();

        // PPO epochs over the rollout
        let num_minibatches = batch.len().div_ceil(mini_batch_size);
        for e in 0..epochs {
            for mb in 0..num_minibatches {
                let start = mb * mini_batch_size;
                let end = (start + mini_batch_size).min(batch.len());
                if start >= end {
                    break;
                }

                // Slice views
                let s_mb = states_t.slice_view(start * state_dim, 1, (end - start) * state_dim);
                let s_mb = s_mb.reshape(vec![(end - start) as i32, state_dim as i32]);
                let a_mb = actions_t
                    .slice_view(start * action_dim, 1, (end - start) * action_dim)
                    .reshape(vec![(end - start) as i32, action_dim as i32]);
                let oldlp_mb = old_logp_t
                    .slice_view(start, 1, end - start)
                    .reshape(vec![(end - start) as i32, 1]);
                let ret_mb = returns_t
                    .slice_view(start, 1, end - start)
                    .reshape(vec![(end - start) as i32, 1]);
                let adv_mb = adv_t
                    .slice_view(start, 1, end - start)
                    .reshape(vec![(end - start) as i32, 1]);

                // Zero grads
                {
                    let mut ps = actor.parameters();
                    actor_opt.zero_grad(&mut ps);
                }
                {
                    let mut ps = critic.parameters();
                    critic_opt.zero_grad(&mut ps);
                }

                // Forward actor and critic
                let (mean_mb, log_std_row) = actor.forward(&s_mb);
                let logp_mb = gaussian_log_prob(&a_mb, &mean_mb, &log_std_row);
                let ratio = logp_mb.sub_tensor(&oldlp_mb).exp(); // exp(new-old)
                let clip_low =
                    Tensor::from_slice(&vec![1.0 - clip_eps; end - start], vec![end - start, 1])
                        .unwrap();
                let clip_high =
                    Tensor::from_slice(&vec![1.0 + clip_eps; end - start], vec![end - start, 1])
                        .unwrap();
                // ratio_clipped = min(max(ratio, low), high) using ReLU identities
                let ratio_ge_low = ratio.sub_tensor(&clip_low).relu().add_tensor(&clip_low);
                let ratio_clipped =
                    clip_high.sub_tensor(&ratio_ge_low.sub_tensor(&clip_high).relu());
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

                // Entropy (approx Gaussian entropy per action)
                let entropy = log_std_row
                    .add_scalar(0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln())
                    .sum_dims(&[1], true)
                    .mean()
                    .mul_scalar(ent_coef);

                let mut loss = actor_loss.add_tensor(&v_loss).sub_tensor(&entropy);
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

                // Occasionally log
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

    println!("=== PPO training finished ===");
    Ok(())
}
