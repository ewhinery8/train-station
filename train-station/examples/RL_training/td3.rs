//! TD3 (Twin Delayed DDPG) - Minimal, self-contained example using Train Station public API
//!
//! Goals:
//! - Keep it small and easy to follow
//! - Reuse `basic_linear_layer.rs` building block (no duplication)
//! - Link optimizer parameters correctly (no cloning of params)
//! - Zero gradients and clear all graphs between iterations
//! - Use only public Train Station APIs + standard Rust
//!
//! Run:
//!   cargo run --release --example td3

use train_station::{
    gradtrack::{clear_all_graphs_known, NoGradTrack},
    optimizers::{Adam, Optimizer},
    Tensor,
};

// Reuse simple LinearLayer to build tiny MLPs (actor/critic)
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
        // Numerical Recipes LCG
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

fn tanh_bounded(x: &Tensor) -> Tensor {
    x.tanh()
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

    fn soft_update_from(&mut self, source: &Self, tau: f32) {
        let _ng = NoGradTrack::new();
        for (t, s) in self.layers.iter_mut().zip(source.layers.iter()) {
            // In-place Polyak update to preserve tensor IDs (no optimizer relink needed)
            let new_w = t
                .weight
                .mul_scalar(1.0 - tau)
                .add_tensor(&s.weight.mul_scalar(tau));
            let new_b = t
                .bias
                .mul_scalar(1.0 - tau)
                .add_tensor(&s.bias.mul_scalar(tau));
            {
                let src = new_w.data();
                let dst = t.weight.data_mut();
                dst.copy_from_slice(src);
            }
            {
                let src = new_b.data();
                let dst = t.bias.data_mut();
                dst.copy_from_slice(src);
            }
            t.weight.set_requires_grad(false);
            t.bias.set_requires_grad(false);
        }
    }
}

// -------------------------------
// Actor and Critic
// -------------------------------

struct Actor {
    net: Mlp,
}

impl Actor {
    fn new(state_dim: usize, action_dim: usize, seed: Option<u64>) -> Self {
        // Smaller net for faster demo: sd -> 64 -> 64 -> ad, tanh output
        let net = Mlp::new(&[state_dim, 64, 64, action_dim], seed);
        Self { net }
    }
    fn forward(&self, state: &Tensor) -> Tensor {
        self.net.forward(state, Some(tanh_bounded))
    }
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.net.parameters()
    }
    fn set_requires_grad_all(&mut self, enable: bool) {
        self.net.set_requires_grad_all(enable);
    }
}

struct Critic {
    net: Mlp,
}

impl Critic {
    fn new(state_dim: usize, action_dim: usize, seed: Option<u64>) -> Self {
        let net = Mlp::new(&[state_dim + action_dim, 64, 64, 1], seed);
        Self { net }
    }
    fn forward(&self, state: &Tensor, action: &Tensor) -> Tensor {
        // Concatenate along feature dim (dim=1) for batched inputs
        // IMPORTANT: use views to preserve gradient graph; cloning would detach autograd
        let s_view = state.view(state.shape().dims().iter().map(|&d| d as i32).collect());
        let a_view = action.view(action.shape().dims().iter().map(|&d| d as i32).collect());
        let sa = Tensor::cat(&[s_view, a_view], 1);
        self.net.forward(&sa, None)
    }
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.net.parameters()
    }
    fn set_requires_grad_all(&mut self, enable: bool) {
        self.net.set_requires_grad_all(enable);
    }
}

// -------------------------------
// Simple continuous control environment: YardEnv
// State: normalized features [pos/3, clamp(vel/1, -1..1), bias(=0)] ; Action: scalar in [-1, 1]
// Dynamics: vel += 0.1*act - 0.01*pos; pos += vel
// Reward: -(pos^2) - 0.1*act^2 ; Episode ends if |pos| > 3 or step >= max_steps
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
        // Normalize to keep critic inputs bounded:
        // - Position is bounded by termination at |pos|>3 → scale by 3 to [-1,1]
        // - Velocity scaled by 1.0 and clamped to [-1,1]
        let pos_n = self.pos / 3.0;
        let vel_n = self.vel.clamp(-1.0, 1.0);
        Tensor::from_slice(&[pos_n, vel_n, 0.0], vec![1, 3]).unwrap()
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
// Replay Buffer
// -------------------------------

struct ReplayBuffer {
    capacity: usize,
    size: usize,
    pos: usize,
    state_dim: usize,
    action_dim: usize,
    states: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<f32>,
    next_states: Vec<f32>,
}

impl ReplayBuffer {
    fn new(capacity: usize, state_dim: usize, action_dim: usize) -> Self {
        Self {
            capacity,
            size: 0,
            pos: 0,
            state_dim,
            action_dim,
            states: vec![0.0; capacity * state_dim],
            actions: vec![0.0; capacity * action_dim],
            rewards: vec![0.0; capacity],
            dones: vec![0.0; capacity],
            next_states: vec![0.0; capacity * state_dim],
        }
    }

    fn push(&mut self, s: &[f32], a: &[f32], r: f32, d: f32, s2: &[f32]) {
        let i = self.pos;
        let so = i * self.state_dim;
        let ao = i * self.action_dim;
        self.states[so..so + self.state_dim].copy_from_slice(s);
        self.actions[ao..ao + self.action_dim].copy_from_slice(a);
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
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let mut s_vec = Vec::with_capacity(batch_size * self.state_dim);
        let mut a_vec = Vec::with_capacity(batch_size * self.action_dim);
        let mut r_vec = Vec::with_capacity(batch_size);
        let mut d_vec = Vec::with_capacity(batch_size);
        let mut s2_vec = Vec::with_capacity(batch_size * self.state_dim);

        for _ in 0..batch_size {
            let idx = rng.sample_index(self.size);
            let so = idx * self.state_dim;
            let ao = idx * self.action_dim;
            s_vec.extend_from_slice(&self.states[so..so + self.state_dim]);
            a_vec.extend_from_slice(&self.actions[ao..ao + self.action_dim]);
            r_vec.push(self.rewards[idx]);
            d_vec.push(self.dones[idx]);
            s2_vec.extend_from_slice(&self.next_states[so..so + self.state_dim]);
        }

        let s = Tensor::from_slice(&s_vec, vec![batch_size, self.state_dim]).unwrap();
        let a = Tensor::from_slice(&a_vec, vec![batch_size, self.action_dim]).unwrap();
        let r = Tensor::from_slice(&r_vec, vec![batch_size, 1]).unwrap();
        let d = Tensor::from_slice(&d_vec, vec![batch_size, 1]).unwrap();
        let s2 = Tensor::from_slice(&s2_vec, vec![batch_size, self.state_dim]).unwrap();
        (s, a, r, d, s2)
    }
}

// -------------------------------
// Helper: gradient clipping by global norm
// -------------------------------

fn clip_gradients(parameters: &mut [&mut Tensor], max_norm: f32, eps: f32) {
    // Compute global L2 norm of all grads
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
                let scaled = g.mul_scalar(scale);
                p.set_grad(scaled);
            }
        }
    }
}

// Compute global L2 norm of gradients across a parameter list (read-only)
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

// Compute L2 norm of parameters (weights/biases) across a parameter list
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

// -------------------------------
// Main: TD3 training on YardEnv
// -------------------------------

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TD3 Example (YardEnv) ===");

    // Environment / problem dims
    let state_dim = 3usize;
    let action_dim = 1usize;

    // Hyperparameters (small for demo)
    let gamma = 0.99f32;
    let tau = 0.005f32; // Polyak
    let policy_noise = 0.2f32; // target smoothing noise stddev
    let exploration_noise = 0.1f32; // behavior policy noise stddev
    let policy_delay = 2usize;
    let batch_size = 64usize;
    let start_steps = 500usize; // random exploration steps
    let total_steps = 1500usize;
    let max_grad_norm = 1.0f32;

    // Models
    let mut actor = Actor::new(state_dim, action_dim, Some(11));
    let mut actor_targ = Actor::new(state_dim, action_dim, Some(12));
    actor_targ.net.copy_from(&actor.net);
    actor_targ.set_requires_grad_all(false);

    let mut critic1 = Critic::new(state_dim, action_dim, Some(21));
    let mut critic2 = Critic::new(state_dim, action_dim, Some(22));
    let mut critic1_targ = Critic::new(state_dim, action_dim, Some(23));
    let mut critic2_targ = Critic::new(state_dim, action_dim, Some(24));
    critic1_targ.net.copy_from(&critic1.net);
    critic2_targ.net.copy_from(&critic2.net);
    critic1_targ.set_requires_grad_all(false);
    critic2_targ.set_requires_grad_all(false);

    // Optimizers
    let mut actor_opt = Adam::with_learning_rate(1e-3);
    for p in actor.parameters() {
        actor_opt.add_parameter(p);
    }

    let mut critic_opt = Adam::with_learning_rate(1e-4);
    for p in critic1.parameters() {
        critic_opt.add_parameter(p);
    }
    for p in critic2.parameters() {
        critic_opt.add_parameter(p);
    }

    // Replay buffer and env
    let mut rb = ReplayBuffer::new(100_000, state_dim, action_dim);
    let mut env = YardEnv::new(1234);
    let mut rng = SmallRng::new(987654321);

    // Reset & metric trackers
    let mut state = env.reset(); // [1, state_dim]
    let mut episode_return = 0.0f32;
    let mut episode = 0usize;
    let mut ema_return: Option<f32> = None;
    let ema_alpha = 0.05f32; // smooth short-term
    let mut best_return = f32::NEG_INFINITY;
    let mut policy_updates: usize = 0;

    for t in 0..total_steps {
        // Select action
        let action_tensor = if t < start_steps {
            let a = rng.uniform(-1.0, 1.0);
            Tensor::from_slice(&[a], vec![1, action_dim]).unwrap()
        } else {
            // Behavior policy with exploration noise
            let _ng = NoGradTrack::new();
            let det = actor.forward(&state);
            let noise = Tensor::randn(vec![1, action_dim], None).mul_scalar(exploration_noise);
            tanh_bounded(&det.add_tensor(&noise))
        };
        let action_value = action_tensor.data()[0];

        // Environment step
        let (next_state, reward, done) = env.step(action_value);
        episode_return += reward;

        // Store transition
        let s_slice = state.data().to_vec();
        let a_slice = action_tensor.data().to_vec();
        let s2_slice = next_state.data().to_vec();
        rb.push(
            &s_slice,
            &a_slice,
            reward,
            if done { 1.0 } else { 0.0 },
            &s2_slice,
        );

        state = if done {
            let st = env.reset();
            // Metrics: update EMA and best
            ema_return = Some(match ema_return {
                None => episode_return,
                Some(prev) => prev * (1.0 - ema_alpha) + ema_alpha * episode_return,
            });
            if episode_return > best_return {
                best_return = episode_return;
            }
            println!(
                "step {:5} | episode {:4} return={:.3} ema={:.3} best={:.3} | rb_size={} | policy_updates={}",
                t,
                episode,
                episode_return,
                ema_return.unwrap_or(episode_return),
                best_return,
                rb.size,
                policy_updates
            );
            episode_return = 0.0;
            episode += 1;
            st
        } else {
            next_state
        };

        // Training
        if rb.can_sample(batch_size) {
            // Sample batch
            let (s, a, r, d, s2) = rb.sample(batch_size, &mut rng);

            // Compute target values y = r + (1-d)*gamma*min(Q1', Q2') using target networks (no grad)
            let target_q = {
                let _ng = NoGradTrack::new();
                // Target actions with smoothing noise (tanh bounds)
                let noise =
                    Tensor::randn(vec![batch_size, action_dim], None).mul_scalar(policy_noise);
                let a_targ = tanh_bounded(&actor_targ.forward(&s2).add_tensor(&noise));
                let q1_t = critic1_targ.forward(&s2, &a_targ);
                let q2_t = critic2_targ.forward(&s2, &a_targ);

                // Elementwise min via data() since this path is no-grad
                let q1d = q1_t.data();
                let q2d = q2_t.data();
                let mut min_vec = Vec::with_capacity(batch_size);
                for i in 0..batch_size {
                    let v1 = q1d[i];
                    let v2 = q2d[i];
                    min_vec.push(v1.min(v2));
                }
                let min_q = Tensor::from_slice(&min_vec, vec![batch_size, 1]).unwrap();
                let not_done = Tensor::ones(vec![batch_size, 1]).sub_tensor(&d);
                r.add_tensor(&not_done.mul_scalar(gamma).mul_tensor(&min_q))
            };

            // Critic update (both critics)
            // Zero grads in a short scope, then drop borrows before forward
            {
                let mut params = {
                    let c_params = critic1.parameters();
                    let c2_params = critic2.parameters();
                    let mut tmp: Vec<&mut Tensor> = Vec::new();
                    tmp.extend(c_params);
                    tmp.extend(c2_params);
                    tmp
                };
                critic_opt.zero_grad(&mut params);
            }

            // Forward current Q estimates
            let q1 = critic1.forward(&s, &a);
            let q2 = critic2.forward(&s, &a);
            let diff1 = q1.sub_tensor(&target_q);
            let diff2 = q2.sub_tensor(&target_q);
            let mut critic_loss = diff1
                .pow_scalar(2.0)
                .mean()
                .add_tensor(&diff2.pow_scalar(2.0).mean());

            // Backward
            critic_loss.backward(None);

            // Optional gradient clipping + step (only for params that received grads)
            {
                let params = {
                    let c_params = critic1.parameters();
                    let c2_params = critic2.parameters();
                    let mut tmp: Vec<&mut Tensor> = Vec::new();
                    tmp.extend(c_params);
                    tmp.extend(c2_params);
                    tmp
                };
                let mut with_grads: Vec<&mut Tensor> = Vec::new();
                for p in params {
                    if p.grad_owned().is_some() {
                        with_grads.push(p);
                    }
                }
                if !with_grads.is_empty() {
                    // Pre-step metrics
                    let grad_norm_before = grad_global_norm(&mut with_grads);
                    clip_gradients(&mut with_grads, max_grad_norm, 1e-6);
                    critic_opt.step(&mut with_grads);
                    critic_opt.zero_grad(&mut with_grads);

                    // Post-step metrics (param norm)
                    let mut for_norm_params = {
                        let c_params = critic1.parameters();
                        let c2_params = critic2.parameters();
                        let mut tmp: Vec<&mut Tensor> = Vec::new();
                        tmp.extend(c_params);
                        tmp.extend(c2_params);
                        tmp
                    };
                    let param_norm = params_l2_norm(&mut for_norm_params);

                    // Print compact critic metrics occasionally
                    if t % 100 == 0 {
                        let q1_mean = q1.mean().value();
                        let q2_mean = q2.mean().value();
                        let tq_mean = target_q.mean().value();
                        println!(
                            "t={:5} | critic_loss={:.4} | q1_mean={:.3} q2_mean={:.3} tq_mean={:.3} | grad_norm={:.3} | crit_param_norm={:.3}",
                            t,
                            critic_loss.value(),
                            q1_mean,
                            q2_mean,
                            tq_mean,
                            grad_norm_before,
                            param_norm
                        );
                    }
                }
            }

            // Delayed policy update
            if t % policy_delay == 0 {
                // Actor update: maximize Q1(s, actor(s)) -> minimize -Q1
                // Zero actor grads before backward
                {
                    let mut a_params: Vec<&mut Tensor> = actor.parameters();
                    actor_opt.zero_grad(&mut a_params);
                }

                let a_pred = actor.forward(&s);
                let q_for_actor = critic1.forward(&s, &a_pred);
                let mut actor_loss = q_for_actor.mul_scalar(-1.0).mean();
                actor_loss.backward(None);

                {
                    let a_params: Vec<&mut Tensor> = actor.parameters();
                    let mut with_grads: Vec<&mut Tensor> = Vec::new();
                    for p in a_params {
                        if p.grad_owned().is_some() {
                            with_grads.push(p);
                        }
                    }
                    if !with_grads.is_empty() {
                        let grad_norm_before = grad_global_norm(&mut with_grads);
                        clip_gradients(&mut with_grads, max_grad_norm, 1e-6);
                        actor_opt.step(&mut with_grads);
                        actor_opt.zero_grad(&mut with_grads);

                        // Post-step param norm
                        let mut for_norm_params = actor.parameters();
                        let param_norm = params_l2_norm(&mut for_norm_params);

                        policy_updates += 1;
                        if t % 200 == 0 {
                            println!(
                                "t={:5} | actor_loss={:.4} | act_grad_norm={:.3} | act_param_norm={:.3} | lr_a={:.4e} lr_c={:.4e} | policy_updates={}",
                                t,
                                actor_loss.value(),
                                grad_norm_before,
                                param_norm,
                                actor_opt.learning_rate(),
                                critic_opt.learning_rate(),
                                policy_updates
                            );
                        }
                    }
                }

                // Target updates (Polyak averaging, no grad)
                actor_targ.net.soft_update_from(&actor.net, tau);
                critic1_targ.net.soft_update_from(&critic1.net, tau);
                critic2_targ.net.soft_update_from(&critic2.net, tau);
            }

            // Clear entire graphs to avoid stale accumulation across iterations
            clear_all_graphs_known();
        }
    }

    println!("=== TD3 training finished ===");
    Ok(())
}
