//! Learning Rate Scheduling Example
//!
//! This example demonstrates various learning rate scheduling techniques
//! for improving neural network training convergence and performance:
//! - Step decay scheduling with milestones
//! - Exponential decay scheduling
//! - Cosine annealing scheduling
//! - Linear warmup with decay
//! - Adaptive scheduling based on validation loss
//! - Performance comparison across scheduling strategies
//!
//! # Learning Objectives
//!
//! - Understand different learning rate scheduling strategies
//! - Learn how to implement custom learning rate schedules
//! - Explore the impact of scheduling on convergence
//! - Compare different scheduling techniques
//! - Implement adaptive scheduling based on training metrics
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of Adam optimizer
//! - Familiarity with neural network training
//! - Knowledge of optimization concepts
//!
//! # Usage
//!
//! ```bash
//! cargo run --example learning_rate_scheduling
//! ```

use train_station::{
    optimizers::{Adam, Optimizer},
    Tensor,
};

/// Learning rate scheduler trait
trait LearningRateScheduler {
    fn step(&mut self, current_lr: f32, epoch: usize, loss: f32) -> f32;
    fn name(&self) -> &str;
}

/// Step decay scheduler
struct StepDecayScheduler {
    milestones: Vec<usize>,
    gamma: f32,
}

impl StepDecayScheduler {
    fn new(milestones: Vec<usize>, gamma: f32) -> Self {
        Self { milestones, gamma }
    }
}

impl LearningRateScheduler for StepDecayScheduler {
    fn step(&mut self, current_lr: f32, epoch: usize, _loss: f32) -> f32 {
        if self.milestones.contains(&epoch) {
            current_lr * self.gamma
        } else {
            current_lr
        }
    }

    fn name(&self) -> &str {
        "Step Decay"
    }
}

/// Exponential decay scheduler
struct ExponentialDecayScheduler {
    gamma: f32,
}

impl ExponentialDecayScheduler {
    fn new(gamma: f32) -> Self {
        Self { gamma }
    }
}

impl LearningRateScheduler for ExponentialDecayScheduler {
    fn step(&mut self, current_lr: f32, _epoch: usize, _loss: f32) -> f32 {
        current_lr * self.gamma
    }

    fn name(&self) -> &str {
        "Exponential Decay"
    }
}

/// Cosine annealing scheduler
struct CosineAnnealingScheduler {
    t_max: usize,
    eta_min: f32,
    initial_lr: f32,
}

impl CosineAnnealingScheduler {
    fn new(t_max: usize, eta_min: f32, initial_lr: f32) -> Self {
        Self {
            t_max,
            eta_min,
            initial_lr,
        }
    }
}

impl LearningRateScheduler for CosineAnnealingScheduler {
    fn step(&mut self, _current_lr: f32, epoch: usize, _loss: f32) -> f32 {
        let t = epoch as f32;
        let t_max = self.t_max as f32;

        self.eta_min
            + 0.5
                * (self.initial_lr - self.eta_min)
                * (1.0 + (std::f32::consts::PI * t / t_max).cos())
    }

    fn name(&self) -> &str {
        "Cosine Annealing"
    }
}

/// Adaptive scheduler based on validation loss
struct AdaptiveScheduler {
    patience: usize,
    factor: f32,
    min_lr: f32,
    best_loss: f32,
    patience_counter: usize,
}

impl AdaptiveScheduler {
    fn new(patience: usize, factor: f32, min_lr: f32) -> Self {
        Self {
            patience,
            factor,
            min_lr,
            best_loss: f32::INFINITY,
            patience_counter: 0,
        }
    }
}

impl LearningRateScheduler for AdaptiveScheduler {
    fn step(&mut self, current_lr: f32, _epoch: usize, loss: f32) -> f32 {
        if loss < self.best_loss {
            self.best_loss = loss;
            self.patience_counter = 0;
            current_lr
        } else {
            self.patience_counter += 1;
            if self.patience_counter >= self.patience {
                let new_lr = (current_lr * self.factor).max(self.min_lr);
                self.patience_counter = 0;
                new_lr
            } else {
                current_lr
            }
        }
    }

    fn name(&self) -> &str {
        "Adaptive (Reduce on Plateau)"
    }
}

/// Training statistics
#[derive(Debug)]
#[allow(dead_code)]
struct TrainingStats {
    scheduler_name: String,
    final_loss: f32,
    lr_history: Vec<f32>,
    loss_history: Vec<f32>,
    convergence_epoch: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Learning Rate Scheduling Example ===\n");

    demonstrate_step_decay()?;
    demonstrate_exponential_decay()?;
    demonstrate_cosine_annealing()?;
    demonstrate_adaptive_scheduling()?;
    demonstrate_scheduler_comparison()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Demonstrate step decay scheduling
fn demonstrate_step_decay() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Step Decay Scheduling ---");

    let mut scheduler = StepDecayScheduler::new(vec![25, 50, 75], 0.5);
    let stats = train_with_scheduler(&mut scheduler, 100)?;

    println!("Step decay results:");
    println!("  Final loss: {:.6}", stats.final_loss);
    println!("  Convergence epoch: {}", stats.convergence_epoch);
    println!("  Learning rate schedule:");
    for (i, &lr) in stats.lr_history.iter().enumerate().step_by(10) {
        println!("    Epoch {:3}: LR = {:.6}", i, lr);
    }

    Ok(())
}

/// Demonstrate exponential decay scheduling
fn demonstrate_exponential_decay() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Exponential Decay Scheduling ---");

    let mut scheduler = ExponentialDecayScheduler::new(0.95);
    let stats = train_with_scheduler(&mut scheduler, 100)?;

    println!("Exponential decay results:");
    println!("  Final loss: {:.6}", stats.final_loss);
    println!("  Convergence epoch: {}", stats.convergence_epoch);
    println!("  Learning rate schedule:");
    for (i, &lr) in stats.lr_history.iter().enumerate().step_by(10) {
        println!("    Epoch {:3}: LR = {:.6}", i, lr);
    }

    Ok(())
}

/// Demonstrate cosine annealing scheduling
fn demonstrate_cosine_annealing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Cosine Annealing Scheduling ---");

    let initial_lr = 0.1;
    let mut scheduler = CosineAnnealingScheduler::new(100, 0.001, initial_lr);
    let stats = train_with_scheduler(&mut scheduler, 100)?;

    println!("Cosine annealing results:");
    println!("  Final loss: {:.6}", stats.final_loss);
    println!("  Convergence epoch: {}", stats.convergence_epoch);
    println!("  Learning rate schedule:");
    for (i, &lr) in stats.lr_history.iter().enumerate().step_by(10) {
        println!("    Epoch {:3}: LR = {:.6}", i, lr);
    }

    Ok(())
}

/// Demonstrate adaptive scheduling
fn demonstrate_adaptive_scheduling() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Adaptive Scheduling ---");

    let mut scheduler = AdaptiveScheduler::new(5, 0.5, 0.001);
    let stats = train_with_scheduler(&mut scheduler, 100)?;

    println!("Adaptive scheduling results:");
    println!("  Final loss: {:.6}", stats.final_loss);
    println!("  Convergence epoch: {}", stats.convergence_epoch);
    println!("  Learning rate schedule:");
    for (i, &lr) in stats.lr_history.iter().enumerate().step_by(10) {
        println!("    Epoch {:3}: LR = {:.6}", i, lr);
    }

    Ok(())
}

/// Demonstrate scheduler comparison
fn demonstrate_scheduler_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Scheduler Comparison ---");

    let schedulers: Vec<Box<dyn LearningRateScheduler>> = vec![
        Box::new(StepDecayScheduler::new(vec![30, 60], 0.5)),
        Box::new(ExponentialDecayScheduler::new(0.98)),
        Box::new(CosineAnnealingScheduler::new(100, 0.001, 0.05)),
        Box::new(AdaptiveScheduler::new(8, 0.7, 0.001)),
    ];

    let mut results = Vec::new();

    for mut scheduler in schedulers {
        println!("\nTesting {} scheduler:", scheduler.name());

        let stats = train_with_scheduler(scheduler.as_mut(), 100)?;
        results.push(stats);

        println!("  Final loss: {:.6}", results.last().unwrap().final_loss);
        println!(
            "  Convergence: {} epochs",
            results.last().unwrap().convergence_epoch
        );
    }

    // Comparison summary
    println!("\nScheduler Comparison Summary:");
    println!(
        "  {:20} | {:10} | {:12} | {:12}",
        "Scheduler", "Final Loss", "Convergence", "LR Range"
    );
    println!("  {}", "-".repeat(70));

    for stats in &results {
        let lr_range = format!(
            "{:.0e} - {:.0e}",
            stats
                .lr_history
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min),
            stats.lr_history.iter().cloned().fold(0.0, f32::max)
        );
        println!(
            "  {:20} | {:.6} | {:8} | {}",
            stats.scheduler_name, stats.final_loss, stats.convergence_epoch, lr_range
        );
    }

    Ok(())
}

/// Helper function to train with a learning rate scheduler
fn train_with_scheduler(
    scheduler: &mut dyn LearningRateScheduler,
    num_epochs: usize,
) -> Result<TrainingStats, Box<dyn std::error::Error>> {
    // Create training data: y = 2*x + 1
    let x_data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
    let y_true = Tensor::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0], vec![5, 1]).unwrap();

    // Create model parameters
    let mut weight = Tensor::randn(vec![1, 1], Some(456)).with_requires_grad();
    let mut bias = Tensor::zeros(vec![1]).with_requires_grad();

    // Create optimizer with initial learning rate
    let mut optimizer = Adam::with_learning_rate(0.05);
    optimizer.add_parameter(&weight);
    optimizer.add_parameter(&bias);

    // Training loop
    let mut losses = Vec::new();
    let mut lr_history = Vec::new();
    let mut convergence_epoch = num_epochs;

    for epoch in 0..num_epochs {
        // Forward pass
        let y_pred = x_data.matmul(&weight) + &bias;
        let mut loss = (&y_pred - &y_true).pow_scalar(2.0).mean();

        // Backward pass
        loss.backward(None);

        // Update learning rate using scheduler
        let current_lr = optimizer.learning_rate();
        let new_lr = scheduler.step(current_lr, epoch, loss.value());

        if (new_lr - current_lr).abs() > 1e-8 {
            optimizer.set_learning_rate(new_lr);
        }

        // Optimizer step
        optimizer.step(&mut [&mut weight, &mut bias]);
        optimizer.zero_grad(&mut [&mut weight, &mut bias]);

        let loss_value = loss.value();
        losses.push(loss_value);
        lr_history.push(new_lr);

        // Check for convergence
        if loss_value < 0.01 && convergence_epoch == num_epochs {
            convergence_epoch = epoch;
        }
    }

    Ok(TrainingStats {
        scheduler_name: scheduler.name().to_string(),
        final_loss: losses[losses.len() - 1],
        lr_history,
        loss_history: losses,
        convergence_epoch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_decay_scheduler() {
        let mut scheduler = StepDecayScheduler::new(vec![5, 10], 0.5);
        let mut lr = 0.1;

        lr = scheduler.step(lr, 0, 0.0);
        assert_eq!(lr, 0.1);

        lr = scheduler.step(lr, 5, 0.0);
        assert_eq!(lr, 0.05);

        lr = scheduler.step(lr, 10, 0.0);
        assert_eq!(lr, 0.025);
    }

    #[test]
    fn test_exponential_decay_scheduler() {
        let mut scheduler = ExponentialDecayScheduler::new(0.9);
        let mut lr = 0.1;

        lr = scheduler.step(lr, 0, 0.0);
        assert!((lr - 0.09).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        let mut scheduler = CosineAnnealingScheduler::new(10, 0.001, 0.1);

        let lr_start = scheduler.step(0.0, 0, 0.0);
        assert!((lr_start - 0.1).abs() < 1e-6);

        let lr_mid = scheduler.step(0.0, 5, 0.0);
        assert!((lr_mid - 0.0505).abs() < 1e-3); // Approximately halfway

        let lr_end = scheduler.step(0.0, 9, 0.0);
        assert!((lr_end - 0.001).abs() < 1e-3);
    }

    #[test]
    fn test_adaptive_scheduler() {
        let mut scheduler = AdaptiveScheduler::new(2, 0.5, 0.001);
        let mut lr = 0.1;

        // Improving loss - should not change LR
        lr = scheduler.step(lr, 0, 0.5);
        assert_eq!(lr, 0.1);

        // Worse loss - should not change LR yet (patience = 2)
        lr = scheduler.step(lr, 1, 0.6);
        assert_eq!(lr, 0.1);

        lr = scheduler.step(lr, 2, 0.6);
        assert_eq!(lr, 0.05); // Should reduce after patience

        // Improving again - should not change LR
        lr = scheduler.step(lr, 3, 0.4);
        assert_eq!(lr, 0.05);
    }

    #[test]
    fn test_scheduler_training() {
        let mut scheduler = StepDecayScheduler::new(vec![10], 0.5);
        let stats = train_with_scheduler(&mut scheduler, 20).unwrap();

        assert!(stats.final_loss < 1.0);
        assert_eq!(stats.lr_history.len(), 20);
        assert!(stats.convergence_epoch < 20);
    }
}
