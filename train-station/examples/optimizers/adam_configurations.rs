//! Adam Configurations Example
//!
//! This example demonstrates different Adam optimizer configurations and their
//! impact on neural network training convergence and performance:
//! - Default Adam configuration for baseline performance
//! - Custom learning rates and their effects
//! - Weight decay regularization techniques
//! - Beta parameter tuning for momentum control
//! - Performance comparison across configurations
//!
//! # Learning Objectives
//!
//! - Understand Adam hyperparameter configuration
//! - Learn how learning rate affects convergence
//! - Explore weight decay for regularization
//! - Compare beta parameters for momentum control
//! - Implement configuration benchmarking workflows
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of tensor operations
//! - Familiarity with neural network training loops
//! - Knowledge of optimization concepts
//!
//! # Usage
//!
//! ```bash
//! cargo run --example adam_configurations
//! ```

use train_station::{
    optimizers::{Adam, AdamConfig, Optimizer},
    Tensor,
};

/// Configuration for training experiments
#[derive(Debug, Clone, PartialEq)]
struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.01,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
        }
    }
}

/// Training statistics for performance analysis
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TrainingStats {
    pub config: TrainingConfig,
    pub final_loss: f32,
    pub loss_history: Vec<f32>,
    pub convergence_epoch: usize,
    pub weight_norm: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Adam Configurations Example ===\n");

    demonstrate_default_adam()?;
    demonstrate_learning_rate_comparison()?;
    demonstrate_weight_decay_comparison()?;
    demonstrate_beta_parameter_tuning()?;
    demonstrate_configuration_benchmarking()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Demonstrate default Adam configuration
fn demonstrate_default_adam() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Default Adam Configuration ---");

    // Create a simple regression problem: y = 2*x + 1
    let x_data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
    let y_true = Tensor::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0], vec![5, 1]).unwrap();

    // Create model parameters
    let mut weight = Tensor::randn(vec![1, 1], Some(42)).with_requires_grad();
    let mut bias = Tensor::zeros(vec![1]).with_requires_grad();

    // Create Adam optimizer with default configuration
    let mut optimizer = Adam::new();
    optimizer.add_parameter(&weight);
    optimizer.add_parameter(&bias);

    println!("Default Adam configuration:");
    println!("  Learning rate: {}", optimizer.learning_rate());
    println!("  Initial weight: {:.6}", weight.value());
    println!("  Initial bias: {:.6}", bias.value());

    // Training loop
    let num_epochs = 50;
    let mut losses = Vec::new();

    for epoch in 0..num_epochs {
        // Forward pass
        let y_pred = x_data.matmul(&weight) + &bias;
        let mut loss = (&y_pred - &y_true).pow_scalar(2.0).mean();

        // Backward pass
        loss.backward(None);

        // Optimizer step
        optimizer.step(&mut [&mut weight, &mut bias]);
        optimizer.zero_grad(&mut [&mut weight, &mut bias]);

        losses.push(loss.value());

        if epoch % 10 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {:3}: Loss = {:.6}", epoch, loss.value());
        }
    }

    // Evaluate final model
    let _final_predictions = x_data.matmul(&weight) + &bias;
    println!("\nFinal model:");
    println!("  Learned weight: {:.6} (target: 2.0)", weight.value());
    println!("  Learned bias: {:.6} (target: 1.0)", bias.value());
    println!("  Final loss: {:.6}", losses[losses.len() - 1]);

    Ok(())
}

/// Demonstrate learning rate comparison
fn demonstrate_learning_rate_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Learning Rate Comparison ---");

    let learning_rates = [0.001, 0.01, 0.1];
    let mut results = Vec::new();

    for &lr in &learning_rates {
        println!("\nTesting learning rate: {}", lr);

        let stats = train_with_config(TrainingConfig {
            learning_rate: lr,
            ..Default::default()
        })?;

        results.push((lr, stats.clone()));

        println!("  Final loss: {:.6}", stats.final_loss);
        println!("  Convergence epoch: {}", stats.convergence_epoch);
    }

    // Compare results
    println!("\nLearning Rate Comparison Summary:");
    for (lr, stats) in &results {
        println!(
            "  LR={:6}: Loss={:.6}, Converged@{}",
            lr, stats.final_loss, stats.convergence_epoch
        );
    }

    Ok(())
}

/// Demonstrate weight decay comparison
fn demonstrate_weight_decay_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Weight Decay Comparison ---");

    let weight_decays = [0.0, 0.001, 0.01];
    let mut results = Vec::new();

    for &wd in &weight_decays {
        println!("\nTesting weight decay: {}", wd);

        let stats = train_with_config(TrainingConfig {
            weight_decay: wd,
            ..Default::default()
        })?;

        results.push((wd, stats.clone()));

        println!("  Final loss: {:.6}", stats.final_loss);
        println!("  Final weight norm: {:.6}", stats.weight_norm);
    }

    // Compare results
    println!("\nWeight Decay Comparison Summary:");
    for (wd, stats) in &results {
        println!(
            "  WD={:6}: Loss={:.6}, Weight Norm={:.6}",
            wd, stats.final_loss, stats.weight_norm
        );
    }

    Ok(())
}

/// Demonstrate beta parameter tuning
fn demonstrate_beta_parameter_tuning() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Beta Parameter Tuning ---");

    let beta_configs = [
        (0.9, 0.999),  // Default
        (0.8, 0.999),  // More aggressive momentum
        (0.95, 0.999), // Less aggressive momentum
        (0.9, 0.99),   // Faster second moment decay
    ];

    let mut results = Vec::new();

    for (i, (beta1, beta2)) in beta_configs.iter().enumerate() {
        println!(
            "\nTesting beta configuration {}: beta1={}, beta2={}",
            i + 1,
            beta1,
            beta2
        );

        let config = TrainingConfig {
            beta1: *beta1,
            beta2: *beta2,
            ..Default::default()
        };

        let stats = train_with_config(config)?;
        results.push(((*beta1, *beta2), stats.clone()));

        println!("  Final loss: {:.6}", stats.final_loss);
        println!("  Convergence epoch: {}", stats.convergence_epoch);
    }

    // Compare results
    println!("\nBeta Parameter Comparison Summary:");
    for ((beta1, beta2), stats) in &results {
        println!(
            "  B1={:4}, B2={:5}: Loss={:.6}, Converged@{}",
            beta1, beta2, stats.final_loss, stats.convergence_epoch
        );
    }

    Ok(())
}

/// Demonstrate configuration benchmarking
fn demonstrate_configuration_benchmarking() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Configuration Benchmarking ---");

    // Define configurations to benchmark
    let configs = vec![
        (
            "Conservative",
            TrainingConfig {
                learning_rate: 0.001,
                weight_decay: 0.001,
                beta1: 0.95,
                ..Default::default()
            },
        ),
        (
            "Balanced",
            TrainingConfig {
                learning_rate: 0.01,
                weight_decay: 0.0,
                beta1: 0.9,
                ..Default::default()
            },
        ),
        (
            "Aggressive",
            TrainingConfig {
                learning_rate: 0.1,
                weight_decay: 0.0,
                beta1: 0.8,
                ..Default::default()
            },
        ),
    ];

    let mut benchmark_results = Vec::new();

    for (name, config) in configs {
        println!("\nBenchmarking {} configuration:", name);

        let start_time = std::time::Instant::now();
        let stats = train_with_config(config.clone())?;
        let elapsed = start_time.elapsed();

        println!("  Training time: {:.2}ms", elapsed.as_millis());
        println!("  Final loss: {:.6}", stats.final_loss);
        println!("  Convergence: {} epochs", stats.convergence_epoch);

        benchmark_results.push((name.to_string(), stats, elapsed));
    }

    // Summary
    println!("\nBenchmarking Summary:");
    for (name, stats, elapsed) in &benchmark_results {
        println!(
            "  {:12}: Loss={:.6}, Time={:4}ms, Converged@{}",
            name,
            stats.final_loss,
            elapsed.as_millis(),
            stats.convergence_epoch
        );
    }

    Ok(())
}

/// Helper function to train with specific configuration
fn train_with_config(config: TrainingConfig) -> Result<TrainingStats, Box<dyn std::error::Error>> {
    // Create training data
    let x_data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
    let y_true = Tensor::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0], vec![5, 1]).unwrap();

    // Create model parameters
    let mut weight = Tensor::randn(vec![1, 1], Some(123)).with_requires_grad();
    let mut bias = Tensor::zeros(vec![1]).with_requires_grad();

    // Create optimizer with custom configuration
    let adam_config = AdamConfig {
        learning_rate: config.learning_rate,
        beta1: config.beta1,
        beta2: config.beta2,
        eps: 1e-8,
        weight_decay: config.weight_decay,
        amsgrad: false,
    };

    let mut optimizer = Adam::with_config(adam_config);
    optimizer.add_parameter(&weight);
    optimizer.add_parameter(&bias);

    // Training loop
    let mut losses = Vec::new();
    let mut convergence_epoch = config.epochs;

    for epoch in 0..config.epochs {
        // Forward pass
        let y_pred = x_data.matmul(&weight) + &bias;
        let mut loss = (&y_pred - &y_true).pow_scalar(2.0).mean();

        // Backward pass
        loss.backward(None);

        // Optimizer step
        optimizer.step(&mut [&mut weight, &mut bias]);
        optimizer.zero_grad(&mut [&mut weight, &mut bias]);

        let loss_value = loss.value();
        losses.push(loss_value);

        // Check for convergence (loss < 0.01)
        if loss_value < 0.01 && convergence_epoch == config.epochs {
            convergence_epoch = epoch;
        }
    }

    Ok(TrainingStats {
        config,
        final_loss: losses[losses.len() - 1],
        loss_history: losses,
        convergence_epoch,
        weight_norm: weight.norm().value(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_adam_convergence() {
        let config = TrainingConfig::default();
        let stats = train_with_config(config).unwrap();

        assert!(stats.final_loss < 1.0);
        assert!(stats.convergence_epoch < config.epochs);
    }

    #[test]
    fn test_learning_rate_effect() {
        let config_slow = TrainingConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let config_fast = TrainingConfig {
            learning_rate: 0.1,
            ..Default::default()
        };

        let stats_slow = train_with_config(config_slow).unwrap();
        let stats_fast = train_with_config(config_fast).unwrap();

        // Faster learning rate should converge faster (lower epoch count)
        assert!(stats_fast.convergence_epoch <= stats_slow.convergence_epoch);
    }

    #[test]
    fn test_weight_decay_effect() {
        let config_no_decay = TrainingConfig {
            weight_decay: 0.0,
            ..Default::default()
        };
        let config_with_decay = TrainingConfig {
            weight_decay: 0.01,
            ..Default::default()
        };

        let stats_no_decay = train_with_config(config_no_decay).unwrap();
        let stats_with_decay = train_with_config(config_with_decay).unwrap();

        // Weight decay should result in smaller weight norms
        assert!(stats_with_decay.weight_norm <= stats_no_decay.weight_norm);
    }
}
