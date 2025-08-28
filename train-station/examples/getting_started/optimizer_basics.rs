//! Optimizer Basics Example
//!
//! This example demonstrates how to use optimizers in Train Station:
//! - Setting up Adam optimizer with parameters
//! - Training a simple linear regression model
//! - Learning rate scheduling and monitoring
//! - Advanced training patterns and analysis
//!
//! # Learning Objectives
//!
//! - Understand optimizer setup and parameter management
//! - Learn to implement basic training loops
//! - Explore learning rate scheduling techniques
//! - Monitor training progress and convergence
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of tensor basics (see tensor_basics.rs)
//! - Familiarity with gradient descent concepts
//!
//! # Usage
//!
//! ```bash
//! cargo run --example optimizer_basics
//! ```

use train_station::{
    optimizers::{Adam, AdamConfig, Optimizer},
    Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Optimizer Basics Example ===\n");

    demonstrate_basic_optimizer_setup();
    demonstrate_linear_regression()?;
    demonstrate_advanced_training()?;
    demonstrate_learning_rate_scheduling()?;
    demonstrate_training_monitoring()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Demonstrate basic optimizer setup and parameter management
fn demonstrate_basic_optimizer_setup() {
    println!("--- Basic Optimizer Setup ---");

    // Create parameters that require gradients
    let weight = Tensor::randn(vec![3, 2], Some(42)).with_requires_grad();
    let bias = Tensor::zeros(vec![2]).with_requires_grad();

    println!("Created parameters:");
    println!(
        "  Weight: shape {:?}, requires_grad: {}",
        weight.shape().dims,
        weight.requires_grad()
    );
    println!(
        "  Bias: shape {:?}, requires_grad: {}",
        bias.shape().dims,
        bias.requires_grad()
    );

    // Create Adam optimizer with default configuration
    let mut optimizer = Adam::new();
    println!(
        "Created Adam optimizer with learning rate: {}",
        optimizer.learning_rate()
    );

    // Add parameters to optimizer
    optimizer.add_parameter(&weight);
    optimizer.add_parameter(&bias);
    println!(
        "Added {} parameters to optimizer",
        optimizer.parameter_count()
    );

    // Create optimizer with custom configuration
    let config = AdamConfig {
        learning_rate: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        amsgrad: false,
    };

    let mut custom_optimizer = Adam::with_config(config);
    custom_optimizer.add_parameter(&weight);
    custom_optimizer.add_parameter(&bias);

    println!(
        "Created custom optimizer with learning rate: {}",
        custom_optimizer.learning_rate()
    );

    // Demonstrate parameter linking
    println!("Parameter linking completed successfully");
}

/// Demonstrate simple linear regression training
fn demonstrate_linear_regression() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Linear Regression Training ---");

    // Create model parameters
    let mut weight = Tensor::randn(vec![1, 1], Some(43)).with_requires_grad();
    let mut bias = Tensor::zeros(vec![1]).with_requires_grad();

    // Create optimizer
    let mut optimizer = Adam::with_learning_rate(0.01);
    optimizer.add_parameter(&weight);
    optimizer.add_parameter(&bias);

    // Create simple training data: y = 2*x + 1
    let x_data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
    let y_true = Tensor::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0], vec![5, 1]).unwrap();

    println!("Training data:");
    println!("  X: {:?}", x_data.data());
    println!("  Y: {:?}", y_true.data());
    println!("  Target: y = 2*x + 1");

    // Training loop
    let num_epochs = 100;
    let mut losses = Vec::new();

    for epoch in 0..num_epochs {
        // Forward pass: y_pred = x * weight + bias
        let y_pred = x_data.matmul(&weight) + &bias;

        // Compute loss: MSE
        let mut loss = (&y_pred - &y_true).pow_scalar(2.0).mean();

        // Backward pass
        loss.backward(None);

        // Optimizer step
        optimizer.step(&mut [&mut weight, &mut bias]);
        optimizer.zero_grad(&mut [&mut weight, &mut bias]);

        losses.push(loss.value());

        // Print progress every 20 epochs
        if epoch % 20 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {:3}: Loss = {:.6}", epoch, loss.value());
        }
    }

    // Evaluate final model
    let final_predictions = x_data.matmul(&weight) + &bias;
    println!("\nFinal model evaluation:");
    println!("  Learned weight: {:.6}", weight.value());
    println!("  Learned bias: {:.6}", bias.value());
    println!("  Predictions vs True:");

    for i in 0..5 {
        let x1 = x_data.data()[i];
        let pred = final_predictions.data()[i];
        let true_val = y_true.data()[i];
        println!(
            "    x={:.1}: pred={:.3}, true={:.1}, error={:.3}",
            x1,
            pred,
            true_val,
            (pred - true_val).abs()
        );
    }

    Ok(())
}

/// Demonstrate advanced training patterns
fn demonstrate_advanced_training() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Advanced Training Patterns ---");

    // Create a more complex model
    let mut weight = Tensor::randn(vec![1, 2], Some(44)).with_requires_grad();
    let mut bias = Tensor::zeros(vec![2]).with_requires_grad();

    // Create optimizer with different learning rate
    let mut optimizer = Adam::with_learning_rate(0.005);
    optimizer.add_parameter(&weight);
    optimizer.add_parameter(&bias);

    // Create training data: y = 2*x + [1, 3]
    let x_data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
    let y_true = Tensor::from_slice(
        &[3.0, 5.0, 7.0, 9.0, 11.0, 6.0, 8.0, 10.0, 12.0, 14.0],
        vec![5, 2],
    )
    .unwrap();

    println!("Advanced training with monitoring:");
    println!("  Initial learning rate: {}", optimizer.learning_rate());

    // Training loop with monitoring
    let num_epochs = 50;
    let mut losses = Vec::new();
    let mut weight_norms = Vec::new();
    let mut gradient_norms = Vec::new();

    for epoch in 0..num_epochs {
        // Forward pass
        let y_pred = x_data.matmul(&weight) + &bias;
        let mut loss = (&y_pred - &y_true).pow_scalar(2.0).mean();

        // Backward pass
        loss.backward(None);

        // Compute gradient norm before optimizer step
        let gradient_norm = weight.grad_by_value().unwrap().norm();

        // Optimizer step
        optimizer.step(&mut [&mut weight, &mut bias]);
        optimizer.zero_grad(&mut [&mut weight, &mut bias]);

        // Learning rate scheduling: reduce every 10 epochs
        if epoch > 0 && epoch % 10 == 0 {
            let current_lr = optimizer.learning_rate();
            let new_lr = current_lr * 0.5;
            optimizer.set_learning_rate(new_lr);
            println!(
                "Epoch {:2}: Reduced learning rate from {:.3} to {:.3}",
                epoch, current_lr, new_lr
            );
        }

        // Record metrics
        losses.push(loss.value());
        weight_norms.push(weight.norm().value());
        gradient_norms.push(gradient_norm.value());

        // Print detailed progress
        if epoch % 10 == 0 || epoch == num_epochs - 1 {
            println!(
                "Epoch {:2}: Loss = {:.6}, Weight Norm = {:.6}, Gradient Norm = {:.6}",
                epoch,
                loss.value(),
                weight.norm().value(),
                gradient_norm.value()
            );
        }
    }

    println!("Final learning rate: {}", optimizer.learning_rate());

    // Analyze training progression
    let initial_loss = losses[0];
    let final_loss = losses[losses.len() - 1];
    let loss_reduction = (initial_loss - final_loss) / initial_loss * 100.0;

    println!("\nTraining Analysis:");
    println!("  Initial loss: {:.6}", initial_loss);
    println!("  Final loss: {:.6}", final_loss);
    println!("  Loss reduction: {:.1}%", loss_reduction);
    println!("  Final weight norm: {:.6}", weight.norm().value());
    println!("  Final bias: {:?}", bias.data());

    Ok(())
}

/// Demonstrate learning rate scheduling
fn demonstrate_learning_rate_scheduling() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Learning Rate Scheduling ---");

    // Create simple model
    let mut weight = Tensor::randn(vec![1, 1], Some(45)).with_requires_grad();
    let mut bias = Tensor::zeros(vec![1]).with_requires_grad();

    // Create optimizer with high initial learning rate
    let mut optimizer = Adam::with_learning_rate(0.1);
    optimizer.add_parameter(&weight);
    optimizer.add_parameter(&bias);

    // Simple data
    let x_data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let y_true = Tensor::from_slice(&[2.0, 4.0, 6.0], vec![3, 1]).unwrap();

    println!("Initial learning rate: {}", optimizer.learning_rate());

    // Training loop with learning rate scheduling
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

        // Learning rate scheduling: reduce every 10 epochs
        if epoch > 0 && epoch % 10 == 0 {
            let current_lr = optimizer.learning_rate();
            let new_lr = current_lr * 0.5;
            optimizer.set_learning_rate(new_lr);
            println!(
                "Epoch {:2}: Reduced learning rate from {:.3} to {:.3}",
                epoch, current_lr, new_lr
            );
        }

        losses.push(loss.value());

        // Print progress
        if epoch % 10 == 0 || epoch == num_epochs - 1 {
            println!(
                "Epoch {:2}: Loss = {:.6}, LR = {:.3}",
                epoch,
                loss.value(),
                optimizer.learning_rate()
            );
        }
    }

    println!("Final learning rate: {}", optimizer.learning_rate());

    Ok(())
}

/// Demonstrate training monitoring and analysis
fn demonstrate_training_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Training Monitoring ---");

    // Create model
    let mut weight = Tensor::randn(vec![1, 1], Some(46)).with_requires_grad();
    let mut bias = Tensor::zeros(vec![1]).with_requires_grad();

    // Create optimizer
    let mut optimizer = Adam::with_learning_rate(0.01);
    optimizer.add_parameter(&weight);
    optimizer.add_parameter(&bias);

    // Training data
    let x_data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
    let y_true = Tensor::from_slice(&[3.0, 5.0, 7.0, 9.0], vec![4, 1]).unwrap();

    // Training loop with comprehensive monitoring
    let num_epochs = 30;
    let mut losses = Vec::new();
    let mut weight_history = Vec::new();
    let mut bias_history = Vec::new();

    for epoch in 0..num_epochs {
        // Forward pass
        let y_pred = x_data.matmul(&weight) + &bias;
        let mut loss = (&y_pred - &y_true).pow_scalar(2.0).mean();

        // Backward pass
        loss.backward(None);

        // Optimizer step
        optimizer.step(&mut [&mut weight, &mut bias]);
        optimizer.zero_grad(&mut [&mut weight, &mut bias]);

        // Record history
        losses.push(loss.value());
        weight_history.push(weight.value());
        bias_history.push(bias.value());

        // Print detailed monitoring
        if epoch % 5 == 0 || epoch == num_epochs - 1 {
            println!(
                "Epoch {:2}: Loss = {:.6}, Weight = {:.6}, Bias = {:.6}",
                epoch,
                loss.value(),
                weight.value(),
                bias.value()
            );
        }
    }

    // Analyze training progression
    println!("\nTraining Analysis:");
    println!("  Initial loss: {:.6}", losses[0]);
    println!("  Final loss: {:.6}", losses[losses.len() - 1]);
    println!(
        "  Loss reduction: {:.1}%",
        (losses[0] - losses[losses.len() - 1]) / losses[0] * 100.0
    );

    // Compute statistics
    let loss_mean = compute_mean(&losses);
    let loss_std = compute_std(&losses);
    let weight_change = (weight_history[weight_history.len() - 1] - weight_history[0]).abs();
    let bias_change = (bias_history[bias_history.len() - 1] - bias_history[0]).abs();

    println!("  Average loss: {:.6} ± {:.6}", loss_mean, loss_std);
    println!("  Weight change: {:.6}", weight_change);
    println!("  Bias change: {:.6}", bias_change);
    println!("  Final weight norm: {:.6}", weight.norm().value());
    println!("  Final bias: {:.6}", bias.value());

    Ok(())
}

/// Compute mean of a vector of f32 values
fn compute_mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

/// Compute standard deviation of a vector of f32 values
fn compute_std(values: &[f32]) -> f32 {
    let mean = compute_mean(values);
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic optimizer functionality
    #[test]
    fn test_basic_optimizer() {
        let mut weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);

        // Simulate a training step
        let mut loss = weight.sum();
        loss.backward(None);
        optimizer.step(&mut [&mut weight]);
        optimizer.zero_grad(&mut [&mut weight]);

        assert_eq!(optimizer.parameter_count(), 1);
        assert!(optimizer.learning_rate() > 0.0);
    }

    /// Test linear regression training
    #[test]
    fn test_linear_regression() {
        let mut weight = Tensor::randn(vec![1, 1], Some(47)).with_requires_grad();
        let mut bias = Tensor::zeros(vec![1]).with_requires_grad();

        let mut optimizer = Adam::with_learning_rate(0.01);
        optimizer.add_parameter(&weight);
        optimizer.add_parameter(&bias);

        let x = Tensor::ones(vec![1, 1]);
        let y_true = Tensor::ones(vec![1, 1]);

        // Single training step
        let y_pred = x.matmul(&weight) + &bias;
        let mut loss = (&y_pred - &y_true).pow_scalar(2.0).mean();

        loss.backward(None);
        optimizer.step(&mut [&mut weight, &mut bias]);
        optimizer.zero_grad(&mut [&mut weight, &mut bias]);

        // Loss should be finite
        assert!(loss.value().is_finite());
    }

    /// Test learning rate scheduling
    #[test]
    fn test_learning_rate_scheduling() {
        let mut optimizer = Adam::with_learning_rate(0.1);
        assert_eq!(optimizer.learning_rate(), 0.1);

        optimizer.set_learning_rate(0.05);
        assert_eq!(optimizer.learning_rate(), 0.05);
    }
}
