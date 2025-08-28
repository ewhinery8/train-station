//! Feed-Forward Network Example
//!
//! This example demonstrates how to build a configurable multi-layer feed-forward network:
//! - Configurable input/output sizes and number of hidden layers
//! - Using the basic linear layer as building blocks
//! - ReLU activation function implementation
//! - Complete training workflow with gradient computation
//! - Comprehensive training loop with 100+ steps
//! - Proper gradient tracking and computation graph connectivity
//! - Zero gradient management between training steps
//!
//! # Learning Objectives
//!
//! - Understand multi-layer network architecture design
//! - Learn to compose linear layers into deeper networks
//! - Implement activation functions and their gradient properties
//! - Master training workflows with proper gradient management
//! - Explore configurable network architectures
//! - Understand gradient flow through multiple layers
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of tensor basics (see getting_started/tensor_basics.rs)
//! - Familiarity with neural network concepts
//! - Knowledge of the basic linear layer (see basic_linear_layer.rs)
//! - Understanding of activation functions and backpropagation
//!
//! # Usage
//!
//! ```bash
//! cargo run --example feedforward_network
//! ```

use std::fs;
use train_station::{
    optimizers::{Adam, Optimizer},
    serialization::StructSerializable,
    NoGradTrack, Tensor,
};

/// ReLU activation function
pub struct ReLU;

impl ReLU {
    /// Apply ReLU activation: max(0, x)
    pub fn forward(input: &Tensor) -> Tensor {
        input.relu()
    }

    /// Apply ReLU activation without gradients
    pub fn forward_no_grad(input: &Tensor) -> Tensor {
        let _guard = NoGradTrack::new();
        Self::forward(input)
    }
}

/// A basic linear layer implementation (reused from basic_linear_layer.rs)
#[derive(Debug)]
pub struct LinearLayer {
    pub weight: Tensor,
    pub bias: Tensor,
    pub input_size: usize,
    pub output_size: usize,
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize, seed: Option<u64>) -> Self {
        let scale = (1.0 / input_size as f32).sqrt();

        let weight = Tensor::randn(vec![input_size, output_size], seed)
            .mul_scalar(scale)
            .with_requires_grad();
        let bias = Tensor::zeros(vec![output_size]).with_requires_grad();

        Self {
            weight,
            bias,
            input_size,
            output_size,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let output = input.matmul(&self.weight);
        output.add_tensor(&self.bias)
    }

    pub fn forward_no_grad(&self, input: &Tensor) -> Tensor {
        let _guard = NoGradTrack::new();
        self.forward(input)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

/// Configuration for feed-forward network
#[derive(Debug, Clone)]
pub struct FeedForwardConfig {
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub output_size: usize,
    pub use_bias: bool,
}

impl Default for FeedForwardConfig {
    fn default() -> Self {
        Self {
            input_size: 4,
            hidden_sizes: vec![8, 4],
            output_size: 2,
            use_bias: true,
        }
    }
}

/// A configurable feed-forward neural network
pub struct FeedForwardNetwork {
    layers: Vec<LinearLayer>,
    config: FeedForwardConfig,
}

impl FeedForwardNetwork {
    /// Create a new feed-forward network with the given configuration
    pub fn new(config: FeedForwardConfig, seed: Option<u64>) -> Self {
        let mut layers = Vec::new();
        let mut current_size = config.input_size;
        let mut current_seed = seed;

        // Create hidden layers
        for &hidden_size in &config.hidden_sizes {
            layers.push(LinearLayer::new(current_size, hidden_size, current_seed));
            current_size = hidden_size;
            current_seed = current_seed.map(|s| s + 1);
        }

        // Create output layer
        layers.push(LinearLayer::new(
            current_size,
            config.output_size,
            current_seed,
        ));

        Self { layers, config }
    }

    /// Forward pass through the entire network
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        // Pass through all layers except the last one with ReLU activation
        for layer in &self.layers[..self.layers.len() - 1] {
            x = layer.forward(&x);
            x = ReLU::forward(&x);
        }

        // Final layer without activation (raw logits)
        if let Some(final_layer) = self.layers.last() {
            x = final_layer.forward(&x);
        }

        x
    }

    /// Forward pass without gradients (for inference)
    pub fn forward_no_grad(&self, input: &Tensor) -> Tensor {
        let _guard = NoGradTrack::new();
        self.forward(input)
    }

    /// Get all parameters for optimization
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the total number of parameters
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        let mut current_size = self.config.input_size;

        for &hidden_size in &self.config.hidden_sizes {
            count += current_size * hidden_size + hidden_size; // weights + bias
            current_size = hidden_size;
        }

        // Output layer
        count += current_size * self.config.output_size + self.config.output_size;

        count
    }

    /// Save network parameters to JSON
    pub fn save_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_path = format!("{}_layer_{}", path, i);
            let weight_path = format!("{}_weight.json", layer_path);
            let bias_path = format!("{}_bias.json", layer_path);

            layer.weight.save_json(&weight_path)?;
            layer.bias.save_json(&bias_path)?;
        }

        println!(
            "Saved feed-forward network to {} ({} layers)",
            path,
            self.layers.len()
        );
        Ok(())
    }

    /// Load network parameters from JSON
    pub fn load_json(
        path: &str,
        config: FeedForwardConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut layers = Vec::new();
        let mut current_size = config.input_size;
        let mut layer_idx = 0;

        // Load hidden layers
        for &hidden_size in &config.hidden_sizes {
            let layer_path = format!("{}_layer_{}", path, layer_idx);
            let weight_path = format!("{}_weight.json", layer_path);
            let bias_path = format!("{}_bias.json", layer_path);

            let weight = Tensor::load_json(&weight_path)?.with_requires_grad();
            let bias = Tensor::load_json(&bias_path)?.with_requires_grad();

            layers.push(LinearLayer {
                weight,
                bias,
                input_size: current_size,
                output_size: hidden_size,
            });

            current_size = hidden_size;
            layer_idx += 1;
        }

        // Load output layer
        let layer_path = format!("{}_layer_{}", path, layer_idx);
        let weight_path = format!("{}_weight.json", layer_path);
        let bias_path = format!("{}_bias.json", layer_path);

        let weight = Tensor::load_json(&weight_path)?.with_requires_grad();
        let bias = Tensor::load_json(&bias_path)?.with_requires_grad();

        layers.push(LinearLayer {
            weight,
            bias,
            input_size: current_size,
            output_size: config.output_size,
        });

        Ok(Self { layers, config })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Feed-Forward Network Example ===\n");

    demonstrate_network_creation();
    demonstrate_forward_pass();
    demonstrate_configurable_architectures();
    demonstrate_training_workflow()?;
    demonstrate_comprehensive_training()?;
    demonstrate_network_serialization()?;
    cleanup_temp_files()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Demonstrate creating different network configurations
fn demonstrate_network_creation() {
    println!("--- Network Creation ---");

    // Default configuration
    let config = FeedForwardConfig::default();
    let network = FeedForwardNetwork::new(config.clone(), Some(42));

    println!("Default network configuration:");
    println!("  Input size: {}", config.input_size);
    println!("  Hidden sizes: {:?}", config.hidden_sizes);
    println!("  Output size: {}", config.output_size);
    println!("  Number of layers: {}", network.num_layers());
    println!("  Total parameters: {}", network.parameter_count());

    // Custom configurations
    let configs = [
        FeedForwardConfig {
            input_size: 2,
            hidden_sizes: vec![4],
            output_size: 1,
            use_bias: true,
        },
        FeedForwardConfig {
            input_size: 8,
            hidden_sizes: vec![16, 8, 4],
            output_size: 3,
            use_bias: true,
        },
        FeedForwardConfig {
            input_size: 10,
            hidden_sizes: vec![20, 15, 10, 5],
            output_size: 2,
            use_bias: true,
        },
    ];

    for (i, config) in configs.iter().enumerate() {
        let network = FeedForwardNetwork::new(config.clone(), Some(42 + i as u64));
        println!("\nCustom network {}:", i + 1);
        println!(
            "  Architecture: {} -> {:?} -> {}",
            config.input_size, config.hidden_sizes, config.output_size
        );
        println!("  Layers: {}", network.num_layers());
        println!("  Parameters: {}", network.parameter_count());
    }
}

/// Demonstrate forward pass through the network
fn demonstrate_forward_pass() {
    println!("\n--- Forward Pass ---");

    let config = FeedForwardConfig {
        input_size: 3,
        hidden_sizes: vec![5, 3],
        output_size: 2,
        use_bias: true,
    };
    let network = FeedForwardNetwork::new(config, Some(43));

    // Single input
    let input = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let output = network.forward(&input);

    println!("Single input forward pass:");
    println!("  Input shape: {:?}", input.shape().dims);
    println!("  Output shape: {:?}", output.shape().dims);
    println!("  Output: {:?}", output.data());
    println!("  Output requires grad: {}", output.requires_grad());

    // Batch input
    let batch_input = Tensor::from_slice(
        &[
            1.0, 2.0, 3.0, // Sample 1
            4.0, 5.0, 6.0, // Sample 2
            7.0, 8.0, 9.0, // Sample 3
        ],
        vec![3, 3],
    )
    .unwrap();
    let batch_output = network.forward(&batch_input);

    println!("Batch input forward pass:");
    println!("  Input shape: {:?}", batch_input.shape().dims);
    println!("  Output shape: {:?}", batch_output.shape().dims);
    println!("  Output requires grad: {}", batch_output.requires_grad());

    // Compare with no-grad version
    let output_no_grad = network.forward_no_grad(&input);
    println!("No-grad comparison:");
    println!("  Same values: {}", output.data() == output_no_grad.data());
    println!("  With grad requires grad: {}", output.requires_grad());
    println!(
        "  No grad requires grad: {}",
        output_no_grad.requires_grad()
    );
}

/// Demonstrate different configurable architectures
fn demonstrate_configurable_architectures() {
    println!("\n--- Configurable Architectures ---");

    let architectures = vec![
        ("Shallow", vec![8]),
        ("Medium", vec![16, 8]),
        ("Deep", vec![32, 16, 8, 4]),
        ("Wide", vec![64, 32]),
        ("Bottleneck", vec![16, 4, 16]),
    ];

    for (name, hidden_sizes) in architectures {
        let config = FeedForwardConfig {
            input_size: 10,
            hidden_sizes,
            output_size: 3,
            use_bias: true,
        };

        let network = FeedForwardNetwork::new(config.clone(), Some(44));

        // Test forward pass
        let test_input = Tensor::randn(vec![5, 10], Some(45)); // Batch of 5
        let output = network.forward_no_grad(&test_input);

        println!("{} network:", name);
        println!("  Architecture: 10 -> {:?} -> 3", config.hidden_sizes);
        println!("  Parameters: {}", network.parameter_count());
        println!("  Test output shape: {:?}", output.shape().dims);
        println!(
            "  Output range: [{:.3}, {:.3}]",
            output.data().iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            output
                .data()
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        );
    }
}

/// Demonstrate basic training workflow
fn demonstrate_training_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Training Workflow ---");

    // Create a simple classification network
    let config = FeedForwardConfig {
        input_size: 2,
        hidden_sizes: vec![4, 3],
        output_size: 1,
        use_bias: true,
    };
    let mut network = FeedForwardNetwork::new(config, Some(46));

    println!("Training network: 2 -> [4, 3] -> 1");

    // Create simple binary classification data: XOR problem
    let x_data = Tensor::from_slice(
        &[
            0.0, 0.0, // -> 0
            0.0, 1.0, // -> 1
            1.0, 0.0, // -> 1
            1.0, 1.0, // -> 0
        ],
        vec![4, 2],
    )
    .unwrap();

    let y_true = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], vec![4, 1]).unwrap();

    println!("Training on XOR problem:");
    println!("  Input shape: {:?}", x_data.shape().dims);
    println!("  Target shape: {:?}", y_true.shape().dims);

    // Create optimizer
    let mut optimizer = Adam::with_learning_rate(0.1);
    let params = network.parameters();
    for param in &params {
        optimizer.add_parameter(param);
    }

    // Training loop
    let num_epochs = 50;
    let mut losses = Vec::new();

    for epoch in 0..num_epochs {
        // Forward pass
        let y_pred = network.forward(&x_data);

        // Compute loss: MSE
        let diff = y_pred.sub_tensor(&y_true);
        let mut loss = diff.pow_scalar(2.0).mean();

        // Backward pass
        loss.backward(None);

        // Optimizer step and zero grad
        let mut params = network.parameters();
        optimizer.step(&mut params);
        optimizer.zero_grad(&mut params);

        losses.push(loss.value());

        // Print progress
        if epoch % 10 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {:2}: Loss = {:.6}", epoch, loss.value());
        }
    }

    // Test final model
    let final_predictions = network.forward_no_grad(&x_data);
    println!("\nFinal predictions vs targets:");
    for i in 0..4 {
        let pred = final_predictions.data()[i];
        let target = y_true.data()[i];
        let input_x = x_data.data()[i * 2];
        let input_y = x_data.data()[i * 2 + 1];
        println!(
            "  [{:.0}, {:.0}] -> pred: {:.3}, target: {:.0}, error: {:.3}",
            input_x,
            input_y,
            pred,
            target,
            (pred - target).abs()
        );
    }

    Ok(())
}

/// Demonstrate comprehensive training with 100+ steps
fn demonstrate_comprehensive_training() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Comprehensive Training (100+ Steps) ---");

    // Create a regression network
    let config = FeedForwardConfig {
        input_size: 3,
        hidden_sizes: vec![8, 6, 4],
        output_size: 2,
        use_bias: true,
    };
    let mut network = FeedForwardNetwork::new(config, Some(47));

    println!("Network architecture: 3 -> [8, 6, 4] -> 2");
    println!("Total parameters: {}", network.parameter_count());

    // Create synthetic regression data
    // Target function: [y1, y2] = [x1 + 2*x2 - x3, x1*x2 + x3]
    let num_samples = 32;
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();

    for i in 0..num_samples {
        let x1 = (i as f32 / num_samples as f32) * 2.0 - 1.0; // [-1, 1]
        let x2 = ((i * 2) as f32 / num_samples as f32) * 2.0 - 1.0;
        let x3 = ((i * 3) as f32 / num_samples as f32) * 2.0 - 1.0;

        let y1 = x1 + 2.0 * x2 - x3;
        let y2 = x1 * x2 + x3;

        x_vec.extend_from_slice(&[x1, x2, x3]);
        y_vec.extend_from_slice(&[y1, y2]);
    }

    let x_data = Tensor::from_slice(&x_vec, vec![num_samples, 3]).unwrap();
    let y_true = Tensor::from_slice(&y_vec, vec![num_samples, 2]).unwrap();

    println!("Training data:");
    println!("  {} samples", num_samples);
    println!("  Input shape: {:?}", x_data.shape().dims);
    println!("  Target shape: {:?}", y_true.shape().dims);

    // Create optimizer with learning rate scheduling
    let mut optimizer = Adam::with_learning_rate(0.01);
    let params = network.parameters();
    for param in &params {
        optimizer.add_parameter(param);
    }

    // Comprehensive training loop (150 epochs)
    let num_epochs = 150;
    let mut losses = Vec::new();
    let mut best_loss = f32::INFINITY;
    let mut patience_counter = 0;
    let patience = 20;

    println!("Starting comprehensive training...");

    for epoch in 0..num_epochs {
        // Forward pass
        let y_pred = network.forward(&x_data);

        // Compute loss: MSE
        let diff = y_pred.sub_tensor(&y_true);
        let mut loss = diff.pow_scalar(2.0).mean();

        // Backward pass
        loss.backward(None);

        // Optimizer step and zero grad
        let mut params = network.parameters();
        optimizer.step(&mut params);
        optimizer.zero_grad(&mut params);

        let current_loss = loss.value();
        losses.push(current_loss);

        // Learning rate scheduling
        if epoch > 0 && epoch % 30 == 0 {
            let new_lr = optimizer.learning_rate() * 0.8;
            optimizer.set_learning_rate(new_lr);
            println!("  Reduced learning rate to {:.4}", new_lr);
        }

        // Early stopping logic
        if current_loss < best_loss {
            best_loss = current_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
        }

        // Print progress
        if epoch % 25 == 0 || epoch == num_epochs - 1 {
            println!(
                "Epoch {:3}: Loss = {:.6}, LR = {:.4}, Best = {:.6}",
                epoch,
                current_loss,
                optimizer.learning_rate(),
                best_loss
            );
        }

        // Early stopping
        if patience_counter >= patience && epoch > 50 {
            println!("Early stopping at epoch {} (patience exceeded)", epoch);
            break;
        }
    }

    // Final evaluation
    let final_predictions = network.forward_no_grad(&x_data);

    // Compute final metrics
    let final_loss = losses[losses.len() - 1];
    let initial_loss = losses[0];
    let loss_reduction = (initial_loss - final_loss) / initial_loss * 100.0;

    println!("\nTraining completed!");
    println!("  Initial loss: {:.6}", initial_loss);
    println!("  Final loss: {:.6}", final_loss);
    println!("  Best loss: {:.6}", best_loss);
    println!("  Loss reduction: {:.1}%", loss_reduction);
    println!("  Final learning rate: {:.4}", optimizer.learning_rate());

    // Sample predictions analysis
    println!("\nSample predictions (first 5):");
    for i in 0..5.min(num_samples) {
        let pred1 = final_predictions.data()[i * 2];
        let pred2 = final_predictions.data()[i * 2 + 1];
        let true1 = y_true.data()[i * 2];
        let true2 = y_true.data()[i * 2 + 1];

        println!(
            "  Sample {}: pred=[{:.3}, {:.3}], true=[{:.3}, {:.3}], error=[{:.3}, {:.3}]",
            i + 1,
            pred1,
            pred2,
            true1,
            true2,
            (pred1 - true1).abs(),
            (pred2 - true2).abs()
        );
    }

    Ok(())
}

/// Demonstrate network serialization
fn demonstrate_network_serialization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Network Serialization ---");

    // Create and train a network
    let config = FeedForwardConfig {
        input_size: 2,
        hidden_sizes: vec![4, 2],
        output_size: 1,
        use_bias: true,
    };
    let mut original_network = FeedForwardNetwork::new(config.clone(), Some(48));

    // Quick training
    let x_data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let y_true = Tensor::from_slice(&[5.0, 11.0], vec![2, 1]).unwrap();

    let mut optimizer = Adam::with_learning_rate(0.01);
    let params = original_network.parameters();
    for param in &params {
        optimizer.add_parameter(param);
    }

    for _ in 0..20 {
        let y_pred = original_network.forward(&x_data);
        let mut loss = (y_pred.sub_tensor(&y_true)).pow_scalar(2.0).mean();
        loss.backward(None);

        let mut params = original_network.parameters();
        optimizer.step(&mut params);
        optimizer.zero_grad(&mut params);
    }

    // Test original network
    let test_input = Tensor::from_slice(&[1.0, 1.0], vec![1, 2]).unwrap();
    let original_output = original_network.forward_no_grad(&test_input);

    println!("Original network output: {:?}", original_output.data());

    // Save network
    original_network.save_json("temp_feedforward_network")?;

    // Load network
    let loaded_network = FeedForwardNetwork::load_json("temp_feedforward_network", config)?;
    let loaded_output = loaded_network.forward_no_grad(&test_input);

    println!("Loaded network output: {:?}", loaded_output.data());

    // Verify consistency
    let match_check = original_output
        .data()
        .iter()
        .zip(loaded_output.data().iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);

    println!(
        "Serialization verification: {}",
        if match_check { "PASSED" } else { "FAILED" }
    );

    Ok(())
}

/// Clean up temporary files
fn cleanup_temp_files() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Cleanup ---");

    // Remove network files
    for i in 0..10 {
        // Assume max 10 layers
        let weight_file = format!("temp_feedforward_network_layer_{}_weight.json", i);
        let bias_file = format!("temp_feedforward_network_layer_{}_bias.json", i);

        if fs::metadata(&weight_file).is_ok() {
            fs::remove_file(&weight_file)?;
            println!("Removed: {}", weight_file);
        }
        if fs::metadata(&bias_file).is_ok() {
            fs::remove_file(&bias_file)?;
            println!("Removed: {}", bias_file);
        }
    }

    println!("Cleanup completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_activation() {
        let input = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![1, 5]).unwrap();
        let output = ReLU::forward(&input);
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

        assert_eq!(output.data(), &expected);
    }

    #[test]
    fn test_network_creation() {
        let config = FeedForwardConfig {
            input_size: 3,
            hidden_sizes: vec![5, 4],
            output_size: 2,
            use_bias: true,
        };
        let network = FeedForwardNetwork::new(config, Some(42));

        assert_eq!(network.num_layers(), 3); // 2 hidden + 1 output
        assert_eq!(network.parameter_count(), 3 * 5 + 5 + 5 * 4 + 4 + 4 * 2 + 2);
        // weights + biases
    }

    #[test]
    fn test_forward_pass() {
        let config = FeedForwardConfig {
            input_size: 2,
            hidden_sizes: vec![3],
            output_size: 1,
            use_bias: true,
        };
        let network = FeedForwardNetwork::new(config, Some(43));

        let input = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]).unwrap();
        let output = network.forward(&input);

        assert_eq!(output.shape().dims, vec![1, 1]);
        assert!(output.requires_grad());
    }

    #[test]
    fn test_batch_forward_pass() {
        let config = FeedForwardConfig {
            input_size: 2,
            hidden_sizes: vec![3],
            output_size: 1,
            use_bias: true,
        };
        let network = FeedForwardNetwork::new(config, Some(44));

        let batch_input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let output = network.forward(&batch_input);

        assert_eq!(output.shape().dims, vec![2, 1]);
    }

    #[test]
    fn test_no_grad_forward() {
        let config = FeedForwardConfig::default();
        let network = FeedForwardNetwork::new(config, Some(45));

        let input = Tensor::randn(vec![1, 4], Some(46));
        let output = network.forward_no_grad(&input);

        assert!(!output.requires_grad());
    }

    #[test]
    fn test_parameter_collection() {
        let config = FeedForwardConfig {
            input_size: 2,
            hidden_sizes: vec![3],
            output_size: 1,
            use_bias: true,
        };
        let mut network = FeedForwardNetwork::new(config, Some(47));

        let params = network.parameters();
        assert_eq!(params.len(), 4); // 2 layers * 2 parameters (weight + bias) each
    }
}
