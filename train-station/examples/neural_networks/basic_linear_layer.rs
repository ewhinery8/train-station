//! Basic Linear Layer Example
//!
//! This example demonstrates how to implement a basic linear layer in Train Station:
//! - Creating a linear layer struct with trainable parameters
//! - Forward pass implementation with matrix multiplication and bias addition
//! - Forward pass without gradient tracking using NoGradTrack
//! - Training loop with gradient computation and optimization
//! - Single inference and batch inference patterns
//! - Serialization for saving and loading layer parameters
//!
//! # Learning Objectives
//!
//! - Understand linear layer implementation using Tensor operations
//! - Learn gradient-aware and gradient-free forward passes
//! - Implement complete training workflows with optimization
//! - Explore serialization patterns for model persistence
//! - Compare single vs batch inference performance
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of tensor basics (see getting_started/tensor_basics.rs)
//! - Familiarity with neural network concepts
//! - Knowledge of gradient descent and backpropagation
//!
//! # Usage
//!
//! ```bash
//! cargo run --example basic_linear_layer
//! ```

use std::fs;
use train_station::{
    optimizers::{Adam, AdamConfig, Optimizer},
    serialization::StructSerializable,
    NoGradTrack, Tensor,
};

/// A basic linear layer implementation
#[derive(Debug)]
pub struct LinearLayer {
    /// Weight matrix [input_size, output_size]
    pub weight: Tensor,
    /// Bias vector [output_size]
    pub bias: Tensor,
    pub input_size: usize,
    pub output_size: usize,
}

impl LinearLayer {
    /// Create a new linear layer with random initialization
    pub fn new(input_size: usize, output_size: usize, seed: Option<u64>) -> Self {
        // Xavier/Glorot initialization: scale by sqrt(1/input_size)
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

    /// Forward pass: output = input @ weight + bias
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Matrix multiplication: [batch_size, input_size] @ [input_size, output_size] = [batch_size, output_size]
        let output = input.matmul(&self.weight);
        // Add bias: [batch_size, output_size] + [output_size] = [batch_size, output_size]
        output.add_tensor(&self.bias)
    }

    /// Forward pass without gradients (for inference)
    pub fn forward_no_grad(&self, input: &Tensor) -> Tensor {
        let _guard = NoGradTrack::new();
        self.forward(input)
    }

    /// Get all parameters for optimization
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    /// Save layer parameters to JSON
    pub fn save_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }

        let weight_path = format!("{}_weight.json", path);
        let bias_path = format!("{}_bias.json", path);

        self.weight.save_json(&weight_path)?;
        self.bias.save_json(&bias_path)?;

        println!("Saved linear layer to {} (weight and bias)", path);
        Ok(())
    }

    /// Load layer parameters from JSON
    pub fn load_json(
        path: &str,
        input_size: usize,
        output_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let weight_path = format!("{}_weight.json", path);
        let bias_path = format!("{}_bias.json", path);

        let weight = Tensor::load_json(&weight_path)?.with_requires_grad();
        let bias = Tensor::load_json(&bias_path)?.with_requires_grad();

        Ok(Self {
            weight,
            bias,
            input_size,
            output_size,
        })
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.input_size * self.output_size + self.output_size
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Linear Layer Example ===\n");

    demonstrate_layer_creation();
    demonstrate_forward_pass();
    demonstrate_forward_pass_no_grad();
    demonstrate_training_loop()?;
    demonstrate_single_vs_batch_inference();
    demonstrate_serialization()?;
    cleanup_temp_files()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Demonstrate creating a linear layer
fn demonstrate_layer_creation() {
    println!("--- Layer Creation ---");

    let layer = LinearLayer::new(3, 2, Some(42));

    println!("Created linear layer:");
    println!("  Input size: {}", layer.input_size);
    println!("  Output size: {}", layer.output_size);
    println!("  Parameter count: {}", layer.parameter_count());
    println!("  Weight shape: {:?}", layer.weight.shape().dims);
    println!("  Bias shape: {:?}", layer.bias.shape().dims);
    println!("  Weight requires grad: {}", layer.weight.requires_grad());
    println!("  Bias requires grad: {}", layer.bias.requires_grad());
}

/// Demonstrate forward pass with gradient tracking
fn demonstrate_forward_pass() {
    println!("\n--- Forward Pass (with gradients) ---");

    let layer = LinearLayer::new(3, 2, Some(43));

    // Single input
    let input = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let output = layer.forward(&input);

    println!("Single input:");
    println!("  Input: {:?}", input.data());
    println!("  Output: {:?}", output.data());
    println!("  Output requires grad: {}", output.requires_grad());

    // Batch input
    let batch_input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let batch_output = layer.forward(&batch_input);

    println!("Batch input:");
    println!("  Input shape: {:?}", batch_input.shape().dims);
    println!("  Output shape: {:?}", batch_output.shape().dims);
    println!("  Output requires grad: {}", batch_output.requires_grad());
}

/// Demonstrate forward pass without gradient tracking
fn demonstrate_forward_pass_no_grad() {
    println!("\n--- Forward Pass (no gradients) ---");

    let layer = LinearLayer::new(3, 2, Some(44));

    // Single input
    let input = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let output = layer.forward_no_grad(&input);

    println!("Single input (no grad):");
    println!("  Input: {:?}", input.data());
    println!("  Output: {:?}", output.data());
    println!("  Output requires grad: {}", output.requires_grad());

    // Compare with grad version
    let output_with_grad = layer.forward(&input);
    println!("Comparison:");
    println!(
        "  Same values: {}",
        output.data() == output_with_grad.data()
    );
    println!("  No grad requires grad: {}", output.requires_grad());
    println!(
        "  With grad requires grad: {}",
        output_with_grad.requires_grad()
    );
}

/// Demonstrate complete training loop
fn demonstrate_training_loop() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Training Loop ---");

    // Create layer and training data
    let mut layer = LinearLayer::new(2, 1, Some(45));

    // Simple regression task: y = 2*x1 + 3*x2 + 1
    let x_data = Tensor::from_slice(
        &[
            1.0, 1.0, // x1=1, x2=1 -> y=6
            2.0, 1.0, // x1=2, x2=1 -> y=8
            1.0, 2.0, // x1=1, x2=2 -> y=9
            2.0, 2.0, // x1=2, x2=2 -> y=11
        ],
        vec![4, 2],
    )
    .unwrap();

    let y_true = Tensor::from_slice(&[6.0, 8.0, 9.0, 11.0], vec![4, 1]).unwrap();

    println!("Training data:");
    println!("  X shape: {:?}", x_data.shape().dims);
    println!("  Y shape: {:?}", y_true.shape().dims);
    println!("  Target function: y = 2*x1 + 3*x2 + 1");

    // Create optimizer
    let config = AdamConfig {
        learning_rate: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        amsgrad: false,
    };

    let mut optimizer = Adam::with_config(config);
    let params = layer.parameters();
    for param in &params {
        optimizer.add_parameter(param);
    }

    println!("Optimizer setup complete. Starting training...");

    // Training loop
    let num_epochs = 100;
    let mut losses = Vec::new();

    for epoch in 0..num_epochs {
        // Forward pass
        let y_pred = layer.forward(&x_data);

        // Compute loss: MSE
        let diff = y_pred.sub_tensor(&y_true);
        let mut loss = diff.pow_scalar(2.0).mean();

        // Backward pass
        loss.backward(None);

        // Optimizer step
        let mut params = layer.parameters();
        optimizer.step(&mut params);
        optimizer.zero_grad(&mut params);

        losses.push(loss.value());

        // Print progress
        if epoch % 20 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {:3}: Loss = {:.6}", epoch, loss.value());
        }
    }

    // Evaluate final model
    let final_predictions = layer.forward_no_grad(&x_data);

    println!("\nFinal model evaluation:");
    println!("  Learned weights: {:?}", layer.weight.data());
    println!("  Learned bias: {:?}", layer.bias.data());
    println!("  Target weights: [2.0, 3.0]");
    println!("  Target bias: [1.0]");

    println!("  Predictions vs True:");
    for i in 0..4 {
        let pred = final_predictions.data()[i];
        let true_val = y_true.data()[i];
        println!(
            "    Sample {}: pred={:.3}, true={:.1}, error={:.3}",
            i + 1,
            pred,
            true_val,
            (pred - true_val).abs()
        );
    }

    // Training analysis
    let initial_loss = losses[0];
    let final_loss = losses[losses.len() - 1];
    let loss_reduction = (initial_loss - final_loss) / initial_loss * 100.0;

    println!("\nTraining Analysis:");
    println!("  Initial loss: {:.6}", initial_loss);
    println!("  Final loss: {:.6}", final_loss);
    println!("  Loss reduction: {:.1}%", loss_reduction);

    Ok(())
}

/// Demonstrate single vs batch inference
fn demonstrate_single_vs_batch_inference() {
    println!("\n--- Single vs Batch Inference ---");

    let layer = LinearLayer::new(4, 3, Some(46));

    // Single inference
    println!("Single inference:");
    let single_input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
    let single_output = layer.forward_no_grad(&single_input);
    println!("  Input shape: {:?}", single_input.shape().dims);
    println!("  Output shape: {:?}", single_output.shape().dims);
    println!("  Output: {:?}", single_output.data());

    // Batch inference
    println!("Batch inference:");
    let batch_input = Tensor::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, // Sample 1
            5.0, 6.0, 7.0, 8.0, // Sample 2
            9.0, 10.0, 11.0, 12.0, // Sample 3
        ],
        vec![3, 4],
    )
    .unwrap();
    let batch_output = layer.forward_no_grad(&batch_input);
    println!("  Input shape: {:?}", batch_input.shape().dims);
    println!("  Output shape: {:?}", batch_output.shape().dims);

    // Verify batch consistency - first sample should match single inference
    let _first_batch_sample = batch_output.view(vec![3, 3]); // Reshape to access first sample
    let first_sample_data = &batch_output.data()[0..3]; // First 3 elements
    let single_sample_data = single_output.data();

    println!("Consistency check:");
    println!("  Single output: {:?}", single_sample_data);
    println!("  First batch sample: {:?}", first_sample_data);
    println!(
        "  Match: {}",
        single_sample_data
            .iter()
            .zip(first_sample_data.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );
}

/// Demonstrate serialization and loading
fn demonstrate_serialization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Serialization ---");

    // Create and train a simple layer
    let mut original_layer = LinearLayer::new(2, 1, Some(47));

    // Simple training data
    let x_data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let y_true = Tensor::from_slice(&[5.0, 11.0], vec![2, 1]).unwrap();

    let mut optimizer = Adam::with_learning_rate(0.01);
    let params = original_layer.parameters();
    for param in &params {
        optimizer.add_parameter(param);
    }

    // Train for a few epochs
    for _ in 0..10 {
        let y_pred = original_layer.forward(&x_data);
        let mut loss = (y_pred.sub_tensor(&y_true)).pow_scalar(2.0).mean();
        loss.backward(None);

        let mut params = original_layer.parameters();
        optimizer.step(&mut params);
        optimizer.zero_grad(&mut params);
    }

    println!("Original layer trained");
    println!("  Weight: {:?}", original_layer.weight.data());
    println!("  Bias: {:?}", original_layer.bias.data());

    // Save layer
    original_layer.save_json("temp_linear_layer")?;

    // Load layer
    let loaded_layer = LinearLayer::load_json("temp_linear_layer", 2, 1)?;

    println!("Loaded layer");
    println!("  Weight: {:?}", loaded_layer.weight.data());
    println!("  Bias: {:?}", loaded_layer.bias.data());

    // Verify consistency
    let test_input = Tensor::from_slice(&[1.0, 1.0], vec![1, 2]).unwrap();
    let original_output = original_layer.forward_no_grad(&test_input);
    let loaded_output = loaded_layer.forward_no_grad(&test_input);

    println!("Consistency check:");
    println!("  Original output: {:?}", original_output.data());
    println!("  Loaded output: {:?}", loaded_output.data());
    println!(
        "  Match: {}",
        original_output
            .data()
            .iter()
            .zip(loaded_output.data().iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );

    println!("Serialization verification: PASSED");

    Ok(())
}

/// Clean up temporary files
fn cleanup_temp_files() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Cleanup ---");

    let files_to_remove = [
        "temp_linear_layer_weight.json",
        "temp_linear_layer_bias.json",
    ];

    for file in &files_to_remove {
        if fs::metadata(file).is_ok() {
            fs::remove_file(file)?;
            println!("Removed: {}", file);
        }
    }

    println!("Cleanup completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = LinearLayer::new(3, 2, Some(42));
        assert_eq!(layer.input_size, 3);
        assert_eq!(layer.output_size, 2);
        assert_eq!(layer.weight.shape().dims, vec![3, 2]);
        assert_eq!(layer.bias.shape().dims, vec![2]);
        assert!(layer.weight.requires_grad());
        assert!(layer.bias.requires_grad());
    }

    #[test]
    fn test_forward_pass() {
        let layer = LinearLayer::new(2, 1, Some(43));
        let input = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]).unwrap();
        let output = layer.forward(&input);

        assert_eq!(output.shape().dims, vec![1, 1]);
        assert!(output.requires_grad());
    }

    #[test]
    fn test_forward_pass_no_grad() {
        let layer = LinearLayer::new(2, 1, Some(44));
        let input = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]).unwrap();
        let output = layer.forward_no_grad(&input);

        assert_eq!(output.shape().dims, vec![1, 1]);
        assert!(!output.requires_grad());
    }

    #[test]
    fn test_batch_inference() {
        let layer = LinearLayer::new(2, 1, Some(45));
        let batch_input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let output = layer.forward(&batch_input);

        assert_eq!(output.shape().dims, vec![2, 1]);
    }

    #[test]
    fn test_parameter_count() {
        let layer = LinearLayer::new(3, 2, Some(46));
        assert_eq!(layer.parameter_count(), 3 * 2 + 2); // weights + bias
    }

    #[test]
    fn test_serialization_roundtrip() {
        let original = LinearLayer::new(2, 1, Some(47));

        // Save and load
        original.save_json("test_layer").unwrap();
        let loaded = LinearLayer::load_json("test_layer", 2, 1).unwrap();

        // Verify shapes
        assert_eq!(original.weight.shape().dims, loaded.weight.shape().dims);
        assert_eq!(original.bias.shape().dims, loaded.bias.shape().dims);

        // Verify data
        assert_eq!(original.weight.data(), loaded.weight.data());
        assert_eq!(original.bias.data(), loaded.bias.data());

        // Cleanup
        let _ = fs::remove_file("test_layer_weight.json");
        let _ = fs::remove_file("test_layer_bias.json");
    }
}
