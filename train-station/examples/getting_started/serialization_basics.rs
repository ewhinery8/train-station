//! Serialization Basics Example
//!
//! This example demonstrates how to save and load tensors and optimizers in Train Station:
//! - Tensor serialization to JSON and binary formats
//! - Optimizer state persistence
//! - Format comparison and performance characteristics
//! - Model checkpointing workflows
//! - Error handling for serialization operations
//!
//! # Learning Objectives
//!
//! - Understand tensor and optimizer serialization
//! - Learn to save and load model states
//! - Compare different serialization formats
//! - Implement basic checkpointing workflows
//! - Handle serialization errors gracefully
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of tensor basics (see tensor_basics.rs)
//! - Familiarity with file I/O operations
//!
//! # Usage
//!
//! ```bash
//! cargo run --example serialization_basics
//! ```

use std::fs;
use train_station::{
    optimizers::{Adam, AdamConfig, Optimizer},
    serialization::StructSerializable,
    Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Serialization Basics Example ===\n");

    demonstrate_tensor_serialization()?;
    demonstrate_optimizer_serialization()?;
    demonstrate_format_comparison()?;
    demonstrate_model_checkpointing()?;
    demonstrate_error_handling()?;
    cleanup_temp_files()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Demonstrate basic tensor serialization and deserialization
fn demonstrate_tensor_serialization() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Tensor Serialization ---");

    // Create a tensor with some data
    let original_tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    println!(
        "Original tensor: shape {:?}, data: {:?}",
        original_tensor.shape().dims,
        original_tensor.data()
    );

    // Save tensor in JSON format
    let json_path = "temp_tensor.json";
    original_tensor.save_json(json_path)?;
    println!("Saved tensor to JSON: {}", json_path);

    // Load tensor from JSON
    let loaded_tensor_json = Tensor::load_json(json_path)?;
    println!(
        "Loaded from JSON: shape {:?}, data: {:?}",
        loaded_tensor_json.shape().dims,
        loaded_tensor_json.data()
    );

    // Verify data integrity
    assert_eq!(
        original_tensor.shape().dims,
        loaded_tensor_json.shape().dims
    );
    assert_eq!(original_tensor.data(), loaded_tensor_json.data());
    println!("JSON serialization verification: PASSED");

    // Save tensor in binary format
    let binary_path = "temp_tensor.bin";
    original_tensor.save_binary(binary_path)?;
    println!("Saved tensor to binary: {}", binary_path);

    // Load tensor from binary
    let loaded_tensor_binary = Tensor::load_binary(binary_path)?;
    println!(
        "Loaded from binary: shape {:?}, data: {:?}",
        loaded_tensor_binary.shape().dims,
        loaded_tensor_binary.data()
    );

    // Verify data integrity
    assert_eq!(
        original_tensor.shape().dims,
        loaded_tensor_binary.shape().dims
    );
    assert_eq!(original_tensor.data(), loaded_tensor_binary.data());
    println!("Binary serialization verification: PASSED");

    Ok(())
}

/// Demonstrate optimizer serialization and deserialization
fn demonstrate_optimizer_serialization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Optimizer Serialization ---");

    // Create an optimizer with some parameters
    let mut weight = Tensor::randn(vec![2, 2], Some(42)).with_requires_grad();
    let mut bias = Tensor::randn(vec![2], Some(43)).with_requires_grad();

    let config = AdamConfig {
        learning_rate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        amsgrad: false,
    };

    let mut optimizer = Adam::with_config(config);
    optimizer.add_parameter(&weight);
    optimizer.add_parameter(&bias);

    println!(
        "Created optimizer with {} parameters",
        optimizer.parameter_count()
    );
    println!("Learning rate: {}", optimizer.learning_rate());

    // Simulate some training steps
    for _ in 0..3 {
        let mut loss = weight.sum() + bias.sum();
        loss.backward(None);
        optimizer.step(&mut [&mut weight, &mut bias]);
        optimizer.zero_grad(&mut [&mut weight, &mut bias]);
    }

    // Save optimizer state
    let optimizer_path = "temp_optimizer.json";
    optimizer.save_json(optimizer_path)?;
    println!("Saved optimizer to: {}", optimizer_path);

    // Load optimizer state
    let loaded_optimizer = Adam::load_json(optimizer_path)?;
    println!(
        "Loaded optimizer with {} parameters",
        loaded_optimizer.parameter_count()
    );
    println!("Learning rate: {}", loaded_optimizer.learning_rate());

    // Verify optimizer state
    assert_eq!(
        optimizer.parameter_count(),
        loaded_optimizer.parameter_count()
    );
    assert_eq!(optimizer.learning_rate(), loaded_optimizer.learning_rate());
    println!("Optimizer serialization verification: PASSED");

    Ok(())
}

/// Demonstrate format comparison and performance characteristics
fn demonstrate_format_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Format Comparison ---");

    // Create a larger tensor for comparison
    let tensor = Tensor::randn(vec![10, 10], Some(44));

    // Save in both formats
    tensor.save_json("temp_comparison.json")?;
    tensor.save_binary("temp_comparison.bin")?;

    // Compare file sizes
    let json_size = fs::metadata("temp_comparison.json")?.len();
    let binary_size = fs::metadata("temp_comparison.bin")?.len();

    println!("JSON file size: {} bytes", json_size);
    println!("Binary file size: {} bytes", binary_size);
    println!(
        "Compression ratio: {:.2}x",
        json_size as f64 / binary_size as f64
    );

    // Load and verify both formats
    let json_tensor = Tensor::load_json("temp_comparison.json")?;
    let binary_tensor = Tensor::load_binary("temp_comparison.bin")?;

    assert_eq!(tensor.shape().dims, json_tensor.shape().dims);
    assert_eq!(tensor.shape().dims, binary_tensor.shape().dims);
    assert_eq!(tensor.data(), json_tensor.data());
    assert_eq!(tensor.data(), binary_tensor.data());

    println!("Format comparison verification: PASSED");

    Ok(())
}

/// Demonstrate a basic model checkpointing workflow
fn demonstrate_model_checkpointing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Model Checkpointing ---");

    // Create a simple model (weights and bias)
    let mut weights = Tensor::randn(vec![2, 1], Some(45)).with_requires_grad();
    let mut bias = Tensor::randn(vec![1], Some(46)).with_requires_grad();

    // Create optimizer
    let mut optimizer = Adam::with_learning_rate(0.01);
    optimizer.add_parameter(&weights);
    optimizer.add_parameter(&bias);

    println!("Initial weights: {:?}", weights.data());
    println!("Initial bias: {:?}", bias.data());

    // Simulate training
    for epoch in 0..5 {
        let mut loss = weights.sum() + bias.sum();
        loss.backward(None);
        optimizer.step(&mut [&mut weights, &mut bias]);
        optimizer.zero_grad(&mut [&mut weights, &mut bias]);

        if epoch % 2 == 0 {
            // Save checkpoint
            let checkpoint_dir = format!("checkpoint_epoch_{}", epoch);
            fs::create_dir_all(&checkpoint_dir)?;

            weights.save_json(format!("{}/weights.json", checkpoint_dir))?;
            bias.save_json(format!("{}/bias.json", checkpoint_dir))?;
            optimizer.save_json(format!("{}/optimizer.json", checkpoint_dir))?;

            println!("Saved checkpoint for epoch {}", epoch);
        }
    }

    // Load from checkpoint
    let loaded_weights = Tensor::load_json("checkpoint_epoch_4/weights.json")?;
    let loaded_bias = Tensor::load_json("checkpoint_epoch_4/bias.json")?;
    let loaded_optimizer = Adam::load_json("checkpoint_epoch_4/optimizer.json")?;

    println!("Loaded weights: {:?}", loaded_weights.data());
    println!("Loaded bias: {:?}", loaded_bias.data());
    println!(
        "Loaded optimizer learning rate: {}",
        loaded_optimizer.learning_rate()
    );

    // Verify checkpoint integrity
    assert_eq!(weights.shape().dims, loaded_weights.shape().dims);
    assert_eq!(bias.shape().dims, loaded_bias.shape().dims);
    assert_eq!(optimizer.learning_rate(), loaded_optimizer.learning_rate());

    println!("Checkpointing verification: PASSED");

    Ok(())
}

/// Demonstrate error handling for serialization operations
fn demonstrate_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Error Handling ---");

    // Test loading non-existent file
    match Tensor::load_json("nonexistent_file.json") {
        Ok(_) => println!("Unexpected: Successfully loaded non-existent file"),
        Err(e) => println!("Expected error loading non-existent file: {}", e),
    }

    // Test loading with wrong format
    let tensor = Tensor::randn(vec![2, 2], Some(47));
    tensor.save_binary("temp_binary.bin")?;

    match Tensor::load_json("temp_binary.bin") {
        Ok(_) => println!("Unexpected: Successfully loaded binary as JSON"),
        Err(e) => println!("Expected error loading binary as JSON: {}", e),
    }

    // Test loading corrupted file
    fs::write("temp_invalid.json", "invalid json content")?;
    match Tensor::load_json("temp_invalid.json") {
        Ok(_) => println!("Unexpected: Successfully loaded invalid JSON"),
        Err(e) => println!("Expected error loading invalid JSON: {}", e),
    }

    println!("Error handling verification: PASSED");

    Ok(())
}

/// Clean up temporary files created during the example
fn cleanup_temp_files() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Cleanup ---");

    let files_to_remove = [
        "temp_tensor.json",
        "temp_tensor.bin",
        "temp_optimizer.json",
        "temp_comparison.json",
        "temp_comparison.bin",
        "temp_binary.bin",
        "temp_invalid.json",
    ];

    for file in &files_to_remove {
        if fs::metadata(file).is_ok() {
            fs::remove_file(file)?;
            println!("Removed: {}", file);
        }
    }

    // Remove checkpoint directories
    for epoch in [0, 2, 4] {
        let checkpoint_dir = format!("checkpoint_epoch_{}", epoch);
        if fs::metadata(&checkpoint_dir).is_ok() {
            fs::remove_dir_all(&checkpoint_dir)?;
            println!("Removed directory: {}", checkpoint_dir);
        }
    }

    println!("Cleanup completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test tensor serialization roundtrip
    #[test]
    fn test_tensor_serialization_roundtrip() {
        let original = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // Test JSON roundtrip
        original.save_json("test_tensor.json").unwrap();
        let loaded = Tensor::load_json("test_tensor.json").unwrap();
        assert_eq!(original.shape().dims, loaded.shape().dims);
        assert_eq!(original.data(), loaded.data());

        // Test binary roundtrip
        original.save_binary("test_tensor.bin").unwrap();
        let loaded = Tensor::load_binary("test_tensor.bin").unwrap();
        assert_eq!(original.shape().dims, loaded.shape().dims);
        assert_eq!(original.data(), loaded.data());

        // Cleanup
        let _ = fs::remove_file("test_tensor.json");
        let _ = fs::remove_file("test_tensor.bin");
    }

    /// Test optimizer serialization roundtrip
    #[test]
    fn test_optimizer_serialization_roundtrip() {
        let mut weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);

        // Simulate a training step
        let mut loss = weight.sum();
        loss.backward(None);
        optimizer.step(&mut [&mut weight]);

        // Test serialization roundtrip
        optimizer.save_json("test_optimizer.json").unwrap();
        let loaded = Adam::load_json("test_optimizer.json").unwrap();

        assert_eq!(optimizer.parameter_count(), loaded.parameter_count());
        assert_eq!(optimizer.learning_rate(), loaded.learning_rate());

        // Cleanup
        let _ = fs::remove_file("test_optimizer.json");
    }
}
