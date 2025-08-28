//! Tensor Basics Example
//!
//! This example demonstrates fundamental tensor concepts in Train Station:
//! - Creating tensors with different initializations
//! - Basic arithmetic operations
//! - Shape manipulation and data access
//! - Utility functions and properties
//!
//! # Learning Objectives
//!
//! - Understand tensor creation and initialization
//! - Learn basic tensor operations and arithmetic
//! - Explore shape manipulation and data access patterns
//! - Discover utility functions for tensor analysis
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of multi-dimensional arrays
//!
//! # Usage
//!
//! ```bash
//! cargo run --example tensor_basics
//! ```

use train_station::Tensor;

fn main() {
    println!("=== Tensor Basics Example ===\n");

    demonstrate_tensor_creation();
    demonstrate_basic_operations();
    demonstrate_shape_operations();
    demonstrate_data_access();
    demonstrate_utility_functions();

    println!("\n=== Example completed successfully! ===");
}

/// Demonstrate different ways to create tensors
fn demonstrate_tensor_creation() {
    println!("--- Tensor Creation ---");

    // Create tensors with different initializations
    let zeros = Tensor::zeros(vec![2, 3]);
    println!(
        "Zeros tensor: shape {:?}, data: {:?}",
        zeros.shape().dims,
        zeros.data()
    );

    let ones = Tensor::ones(vec![3, 2]);
    println!(
        "Ones tensor: shape {:?}, data: {:?}",
        ones.shape().dims,
        ones.data()
    );

    // Create tensor from slice
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let from_slice = Tensor::from_slice(&data, vec![2, 3]).unwrap();
    println!(
        "From slice: shape {:?}, data: {:?}",
        from_slice.shape().dims,
        from_slice.data()
    );

    // Create tensor with specific value
    let mut filled = Tensor::new(vec![2, 2]);
    {
        let data = filled.data_mut();
        for value in data.iter_mut() {
            *value = 42.0;
        }
    }
    println!("Filled with 42: {:?}", filled.data());

    // Create tensor with random data
    let random = Tensor::randn(vec![2, 2], Some(42));
    println!(
        "Random tensor: shape {:?}, data: {:?}",
        random.shape().dims,
        random.data()
    );
}

/// Demonstrate basic arithmetic operations
fn demonstrate_basic_operations() {
    println!("\n--- Basic Operations ---");

    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    // Addition
    let sum = a.add_tensor(&b);
    println!("A + B: {:?}", sum.data());

    // Subtraction
    let diff = a.sub_tensor(&b);
    println!("A - B: {:?}", diff.data());

    // Multiplication
    let product = a.mul_tensor(&b);
    println!("A * B: {:?}", product.data());

    // Division
    let quotient = a.div_tensor(&b);
    println!("A / B: {:?}", quotient.data());

    // Scalar operations
    let scalar_add = a.add_scalar(5.0);
    println!("A + 5.0: {:?}", scalar_add.data());

    let scalar_mul = a.mul_scalar(2.0);
    println!("A * 2.0: {:?}", scalar_mul.data());
}

/// Demonstrate shape manipulation operations
fn demonstrate_shape_operations() {
    println!("\n--- Shape Operations ---");

    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    println!(
        "Original: shape {:?}, data: {:?}",
        tensor.shape().dims,
        tensor.data()
    );

    // Reshape (view)
    let reshaped = tensor.view(vec![3, 2]);
    println!(
        "Reshaped to [3, 2]: shape {:?}, data: {:?}",
        reshaped.shape().dims,
        reshaped.data()
    );

    // Create a different shaped tensor for demonstration
    let tensor_2d = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    println!(
        "2D tensor: shape {:?}, data: {:?}",
        tensor_2d.shape().dims,
        tensor_2d.data()
    );

    // Create a 1D tensor
    let tensor_1d = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    println!(
        "1D tensor: shape {:?}, data: {:?}",
        tensor_1d.shape().dims,
        tensor_1d.data()
    );
}

/// Demonstrate data access patterns
fn demonstrate_data_access() {
    println!("\n--- Data Access ---");

    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    // Access individual elements
    println!("Element [0, 0]: {}", tensor.get(&[0, 0]));
    println!("Element [0, 1]: {}", tensor.get(&[0, 1]));
    println!("Element [1, 0]: {}", tensor.get(&[1, 0]));
    println!("Element [1, 1]: {}", tensor.get(&[1, 1]));

    // Access data as slice
    let data = tensor.data();
    println!("Data as slice: {:?}", data);

    // Iterate over elements
    println!("Elements:");
    for (i, &value) in data.iter().enumerate() {
        println!("  [{}]: {}", i, value);
    }
}

/// Demonstrate utility functions
fn demonstrate_utility_functions() {
    println!("\n--- Utility Functions ---");

    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    // Basic properties
    println!("Shape: {:?}", tensor.shape().dims);
    println!("Size: {}", tensor.size());
    println!("Is contiguous: {}", tensor.is_contiguous());
    println!("Device: {:?}", tensor.device());

    // Mathematical operations
    let sum = tensor.sum();
    println!("Sum: {}", sum.value());

    let mean = tensor.mean();
    println!("Mean: {}", mean.value());

    let norm = tensor.norm();
    println!("Norm: {}", norm.value());

    // Device placement
    let cpu_tensor = Tensor::zeros_on_device(vec![3, 3], train_station::Device::cpu());
    println!(
        "CPU tensor: shape {:?}, device: {:?}",
        cpu_tensor.shape().dims,
        cpu_tensor.device()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::zeros(vec![2, 3]);
        assert_eq!(tensor.shape().dims, vec![2, 3]);
        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_basic_operations() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();

        let sum = a.add_tensor(&b);
        assert_eq!(sum.data(), &[4.0, 6.0]);

        let scalar_add = a.add_scalar(5.0);
        assert_eq!(scalar_add.data(), &[6.0, 7.0]);
    }

    #[test]
    fn test_shape_operations() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let reshaped = tensor.view(vec![4]);
        assert_eq!(reshaped.shape().dims, vec![4]);
        assert_eq!(reshaped.data(), tensor.data());
    }

    #[test]
    fn test_data_access() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        assert_eq!(tensor.get(&[0, 0]), 1.0);
        assert_eq!(tensor.get(&[0, 1]), 2.0);
        assert_eq!(tensor.get(&[1, 0]), 3.0);
        assert_eq!(tensor.get(&[1, 1]), 4.0);
    }
}
