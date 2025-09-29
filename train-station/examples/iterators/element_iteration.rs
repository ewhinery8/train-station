//! # Element Iteration - Basic Tensor Element Processing
//!
//! ## Overview
//!
//! This example demonstrates the fundamental tensor iterator functionality in Train Station,
//! showing how to iterate over tensor elements as individual view tensors. Each element
//! becomes a proper Tensor of shape [1] that supports all existing tensor operations
//! and gradient tracking.
//!
//! ## Learning Objectives
//!
//! - Understand basic tensor element iteration
//! - Learn standard iterator trait methods
//! - Master element-wise transformations
//! - Explore gradient tracking through iterations
//!
//! ## Prerequisites
//!
//! - Basic Rust knowledge and iterator concepts
//! - Understanding of tensor basics (see getting_started/tensor_basics.rs)
//! - Familiarity with functional programming patterns
//!
//! ## Key Concepts Demonstrated
//!
//! - **Element Views**: Each element becomes a true tensor view of shape [1]
//! - **Standard Library Integration**: Full compatibility with Rust's iterator traits
//! - **Gradient Tracking**: Automatic gradient propagation through element operations
//! - **Zero-Copy Semantics**: True views with shared memory allocation
//! - **NoGrad Fast Paths**: Use `with_no_grad` and `data().iter()` to stream values directly for
//!   maximum throughput when gradients are not needed
//!
//! ## Example Code Structure
//!
//! 1. **Basic Iteration**: Simple element access and transformation
//! 2. **Standard Methods**: Using Iterator trait methods (map, filter, collect)
//! 3. **Gradient Tracking**: Demonstrating autograd through element operations
//! 4. **Advanced Patterns**: Complex iterator chains and transformations
//! 5. **NoGrad & Streaming**: Raw data iteration + accelerated collection for inference
//!
//! When to use which iterator path:
//! - Use `iter()` for multi-dim outer iteration (row-major slices). Combine with
//!   `collect_shape([..])` to preserve shape after per-slice transforms.
//! - Use `iter_flat()` for scalar element transforms requiring GradTrack. Combine with
//!   `collect_shape` to reshape the output efficiently instead of manual concatenation.
//! - Use `chunks()` or `iter_fast_chunks()` to process large 1D (or flattened) tensors
//!   in cache-friendly blocks; `collect_shape` to reassemble.
//! - For inference-only value pipelines, use `with_no_grad` + `data().iter().copied()` and
//!   `collect_shape` to stream directly into the destination tensor.
//!
//! ## Expected Output
//!
//! The example will demonstrate various iteration patterns, showing element-wise
//! transformations, gradient tracking, and performance characteristics of the
//! tensor iterator system.
//!
//! ## Performance Notes
//!
//! - View creation is O(1) per element with true zero-copy semantics
//! - Memory overhead is ~64 bytes per view tensor (no data copying)
//! - All operations leverage existing SIMD-optimized tensor implementations
//!
//! ## Next Steps
//!
//! - Explore advanced_patterns.rs for complex iterator chains
//! - Study performance_optimization.rs for large-scale processing
//! - Review tensor operations for element-wise mathematical functions

use train_station::tensor::{TensorCollectExt, ValuesCollectExt};
use train_station::{gradtrack::with_no_grad, Tensor};

/// Main example function demonstrating basic element iteration
///
/// This function serves as the primary educational entry point,
/// with extensive inline comments explaining each step.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Element Iteration Example");

    demonstrate_basic_iteration()?;
    demonstrate_standard_methods()?;
    demonstrate_gradient_tracking()?;
    demonstrate_advanced_patterns()?;
    demonstrate_row_wise_collect_shape()?;
    demonstrate_nograd_and_streaming()?;

    println!("Element Iteration Example completed successfully!");
    Ok(())
}

/// Demonstrate basic tensor element iteration
///
/// Shows how to create iterators over tensor elements and perform
/// simple element-wise operations.
fn demonstrate_basic_iteration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Basic Element Iteration ---");

    // Create a simple tensor for demonstration
    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;
    println!("Original tensor: {:?}", tensor.data());

    // Basic iteration with for loop
    println!("\nBasic iteration with for loop:");
    for (i, element) in tensor.iter().enumerate() {
        println!(
            "  Element {}: value = {:.1}, shape = {:?}",
            i,
            element.value(),
            element.shape().dims()
        );
    }

    // Element-wise transformation
    println!("\nElement-wise transformation (2x + 1):");
    let transformed: Tensor = tensor
        .iter()
        .map(|elem| elem.mul_scalar(2.0).add_scalar(1.0))
        .collect();
    println!("  Result: {:?}", transformed.data());

    // Filtering elements
    println!("\nFiltering elements (values > 3.0):");
    let filtered: Tensor = tensor.iter().filter(|elem| elem.value() > 3.0).collect();
    println!("  Filtered: {:?}", filtered.data());

    Ok(())
}

/// Demonstrate standard iterator trait methods
///
/// Shows compatibility with Rust's standard library iterator methods
/// and demonstrates various functional programming patterns.
fn demonstrate_standard_methods() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Standard Iterator Methods ---");

    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;

    // Using map for transformations
    println!("\nMap transformation (square each element):");
    let squared: Tensor = tensor.iter().map(|elem| elem.pow_scalar(2.0)).collect();
    println!("  Squared: {:?}", squared.data());

    // Using enumerate for indexed operations
    println!("\nEnumerate with indexed operations:");
    let indexed: Tensor = tensor
        .iter()
        .enumerate()
        .map(|(i, elem)| elem.add_scalar(i as f32))
        .collect();
    println!("  Indexed: {:?}", indexed.data());

    // Using fold for reduction
    println!("\nFold for sum calculation:");
    let sum: f32 = tensor.iter().fold(0.0, |acc, elem| acc + elem.value());
    println!("  Sum: {:.1}", sum);

    // Using find for element search
    println!("\nFind specific element:");
    if let Some(found) = tensor.iter().find(|elem| elem.value() == 3.0) {
        println!("  Found element with value 3.0: {:.1}", found.value());
    }

    // Using any/all for condition checking
    println!("\nCondition checking:");
    let all_positive = tensor.iter().all(|elem| elem.value() > 0.0);
    let any_large = tensor.iter().any(|elem| elem.value() > 4.0);
    println!("  All positive: {}", all_positive);
    println!("  Any > 4.0: {}", any_large);

    Ok(())
}

/// Demonstrate gradient tracking through element operations
///
/// Shows how gradient tracking works seamlessly through iterator
/// operations, maintaining the computational graph for backpropagation.
fn demonstrate_gradient_tracking() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Gradient Tracking ---");

    // Create a tensor with gradient tracking enabled
    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3])?.with_requires_grad();
    println!("Input tensor (requires_grad): {:?}", tensor.data());

    // Perform element-wise operations through iteration
    let result: Tensor = tensor
        .iter()
        .map(|elem| {
            // Apply a complex transformation: (x^2 + 1) * 2
            elem.pow_scalar(2.0).add_scalar(1.0).mul_scalar(2.0)
        })
        .collect();

    println!("Result tensor: {:?}", result.data());
    println!("Result requires_grad: {}", result.requires_grad());

    // Compute gradients
    let mut loss = result.sum();
    loss.backward(None);

    println!("Loss: {:.6}", loss.value());
    println!("Input gradients: {:?}", tensor.grad().map(|g| g.data()));

    Ok(())
}

/// Demonstrate advanced iterator patterns
///
/// Shows complex iterator chains and advanced functional programming
/// patterns for sophisticated data processing workflows.
fn demonstrate_advanced_patterns() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Advanced Iterator Patterns ---");

    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6])?;
    println!("Input tensor: {:?}", tensor.data());

    // Complex chain: enumerate -> filter -> map -> collect
    println!("\nComplex chain (even indices only, add index to value):");
    let result: Tensor = tensor
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0) // Take even indices
        .map(|(i, elem)| elem.add_scalar(i as f32)) // Add index to value
        .collect();
    println!("  Result: {:?}", result.data());

    // Using take and skip for windowing
    println!("\nWindowing with take and skip:");
    let window1: Tensor = tensor.iter().take(3).collect();
    let window2: Tensor = tensor.iter().skip(2).take(3).collect();
    println!("  Window 1 (first 3): {:?}", window1.data());
    println!("  Window 2 (middle 3): {:?}", window2.data());

    // Using rev() for reverse iteration
    println!("\nReverse iteration:");
    let reversed: Tensor = tensor.iter().rev().collect();
    println!("  Reversed: {:?}", reversed.data());

    // Chaining with mathematical operations
    println!("\nMathematical operation chain:");
    let math_result: Tensor = tensor
        .iter()
        .map(|elem| elem.exp()) // e^x
        .filter(|elem| elem.value() < 50.0) // Filter large values
        .map(|elem| elem.log()) // ln(x)
        .collect();
    println!("  Math chain result: {:?}", math_result.data());

    // Using zip for element-wise combinations
    println!("\nElement-wise combination with zip:");
    let tensor2 = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![6])?;
    let combined: Tensor = tensor
        .iter()
        .zip(tensor2.iter())
        .map(|(a, b)| a.mul_tensor(&b)) // Element-wise multiplication
        .collect();
    println!("  Combined: {:?}", combined.data());

    Ok(())
}

/// Demonstrate per-row transforms with shape-preserving collection
///
/// Shows how to use `iter()` over the outer dimension on a 2D tensor and
/// `collect_shape([..])` to maintain the original shape after mapping.
fn demonstrate_row_wise_collect_shape() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Row-wise iteration with collect_shape ---");
    let mat = Tensor::from_slice(&(1..=12).map(|x| x as f32).collect::<Vec<_>>(), vec![3, 4])?;
    println!("Input shape: {:?}", mat.shape().dims());

    // Map each row: 1.1*x + 0.5, then collect back to [3,4]
    let out: Tensor = mat
        .iter()
        .map(|row| row.mul_scalar(1.1).add_scalar(0.5))
        .collect_shape(vec![3, 4]);
    println!("  Output shape: {:?}", out.shape().dims());

    Ok(())
}

/// Demonstrate NoGrad fast paths and raw data streaming
///
/// Highlights how to get maximum performance in inference by:
/// - Disabling gradient tracking with `with_no_grad`
/// - Iterating raw values via `tensor.data().iter().copied()`
/// - Using `collect_shape` to stream directly into destination tensors
fn demonstrate_nograd_and_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- NoGrad & Streaming (Inference Fast Paths) ---");

    let input = Tensor::from_slice(
        &(0..24).map(|i| i as f32 * 0.25).collect::<Vec<_>>(),
        vec![4, 6],
    )?;
    println!("Input shape: {:?}", input.shape().dims());

    // NoGrad: stream values directly and reshape
    let out = with_no_grad(|| {
        input
            .data()
            .iter()
            .copied()
            .map(|x| 1.2 * x - 0.3)
            .collect_shape(vec![4, 6])
    });
    println!(
        "  NoGrad streamed map (1.2x-0.3) -> shape {:?}",
        out.shape().dims()
    );

    // Compare to view-based element iteration in NoGrad
    let out_view: Tensor = with_no_grad(|| {
        input
            .iter()
            .map(|e| e.mul_scalar(1.2).add_scalar(-0.3))
            .collect_shape(vec![4, 6])
    });
    println!(
        "  NoGrad view-based map shape {:?}",
        out_view.shape().dims()
    );

    // Quick parity check
    assert_eq!(out.data(), out_view.data());
    println!("  Parity check passed.");

    // Show simple flatten + collect back to a different shape
    let reshaped = with_no_grad(|| input.data().iter().copied().collect_shape(vec![6, 4]));
    println!(
        "  Reshaped via streaming collect_shape: {:?}",
        reshaped.shape().dims()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic iteration functionality
    #[test]
    fn test_basic_iteration() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let elements: Vec<Tensor> = tensor.iter().collect();

        assert_eq!(elements.len(), 3);
        assert_eq!(elements[0].value(), 1.0);
        assert_eq!(elements[1].value(), 2.0);
        assert_eq!(elements[2].value(), 3.0);
    }

    /// Test element-wise transformation
    #[test]
    fn test_element_transformation() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let doubled: Tensor = tensor.iter().map(|elem| elem.mul_scalar(2.0)).collect();

        assert_eq!(doubled.data(), &[2.0, 4.0, 6.0]);
    }

    /// Test gradient tracking
    #[test]
    fn test_gradient_tracking() {
        let tensor = Tensor::from_slice(&[1.0, 2.0], vec![2])
            .unwrap()
            .with_requires_grad();

        let result: Tensor = tensor.iter().map(|elem| elem.mul_scalar(2.0)).collect();

        assert!(result.requires_grad());
        assert_eq!(result.data(), &[2.0, 4.0]);
    }
}
