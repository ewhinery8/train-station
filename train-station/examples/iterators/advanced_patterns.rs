//! # Advanced Iterator Patterns - Complex Data Processing Workflows
//!
//! ## Overview
//!
//! This example demonstrates sophisticated iterator patterns and complex data processing
//! workflows using Train Station's tensor iterator system. It showcases advanced
//! functional programming techniques, data transformation pipelines, and real-world
//! processing scenarios.
//!
//! ## Learning Objectives
//!
//! - Master complex iterator chains and transformations
//! - Learn advanced functional programming patterns
//! - Understand data processing pipeline design
//! - Explore real-world tensor processing scenarios
//!
//! ## Prerequisites
//!
//! - Understanding of basic iterator concepts (see element_iteration.rs)
//! - Familiarity with functional programming patterns
//! - Knowledge of tensor operations and gradient tracking
//! - Experience with data processing workflows
//!
//! ## Key Concepts Demonstrated
//!
//! - **Pipeline Processing**: Multi-stage data transformation workflows
//! - **Conditional Processing**: Dynamic filtering and transformation based on data
//! - **Batch Operations**: Efficient processing of large datasets
//! - **Error Handling**: Robust processing with fallback strategies
//! - **Performance Optimization**: Memory-efficient processing patterns
//!
//! ## Example Code Structure
//!
//! 1. **Data Pipeline Processing**: Multi-stage transformation workflows
//! 2. **Conditional Processing**: Dynamic filtering and transformation
//! 3. **Batch Operations**: Efficient large-scale processing
//! 4. **Real-world Scenarios**: Practical data processing applications
//!
//! ## Expected Output
//!
//! The example will demonstrate complex data processing workflows, showing
//! how to build sophisticated transformation pipelines using iterator patterns
//! while maintaining performance and gradient tracking capabilities.
//!
//! ## Performance Notes
//!
//! - Pipeline processing minimizes memory allocations
//! - Conditional processing avoids unnecessary computations
//! - Batch operations leverage SIMD optimizations
//! - Lazy evaluation patterns improve memory efficiency

use train_station::Tensor;

/// Main example function demonstrating advanced iterator patterns
///
/// This function showcases sophisticated data processing workflows
/// using complex iterator chains and transformation pipelines.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Advanced Iterator Patterns Example");

    demonstrate_data_pipeline()?;
    demonstrate_conditional_processing()?;
    demonstrate_batch_operations()?;
    demonstrate_real_world_scenarios()?;

    println!("Advanced Iterator Patterns Example completed successfully!");
    Ok(())
}

/// Demonstrate multi-stage data processing pipeline
///
/// Shows how to build sophisticated transformation workflows using
/// iterator chains for data preprocessing and feature engineering.
fn demonstrate_data_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Data Processing Pipeline ---");

    // Simulate raw sensor data with noise
    let raw_data: Vec<f32> = (0..20)
        .map(|i| {
            let base = i as f32 * 0.5;
            let noise = (i % 3) as f32 * 0.1;
            base + noise
        })
        .collect();

    let tensor = Tensor::from_slice(&raw_data, vec![20])?;
    println!("Raw sensor data: {:?}", tensor.data());

    // Multi-stage processing pipeline
    println!("\nProcessing pipeline:");
    println!("1. Normalize data (z-score)");
    println!("2. Apply smoothing filter");
    println!("3. Detect outliers");
    println!("4. Apply feature scaling");

    // Stage 1: Normalization
    let mean = tensor.mean().value();
    let std = tensor.std().value();
    let normalized: Tensor = tensor
        .iter()
        .map(|elem| elem.sub_scalar(mean).div_scalar(std))
        .collect();
    println!(
        "  Normalized (mean={:.3}, std={:.3}): {:?}",
        mean,
        std,
        normalized.data()
    );

    // Stage 2: Smoothing (simple moving average)
    let smoothed: Tensor = normalized
        .iter()
        .enumerate()
        .map(|(i, elem)| {
            if i == 0 || i == normalized.size() - 1 {
                elem.clone()
            } else {
                // Simple 3-point average
                let prev = normalized.element_view(i - 1);
                let next = normalized.element_view(i + 1);
                elem.add_tensor(&prev).add_tensor(&next).div_scalar(3.0)
            }
        })
        .collect();
    println!("  Smoothed: {:?}", smoothed.data());

    // Stage 3: Outlier detection and removal
    let outlier_threshold = 2.0;
    let cleaned: Tensor = smoothed
        .iter()
        .filter(|elem| elem.value().abs() < outlier_threshold)
        .collect();
    println!(
        "  Outliers removed (threshold={}): {:?}",
        outlier_threshold,
        cleaned.data()
    );

    // Stage 4: Feature scaling to [0, 1] range
    let min_val = cleaned
        .iter()
        .map(|e| e.value())
        .fold(f32::INFINITY, f32::min);
    let max_val = cleaned
        .iter()
        .map(|e| e.value())
        .fold(f32::NEG_INFINITY, f32::max);
    let scaled: Tensor = cleaned
        .iter()
        .map(|elem| elem.sub_scalar(min_val).div_scalar(max_val - min_val))
        .collect();
    println!("  Scaled to [0,1]: {:?}", scaled.data());

    Ok(())
}

/// Demonstrate conditional processing patterns
///
/// Shows how to implement dynamic filtering and transformation
/// based on data characteristics and conditions.
fn demonstrate_conditional_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Conditional Processing ---");

    // Create data with mixed characteristics
    let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0];
    let tensor = Tensor::from_slice(&data, vec![10])?;
    println!("Input data: {:?}", tensor.data());

    // Conditional transformation based on sign
    println!("\nConditional transformation (positive/negative handling):");
    let processed: Tensor = tensor
        .iter()
        .map(|elem| {
            let val = elem.value();
            if val > 0.0 {
                elem.pow_scalar(2.0) // Square positive values
            } else {
                elem.mul_scalar(-1.0).sqrt() // Square root of absolute negative values
            }
        })
        .collect();
    println!("  Processed: {:?}", processed.data());

    // Adaptive filtering based on local statistics
    println!("\nAdaptive filtering (remove values > 2 std from local mean):");
    let window_size = 3;
    let adaptive_filtered: Tensor = tensor
        .iter()
        .enumerate()
        .filter(|(i, elem)| {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(tensor.size());

            // Calculate local mean and std
            let local_values: Vec<f32> = (start..end)
                .map(|j| tensor.element_view(j).value())
                .collect();

            let local_mean = local_values.iter().sum::<f32>() / local_values.len() as f32;
            let local_variance = local_values
                .iter()
                .map(|v| (v - local_mean).powi(2))
                .sum::<f32>()
                / local_values.len() as f32;
            let local_std = local_variance.sqrt();

            let threshold = local_mean + 2.0 * local_std;
            elem.value() <= threshold
        })
        .map(|(_, elem)| elem)
        .collect();
    println!("  Adaptive filtered: {:?}", adaptive_filtered.data());

    // Multi-condition processing
    println!("\nMulti-condition processing:");
    let multi_processed: Tensor = tensor
        .iter()
        .map(|elem| {
            let val = elem.value();
            match () {
                _ if val > 5.0 => elem.mul_scalar(2.0), // Double large values
                _ if val < -5.0 => elem.div_scalar(2.0), // Halve small values
                _ if val.abs() < 2.0 => elem.add_scalar(1.0), // Add 1 to small values
                _ => elem.clone(),                      // Keep others unchanged
            }
        })
        .collect();
    println!("  Multi-condition: {:?}", multi_processed.data());

    Ok(())
}

/// Demonstrate batch processing operations
///
/// Shows efficient processing of large datasets using iterator
/// patterns and batch operations for performance optimization.
fn demonstrate_batch_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Batch Operations ---");

    // Create a larger dataset for batch processing
    let size = 100;
    let data: Vec<f32> = (0..size)
        .map(|i| {
            let x = i as f32 / size as f32;
            x * x + 0.1 * (i % 7) as f32 // Quadratic with some noise
        })
        .collect();

    let tensor = Tensor::from_slice(&data, vec![size])?;
    println!("Dataset size: {}", tensor.size());

    // Batch processing with windowing
    println!("\nBatch processing with sliding windows:");
    let batch_size = 10;
    let batches: Vec<Tensor> = tensor
        .iter()
        .collect::<Vec<_>>()
        .chunks(batch_size)
        .map(|chunk| {
            // Process each batch independently
            chunk
                .iter()
                .map(|elem| elem.pow_scalar(2.0).add_scalar(1.0))
                .collect()
        })
        .collect();

    println!(
        "  Processed {} batches of size {}",
        batches.len(),
        batch_size
    );
    for (i, batch) in batches.iter().enumerate() {
        println!(
            "    Batch {}: mean={:.3}, std={:.3}",
            i,
            batch.mean().value(),
            batch.std().value()
        );
    }

    // Parallel-like processing with stride
    println!("\nStrided processing (every nth element):");
    let stride = 5;
    let strided: Tensor = tensor
        .iter()
        .enumerate()
        .filter(|(i, _)| i % stride == 0)
        .map(|(_, elem)| elem)
        .collect();
    println!("  Strided (every {}th): {:?}", stride, strided.data());

    // Hierarchical processing
    println!("\nHierarchical processing (coarse to fine):");
    let coarse: Tensor = tensor
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 == 0) // Take every 4th element
        .map(|(_, elem)| elem)
        .collect();

    let fine: Tensor = tensor
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 != 0) // Take the rest
        .map(|(_, elem)| elem)
        .collect();

    println!("  Coarse (every 4th): {:?}", coarse.data());
    println!("  Fine (rest): {:?}", fine.data());

    // Combine coarse and fine with different processing
    let combined: Tensor = coarse
        .iter()
        .map(|elem| elem.mul_scalar(2.0)) // Scale coarse
        .chain(fine.iter().map(|elem| elem.div_scalar(2.0))) // Scale fine
        .collect();
    println!("  Combined: {:?}", combined.data());

    Ok(())
}

/// Demonstrate real-world processing scenarios
///
/// Shows practical applications of iterator patterns for
/// common data processing tasks in machine learning and analytics.
fn demonstrate_real_world_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Real-world Scenarios ---");

    // Scenario 1: Time series analysis
    println!("\nScenario 1: Time Series Analysis");
    let time_series: Vec<f32> = (0..24)
        .map(|hour| {
            let base = 20.0 + 10.0 * (hour as f32 * std::f32::consts::PI / 12.0).sin();
            base + (hour % 3) as f32 * 2.0 // Add some noise
        })
        .collect();

    let series = Tensor::from_slice(&time_series, vec![24])?;
    println!("  Time series (24 hours): {:?}", series.data());

    // Calculate moving average
    let window_size = 3;
    let moving_avg: Tensor = series
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(series.size());
            let window = series.iter_range(start, end);
            window.fold(0.0, |acc, elem| acc + elem.value()) / (end - start) as f32
        })
        .map(|val| Tensor::from_slice(&[val], vec![1]).unwrap())
        .collect();
    println!(
        "  Moving average (window={}): {:?}",
        window_size,
        moving_avg.data()
    );

    // Scenario 2: Feature engineering
    println!("\nScenario 2: Feature Engineering");
    let features = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;
    println!("  Original features: {:?}", features.data());

    // Create polynomial features
    let poly_features: Tensor = features
        .iter()
        .flat_map(|elem| {
            vec![
                elem.clone(),         // x^1
                elem.pow_scalar(2.0), // x^2
                elem.pow_scalar(3.0), // x^3
            ]
        })
        .collect();
    println!(
        "  Polynomial features (x, x^2, x^3): {:?}",
        poly_features.data()
    );

    // Scenario 3: Data augmentation
    println!("\nScenario 3: Data Augmentation");
    let original = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3])?;
    println!("  Original data: {:?}", original.data());

    // Augment with noise and scaling
    let augmented: Tensor = original
        .iter()
        .flat_map(|elem| {
            vec![
                elem.clone(),         // Original
                elem.add_scalar(0.1), // Add noise
                elem.sub_scalar(0.1), // Subtract noise
                elem.mul_scalar(1.1), // Scale up
                elem.mul_scalar(0.9), // Scale down
            ]
        })
        .collect();
    println!("  Augmented data: {:?}", augmented.data());

    // Scenario 4: Statistical analysis
    println!("\nScenario 4: Statistical Analysis");
    let sample_data = Tensor::from_slice(&[1.1, 2.3, 1.8, 2.1, 1.9, 2.0, 1.7, 2.2], vec![8])?;
    println!("  Sample data: {:?}", sample_data.data());

    // Calculate various statistics
    let mean = sample_data.mean().value();
    let std = sample_data.std().value();
    let min = sample_data
        .iter()
        .map(|e| e.value())
        .fold(f32::INFINITY, f32::min);
    let max = sample_data
        .iter()
        .map(|e| e.value())
        .fold(f32::NEG_INFINITY, f32::max);

    // Z-score normalization
    let z_scores: Tensor = sample_data
        .iter()
        .map(|elem| elem.sub_scalar(mean).div_scalar(std))
        .collect();

    println!(
        "  Statistics: mean={:.3}, std={:.3}, min={:.3}, max={:.3}",
        mean, std, min, max
    );
    println!("  Z-scores: {:?}", z_scores.data());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test data pipeline processing
    #[test]
    fn test_data_pipeline() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let normalized: Tensor = tensor.iter().map(|elem| elem.mul_scalar(2.0)).collect();

        assert_eq!(normalized.data(), &[2.0, 4.0, 6.0, 8.0]);
    }

    /// Test conditional processing
    #[test]
    fn test_conditional_processing() {
        let tensor = Tensor::from_slice(&[1.0, -2.0, 3.0], vec![3]).unwrap();
        let processed: Tensor = tensor
            .iter()
            .map(|elem| {
                if elem.value() > 0.0 {
                    elem.mul_scalar(2.0)
                } else {
                    elem.abs()
                }
            })
            .collect();

        assert_eq!(processed.data(), &[2.0, 2.0, 6.0]);
    }

    /// Test batch operations
    #[test]
    fn test_batch_operations() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let strided: Tensor = tensor
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, elem)| elem)
            .collect();

        assert_eq!(strided.data(), &[1.0, 3.0]);
    }
}
