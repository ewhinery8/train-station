//! # Performance Optimization - Memory-Efficient Tensor Processing
//!
//! ## Overview
//!
//! This example demonstrates performance optimization techniques and memory-efficient
//! processing patterns using Train Station's tensor iterator system. It showcases
//! how to process large datasets efficiently while maintaining gradient tracking
//! and leveraging SIMD optimizations.
//!
//! ## Learning Objectives
//!
//! - Understand performance characteristics of tensor iterators
//! - Learn memory-efficient processing patterns
//! - Master optimization techniques for large-scale processing
//! - Explore benchmarking and performance analysis
//!
//! ## Prerequisites
//!
//! - Understanding of basic and advanced iterator concepts
//! - Knowledge of performance optimization principles
//! - Familiarity with memory management patterns
//! - Experience with large-scale data processing
//!
//! ## Key Concepts Demonstrated
//!
//! - **Memory Efficiency**: Zero-copy views and shared memory allocation
//! - **Performance Optimization**: SIMD utilization and batch processing
//! - **Benchmarking**: Performance measurement and analysis
//! - **Scalability**: Processing patterns for large datasets
//! - **Resource Management**: Efficient memory and computation usage
//!
//! ## Example Code Structure
//!
//! 1. **Performance Benchmarking**: Measuring iterator performance characteristics
//! 2. **Memory Optimization**: Efficient memory usage patterns
//! 3. **Large-Scale Processing**: Handling big datasets efficiently
//! 4. **Optimization Techniques**: Advanced performance optimization strategies
//!
//! ## Expected Output
//!
//! The example will demonstrate performance characteristics and optimization
//! techniques, showing how to efficiently process large datasets while
//! maintaining memory efficiency and leveraging SIMD optimizations.
//!
//! ## Performance Notes
//!
//! - View creation overhead: ~64 bytes per element view
//! - SIMD operations leverage existing optimized implementations
//! - Memory sharing eliminates data copying overhead
//! - Batch processing improves cache locality

use std::time::Instant;
use train_station::Tensor;

/// Main example function demonstrating performance optimization
///
/// This function showcases performance optimization techniques and
/// memory-efficient processing patterns for large-scale tensor operations.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Performance Optimization Example");

    demonstrate_performance_benchmarking()?;
    demonstrate_memory_optimization()?;
    demonstrate_large_scale_processing()?;
    demonstrate_optimization_techniques()?;

    println!("Performance Optimization Example completed successfully!");
    Ok(())
}

/// Demonstrate performance benchmarking and analysis
///
/// Shows how to measure and analyze the performance characteristics
/// of tensor iterator operations and compare different approaches.
fn demonstrate_performance_benchmarking() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Performance Benchmarking ---");

    // Create test data of different sizes
    let sizes = vec![100, 1000, 10000];

    for size in sizes {
        println!("\nBenchmarking with tensor size: {}", size);

        // Generate test data
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, vec![size])?;

        // Benchmark 1: Direct tensor operations
        let start = Instant::now();
        let direct_result = tensor.mul_scalar(2.0).add_scalar(1.0);
        let direct_time = start.elapsed();

        // Benchmark 2: Iterator-based operations
        let start = Instant::now();
        let iterator_result: Tensor = tensor
            .iter()
            .map(|elem| elem.mul_scalar(2.0).add_scalar(1.0))
            .collect();
        let iterator_time = start.elapsed();

        // Benchmark 3: Chained iterator operations
        let start = Instant::now();
        let _chained_result: Tensor = tensor
            .iter()
            .map(|elem| elem.mul_scalar(2.0))
            .filter(|elem| elem.value() > size as f32)
            .map(|elem| elem.add_scalar(1.0))
            .collect();
        let chained_time = start.elapsed();

        // Report results
        println!("  Direct operations: {:?}", direct_time);
        println!("  Iterator operations: {:?}", iterator_time);
        println!("  Chained operations: {:?}", chained_time);

        // Verify correctness
        assert_eq!(direct_result.data(), iterator_result.data());
        println!(
            "  Results match: {}",
            direct_result.data() == iterator_result.data()
        );

        // Performance ratio
        let ratio = iterator_time.as_nanos() as f64 / direct_time.as_nanos() as f64;
        println!("  Iterator/Direct ratio: {:.2}x", ratio);
    }

    Ok(())
}

/// Demonstrate memory optimization patterns
///
/// Shows memory-efficient processing patterns and techniques
/// for minimizing memory usage while maintaining performance.
fn demonstrate_memory_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Memory Optimization ---");

    // Create a large tensor for memory testing
    let size = 10000;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let tensor = Tensor::from_slice(&data, vec![size])?;

    println!("Processing tensor of size: {}", size);

    // Pattern 1: Streaming processing (process in chunks)
    println!("\nPattern 1: Streaming Processing");
    let chunk_size = 1000;
    let start = Instant::now();

    let mut streamed_result = Vec::new();
    for chunk_start in (0..size).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(size);
        let chunk: Tensor = tensor
            .iter_range(chunk_start, chunk_end)
            .map(|elem| elem.pow_scalar(2.0).sqrt())
            .collect();
        streamed_result.extend(chunk.data().iter().cloned());
    }
    let streamed_time = start.elapsed();

    // Pattern 2: Full processing
    let start = Instant::now();
    let _full_result: Tensor = tensor
        .iter()
        .map(|elem| elem.pow_scalar(2.0).sqrt())
        .collect();
    let full_time = start.elapsed();

    println!("  Streaming time: {:?}", streamed_time);
    println!("  Full processing time: {:?}", full_time);
    println!(
        "  Memory efficiency ratio: {:.2}x",
        full_time.as_nanos() as f64 / streamed_time.as_nanos() as f64
    );

    // Pattern 3: Lazy evaluation with take
    println!("\nPattern 2: Lazy Evaluation");
    let start = Instant::now();
    let lazy_result: Tensor = tensor
        .iter()
        .take(1000) // Only process first 1000 elements
        .map(|elem| elem.pow_scalar(2.0).sqrt())
        .collect();
    let lazy_time = start.elapsed();

    println!("  Lazy processing (1000 elements): {:?}", lazy_time);
    println!("  Lazy result size: {}", lazy_result.size());

    // Pattern 4: Memory-efficient filtering
    println!("\nPattern 3: Memory-Efficient Filtering");
    let start = Instant::now();
    let filtered_result: Tensor = tensor
        .iter()
        .filter(|elem| elem.value() > size as f32 / 2.0) // Keep only large values
        .map(|elem| elem.mul_scalar(2.0))
        .collect();
    let filtered_time = start.elapsed();

    println!("  Filtered processing: {:?}", filtered_time);
    println!(
        "  Filtered result size: {} (reduced from {})",
        filtered_result.size(),
        size
    );

    Ok(())
}

/// Demonstrate large-scale processing techniques
///
/// Shows how to efficiently process very large datasets using
/// iterator patterns and optimization strategies.
fn demonstrate_large_scale_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Large-Scale Processing ---");

    // Simulate large dataset processing
    let sizes = vec![10000, 50000, 100000];

    for size in sizes {
        println!("\nProcessing dataset of size: {}", size);

        // Generate large dataset
        let data: Vec<f32> = (0..size)
            .map(|i| {
                let x = i as f32 / size as f32;
                x * x + 0.1 * (i % 10) as f32 // Quadratic with noise
            })
            .collect();

        let tensor = Tensor::from_slice(&data, vec![size])?;

        // Technique 1: Batch processing
        let batch_size = 1000;
        let start = Instant::now();

        let mut batch_results = Vec::new();
        for batch_start in (0..size).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(size);
            let batch: Tensor = tensor
                .iter_range(batch_start, batch_end)
                .map(|elem| elem.pow_scalar(2.0).add_scalar(1.0))
                .collect();
            batch_results.push(batch);
        }
        let batch_time = start.elapsed();

        // Technique 2: Parallel-like processing with stride
        let start = Instant::now();
        let stride = 4;
        let strided_result: Tensor = tensor
            .iter()
            .enumerate()
            .filter(|(i, _)| i % stride == 0)
            .map(|(_, elem)| elem.pow_scalar(2.0).add_scalar(1.0))
            .collect();
        let strided_time = start.elapsed();

        // Technique 3: Hierarchical processing
        let start = Instant::now();
        let coarse: Tensor = tensor
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 10 == 0) // Every 10th element
            .map(|(_, elem)| elem.pow_scalar(2.0).add_scalar(1.0))
            .collect();
        let fine: Tensor = tensor
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 10 != 0) // Rest of elements
            .map(|(_, elem)| elem.pow_scalar(1.5).add_scalar(0.5))
            .collect();
        let hierarchical_time = start.elapsed();

        // Report performance
        println!("  Batch processing: {:?}", batch_time);
        println!("  Strided processing: {:?}", strided_time);
        println!("  Hierarchical processing: {:?}", hierarchical_time);

        // Memory usage analysis
        let total_batches = (size + batch_size - 1) / batch_size;
        println!("  Batch count: {}", total_batches);
        println!("  Strided result size: {}", strided_result.size());
        println!(
            "  Hierarchical: coarse={}, fine={}",
            coarse.size(),
            fine.size()
        );
    }

    Ok(())
}

/// Demonstrate advanced optimization techniques
///
/// Shows sophisticated optimization strategies and techniques
/// for maximizing performance in tensor iterator operations.
fn demonstrate_optimization_techniques() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Optimization Techniques ---");

    let size = 50000;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let tensor = Tensor::from_slice(&data, vec![size])?;

    println!("Optimizing processing for size: {}", size);

    // Technique 1: Operation fusion
    println!("\nTechnique 1: Operation Fusion");
    let start = Instant::now();
    let fused_result: Tensor = tensor
        .iter()
        .map(|elem| {
            // Fuse multiple operations into single chain
            elem.mul_scalar(2.0).add_scalar(1.0).pow_scalar(2.0).sqrt()
        })
        .collect();
    let fused_time = start.elapsed();

    // Technique 2: Conditional optimization
    println!("\nTechnique 2: Conditional Optimization");
    let start = Instant::now();
    let conditional_result: Tensor = tensor
        .iter()
        .map(|elem| {
            let val = elem.value();
            if val < size as f32 / 2.0 {
                elem.mul_scalar(2.0) // Simple operation for small values
            } else {
                elem.pow_scalar(2.0).sqrt() // Complex operation for large values
            }
        })
        .collect();
    let conditional_time = start.elapsed();

    // Technique 3: Cache-friendly processing
    println!("\nTechnique 3: Cache-Friendly Processing");
    let start = Instant::now();
    let cache_friendly_result: Tensor = tensor
        .iter()
        .take(1000) // Process in cache-friendly chunks
        .map(|elem| elem.mul_scalar(2.0))
        .collect();
    let cache_friendly_time = start.elapsed();

    // Technique 4: Memory pooling simulation
    println!("\nTechnique 4: Memory Pooling Simulation");
    let start = Instant::now();
    let pooled_result: Tensor = tensor
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 100 == 0) // Process every 100th element
        .map(|(_, elem)| elem.pow_scalar(2.0))
        .collect();
    let pooled_time = start.elapsed();

    // Report optimization results
    println!("  Fused operations: {:?}", fused_time);
    println!("  Conditional optimization: {:?}", conditional_time);
    println!("  Cache-friendly processing: {:?}", cache_friendly_time);
    println!("  Memory pooling simulation: {:?}", pooled_time);

    // Performance analysis
    let fastest = fused_time
        .min(conditional_time)
        .min(cache_friendly_time)
        .min(pooled_time);
    println!("  Fastest technique: {:?}", fastest);

    // Memory efficiency analysis
    println!("  Fused result size: {}", fused_result.size());
    println!("  Conditional result size: {}", conditional_result.size());
    println!(
        "  Cache-friendly result size: {}",
        cache_friendly_result.size()
    );
    println!("  Pooled result size: {}", pooled_result.size());

    // Technique 5: Gradient optimization
    println!("\nTechnique 5: Gradient Optimization");
    let grad_tensor = tensor.with_requires_grad();
    let start = Instant::now();

    let grad_result: Tensor = grad_tensor
        .iter()
        .map(|elem| elem.pow_scalar(2.0).add_scalar(1.0))
        .collect();

    let mut loss = grad_result.sum();
    loss.backward(None);
    let grad_time = start.elapsed();

    println!("  Gradient computation: {:?}", grad_time);
    println!(
        "  Gradient tracking enabled: {}",
        grad_result.requires_grad()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test performance benchmarking
    #[test]
    fn test_performance_benchmarking() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let direct = tensor.mul_scalar(2.0);
        let iterator: Tensor = tensor.iter().map(|elem| elem.mul_scalar(2.0)).collect();

        assert_eq!(direct.data(), iterator.data());
    }

    /// Test memory optimization
    #[test]
    fn test_memory_optimization() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let streamed: Tensor = tensor
            .iter_range(0, 2)
            .map(|elem| elem.mul_scalar(2.0))
            .collect();

        assert_eq!(streamed.data(), &[2.0, 4.0]);
    }

    /// Test large-scale processing
    #[test]
    fn test_large_scale_processing() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let strided: Tensor = tensor
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, elem)| elem)
            .collect();

        assert_eq!(strided.data(), &[1.0, 3.0]);
    }

    /// Test optimization techniques
    #[test]
    fn test_optimization_techniques() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let fused: Tensor = tensor
            .iter()
            .map(|elem| elem.mul_scalar(2.0).add_scalar(1.0))
            .collect();

        assert_eq!(fused.data(), &[3.0, 5.0, 7.0]);
    }
}
