//! Matrix multiplication operation performance benchmark
//!
//! This benchmark tests the performance of tensor matrix multiplication operations
//! against LibTorch, including both 2D and higher-dimensional matrix multiplication.
//!
//! Test Coverage:
//! - Square matrices (n×n * n×n)
//! - Non-square matrices (m×n * n×p)
//! - Batch matrix multiplication (batch×m×n * batch×n×p)
//! - Dot products (1×n * n×1)
//! - Outer products (n×1 * 1×n)
//!
//! Results are saved to a common JSON file for analysis.

use libtorch_validation::performance::{tensor::ops::MatmulPerformanceTester, PerformanceConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Matrix Multiplication Performance Benchmark");
    println!("=================================================");

    // Expanded shape families
    let mut shape_pairs: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();

    // Square families
    for &d in &[16usize, 32, 64, 128, 256, 512, 1024] {
        shape_pairs.push((vec![d, d], vec![d, d]));
    }

    // Tall-skinny: M >> K, moderate N
    for &(m, k, n) in &[
        (512usize, 32usize, 64usize),
        (1024, 64, 64),
        (2048, 64, 128),
    ] {
        shape_pairs.push((vec![m, k], vec![k, n]));
    }

    // Short-wide: small M, large N
    for &(m, k, n) in &[
        (32usize, 128usize, 1024usize),
        (64, 128, 2048),
        (64, 256, 2048),
    ] {
        shape_pairs.push((vec![m, k], vec![k, n]));
    }

    // GEMV: m==1 or n==1
    for &k in &[64usize, 128, 256, 1024] {
        shape_pairs.push((vec![1, k], vec![k, 256])); // row-vector x matrix
        shape_pairs.push((vec![256, k], vec![k, 1])); // matrix x col-vector
    }

    // Batched smalls
    for &(b, m, k, n) in &[
        (8usize, 16usize, 16usize, 16usize),
        (16, 32, 32, 32),
        (32, 32, 64, 32),
    ] {
        shape_pairs.push((vec![b, m, k], vec![b, k, n]));
    }

    // Create performance tester with custom configuration
    let config = PerformanceConfig {
        iterations: 1000,
        warmup_iterations: 50,
        verbose: true,
    };

    let mut tester = MatmulPerformanceTester::with_config(config);

    // Run all matrix multiplication performance tests with shape pairs
    let results = tester.test_all_operations_with_shapes(&shape_pairs);

    // Save results to common JSON file
    tester.tester().save_results("matmul_performance.json")?;

    println!("\nBenchmark completed successfully!");
    println!("Results saved to: matmul_performance.json");
    println!("Total matrix multiplication tests run: {}", results.len());
    println!("Coverage includes square, tall-skinny, short-wide, gemv, and batched smalls");

    Ok(())
}
