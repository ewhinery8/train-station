//! Permute operation performance benchmark
//!
//! This benchmark tests the performance of tensor permute operations
//! against LibTorch, including 3D permute operations.
//! Results are saved to a common JSON file for analysis.

use libtorch_validation::performance::{
    generate_test_shapes_with_config, tensor::transform::PermutePerformanceTester,
    PerformanceConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Permute Performance Benchmark");
    println!("====================================");

    // Configure test dimensions and batch sizes
    let test_dims = &[16, 32, 64, 128, 256]; // Custom dimensions
    let batch_sizes = &[16, 32]; // Custom batch sizes

    println!("Test dimensions: {:?}", test_dims);
    println!("Batch sizes: {:?}", batch_sizes);

    // Create performance tester with custom configuration
    let config = PerformanceConfig {
        iterations: 1000,
        warmup_iterations: 10,
        verbose: true,
    };

    let mut tester = PermutePerformanceTester::with_config(config);

    // Generate custom test shapes
    let test_shapes = generate_test_shapes_with_config(test_dims, batch_sizes);
    println!("Generated {} test shapes", test_shapes.len());

    // Run all permute performance tests with custom shapes
    let results = tester.test_all_operations_with_shapes(&test_shapes);

    // Save results to common JSON file
    tester.tester().save_results("permute_performance.json")?;

    println!("\nBenchmark completed successfully!");
    println!("Results saved to: permute_performance.json");
    println!("Total tests run: {}", results.len());

    Ok(())
}
