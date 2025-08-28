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

    // Define comprehensive matrix multiplication test cases
    let shape_pairs = vec![
        // 2D square matrices
        (vec![16, 16], vec![16, 16]),
        (vec![32, 32], vec![32, 32]),
        (vec![64, 64], vec![64, 64]),
        (vec![128, 128], vec![128, 128]),
        (vec![256, 256], vec![256, 256]),
        (vec![512, 512], vec![512, 512]),
        (vec![1024, 1024], vec![1024, 1024]),
        // 2D non-square matrices
        (vec![64, 32], vec![32, 64]),     // 2×3 * 3×4 = 2×4
        (vec![32, 64], vec![64, 32]),     // 4×5 * 5×6 = 4×6
        (vec![128, 64], vec![64, 128]),   // 10×20 * 20×15 = 10×15
        (vec![64, 128], vec![128, 64]),   // 1×100 * 100×1 = 1×1 (dot product)
        (vec![256, 128], vec![128, 256]), // 100×1 * 1×100 = 100×100 (outer product)
        // 3D batch matrices
        (vec![16, 32, 32], vec![16, 32, 32]),
        (vec![32, 64, 64], vec![32, 64, 64]),
        (vec![64, 128, 128], vec![64, 128, 128]),
        (vec![16, 32, 32], vec![16, 32, 32]),
        (vec![32, 64, 64], vec![32, 64, 64]),
        (vec![64, 128, 128], vec![64, 128, 128]),
    ];

    // Create performance tester with custom configuration
    let config = PerformanceConfig {
        iterations: 100,
        warmup_iterations: 10,
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
    println!("Coverage includes square, non-square, and batch matrix multiplications");

    Ok(())
}
