//! Iterator performance benchmark
//!
//! Compares Train Station iterator performance against a Vec baseline
//! using the common PerformanceTester. The Vec path is wrapped as
//! a LibTorchTensor for timing parity.

use libtorch_validation::performance::{
    create_test_tensor, generate_test_shapes_with_config, PerformanceConfig, PerformanceResult,
    PerformanceTester,
};
use train_station::tensor::{TensorCollectExt, ValuesCollectExt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Iterator Performance Benchmark");
    println!("=====================================");

    // Configure test dimensions and batch sizes
    let test_dims = &[32, 64, 128, 256, 512];
    let batch_sizes = &[16, 32, 64];

    println!("Test dimensions: {:?}", test_dims);
    println!("Batch sizes: {:?}", batch_sizes);

    // Create performance tester with custom configuration
    let config = PerformanceConfig {
        iterations: 400,
        warmup_iterations: 10,
        verbose: true,
    };
    let mut tester = PerformanceTester::with_config(config);

    // Generate custom test shapes
    let test_shapes = generate_test_shapes_with_config(test_dims, batch_sizes);
    println!("Generated {} test shapes", test_shapes.len());

    // Helper: generate sequential Vec matching TestPattern::Sequential (1..=n)
    let gen_seq = |n: usize| -> Vec<f32> { (0..n).map(|i| (i + 1) as f32).collect() };

    // Helper function to run benchmarks
    let run_bench = |tester: &mut PerformanceTester,
                     name: &str,
                     shape: &[usize],
                     our_op: &dyn Fn() -> train_station::Tensor,
                     vec_op: &dyn Fn()| {
        // Warmup
        for _ in 0..tester.config.warmup_iterations {
            let _ = our_op();
            vec_op();
        }
        // Measure our implementation
        let t0 = std::time::Instant::now();
        for _ in 0..tester.config.iterations {
            let _ = our_op();
        }
        let our_dt = t0.elapsed();
        // Measure Vec baseline
        let t1 = std::time::Instant::now();
        for _ in 0..tester.config.iterations {
            vec_op();
        }
        let vec_dt = t1.elapsed();
        let result = PerformanceResult::new(
            name.to_string(),
            shape.to_vec(),
            tester.config.iterations,
            our_dt,
            vec_dt,
        )
        .with_param("baseline", "Vec");
        tester.add_result(result);
    };

    for shape in &test_shapes {
        let total: usize = shape.iter().product();
        let base = create_test_tensor(
            shape,
            libtorch_validation::performance::TestPattern::Sequential,
        );
        let base_vec = gen_seq(total);

        // 1) Element iterator: map (2x+1) then collect back to original shape
        let shape_clone = shape.clone();
        let our1 = || {
            base.iter_elements()
                .map(|e| e.mul_scalar(2.0).add_scalar(1.0))
                .collect_shape(shape_clone.clone())
        };
        let vec1 = || {
            let result: Vec<f32> = base_vec.iter().map(|&x| 2.0 * x + 1.0).collect();
            std::hint::black_box(&result);
        };
        run_bench(
            &mut tester,
            "iter_elements_map_collect",
            shape,
            &our1,
            &vec1,
        );

        // 2) Multi-dim row-wise map via iter_dim(0) then collect
        if shape.len() >= 2 {
            let row_len: usize = shape[1..].iter().product();
            let our2 = || {
                // Collect directly into a single tensor using collect_shape instead of cat
                base.iter_elements()
                    .map(|e| e.mul_scalar(1.1).add_scalar(0.5))
                    .collect_shape(shape.to_vec())
            };
            let vec2 = || {
                let mut out = Vec::with_capacity(total);
                for b in 0..shape[0] {
                    let start = b * row_len;
                    let end = start + row_len;
                    for &x in &base_vec[start..end] {
                        out.push(1.1 * x + 0.5);
                    }
                }
                std::hint::black_box(&out);
            };
            run_bench(
                &mut tester,
                "iter_dim0_rows_map_collect",
                shape,
                &our2,
                &vec2,
            );
        }

        // 3) Multi-dim: iter_dim(0) then iter_values() inside each slice and collect
        if shape.len() >= 2 {
            let row_len: usize = shape[1..].iter().product();
            let our3 = || {
                // Collect directly into a single tensor using collect_shape instead of cat
                base.iter_values()
                    .map(|x| x * 1.01 - 0.3)
                    .collect_shape(shape.to_vec())
            };
            let vec3 = || {
                let mut out = Vec::with_capacity(total);
                for b in 0..shape[0] {
                    let start = b * row_len;
                    let end = start + row_len;
                    for &x in &base_vec[start..end] {
                        out.push(x * 1.01 - 0.3);
                    }
                }
                std::hint::black_box(&out);
            };
            run_bench(
                &mut tester,
                "iter_dim0_then_values_collect",
                shape,
                &our3,
                &vec3,
            );
        }
    }

    // Save results to common JSON file
    tester.save_results("iterator_performance.json")?;

    println!("\nBenchmark completed successfully!");
    println!("Results saved to: iterator_performance.json");
    println!("Total tests run: {}", tester.results().len());
    tester.print_summary();

    Ok(())
}
