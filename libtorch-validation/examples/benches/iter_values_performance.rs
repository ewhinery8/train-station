//! Value-iteration performance benchmark
//!
//! Compares Train Station element-view-based value pipelines against a Vec baseline
//! using the common PerformanceTester infrastructure but substituting Vec
//! where LibTorch would normally be.

use libtorch_validation::performance::{
    create_test_tensor, generate_test_shapes_with_config, PerformanceConfig, PerformanceResult,
    PerformanceTester,
};
use train_station::gradtrack::with_no_grad;
use train_station::tensor::ValuesCollectExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running value-iteration Performance Benchmark");
    println!("==========================================");

    // Configure test dimensions and batch sizes
    let test_dims = &[32, 64, 128, 256, 512];
    let batch_sizes = &[16, 32, 64];

    println!("Test dimensions: {:?}", test_dims);
    println!("Batch sizes: {:?}", batch_sizes);

    // Create performance tester with custom configuration
    let config = PerformanceConfig {
        iterations: 100,
        warmup_iterations: 10,
        verbose: true,
    };
    let mut tester = PerformanceTester::with_config(config);

    // Generate custom test shapes
    let test_shapes = generate_test_shapes_with_config(test_dims, batch_sizes);
    println!("Generated {} test shapes", test_shapes.len());

    // Helper: generate sequential Vec matching TestPattern::Sequential (1..=n)
    let gen_seq = |n: usize| -> Vec<f32> { (0..n).map(|i| (i + 1) as f32).collect() };

    // Helper function to run benchmarks (grad-enabled path)
    let run_bench = |tester: &mut PerformanceTester,
                     name: &str,
                     shape: &[usize],
                     our_op: &dyn Fn() -> train_station::Tensor,
                     vec_op: &dyn Fn()| {
        for _ in 0..tester.config.warmup_iterations {
            let _ = our_op();
            vec_op();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..tester.config.iterations {
            let _ = our_op();
        }
        let our_dt = t0.elapsed();

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

    // Helper function to run benchmarks under NoGrad (inference path)
    let run_bench_nograd = |tester: &mut PerformanceTester,
                            name: &str,
                            shape: &[usize],
                            our_op: &dyn Fn() -> train_station::Tensor,
                            vec_op: &dyn Fn()| {
        // Warmup
        with_no_grad(|| {
            for _ in 0..tester.config.warmup_iterations {
                let _ = our_op();
                vec_op();
            }
        });
        // Measure our op
        let our_dt = {
            let t0 = std::time::Instant::now();
            with_no_grad(|| {
                for _ in 0..tester.config.iterations {
                    let _ = our_op();
                }
            });
            t0.elapsed()
        };
        // Measure Vec baseline (no guard needed)
        let vec_dt = {
            let t1 = std::time::Instant::now();
            for _ in 0..tester.config.iterations {
                vec_op();
            }
            t1.elapsed()
        };

        let result = PerformanceResult::new(
            name.to_string(),
            shape.to_vec(),
            tester.config.iterations,
            our_dt,
            vec_dt,
        )
        .with_param("baseline", "Vec")
        .with_param("mode", "NoGrad");
        tester.add_result(result);
    };

    for shape in &test_shapes {
        println!("Running benchmark for shape: {:?}", shape);
        let total: usize = shape.iter().product();
        let base = create_test_tensor(
            shape,
            libtorch_validation::performance::TestPattern::Sequential,
        );
        let base_vec = gen_seq(total);

        // 1) Flat value pipeline via element views: map (2x+1) then collect into same shape
        let shape_clone = shape.clone();
        let our1 = || {
            base.iter_elements()
                .map(|e| 2.0 * e.value() + 1.0)
                .collect_shape(shape_clone.clone())
        };
        let vec1 = || {
            let result: Vec<f32> = base_vec.iter().map(|&x| 2.0 * x + 1.0).collect();
            std::hint::black_box(&result);
        };
        run_bench(&mut tester, "iter_values_map_collect", shape, &our1, &vec1);

        // 1b) NoGrad + raw data iterator: data().iter().copied() -> collect_shape
        let shape_clone_ng = shape.clone();
        let our1_ng = || {
            base.data()
                .iter()
                .copied()
                .collect_shape(shape_clone_ng.clone())
        };
        run_bench_nograd(
            &mut tester,
            "data_iter_map_collect_nograd",
            shape,
            &our1_ng,
            &vec1,
        );

        // 2) Reduce via element views: compute sum to scalar vs Vec sum
        let our2 = || {
            let s: f32 = base.iter_elements().map(|e| e.value()).sum();
            train_station::Tensor::from_slice(&[s], vec![1]).unwrap()
        };
        let vec2 = || {
            let result: f32 = base_vec.iter().copied().sum();
            std::hint::black_box(result);
        };
        run_bench(&mut tester, "iter_values_sum_reduce", shape, &our2, &vec2);

        // 2b) NoGrad + raw data sum
        let our2_ng = || {
            let s: f32 = base.data().iter().copied().sum();
            train_station::Tensor::from_slice(&[s], vec![1]).unwrap()
        };
        run_bench_nograd(
            &mut tester,
            "data_iter_sum_reduce_nograd",
            shape,
            &our2_ng,
            &vec2,
        );

        // 3) Multi-dim: iter over dim 0, per-slice values map, then collect
        if shape.len() >= 2 {
            let row_len: usize = shape[1..].iter().product();
            let our3 = || {
                // Collect directly into a single tensor using collect_shape instead of cat
                base.iter_elements()
                    .map(|e| e.value() * 0.75 - 0.2)
                    .collect_shape(shape.to_vec())
            };
            let vec3 = || {
                let mut out = Vec::with_capacity(total);
                for b in 0..shape[0] {
                    let start = b * row_len;
                    let end = start + row_len;
                    for &x in &base_vec[start..end] {
                        out.push(x * 0.75 - 0.2);
                    }
                }
                std::hint::black_box(&out);
            };
            run_bench(&mut tester, "iter_dim0_values_map_cat", shape, &our3, &vec3);

            // 3b) NoGrad + raw data map across full tensor
            let shape_clone3 = shape.clone();
            let our3_ng = || {
                base.data()
                    .iter()
                    .copied()
                    .map(|x| x * 0.75 - 0.2)
                    .collect_shape(shape_clone3.clone())
            };
            run_bench_nograd(
                &mut tester,
                "data_iter_values_map_collect_nograd",
                shape,
                &our3_ng,
                &vec3,
            );
        }
    }

    // Save results to common JSON file
    tester.save_results("iter_values_performance.json")?;

    println!("\nBenchmark completed successfully!");
    println!("Results saved to: iter_values_performance.json");
    println!("Total tests run: {}", tester.results().len());
    tester.print_summary();

    Ok(())
}
