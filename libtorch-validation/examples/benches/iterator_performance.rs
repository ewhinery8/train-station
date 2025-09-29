//! Iterator performance benchmark
//!
//! Compares Train Station iterator performance against a Vec baseline
//! using the common PerformanceTester. The Vec path is wrapped as
//! a LibTorchTensor for timing parity.

use libtorch_validation::performance::{
    create_test_tensor, generate_test_shapes_with_config, PerformanceConfig, PerformanceResult,
    PerformanceTester,
};
use train_station::gradtrack::with_no_grad;
use train_station::tensor::TensorCollectExt;

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

    // Helper function to run benchmarks (grad-enabled path)
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

    // Helper function for NoGrad benchmarks (inference path)
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
        // Our op
        let our_dt = {
            let t0 = std::time::Instant::now();
            with_no_grad(|| {
                for _ in 0..tester.config.iterations {
                    let _ = our_op();
                }
            });
            t0.elapsed()
        };
        // Vec baseline
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
        let total: usize = shape.iter().product();
        let base = create_test_tensor(
            shape,
            libtorch_validation::performance::TestPattern::Sequential,
        );
        let base_vec = gen_seq(total);

        // 1) Multi-dim outer iteration: map rows then collect back to original shape
        if shape.len() >= 2 {
            let row_len: usize = shape[1..].iter().product();
            let our_rows = || {
                base.iter() // outermost dimension
                    .map(|row| row.mul_scalar(1.1).add_scalar(0.5))
                    .collect_shape(shape.to_vec())
            };
            let vec_rows = || {
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
                &our_rows,
                &vec_rows,
            );

            // 1b) Same with gradients disabled (iterator path only)
            let our_rows_ng = || {
                with_no_grad(|| {
                    base.iter()
                        .map(|row| row.mul_scalar(1.1).add_scalar(0.5))
                        .collect_shape(shape.to_vec())
                })
            };
            run_bench_nograd(
                &mut tester,
                "iter_dim0_rows_map_collect_nograd",
                shape,
                &our_rows_ng,
                &vec_rows,
            );
        }

        // 2) Flattened element-wise map via iter_elements then collect back to original shape
        let shape_clone = shape.to_vec();
        let our_flat = || {
            base.iter_elements()
                .map(|e| e.mul_scalar(2.0).add_scalar(1.0))
                .collect_shape(shape_clone.clone())
        };
        let vec_flat = || {
            let result: Vec<f32> = base_vec.iter().map(|&x| 2.0 * x + 1.0).collect();
            std::hint::black_box(&result);
        };
        run_bench(
            &mut tester,
            "iter_elements_map_collect",
            shape,
            &our_flat,
            &vec_flat,
        );

        // 2b) Same with gradients disabled (iterator path only)
        let shape_clone_ng = shape.to_vec();
        let our_flat_ng = || {
            with_no_grad(|| {
                base.iter_elements()
                    .map(|e| e.mul_scalar(2.0).add_scalar(1.0))
                    .collect_shape(shape_clone_ng.clone())
            })
        };
        run_bench_nograd(
            &mut tester,
            "iter_elements_map_collect_nograd",
            shape,
            &our_flat_ng,
            &vec_flat,
        );

        // 3) Chunks over flattened tensor: map chunks then collect back
        let flat_shape = vec![total];
        let base_flat = base.view(vec![total as i32]);
        let chunk_size = 8192usize.min(total.max(1));
        let our_chunks = || {
            base_flat
                .chunks(chunk_size)
                .map(|c| c.mul_scalar(0.75).add_scalar(-0.2))
                .collect_shape(flat_shape.clone())
        };
        let vec_chunks = || {
            let mut out = Vec::with_capacity(total);
            for ch in base_vec.chunks(chunk_size) {
                for &x in ch {
                    out.push(0.75 * x - 0.2);
                }
            }
            std::hint::black_box(&out);
        };
        run_bench(
            &mut tester,
            "iter_chunks_flat_map_collect",
            shape,
            &our_chunks,
            &vec_chunks,
        );

        // 3b) NoGrad + chunks (iterator path only)
        let our_chunks_ng = || {
            with_no_grad(|| {
                base_flat
                    .chunks(chunk_size)
                    .map(|c| c.mul_scalar(0.75).add_scalar(-0.2))
                    .collect_shape(flat_shape.clone())
            })
        };
        run_bench_nograd(
            &mut tester,
            "iter_chunks_flat_map_collect_nograd",
            shape,
            &our_chunks_ng,
            &vec_chunks,
        );

        // 4) Flatten via iter().flat_map(row.iter()) then map and collect
        let our_flatten_rows = || {
            // Avoid returning references to row by collecting each row first
            let rows: Vec<_> = base.iter().collect();
            rows.iter()
                .flat_map(|r| r.iter())
                .map(|e| e.mul_scalar(0.9))
                .collect_shape(vec![total])
        };
        let vec_flatten_rows = || {
            let result: Vec<f32> = base_vec.iter().map(|&x| 0.9 * x).collect();
            std::hint::black_box(&result);
        };
        run_bench(
            &mut tester,
            "iter_dim0_flatmap_iter_map_collect",
            shape,
            &our_flatten_rows,
            &vec_flatten_rows,
        );

        // 4b) Filtered flattened iteration (every 2nd) then collect
        let filtered_len = total.div_ceil(2);
        let our_filter = || {
            base.iter_elements()
                .enumerate()
                .filter(|(i, _)| i % 2 == 0)
                .map(|(_, e)| e.add_scalar(3.0))
                .collect_shape(vec![filtered_len])
        };
        let vec_filter = || {
            let result: Vec<f32> = base_vec
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 0)
                .map(|(_, &x)| x + 3.0)
                .collect();
            std::hint::black_box(&result);
        };
        run_bench(
            &mut tester,
            "iter_elements_filter_take_map_collect",
            shape,
            &our_filter,
            &vec_filter,
        );

        // 4c) NoGrad filtered flattened iteration (iterator only)
        let our_filter_ng = || {
            with_no_grad(|| {
                base.iter_elements()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .map(|(_, e)| e.add_scalar(3.0))
                    .collect_shape(vec![filtered_len])
            })
        };
        run_bench_nograd(
            &mut tester,
            "iter_elements_filter_take_map_collect_nograd",
            shape,
            &our_filter_ng,
            &vec_filter,
        );
    }

    // Save results to common JSON file
    tester.save_results("iterator_performance.json")?;

    println!("\nBenchmark completed successfully!");
    println!("Results saved to: iterator_performance.json");
    println!("Total tests run: {}", tester.results().len());
    tester.print_summary();

    Ok(())
}
