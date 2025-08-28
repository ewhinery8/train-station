//! Core performance testing utilities and benchmarking framework
//!
//! Provides comprehensive performance comparison functionality for benchmarking
//! tensor operations against LibTorch reference implementation.

use crate::ffi::LibTorchTensor;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use train_station::serialization::StructSerializable;
use train_station::Tensor;

/// Comprehensive performance metrics for a single operation
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    /// Operation name (e.g., "add_tensor", "mul_scalar")
    pub operation: String,
    /// Tensor shape tested
    pub shape: Vec<usize>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Our implementation average time per operation (nanoseconds)
    pub our_avg_time_ns: u64,
    /// LibTorch average time per operation (nanoseconds)
    pub libtorch_avg_time_ns: u64,
    /// Performance ratio (our_time / libtorch_time) - lower is better
    pub performance_ratio: f64,
    /// Operations per second - our implementation
    pub our_ops_per_sec: f64,
    /// Operations per second - LibTorch
    pub libtorch_ops_per_sec: f64,
    /// Speedup factor (positive means we're faster, negative means slower)
    pub speedup_factor: f64,
    /// Memory usage estimation (elements processed)
    pub elements_processed: usize,
    /// Additional test parameters (scalar values, dimensions, etc.)
    pub test_params: HashMap<String, String>,
    /// Timestamp when test was performed (seconds since Unix epoch)
    pub timestamp: u64,
}

impl PerformanceResult {
    /// Create a new performance result
    pub fn new(
        operation: String,
        shape: Vec<usize>,
        iterations: usize,
        our_total_time: Duration,
        libtorch_total_time: Duration,
    ) -> Self {
        let our_total_ns = our_total_time.as_nanos() as u64;
        let libtorch_total_ns = libtorch_total_time.as_nanos() as u64;

        let our_avg_ns = our_total_ns / iterations as u64;
        let libtorch_avg_ns = libtorch_total_ns / iterations as u64;

        let performance_ratio = our_avg_ns as f64 / libtorch_avg_ns as f64;

        let our_ops_per_sec = 1_000_000_000.0 / our_avg_ns as f64;
        let libtorch_ops_per_sec = 1_000_000_000.0 / libtorch_avg_ns as f64;

        let speedup_factor = if performance_ratio > 1.0 {
            -(performance_ratio) // Negative means we're slower
        } else {
            1.0 / performance_ratio // Positive means we're faster
        };

        let elements_processed = shape.iter().product();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();

        PerformanceResult {
            operation,
            shape,
            iterations,
            our_avg_time_ns: our_avg_ns,
            libtorch_avg_time_ns: libtorch_avg_ns,
            performance_ratio,
            our_ops_per_sec,
            libtorch_ops_per_sec,
            speedup_factor,
            elements_processed,
            test_params: HashMap::new(),
            timestamp,
        }
    }

    /// Add a test parameter
    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.test_params.insert(key.to_string(), value.to_string());
        self
    }

    /// Check if our implementation is faster
    pub fn is_faster(&self) -> bool {
        self.speedup_factor > 1.0
    }

    /// Get performance status as a string
    pub fn status(&self) -> String {
        if self.is_faster() {
            format!("{:.2}x FASTER", self.speedup_factor)
        } else {
            format!("{:.2}x SLOWER", -self.speedup_factor)
        }
    }

    /// Print detailed performance summary to console
    pub fn print_summary(&self) {
        println!("\n=== Performance Result: {} ===", self.operation);
        println!("Shape: {:?}", self.shape);
        println!("Elements: {}", self.elements_processed);
        println!("Iterations: {}", self.iterations);
        println!();
        println!(
            "Train Station: {:.2} ms avg ({:.0} ops/sec)",
            self.our_avg_time_ns as f64 / 1_000_000.0,
            self.our_ops_per_sec
        );
        println!(
            "LibTorch:      {:.2} ms avg ({:.0} ops/sec)",
            self.libtorch_avg_time_ns as f64 / 1_000_000.0,
            self.libtorch_ops_per_sec
        );
        println!();
        println!("Result: {}", self.status());
        println!("Ratio: {:.3}", self.performance_ratio);

        if !self.test_params.is_empty() {
            println!("\nParameters:");
            for (key, value) in &self.test_params {
                println!("  {}: {}", key, value);
            }
        }
        println!("{}=", "=".repeat(50));
    }
}

impl StructSerializable for PerformanceResult {
    fn to_serializer(&self) -> train_station::serialization::StructSerializer {
        use train_station::serialization::StructSerializer;

        StructSerializer::new()
            .field("operation", &self.operation)
            .field("shape", &self.shape)
            .field("iterations", &self.iterations)
            .field("our_avg_time_ns", &self.our_avg_time_ns)
            .field("libtorch_avg_time_ns", &self.libtorch_avg_time_ns)
            .field("performance_ratio", &self.performance_ratio)
            .field("our_ops_per_sec", &self.our_ops_per_sec)
            .field("libtorch_ops_per_sec", &self.libtorch_ops_per_sec)
            .field("speedup_factor", &self.speedup_factor)
            .field("elements_processed", &self.elements_processed)
            .field("test_params", &self.test_params)
            .field("timestamp", &self.timestamp)
    }

    fn from_deserializer(
        deserializer: &mut train_station::serialization::StructDeserializer,
    ) -> train_station::serialization::SerializationResult<Self> {
        Ok(PerformanceResult {
            operation: deserializer.field("operation")?,
            shape: deserializer.field("shape")?,
            iterations: deserializer.field("iterations")?,
            our_avg_time_ns: deserializer.field("our_avg_time_ns")?,
            libtorch_avg_time_ns: deserializer.field("libtorch_avg_time_ns")?,
            performance_ratio: deserializer.field("performance_ratio")?,
            our_ops_per_sec: deserializer.field("our_ops_per_sec")?,
            libtorch_ops_per_sec: deserializer.field("libtorch_ops_per_sec")?,
            speedup_factor: deserializer.field("speedup_factor")?,
            elements_processed: deserializer.field("elements_processed")?,
            test_params: deserializer.field("test_params")?,
            timestamp: deserializer.field("timestamp")?,
        })
    }
}

// Implement Serializable trait for PerformanceResult
impl train_station::serialization::Serializable for PerformanceResult {
    fn to_json(&self) -> train_station::serialization::SerializationResult<String> {
        self.to_serializer().to_json()
    }

    fn from_json(json: &str) -> train_station::serialization::SerializationResult<Self> {
        let mut deserializer = train_station::serialization::StructDeserializer::from_json(json)?;
        Self::from_deserializer(&mut deserializer)
    }

    fn to_binary(&self) -> train_station::serialization::SerializationResult<Vec<u8>> {
        self.to_serializer().to_binary()
    }

    fn from_binary(data: &[u8]) -> train_station::serialization::SerializationResult<Self> {
        let mut deserializer = train_station::serialization::StructDeserializer::from_binary(data)?;
        Self::from_deserializer(&mut deserializer)
    }
}

/// Configuration for performance testing
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Number of iterations for each test
    pub iterations: usize,
    /// Number of warmup iterations (not counted in results)
    pub warmup_iterations: usize,
    /// Whether to print detailed console output
    pub verbose: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        PerformanceConfig {
            iterations: 1000,
            warmup_iterations: 10,
            verbose: true,
        }
    }
}

/// Core performance testing framework
pub struct PerformanceTester {
    pub config: PerformanceConfig,
    results: Vec<PerformanceResult>,
}

impl PerformanceTester {
    /// Create a new performance tester with default configuration
    pub fn new() -> Self {
        PerformanceTester {
            config: PerformanceConfig::default(),
            results: Vec::new(),
        }
    }

    /// Create a new performance tester with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        PerformanceTester {
            config,
            results: Vec::new(),
        }
    }

    /// Add a result to the tester
    pub fn add_result(&mut self, result: PerformanceResult) {
        self.results.push(result);
    }

    /// Add multiple results to the tester
    pub fn add_results(&mut self, results: Vec<PerformanceResult>) {
        self.results.extend(results);
    }

    /// Benchmark a tensor operation against LibTorch
    pub fn benchmark_operation<F, G>(
        &mut self,
        operation_name: &str,
        shape: &[usize],
        our_op: F,
        libtorch_op: G,
    ) -> PerformanceResult
    where
        F: Fn() -> Tensor,
        G: Fn() -> Result<LibTorchTensor, String>,
    {
        if self.config.verbose {
            println!("Benchmarking {} with shape {:?}...", operation_name, shape);
        }

        // Warmup runs
        for _ in 0..self.config.warmup_iterations {
            let _ = our_op();
            let _ = libtorch_op();
        }

        // Benchmark our implementation
        let our_start = Instant::now();
        for _ in 0..self.config.iterations {
            let _ = our_op();
        }
        let our_total_time = our_start.elapsed();

        // Benchmark LibTorch
        let libtorch_start = Instant::now();
        for _ in 0..self.config.iterations {
            let _ = libtorch_op().expect("LibTorch operation failed");
        }
        let libtorch_total_time = libtorch_start.elapsed();

        // Create result
        let result = PerformanceResult::new(
            operation_name.to_string(),
            shape.to_vec(),
            self.config.iterations,
            our_total_time,
            libtorch_total_time,
        );

        if self.config.verbose {
            result.print_summary();
        }

        // Store result
        self.results.push(result.clone());

        result
    }

    /// Benchmark a tensor operation with custom parameters
    pub fn benchmark_with_params<F, G>(
        &mut self,
        operation_name: &str,
        shape: &[usize],
        params: &[(&str, &str)],
        our_op: F,
        libtorch_op: G,
    ) -> PerformanceResult
    where
        F: Fn() -> Tensor,
        G: Fn() -> Result<LibTorchTensor, String>,
    {
        let mut result = self.benchmark_operation(operation_name, shape, our_op, libtorch_op);

        // Add parameters
        for (key, value) in params {
            result = result.with_param(key, value);
        }

        // Update stored result
        if let Some(last_result) = self.results.last_mut() {
            *last_result = result.clone();
        }

        result
    }

    /// Save all results to a file in the specified format
    ///
    /// This method serializes all performance results to either JSON or binary format
    /// and writes them to the specified file. The format is automatically detected
    /// based on the file extension. The JSON output format is a bare array that matches
    /// what Python visualization scripts expect.
    ///
    /// # Arguments
    ///
    /// * `filename` - File path where results should be saved
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error on failure
    ///
    /// # Examples
    ///
    /// ```ignore
    /// tester.save_results("performance_results.json")?;
    /// tester.save_results("performance_results.bin")?;
    /// ```
    pub fn save_results(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::Write;
        use std::path::Path;
        use train_station::serialization::Format;

        let path = Path::new(filename);
        let format = if let Some(extension) = path.extension() {
            match extension.to_str() {
                Some("json") => Format::Json,
                Some("bin") => Format::Binary,
                _ => Format::Json, // Default to JSON for unknown extensions
            }
        } else {
            Format::Json // Default to JSON if no extension
        };

        match format {
            Format::Json => {
                // Create JSON array format that Python visualization scripts expect
                let mut json_results = Vec::new();
                for result in &self.results {
                    let result_json = result.to_json()?;
                    json_results.push(result_json);
                }

                // Create JSON array manually
                let json_array = format!("[{}]", json_results.join(","));

                let mut file = File::create(filename)?;
                file.write_all(json_array.as_bytes())?;
                file.flush()?;
            }
            Format::Binary => {
                // For binary format, use struct wrapper for proper serialization
                struct ResultsWrapper {
                    results: Vec<PerformanceResult>,
                }

                impl StructSerializable for ResultsWrapper {
                    fn to_serializer(&self) -> train_station::serialization::StructSerializer {
                        train_station::serialization::StructSerializer::new()
                            .field("results", &self.results)
                    }

                    fn from_deserializer(
                        deserializer: &mut train_station::serialization::StructDeserializer,
                    ) -> train_station::serialization::SerializationResult<Self>
                    {
                        Ok(ResultsWrapper {
                            results: deserializer.field("results")?,
                        })
                    }
                }

                let wrapper = ResultsWrapper {
                    results: self.results.clone(),
                };
                wrapper.save_binary(filename)?;
            }
        }
        Ok(())
    }

    /// Get all performance results
    pub fn results(&self) -> &[PerformanceResult] {
        &self.results
    }

    /// Print summary of all results
    pub fn print_summary(&self) {
        if self.results.is_empty() {
            println!("No performance results to display.");
            return;
        }

        println!("\n{}", "=".repeat(70));
        println!(
            "           PERFORMANCE SUMMARY ({} tests)",
            self.results.len()
        );
        println!("{}", "=".repeat(70));

        let mut faster_count = 0;
        let mut total_speedup = 0.0;

        for result in &self.results {
            if result.is_faster() {
                faster_count += 1;
                total_speedup += result.speedup_factor;
            } else {
                total_speedup -= result.speedup_factor;
            }

            println!(
                "{:<20} {:>15} {:>20}",
                result.operation,
                format!("{:?}", result.shape),
                result.status()
            );
        }

        println!("{}", "=".repeat(70));
        println!(
            "Tests where Train Station is faster: {}/{}",
            faster_count,
            self.results.len()
        );

        let avg_performance = total_speedup / self.results.len() as f64;
        if avg_performance > 0.0 {
            println!("Average performance: {:.2}x FASTER", avg_performance);
        } else {
            println!("Average performance: {:.2}x SLOWER", -avg_performance);
        }
        println!("{}", "=".repeat(70));
    }

    /// Clear all stored results
    pub fn clear_results(&mut self) {
        self.results.clear();
    }
}

impl Default for PerformanceTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard test dimensions for comprehensive performance testing
pub const TEST_DIMS: &[usize] = &[32, 64, 128, 256];

/// Standard batch sizes for 3D tensor testing
pub const BATCH_SIZES: &[usize] = &[32, 64, 128];

/// Generate all test shapes for comprehensive benchmarking
pub fn generate_test_shapes() -> Vec<Vec<usize>> {
    generate_test_shapes_with_config(TEST_DIMS, BATCH_SIZES)
}

/// Generate test shapes with custom dimensions and batch sizes
pub fn generate_test_shapes_with_config(
    test_dims: &[usize],
    batch_sizes: &[usize],
) -> Vec<Vec<usize>> {
    let mut shapes = Vec::new();

    // 1D shapes
    for &dim in test_dims {
        shapes.push(vec![dim]);
    }

    // 2D shapes (square matrices)
    for &dim in test_dims {
        shapes.push(vec![dim, dim]);
    }

    // 3D shapes (batch, height, width) - realistic batch sizes
    for &batch in batch_sizes {
        for &dim in test_dims {
            // Reasonable feature sizes for 3D
            shapes.push(vec![batch, dim, dim]);
        }
    }

    shapes
}

/// Utility function to create tensor with specific data pattern for consistent benchmarking
pub fn create_test_tensor(shape: &[usize], pattern: TestPattern) -> Tensor {
    let mut tensor = Tensor::new(shape.to_vec());
    let size = tensor.size();

    unsafe {
        let ptr = tensor.as_mut_ptr();
        match pattern {
            TestPattern::Ones => {
                for i in 0..size {
                    *ptr.add(i) = 1.0;
                }
            }
            TestPattern::Sequential => {
                for i in 0..size {
                    *ptr.add(i) = (i + 1) as f32;
                }
            }
            TestPattern::Random => {
                for i in 0..size {
                    *ptr.add(i) = ((i * 37 + 17) % 100) as f32 / 100.0; // Pseudo-random
                }
            }
        }
    }

    tensor
}

/// Test data patterns for consistent benchmarking
#[derive(Debug, Clone, Copy)]
pub enum TestPattern {
    Ones,
    Sequential,
    Random,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_result_creation() {
        let our_time = Duration::from_millis(100);
        let libtorch_time = Duration::from_millis(150);

        let result = PerformanceResult::new(
            "test_op".to_string(),
            vec![10, 10],
            1000,
            our_time,
            libtorch_time,
        );

        assert_eq!(result.operation, "test_op");
        assert_eq!(result.shape, vec![10, 10]);
        assert_eq!(result.iterations, 1000);
        assert!(result.is_faster()); // We should be faster
        assert!(result.speedup_factor > 1.0);
    }

    #[test]
    fn test_performance_tester() {
        let mut tester = PerformanceTester::new();

        let result = tester.benchmark_operation(
            "test",
            &[2, 2],
            || Tensor::ones(vec![2, 2]),
            || LibTorchTensor::ones(&[2, 2]),
        );

        assert_eq!(result.operation, "test");
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(tester.results().len(), 1);
    }

    #[test]
    fn test_generate_test_shapes() {
        let shapes = generate_test_shapes();

        // Should have 1D, 2D, and 3D shapes
        assert!(shapes.iter().any(|s| s.len() == 1));
        assert!(shapes.iter().any(|s| s.len() == 2));
        assert!(shapes.iter().any(|s| s.len() == 3));

        // Should include all test dimensions
        for &dim in TEST_DIMS {
            assert!(shapes.iter().any(|s| s.contains(&dim)));
        }
    }

    #[test]
    fn test_create_test_tensor() {
        let tensor = create_test_tensor(&[3, 3], TestPattern::Ones);
        assert_eq!(tensor.shape().dims, vec![3, 3]);

        // Check that data pattern is applied
        unsafe {
            for i in 0..tensor.size() {
                assert_eq!(*tensor.as_ptr().add(i), 1.0);
            }
        }
    }
}
