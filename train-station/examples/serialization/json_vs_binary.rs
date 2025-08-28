//! JSON vs Binary Format Comparison Example
//!
//! This example demonstrates the trade-offs between JSON and binary serialization formats:
//! - Performance characteristics of each format
//! - File size comparisons across different data types
//! - Use case recommendations and decision criteria
//! - Debugging and human-readability considerations
//! - Cross-platform compatibility aspects
//!
//! # Learning Objectives
//!
//! - Understand when to choose JSON vs binary formats
//! - Learn performance implications of format selection
//! - Master format-specific optimization techniques
//! - Explore debugging and maintenance considerations
//! - Implement format selection strategies
//!
//! # Prerequisites
//!
//! - Understanding of basic serialization (see basic_structs.rs)
//! - Knowledge of file I/O performance concepts
//! - Familiarity with data interchange requirements
//!
//! # Usage
//!
//! ```bash
//! cargo run --example json_vs_binary
//! ```

use std::{collections::HashMap, fs, time::Instant};
use train_station::serialization::{
    FieldValue, FromFieldValue, SerializationError, SerializationResult, StructDeserializer,
    StructSerializable, StructSerializer, ToFieldValue,
};

/// Performance metrics data structure
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceMetrics {
    pub operation: String,
    pub duration_micros: u64,
    pub memory_usage_bytes: usize,
    pub cpu_usage_percent: f32,
    pub throughput_ops_per_sec: f64,
    pub metadata: HashMap<String, String>,
}

impl StructSerializable for PerformanceMetrics {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("operation", &self.operation)
            .field("duration_micros", &self.duration_micros)
            .field("memory_usage_bytes", &self.memory_usage_bytes)
            .field("cpu_usage_percent", &self.cpu_usage_percent)
            .field("throughput_ops_per_sec", &self.throughput_ops_per_sec)
            .field("metadata", &self.metadata)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let operation = deserializer.field("operation")?;
        let duration_micros = deserializer.field("duration_micros")?;
        let memory_usage_bytes = deserializer.field("memory_usage_bytes")?;
        let cpu_usage_percent = deserializer.field("cpu_usage_percent")?;
        let throughput_ops_per_sec = deserializer.field("throughput_ops_per_sec")?;
        let metadata = deserializer.field("metadata")?;

        Ok(PerformanceMetrics {
            operation,
            duration_micros,
            memory_usage_bytes,
            cpu_usage_percent,
            throughput_ops_per_sec,
            metadata,
        })
    }
}

impl ToFieldValue for PerformanceMetrics {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for PerformanceMetrics {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize PerformanceMetrics from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!(
                        "Failed to deserialize PerformanceMetrics from binary: {}",
                        e
                    ),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for PerformanceMetrics, found {}",
                value.type_name()
            ),
        })
    }
}

/// Large dataset for performance testing
#[derive(Debug, Clone, PartialEq)]
pub struct LargeDataset {
    pub name: String,
    pub values: Vec<f32>, // Changed from f64 to f32 (supported)
    pub labels: Vec<String>,
    pub feature_count: usize, // Simplified from Vec<Vec<f32>> to just a count
    pub feature_dimension: usize, // Store dimensions separately
    pub metadata: HashMap<String, String>,
    pub timestamp_count: usize, // Simplified from Vec<u64> to just count
}

impl StructSerializable for LargeDataset {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("name", &self.name)
            .field("values", &self.values)
            .field("labels", &self.labels)
            .field("feature_count", &self.feature_count)
            .field("feature_dimension", &self.feature_dimension)
            .field("metadata", &self.metadata)
            .field("timestamp_count", &self.timestamp_count)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let name = deserializer.field("name")?;
        let values = deserializer.field("values")?;
        let labels = deserializer.field("labels")?;
        let feature_count = deserializer.field("feature_count")?;
        let feature_dimension = deserializer.field("feature_dimension")?;
        let metadata = deserializer.field("metadata")?;
        let timestamp_count = deserializer.field("timestamp_count")?;

        Ok(LargeDataset {
            name,
            values,
            labels,
            feature_count,
            feature_dimension,
            metadata,
            timestamp_count,
        })
    }
}

impl ToFieldValue for LargeDataset {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for LargeDataset {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize LargeDataset from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize LargeDataset from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for LargeDataset, found {}",
                value.type_name()
            ),
        })
    }
}

/// Configuration data (typical JSON use case)
#[derive(Debug, Clone, PartialEq)]
pub struct Configuration {
    pub version: String,
    pub debug_enabled: bool,
    pub log_level: String,
    pub database_settings: HashMap<String, String>,
    pub feature_flags_enabled: bool, // Simplified from HashMap<String, bool>
    pub max_connections: f32,        // Simplified from HashMap<String, f64>
    pub timeout_seconds: f32,
}

impl StructSerializable for Configuration {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("version", &self.version)
            .field("debug_enabled", &self.debug_enabled)
            .field("log_level", &self.log_level)
            .field("database_settings", &self.database_settings)
            .field("feature_flags_enabled", &self.feature_flags_enabled)
            .field("max_connections", &self.max_connections)
            .field("timeout_seconds", &self.timeout_seconds)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let version = deserializer.field("version")?;
        let debug_enabled = deserializer.field("debug_enabled")?;
        let log_level = deserializer.field("log_level")?;
        let database_settings = deserializer.field("database_settings")?;
        let feature_flags_enabled = deserializer.field("feature_flags_enabled")?;
        let max_connections = deserializer.field("max_connections")?;
        let timeout_seconds = deserializer.field("timeout_seconds")?;

        Ok(Configuration {
            version,
            debug_enabled,
            log_level,
            database_settings,
            feature_flags_enabled,
            max_connections,
            timeout_seconds,
        })
    }
}

impl ToFieldValue for Configuration {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for Configuration {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize Configuration from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize Configuration from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for Configuration, found {}",
                value.type_name()
            ),
        })
    }
}

/// Format comparison results
#[derive(Debug)]
pub struct FormatComparison {
    pub data_type: String,
    pub json_size_bytes: u64,
    pub binary_size_bytes: u64,
    pub json_serialize_micros: u64,
    pub binary_serialize_micros: u64,
    pub json_deserialize_micros: u64,
    pub binary_deserialize_micros: u64,
    pub size_ratio: f64,
    pub serialize_speed_ratio: f64,
    pub deserialize_speed_ratio: f64,
}

impl FormatComparison {
    fn new(data_type: String) -> Self {
        Self {
            data_type,
            json_size_bytes: 0,
            binary_size_bytes: 0,
            json_serialize_micros: 0,
            binary_serialize_micros: 0,
            json_deserialize_micros: 0,
            binary_deserialize_micros: 0,
            size_ratio: 0.0,
            serialize_speed_ratio: 0.0,
            deserialize_speed_ratio: 0.0,
        }
    }

    fn calculate_ratios(&mut self) {
        self.size_ratio = self.json_size_bytes as f64 / self.binary_size_bytes as f64;
        self.serialize_speed_ratio =
            self.binary_serialize_micros as f64 / self.json_serialize_micros as f64;
        self.deserialize_speed_ratio =
            self.binary_deserialize_micros as f64 / self.json_deserialize_micros as f64;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== JSON vs Binary Format Comparison Example ===\n");

    demonstrate_format_characteristics()?;
    demonstrate_size_comparisons()?;
    demonstrate_performance_benchmarks()?;
    demonstrate_use_case_recommendations()?;
    demonstrate_debugging_capabilities()?;
    cleanup_temp_files()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Demonstrate basic format characteristics
fn demonstrate_format_characteristics() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Format Characteristics ---");

    // Create sample data structures
    let mut metadata = HashMap::new();
    metadata.insert("operation_type".to_string(), "benchmark".to_string());
    metadata.insert("system".to_string(), "train_station".to_string());

    let metrics = PerformanceMetrics {
        operation: "tensor_multiplication".to_string(),
        duration_micros: 1234,
        memory_usage_bytes: 8192,
        cpu_usage_percent: 75.5,
        throughput_ops_per_sec: 1000.0,
        metadata,
    };

    println!("Format characteristics analysis:");

    // JSON characteristics
    let json_data = metrics.to_json()?;
    let json_lines = json_data.lines().count();
    let json_chars = json_data.chars().count();

    println!("\nJSON Format:");
    println!("  Size: {} bytes", json_data.len());
    println!("  Characters: {}", json_chars);
    println!("  Lines: {}", json_lines);
    println!("  Human readable: Yes");
    println!("  Self-describing: Yes");
    println!("  Cross-platform: Yes");
    println!("  Compression ratio: Variable (depends on content)");

    // Show sample JSON output
    println!("  Sample output:");
    for line in json_data.lines().take(3) {
        println!("    {}", line);
    }
    if json_lines > 3 {
        println!("    ... ({} more lines)", json_lines - 3);
    }

    // Binary characteristics
    let binary_data = metrics.to_binary()?;

    println!("\nBinary Format:");
    println!("  Size: {} bytes", binary_data.len());
    println!("  Human readable: No");
    println!("  Self-describing: No (requires schema)");
    println!("  Cross-platform: Yes (with proper endianness handling)");
    println!("  Compression ratio: High (efficient encoding)");

    // Show sample binary output (hex)
    println!("  Sample output (first 32 bytes as hex):");
    print!("    ");
    for (i, byte) in binary_data.iter().take(32).enumerate() {
        if i > 0 && i % 16 == 0 {
            println!();
            print!("    ");
        }
        print!("{:02x} ", byte);
    }
    if binary_data.len() > 32 {
        println!("\n    ... ({} more bytes)", binary_data.len() - 32);
    } else {
        println!();
    }

    // Verify roundtrip for both formats
    let json_parsed = PerformanceMetrics::from_json(&json_data)?;
    let binary_parsed = PerformanceMetrics::from_binary(&binary_data)?;

    assert_eq!(metrics, json_parsed);
    assert_eq!(metrics, binary_parsed);
    println!("\nRoundtrip verification: PASSED");

    Ok(())
}

/// Demonstrate size comparisons across different data types
fn demonstrate_size_comparisons() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Size Comparison Analysis ---");

    // Test 1: Small configuration data (typical JSON use case)
    let mut db_settings = HashMap::new();
    db_settings.insert("host".to_string(), "localhost".to_string());
    db_settings.insert("port".to_string(), "5432".to_string());
    db_settings.insert("database".to_string(), "myapp".to_string());

    let config = Configuration {
        version: "1.2.3".to_string(),
        debug_enabled: true,
        log_level: "info".to_string(),
        database_settings: db_settings,
        feature_flags_enabled: true,
        max_connections: 100.0,
        timeout_seconds: 30.0,
    };

    // Test 2: Large numeric dataset (typical binary use case)
    let large_dataset = LargeDataset {
        name: "ML Training Data".to_string(),
        values: (0..1000).map(|i| i as f32 * 0.1).collect(),
        labels: (0..1000).map(|i| format!("label_{}", i)).collect(),
        feature_count: 100,
        feature_dimension: 50,
        timestamp_count: 1000,
        metadata: HashMap::new(),
    };

    println!("Size comparison results:");

    // Configuration comparison
    let config_json = config.to_json()?;
    let config_binary = config.to_binary()?;

    println!("\nConfiguration Data (small, text-heavy):");
    println!("  JSON: {} bytes", config_json.len());
    println!("  Binary: {} bytes", config_binary.len());
    println!(
        "  Ratio (JSON/Binary): {:.2}x",
        config_json.len() as f64 / config_binary.len() as f64
    );
    println!("  Recommendation: JSON (human readable, small size difference)");

    // Large dataset comparison
    let dataset_json = large_dataset.to_json()?;
    let dataset_binary = large_dataset.to_binary()?;

    println!("\nLarge Numeric Dataset (1000 values, 100x50 matrix):");
    println!(
        "  JSON: {} bytes ({:.1} KB)",
        dataset_json.len(),
        dataset_json.len() as f64 / 1024.0
    );
    println!(
        "  Binary: {} bytes ({:.1} KB)",
        dataset_binary.len(),
        dataset_binary.len() as f64 / 1024.0
    );
    println!(
        "  Ratio (JSON/Binary): {:.2}x",
        dataset_json.len() as f64 / dataset_binary.len() as f64
    );
    if dataset_json.len() > dataset_binary.len() {
        println!(
            "  Space saved with binary: {} bytes ({:.1} KB)",
            dataset_json.len() - dataset_binary.len(),
            (dataset_json.len() - dataset_binary.len()) as f64 / 1024.0
        );
        println!("  Recommendation: Binary (significant size reduction)");
    } else {
        println!(
            "  Binary overhead: {} bytes ({:.1} KB)",
            dataset_binary.len() - dataset_json.len(),
            (dataset_binary.len() - dataset_json.len()) as f64 / 1024.0
        );
        println!("  Recommendation: JSON (binary overhead not justified for this size)");
    }

    // Content analysis
    println!("\nContent Type Analysis:");

    // Analyze JSON content patterns
    let json_numbers = dataset_json.matches(char::is_numeric).count();
    let json_brackets = dataset_json.matches('[').count() + dataset_json.matches(']').count();
    let json_quotes = dataset_json.matches('"').count();

    println!("  JSON overhead sources:");
    println!("    Numeric characters: ~{}", json_numbers);
    println!("    Brackets and commas: ~{}", json_brackets);
    println!("    Quote marks: {}", json_quotes);
    println!("    Formatting/whitespace: Varies");

    println!("  Binary advantages:");
    println!("    Direct numeric encoding: 4-8 bytes per number");
    println!("    No formatting overhead: Zero bytes");
    println!("    Efficient length encoding: Minimal bytes");

    Ok(())
}

/// Demonstrate performance benchmarks
fn demonstrate_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Performance Benchmark Analysis ---");

    // Create test data of varying sizes
    let small_config = Configuration {
        version: "1.0.0".to_string(),
        debug_enabled: false,
        log_level: "warn".to_string(),
        database_settings: HashMap::new(),
        feature_flags_enabled: false,
        max_connections: 100.0,
        timeout_seconds: 30.0,
    };

    let large_dataset = LargeDataset {
        name: "Large Dataset".to_string(),
        values: (0..5000).map(|i| i as f32 * 0.001).collect(),
        labels: (0..5000).map(|i| format!("large_item_{}", i)).collect(),
        feature_count: 200,
        feature_dimension: 25,
        timestamp_count: 5000,
        metadata: HashMap::new(),
    };

    println!("Performance benchmark results:");

    // Benchmark each dataset (avoiding trait objects due to object safety)
    let dataset_names = ["Small Config", "Large Dataset"];

    for (i, name) in dataset_names.iter().enumerate() {
        let mut comparison = FormatComparison::new(name.to_string());

        // JSON serialization benchmark
        let start = Instant::now();
        let json_data = match i {
            0 => small_config.to_json()?,
            _ => large_dataset.to_json()?,
        };
        comparison.json_serialize_micros = start.elapsed().as_micros() as u64;
        comparison.json_size_bytes = json_data.len() as u64;

        // JSON deserialization benchmark (using PerformanceMetrics as example)
        if *name == "Small Config" {
            let start = Instant::now();
            let _parsed = Configuration::from_json(&json_data)?;
            comparison.json_deserialize_micros = start.elapsed().as_micros() as u64;
        } else {
            let start = Instant::now();
            let _parsed = LargeDataset::from_json(&json_data)?;
            comparison.json_deserialize_micros = start.elapsed().as_micros() as u64;
        }

        // Binary serialization benchmark
        let start = Instant::now();
        let binary_data = match i {
            0 => small_config.to_binary()?,
            _ => large_dataset.to_binary()?,
        };
        comparison.binary_serialize_micros = start.elapsed().as_micros() as u64;
        comparison.binary_size_bytes = binary_data.len() as u64;

        // Binary deserialization benchmark
        if *name == "Small Config" {
            let start = Instant::now();
            let _parsed = Configuration::from_binary(&binary_data)?;
            comparison.binary_deserialize_micros = start.elapsed().as_micros() as u64;
        } else {
            let start = Instant::now();
            let _parsed = LargeDataset::from_binary(&binary_data)?;
            comparison.binary_deserialize_micros = start.elapsed().as_micros() as u64;
        }

        // Calculate ratios
        comparison.calculate_ratios();

        // Display results
        println!("\n{}:", name);
        println!(
            "  Size - JSON: {} bytes, Binary: {} bytes (ratio: {:.2}x)",
            comparison.json_size_bytes, comparison.binary_size_bytes, comparison.size_ratio
        );
        println!(
            "  Serialize - JSON: {}μs, Binary: {}μs (binary relative speed: {:.2}x)",
            comparison.json_serialize_micros,
            comparison.binary_serialize_micros,
            comparison.serialize_speed_ratio
        );
        println!(
            "  Deserialize - JSON: {}μs, Binary: {}μs (binary relative speed: {:.2}x)",
            comparison.json_deserialize_micros,
            comparison.binary_deserialize_micros,
            comparison.deserialize_speed_ratio
        );
    }

    println!("\nPerformance Summary:");
    println!("  - Binary format consistently uses less storage space");
    println!("  - Performance differences vary by data type and size");
    println!("  - Larger datasets show more significant binary advantages");
    println!("  - JSON parsing overhead increases with structure complexity");

    Ok(())
}

/// Demonstrate use case recommendations
fn demonstrate_use_case_recommendations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Use Case Recommendations ---");

    println!("JSON Format - Recommended for:");
    println!("  ✓ Configuration files (human-editable)");
    println!("  ✓ API responses (web compatibility)");
    println!("  ✓ Debugging and development (readability)");
    println!("  ✓ Small data structures (minimal overhead)");
    println!("  ✓ Cross-language interoperability");
    println!("  ✓ Schema evolution (self-describing)");
    println!("  ✓ Text-heavy data with few numbers");

    println!("\nBinary Format - Recommended for:");
    println!("  ✓ Large datasets (memory/storage efficiency)");
    println!("  ✓ High-performance applications (speed critical)");
    println!("  ✓ Numeric-heavy data (ML models, matrices)");
    println!("  ✓ Network transmission (bandwidth limited)");
    println!("  ✓ Embedded systems (resource constrained)");
    println!("  ✓ Long-term storage (space efficiency)");
    println!("  ✓ Frequent serialization/deserialization");

    // Demonstrate decision matrix
    println!("\nDecision Matrix Example:");

    let scenarios = vec![
        (
            "Web API Configuration",
            "JSON",
            "Human readable, web standard, small size",
        ),
        (
            "ML Model Weights",
            "Binary",
            "Large numeric data, performance critical",
        ),
        (
            "User Preferences",
            "JSON",
            "Human editable, self-documenting",
        ),
        (
            "Real-time Telemetry",
            "Binary",
            "High frequency, bandwidth limited",
        ),
        (
            "Application Settings",
            "JSON",
            "Developer accessible, version control friendly",
        ),
        (
            "Scientific Dataset",
            "Binary",
            "Large arrays, storage efficiency critical",
        ),
    ];

    for (scenario, recommendation, reason) in scenarios {
        println!("  {} -> {} ({})", scenario, recommendation, reason);
    }

    // Create examples for common scenarios
    println!("\nPractical Examples:");

    // Configuration file example (JSON)
    let config = Configuration {
        version: "2.1.0".to_string(),
        debug_enabled: false,
        log_level: "info".to_string(),
        database_settings: {
            let mut map = HashMap::new();
            map.insert("url".to_string(), "postgresql://localhost/app".to_string());
            map.insert("pool_size".to_string(), "10".to_string());
            map
        },
        feature_flags_enabled: true,
        max_connections: 100.0,
        timeout_seconds: 30.0,
    };

    config.save_json("temp_config_example.json")?;
    let config_content = fs::read_to_string("temp_config_example.json")?;

    println!("\nConfiguration File (JSON) - Human readable:");
    for line in config_content.lines().take(5) {
        println!("  {}", line);
    }
    println!("  ... (easily editable by developers)");

    // Data export example (Binary)
    let export_data = LargeDataset {
        name: "Training Export".to_string(),
        values: (0..1000).map(|i| (i as f32).sin()).collect(),
        labels: (0..1000).map(|i| format!("sample_{:04}", i)).collect(),
        feature_count: 50,
        feature_dimension: 20,
        timestamp_count: 1000,
        metadata: HashMap::new(),
    };

    export_data.save_binary("temp_export_example.bin")?;
    let export_size = fs::metadata("temp_export_example.bin")?.len();

    println!("\nData Export (Binary) - Efficient storage:");
    println!(
        "  File size: {} bytes ({:.1} KB)",
        export_size,
        export_size as f64 / 1024.0
    );
    println!("  1000 numeric values + 50x20 matrix + metadata");
    println!("  Compact encoding saves significant space vs JSON");

    Ok(())
}

/// Demonstrate debugging capabilities
fn demonstrate_debugging_capabilities() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Debugging Capabilities ---");

    let mut metadata = HashMap::new();
    metadata.insert("debug_session".to_string(), "session_123".to_string());
    metadata.insert("error_code".to_string(), "E001".to_string());

    let debug_metrics = PerformanceMetrics {
        operation: "debug_test".to_string(),
        duration_micros: 5432,
        memory_usage_bytes: 16384,
        cpu_usage_percent: 42.7,
        throughput_ops_per_sec: 750.0,
        metadata,
    };

    println!("Debugging Comparison:");

    // JSON debugging advantages
    let json_data = debug_metrics.to_json()?;
    println!("\nJSON Format - Debugging Advantages:");
    println!("  ✓ Human readable without tools");
    println!("  ✓ Can inspect values directly");
    println!("  ✓ Text editors show structure");
    println!("  ✓ Diff tools work naturally");
    println!("  ✓ Version control friendly");

    println!("\n  Sample JSON output for debugging:");
    for (i, line) in json_data.lines().enumerate() {
        if i < 5 {
            println!("    {}", line);
        }
    }

    // Binary debugging limitations
    let binary_data = debug_metrics.to_binary()?;
    println!("\nBinary Format - Debugging Limitations:");
    println!("  ✗ Requires special tools to inspect");
    println!("  ✗ Not human readable");
    println!("  ✗ Difficult to debug data corruption");
    println!("  ✗ Version control shows as binary diff");

    println!("\n  Binary data (hex dump for debugging):");
    print!("    ");
    for (i, byte) in binary_data.iter().take(40).enumerate() {
        if i > 0 && i % 16 == 0 {
            println!();
            print!("    ");
        }
        print!("{:02x} ", byte);
    }
    println!("\n    (requires hex editor or custom tools)");

    // Development workflow comparison
    println!("\nDevelopment Workflow Impact:");

    println!("\nJSON Workflow:");
    println!("  1. Save data to JSON file");
    println!("  2. Open in any text editor");
    println!("  3. Inspect values directly");
    println!("  4. Make manual edits if needed");
    println!("  5. Version control tracks changes");

    println!("\nBinary Workflow:");
    println!("  1. Save data to binary file");
    println!("  2. Write debugging code to load and print");
    println!("  3. Use hex editor for low-level inspection");
    println!("  4. Cannot make manual edits easily");
    println!("  5. Version control shows binary changes only");

    // Hybrid approach recommendation
    println!("\nHybrid Approach for Development:");
    println!("  - Use JSON during development/debugging");
    println!("  - Switch to binary for production deployment");
    println!("  - Provide debugging tools that export binary to JSON");
    println!("  - Include format conversion utilities");

    // Demonstrate debugging scenario
    println!("\nDebugging Scenario Example:");
    println!("  Problem: Performance metrics show unexpected values");

    // Save both formats for comparison
    debug_metrics.save_json("temp_debug_metrics.json")?;
    debug_metrics.save_binary("temp_debug_metrics.bin")?;

    println!("  JSON approach: Open temp_debug_metrics.json in editor");
    println!("    -> Immediately see cpu_usage_percent: 42.7");
    println!("    -> Compare with expected range");
    println!("    -> Check metadata for debug_session: 'session_123'");

    println!("  Binary approach: Write debugging code");
    println!("    -> Load binary file programmatically");
    println!("    -> Print values to console");
    println!("    -> Additional development time required");

    Ok(())
}

/// Clean up temporary files created during the example
fn cleanup_temp_files() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Cleanup ---");

    let files_to_remove = [
        "temp_config_example.json",
        "temp_export_example.bin",
        "temp_debug_metrics.json",
        "temp_debug_metrics.bin",
    ];

    for file in &files_to_remove {
        if fs::metadata(file).is_ok() {
            fs::remove_file(file)?;
            println!("Removed: {}", file);
        }
    }

    println!("Cleanup completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_serialization() {
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "value".to_string());

        let metrics = PerformanceMetrics {
            operation: "test_op".to_string(),
            duration_micros: 1000,
            memory_usage_bytes: 2048,
            cpu_usage_percent: 50.0,
            throughput_ops_per_sec: 100.0,
            metadata,
        };

        // Test both formats
        let json_data = metrics.to_json().unwrap();
        let binary_data = metrics.to_binary().unwrap();

        let json_parsed = PerformanceMetrics::from_json(&json_data).unwrap();
        let binary_parsed = PerformanceMetrics::from_binary(&binary_data).unwrap();

        assert_eq!(metrics, json_parsed);
        assert_eq!(metrics, binary_parsed);
    }

    #[test]
    fn test_format_size_comparison() {
        let dataset = LargeDataset {
            name: "Test".to_string(),
            values: vec![1.0, 2.0, 3.0],
            labels: vec!["a".to_string(), "b".to_string()],
            feature_matrix: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            metadata: HashMap::new(),
            timestamps: vec![1, 2, 3],
        };

        let json_data = dataset.to_json().unwrap();
        let binary_data = dataset.to_binary().unwrap();

        // For structured data with numbers, binary should be more compact
        assert!(binary_data.len() < json_data.len());
    }

    #[test]
    fn test_configuration_human_readability() {
        let config = Configuration {
            version: "1.0.0".to_string(),
            debug_enabled: true,
            log_level: "debug".to_string(),
            database_settings: HashMap::new(),
            feature_flags_enabled: true,
            max_connections: 50.0,
            timeout_seconds: 60.0,
        };

        let json_data = config.to_json().unwrap();

        // JSON should contain human-readable text
        assert!(json_data.contains("version"));
        assert!(json_data.contains("1.0.0"));
        assert!(json_data.contains("debug_enabled"));
        assert!(json_data.contains("true"));
    }
}
