//! Error Handling and Validation Example
//!
//! This example demonstrates robust error handling patterns for serialization:
//! - Common serialization error scenarios and recovery
//! - Data validation and schema evolution
//! - Graceful degradation strategies
//! - Error reporting and debugging techniques
//! - Production-ready error handling patterns
//!
//! # Learning Objectives
//!
//! - Understand common serialization failure modes
//! - Learn error recovery and fallback strategies
//! - Master data validation techniques
//! - Explore schema evolution patterns
//! - Implement production-ready error handling
//!
//! # Prerequisites
//!
//! - Understanding of basic serialization (see basic_structs.rs)
//! - Knowledge of Rust error handling patterns
//! - Familiarity with data validation concepts
//!
//! # Usage
//!
//! ```bash
//! cargo run --example error_handling
//! ```

use std::{collections::HashMap, fs, io::Write};
use train_station::serialization::{
    FieldValue, FromFieldValue, SerializationError, SerializationResult, StructDeserializer,
    StructSerializable, StructSerializer, ToFieldValue,
};

/// Versioned data structure for schema evolution testing
#[derive(Debug, Clone, PartialEq)]
pub struct VersionedData {
    pub version: u32,
    pub name: String,
    pub value: f64,
    // New fields for schema evolution
    pub optional_field: Option<String>,
    pub new_field: Option<i32>,
}

impl StructSerializable for VersionedData {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("version", &self.version)
            .field("name", &self.name)
            .field("value", &self.value)
            .field("optional_field", &self.optional_field)
            .field("new_field", &self.new_field)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let version = deserializer.field("version")?;
        let name = deserializer.field("name")?;
        let value = deserializer.field("value")?;

        // Handle optional fields gracefully for schema evolution
        let optional_field = deserializer.field_optional("optional_field")?;
        let new_field = deserializer.field_optional("new_field")?;

        // Validate version compatibility
        if version > 3 {
            return Err(SerializationError::ValidationFailed {
                field: "version".to_string(),
                message: format!("Unsupported version: {}. Maximum supported: 3", version),
            });
        }

        Ok(VersionedData {
            version,
            name,
            value,
            optional_field,
            new_field,
        })
    }
}

impl ToFieldValue for VersionedData {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for VersionedData {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize VersionedData from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize VersionedData from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for VersionedData, found {}",
                value.type_name()
            ),
        })
    }
}

/// Validated user input with constraints
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedUserInput {
    pub username: String,
    pub email: String,
    pub age: u16,
    pub preferences: HashMap<String, String>,
}

impl StructSerializable for ValidatedUserInput {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("username", &self.username)
            .field("email", &self.email)
            .field("age", &self.age)
            .field("preferences", &self.preferences)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let username: String = deserializer.field("username")?;
        let email: String = deserializer.field("email")?;
        let age: u16 = deserializer.field("age")?;
        let preferences: HashMap<String, String> = deserializer.field("preferences")?;

        // Validate username
        if username.is_empty() || username.len() > 50 {
            return Err(SerializationError::ValidationFailed {
                field: "username".to_string(),
                message: "Username must be 1-50 characters long".to_string(),
            });
        }

        if !username
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return Err(SerializationError::ValidationFailed {
                field: "username".to_string(),
                message:
                    "Username can only contain alphanumeric characters, underscores, and hyphens"
                        .to_string(),
            });
        }

        // Validate email (basic check)
        if !email.contains('@') || !email.contains('.') || email.len() < 5 {
            return Err(SerializationError::ValidationFailed {
                field: "email".to_string(),
                message: "Invalid email format".to_string(),
            });
        }

        // Validate age
        if !(13..=120).contains(&age) {
            return Err(SerializationError::ValidationFailed {
                field: "age".to_string(),
                message: "Age must be between 13 and 120".to_string(),
            });
        }

        // Validate preferences
        if preferences.len() > 20 {
            return Err(SerializationError::ValidationFailed {
                field: "preferences".to_string(),
                message: "Too many preferences (maximum 20)".to_string(),
            });
        }

        for (key, value) in &preferences {
            if key.len() > 50 || value.len() > 200 {
                return Err(SerializationError::ValidationFailed {
                    field: "preferences".to_string(),
                    message: format!("Preference key/value too long: {}", key),
                });
            }
        }

        Ok(ValidatedUserInput {
            username,
            email,
            age,
            preferences,
        })
    }
}

impl ToFieldValue for ValidatedUserInput {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for ValidatedUserInput {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize ValidatedUserInput from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!(
                        "Failed to deserialize ValidatedUserInput from binary: {}",
                        e
                    ),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for ValidatedUserInput, found {}",
                value.type_name()
            ),
        })
    }
}

/// Recovery helper for handling partial data
#[derive(Debug, Clone, PartialEq)]
pub struct RecoverableData {
    pub critical_field: String,
    pub important_field: Option<String>,
    pub optional_field: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl StructSerializable for RecoverableData {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("critical_field", &self.critical_field)
            .field("important_field", &self.important_field)
            .field("optional_field", &self.optional_field)
            .field("metadata", &self.metadata)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        // Critical field - must exist
        let critical_field = deserializer.field("critical_field")?;

        // Important field - try to recover if missing
        let important_field = deserializer.field_optional("important_field")?;

        // Optional field - graceful fallback
        let optional_field = deserializer.field_optional("optional_field")?;

        // Metadata - recover what we can
        let metadata = deserializer.field_or("metadata", HashMap::new())?;

        Ok(RecoverableData {
            critical_field,
            important_field,
            optional_field,
            metadata,
        })
    }
}

impl ToFieldValue for RecoverableData {
    fn to_field_value(&self) -> FieldValue {
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for RecoverableData {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize RecoverableData from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize RecoverableData from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for RecoverableData, found {}",
                value.type_name()
            ),
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Error Handling and Validation Example ===\n");

    demonstrate_common_error_scenarios()?;
    demonstrate_validation_patterns()?;
    demonstrate_schema_evolution()?;
    demonstrate_recovery_strategies()?;
    demonstrate_production_error_handling()?;
    cleanup_temp_files()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Demonstrate common serialization error scenarios
fn demonstrate_common_error_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Common Error Scenarios ---");

    fs::create_dir_all("temp_error_tests")?;

    // Scenario 1: Corrupted JSON file
    println!("1. Corrupted JSON File:");
    let corrupted_json = r#"{"name": "test", "value": 42, "incomplete"#;
    fs::write("temp_error_tests/corrupted.json", corrupted_json)?;

    match VersionedData::load_json("temp_error_tests/corrupted.json") {
        Ok(_) => println!("   Unexpected: Corrupted JSON was parsed successfully"),
        Err(e) => println!("   Expected error: {}", e),
    }

    // Scenario 2: Missing required fields
    println!("\n2. Missing Required Fields:");
    let incomplete_json = r#"{"name": "test"}"#;
    fs::write("temp_error_tests/incomplete.json", incomplete_json)?;

    match VersionedData::load_json("temp_error_tests/incomplete.json") {
        Ok(_) => println!("   Unexpected: Incomplete JSON was parsed successfully"),
        Err(e) => println!("   Expected error: {}", e),
    }

    // Scenario 3: Type mismatches
    println!("\n3. Type Mismatch:");
    let type_mismatch_json = r#"{"version": "not_a_number", "name": "test", "value": 42.0}"#;
    fs::write("temp_error_tests/type_mismatch.json", type_mismatch_json)?;

    match VersionedData::load_json("temp_error_tests/type_mismatch.json") {
        Ok(_) => println!("   Unexpected: Type mismatch was handled gracefully"),
        Err(e) => println!("   Expected error: {}", e),
    }

    // Scenario 4: File not found
    println!("\n4. File Not Found:");
    match VersionedData::load_json("temp_error_tests/nonexistent.json") {
        Ok(_) => println!("   Unexpected: Non-existent file was loaded"),
        Err(e) => println!("   Expected error: {}", e),
    }

    // Scenario 5: Binary format mismatch
    println!("\n5. Binary Format Mismatch:");
    let invalid_binary = vec![0xFF, 0xFF, 0xFF, 0xFF]; // Invalid binary data
    fs::write("temp_error_tests/invalid.bin", invalid_binary)?;

    match VersionedData::load_binary("temp_error_tests/invalid.bin") {
        Ok(_) => println!("   Unexpected: Invalid binary was parsed successfully"),
        Err(e) => println!("   Expected error: {}", e),
    }

    // Scenario 6: Wrong format loading
    println!("\n6. Wrong Format Loading:");
    let valid_data = VersionedData {
        version: 1,
        name: "test".to_string(),
        value: 42.0,
        optional_field: None,
        new_field: None,
    };
    valid_data.save_binary("temp_error_tests/valid.bin")?;

    // Try to load binary file as JSON
    match VersionedData::load_json("temp_error_tests/valid.bin") {
        Ok(_) => println!("   Unexpected: Binary file was loaded as JSON"),
        Err(e) => println!("   Expected error: {}", e),
    }

    Ok(())
}

/// Demonstrate validation patterns
fn demonstrate_validation_patterns() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Validation Patterns ---");

    println!("Testing input validation with various scenarios:");

    // Valid input
    println!("\n1. Valid Input:");
    let mut valid_preferences = HashMap::new();
    valid_preferences.insert("theme".to_string(), "dark".to_string());
    valid_preferences.insert("language".to_string(), "en".to_string());

    let valid_input = ValidatedUserInput {
        username: "john_doe".to_string(),
        email: "john@example.com".to_string(),
        age: 25,
        preferences: valid_preferences,
    };

    match valid_input.to_json() {
        Ok(json) => {
            println!("   ✓ Valid input serialized successfully");
            match ValidatedUserInput::from_json(&json) {
                Ok(_) => println!("   ✓ Valid input deserialized successfully"),
                Err(e) => println!("   ✗ Deserialization failed: {}", e),
            }
        }
        Err(e) => println!("   ✗ Serialization failed: {}", e),
    }

    // Test validation errors
    let validation_tests = vec![
        (
            "Empty username",
            ValidatedUserInput {
                username: "".to_string(),
                email: "test@example.com".to_string(),
                age: 25,
                preferences: HashMap::new(),
            },
        ),
        (
            "Invalid username characters",
            ValidatedUserInput {
                username: "user@name!".to_string(),
                email: "test@example.com".to_string(),
                age: 25,
                preferences: HashMap::new(),
            },
        ),
        (
            "Invalid email",
            ValidatedUserInput {
                username: "username".to_string(),
                email: "invalid_email".to_string(),
                age: 25,
                preferences: HashMap::new(),
            },
        ),
        (
            "Age too low",
            ValidatedUserInput {
                username: "username".to_string(),
                email: "test@example.com".to_string(),
                age: 10,
                preferences: HashMap::new(),
            },
        ),
        (
            "Age too high",
            ValidatedUserInput {
                username: "username".to_string(),
                email: "test@example.com".to_string(),
                age: 150,
                preferences: HashMap::new(),
            },
        ),
    ];

    for (description, invalid_input) in validation_tests {
        println!("\n2. {}:", description);
        match invalid_input.to_json() {
            Ok(json) => match ValidatedUserInput::from_json(&json) {
                Ok(_) => println!("   ✗ Unexpected: Invalid input was accepted"),
                Err(e) => println!("   ✓ Expected validation error: {}", e),
            },
            Err(e) => println!("   ✗ Serialization error: {}", e),
        }
    }

    // Test preferences validation
    println!("\n3. Preferences Validation:");
    let mut too_many_preferences = HashMap::new();
    for i in 0..25 {
        too_many_preferences.insert(format!("pref_{}", i), "value".to_string());
    }

    let invalid_prefs_input = ValidatedUserInput {
        username: "username".to_string(),
        email: "test@example.com".to_string(),
        age: 25,
        preferences: too_many_preferences,
    };

    match invalid_prefs_input.to_json() {
        Ok(json) => match ValidatedUserInput::from_json(&json) {
            Ok(_) => println!("   ✗ Unexpected: Too many preferences were accepted"),
            Err(e) => println!("   ✓ Expected validation error: {}", e),
        },
        Err(e) => println!("   ✗ Serialization error: {}", e),
    }

    Ok(())
}

/// Demonstrate schema evolution patterns
fn demonstrate_schema_evolution() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Schema Evolution Patterns ---");

    fs::create_dir_all("temp_schema_tests")?;

    // Create data with different schema versions
    println!("Creating data with different schema versions:");

    // Version 1 data (minimal)
    let v1_json = r#"{
        "version": 1,
        "name": "legacy_data",
        "value": 123.45
    }"#;
    fs::write("temp_schema_tests/v1_data.json", v1_json)?;
    println!("  ✓ Version 1 data created (minimal fields)");

    // Version 2 data (with optional field)
    let v2_json = r#"{
        "version": 2,
        "name": "v2_data",
        "value": 678.90,
        "optional_field": "added_in_v2"
    }"#;
    fs::write("temp_schema_tests/v2_data.json", v2_json)?;
    println!("  ✓ Version 2 data created (with optional field)");

    // Version 3 data (with all fields)
    let v3_data = VersionedData {
        version: 3,
        name: "v3_data".to_string(),
        value: 999.99,
        optional_field: Some("present".to_string()),
        new_field: Some(42),
    };
    v3_data.save_json("temp_schema_tests/v3_data.json")?;
    println!("  ✓ Version 3 data created (all fields)");

    // Test backward compatibility
    println!("\nTesting backward compatibility:");

    // Load v1 data with current deserializer
    match VersionedData::load_json("temp_schema_tests/v1_data.json") {
        Ok(data) => {
            println!("  ✓ V1 data loaded successfully:");
            println!("    Name: {}", data.name);
            println!("    Value: {}", data.value);
            println!("    Optional field: {:?}", data.optional_field);
            println!("    New field: {:?}", data.new_field);
        }
        Err(e) => println!("  ✗ Failed to load V1 data: {}", e),
    }

    // Load v2 data with current deserializer
    match VersionedData::load_json("temp_schema_tests/v2_data.json") {
        Ok(data) => {
            println!("  ✓ V2 data loaded successfully:");
            println!("    Name: {}", data.name);
            println!("    Value: {}", data.value);
            println!("    Optional field: {:?}", data.optional_field);
            println!("    New field: {:?}", data.new_field);
        }
        Err(e) => println!("  ✗ Failed to load V2 data: {}", e),
    }

    // Test future version rejection
    println!("\nTesting future version handling:");
    let future_version_json = r#"{
        "version": 99,
        "name": "future_data",
        "value": 123.45,
        "unknown_field": "should_be_ignored"
    }"#;
    fs::write("temp_schema_tests/future_data.json", future_version_json)?;

    match VersionedData::load_json("temp_schema_tests/future_data.json") {
        Ok(_) => println!("  ✗ Unexpected: Future version was accepted"),
        Err(e) => println!("  ✓ Expected rejection of future version: {}", e),
    }

    // Demonstrate migration strategy
    println!("\nDemonstrating migration strategy:");
    println!("  Strategy: Load old format, upgrade to new format, save");

    // Simulate migrating v1 data to v3 format
    let v1_loaded = VersionedData::load_json("temp_schema_tests/v1_data.json")?;
    let v1_upgraded = VersionedData {
        version: 3,
        name: v1_loaded.name,
        value: v1_loaded.value,
        optional_field: Some("migrated_default".to_string()),
        new_field: Some(0),
    };

    v1_upgraded.save_json("temp_schema_tests/v1_migrated.json")?;
    println!("  ✓ V1 data migrated to V3 format");

    Ok(())
}

/// Demonstrate recovery strategies
fn demonstrate_recovery_strategies() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Recovery Strategies ---");

    fs::create_dir_all("temp_recovery_tests")?;

    // Strategy 1: Graceful degradation
    println!("1. Graceful Degradation Strategy:");

    // Create complete data
    let complete_data = RecoverableData {
        critical_field: "essential_info".to_string(),
        important_field: Some("important_info".to_string()),
        optional_field: Some("nice_to_have".to_string()),
        metadata: {
            let mut map = HashMap::new();
            map.insert("key1".to_string(), "value1".to_string());
            map.insert("key2".to_string(), "value2".to_string());
            map
        },
    };

    // Save complete data
    complete_data.save_json("temp_recovery_tests/complete.json")?;

    // Create partial data (missing some fields)
    let partial_json = r#"{
        "critical_field": "essential_info",
        "optional_field": "nice_to_have"
    }"#;
    fs::write("temp_recovery_tests/partial.json", partial_json)?;

    // Load partial data and demonstrate recovery
    match RecoverableData::load_json("temp_recovery_tests/partial.json") {
        Ok(recovered) => {
            println!("  ✓ Partial data recovered successfully:");
            println!("    Critical field: {}", recovered.critical_field);
            println!(
                "    Important field: {:?} (missing, set to None)",
                recovered.important_field
            );
            println!("    Optional field: {:?}", recovered.optional_field);
            println!(
                "    Metadata: {} entries (defaulted to empty)",
                recovered.metadata.len()
            );
        }
        Err(e) => println!("  ✗ Recovery failed: {}", e),
    }

    // Strategy 2: Error context preservation
    println!("\n2. Error Context Preservation:");

    let malformed_json = r#"{
        "critical_field": "essential_info",
        "important_field": 12345,
        "metadata": "not_a_map"
    }"#;
    fs::write("temp_recovery_tests/malformed.json", malformed_json)?;

    match RecoverableData::load_json("temp_recovery_tests/malformed.json") {
        Ok(_) => println!("  ✗ Unexpected: Malformed data was accepted"),
        Err(e) => {
            println!("  ✓ Error context preserved:");
            println!("    Error: {}", e);
            println!("    Error type: {:?}", std::mem::discriminant(&e));
        }
    }

    // Strategy 3: Fallback data sources
    println!("\n3. Fallback Data Sources:");

    // Primary source (corrupted)
    let corrupted_primary = "corrupted data";
    fs::write("temp_recovery_tests/primary.json", corrupted_primary)?;

    // Backup source (valid)
    let backup_data = RecoverableData {
        critical_field: "backup_critical".to_string(),
        important_field: Some("backup_important".to_string()),
        optional_field: None,
        metadata: HashMap::new(),
    };
    backup_data.save_json("temp_recovery_tests/backup.json")?;

    // Default fallback
    let default_data = RecoverableData {
        critical_field: "default_critical".to_string(),
        important_field: None,
        optional_field: None,
        metadata: HashMap::new(),
    };

    println!("  Attempting to load data with fallback chain:");

    // Try primary source
    let loaded_data = match RecoverableData::load_json("temp_recovery_tests/primary.json") {
        Ok(data) => {
            println!("    ✓ Loaded from primary source");
            data
        }
        Err(_) => {
            println!("    ✗ Primary source failed, trying backup");

            // Try backup source
            match RecoverableData::load_json("temp_recovery_tests/backup.json") {
                Ok(data) => {
                    println!("    ✓ Loaded from backup source");
                    data
                }
                Err(_) => {
                    println!("    ✗ Backup source failed, using default");
                    default_data
                }
            }
        }
    };

    println!("  Final loaded data:");
    println!("    Critical field: {}", loaded_data.critical_field);

    Ok(())
}

/// Demonstrate production-ready error handling
fn demonstrate_production_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Production Error Handling ---");

    fs::create_dir_all("temp_production_tests")?;

    // Error logging and monitoring
    println!("1. Error Logging and Monitoring:");

    let test_data = VersionedData {
        version: 2,
        name: "production_test".to_string(),
        value: 42.0,
        optional_field: Some("test".to_string()),
        new_field: None,
    };

    // Create error log for demonstration
    let mut error_log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("temp_production_tests/error.log")?;

    // Function to log errors in production format
    let mut log_error =
        |error: &SerializationError, context: &str| -> Result<(), Box<dyn std::error::Error>> {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();

            writeln!(error_log, "[{}] ERROR in {}: {}", timestamp, context, error)?;
            Ok(())
        };

    // Simulate various error scenarios with logging
    let error_scenarios = vec![
        ("corrupted_file.json", "invalid json content"),
        ("missing_fields.json", r#"{"version": 1}"#),
        (
            "type_error.json",
            r#"{"version": "not_number", "name": "test", "value": 42.0}"#,
        ),
    ];

    for (filename, content) in error_scenarios {
        let filepath = format!("temp_production_tests/{}", filename);
        fs::write(&filepath, content)?;

        match VersionedData::load_json(&filepath) {
            Ok(_) => println!("  ✗ Unexpected success for {}", filename),
            Err(e) => {
                log_error(&e, &format!("load_config({})", filename))?;
                println!("  ✓ Error logged for {}: {}", filename, e);
            }
        }
    }

    // Health check pattern
    println!("\n2. Health Check Pattern:");

    let health_check = || -> Result<bool, SerializationError> {
        // Check if we can serialize/deserialize basic data
        let test_data = VersionedData {
            version: 1,
            name: "health_check".to_string(),
            value: 1.0,
            optional_field: None,
            new_field: None,
        };

        let serialized = test_data.to_json()?;
        let _deserialized = VersionedData::from_json(&serialized)?;
        Ok(true)
    };

    match health_check() {
        Ok(_) => println!("  ✓ Serialization system health check passed"),
        Err(e) => {
            log_error(&e, "health_check")?;
            println!("  ✗ Serialization system health check failed: {}", e);
        }
    }

    // Circuit breaker pattern simulation
    println!("\n3. Circuit Breaker Pattern:");

    struct CircuitBreaker {
        failure_count: u32,
        failure_threshold: u32,
        is_open: bool,
    }

    impl CircuitBreaker {
        fn new(threshold: u32) -> Self {
            Self {
                failure_count: 0,
                failure_threshold: threshold,
                is_open: false,
            }
        }

        fn call<F, T>(&mut self, operation: F) -> Result<T, String>
        where
            F: FnOnce() -> Result<T, SerializationError>,
        {
            if self.is_open {
                return Err("Circuit breaker is open".to_string());
            }

            match operation() {
                Ok(result) => {
                    self.failure_count = 0; // Reset on success
                    Ok(result)
                }
                Err(e) => {
                    self.failure_count += 1;
                    if self.failure_count >= self.failure_threshold {
                        self.is_open = true;
                        println!(
                            "    Circuit breaker opened after {} failures",
                            self.failure_count
                        );
                    }
                    Err(e.to_string())
                }
            }
        }
    }

    let mut circuit_breaker = CircuitBreaker::new(3);

    // Simulate operations that fail
    for i in 1..=5 {
        let result = circuit_breaker
            .call(|| VersionedData::load_json("temp_production_tests/corrupted_file.json"));

        match result {
            Ok(_) => println!("  Operation {} succeeded", i),
            Err(e) => println!("  Operation {} failed: {}", i, e),
        }
    }

    // Retry mechanism
    println!("\n4. Retry Mechanism:");

    let retry_operation = |max_attempts: u32| -> Result<VersionedData, String> {
        for attempt in 1..=max_attempts {
            println!("    Attempt {}/{}", attempt, max_attempts);

            // Try different sources in order
            let sources = vec![
                "temp_production_tests/corrupted_file.json",
                "temp_production_tests/missing_fields.json",
                "temp_production_tests/backup_valid.json",
            ];

            if attempt == max_attempts {
                // On final attempt, create valid backup
                test_data
                    .save_json("temp_production_tests/backup_valid.json")
                    .map_err(|e| format!("Failed to create backup: {}", e))?;
            }

            for source in &sources {
                match VersionedData::load_json(source) {
                    Ok(data) => {
                        println!("    ✓ Succeeded loading from {}", source);
                        return Ok(data);
                    }
                    Err(_) => {
                        println!("    ✗ Failed to load from {}", source);
                        continue;
                    }
                }
            }

            if attempt < max_attempts {
                println!("    Waiting before retry...");
                // In real code, would sleep here
            }
        }

        Err("All retry attempts exhausted".to_string())
    };

    match retry_operation(3) {
        Ok(data) => println!("  ✓ Retry succeeded: {}", data.name),
        Err(e) => println!("  ✗ Retry failed: {}", e),
    }

    Ok(())
}

/// Clean up temporary files created during the example
fn cleanup_temp_files() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Cleanup ---");

    let directories_to_remove = [
        "temp_error_tests",
        "temp_schema_tests",
        "temp_recovery_tests",
        "temp_production_tests",
    ];

    for dir in &directories_to_remove {
        if std::path::Path::new(dir).exists() {
            fs::remove_dir_all(dir)?;
            println!("Removed directory: {}", dir);
        }
    }

    println!("Cleanup completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_versioned_data_validation() {
        // Valid version should work
        let valid_data = VersionedData {
            version: 2,
            name: "test".to_string(),
            value: 1.0,
            optional_field: None,
            new_field: None,
        };

        let json = valid_data.to_json().unwrap();
        let parsed = VersionedData::from_json(&json).unwrap();
        assert_eq!(valid_data, parsed);

        // Invalid version should fail
        let invalid_json = r#"{"version": 99, "name": "test", "value": 1.0}"#;
        assert!(VersionedData::from_json(invalid_json).is_err());
    }

    #[test]
    fn test_user_input_validation() {
        // Valid input
        let valid_input = ValidatedUserInput {
            username: "valid_user".to_string(),
            email: "user@example.com".to_string(),
            age: 25,
            preferences: HashMap::new(),
        };

        let json = valid_input.to_json().unwrap();
        let parsed = ValidatedUserInput::from_json(&json).unwrap();
        assert_eq!(valid_input, parsed);

        // Invalid username
        let invalid_input = ValidatedUserInput {
            username: "".to_string(),
            email: "user@example.com".to_string(),
            age: 25,
            preferences: HashMap::new(),
        };

        let json = invalid_input.to_json().unwrap();
        assert!(ValidatedUserInput::from_json(&json).is_err());
    }

    #[test]
    fn test_recoverable_data_fallbacks() {
        // Complete data should work normally
        let complete = RecoverableData {
            critical_field: "critical".to_string(),
            important_field: Some("important".to_string()),
            optional_field: Some("optional".to_string()),
            metadata: HashMap::new(),
        };

        let json = complete.to_json().unwrap();
        let parsed = RecoverableData::from_json(&json).unwrap();
        assert_eq!(complete, parsed);

        // Partial data should recover gracefully
        let partial_json = r#"{"critical_field": "critical"}"#;
        let parsed = RecoverableData::from_json(partial_json).unwrap();
        assert_eq!(parsed.critical_field, "critical");
        assert!(parsed.important_field.is_none());
        assert!(parsed.optional_field.is_none());
        assert!(parsed.metadata.is_empty());
    }
}
