//! Basic Struct Serialization Example
//!
//! This example demonstrates fundamental serialization patterns in Train Station:
//! - Implementing StructSerializable for custom structs
//! - Basic field types and their serialization behavior
//! - JSON and binary format roundtrip operations
//! - Best practices for struct design and serialization
//! - File persistence and loading workflows
//!
//! # Learning Objectives
//!
//! - Understand StructSerializable trait implementation
//! - Learn field-by-field serialization patterns
//! - Master basic data type serialization
//! - Explore format selection criteria
//! - Implement robust save/load workflows
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of struct definitions
//! - Familiarity with file I/O concepts
//!
//! # Usage
//!
//! ```bash
//! cargo run --example basic_structs
//! ```

use std::{collections::HashMap, fs};
use train_station::serialization::{
    FieldValue, FromFieldValue, SerializationError, SerializationResult, StructDeserializer,
    StructSerializable, StructSerializer, ToFieldValue,
};

/// Simple user profile struct demonstrating basic field types
#[derive(Debug, Clone, PartialEq)]
pub struct UserProfile {
    pub id: u32,
    pub username: String,
    pub email: String,
    pub age: i32, // Changed from u16 to i32 for better JSON compatibility
    pub is_active: bool,
    pub score: f32, // Changed from f64 to f32 for better compatibility
}

impl StructSerializable for UserProfile {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("id", &self.id)
            .field("username", &self.username)
            .field("email", &self.email)
            .field("age", &self.age)
            .field("is_active", &self.is_active)
            .field("score", &self.score)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let id = deserializer.field("id")?;
        let username = deserializer.field("username")?;
        let email = deserializer.field("email")?;
        let age = deserializer.field("age")?;
        let is_active = deserializer.field("is_active")?;
        let score = deserializer.field("score")?;

        Ok(UserProfile {
            id,
            username,
            email,
            age,
            is_active,
            score,
        })
    }
}

impl ToFieldValue for UserProfile {
    fn to_field_value(&self) -> FieldValue {
        // Convert to JSON and then parse as FieldValue for nested object handling
        match self.to_json() {
            Ok(json_str) => {
                // For examples, we'll serialize as JSON string for simplicity
                FieldValue::from_json_object(json_str)
            }
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for UserProfile {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize UserProfile from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize UserProfile from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for UserProfile, found {}",
                value.type_name()
            ),
        })
    }
}

/// Application settings struct with optional fields and collections
#[derive(Debug, Clone, PartialEq)]
pub struct AppSettings {
    pub app_name: String,
    pub version: String,
    pub debug_mode: bool,
    pub max_connections: u32,
    pub timeout_seconds: f32,
    pub features: Vec<String>,
    pub environment_vars: HashMap<String, String>,
    pub optional_database_url: Option<String>,
}

impl StructSerializable for AppSettings {
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("app_name", &self.app_name)
            .field("version", &self.version)
            .field("debug_mode", &self.debug_mode)
            .field("max_connections", &self.max_connections)
            .field("timeout_seconds", &self.timeout_seconds)
            .field("features", &self.features)
            .field("environment_vars", &self.environment_vars)
            .field("optional_database_url", &self.optional_database_url)
    }

    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let app_name = deserializer.field("app_name")?;
        let version = deserializer.field("version")?;
        let debug_mode = deserializer.field("debug_mode")?;
        let max_connections = deserializer.field("max_connections")?;
        let timeout_seconds = deserializer.field("timeout_seconds")?;
        let features = deserializer.field("features")?;
        let environment_vars = deserializer.field("environment_vars")?;
        let optional_database_url = deserializer.field("optional_database_url")?;

        Ok(AppSettings {
            app_name,
            version,
            debug_mode,
            max_connections,
            timeout_seconds,
            features,
            environment_vars,
            optional_database_url,
        })
    }
}

impl ToFieldValue for AppSettings {
    fn to_field_value(&self) -> FieldValue {
        // Convert to JSON and then parse as FieldValue for nested object handling
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("serialization_error".to_string()),
        }
    }
}

impl FromFieldValue for AppSettings {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try JSON object first
        if let Ok(json_data) = value.as_json_object() {
            return Self::from_json(json_data).map_err(|e| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Failed to deserialize AppSettings from JSON: {}", e),
            });
        }

        // Try binary object
        if let Ok(binary_data) = value.as_binary_object() {
            return Self::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize AppSettings from binary: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: format!(
                "Expected JsonObject or BinaryObject for AppSettings, found {}",
                value.type_name()
            ),
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Struct Serialization Example ===\n");

    demonstrate_user_profile_serialization()?;
    demonstrate_app_settings_serialization()?;
    demonstrate_format_comparison()?;
    demonstrate_roundtrip_verification()?;
    demonstrate_field_access_patterns()?;
    cleanup_temp_files()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Demonstrate basic struct serialization with simple field types
fn demonstrate_user_profile_serialization() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- User Profile Serialization ---");

    // Create a user profile with various field types
    let user = UserProfile {
        id: 12345,
        username: "alice_cooper".to_string(),
        email: "alice@example.com".to_string(),
        age: 28,
        is_active: true,
        score: 95.7,
    };

    println!("Original user profile:");
    println!("  ID: {}", user.id);
    println!("  Username: {}", user.username);
    println!("  Email: {}", user.email);
    println!("  Age: {}", user.age);
    println!("  Active: {}", user.is_active);
    println!("  Score: {}", user.score);

    // Serialize to JSON
    let json_data = user.to_json()?;
    println!("\nSerialized to JSON:");
    println!("{}", json_data);

    // Save to JSON file
    user.save_json("temp_user_profile.json")?;
    println!("Saved to file: temp_user_profile.json");

    // Load from JSON file
    let loaded_user = UserProfile::load_json("temp_user_profile.json")?;
    println!("\nLoaded user profile:");
    println!("  ID: {}", loaded_user.id);
    println!("  Username: {}", loaded_user.username);
    println!("  Email: {}", loaded_user.email);
    println!("  Age: {}", loaded_user.age);
    println!("  Active: {}", loaded_user.is_active);
    println!("  Score: {}", loaded_user.score);

    // Verify data integrity
    assert_eq!(user, loaded_user);
    println!("Data integrity verification: PASSED");

    Ok(())
}

/// Demonstrate serialization with collections and optional fields
fn demonstrate_app_settings_serialization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- App Settings Serialization ---");

    // Create app settings with collections and optional fields
    let mut env_vars = HashMap::new();
    env_vars.insert("LOG_LEVEL".to_string(), "info".to_string());
    env_vars.insert("PORT".to_string(), "8080".to_string());
    env_vars.insert("HOST".to_string(), "localhost".to_string());

    let settings = AppSettings {
        app_name: "Train Station Example".to_string(),
        version: "1.0.0".to_string(),
        debug_mode: true,
        max_connections: 100,
        timeout_seconds: 30.5,
        features: vec![
            "authentication".to_string(),
            "logging".to_string(),
            "metrics".to_string(),
        ],
        environment_vars: env_vars,
        optional_database_url: Some("postgresql://localhost:5432/mydb".to_string()),
    };

    println!("Original app settings:");
    println!("  App Name: {}", settings.app_name);
    println!("  Version: {}", settings.version);
    println!("  Debug Mode: {}", settings.debug_mode);
    println!("  Max Connections: {}", settings.max_connections);
    println!("  Timeout: {} seconds", settings.timeout_seconds);
    println!("  Features: {:?}", settings.features);
    println!("  Environment Variables: {:?}", settings.environment_vars);
    println!("  Database URL: {:?}", settings.optional_database_url);

    // Serialize to binary format for efficient storage
    let binary_data = settings.to_binary()?;
    println!("\nSerialized to binary: {} bytes", binary_data.len());

    // Save to binary file
    settings.save_binary("temp_app_settings.bin")?;
    println!("Saved to file: temp_app_settings.bin");

    // Load from binary file
    let loaded_settings = AppSettings::load_binary("temp_app_settings.bin")?;
    println!("\nLoaded app settings:");
    println!("  App Name: {}", loaded_settings.app_name);
    println!("  Version: {}", loaded_settings.version);
    println!("  Debug Mode: {}", loaded_settings.debug_mode);
    println!("  Features count: {}", loaded_settings.features.len());
    println!(
        "  Environment variables count: {}",
        loaded_settings.environment_vars.len()
    );

    // Verify data integrity
    assert_eq!(settings, loaded_settings);
    println!("Data integrity verification: PASSED");

    Ok(())
}

/// Demonstrate format comparison between JSON and binary
fn demonstrate_format_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Format Comparison ---");

    let user = UserProfile {
        id: 98765,
        username: "bob_builder".to_string(),
        email: "bob@construction.com".to_string(),
        age: 35,
        is_active: false,
        score: 87.2,
    };

    // Save in both formats
    user.save_json("temp_format_comparison.json")?;
    user.save_binary("temp_format_comparison.bin")?;

    // Compare file sizes
    let json_size = fs::metadata("temp_format_comparison.json")?.len();
    let binary_size = fs::metadata("temp_format_comparison.bin")?.len();

    println!("Format comparison for UserProfile:");
    println!("  JSON file size: {} bytes", json_size);
    println!("  Binary file size: {} bytes", binary_size);
    println!(
        "  Size ratio (JSON/Binary): {:.2}x",
        json_size as f64 / binary_size as f64
    );

    // Demonstrate readability
    let json_content = fs::read_to_string("temp_format_comparison.json")?;
    println!("\nJSON format (human-readable):");
    println!("{}", json_content);

    println!("\nBinary format (first 32 bytes as hex):");
    let binary_content = fs::read("temp_format_comparison.bin")?;
    for (i, byte) in binary_content.iter().take(32).enumerate() {
        if i % 16 == 0 && i > 0 {
            println!();
        }
        print!("{:02x} ", byte);
    }
    println!("\n... ({} total bytes)", binary_content.len());

    // Load and verify both formats produce identical results
    let json_loaded = UserProfile::load_json("temp_format_comparison.json")?;
    let binary_loaded = UserProfile::load_binary("temp_format_comparison.bin")?;

    assert_eq!(json_loaded, binary_loaded);
    println!("\nFormat consistency verification: PASSED");

    Ok(())
}

/// Demonstrate roundtrip verification with multiple data variations
fn demonstrate_roundtrip_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Roundtrip Verification ---");

    // Test various data patterns
    let test_users = [
        UserProfile {
            id: 0,
            username: "".to_string(),
            email: "empty@test.com".to_string(),
            age: 0,
            is_active: false,
            score: 0.0,
        },
        UserProfile {
            id: u32::MAX,
            username: "maximal_user_with_very_long_name_123456789".to_string(),
            email: "test@verylongdomainname.example.org".to_string(),
            age: i32::MAX,
            is_active: true,
            score: 999999.5,
        },
        UserProfile {
            id: 42,
            username: "unicode_tëst_🦀".to_string(),
            email: "unicode@tëst.com".to_string(),
            age: 25,
            is_active: true,
            score: -123.456,
        },
    ];

    println!(
        "Testing roundtrip serialization with {} variations:",
        test_users.len()
    );

    for (i, user) in test_users.iter().enumerate() {
        println!(
            "  Test case {}: ID={}, Username='{}'",
            i + 1,
            user.id,
            user.username
        );

        // JSON roundtrip
        let json_data = user.to_json()?;
        let json_parsed = UserProfile::from_json(&json_data)?;
        assert_eq!(*user, json_parsed);

        // Binary roundtrip
        let binary_data = user.to_binary()?;
        let binary_parsed = UserProfile::from_binary(&binary_data)?;
        assert_eq!(*user, binary_parsed);

        println!("    JSON roundtrip: PASSED");
        println!("    Binary roundtrip: PASSED");
    }

    println!("All roundtrip tests: PASSED");

    Ok(())
}

/// Demonstrate field access patterns and validation
fn demonstrate_field_access_patterns() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Field Access Patterns ---");

    let settings = AppSettings {
        app_name: "Field Test App".to_string(),
        version: "2.1.0".to_string(),
        debug_mode: false,
        max_connections: 50,
        timeout_seconds: 15.0,
        features: vec!["basic".to_string(), "advanced".to_string()],
        environment_vars: HashMap::new(),
        optional_database_url: None,
    };

    // Convert to JSON to inspect structure
    let json_data = settings.to_json()?;
    println!("JSON structure for field inspection:");

    // Count approximate fields by counting field separators
    let field_count = json_data.matches(':').count();
    println!("Estimated fields: {}", field_count);

    // Show structure (first few lines)
    let lines: Vec<&str> = json_data.lines().take(5).collect();
    for line in lines {
        println!("  {}", line.trim());
    }
    if json_data.lines().count() > 5 {
        println!("  ... ({} more lines)", json_data.lines().count() - 5);
    }

    // Demonstrate optional field handling
    println!("\nOptional field handling:");
    println!(
        "  Database URL is None: {}",
        settings.optional_database_url.is_none()
    );

    // Create version with optional field populated
    let settings_with_db = AppSettings {
        optional_database_url: Some("sqlite:///tmp/test.db".to_string()),
        ..settings.clone()
    };

    println!(
        "  Database URL with value: {:?}",
        settings_with_db.optional_database_url
    );

    // Verify both versions serialize/deserialize correctly
    let json_none = settings.to_json()?;
    let json_some = settings_with_db.to_json()?;

    let parsed_none = AppSettings::from_json(&json_none)?;
    let parsed_some = AppSettings::from_json(&json_some)?;

    assert_eq!(settings, parsed_none);
    assert_eq!(settings_with_db, parsed_some);
    assert!(parsed_none.optional_database_url.is_none());
    assert!(parsed_some.optional_database_url.is_some());

    println!("Optional field serialization: PASSED");

    Ok(())
}

/// Clean up temporary files created during the example
fn cleanup_temp_files() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Cleanup ---");

    let files_to_remove = [
        "temp_user_profile.json",
        "temp_app_settings.bin",
        "temp_format_comparison.json",
        "temp_format_comparison.bin",
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
    fn test_user_profile_serialization() {
        let user = UserProfile {
            id: 123,
            username: "test_user".to_string(),
            email: "test@example.com".to_string(),
            age: 30,
            is_active: true,
            score: 88.5,
        };

        // Test JSON roundtrip
        let json_data = user.to_json().unwrap();
        let parsed_user = UserProfile::from_json(&json_data).unwrap();
        assert_eq!(user, parsed_user);

        // Test binary roundtrip
        let binary_data = user.to_binary().unwrap();
        let parsed_user = UserProfile::from_binary(&binary_data).unwrap();
        assert_eq!(user, parsed_user);
    }

    #[test]
    fn test_app_settings_serialization() {
        let mut env_vars = HashMap::new();
        env_vars.insert("TEST".to_string(), "value".to_string());

        let settings = AppSettings {
            app_name: "Test App".to_string(),
            version: "1.0.0".to_string(),
            debug_mode: true,
            max_connections: 10,
            timeout_seconds: 5.0,
            features: vec!["test".to_string()],
            environment_vars: env_vars,
            optional_database_url: Some("test://db".to_string()),
        };

        // Test JSON roundtrip
        let json_data = settings.to_json().unwrap();
        let parsed_settings = AppSettings::from_json(&json_data).unwrap();
        assert_eq!(settings, parsed_settings);

        // Test binary roundtrip
        let binary_data = settings.to_binary().unwrap();
        let parsed_settings = AppSettings::from_binary(&binary_data).unwrap();
        assert_eq!(settings, parsed_settings);
    }

    #[test]
    fn test_optional_field_handling() {
        let settings_none = AppSettings {
            app_name: "Test".to_string(),
            version: "1.0.0".to_string(),
            debug_mode: false,
            max_connections: 1,
            timeout_seconds: 1.0,
            features: vec![],
            environment_vars: HashMap::new(),
            optional_database_url: None,
        };

        let settings_some = AppSettings {
            optional_database_url: Some("db://test".to_string()),
            ..settings_none.clone()
        };

        // Test both variants
        let json_none = settings_none.to_json().unwrap();
        let json_some = settings_some.to_json().unwrap();

        let parsed_none = AppSettings::from_json(&json_none).unwrap();
        let parsed_some = AppSettings::from_json(&json_some).unwrap();

        assert!(parsed_none.optional_database_url.is_none());
        assert!(parsed_some.optional_database_url.is_some());
        assert_eq!(settings_none, parsed_none);
        assert_eq!(settings_some, parsed_some);
    }
}
