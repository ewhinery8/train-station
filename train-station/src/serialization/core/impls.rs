//! Core serialization trait implementations for standard Rust types
//!
//! This module provides comprehensive implementations of `ToFieldValue` and `FromFieldValue` traits
//! for all standard Rust types and common collections. These implementations enable seamless
//! serialization and deserialization of structured data through the core serialization framework.
//!
//! # Purpose
//!
//! The implementations module serves as the foundation for type-safe serialization by providing:
//! - Automatic conversion between Rust types and `FieldValue` representations
//! - Support for primitive types, collections, and custom serializable types
//! - JSON-compatible serialization formats for human-readable data
//! - Binary-compatible formats for efficient storage and transmission
//! - Error handling with detailed validation messages
//!
//! # Supported Types
//!
//! ## Primitive Types
//! - **Integers**: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `usize`
//! - **Floating Point**: `f32`, `f64`
//! - **Boolean**: `bool`
//! - **String Types**: `String`, `&str`, `str`
//!
//! ## Collections
//! - **Vectors**: `Vec<T>` for all supported types
//! - **Arrays**: Fixed-size arrays for common types
//! - **HashMaps**: `HashMap<String, String>` for key-value data
//! - **Options**: `Option<T>` for optional values
//!
//! ## Custom Types
//! - **Serializable Objects**: Types implementing `Serializable` trait
//! - **Struct Types**: Types implementing `StructSerializable` trait
//!
//! # Serialization Strategy
//!
//! ## JSON Compatibility
//! - Numeric vectors are serialized as human-readable arrays
//! - String data maintains UTF-8 encoding
//! - Boolean values are preserved as true/false
//! - Collections maintain their structure and relationships
//!
//! ## Binary Efficiency
//! - Raw byte arrays use compact binary representation
//! - Numeric data uses native endianness
//! - String data includes length prefixes for efficient parsing
//! - Collections use optimized storage formats
//!
//! ## Error Handling
//! - Detailed validation error messages with field names
//! - Type mismatch detection with clear error descriptions
//! - Array element validation with indexed error reporting
//! - Backward compatibility for legacy data formats
//!
//! # Usage Patterns
//!
//! The trait implementations are automatically used when:
//! - Serializing structs with `StructSerializer`
//! - Deserializing data with `StructDeserializer`
//! - Converting between `FieldValue` and concrete types
//! - Handling nested serializable objects
//!
//! # Thread Safety
//!
//! All implementations are thread-safe and can be used concurrently across multiple threads.
//! No shared state is maintained between serialization operations.

use super::error::{SerializationError, SerializationResult};
use super::traits::{FromFieldValue, StructSerializable, ToFieldValue};
use super::types::FieldValue;
use std::collections::HashMap;

// ===== ToFieldValue implementations =====

/// Converts boolean values to FieldValue representation
///
/// Boolean values are stored as native boolean types in both JSON and binary formats,
/// maintaining their logical meaning across serialization boundaries.
impl ToFieldValue for bool {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_bool(*self)
    }
}

/// Converts 8-bit signed integers to FieldValue representation
///
/// Small integers are stored efficiently in both JSON and binary formats,
/// with proper range validation during deserialization.
impl ToFieldValue for i8 {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_i8(*self)
    }
}

/// Converts 16-bit signed integers to FieldValue representation
///
/// Medium-sized integers provide a balance between range and storage efficiency
/// for most common use cases.
impl ToFieldValue for i16 {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_i16(*self)
    }
}

/// Converts 32-bit signed integers to FieldValue representation
///
/// Standard integer type used for most numeric data, providing sufficient range
/// for typical application needs.
impl ToFieldValue for i32 {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_i32(*self)
    }
}

/// Converts 64-bit signed integers to FieldValue representation
///
/// Large integer type for values requiring extended range, such as timestamps,
/// file sizes, or large counts.
impl ToFieldValue for i64 {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_i64(*self)
    }
}

/// Converts 8-bit unsigned integers to FieldValue representation
///
/// Small unsigned integers for values that are always non-negative,
/// such as array indices or small counts.
impl ToFieldValue for u8 {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_u8(*self)
    }
}

/// Converts 16-bit unsigned integers to FieldValue representation
///
/// Medium-sized unsigned integers for values requiring positive range
/// with moderate storage efficiency.
impl ToFieldValue for u16 {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_u16(*self)
    }
}

/// Converts 32-bit unsigned integers to FieldValue representation
///
/// Standard unsigned integer type for large positive values,
/// commonly used for sizes, counts, and identifiers.
impl ToFieldValue for u32 {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_u32(*self)
    }
}

/// Converts 64-bit unsigned integers to FieldValue representation
///
/// Large unsigned integer type for very large positive values,
/// such as memory addresses or large file sizes.
impl ToFieldValue for u64 {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_u64(*self)
    }
}

/// Converts platform-specific size integers to FieldValue representation
///
/// Size type that adapts to the platform's pointer size, commonly used
/// for array lengths, memory sizes, and indexing operations.
impl ToFieldValue for usize {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_usize(*self)
    }
}

/// Converts 32-bit floating point numbers to FieldValue representation
///
/// Single-precision floating point for values requiring decimal precision
/// with moderate storage requirements.
impl ToFieldValue for f32 {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_f32(*self)
    }
}

/// Converts 64-bit floating point numbers to FieldValue representation
///
/// Double-precision floating point for values requiring high decimal precision,
/// commonly used for scientific calculations and financial data.
impl ToFieldValue for f64 {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_f64(*self)
    }
}

/// Converts string slices to FieldValue representation
///
/// String slices are converted to owned strings for serialization,
/// maintaining UTF-8 encoding and preserving all character data.
impl ToFieldValue for str {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_string_slice(self)
    }
}

/// Converts string references to FieldValue representation
///
/// String references are converted to owned strings for serialization,
/// ensuring data ownership and thread safety.
impl ToFieldValue for &str {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_string_slice(self)
    }
}

/// Converts owned strings to FieldValue representation
///
/// Owned strings are cloned for serialization to maintain data ownership
/// while preserving the original string for further use.
impl ToFieldValue for String {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_string(self.clone())
    }
}

/// Converts byte slices to FieldValue representation
///
/// Byte slices are converted to owned byte vectors for serialization,
/// preserving raw binary data without encoding assumptions.
impl ToFieldValue for [u8] {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_bytes(self.to_vec())
    }
}

/// Converts byte vectors to FieldValue representation
///
/// Byte vectors are cloned for serialization to maintain data ownership
/// while preserving the original vector for further use.
impl ToFieldValue for Vec<u8> {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_bytes(self.clone())
    }
}

/// Converts integer vectors to FieldValue representation
///
/// Integer vectors are serialized as JSON-compatible arrays for human readability,
/// while maintaining efficient binary storage for performance-critical applications.
impl ToFieldValue for Vec<i32> {
    fn to_field_value(&self) -> FieldValue {
        // Use Array format for JSON compatibility (human-readable numbers)
        let mut values = Vec::new();
        for &val in self {
            values.push(FieldValue::from_i32(val));
        }
        FieldValue::from_array(values)
    }
}

/// Converts size vectors to FieldValue representation
///
/// Size vectors are serialized as JSON-compatible arrays, adapting to the platform's
/// pointer size while maintaining cross-platform compatibility.
impl ToFieldValue for Vec<usize> {
    fn to_field_value(&self) -> FieldValue {
        // Use Array format for JSON compatibility (human-readable numbers)
        let mut values = Vec::new();
        for &val in self {
            values.push(FieldValue::from_usize(val));
        }
        FieldValue::from_array(values)
    }
}

/// Converts single-precision float vectors to FieldValue representation
///
/// Float vectors are serialized as JSON-compatible arrays to maintain human readability
/// while preserving numerical precision for scientific and engineering applications.
impl ToFieldValue for Vec<f32> {
    fn to_field_value(&self) -> FieldValue {
        // CRITICAL: JSON must remain human-readable as number arrays
        // Use Array format for JSON compatibility (not bytes)
        let mut values = Vec::new();
        for &val in self {
            values.push(FieldValue::from_f32(val));
        }
        FieldValue::from_array(values)
    }
}

/// Converts double-precision float vectors to FieldValue representation
///
/// Double-precision float vectors maintain high numerical accuracy for scientific
/// calculations while providing JSON-compatible serialization for data exchange.
impl ToFieldValue for Vec<f64> {
    fn to_field_value(&self) -> FieldValue {
        // Use Array format for JSON compatibility (human-readable numbers)
        let mut values = Vec::new();
        for &val in self {
            values.push(FieldValue::from_f64(val));
        }
        FieldValue::from_array(values)
    }
}

/// Converts boolean vectors to FieldValue representation
///
/// Boolean vectors are serialized as JSON-compatible arrays, maintaining logical
/// relationships and providing clear true/false representation.
impl ToFieldValue for Vec<bool> {
    fn to_field_value(&self) -> FieldValue {
        let mut values = Vec::new();
        for &val in self {
            values.push(FieldValue::from_bool(val));
        }
        FieldValue::from_array(values)
    }
}

/// Converts string vectors to FieldValue representation
///
/// String vectors are serialized as JSON-compatible arrays, preserving UTF-8 encoding
/// and maintaining string relationships for text processing applications.
impl ToFieldValue for Vec<String> {
    fn to_field_value(&self) -> FieldValue {
        let mut values = Vec::new();
        for val in self {
            values.push(FieldValue::from_string(val.clone()));
        }
        FieldValue::from_array(values)
    }
}

/// Converts string hash maps to FieldValue representation
///
/// String hash maps are serialized as JSON objects, maintaining key-value relationships
/// for configuration data, metadata, and structured text storage.
impl ToFieldValue for HashMap<String, String> {
    fn to_field_value(&self) -> FieldValue {
        let mut object = HashMap::new();
        for (key, value) in self {
            object.insert(key.clone(), FieldValue::from_string(value.clone()));
        }
        FieldValue::from_object(object)
    }
}

/// Converts fixed-size byte arrays to FieldValue representation
///
/// Fixed-size byte arrays are converted to vectors for serialization,
/// maintaining the original data while providing flexible storage.
impl<const N: usize> ToFieldValue for [u8; N] {
    fn to_field_value(&self) -> FieldValue {
        FieldValue::from_bytes(self.to_vec())
    }
}

/// Converts fixed-size integer arrays to FieldValue representation
///
/// Fixed-size integer arrays are serialized as JSON-compatible arrays,
/// maintaining the original structure while providing human-readable format.
impl<const N: usize> ToFieldValue for [i32; N] {
    fn to_field_value(&self) -> FieldValue {
        let mut values = Vec::new();
        for &val in self {
            values.push(FieldValue::from_i32(val));
        }
        FieldValue::from_array(values)
    }
}

/// Converts optional values to FieldValue representation
///
/// Optional values are serialized with explicit None/Some representation,
/// maintaining the optional nature of the data across serialization boundaries.
impl<T> ToFieldValue for Option<T>
where
    T: ToFieldValue,
{
    fn to_field_value(&self) -> FieldValue {
        match self {
            Some(value) => FieldValue::from_optional(Some(value.to_field_value())),
            None => FieldValue::from_optional(None),
        }
    }
}

/// Converts serializable struct vectors to FieldValue representation
///
/// Struct vectors are serialized as arrays of objects, maintaining the structure
/// and relationships of complex data while providing JSON-compatible format.
impl<T> ToFieldValue for Vec<T>
where
    T: StructSerializable,
{
    fn to_field_value(&self) -> FieldValue {
        let field_values: Vec<FieldValue> = self
            .iter()
            .map(|item| {
                // Convert struct to field-value map and then to Object
                let serializer = item.to_serializer();
                let mut object = HashMap::new();
                for (field_name, field_value) in serializer.fields {
                    object.insert(field_name, field_value);
                }
                FieldValue::from_object(object)
            })
            .collect();
        FieldValue::from_array(field_values)
    }
}

/// Converts serializable objects to FieldValue representation
///
/// Serializable objects are converted using their JSON representation,
/// providing format-aware serialization for complex data structures.
impl<T> ToFieldValue for T
where
    T: crate::serialization::Serializable,
{
    fn to_field_value(&self) -> FieldValue {
        // This will be handled by format-aware serialization
        // For now, default to JSON serialization
        match self.to_json() {
            Ok(json_str) => FieldValue::from_json_object(json_str),
            Err(_) => FieldValue::from_string("ERROR: Failed to serialize object".to_string()),
        }
    }
}

// ===== FromFieldValue implementations =====

/// Converts FieldValue representation back to boolean values
///
/// Validates that the FieldValue contains a boolean type and returns
/// the corresponding boolean value with detailed error reporting.
impl FromFieldValue for bool {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_bool()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected bool, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to 8-bit signed integers
///
/// Validates that the FieldValue contains an i8 type and returns
/// the corresponding integer value with range validation.
impl FromFieldValue for i8 {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_i8()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected i8, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to 16-bit signed integers
///
/// Validates that the FieldValue contains an i16 type and returns
/// the corresponding integer value with range validation.
impl FromFieldValue for i16 {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_i16()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected i16, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to 32-bit signed integers
///
/// Validates that the FieldValue contains an i32 type and returns
/// the corresponding integer value with range validation.
impl FromFieldValue for i32 {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_i32()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected i32, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to 64-bit signed integers
///
/// Validates that the FieldValue contains an i64 type and returns
/// the corresponding integer value with range validation.
impl FromFieldValue for i64 {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_i64()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected i64, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to 8-bit unsigned integers
///
/// Validates that the FieldValue contains a u8 type and returns
/// the corresponding integer value with range validation.
impl FromFieldValue for u8 {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_u8()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected u8, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to 16-bit unsigned integers
///
/// Validates that the FieldValue contains a u16 type and returns
/// the corresponding integer value with range validation.
impl FromFieldValue for u16 {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_u16()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected u16, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to 32-bit unsigned integers
///
/// Validates that the FieldValue contains a u32 type and returns
/// the corresponding integer value with range validation.
impl FromFieldValue for u32 {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_u32()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected u32, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to 64-bit unsigned integers
///
/// Validates that the FieldValue contains a u64 type and returns
/// the corresponding integer value with range validation.
impl FromFieldValue for u64 {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_u64()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected u64, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to platform-specific size integers
///
/// Validates that the FieldValue contains a usize type and returns
/// the corresponding integer value with platform-appropriate range validation.
impl FromFieldValue for usize {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_usize()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected usize, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to 32-bit floating point numbers
///
/// Validates that the FieldValue contains an f32 type and returns
/// the corresponding floating point value with precision validation.
impl FromFieldValue for f32 {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_f32()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected f32, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to 64-bit floating point numbers
///
/// Validates that the FieldValue contains an f64 type and returns
/// the corresponding floating point value with precision validation.
impl FromFieldValue for f64 {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_f64()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected f64, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to byte vectors
///
/// Validates that the FieldValue contains byte data and returns
/// the corresponding byte vector with proper error handling.
impl FromFieldValue for Vec<u8> {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .to_bytes()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected Vec<u8>, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to integer vectors
///
/// Validates that the FieldValue contains an array of integers and returns
/// the corresponding vector with element-by-element validation.
impl FromFieldValue for Vec<i32> {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Use Array format for JSON compatibility (human-readable numbers)
        value
            .as_array()
            .and_then(|array| {
                let mut result = Vec::new();
                for field_value in array {
                    result.push(field_value.as_i32().map_err(|_| {
                        SerializationError::ValidationFailed {
                            field: field_name.to_string(),
                            message: format!("Expected i32 in array, found {:?}", field_value),
                        }
                    })?);
                }
                Ok(result)
            })
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected Vec<i32>, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to size vectors
///
/// Validates that the FieldValue contains an array of size integers and returns
/// the corresponding vector with platform-appropriate validation.
impl FromFieldValue for Vec<usize> {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Use Array format for JSON compatibility (human-readable numbers)
        value
            .as_array()
            .and_then(|array| {
                let mut result = Vec::new();
                for field_value in array {
                    result.push(field_value.as_usize().map_err(|_| {
                        SerializationError::ValidationFailed {
                            field: field_name.to_string(),
                            message: format!("Expected usize in array, found {:?}", field_value),
                        }
                    })?);
                }
                Ok(result)
            })
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected Vec<usize>, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to single-precision float vectors
///
/// Validates that the FieldValue contains an array of f32 values and returns
/// the corresponding vector with indexed error reporting for debugging.
impl FromFieldValue for Vec<f32> {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Use Array format for JSON compatibility (human-readable numbers)
        value
            .as_array()
            .and_then(|array| {
                let mut result = Vec::new();
                for (index, field_value) in array.iter().enumerate() {
                    let element_field_name = format!("{}[{}]", field_name, index);
                    result.push(field_value.as_f32().map_err(|_| {
                        SerializationError::ValidationFailed {
                            field: element_field_name,
                            message: format!("Expected f32 in array, found {:?}", field_value),
                        }
                    })?);
                }
                Ok(result)
            })
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected Vec<f32>, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to double-precision float vectors
///
/// Validates that the FieldValue contains an array of f64 values and returns
/// the corresponding vector with precision validation.
impl FromFieldValue for Vec<f64> {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Use Array format for JSON compatibility (human-readable numbers)
        value
            .as_array()
            .and_then(|array| {
                let mut result = Vec::new();
                for field_value in array {
                    result.push(field_value.as_f64().map_err(|_| {
                        SerializationError::ValidationFailed {
                            field: field_name.to_string(),
                            message: format!("Expected f64 in array, found {:?}", field_value),
                        }
                    })?);
                }
                Ok(result)
            })
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected Vec<f64>, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to boolean vectors
///
/// Validates that the FieldValue contains an array of boolean values and returns
/// the corresponding vector with logical validation.
impl FromFieldValue for Vec<bool> {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_array()
            .and_then(|array| {
                let mut result = Vec::new();
                for field_value in array {
                    result.push(field_value.as_bool().map_err(|_| {
                        SerializationError::ValidationFailed {
                            field: field_name.to_string(),
                            message: format!("Expected bool in array, found {:?}", field_value),
                        }
                    })?);
                }
                Ok(result)
            })
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected Vec<bool>, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to string vectors
///
/// Validates that the FieldValue contains an array of strings and returns
/// the corresponding vector with UTF-8 validation.
impl FromFieldValue for Vec<String> {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        value
            .as_array()
            .and_then(|array| {
                let mut result = Vec::new();
                for field_value in array {
                    result.push(
                        field_value
                            .as_string()
                            .map(|s| s.to_string())
                            .map_err(|_| SerializationError::ValidationFailed {
                                field: field_name.to_string(),
                                message: format!(
                                    "Expected String in array, found {:?}",
                                    field_value
                                ),
                            })?,
                    );
                }
                Ok(result)
            })
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected Vec<String>, found {:?}", value),
            })
    }
}

/// Converts FieldValue representation back to string hash maps
///
/// Validates that the FieldValue contains key-value pairs and returns
/// the corresponding hash map with flexible value type conversion.
impl FromFieldValue for HashMap<String, String> {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::Object(object) => {
                let mut result = HashMap::new();
                for (key, field_value) in object {
                    // Convert any FieldValue to a string representation for HashMap<String, String>
                    let string_value = match field_value {
                        FieldValue::String(s) => s,
                        FieldValue::Bool(b) => b.to_string(),
                        FieldValue::I8(i) => i.to_string(),
                        FieldValue::I16(i) => i.to_string(),
                        FieldValue::I32(i) => i.to_string(),
                        FieldValue::I64(i) => i.to_string(),
                        FieldValue::U8(u) => u.to_string(),
                        FieldValue::U16(u) => u.to_string(),
                        FieldValue::U32(u) => u.to_string(),
                        FieldValue::U64(u) => u.to_string(),
                        FieldValue::Usize(u) => u.to_string(),
                        FieldValue::F32(f) => f.to_string(),
                        FieldValue::F64(f) => f.to_string(),
                        FieldValue::Bytes(bytes) => {
                            // Try to interpret bytes as hex string (if that's how they were stored)
                            String::from_utf8(bytes.clone()).unwrap_or_else(|_| {
                                // If not valid UTF-8, convert to hex representation
                                {
                                    let mut hex_string = String::with_capacity(bytes.len() * 2);
                                    for b in bytes {
                                        hex_string.push_str(&format!("{:02x}", b));
                                    }
                                    hex_string
                                }
                            })
                        }
                        FieldValue::JsonObject(json) => json,
                        _ => {
                            return Err(SerializationError::ValidationFailed {
                                field: format!("{}[{}]", field_name, key),
                                message: format!(
                                    "Cannot convert {} to string",
                                    field_value.type_name()
                                ),
                            });
                        }
                    };
                    result.insert(key, string_value);
                }
                Ok(result)
            }
            // Backward compatibility: support old array format
            FieldValue::Array(array) => {
                let mut result = HashMap::new();
                for field_value in array {
                    let key_value_str = field_value.as_string().map_err(|_| {
                        SerializationError::ValidationFailed {
                            field: field_name.to_string(),
                            message: format!("Expected string in array, found {:?}", field_value),
                        }
                    })?;

                    // Parse "key: value" format
                    if let Some(colon_pos) = key_value_str.find(':') {
                        let key = key_value_str[..colon_pos].trim().to_string();
                        let value = key_value_str[colon_pos + 1..].trim().to_string();
                        result.insert(key, value);
                    } else {
                        return Err(SerializationError::ValidationFailed {
                            field: field_name.to_string(),
                            message: format!("Invalid key-value format: {}", key_value_str),
                        });
                    }
                }
                Ok(result)
            }
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Expected Object or Array for HashMap<String, String>, found {:?}",
                    value.type_name()
                ),
            }),
        }
    }
}

/// Converts FieldValue representation back to strings
///
/// Validates that the FieldValue contains string data and returns
/// the corresponding string with support for both direct strings and JSON objects.
impl FromFieldValue for String {
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try string first, then JSON object
        if let Ok(s) = value.as_string() {
            Ok(s.to_string())
        } else if let Ok(json_str) = value.as_json_object() {
            Ok(json_str.to_string())
        } else {
            Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Expected String or JSON object, found {:?}", value),
            })
        }
    }
}

/// Converts FieldValue representation back to optional values
///
/// Validates that the FieldValue contains optional data and returns
/// the corresponding optional value with proper None/Some handling.
impl<T> FromFieldValue for Option<T>
where
    T: FromFieldValue,
{
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::Optional(Some(inner)) => T::from_field_value(*inner, field_name).map(Some),
            FieldValue::Optional(None) => Ok(None),
            // Also allow direct values for convenience
            _ => T::from_field_value(value, field_name).map(Some),
        }
    }
}

/// Converts FieldValue representation back to serializable struct vectors
///
/// Validates that the FieldValue contains an array of struct objects and returns
/// the corresponding vector with struct-by-struct deserialization.
impl<T> FromFieldValue for Vec<T>
where
    T: StructSerializable,
{
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::Array(array) => {
                let mut result = Vec::with_capacity(array.len());
                for (index, item) in array.into_iter().enumerate() {
                    match item {
                        FieldValue::Object(object) => {
                            // Convert Object back to StructDeserializer
                            let deserializer =
                                crate::serialization::core::StructDeserializer { fields: object };
                            let mut deserializer = deserializer;
                            let parsed_item =
                                T::from_deserializer(&mut deserializer).map_err(|e| {
                                    SerializationError::ValidationFailed {
                                        field: format!("{}[{}]", field_name, index),
                                        message: format!("Failed to deserialize struct: {}", e),
                                    }
                                })?;
                            result.push(parsed_item);
                        }
                        // Backward compatibility: support old JSON string format
                        FieldValue::String(json_string) => {
                            let parsed_item = T::from_json(&json_string).map_err(|e| {
                                SerializationError::ValidationFailed {
                                    field: format!("{}[{}]", field_name, index),
                                    message: format!("Failed to parse JSON: {}", e),
                                }
                            })?;
                            result.push(parsed_item);
                        }
                        _ => {
                            return Err(SerializationError::ValidationFailed {
                                field: format!("{}[{}]", field_name, index),
                                message: format!(
                                    "Expected Object or String, found {}",
                                    item.type_name()
                                ),
                            });
                        }
                    }
                }
                Ok(result)
            }
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Expected Array for Vec<{}>, found {}",
                    std::any::type_name::<T>(),
                    value.type_name()
                ),
            }),
        }
    }
}
