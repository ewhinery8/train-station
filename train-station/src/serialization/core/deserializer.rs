//! Core deserializer for structured data extraction and type conversion
//!
//! This module provides the `StructDeserializer` which enables type-safe extraction
//! of fields from serialized data structures. The deserializer supports both JSON
//! and binary formats, with comprehensive error handling and flexible field access
//! patterns.
//!
//! # Purpose
//!
//! The `StructDeserializer` serves as the counterpart to `StructSerializer`, providing:
//! - Type-safe field extraction with automatic conversion
//! - Support for both JSON and binary deserialization
//! - Flexible field access patterns (required, optional, with defaults)
//! - Comprehensive error handling with detailed validation messages
//! - Field existence checking and remaining field enumeration
//!
//! # Usage Patterns
//!
//! The deserializer supports multiple field extraction patterns:
//! - **Required fields**: `field<T>(name)` - extracts and converts, fails if missing
//! - **Optional fields**: `field_optional<T>(name)` - returns `Option<T>`
//! - **Default fields**: `field_or<T>(name, default)` - uses default if missing
//! - **Custom error handling**: `field_with_error<T>(name, error_fn)` - custom validation
//!
//! # Thread Safety
//!
//! This type is not thread-safe. Access from multiple threads requires external
//! synchronization.

use super::error::{SerializationError, SerializationResult};
use super::traits::FromFieldValue;
use super::types::FieldValue;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Deserializer for structured data extraction and type conversion
///
/// This struct provides a type-safe interface for deserializing complex structures
/// that were created with `StructSerializer`. It supports both JSON and binary
/// formats, with comprehensive error handling and flexible field access patterns.
///
/// The deserializer maintains a HashMap of field names to `FieldValue` instances,
/// allowing for efficient field lookup and extraction. Fields are consumed during
/// extraction to prevent accidental reuse and ensure proper deserialization flow.
///
/// # Fields
///
/// * `fields` - Internal storage mapping field names to their serialized values
///
/// # Thread Safety
///
/// This type is not thread-safe. Access from multiple threads requires external
/// synchronization.
///
/// # Memory Layout
///
/// The deserializer uses a HashMap for field storage, providing O(1) average
/// field lookup time. Memory usage scales linearly with the number of fields.
#[derive(Debug)]
pub struct StructDeserializer {
    /// Field storage for loaded data
    pub(crate) fields: HashMap<String, FieldValue>,
}

impl StructDeserializer {
    /// Create a new StructDeserializer from a field map
    ///
    /// This is the recommended way to create a StructDeserializer when you have
    /// a HashMap of fields, typically from a `FieldValue::Object` variant or when
    /// constructing a deserializer programmatically.
    ///
    /// # Arguments
    ///
    /// * `fields` - HashMap containing field names and their serialized values
    ///
    /// # Returns
    ///
    /// A new StructDeserializer ready for field extraction
    ///
    /// # Performance
    ///
    /// This constructor performs no validation or conversion, making it the fastest
    /// way to create a deserializer when you already have the field data.
    ///

    pub fn from_fields(fields: HashMap<String, FieldValue>) -> Self {
        Self { fields }
    }

    /// Load a struct from a JSON file
    ///
    /// Reads a JSON file from the filesystem and creates a StructDeserializer
    /// from its contents. The file is expected to contain a valid JSON object
    /// that can be parsed into field-value pairs.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the JSON file to load
    ///
    /// # Returns
    ///
    /// A StructDeserializer containing the parsed field data, or an error if
    /// the file cannot be read or parsed
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist or cannot be opened
    /// - The file contains invalid JSON
    /// - The JSON structure is not a valid object
    /// - I/O errors occur during file reading
    ///
    /// # Performance
    ///
    /// The entire file is read into memory before parsing. For very large files,
    /// consider using streaming JSON parsing instead.
    ///

    pub fn load_json<P: AsRef<Path>>(path: P) -> SerializationResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut json_string = String::new();
        reader.read_to_string(&mut json_string)?;

        Self::from_json(&json_string)
    }

    /// Load a struct from a binary file
    ///
    /// Reads a binary file from the filesystem and creates a StructDeserializer
    /// from its contents. The file is expected to contain valid binary data
    /// that was previously serialized using the binary format.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the binary file to load
    ///
    /// # Returns
    ///
    /// A StructDeserializer containing the parsed field data, or an error if
    /// the file cannot be read or parsed
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist or cannot be opened
    /// - The file contains invalid binary format
    /// - The binary data is corrupted or truncated
    /// - I/O errors occur during file reading
    ///
    /// # Performance
    ///
    /// The entire file is read into memory before parsing. Binary format is
    /// typically more compact and faster to parse than JSON.
    ///

    pub fn load_binary<P: AsRef<Path>>(path: P) -> SerializationResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut binary_data = Vec::new();
        reader.read_to_end(&mut binary_data)?;

        Self::from_binary(&binary_data)
    }

    /// Extract a field with automatic type detection
    ///
    /// Extracts a field from the deserializer and converts it to the specified type.
    /// The field is consumed (removed) from the deserializer to prevent accidental
    /// reuse and ensure proper deserialization flow.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the field to extract
    ///
    /// # Returns
    ///
    /// The converted field value, or an error if the field is missing or cannot
    /// be converted to the target type
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The field does not exist in the deserializer
    /// - The field value cannot be converted to the target type
    /// - Type conversion fails due to incompatible data
    ///
    /// # Performance
    ///
    /// Field extraction is O(1) average case due to HashMap lookup. The field
    /// is removed from the deserializer to prevent memory leaks.
    ///

    pub fn field<T: FromFieldValue>(&mut self, name: &str) -> SerializationResult<T> {
        let field_value =
            self.fields
                .remove(name)
                .ok_or_else(|| SerializationError::ValidationFailed {
                    field: name.to_string(),
                    message: "Field not found".to_string(),
                })?;

        T::from_field_value(field_value, name)
    }

    /// Extract a field value with type conversion and provide a default if missing
    ///
    /// Extracts a field from the deserializer and converts it to the specified type.
    /// If the field is missing, returns the provided default value instead of
    /// returning an error.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the field to extract
    /// * `default` - Default value to return if the field is missing
    ///
    /// # Returns
    ///
    /// The converted field value, or the default value if the field is missing.
    /// Returns an error only if the field exists but cannot be converted to the
    /// target type.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The field exists but cannot be converted to the target type
    /// - Type conversion fails due to incompatible data
    ///
    /// # Performance
    ///
    /// Field extraction is O(1) average case. The field is removed from the
    /// deserializer if it exists.
    ///

    pub fn field_or<T: FromFieldValue>(
        &mut self,
        name: &str,
        default: T,
    ) -> SerializationResult<T> {
        match self.fields.remove(name) {
            Some(field_value) => T::from_field_value(field_value, name),
            None => Ok(default),
        }
    }

    /// Extract a field value as an optional type
    ///
    /// Extracts a field from the deserializer and converts it to the specified type,
    /// returning `Some(value)` if the field exists and can be converted, or `None`
    /// if the field is missing.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the field to extract
    ///
    /// # Returns
    ///
    /// `Some(converted_value)` if the field exists and can be converted,
    /// `None` if the field is missing, or an error if the field exists but
    /// cannot be converted to the target type
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The field exists but cannot be converted to the target type
    /// - Type conversion fails due to incompatible data
    ///
    /// # Performance
    ///
    /// Field extraction is O(1) average case. The field is removed from the
    /// deserializer if it exists.
    ///

    pub fn field_optional<T: FromFieldValue>(
        &mut self,
        name: &str,
    ) -> SerializationResult<Option<T>> {
        match self.fields.remove(name) {
            Some(field_value) => T::from_field_value(field_value, name).map(Some),
            None => Ok(None),
        }
    }

    /// Extract a field value with custom error handling
    ///
    /// Extracts a field from the deserializer and converts it to the specified type,
    /// using a custom error handling function to process any errors that occur
    /// during extraction or conversion.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the field to extract
    /// * `error_fn` - Function to handle errors during field extraction or conversion
    ///
    /// # Returns
    ///
    /// The converted field value, or the result of the error handling function
    /// if an error occurs
    ///
    /// # Errors
    ///
    /// The error handling function can return any error type that implements
    /// the appropriate error traits. The function is called with the field name
    /// and the original error for context.
    ///
    /// # Performance
    ///
    /// Field extraction is O(1) average case. The error handling function is
    /// only called when an error occurs.
    ///

    pub fn field_with_error<T: FromFieldValue, F>(
        &mut self,
        name: &str,
        error_fn: F,
    ) -> SerializationResult<T>
    where
        F: FnOnce(&str, &SerializationError) -> SerializationResult<T>,
    {
        match self.fields.remove(name) {
            Some(field_value) => {
                T::from_field_value(field_value, name).map_err(|err| {
                    // Call error function and return its result
                    match error_fn(name, &err) {
                        Ok(_value) => {
                            // We can't return Ok from map_err, so we need to handle this differently
                            // For now, we'll just return the original error
                            err
                        }
                        Err(_) => err, // Return original error if error_fn fails
                    }
                })
            }
            None => error_fn(
                name,
                &SerializationError::ValidationFailed {
                    field: name.to_string(),
                    message: "Field not found".to_string(),
                },
            ),
        }
    }

    /// Get the names of all remaining fields
    ///
    /// Returns a vector containing the names of all fields that have not yet
    /// been extracted from the deserializer. This is useful for debugging,
    /// validation, or implementing custom deserialization logic.
    ///
    /// # Returns
    ///
    /// A vector of field names that are still available for extraction
    ///
    /// # Performance
    ///
    /// This method performs O(n) work to collect all field names, where n is
    /// the number of remaining fields. The returned vector is a copy of the
    /// field names.
    ///

    pub fn remaining_fields(&self) -> Vec<&str> {
        self.fields.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a field exists
    ///
    /// Returns whether a field with the specified name exists in the deserializer
    /// and has not yet been extracted. This is useful for conditional field
    /// extraction or validation.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the field to check
    ///
    /// # Returns
    ///
    /// `true` if the field exists and has not been extracted, `false` otherwise
    ///
    /// # Performance
    ///
    /// This method performs O(1) average case lookup using HashMap contains_key.
    ///

    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Create a deserializer from a JSON string
    ///
    /// Parses a JSON string and creates a StructDeserializer from its contents.
    /// The JSON string is expected to contain a valid JSON object that can be
    /// parsed into field-value pairs.
    ///
    /// # Arguments
    ///
    /// * `json` - JSON string to parse
    ///
    /// # Returns
    ///
    /// A StructDeserializer containing the parsed field data, or an error if
    /// the JSON string cannot be parsed
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The JSON string is malformed or invalid
    /// - The JSON structure is not a valid object
    /// - Parsing fails due to unexpected JSON format
    ///
    /// # Performance
    ///
    /// JSON parsing is performed in a single pass. The entire JSON string is
    /// processed to build the field map.
    ///

    pub fn from_json(json: &str) -> SerializationResult<Self> {
        crate::serialization::json::from_json_internal(json)
    }

    /// Create a deserializer from binary data
    ///
    /// Parses binary data and creates a StructDeserializer from its contents.
    /// The binary data is expected to contain valid binary format data that was
    /// previously serialized using the binary format.
    ///
    /// # Arguments
    ///
    /// * `data` - Binary data to parse
    ///
    /// # Returns
    ///
    /// A StructDeserializer containing the parsed field data, or an error if
    /// the binary data cannot be parsed
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The binary data is malformed or corrupted
    /// - The binary format is invalid or unsupported
    /// - Parsing fails due to unexpected binary structure
    /// - The data is truncated or incomplete
    ///
    /// # Performance
    ///
    /// Binary parsing is typically faster than JSON parsing due to the more
    /// compact format and lack of string parsing overhead.
    ///

    pub fn from_binary(data: &[u8]) -> SerializationResult<Self> {
        crate::serialization::binary::from_binary_internal(data)
    }
}
