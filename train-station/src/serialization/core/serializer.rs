//! Core serializer for structured data serialization
//!
//! This module provides the `StructSerializer` which enables building complex serializable
//! structures using a fluent interface without requiring trait imports. The serializer
//! supports both JSON and binary formats with automatic type detection and field validation.
//!
//! # Purpose
//!
//! The serializer module provides:
//! - Fluent interface for building structured data without trait imports
//! - Automatic type detection and conversion through `ToFieldValue` trait
//! - Support for both JSON and binary serialization formats
//! - File I/O operations with proper error handling
//! - Type-safe field registration with insertion order preservation
//! - Memory-efficient serialization with pre-allocated capacity options
//!
//! # Usage Patterns
//!
//! The `StructSerializer` is designed for building complex data structures
//! with automatic type detection and fluent method chaining.
//!
//! # Serialization Formats
//!
//! ## JSON Format
//! - Human-readable text format for debugging and data exchange
//! - UTF-8 encoded with proper escaping
//! - Maintains field order and type information
//! - Compatible with standard JSON parsers
//!
//! ## Binary Format
//! - Compact binary representation for efficient storage
//! - Platform-independent with proper endianness handling
//! - Includes type information and field metadata
//! - Optimized for fast serialization and deserialization
//!
//! # Field Types
//!
//! The serializer supports all types implementing `ToFieldValue`:
//! - **Primitive types**: `bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `usize`, `f32`, `f64`
//! - **String types**: `String`, `&str`
//! - **Collections**: `Vec<T>`, `HashMap<String, String>`, `Option<T>`
//! - **Custom types**: Any type implementing `ToFieldValue` trait
//!
//! # Error Handling
//!
//! All serialization operations return `SerializationResult<T>` which provides:
//! - Detailed error messages for debugging
//! - Type-specific error information
//! - File I/O error propagation
//! - Validation error reporting
//!
//! # Thread Safety
//!
//! The `StructSerializer` is not thread-safe and should not be shared between threads.
//! Each thread should create its own serializer instance for concurrent operations.

use super::error::SerializationResult;
use super::traits::ToFieldValue;
use super::types::FieldValue;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Builder for serializing structured data with fluent interface
///
/// This struct provides a fluent interface for building complex serializable structures
/// without requiring trait imports. It maintains type safety while allowing for future
/// extensibility with field attributes and customization options.
///
/// The serializer supports both JSON and binary formats, with automatic type detection
/// through the `ToFieldValue` trait. Fields are stored in insertion order to maintain
/// consistent serialization output.
///
/// # Fields
///
/// * `fields` - Vector of field name-value pairs preserving insertion order
///
/// # Examples
///
/// The serializer provides a fluent interface for building structured data
/// with automatic type detection and support for both JSON and binary formats.
///
/// # Thread Safety
///
/// This type is not thread-safe. Each thread should create its own instance
/// for concurrent serialization operations.
#[derive(Debug)]
pub struct StructSerializer {
    /// Field storage with insertion order preservation
    ///
    /// Vector of (field_name, field_value) pairs that maintains the order
    /// in which fields were added to the serializer. This ensures consistent
    /// output across multiple serialization runs.
    pub(crate) fields: Vec<(String, FieldValue)>,
}

impl StructSerializer {
    /// Creates a new empty struct serializer
    ///
    /// Returns a new `StructSerializer` with an empty field collection.
    /// Use this constructor when you don't know the final number of fields
    /// in advance.
    ///
    /// # Returns
    ///
    /// A new `StructSerializer` instance ready for field registration
    ///
    /// # Examples
    ///
    /// Creates a new empty serializer ready for field registration.
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Creates a new struct serializer with pre-allocated capacity
    ///
    /// Returns a new `StructSerializer` with enough capacity to hold the specified
    /// number of fields without reallocating. This is useful for performance-critical
    /// applications where you know the number of fields in advance.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of fields the serializer should be able to hold without reallocating
    ///
    /// # Returns
    ///
    /// A new `StructSerializer` instance with pre-allocated capacity
    ///
    /// # Examples
    ///
    /// Creates a serializer with pre-allocated capacity for performance optimization.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            fields: Vec::with_capacity(capacity),
        }
    }

    /// Registers a field with automatic type detection
    ///
    /// Adds a field to the serializer with automatic type conversion through the
    /// `ToFieldValue` trait. The field name is preserved as a string, and the value
    /// is converted to a `FieldValue` for serialization.
    ///
    /// This method consumes `self` and returns a new `StructSerializer` with the
    /// field added, enabling fluent method chaining.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name as a string slice
    /// * `value` - Field value that implements `ToFieldValue`
    ///
    /// # Returns
    ///
    /// A new `StructSerializer` with the field added
    ///
    /// # Examples
    ///
    /// Adds a field with automatic type conversion and enables fluent method chaining.
    pub fn field<T>(mut self, name: &str, value: &T) -> Self
    where
        T: ToFieldValue,
    {
        let field_value = value.to_field_value();
        self.fields.push((name.to_string(), field_value));
        self
    }

    /// Converts the struct to a JSON string
    ///
    /// Serializes all registered fields to a JSON string format. The JSON output
    /// is human-readable and maintains field order and type information.
    ///
    /// This method consumes the serializer, preventing further field additions.
    ///
    /// # Returns
    ///
    /// `Ok(String)` containing the JSON representation on success
    /// `Err(SerializationError)` if serialization fails
    ///
    /// # Examples
    ///
    /// Serializes all fields to human-readable JSON format with proper escaping.
    pub fn to_json(self) -> SerializationResult<String> {
        crate::serialization::json::to_json_internal(self)
    }

    /// Converts the struct to binary data
    ///
    /// Serializes all registered fields to a compact binary format. The binary
    /// output is platform-independent and optimized for fast serialization
    /// and deserialization.
    ///
    /// This method consumes the serializer, preventing further field additions.
    ///
    /// # Returns
    ///
    /// `Ok(Vec<u8>)` containing the binary representation on success
    /// `Err(SerializationError)` if serialization fails
    ///
    /// # Examples
    ///
    /// Serializes all fields to compact binary format for efficient storage.
    pub fn to_binary(self) -> SerializationResult<Vec<u8>> {
        crate::serialization::binary::to_binary_internal(self)
    }

    /// Saves the struct to a JSON file
    ///
    /// Serializes all registered fields to JSON format and writes the result
    /// to the specified file path. The file is created if it doesn't exist,
    /// or truncated if it already exists.
    ///
    /// This method consumes the serializer, preventing further field additions.
    ///
    /// # Arguments
    ///
    /// * `path` - File path where the JSON data should be written
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful file write
    /// `Err(SerializationError)` if serialization or file I/O fails
    ///
    /// # Examples
    ///
    /// Saves serialized data to a JSON file with proper file I/O handling.
    pub fn save_json<P: AsRef<Path>>(self, path: P) -> SerializationResult<()> {
        let json_string = self.to_json()?;

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::new(file);
        writer.write_all(json_string.as_bytes())?;
        writer.flush()?;

        Ok(())
    }

    /// Saves the struct to a binary file
    ///
    /// Serializes all registered fields to binary format and writes the result
    /// to the specified file path. The file is created if it doesn't exist,
    /// or truncated if it already exists.
    ///
    /// This method consumes the serializer, preventing further field additions.
    ///
    /// # Arguments
    ///
    /// * `path` - File path where the binary data should be written
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful file write
    /// `Err(SerializationError)` if serialization or file I/O fails
    ///
    /// # Examples
    ///
    /// Saves serialized data to a binary file with proper file I/O handling.
    pub fn save_binary<P: AsRef<Path>>(self, path: P) -> SerializationResult<()> {
        let binary_data = self.to_binary()?;

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::new(file);
        writer.write_all(&binary_data)?;
        writer.flush()?;

        Ok(())
    }
}

impl Default for StructSerializer {
    /// Creates a default struct serializer
    ///
    /// Returns a new `StructSerializer` with default settings, equivalent to
    /// calling `StructSerializer::new()`.
    ///
    /// # Returns
    ///
    /// A new `StructSerializer` instance with empty field collection
    ///
    /// # Examples
    ///
    /// Creates a default serializer with empty field collection.
    fn default() -> Self {
        Self::new()
    }
}
