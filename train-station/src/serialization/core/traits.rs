//! Core traits for serialization operations and type conversion
//!
//! This module defines the fundamental traits that form the backbone of the serialization
//! system. These traits provide the interface for converting between Rust types and
//! serializable field values, enabling type-safe serialization and deserialization
//! with compile-time guarantees.
//!
//! # Purpose
//!
//! The traits module provides:
//! - **Type conversion traits**: `ToFieldValue` and `FromFieldValue` for automatic type handling
//! - **Struct serialization trait**: `StructSerializable` for complete struct serialization
//! - **Zero-dependency design**: No external dependencies, pure Rust implementation
//! - **Compile-time safety**: Type-safe conversions with compile-time validation
//! - **Extensible interface**: Easy to implement for custom types
//! - **Dual format support**: JSON and binary serialization through unified traits
//!
//! # Core Traits
//!
//! ## ToFieldValue
//! Converts Rust types to `FieldValue` for serialization. Provides automatic type
//! detection and conversion without requiring specialized serialization methods.
//!
//! ## FromFieldValue
//! Converts `FieldValue` back to concrete Rust types during deserialization.
//! Ensures type safety and provides detailed error reporting for conversion failures.
//!
//! ## StructSerializable
//! Complete trait for struct serialization and deserialization. Provides both
//! manual field-by-field control and convenient file I/O operations.
//!
//! # Implementation Patterns
//!
//! The traits are designed to work together seamlessly:
//!
//! - **Automatic conversion**: `ToFieldValue` enables generic serialization
//! - **Type safety**: `FromFieldValue` ensures correct type reconstruction
//! - **Complete workflow**: `StructSerializable` provides end-to-end serialization
//! - **Error handling**: All traits return `SerializationResult` for robust error handling
//!
//! # Thread Safety
//!
//! All traits are designed to be thread-safe. Implementations should ensure that
//! trait methods can be called concurrently without data races.

use super::error::SerializationResult;
use super::types::FieldValue;
use std::path::Path;

/// Trait for converting values to FieldValue for serialization
///
/// This trait provides the foundation for automatic type conversion during serialization.
/// It allows the `StructSerializer` to handle different types generically without
/// requiring specialized methods for each type. The conversion should be lossless
/// and reversible to ensure data integrity.
///
/// # Purpose
///
/// The `ToFieldValue` trait enables:
/// - **Generic serialization**: Single interface for all serializable types
/// - **Type safety**: Compile-time guarantees for conversion correctness
/// - **Automatic detection**: No manual type specification required
/// - **Extensibility**: Easy to implement for custom types
/// - **Performance**: Zero-cost abstractions for type conversion
///
/// # Required Methods
///
/// * `to_field_value` - Convert the value to a `FieldValue` for serialization
///
/// # Examples
///
/// The trait enables automatic type conversion for serialization with compile-time safety.
///
/// # Implementors
///
/// Common types that implement this trait include:
/// * **Primitive types**: `bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `usize`, `f32`, `f64`
/// * **String types**: `String`, `&str`
/// * **Collections**: `Vec<T>`, `HashMap<String, String>`, `Option<T>`
/// * **Custom types**: Any type implementing `ToFieldValue`
///
/// # Thread Safety
///
/// Implementations should be thread-safe and not modify any shared state during conversion.
pub trait ToFieldValue {
    /// Converts the value to a FieldValue for serialization
    ///
    /// This method should convert the implementing type into the appropriate
    /// `FieldValue` variant. The conversion should be lossless and reversible
    /// to ensure data integrity during serialization and deserialization cycles.
    ///
    /// # Returns
    ///
    /// A `FieldValue` representing the serialized form of the value
    ///
    /// # Examples
    ///
    /// Converts various types to their serializable representation with automatic type detection.
    fn to_field_value(&self) -> FieldValue;
}

/// Trait for converting FieldValue back to concrete types during deserialization
///
/// This trait provides the foundation for automatic type reconstruction during
/// deserialization. It allows the `StructDeserializer` to handle different types
/// generically while ensuring type safety and providing detailed error reporting
/// for conversion failures.
///
/// # Purpose
///
/// The `FromFieldValue` trait enables:
/// - **Generic deserialization**: Single interface for all deserializable types
/// - **Type safety**: Compile-time guarantees for conversion correctness
/// - **Error handling**: Detailed error reporting for conversion failures
/// - **Extensibility**: Easy to implement for custom types
/// - **Validation**: Runtime type checking and validation
///
/// # Required Methods
///
/// * `from_field_value` - Convert a `FieldValue` to the concrete type
///
/// # Examples
///
/// The trait enables automatic type reconstruction during deserialization with error handling.
///
/// # Implementors
///
/// Common types that implement this trait include:
/// * **Primitive types**: `bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `usize`, `f32`, `f64`
/// * **String types**: `String`
/// * **Collections**: `Vec<T>`, `HashMap<String, String>`, `Option<T>`
/// * **Custom types**: Any type implementing `FromFieldValue`
///
/// # Thread Safety
///
/// Implementations should be thread-safe and not modify any shared state during conversion.
pub trait FromFieldValue: Sized {
    /// Converts a FieldValue to the concrete type
    ///
    /// This method should convert the `FieldValue` into the implementing type.
    /// If the `FieldValue` variant doesn't match the expected type, an error
    /// should be returned with descriptive information including the field name
    /// for debugging purposes.
    ///
    /// # Arguments
    ///
    /// * `value` - The `FieldValue` to convert
    /// * `field_name` - Name of the field being converted (for error reporting)
    ///
    /// # Returns
    ///
    /// `Ok(Self)` on successful conversion
    /// `Err(SerializationError)` if the conversion fails
    ///
    /// # Examples
    ///
    /// Handles type conversion with proper error reporting and field name context.
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self>;
}

/// Trait for structs that can be serialized and deserialized using the struct builder pattern
///
/// This trait provides a complete serialization and deserialization interface for
/// structs, offering both manual field-by-field control and convenient file I/O
/// operations. It serves as a zero-dependency alternative to serde's derive macros,
/// allowing structs to be easily serialized and deserialized with minimal boilerplate.
///
/// # Purpose
///
/// The `StructSerializable` trait provides:
/// - **Complete serialization workflow**: From struct to multiple output formats
/// - **File I/O operations**: Direct save/load to JSON and binary files
/// - **String/binary conversion**: In-memory serialization and deserialization
/// - **Type safety**: Compile-time guarantees for struct serialization
/// - **Error handling**: Comprehensive error reporting for all operations
/// - **Zero dependencies**: Pure Rust implementation without external crates
///
/// # Required Methods
///
/// * `to_serializer` - Convert the struct to a `StructSerializer`
/// * `from_deserializer` - Create the struct from a `StructDeserializer`
///
/// # Provided Methods
///
/// * `save_json` - Save the struct to a JSON file
/// * `save_binary` - Save the struct to a binary file
/// * `load_json` - Load the struct from a JSON file
/// * `load_binary` - Load the struct from a binary file
/// * `to_json` - Convert the struct to a JSON string
/// * `to_binary` - Convert the struct to binary data
/// * `from_json` - Create the struct from a JSON string
/// * `from_binary` - Create the struct from binary data
///
/// # Examples
///
/// The trait provides comprehensive serialization capabilities with file I/O and format conversion.
///
/// # Implementors
///
/// Common types that implement this trait include:
/// * **Configuration structs**: Settings, config files, parameters
/// * **Data models**: Business logic entities, domain objects
/// * **Serializable objects**: Any struct requiring persistence
/// * **Custom types**: User-defined serializable structures
///
/// # Thread Safety
///
/// Implementations should be thread-safe. The trait methods should not modify
/// any shared state and should be safe to call concurrently.
pub trait StructSerializable: Sized {
    /// Converts the struct to a StructSerializer
    ///
    /// This method should create a `StructSerializer` and register all fields
    /// that should be serialized. The implementation should use the fluent
    /// interface to build the serializer with all relevant fields.
    ///
    /// # Returns
    ///
    /// A `StructSerializer` containing all the struct's serializable fields
    ///
    /// # Examples
    ///
    /// Creates a serializer with all struct fields using the fluent interface.
    fn to_serializer(&self) -> crate::serialization::core::StructSerializer;

    /// Creates the struct from a StructDeserializer
    ///
    /// This method should extract all fields from the deserializer and
    /// construct a new instance of the struct. It should handle field
    /// extraction errors gracefully and provide meaningful error messages.
    ///
    /// # Arguments
    ///
    /// * `deserializer` - The deserializer containing the struct's field data
    ///
    /// # Returns
    ///
    /// `Ok(Self)` on successful struct construction
    /// `Err(SerializationError)` if field extraction or construction fails
    ///
    /// # Examples
    ///
    /// Extracts fields and constructs the struct with proper error handling.
    fn from_deserializer(
        deserializer: &mut crate::serialization::core::StructDeserializer,
    ) -> SerializationResult<Self>;

    /// Saves the struct to a JSON file
    ///
    /// Serializes the struct to JSON format and writes it to the specified file path.
    /// The file is created if it doesn't exist, or truncated if it already exists.
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
    /// Saves struct data to a JSON file with proper file I/O handling.
    fn save_json<P: AsRef<Path>>(&self, path: P) -> SerializationResult<()> {
        self.to_serializer().save_json(path)
    }

    /// Saves the struct to a binary file
    ///
    /// Serializes the struct to binary format and writes it to the specified file path.
    /// The file is created if it doesn't exist, or truncated if it already exists.
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
    /// Saves struct data to a binary file with proper file I/O handling.
    fn save_binary<P: AsRef<Path>>(&self, path: P) -> SerializationResult<()> {
        self.to_serializer().save_binary(path)
    }

    /// Loads the struct from a JSON file
    ///
    /// Reads JSON data from the specified file path and deserializes it into
    /// a new instance of the struct.
    ///
    /// # Arguments
    ///
    /// * `path` - File path containing the JSON data to read
    ///
    /// # Returns
    ///
    /// `Ok(Self)` on successful deserialization
    /// `Err(SerializationError)` if file I/O or deserialization fails
    ///
    /// # Examples
    ///
    /// Loads struct data from a JSON file with proper error handling.
    fn load_json<P: AsRef<Path>>(path: P) -> SerializationResult<Self> {
        let mut deserializer = crate::serialization::core::StructDeserializer::load_json(path)?;
        Self::from_deserializer(&mut deserializer)
    }

    /// Loads the struct from a binary file
    ///
    /// Reads binary data from the specified file path and deserializes it into
    /// a new instance of the struct.
    ///
    /// # Arguments
    ///
    /// * `path` - File path containing the binary data to read
    ///
    /// # Returns
    ///
    /// `Ok(Self)` on successful deserialization
    /// `Err(SerializationError)` if file I/O or deserialization fails
    ///
    /// # Examples
    ///
    /// Loads struct data from a binary file with proper error handling.
    fn load_binary<P: AsRef<Path>>(path: P) -> SerializationResult<Self> {
        let mut deserializer = crate::serialization::core::StructDeserializer::load_binary(path)?;
        Self::from_deserializer(&mut deserializer)
    }

    /// Converts the struct to a JSON string
    ///
    /// Serializes the struct to a human-readable JSON string format.
    /// The JSON output maintains field order and includes proper escaping.
    ///
    /// # Returns
    ///
    /// `Ok(String)` containing the JSON representation on success
    /// `Err(SerializationError)` if serialization fails
    ///
    /// # Examples
    ///
    /// Converts struct to JSON string with proper escaping and formatting.
    fn to_json(&self) -> SerializationResult<String> {
        self.to_serializer().to_json()
    }

    /// Converts the struct to binary data
    ///
    /// Serializes the struct to a compact binary format optimized for
    /// efficient storage and transmission.
    ///
    /// # Returns
    ///
    /// `Ok(Vec<u8>)` containing the binary representation on success
    /// `Err(SerializationError)` if serialization fails
    ///
    /// # Examples
    ///
    /// Converts struct to binary data with efficient compact format.
    fn to_binary(&self) -> SerializationResult<Vec<u8>> {
        self.to_serializer().to_binary()
    }

    /// Creates the struct from a JSON string
    ///
    /// Deserializes a JSON string into a new instance of the struct.
    /// The JSON should contain all required fields in the expected format.
    ///
    /// # Arguments
    ///
    /// * `json` - JSON string containing the struct data
    ///
    /// # Returns
    ///
    /// `Ok(Self)` on successful deserialization
    /// `Err(SerializationError)` if JSON parsing or deserialization fails
    ///
    /// # Examples
    ///
    /// Creates struct from JSON string with proper parsing and validation.
    fn from_json(json: &str) -> SerializationResult<Self> {
        let mut deserializer = crate::serialization::core::StructDeserializer::from_json(json)?;
        Self::from_deserializer(&mut deserializer)
    }

    /// Creates the struct from binary data
    ///
    /// Deserializes binary data into a new instance of the struct.
    /// The binary data should contain all required fields in the expected format.
    ///
    /// # Arguments
    ///
    /// * `data` - Binary data containing the struct data
    ///
    /// # Returns
    ///
    /// `Ok(Self)` on successful deserialization
    /// `Err(SerializationError)` if binary parsing or deserialization fails
    ///
    /// # Examples
    ///
    /// Creates struct from binary data with proper parsing and validation.
    fn from_binary(data: &[u8]) -> SerializationResult<Self> {
        let mut deserializer = crate::serialization::core::StructDeserializer::from_binary(data)?;
        Self::from_deserializer(&mut deserializer)
    }
}
