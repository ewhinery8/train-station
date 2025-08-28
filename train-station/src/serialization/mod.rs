//! Serialization and deserialization system for Train Station objects
//!
//! This module provides a robust, zero-dependency serialization framework that enables
//! persistent storage and model checkpointing for all Train Station objects. The system
//! supports both human-readable JSON format for debugging and efficient binary format
//! for production deployment.
//!
//! # Design Philosophy
//!
//! The serialization system follows Train Station's core principles:
//! - **Zero external dependencies**: Uses only the standard library
//! - **Maximum performance**: Optimized binary format for production use
//! - **Safety first**: Comprehensive validation and error handling
//! - **Future-proof**: Generic trait-based design for extensibility
//!
//! # Supported Formats
//!
//! ## JSON Format
//! Human-readable format suitable for:
//! - Model inspection and debugging
//! - Configuration files and version control
//! - Cross-language interoperability
//! - Development and testing workflows
//!
//! ## Binary Format
//! Optimized binary format suitable for:
//! - Production model deployment
//! - High-frequency checkpointing
//! - Network transmission and storage
//! - Memory and storage-constrained environments
//!
//! # Organization
//!
//! - `core/` - Core serialization types, traits, and functionality
//! - `binary/` - Binary format serialization and deserialization
//! - `json/` - JSON format serialization and deserialization
//!
//! # Examples
//!
//! Basic serialization usage:
//!
//! ```
//! use train_station::serialization::{StructSerializer, StructDeserializer, Format};
//! use std::collections::HashMap;
//!
//! // Create a simple data structure
//! let mut data = HashMap::new();
//! data.insert("name".to_string(), "test".to_string());
//! data.insert("value".to_string(), "42".to_string());
//!
//! // Serialize to JSON
//! let serializer = StructSerializer::new()
//!     .field("data", &data)
//!     .field("version", &1u32);
//! let json = serializer.to_json().unwrap();
//! assert!(json.contains("test"));
//!
//! // Deserialize from JSON
//! let mut deserializer = StructDeserializer::from_json(&json).unwrap();
//! let loaded_data: HashMap<String, String> = deserializer.field("data").unwrap();
//! let version: u32 = deserializer.field("version").unwrap();
//! assert_eq!(loaded_data.get("name").unwrap(), "test");
//! assert_eq!(version, 1);
//! ```
//!
//! # Thread Safety
//!
//! All serialization operations are thread-safe and can be performed concurrently
//! on different objects. The underlying file I/O operations use standard library
//! primitives that provide appropriate synchronization.
//!
//! # Error Handling
//!
//! The serialization system provides comprehensive error handling through the
//! `SerializationError` type, which includes detailed information about what
//! went wrong during serialization or deserialization. All operations return
//! `Result` types to ensure errors are handled explicitly.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub(crate) mod binary;
pub(crate) mod core;
pub(crate) mod json;

// Re-export core functionality for convenience
pub use core::{
    FieldValue, FromFieldValue, SerializationError, SerializationResult, StructDeserializer,
    StructSerializable, StructSerializer, ToFieldValue,
};

/// Serialization format options for saving and loading objects
///
/// This enum defines the available serialization formats supported by the
/// Train Station serialization system. Each format has specific use cases
/// and performance characteristics.
///
/// # Variants
///
/// * `Json` - Human-readable JSON format for debugging and inspection
/// * `Binary` - Efficient binary format for production deployment
///
/// # Examples
///
/// ```
/// use train_station::serialization::Format;
///
/// // Check format variants
/// let json_format = Format::Json;
/// let binary_format = Format::Binary;
/// assert_ne!(json_format, binary_format);
/// ```
///
/// # Performance Considerations
///
/// - **JSON**: Larger file sizes, slower serialization, human-readable
/// - **Binary**: Smaller file sizes, faster serialization, machine-optimized
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Human-readable JSON format
    ///
    /// Suitable for debugging, configuration files, and cross-language
    /// interoperability. Produces larger files but allows easy inspection
    /// and modification of serialized data.
    Json,
    /// Efficient binary format
    ///
    /// Optimized for production deployment with minimal file sizes and
    /// maximum serialization speed. Not human-readable but provides
    /// the best performance characteristics.
    Binary,
}

/// Core serialization trait for Train Station objects
///
/// This trait provides a unified interface for saving and loading objects in multiple
/// formats. All serializable objects must implement this trait to enable persistent
/// storage and model checkpointing. The trait includes both file-based and writer-based
/// operations for maximum flexibility.
///
/// # Required Methods
///
/// * `to_json` - Serialize the object to JSON format
/// * `from_json` - Deserialize an object from JSON format
/// * `to_binary` - Serialize the object to binary format
/// * `from_binary` - Deserialize an object from binary format
///
/// # Provided Methods
///
/// * `save` - Save the object to a file in the specified format
/// * `save_to_writer` - Save the object to a writer in the specified format
/// * `load` - Load an object from a file in the specified format
/// * `load_from_reader` - Load an object from a reader in the specified format
///
/// # Safety
///
/// Implementations must ensure that:
/// - Serialized data contains all necessary information for reconstruction
/// - Deserialization validates all input data thoroughly
/// - Memory safety is maintained during reconstruction
/// - No undefined behavior occurs with malformed input
///
/// # Examples
///
/// ```
/// use train_station::serialization::{Serializable, Format, SerializationResult};
///
/// // Example implementation for a simple struct
/// #[derive(Debug, PartialEq)]
/// struct SimpleData {
///     value: i32,
/// }
///
/// impl Serializable for SimpleData {
///     fn to_json(&self) -> SerializationResult<String> {
///         Ok(format!(r#"{{"value":{}}}"#, self.value))
///     }
///     
///     fn from_json(json: &str) -> SerializationResult<Self> {
///         // Simple parsing for demonstration
///         if let Some(start) = json.find("value\":") {
///             let value_str = &json[start + 7..];
///             if let Some(end) = value_str.find('}') {
///                 let value: i32 = value_str[..end].parse()
///                     .map_err(|_| "Invalid number format")?;
///                 return Ok(SimpleData { value });
///             }
///         }
///         Err("Invalid JSON format".into())
///     }
///     
///     fn to_binary(&self) -> SerializationResult<Vec<u8>> {
///         Ok(self.value.to_le_bytes().to_vec())
///     }
///     
///     fn from_binary(data: &[u8]) -> SerializationResult<Self> {
///         if data.len() != 4 {
///             return Err("Invalid binary data length".into());
///         }
///         let value = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
///         Ok(SimpleData { value })
///     }
/// }
///
/// // Usage example
/// let data = SimpleData { value: 42 };
/// let json = data.to_json().unwrap();
/// let loaded = SimpleData::from_json(&json).unwrap();
/// assert_eq!(data, loaded);
/// ```
///
/// # Implementors
///
/// Common types that implement this trait include:
/// * `Tensor` - For serializing tensor data and metadata
/// * `AdamConfig` - For serializing optimizer configuration
/// * `SerializableAdam` - For serializing optimizer state
pub trait Serializable: Sized {
    /// Save the object to a file in the specified format
    ///
    /// This method creates or overwrites a file at the specified path and writes
    /// the serialized object data in the requested format. The file is created
    /// with write permissions and truncated if it already exists.
    ///
    /// # Arguments
    ///
    /// * `path` - File path where the object should be saved
    /// * `format` - Serialization format (JSON or Binary)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::serialization::{Serializable, Format, SerializationResult};
    /// use std::io::Write;
    ///
    /// // Simple example struct
    /// struct TestData { value: i32 }
    /// impl Serializable for TestData {
    ///     fn to_json(&self) -> SerializationResult<String> {
    ///         Ok(format!(r#"{{"value":{}}}"#, self.value))
    ///     }
    ///     fn from_json(json: &str) -> SerializationResult<Self> {
    ///         Ok(TestData { value: 42 }) // Simplified for example
    ///     }
    ///     fn to_binary(&self) -> SerializationResult<Vec<u8>> {
    ///         Ok(self.value.to_le_bytes().to_vec())
    ///     }
    ///     fn from_binary(_data: &[u8]) -> SerializationResult<Self> {
    ///         Ok(TestData { value: 42 }) // Simplified for example
    ///     }
    /// }
    ///
    /// let data = TestData { value: 42 };
    ///
    /// // Save to temporary file (cleanup handled by temp directory)
    /// let temp_dir = std::env::temp_dir();
    /// let json_path = temp_dir.join("test_data.json");
    /// data.save(&json_path, Format::Json).unwrap();
    ///
    /// // Verify file was created
    /// assert!(json_path.exists());
    ///
    /// // Clean up
    /// std::fs::remove_file(&json_path).ok();
    /// ```
    fn save<P: AsRef<Path>>(&self, path: P, format: Format) -> SerializationResult<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::new(file);
        self.save_to_writer(&mut writer, format)
    }

    /// Save the object to a writer in the specified format
    ///
    /// This method serializes the object and writes the data to the provided writer.
    /// The writer is flushed after writing to ensure all data is written. This method
    /// is useful for streaming serialization or writing to non-file destinations.
    ///
    /// # Arguments
    ///
    /// * `writer` - Writer to output serialized data
    /// * `format` - Serialization format (JSON or Binary)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    fn save_to_writer<W: Write>(&self, writer: &mut W, format: Format) -> SerializationResult<()> {
        match format {
            Format::Json => {
                let json_data = self.to_json()?;
                writer.write_all(json_data.as_bytes())?;
            }
            Format::Binary => {
                let binary_data = self.to_binary()?;
                writer.write_all(&binary_data)?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    /// Load an object from a file in the specified format
    ///
    /// This method reads the entire file content and deserializes it into an object
    /// of the implementing type. The file must exist and contain valid serialized
    /// data in the specified format.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to read from
    /// * `format` - Expected serialization format
    ///
    /// # Returns
    ///
    /// The deserialized object on success, or `SerializationError` on failure
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::serialization::{Serializable, Format, SerializationResult};
    /// use std::io::Write;
    ///
    /// // Simple example struct
    /// #[derive(Debug, PartialEq)]
    /// struct TestData { value: i32 }
    /// impl Serializable for TestData {
    ///     fn to_json(&self) -> SerializationResult<String> {
    ///         Ok(format!(r#"{{"value":{}}}"#, self.value))
    ///     }
    ///     fn from_json(json: &str) -> SerializationResult<Self> {
    ///         // Simple parsing for demonstration
    ///         if json.contains("42") {
    ///             Ok(TestData { value: 42 })
    ///         } else {
    ///             Ok(TestData { value: 0 })
    ///         }
    ///     }
    ///     fn to_binary(&self) -> SerializationResult<Vec<u8>> {
    ///         Ok(self.value.to_le_bytes().to_vec())
    ///     }
    ///     fn from_binary(data: &[u8]) -> SerializationResult<Self> {
    ///         if data.len() >= 4 {
    ///             let value = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    ///             Ok(TestData { value })
    ///         } else {
    ///             Ok(TestData { value: 0 })
    ///         }
    ///     }
    /// }
    ///
    /// let original = TestData { value: 42 };
    ///
    /// // Save and load from temporary file
    /// let temp_dir = std::env::temp_dir();
    /// let json_path = temp_dir.join("test_load.json");
    /// original.save(&json_path, Format::Json).unwrap();
    ///
    /// let loaded = TestData::load(&json_path, Format::Json).unwrap();
    /// assert_eq!(original, loaded);
    ///
    /// // Clean up
    /// std::fs::remove_file(&json_path).ok();
    /// ```
    fn load<P: AsRef<Path>>(path: P, format: Format) -> SerializationResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::load_from_reader(&mut reader, format)
    }

    /// Load an object from a reader in the specified format
    ///
    /// This method reads all available data from the provided reader and deserializes
    /// it into an object of the implementing type. The reader must contain complete
    /// serialized data in the specified format.
    ///
    /// # Arguments
    ///
    /// * `reader` - Reader containing serialized data
    /// * `format` - Expected serialization format
    ///
    /// # Returns
    ///
    /// The deserialized object on success, or `SerializationError` on failure
    fn load_from_reader<R: Read>(reader: &mut R, format: Format) -> SerializationResult<Self> {
        match format {
            Format::Json => {
                let mut json_data = String::new();
                reader.read_to_string(&mut json_data)?;
                Self::from_json(&json_data)
            }
            Format::Binary => {
                let mut binary_data = Vec::new();
                reader.read_to_end(&mut binary_data)?;
                Self::from_binary(&binary_data)
            }
        }
    }

    /// Serialize the object to JSON format
    ///
    /// This method converts the object into a human-readable JSON string representation.
    /// The JSON format is suitable for debugging, configuration files, and cross-language
    /// interoperability.
    ///
    /// # Returns
    ///
    /// JSON string representation of the object on success, or `SerializationError` on failure
    fn to_json(&self) -> SerializationResult<String>;

    /// Deserialize an object from JSON format
    ///
    /// This method parses a JSON string and reconstructs an object of the implementing
    /// type. The JSON must contain all necessary fields and data in the expected format.
    ///
    /// # Arguments
    ///
    /// * `json` - JSON string containing serialized object
    ///
    /// # Returns
    ///
    /// The deserialized object on success, or `SerializationError` on failure
    fn from_json(json: &str) -> SerializationResult<Self>;

    /// Serialize the object to binary format
    ///
    /// This method converts the object into a compact binary representation optimized
    /// for storage and transmission. The binary format provides maximum performance
    /// and minimal file sizes.
    ///
    /// # Returns
    ///
    /// Binary representation of the object on success, or `SerializationError` on failure
    fn to_binary(&self) -> SerializationResult<Vec<u8>>;

    /// Deserialize an object from binary format
    ///
    /// This method parses binary data and reconstructs an object of the implementing
    /// type. The binary data must contain complete serialized information in the
    /// expected format.
    ///
    /// # Arguments
    ///
    /// * `data` - Binary data containing serialized object
    ///
    /// # Returns
    ///
    /// The deserialized object on success, or `SerializationError` on failure
    fn from_binary(data: &[u8]) -> SerializationResult<Self>;
}

/// Utility functions for common serialization tasks
///
/// This module provides helper functions for format detection, file extension
/// management, and size estimation for serialization operations. These functions
/// are used internally by the serialization system to support file operations
/// and provide estimates for memory allocation.
///
/// # Purpose
///
/// The utilities in this module handle:
/// - File extension mapping for different serialization formats
/// - Automatic format detection based on file paths
/// - Size estimation for binary serialization planning
/// - Common helper functions used across the serialization system
pub(crate) mod utils {
    #[cfg(test)]
    use super::Format;
    #[cfg(test)]
    use std::path::Path;

    /// Get the appropriate file extension for a format
    ///
    /// Returns the standard file extension associated with each serialization format.
    /// This is useful for automatically determining file extensions when saving
    /// or for format detection based on file paths.
    ///
    /// # Arguments
    ///
    /// * `format` - The serialization format
    ///
    /// # Returns
    ///
    /// The file extension as a string slice
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::serialization::Format;
    ///
    /// // This function is internal, but demonstrates the concept
    /// fn format_extension(format: Format) -> &'static str {
    ///     match format {
    ///         Format::Json => "json",
    ///         Format::Binary => "bin",
    ///     }
    /// }
    ///
    /// assert_eq!(format_extension(Format::Json), "json");
    /// assert_eq!(format_extension(Format::Binary), "bin");
    /// ```
    #[cfg(test)]
    pub(crate) fn format_extension(format: Format) -> &'static str {
        match format {
            Format::Json => "json",
            Format::Binary => "bin",
        }
    }

    /// Detect format from file extension
    ///
    /// Attempts to determine the serialization format based on the file extension.
    /// Supports case-insensitive extension matching for common format extensions.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to analyze
    ///
    /// # Returns
    ///
    /// `Some(Format)` if the extension is recognized, `None` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::serialization::Format;
    /// use std::path::Path;
    ///
    /// // This function is internal, but demonstrates the concept
    /// fn detect_format<P: AsRef<Path>>(path: P) -> Option<Format> {
    ///     path.as_ref()
    ///         .extension()
    ///         .and_then(|ext| ext.to_str())
    ///         .and_then(|ext| match ext.to_lowercase().as_str() {
    ///             "json" => Some(Format::Json),
    ///             "bin" => Some(Format::Binary),
    ///             _ => None,
    ///         })
    /// }
    ///
    /// assert_eq!(detect_format("model.json"), Some(Format::Json));
    /// assert_eq!(detect_format("model.JSON"), Some(Format::Json));
    /// assert_eq!(detect_format("model.bin"), Some(Format::Binary));
    /// assert_eq!(detect_format("model.txt"), None);
    /// ```
    #[cfg(test)]
    pub(crate) fn detect_format<P: AsRef<Path>>(path: P) -> Option<Format> {
        path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "json" => Some(Format::Json),
                "bin" => Some(Format::Binary),
                _ => None,
            })
    }

    /// Estimate serialized size for binary format
    ///
    /// Provides a rough estimate of the binary serialized size based on the number
    /// of tensors, total elements, and metadata fields. This is useful for memory
    /// allocation and storage planning.
    ///
    /// # Arguments
    ///
    /// * `tensor_count` - Number of tensors to be serialized
    /// * `total_elements` - Total number of elements across all tensors
    /// * `metadata_fields` - Number of metadata fields per tensor
    ///
    /// # Returns
    ///
    /// Estimated size in bytes for the binary serialization
    ///
    /// # Examples
    ///
    /// ```
    /// // This function is internal, but demonstrates the concept
    /// fn estimate_binary_size(
    ///     tensor_count: usize,
    ///     total_elements: usize,
    ///     metadata_fields: usize,
    /// ) -> usize {
    ///     // Header + magic number + version
    ///     let header_size = 16;
    ///     // Tensor data (f32 per element)
    ///     let data_size = total_elements * 4;
    ///     // Shape information (dimensions, strides, metadata)
    ///     let shape_size = tensor_count * (metadata_fields * 8 + 64);
    ///     header_size + data_size + shape_size
    /// }
    ///
    /// let estimated_size = estimate_binary_size(3, 1000, 5);
    /// assert!(estimated_size > 4000); // At least data size
    /// ```
    #[cfg(test)]
    pub(crate) fn estimate_binary_size(
        tensor_count: usize,
        total_elements: usize,
        metadata_fields: usize,
    ) -> usize {
        // Header + magic number + version
        let header_size = 16;

        // Tensor data (f32 per element)
        let data_size = total_elements * 4;

        // Shape information (dimensions, strides, metadata)
        let shape_size = tensor_count * (metadata_fields * 8 + 64);

        header_size + data_size + shape_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(utils::detect_format("model.json"), Some(Format::Json));
        assert_eq!(utils::detect_format("model.JSON"), Some(Format::Json)); // Case insensitive
        assert_eq!(utils::detect_format("model.bin"), Some(Format::Binary));
        assert_eq!(utils::detect_format("model.BIN"), Some(Format::Binary)); // Case insensitive
        assert_eq!(utils::detect_format("model.txt"), None);
        assert_eq!(utils::detect_format("model"), None);
        assert_eq!(utils::detect_format(""), None);
    }

    #[test]
    fn test_format_extensions() {
        assert_eq!(utils::format_extension(Format::Json), "json");
        assert_eq!(utils::format_extension(Format::Binary), "bin");
    }

    #[test]
    fn test_binary_size_estimation() {
        // Single tensor with 1000 elements
        let estimated = utils::estimate_binary_size(1, 1000, 5);
        assert!(estimated >= 4000); // At least data size (1000 * 4 bytes)
        assert!(estimated <= 5000); // Reasonable metadata overhead

        // Multiple tensors
        let estimated_multi = utils::estimate_binary_size(3, 3000, 5);
        assert!(estimated_multi >= 12000); // At least data size (3000 * 4 bytes)
        assert!(estimated_multi > estimated * 2); // Should be larger than single tensor
    }
}
