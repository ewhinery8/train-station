//! Binary serialization and deserialization for Train Station objects
//!
//! This module provides efficient binary serialization using a custom format optimized
//! for performance and storage efficiency. The binary format is designed for production
//! deployment with minimal file sizes and maximum serialization speed.
//!
//! # Design Philosophy
//!
//! The binary format prioritizes:
//! - **Performance**: Optimized for fast serialization and deserialization
//! - **Storage Efficiency**: Minimal overhead and compact representation
//! - **Data Integrity**: Built-in validation and checksum support
//! - **Extensibility**: Type-length-value encoding for future compatibility
//! - **Cross-Platform**: Little-endian byte ordering for x86/x64 compatibility
//!
//! # Binary Format Specification
//!
//! The binary format uses a structured layout with fixed-size headers for fast seeking
//! and type-length-value (TLV) encoding for extensibility. All data is stored in
//! little-endian byte order for compatibility with x86/x64 architectures.
//!
//! ## File Format Layout
//!
//! ```text
//! [Magic Number: 4 bytes] - File format identifier (0x54535F42)
//! [Version: 4 bytes]      - Format version for compatibility
//! [Checksum: 4 bytes]     - CRC32 of remaining data (reserved)
//! [Data Length: 8 bytes]  - Total data length in bytes
//! [Object Type: 4 bytes]  - Object type identifier
//! [Object Data: N bytes]  - Serialized object data
//! ```
//!
//! ## Data Types Encoding
//!
//! - **Primitive Types**: Fixed-size encoding for maximum performance
//!   - `u8`, `i8`: 1 byte
//!   - `u16`, `i16`: 2 bytes (little-endian)
//!   - `u32`, `i32`, `f32`: 4 bytes (little-endian)
//!   - `u64`, `i64`: 8 bytes (little-endian)
//!   - `usize`: 8 bytes (little-endian, platform independent)
//!   - `bool`: 1 byte (0 = false, 1 = true)
//!
//! - **Complex Types**: Length-prefixed encoding
//!   - `String`: [length: 4 bytes][utf8 data: N bytes]
//!   - `Vec<T>`: [length: 8 bytes][elements: N * sizeof(T) bytes]
//!   - `Option<T>`: [present: 1 byte][data: sizeof(T) bytes if present]
//!
//! # Operations
//!
//! * `serialize_with_header()` - Serialize objects with format headers
//! * `deserialize_with_header()` - Deserialize objects with format validation
//! * `to_binary_internal()` - Convert struct serializers to binary format
//! * `from_binary_internal()` - Convert binary data to struct deserializers
//!
//! # Performance Characteristics
//!
//! * **Serialization Speed**: ~2-5x faster than JSON for large objects
//! * **File Size**: ~50-70% smaller than equivalent JSON representation
//! * **Memory Usage**: Streaming support for large objects
//! * **Seeking**: Fixed headers enable fast file navigation
//! * **Validation**: Minimal overhead for safety checks
//!
//! # Examples
//!
//! ```
//! use train_station::serialization::{StructSerializer, Format};
//!
//! // Create a simple struct serializer
//! let serializer = StructSerializer::new()
//!     .field("id", &42u32)
//!     .field("name", &"test".to_string());
//!
//! // Serialize to binary using the public API
//! let binary_data = serializer.to_binary().unwrap();
//! assert!(!binary_data.is_empty());
//! ```
//!
//! ```
//! use train_station::serialization::{StructSerializer, StructDeserializer, Format};
//!
//! // Create and serialize data
//! let serializer = StructSerializer::new()
//!     .field("value", &3.14f32)
//!     .field("active", &true);
//!
//! let binary_data = serializer.to_binary().unwrap();
//!
//! // Deserialize the data using the public API
//! let mut deserializer = StructDeserializer::from_binary(&binary_data).unwrap();
//! let value: f32 = deserializer.field("value").unwrap();
//! let active: bool = deserializer.field("active").unwrap();
//!
//! assert_eq!(value, 3.14);
//! assert_eq!(active, true);
//! ```
//!
//! ```
//! use train_station::{Tensor, serialization::StructSerializable};
//!
//! // Use binary format through the public Tensor API
//! let tensor = Tensor::new(vec![2, 3]);
//!
//! // Serialize tensor to binary format
//! let binary_data = tensor.to_binary().unwrap();
//! assert!(!binary_data.is_empty());
//!
//! // Deserialize tensor from binary format
//! let loaded_tensor = Tensor::from_binary(&binary_data).unwrap();
//! assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
//! ```
//!
//! # Safety and Validation
//!
//! The binary format includes comprehensive safety measures:
//! - **Bounds Checking**: All reads validate data availability
//! - **Size Limits**: Collections are limited to prevent memory exhaustion
//! - **Type Validation**: Object types are verified during deserialization
//! - **UTF-8 Validation**: String data is validated for proper encoding
//! - **Magic Number**: File format identification prevents corruption
//! - **Version Checking**: Format compatibility validation
//!
//! # Error Handling
//!
//! The binary serialization system provides detailed error information:
//! - **Format Errors**: Invalid binary structure or corrupted data
//! - **Version Mismatches**: Incompatible format versions
//! - **Type Errors**: Unexpected object types during deserialization
//! - **Validation Errors**: Data integrity and bounds violations
//!
//! # Components
//!
//! - `writer.rs` - Binary writer with endianness handling and byte counting
//! - `reader.rs` - Binary reader with validation and bounds checking
//! - `types.rs` - Object type identifiers and format constants
//! - `crc32.rs` - Data integrity validation implementation
//! - `header.rs` - File format validation and metadata handling
//! - `structs.rs` - Struct serialization and deserialization utilities
//!
//! # Integration with Core Serialization
//!
//! This module integrates with the core serialization system:
//! - **StructSerializer**: Convert to binary format for storage
//! - **StructDeserializer**: Load from binary format for reconstruction
//! - **FieldValue**: Efficient binary encoding of field data
//! - **ObjectType**: Type-safe object identification and validation

/// CRC32 checksum implementation for data integrity validation
///
/// Provides fast CRC32 calculation for binary data validation and corruption detection.
/// Used internally for ensuring data integrity during serialization and deserialization.
pub mod crc32;

/// Binary serialization error types and handling
///
/// Contains error types specific to binary format operations including format errors,
/// version mismatches, and data corruption detection.
pub mod error;

/// Binary format header management and validation
///
/// Handles the binary file format headers including magic numbers, version checking,
/// and object type validation for safe deserialization.
pub mod header;

/// Binary data reader with validation and bounds checking
///
/// Provides safe reading of binary data with automatic endianness handling,
/// bounds checking, and type validation for all primitive and complex types.
pub mod reader;

/// Struct serialization and deserialization utilities
///
/// Internal utilities for converting between StructSerializer/StructDeserializer
/// and binary format representation with efficient encoding.
pub mod structs;

/// Object type identifiers and format constants
///
/// Defines the object type system used for type-safe binary serialization
/// and format version constants for compatibility checking.
pub mod types;

/// Binary data writer with endianness handling and byte counting
///
/// Provides efficient writing of binary data with automatic endianness conversion,
/// byte counting, and optimized encoding for all supported data types.
pub mod writer;

/// Binary format error type for detailed error reporting
///
/// Represents errors that can occur during binary serialization and deserialization,
/// including format corruption, version mismatches, and validation failures.
pub use error::BinaryError;

/// High-level serialization and deserialization functions with header validation
///
/// * `serialize_with_header` - Serialize objects with complete binary format headers
/// * `deserialize_with_header` - Deserialize objects with format validation and type checking
pub use header::{deserialize_with_header, serialize_with_header};

/// Binary data reader for safe deserialization operations
///
/// Provides validated reading of binary data with bounds checking, endianness handling,
/// and automatic type conversion for all supported primitive and complex types.
pub use reader::BinaryReader;

/// Internal struct serialization utilities for binary format conversion
///
/// * `to_binary_internal` - Convert StructSerializer to binary format
/// * `from_binary_internal` - Convert binary data to StructDeserializer
///
/// These functions are used internally by the serialization system and should not
/// be called directly by user code. Use the public Serializable trait methods instead.
pub use structs::{from_binary_internal, to_binary_internal};

/// Object type identifier for type-safe binary serialization
///
/// Enum representing the different types of objects that can be serialized
/// in binary format, used for validation during deserialization.
pub use types::ObjectType;

/// Binary data writer for efficient serialization operations
///
/// Provides optimized writing of binary data with automatic endianness conversion,
/// byte counting, and efficient encoding for maximum serialization performance.
pub use writer::BinaryWriter;
