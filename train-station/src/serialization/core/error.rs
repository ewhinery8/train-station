//! Core error types for serialization operations
//!
//! This module provides the fundamental error types and result aliases used throughout
//! the serialization system. It defines the main `SerializationError` enum that covers
//! all possible failure modes during serialization and deserialization operations.
//!
//! # Purpose
//!
//! The error system provides:
//! - Comprehensive error coverage for all serialization operations
//! - Detailed context for debugging and error recovery
//! - Format-specific error delegation to specialized modules
//! - Standard error trait implementations for integration
//! - Convenient result type aliases for cleaner code
//!
//! # Error Categories
//!
//! The `SerializationError` enum covers several categories of errors:
//! - **I/O Errors**: File operations, network issues, stream handling
//! - **Format Errors**: JSON and binary format-specific parsing errors
//! - **Validation Errors**: Data validation failures during deserialization
//! - **Shape Errors**: Tensor dimension mismatches
//! - **Memory Errors**: Allocation failures and resource constraints
//! - **Custom Errors**: Generic errors with custom messages
//!
//! # Usage Patterns
//!
//! The error system provides comprehensive error handling for all serialization
//! operations with detailed context for debugging and error recovery.
//!
//! # Error Handling
//!
//! All serialization operations return `SerializationResult<T>` which provides:
//! - Detailed error context for debugging
//! - Source error chaining for root cause analysis
//! - Human-readable error messages
//! - Structured error data for programmatic handling
//!
//! # Thread Safety
//!
//! All error types are thread-safe and can be shared across threads without
//! additional synchronization.

use std::fmt;
use std::io;

/// Result type for serialization operations
///
/// This type alias provides a convenient way to express the result of any
/// serialization or deserialization operation. It uses the standard `Result`
/// type with `SerializationError` as the error variant.
///

pub type SerializationResult<T> = Result<T, SerializationError>;

/// Main error type for serialization operations
///
/// This enum covers all possible failure modes during serialization and deserialization,
/// providing detailed context for debugging and error recovery. Format-specific errors
/// are delegated to their respective modules while maintaining a unified error interface.
///
/// # Error Variants
///
/// The enum provides comprehensive coverage of serialization failure modes:
/// - **I/O errors**: File operations, network issues, stream handling
/// - **Format errors**: JSON and binary format-specific parsing errors
/// - **Validation errors**: Data validation failures during deserialization
/// - **Shape errors**: Tensor dimension mismatches
/// - **Memory errors**: Allocation failures and resource constraints
/// - **Custom errors**: Generic errors with custom messages
///
/// # Error Context
///
/// Each error variant provides detailed context to aid in debugging:
/// - Field names and validation messages for validation errors
/// - Expected vs actual tensor dimensions for shape errors
/// - Memory allocation details for allocation failures
/// - Format-specific error information for parsing errors
///
/// # Thread Safety
///
/// This type is thread-safe and can be shared across threads without
/// additional synchronization.
///

#[derive(Debug)]
pub enum SerializationError {
    /// I/O errors during file operations, network issues, or stream handling
    ///
    /// This variant wraps standard I/O errors that occur during serialization
    /// operations, such as file reading/writing, network communication, or
    /// stream processing. The underlying I/O error is preserved for detailed
    /// error analysis.
    ///
    /// # Common Causes
    ///
    /// - File not found or permission denied
    /// - Disk space exhaustion
    /// - Network connectivity issues
    /// - Stream corruption or interruption
    /// - Device I/O failures
    Io(io::Error),

    /// JSON-specific format and parsing errors
    ///
    /// This variant delegates to the JSON module's error type for format-specific
    /// JSON parsing and validation errors. It provides detailed information about
    /// JSON syntax errors, structural issues, and parsing failures.
    ///
    /// # Common Causes
    ///
    /// - Invalid JSON syntax
    /// - Malformed JSON structure
    /// - Encoding issues
    /// - Unexpected token types
    /// - JSON depth limits exceeded
    Json(crate::serialization::json::JsonError),

    /// Binary-specific format and parsing errors
    ///
    /// This variant delegates to the binary module's error type for format-specific
    /// binary parsing and validation errors. It provides detailed information about
    /// binary format issues, corruption, and parsing failures.
    ///
    /// # Common Causes
    ///
    /// - Invalid binary format
    /// - Corrupted binary data
    /// - Version mismatches
    /// - Magic number validation failures
    /// - Truncated binary streams
    Binary(crate::serialization::binary::BinaryError),

    /// Data validation failed during deserialization
    ///
    /// This variant indicates that data validation failed during the deserialization
    /// process. It provides the specific field name and a human-readable message
    /// explaining why the validation failed.
    ///
    /// # Fields
    ///
    /// * `field` - Name of the field that failed validation
    /// * `message` - Human-readable message explaining why validation failed
    ///
    /// # Common Causes
    ///
    /// - Invalid field values
    /// - Missing required fields
    /// - Type conversion failures
    /// - Constraint violations
    /// - Business logic validation failures
    ValidationFailed {
        /// Name of the field that failed validation
        field: String,
        /// Human-readable message explaining why validation failed
        message: String,
    },

    /// Tensor shape or size mismatch
    ///
    /// This variant indicates that a tensor's dimensions don't match the expected
    /// shape during deserialization or validation. It provides both the expected
    /// and actual dimensions for debugging.
    ///
    /// # Fields
    ///
    /// * `expected_dims` - Expected tensor dimensions
    /// * `found_dims` - Actual tensor dimensions found
    ///
    /// # Common Causes
    ///
    /// - Incorrect tensor serialization
    /// - Version incompatibilities
    /// - Manual data corruption
    /// - Serialization format changes
    /// - Dimension calculation errors
    ShapeMismatch {
        /// Expected tensor dimensions
        expected_dims: Vec<usize>,
        /// Actual tensor dimensions found
        found_dims: Vec<usize>,
    },

    /// Memory allocation failed
    ///
    /// This variant indicates that a memory allocation request failed during
    /// serialization or deserialization. It provides details about the requested
    /// size and available memory (if known).
    ///
    /// # Fields
    ///
    /// * `requested_size` - Number of bytes that were requested
    /// * `available_memory` - Number of bytes available (if known)
    ///
    /// # Common Causes
    ///
    /// - Insufficient system memory
    /// - Memory fragmentation
    /// - Resource limits exceeded
    /// - Large tensor allocations
    /// - Memory pool exhaustion
    AllocationFailed {
        /// Number of bytes that were requested
        requested_size: usize,
        /// Number of bytes available (if known)
        available_memory: Option<usize>,
    },

    /// Generic error with custom message
    ///
    /// This variant provides a catch-all for generic serialization errors that
    /// don't fit into the other specific categories. It allows for custom error
    /// messages while maintaining the unified error interface.
    ///
    /// # Common Uses
    ///
    /// - Custom validation logic
    /// - Business rule violations
    /// - Unsupported operations
    /// - Configuration errors
    /// - Third-party integration issues
    Custom(String),

    // Compatibility variants - these will be removed after refactoring
    /// @deprecated Use Json(JsonError::Format) instead
    ///
    /// This variant is deprecated and will be removed in a future version.
    /// Use `Json(JsonError::Format)` instead for JSON format errors.
    ///
    /// # Migration
    ///
    /// Replace usage with the new format-specific error variants.
    JsonFormat {
        message: String,
        line: Option<usize>,
        column: Option<usize>,
    },

    /// @deprecated Use Binary(BinaryError::Format) instead
    ///
    /// This variant is deprecated and will be removed in a future version.
    /// Use `Binary(BinaryError::Format)` instead for binary format errors.
    ///
    /// # Migration
    ///
    /// Replace usage with the new format-specific error variants.
    BinaryFormat {
        message: String,
        position: Option<usize>,
    },

    /// @deprecated Use Binary(BinaryError::VersionMismatch) instead
    ///
    /// This variant is deprecated and will be removed in a future version.
    /// Use `Binary(BinaryError::VersionMismatch)` instead for version mismatch errors.
    ///
    /// # Migration
    ///
    /// Replace usage with the new format-specific error variants.
    VersionMismatch { expected: u32, found: u32 },

    /// @deprecated Use Binary(BinaryError::InvalidMagic) instead
    ///
    /// This variant is deprecated and will be removed in a future version.
    /// Use `Binary(BinaryError::InvalidMagic)` instead for invalid magic number errors.
    ///
    /// # Migration
    ///
    /// Replace usage with the new format-specific error variants.
    InvalidMagic { expected: u32, found: u32 },
}

impl fmt::Display for SerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerializationError::Io(err) => {
                write!(f, "I/O error during serialization: {}", err)
            }
            SerializationError::Json(err) => {
                write!(f, "JSON error: {}", err)
            }
            SerializationError::Binary(err) => {
                write!(f, "Binary error: {}", err)
            }
            SerializationError::ValidationFailed { field, message } => {
                write!(f, "Validation failed for field '{}': {}", field, message)
            }
            SerializationError::ShapeMismatch {
                expected_dims,
                found_dims,
            } => {
                write!(
                    f,
                    "Shape mismatch: expected {:?}, found {:?}",
                    expected_dims, found_dims
                )
            }
            SerializationError::AllocationFailed {
                requested_size,
                available_memory,
            } => match available_memory {
                Some(available) => {
                    write!(
                        f,
                        "Memory allocation failed: requested {} bytes, {} bytes available",
                        requested_size, available
                    )
                }
                None => {
                    write!(
                        f,
                        "Memory allocation failed: requested {} bytes",
                        requested_size
                    )
                }
            },
            SerializationError::Custom(message) => {
                write!(f, "Serialization error: {}", message)
            }
            // Compatibility variants
            SerializationError::JsonFormat {
                message,
                line,
                column,
            } => match (line, column) {
                (Some(l), Some(c)) => {
                    write!(
                        f,
                        "JSON format error at line {}, column {}: {}",
                        l, c, message
                    )
                }
                (Some(l), None) => {
                    write!(f, "JSON format error at line {}: {}", l, message)
                }
                _ => {
                    write!(f, "JSON format error: {}", message)
                }
            },
            SerializationError::BinaryFormat { message, position } => match position {
                Some(pos) => {
                    write!(f, "Binary format error at position {}: {}", pos, message)
                }
                None => {
                    write!(f, "Binary format error: {}", message)
                }
            },
            SerializationError::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "Version mismatch: expected {}, found {}",
                    expected, found
                )
            }
            SerializationError::InvalidMagic { expected, found } => {
                write!(
                    f,
                    "Invalid magic number: expected 0x{:08X}, found 0x{:08X}",
                    expected, found
                )
            }
        }
    }
}

impl std::error::Error for SerializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SerializationError::Io(err) => Some(err),
            SerializationError::Json(err) => Some(err),
            SerializationError::Binary(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for SerializationError {
    /// Convert an I/O error to a SerializationError
    ///
    /// This implementation allows I/O errors to be automatically converted
    /// to SerializationError instances, enabling seamless error propagation
    /// in serialization operations.
    ///

    fn from(err: io::Error) -> Self {
        SerializationError::Io(err)
    }
}

impl From<crate::serialization::json::JsonError> for SerializationError {
    /// Convert a JSON error to a SerializationError
    ///
    /// This implementation allows JSON-specific errors to be automatically
    /// converted to SerializationError instances, maintaining error context
    /// while providing a unified error interface.
    ///

    fn from(err: crate::serialization::json::JsonError) -> Self {
        SerializationError::Json(err)
    }
}

impl From<crate::serialization::binary::BinaryError> for SerializationError {
    /// Convert a binary error to a SerializationError
    ///
    /// This implementation allows binary-specific errors to be automatically
    /// converted to SerializationError instances, maintaining error context
    /// while providing a unified error interface.
    ///

    fn from(err: crate::serialization::binary::BinaryError) -> Self {
        SerializationError::Binary(err)
    }
}

impl From<String> for SerializationError {
    /// Convert a String to a SerializationError
    ///
    /// This implementation allows String values to be automatically converted
    /// to SerializationError::Custom instances, providing a convenient way
    /// to create custom error messages.
    ///

    fn from(message: String) -> Self {
        SerializationError::Custom(message)
    }
}

impl From<&str> for SerializationError {
    /// Convert a string slice to a SerializationError
    ///
    /// This implementation allows string slices to be automatically converted
    /// to SerializationError::Custom instances, providing a convenient way
    /// to create custom error messages from string literals.
    ///

    fn from(message: &str) -> Self {
        SerializationError::Custom(message.to_string())
    }
}

impl SerializationError {
    /// Create a JSON format error
    ///
    /// This method provides a convenient way to create JSON format errors
    /// with optional line and column information for debugging.
    ///
    /// # Arguments
    ///
    /// * `message` - Human-readable error message describing the JSON format problem
    /// * `line` - Line number where the error occurred (1-based, if available)
    /// * `column` - Column number where the error occurred (1-based, if available)
    ///
    /// # Returns
    ///
    /// A SerializationError::Json variant with the specified format error
    ///

    pub fn json_format(message: String, line: Option<usize>, column: Option<usize>) -> Self {
        SerializationError::Json(crate::serialization::json::JsonError::Format {
            message,
            line,
            column,
        })
    }

    /// Create a binary format error
    ///
    /// This method provides a convenient way to create binary format errors
    /// with optional position information for debugging.
    ///
    /// # Arguments
    ///
    /// * `message` - Human-readable error message describing the binary format problem
    /// * `position` - Byte position where the error occurred (if available)
    ///
    /// # Returns
    ///
    /// A SerializationError::Binary variant with the specified format error
    ///

    pub fn binary_format(message: String, position: Option<usize>) -> Self {
        SerializationError::Binary(crate::serialization::binary::BinaryError::Format {
            message,
            position,
        })
    }

    /// Create a binary version mismatch error
    ///
    /// This method provides a convenient way to create binary version mismatch
    /// errors when the expected and actual format versions don't match.
    ///
    /// # Arguments
    ///
    /// * `expected` - Expected format version that the deserializer supports
    /// * `found` - Actual format version found in the binary data
    ///
    /// # Returns
    ///
    /// A SerializationError::Binary variant with the specified version mismatch error
    ///

    pub fn binary_version_mismatch(expected: u32, found: u32) -> Self {
        SerializationError::Binary(crate::serialization::binary::BinaryError::VersionMismatch {
            expected,
            found,
        })
    }

    /// Create a binary invalid magic error
    ///
    /// This method provides a convenient way to create binary invalid magic
    /// errors when the binary data doesn't start with the expected magic number.
    ///
    /// # Arguments
    ///
    /// * `expected` - Expected magic number value for the binary format
    /// * `found` - Actual magic number found at the beginning of the data
    ///
    /// # Returns
    ///
    /// A SerializationError::Binary variant with the specified invalid magic error
    ///

    pub fn binary_invalid_magic(expected: u32, found: u32) -> Self {
        SerializationError::Binary(crate::serialization::binary::BinaryError::InvalidMagic {
            expected,
            found,
        })
    }
}
