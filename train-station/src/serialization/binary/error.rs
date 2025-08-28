//! Binary-specific error types for serialization and deserialization
//!
//! This module provides comprehensive error types for binary serialization and parsing
//! operations. These error types are designed to provide detailed information about
//! what went wrong during binary data processing, including position information
//! and specific error contexts.
//!
//! # Error Types
//!
//! The module defines a single `BinaryError` enum with variants for different types
//! of binary processing errors:
//!
//! - **Format**: Invalid binary format or corrupted data with optional position information
//! - **VersionMismatch**: Version incompatibility between serializer and deserializer
//! - **InvalidMagic**: Invalid magic number in binary format header
//! - **UnsupportedFeature**: Attempted to use a feature not supported in the current version
//!
//! # Usage
//!
//! These error types are used internally by the binary serialization system to
//! provide detailed error information when binary data cannot be processed correctly.
//! They implement standard error traits for integration with Rust's error handling
//! ecosystem.
//!
//! # Thread Safety
//!
//! All error types in this module are thread-safe and can be safely shared between
//! threads.

use std::fmt;

/// Binary-specific error type for serialization and deserialization operations
///
/// This enum provides comprehensive error information for binary data processing
/// operations. Each variant contains specific details about what went wrong,
/// enabling precise error diagnosis and debugging.
///
/// # Variants
///
/// * `Format` - Invalid binary format or corrupted data with optional position information
/// * `VersionMismatch` - Version incompatibility between serializer and deserializer
/// * `InvalidMagic` - Invalid magic number in binary format header
/// * `UnsupportedFeature` - Attempted to use a feature not supported in the current version
///
/// # Error Handling
///
/// This error type implements standard Rust error traits (`Display`, `Error`, `Debug`)
/// for seamless integration with Rust's error handling ecosystem. Error messages
/// are designed to be human-readable and provide actionable debugging information.
///
/// # Thread Safety
///
/// This type is thread-safe and can be safely shared between threads.
#[derive(Debug)]
pub enum BinaryError {
    /// Invalid binary format or corrupted data with optional position information
    ///
    /// This variant indicates that the binary data format is invalid or corrupted.
    /// It provides a human-readable error message and optionally the byte position
    /// where the error occurred, enabling precise debugging of binary format issues.
    ///
    /// # Fields
    ///
    /// * `message` - Human-readable error message describing the binary format problem
    /// * `position` - Byte position where the error occurred (if available)
    ///
    /// # Common Causes
    ///
    /// - Corrupted binary data during transmission or storage
    /// - Incorrect binary format specification
    /// - Truncated binary data
    /// - Invalid field values or type mismatches
    Format {
        /// Human-readable error message describing the binary format problem
        message: String,
        /// Byte position where the error occurred (if available)
        position: Option<usize>,
    },

    /// Version mismatch between serializer and deserializer
    ///
    /// This variant indicates that the binary data was written with a different
    /// format version than what the deserializer expects. This typically occurs
    /// when trying to read data written by a newer or older version of the
    /// serialization system.
    ///
    /// # Fields
    ///
    /// * `expected` - Expected format version that the deserializer supports
    /// * `found` - Actual format version found in the binary data
    ///
    /// # Common Causes
    ///
    /// - Reading data written by a newer version of the library
    /// - Reading data written by an older version of the library
    /// - Format version incompatibility between different systems
    VersionMismatch {
        /// Expected format version that the deserializer supports
        expected: u32,
        /// Actual format version found in the binary data
        found: u32,
    },

    /// Invalid magic number in binary format header
    ///
    /// This variant indicates that the binary data does not start with the expected
    /// magic number. Magic numbers are used to identify binary format files and
    /// ensure that the data is in the expected format.
    ///
    /// # Fields
    ///
    /// * `expected` - Expected magic number value for the binary format
    /// * `found` - Actual magic number found at the beginning of the data
    ///
    /// # Common Causes
    ///
    /// - Reading a file that is not in the expected binary format
    /// - Corrupted binary data header
    /// - Incorrect file type or format
    /// - Data written by a different serialization system
    InvalidMagic {
        /// Expected magic number value for the binary format
        expected: u32,
        /// Actual magic number found at the beginning of the data
        found: u32,
    },

    /// Unsupported feature or format version
    ///
    /// This variant indicates that the binary data contains a feature that is not
    /// supported by the current version of the deserializer. This typically occurs
    /// when reading data written by a newer version that includes features not yet
    /// implemented in the current version.
    ///
    /// # Fields
    ///
    /// * `feature` - Name of the unsupported feature that was encountered
    /// * `version` - Version where this feature was introduced
    ///
    /// # Common Causes
    ///
    /// - Reading data written by a newer version of the library
    /// - Feature not yet implemented in the current version
    /// - Experimental features not enabled in the current build
    UnsupportedFeature {
        /// Name of the unsupported feature that was encountered
        feature: String,
        /// Version where this feature was introduced
        version: u32,
    },
}

impl fmt::Display for BinaryError {
    /// Formats the binary error for human-readable display
    ///
    /// This implementation provides detailed, human-readable error messages for
    /// each variant of the BinaryError enum. The messages include all relevant
    /// information to help diagnose and debug binary serialization issues.
    ///
    /// # Format Examples
    ///
    /// - Format errors: "Binary format error at position 42: Invalid field type"
    /// - Version mismatches: "Binary version mismatch: expected 1, found 2"
    /// - Invalid magic: "Invalid binary magic number: expected 0x54535F42, found 0x00000000"
    /// - Unsupported features: "Unsupported binary feature 'compression' in version 2"
    ///
    /// # Returns
    ///
    /// A formatted string representation of the error suitable for logging,
    /// debugging, and user-facing error messages.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryError::Format { message, position } => match position {
                Some(pos) => {
                    write!(f, "Binary format error at position {}: {}", pos, message)
                }
                None => {
                    write!(f, "Binary format error: {}", message)
                }
            },
            BinaryError::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "Binary version mismatch: expected {}, found {}",
                    expected, found
                )
            }
            BinaryError::InvalidMagic { expected, found } => {
                write!(
                    f,
                    "Invalid binary magic number: expected 0x{:08X}, found 0x{:08X}",
                    expected, found
                )
            }
            BinaryError::UnsupportedFeature { feature, version } => {
                write!(
                    f,
                    "Unsupported binary feature '{}' in version {}",
                    feature, version
                )
            }
        }
    }
}

impl std::error::Error for BinaryError {}
