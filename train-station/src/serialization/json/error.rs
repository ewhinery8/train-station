//! JSON-specific error types for serialization and parsing operations
//!
//! This module provides comprehensive error handling for JSON serialization and parsing
//! operations within the Train Station project. It includes detailed error information
//! with location tracking and feature compatibility checking.
//!
//! # Error Types
//!
//! - **Format Errors**: Invalid JSON syntax, structure, or parsing issues with precise
//!   location information (line and column numbers)
//! - **Unsupported Feature Errors**: JSON features not supported by the current parser
//!   version with feature name and version information
//!
//! # Usage
//!
//! The `JsonError` enum is used throughout the JSON serialization system to provide
//! detailed error information for debugging and error handling. It implements the
//! standard `std::error::Error` trait for integration with Rust's error handling
//! ecosystem.
//!
//! # Implementation Details
//!
//! This module is part of the internal JSON serialization system and provides
//! error types that are used internally by the serialization framework. The error
//! types include detailed location information and feature compatibility checking
//! to help developers debug JSON parsing issues effectively.

use std::fmt;

/// JSON-specific error type for serialization and parsing operations
///
/// This enum provides detailed error information for JSON-related operations,
/// including format errors with location tracking and unsupported feature
/// notifications. It implements the standard `std::error::Error` trait for
/// seamless integration with Rust's error handling ecosystem.
///
/// # Error Variants
///
/// - **Format**: Invalid JSON syntax, structure, or parsing issues with optional
///   line and column location information
/// - **UnsupportedFeature**: JSON features not supported by the current parser
///   version with feature name and version details
///
/// # Error Handling
///
/// The `JsonError` type is designed to provide maximum debugging information
/// while maintaining compatibility with standard Rust error handling patterns.
/// Location information is provided when available to help identify the exact
/// source of JSON parsing issues.
///
/// # Thread Safety
///
/// This type is thread-safe and can be safely shared between threads.
///
/// # Implementation Notes
///
/// This error type is used internally by the JSON serialization system and
/// provides comprehensive error information for debugging JSON parsing issues.
/// The error messages include location information when available and detailed
/// descriptions of the specific JSON problems encountered.
#[derive(Debug)]
pub enum JsonError {
    /// Invalid JSON format or structure with optional location information
    ///
    /// This variant represents errors that occur during JSON parsing or validation,
    /// such as syntax errors, structural issues, or malformed JSON data. It includes
    /// optional line and column information to help identify the exact location
    /// of the error in the source JSON.
    ///
    /// # Fields
    ///
    /// * `message` - Human-readable error message describing the JSON problem
    /// * `line` - Line number where the error occurred (1-based, if available)
    /// * `column` - Column number where the error occurred (1-based, if available)
    ///
    /// # Common Causes
    ///
    /// - Invalid JSON syntax (missing quotes, brackets, etc.)
    /// - Malformed UTF-8 sequences
    /// - Unexpected token types
    /// - Structural validation failures
    /// - Depth limit exceeded
    ///
    /// # Error Message Format
    ///
    /// The error message format varies based on available location information:
    /// - With line and column: "JSON format error at line 5, column 12: message"
    /// - With line only: "JSON format error at line 5: message"
    /// - Without location: "JSON format error: message"
    Format {
        /// Human-readable error message describing the JSON problem
        message: String,
        /// Line number where the error occurred (1-based, if available)
        line: Option<usize>,
        /// Column number where the error occurred (1-based, if available)
        column: Option<usize>,
    },

    /// Unsupported feature or format version
    ///
    /// This variant represents errors that occur when the JSON parser encounters
    /// features that are not supported by the current implementation. This is
    /// typically used for version compatibility or when parsing JSON with
    /// experimental or non-standard features.
    ///
    /// # Fields
    ///
    /// * `feature` - Name of the unsupported feature
    /// * `version` - Version where this feature was introduced
    ///
    /// # Common Causes
    ///
    /// - JSON comments (not supported in standard JSON)
    /// - Trailing commas in objects or arrays
    /// - Non-standard number formats
    /// - Custom JSON extensions
    ///
    /// # Error Message Format
    ///
    /// The error message format is: "Unsupported JSON feature 'feature' in version X"
    UnsupportedFeature {
        /// Name of the unsupported feature
        feature: String,
        /// Version where this feature was introduced
        version: u32,
    },
}

impl fmt::Display for JsonError {
    /// Formats the JSON error for display
    ///
    /// Provides a human-readable representation of the error, including
    /// location information when available. The format varies based on
    /// the error variant and available information.
    ///
    /// # Format Examples
    ///
    /// - Format error with line and column: "JSON format error at line 5, column 12: message"
    /// - Format error with line only: "JSON format error at line 5: message"
    /// - Format error without location: "JSON format error: message"
    /// - Unsupported feature: "Unsupported JSON feature 'feature' in version X"
    ///
    /// # Arguments
    ///
    /// * `f` - The formatter to write the error message to
    ///
    /// # Returns
    ///
    /// `fmt::Result` indicating success or failure of the formatting operation
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonError::Format {
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
            JsonError::UnsupportedFeature { feature, version } => {
                write!(
                    f,
                    "Unsupported JSON feature '{}' in version {}",
                    feature, version
                )
            }
        }
    }
}

impl std::error::Error for JsonError {
    /// Provides error source information
    ///
    /// Since `JsonError` is a leaf error type (does not wrap other errors),
    /// this implementation always returns `None`.
    ///
    /// # Returns
    ///
    /// `None` - This error type does not have a source error
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
