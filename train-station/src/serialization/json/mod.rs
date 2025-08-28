//! JSON serialization and deserialization for Train Station objects
//!
//! This module provides human-readable JSON serialization using a custom parser
//! and formatter optimized for Train Station objects. The JSON format is designed
//! for debugging, configuration files, and cross-language interoperability while
//! maintaining full compliance with the JSON specification.
//!
//! # Purpose
//!
//! The JSON module serves as the human-readable serialization format for the
//! Train Station library, providing:
//! - **Debugging support**: Readable output for development and troubleshooting
//! - **Configuration files**: Human-editable configuration and settings
//! - **Cross-language interoperability**: Standard JSON format for external tools
//! - **Data inspection**: Easy visualization of serialized data structures
//! - **Documentation**: Self-documenting data format for API responses
//!
//! # Design Philosophy
//!
//! The JSON format prioritizes:
//! - **Human Readability**: Clear, formatted output for debugging and inspection
//! - **Cross-Platform Compatibility**: Standard JSON format for interoperability
//! - **Type Safety**: Strongly typed parsing with comprehensive error handling
//! - **Performance**: Optimized parsing and formatting for large objects
//! - **Extensibility**: Support for custom types and nested structures
//! - **Zero Dependencies**: Pure Rust implementation without external JSON libraries
//!
//! # JSON Format Features
//!
//! The JSON implementation supports the complete JSON specification:
//! - **Primitive Types**: null, boolean, number, string
//! - **Complex Types**: arrays and objects with nested structures
//! - **Unicode Support**: Full UTF-8 encoding with proper escaping
//! - **Number Precision**: Maintains floating-point precision for scientific notation
//! - **Whitespace Handling**: Flexible whitespace and formatting options
//! - **Error Recovery**: Attempts to recover from common parsing errors
//!
//! # Module Organization
//!
//! The JSON module is organized into focused submodules:
//!
//! - **`value.rs`** - JSON value representation and manipulation
//! - **`parser.rs`** - JSON parsing with error recovery and validation
//! - **`formatter.rs`** - JSON formatting with pretty-printing support
//! - **`escape.rs`** - String escaping and unescaping utilities
//! - **`structs.rs`** - Internal serialization structures and utilities
//! - **`error.rs`** - JSON-specific error types and handling
//!
//! # Core Components
//!
//! ## JsonValue
//! Universal JSON value representation supporting all JSON types with type-safe
//! access methods and comprehensive validation.
//!
//! ## Parser
//! High-performance JSON parser with detailed error reporting, line/column tracking,
//! and recovery mechanisms for common parsing errors.
//!
//! ## Formatter
//! Configurable JSON formatter supporting both compact and pretty-printed output
//! with customizable indentation and formatting options.
//!
//! ## Escape Utilities
//! Comprehensive string escaping and unescaping for proper JSON string handling,
//! supporting all Unicode characters and control sequences.
//!
//! # Error Handling
//!
//! The JSON parser provides comprehensive error handling with detailed diagnostics:
//!
//! - **Syntax Errors**: Detailed error messages with line and column information
//! - **Type Errors**: Clear indication of expected vs actual types
//! - **Validation**: Comprehensive validation of JSON structure and content
//! - **Recovery**: Attempts to recover from common parsing errors
//! - **Context**: Error messages include surrounding context for debugging
//!
//! # Performance Characteristics
//!
//! The JSON implementation is optimized for typical use cases:
//!
//! - **Parsing Speed**: Optimized for typical JSON document sizes (1KB-1MB)
//! - **Memory Usage**: Efficient memory allocation with minimal overhead
//! - **Formatting**: Fast pretty-printing with configurable indentation
//! - **Validation**: Minimal overhead for type checking and validation
//! - **String Handling**: Optimized UTF-8 processing and escaping
//!
//! # Thread Safety
//!
//! All JSON components are thread-safe and can be used concurrently across
//! multiple threads. No shared state is maintained between operations.
//!
//! # Integration with Serialization Framework
//!
//! The JSON module integrates seamlessly with the core serialization framework:
//!
//! - **FieldValue Conversion**: Automatic conversion between JSON and FieldValue types
//! - **Struct Serialization**: Direct support for StructSerializer/StructDeserializer
//! - **Error Propagation**: Unified error handling through SerializationError
//! - **Format Consistency**: Consistent behavior across all serialization formats

// ===== Module Declarations =====

/// JSON-specific error types and error handling utilities
pub mod error;

/// String escaping and unescaping utilities for JSON string handling
pub mod escape;

/// JSON formatting and pretty-printing functionality
pub mod formatter;

/// JSON parsing and validation functionality
pub mod parser;

/// Internal serialization structures and utilities
pub mod structs;

/// JSON value representation and manipulation
pub mod value;

// ===== Public Exports =====

/// JSON-specific error type for parsing and formatting operations
///
/// Provides detailed error information including line/column numbers and
/// context for debugging JSON parsing and formatting issues.
pub use error::JsonError;

/// High-performance JSON parser function
///
/// Parses JSON strings into `JsonValue` objects with comprehensive error
/// reporting and recovery mechanisms for common parsing errors.
pub use parser::parse;

/// Internal JSON serialization utilities for the core serialization framework
///
/// These functions provide the bridge between the JSON module and the core
/// serialization system, handling conversion between JSON and FieldValue types.
pub use structs::{from_json_internal, to_json_internal};

/// Universal JSON value representation
///
/// Supports all JSON types (null, boolean, number, string, array, object)
/// with type-safe access methods and comprehensive validation capabilities.
pub use value::JsonValue;
