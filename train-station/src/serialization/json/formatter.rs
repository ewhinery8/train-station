//! JSON formatting with pretty-printing support
//!
//! This module provides comprehensive JSON formatting capabilities with both compact
//! and pretty-printed output options. It includes proper string escaping, configurable
//! indentation, and consistent key ordering for human-readable output and debugging.
//!
//! # Purpose
//!
//! The JSON formatter module serves as the output formatting layer for the JSON
//! serialization system, providing:
//! - **Pretty printing**: Human-readable formatted output with proper indentation
//! - **Compact formatting**: Minimal whitespace for efficient storage and transmission
//! - **String escaping**: Proper JSON string escaping for all Unicode characters
//! - **Consistent output**: Deterministic formatting with sorted object keys
//! - **Performance**: Optimized formatting for large JSON structures
//!
//! # Formatting Options
//!
//! ## Pretty Printing
//! - **2-space indentation**: Consistent indentation for readability
//! - **Line breaks**: Proper line breaks for complex structures
//! - **Key sorting**: Deterministic object key ordering
//! - **String escaping**: Full Unicode and control character escaping
//!
//! ## Compact Formatting
//! - **Minimal whitespace**: No unnecessary spaces or line breaks
//! - **Efficient storage**: Optimized for file size and network transmission
//! - **Consistent output**: Same key sorting and escaping as pretty format
//! - **Fast formatting**: Minimal processing overhead
//!
//! # Core Components
//!
//! ## Public Functions
//! - **`format_pretty`**: High-level pretty printing with error handling
//! - **`format_compact`**: High-level compact formatting with error handling
//!
//! ## Internal Methods
//! - **`write_pretty`**: Low-level pretty printing implementation
//! - **`write_compact`**: Low-level compact formatting implementation
//!
//! # Performance Characteristics
//!
//! - **String allocation**: Efficient string buffer management
//! - **Memory usage**: Minimal temporary allocations during formatting
//! - **Formatting speed**: Optimized for typical JSON document sizes
//! - **Key sorting**: Efficient sorting for consistent object output
//!
//! # Thread Safety
//!
//! All formatting functions are thread-safe and can be used concurrently
//! across multiple threads. No shared state is maintained between operations.

use super::escape::escape_string;
use super::value::JsonValue;
use crate::serialization::core::{SerializationError, SerializationResult};
use std::fmt::Write;

/// Format a JSON value with pretty printing for human readability
///
/// This function formats a JSON value with proper indentation, line breaks, and
/// consistent key ordering for optimal human readability. It uses 2-space indentation
/// by default and sorts object keys alphabetically for deterministic output.
///
/// The pretty-printed format is ideal for:
/// - **Debugging**: Easy to read and inspect JSON data
/// - **Configuration files**: Human-editable JSON documents
/// - **Documentation**: Self-documenting data structures
/// - **Development**: Clear visualization of complex nested structures
///
/// # Arguments
///
/// * `value` - The JSON value to format with pretty printing
///
/// # Returns
///
/// `Ok(String)` containing the pretty-printed JSON on success
/// `Err(SerializationError)` if formatting fails
///
/// # Formatting Features
///
/// - **2-space indentation**: Consistent indentation for readability
/// - **Line breaks**: Proper line breaks for arrays and objects
/// - **Key sorting**: Alphabetical sorting of object keys for consistency
/// - **String escaping**: Full Unicode and control character escaping
/// - **Number formatting**: Proper handling of integers vs floating-point numbers
///
/// # Performance Characteristics
///
/// - **Memory efficient**: Uses string buffer for minimal allocations
/// - **Fast formatting**: Optimized for typical JSON document sizes
/// - **Deterministic output**: Consistent formatting across multiple runs
#[allow(unused)]
pub fn format_pretty(value: &JsonValue) -> SerializationResult<String> {
    let mut result = String::new();
    value
        .write_pretty(&mut result, 0)
        .map_err(|_| SerializationError::Custom("JSON formatting failed".to_string()))?;
    Ok(result)
}

/// Format a JSON value in compact form for efficient storage and transmission
///
/// This function formats a JSON value without unnecessary whitespace, line breaks,
/// or indentation for minimal file size and optimal network transmission efficiency.
/// It maintains the same key sorting and string escaping as the pretty format for
/// consistency.
///
/// The compact format is ideal for:
/// - **Network transmission**: Minimal bandwidth usage
/// - **Storage efficiency**: Reduced file sizes
/// - **Performance**: Faster parsing due to reduced character count
/// - **Embedded systems**: Memory-efficient JSON representation
/// - **API responses**: Efficient data transfer
///
/// # Arguments
///
/// * `value` - The JSON value to format in compact form
///
/// # Returns
///
/// `Ok(String)` containing the compact JSON on success
/// `Err(SerializationError)` if formatting fails
///
/// # Formatting Features
///
/// - **No whitespace**: Minimal character count for efficiency
/// - **Key sorting**: Alphabetical sorting of object keys for consistency
/// - **String escaping**: Full Unicode and control character escaping
/// - **Number formatting**: Proper handling of integers vs floating-point numbers
/// - **Deterministic output**: Consistent formatting across multiple runs
///
/// # Performance Characteristics
///
/// - **Memory efficient**: Uses string buffer for minimal allocations
/// - **Fast formatting**: Optimized for minimal processing overhead
/// - **Compact output**: Minimal character count for storage and transmission
#[allow(unused)]
pub fn format_compact(value: &JsonValue) -> SerializationResult<String> {
    let mut result = String::new();
    value
        .write_compact(&mut result)
        .map_err(|_| SerializationError::Custom("JSON formatting failed".to_string()))?;
    Ok(result)
}

impl JsonValue {
    /// Write the JSON value with pretty printing to a string buffer
    ///
    /// This internal method handles the low-level pretty printing implementation,
    /// writing formatted JSON with proper indentation, line breaks, and consistent
    /// key ordering. It recursively processes nested structures and maintains
    /// proper formatting throughout the JSON hierarchy.
    ///
    /// # Arguments
    ///
    /// * `f` - The string buffer to write the formatted JSON to
    /// * `indent` - Current indentation level (number of 2-space increments)
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful formatting
    /// `Err(std::fmt::Error)` if writing to the buffer fails
    ///
    /// # Implementation Details
    ///
    /// - **Indentation**: Uses 2-space increments for consistent formatting
    /// - **Line breaks**: Adds proper line breaks for arrays and objects
    /// - **Key sorting**: Sorts object keys alphabetically for deterministic output
    /// - **String escaping**: Applies proper JSON string escaping
    /// - **Number formatting**: Handles integer vs floating-point representation
    /// - **Empty structures**: Special handling for empty arrays and objects
    #[allow(unused)]
    fn write_pretty(&self, f: &mut String, indent: usize) -> std::fmt::Result {
        match self {
            JsonValue::Null => write!(f, "null"),
            JsonValue::Bool(b) => write!(f, "{}", b),
            JsonValue::Number(n) => {
                if n.fract() == 0.0 {
                    write!(f, "{:.0}", n)
                } else {
                    write!(f, "{}", n)
                }
            }
            JsonValue::String(s) => {
                write!(f, "\"")?;
                write!(f, "{}", escape_string(s))?;
                write!(f, "\"")
            }
            JsonValue::Array(arr) => {
                if arr.is_empty() {
                    write!(f, "[]")
                } else {
                    writeln!(f, "[")?;
                    for (i, item) in arr.iter().enumerate() {
                        write!(f, "{}", "  ".repeat(indent + 1))?;
                        item.write_pretty(f, indent + 1)?;
                        if i < arr.len() - 1 {
                            writeln!(f, ",")?;
                        } else {
                            writeln!(f)?;
                        }
                    }
                    write!(f, "{}]", "  ".repeat(indent))
                }
            }
            JsonValue::Object(obj) => {
                if obj.is_empty() {
                    write!(f, "{{}}")
                } else {
                    writeln!(f, "{{")?;
                    let mut entries: Vec<_> = obj.iter().collect();
                    entries.sort_by(|a, b| a.0.cmp(b.0)); // Sort keys for consistent output

                    for (i, (key, value)) in entries.iter().enumerate() {
                        write!(f, "{}", "  ".repeat(indent + 1))?;
                        write!(f, "\"{}\": ", escape_string(key))?;
                        value.write_pretty(f, indent + 1)?;
                        if i < entries.len() - 1 {
                            writeln!(f, ",")?;
                        } else {
                            writeln!(f)?;
                        }
                    }
                    write!(f, "{}}}", "  ".repeat(indent))
                }
            }
        }
    }

    /// Write the JSON value in compact form to a string buffer
    ///
    /// This internal method handles the low-level compact formatting implementation,
    /// writing JSON without unnecessary whitespace, line breaks, or indentation.
    /// It recursively processes nested structures while maintaining minimal
    /// character count for optimal storage and transmission efficiency.
    ///
    /// # Arguments
    ///
    /// * `f` - The string buffer to write the compact JSON to
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful formatting
    /// `Err(std::fmt::Error)` if writing to the buffer fails
    ///
    /// # Implementation Details
    ///
    /// - **No whitespace**: Minimal character count for efficiency
    /// - **Key sorting**: Sorts object keys alphabetically for consistency
    /// - **String escaping**: Applies proper JSON string escaping
    /// - **Number formatting**: Handles integer vs floating-point representation
    /// - **Comma handling**: Proper comma placement between array/object elements
    /// - **Empty structures**: Special handling for empty arrays and objects
    #[allow(unused)]
    fn write_compact(&self, f: &mut String) -> std::fmt::Result {
        match self {
            JsonValue::Null => write!(f, "null"),
            JsonValue::Bool(b) => write!(f, "{}", b),
            JsonValue::Number(n) => {
                if n.fract() == 0.0 {
                    write!(f, "{:.0}", n)
                } else {
                    write!(f, "{}", n)
                }
            }
            JsonValue::String(s) => {
                write!(f, "\"")?;
                write!(f, "{}", escape_string(s))?;
                write!(f, "\"")
            }
            JsonValue::Array(arr) => {
                write!(f, "[")?;
                for (i, item) in arr.iter().enumerate() {
                    item.write_compact(f)?;
                    if i < arr.len() - 1 {
                        write!(f, ",")?;
                    }
                }
                write!(f, "]")
            }
            JsonValue::Object(obj) => {
                write!(f, "{{")?;
                let mut entries: Vec<_> = obj.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0)); // Sort keys for consistent output

                for (i, (key, value)) in entries.iter().enumerate() {
                    write!(f, "\"{}\":", escape_string(key))?;
                    value.write_compact(f)?;
                    if i < entries.len() - 1 {
                        write!(f, ",")?;
                    }
                }
                write!(f, "}}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_format_pretty() {
        let value = JsonValue::object_from_iter(vec![
            ("name".to_string(), JsonValue::String("test".to_string())),
            ("value".to_string(), JsonValue::Number(42.0)),
            (
                "array".to_string(),
                JsonValue::Array(vec![
                    JsonValue::Number(1.0),
                    JsonValue::Number(2.0),
                    JsonValue::Number(3.0),
                ]),
            ),
        ]);

        let formatted = format_pretty(&value).unwrap();
        assert!(formatted.contains("\n"));
        assert!(formatted.contains("  ")); // Indentation
        assert!(formatted.contains("name"));
        assert!(formatted.contains("value"));
    }

    #[test]
    fn test_format_compact() {
        let value = JsonValue::object_from_iter(vec![
            ("name".to_string(), JsonValue::String("test".to_string())),
            ("value".to_string(), JsonValue::Number(42.0)),
        ]);

        let compact = format_compact(&value).unwrap();
        assert!(!compact.contains('\n'));
        assert!(!compact.contains("  ")); // No indentation
        assert_eq!(compact, r#"{"name":"test","value":42}"#);
    }

    #[test]
    fn test_format_comprehensive() {
        // Test complex nested structure
        let value = JsonValue::object_from_iter(vec![
            ("name".to_string(), JsonValue::String("test".to_string())),
            ("value".to_string(), JsonValue::Number(42.0)),
            ("enabled".to_string(), JsonValue::Bool(true)),
            ("null_value".to_string(), JsonValue::Null),
            (
                "array".to_string(),
                JsonValue::array(vec![
                    JsonValue::Number(1.0),
                    JsonValue::Number(2.0),
                    JsonValue::Number(3.0),
                ]),
            ),
            (
                "nested".to_string(),
                JsonValue::object_from_iter(vec![
                    ("inner".to_string(), JsonValue::String("value".to_string())),
                    (
                        "number".to_string(),
                        JsonValue::Number(std::f64::consts::PI),
                    ),
                ]),
            ),
        ]);

        // Test compact formatting
        let compact = format_compact(&value).unwrap();
        assert!(!compact.contains('\n'));
        assert!(!compact.contains("  "));
        assert!(compact.contains("name"));
        assert!(compact.contains("value"));
        assert!(compact.contains("enabled"));
        assert!(compact.contains("null_value"));
        assert!(compact.contains("array"));
        assert!(compact.contains("nested"));
        assert!(compact.contains("[1,2,3]"));
        assert!(compact.contains("inner"));

        // Test pretty formatting
        let pretty = format_pretty(&value).unwrap();
        assert!(pretty.contains('\n'));
        assert!(pretty.contains("  "));
        assert!(pretty.contains("name"));
        assert!(pretty.contains("value"));
        assert!(pretty.contains("enabled"));
        assert!(pretty.contains("null_value"));
        assert!(pretty.contains("array"));
        assert!(pretty.contains("nested"));
        assert!(pretty.contains("["));
        assert!(pretty.contains("]"));
        assert!(pretty.contains("inner"));
    }

    #[test]
    fn test_format_edge_cases() {
        // Test empty objects and arrays
        let empty_object = JsonValue::object(HashMap::new());
        assert_eq!(format_compact(&empty_object).unwrap(), "{}");
        assert_eq!(format_pretty(&empty_object).unwrap(), "{}");

        let empty_array = JsonValue::array(vec![]);
        assert_eq!(format_compact(&empty_array).unwrap(), "[]");
        assert_eq!(format_pretty(&empty_array).unwrap(), "[]");

        // Test primitive values
        assert_eq!(format_compact(&JsonValue::Null).unwrap(), "null");
        assert_eq!(format_pretty(&JsonValue::Null).unwrap(), "null");

        assert_eq!(format_compact(&JsonValue::Bool(true)).unwrap(), "true");
        assert_eq!(format_pretty(&JsonValue::Bool(true)).unwrap(), "true");

        assert_eq!(format_compact(&JsonValue::Bool(false)).unwrap(), "false");
        assert_eq!(format_pretty(&JsonValue::Bool(false)).unwrap(), "false");

        assert_eq!(format_compact(&JsonValue::Number(42.0)).unwrap(), "42");
        assert_eq!(format_pretty(&JsonValue::Number(42.0)).unwrap(), "42");

        assert_eq!(
            format_compact(&JsonValue::Number(std::f64::consts::PI)).unwrap(),
            "3.141592653589793"
        );
        assert_eq!(
            format_pretty(&JsonValue::Number(std::f64::consts::PI)).unwrap(),
            "3.141592653589793"
        );

        // Test strings with special characters
        let special_string = JsonValue::String("Hello\nWorld\twith \"quotes\"".to_string());
        let compact = format_compact(&special_string).unwrap();
        assert!(compact.contains("\\n"));
        assert!(compact.contains("\\t"));
        assert!(compact.contains("\\\""));

        let pretty = format_pretty(&special_string).unwrap();
        assert!(pretty.contains("\\n"));
        assert!(pretty.contains("\\t"));
        assert!(pretty.contains("\\\""));
    }

    #[test]
    fn test_format_consistency() {
        // Test that compact and pretty produce equivalent JSON (ignoring whitespace)
        let value = JsonValue::object_from_iter(vec![
            ("name".to_string(), JsonValue::String("test".to_string())),
            ("value".to_string(), JsonValue::Number(42.0)),
            (
                "array".to_string(),
                JsonValue::array(vec![
                    JsonValue::Number(1.0),
                    JsonValue::Number(2.0),
                    JsonValue::Number(3.0),
                ]),
            ),
        ]);

        let compact = format_compact(&value).unwrap();
        let pretty = format_pretty(&value).unwrap();

        // Remove all whitespace from pretty format
        let pretty_no_ws: String = pretty.chars().filter(|c| !c.is_whitespace()).collect();
        let compact_no_ws: String = compact.chars().filter(|c| !c.is_whitespace()).collect();

        assert_eq!(pretty_no_ws, compact_no_ws);
    }
}
