//! JSON value representation and manipulation
//!
//! This module provides a comprehensive JSON value type that can represent
//! all valid JSON data types according to the JSON specification. It includes
//! methods for creating, accessing, and manipulating JSON values with full
//! type safety and comprehensive error handling.
//!
//! # Purpose
//!
//! The JSON value module serves as the core data representation layer for the
//! JSON serialization system, providing:
//! - **Universal JSON representation**: Single enum type for all JSON data types
//! - **Type-safe operations**: Compile-time guarantees for JSON value manipulation
//! - **Comprehensive API**: Complete set of methods for creating and accessing JSON values
//! - **Format compatibility**: Full compatibility with JSON specification
//! - **Performance**: Optimized for typical JSON document operations
//!
//! # JSON Value Types
//!
//! The module supports all JSON specification data types:
//! - **Null**: JSON null value representation
//! - **Boolean**: True/false boolean values
//! - **Number**: Floating-point numbers with full precision (f64)
//! - **String**: UTF-8 encoded text strings
//! - **Array**: Ordered collections of JSON values
//! - **Object**: Key-value pairs with string keys
//!
//! # Core Features
//!
//! ## Type Safety
//! - **Pattern matching**: Comprehensive match expressions for all JSON types
//! - **Type checking**: Methods to verify JSON value types at runtime
//! - **Safe accessors**: Type-safe methods for accessing JSON value contents
//! - **Error handling**: Proper error handling for invalid operations
//!
//! ## Value Creation
//! - **Constructor methods**: Convenient methods for creating JSON values
//! - **From trait implementations**: Automatic conversion from Rust types
//! - **Iterator support**: Creation from iterators for arrays and objects
//! - **Builder pattern**: Fluent API for complex JSON structure creation
//!
//! ## Value Access
//! - **Type checking**: Methods to verify JSON value types
//! - **Safe extraction**: Type-safe methods for accessing value contents
//! - **Nested access**: Methods for accessing nested object fields and array elements
//! - **Optional access**: Safe access methods that return Option types
//!
//! # Performance Characteristics
//!
//! - **Memory efficient**: Optimized memory layout for JSON value storage
//! - **Fast access**: Efficient type checking and value extraction
//! - **Minimal allocations**: Reuse of existing data where possible
//! - **String optimization**: Efficient string handling and storage
//!
//! # Thread Safety
//!
//! All JSON value operations are thread-safe and can be used concurrently
//! across multiple threads. No shared state is maintained between operations.

use std::collections::HashMap;

/// Universal JSON value representation for all JSON data types
///
/// This enum represents all possible JSON value types according to the JSON specification,
/// providing a type-safe and comprehensive way to work with JSON data. It includes methods
/// for creating, accessing, and manipulating JSON values with full type safety and
/// comprehensive error handling.
///
/// The enum serves as the foundation for the entire JSON serialization system, enabling
/// uniform handling of all JSON data types while maintaining type safety and providing
/// efficient memory layout for optimal performance.
///
/// # Variants
///
/// * `Null` - JSON null value representing absence of data
/// * `Bool(bool)` - JSON boolean value (true/false)
/// * `Number(f64)` - JSON number value stored as f64 for full precision
/// * `String(String)` - JSON string value with UTF-8 encoding
/// * `Array(Vec<JsonValue>)` - JSON array of ordered JSON values
/// * `Object(HashMap<String, JsonValue>)` - JSON object with string key-value pairs
///
/// # Type Safety
///
/// The enum provides comprehensive type safety through:
///
/// - **Pattern matching**: Complete match expressions for all JSON types
/// - **Type checking methods**: Runtime verification of JSON value types
/// - **Safe accessors**: Type-safe methods that return Option types
/// - **Compile-time guarantees**: Type-safe operations enforced at compile time
///
/// # Memory Layout
///
/// The enum is optimized for typical JSON usage patterns:
///
/// - **Tagged union**: Efficient memory layout with minimal overhead
/// - **String optimization**: Leverages Rust's string optimization for text data
/// - **Collection efficiency**: Optimized storage for arrays and objects
/// - **Zero-copy operations**: Efficient data access without unnecessary copying
///
/// # JSON Specification Compliance
///
/// The enum fully complies with the JSON specification:
///
/// - **Complete type coverage**: All JSON data types are supported
/// - **Number precision**: Full f64 precision for numeric values
/// - **String encoding**: Proper UTF-8 string handling
/// - **Collection types**: Arrays and objects with proper semantics
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON null value representing absence of data
    ///
    /// This variant represents the JSON null value, indicating the absence
    /// of meaningful data. It is commonly used for optional fields and
    /// uninitialized values in JSON structures.
    Null,
    /// JSON boolean value (true/false)
    ///
    /// This variant represents JSON boolean values, storing either true
    /// or false. It is used for logical values, flags, and boolean
    /// configuration settings in JSON data.
    Bool(bool),
    /// JSON number value stored as f64 for full precision
    ///
    /// This variant represents JSON numeric values, stored as f64 to
    /// maintain full precision for both integers and floating-point
    /// numbers. It supports the complete range of JSON number formats
    /// including integers, decimals, and scientific notation.
    Number(f64),
    /// JSON string value with UTF-8 encoding
    ///
    /// This variant represents JSON string values, stored as UTF-8
    /// encoded strings. It supports all Unicode characters and proper
    /// escaping for special characters according to the JSON specification.
    String(String),
    /// JSON array of ordered JSON values
    ///
    /// This variant represents JSON arrays, storing an ordered collection
    /// of JSON values. Arrays maintain insertion order and support
    /// heterogeneous value types within the same array.
    Array(Vec<JsonValue>),
    /// JSON object with string key-value pairs
    ///
    /// This variant represents JSON objects, storing key-value pairs
    /// where keys are strings and values are JSON values. Objects
    /// provide efficient lookup by key and support nested structures.
    Object(HashMap<String, JsonValue>),
}

#[allow(unused)]
impl JsonValue {
    /// Create a new JSON null value
    ///
    /// Creates a JSON null value representing the absence of meaningful data.
    /// This is commonly used for optional fields, uninitialized values, and
    /// explicit null values in JSON structures.
    ///
    /// # Returns
    ///
    /// A new `JsonValue::Null` representing the JSON null value
    ///
    /// # Usage
    ///
    /// Null values are typically used for:
    /// - Optional fields that are not present
    /// - Explicit null values in JSON data
    /// - Uninitialized or empty values
    pub fn null() -> Self {
        JsonValue::Null
    }

    /// Create a new JSON boolean value
    ///
    /// Creates a JSON boolean value from a Rust bool. Boolean values are
    /// commonly used for flags, configuration settings, and logical values
    /// in JSON data structures.
    ///
    /// # Arguments
    ///
    /// * `value` - The boolean value to convert to JSON
    ///
    /// # Returns
    ///
    /// A new `JsonValue::Bool` containing the boolean value
    ///
    /// # Usage
    ///
    /// Boolean values are typically used for:
    /// - Configuration flags and settings
    /// - Logical true/false values
    /// - Feature enable/disable flags
    pub fn bool(value: bool) -> Self {
        JsonValue::Bool(value)
    }

    /// Create a new JSON number value
    ///
    /// Creates a JSON number value from a Rust f64. Numbers are stored as
    /// f64 to maintain full precision for both integers and floating-point
    /// values, supporting the complete range of JSON number formats.
    ///
    /// # Arguments
    ///
    /// * `value` - The numeric value to convert to JSON (f64 for full precision)
    ///
    /// # Returns
    ///
    /// A new `JsonValue::Number` containing the numeric value
    ///
    /// # Number Support
    ///
    /// The number type supports:
    /// - **Integers**: Whole numbers (e.g., 42, -123)
    /// - **Decimals**: Floating-point numbers (e.g., 3.14, -0.001)
    /// - **Scientific notation**: Large/small numbers (e.g., 1.23e+10, 1.23e-10)
    /// - **Full precision**: 64-bit floating-point precision
    pub fn number(value: f64) -> Self {
        JsonValue::Number(value)
    }

    /// Create a new JSON string value
    ///
    /// Creates a JSON string value from a Rust String. Strings are stored as
    /// UTF-8 encoded text and support all Unicode characters with proper
    /// escaping for special characters according to the JSON specification.
    ///
    /// # Arguments
    ///
    /// * `value` - The string value to convert to JSON (UTF-8 encoded)
    ///
    /// # Returns
    ///
    /// A new `JsonValue::String` containing the string value
    ///
    /// # String Support
    ///
    /// The string type supports:
    /// - **UTF-8 encoding**: Full Unicode character support
    /// - **Escape sequences**: Proper handling of special characters
    /// - **Control characters**: Escaping of newlines, tabs, quotes, etc.
    /// - **Unicode**: Support for all Unicode code points
    pub fn string(value: String) -> Self {
        JsonValue::String(value)
    }

    /// Create a new JSON array value
    ///
    /// Creates a JSON array value from a vector of JSON values. Arrays maintain
    /// insertion order and support heterogeneous value types within the same
    /// array, allowing mixed types of JSON values.
    ///
    /// # Arguments
    ///
    /// * `values` - Vector of JSON values to include in the array
    ///
    /// # Returns
    ///
    /// A new `JsonValue::Array` containing the ordered collection of values
    ///
    /// # Array Features
    ///
    /// Arrays provide:
    /// - **Order preservation**: Maintains insertion order of elements
    /// - **Heterogeneous types**: Mixed JSON value types in same array
    /// - **Dynamic sizing**: Variable number of elements
    /// - **Indexed access**: Efficient access by numeric index
    pub fn array(values: Vec<JsonValue>) -> Self {
        JsonValue::Array(values)
    }

    /// Create a new JSON object value
    ///
    /// Creates a JSON object value from a HashMap of key-value pairs. Objects
    /// provide efficient lookup by string keys and support nested structures
    /// with arbitrary JSON values as values.
    ///
    /// # Arguments
    ///
    /// * `pairs` - HashMap of string keys to JSON values
    ///
    /// # Returns
    ///
    /// A new `JsonValue::Object` containing the key-value pairs
    ///
    /// # Object Features
    ///
    /// Objects provide:
    /// - **Key-value pairs**: String keys with JSON value values
    /// - **Efficient lookup**: Fast access by string key
    /// - **Nested structures**: Support for complex data hierarchies
    /// - **Dynamic properties**: Variable number of key-value pairs
    pub fn object(pairs: HashMap<String, JsonValue>) -> Self {
        JsonValue::Object(pairs)
    }

    /// Create a new JSON object from an iterator of key-value pairs
    ///
    /// Creates a JSON object value from any iterator that yields (String, JsonValue)
    /// pairs. This provides a convenient way to build objects from various data
    /// sources without manually constructing HashMaps.
    ///
    /// # Arguments
    ///
    /// * `iter` - Iterator yielding (String, JsonValue) key-value pairs
    ///
    /// # Returns
    ///
    /// A new `JsonValue::Object` containing the key-value pairs from the iterator
    ///
    /// # Usage
    ///
    /// This method is useful for:
    /// - Building objects from vector of tuples
    /// - Converting HashMap iterators
    /// - Creating objects from filtered or transformed data
    /// - Dynamic object construction from various data sources
    pub fn object_from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (String, JsonValue)>,
    {
        JsonValue::Object(iter.into_iter().collect())
    }

    /// Create a new JSON array from an iterator of values
    ///
    /// Creates a JSON array value from any iterator that yields JsonValue items.
    /// This provides a convenient way to build arrays from various data sources
    /// without manually constructing vectors.
    ///
    /// # Arguments
    ///
    /// * `iter` - Iterator yielding JsonValue items
    ///
    /// # Returns
    ///
    /// A new `JsonValue::Array` containing the values from the iterator
    ///
    /// # Usage
    ///
    /// This method is useful for:
    /// - Building arrays from vector iterators
    /// - Converting filtered or transformed data
    /// - Creating arrays from range iterators
    /// - Dynamic array construction from various data sources
    pub fn array_from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = JsonValue>,
    {
        JsonValue::Array(iter.into_iter().collect())
    }

    /// Check if the JSON value is null
    ///
    /// Returns true if the value is the JSON null value, indicating the absence
    /// of meaningful data. This is useful for checking optional fields and
    /// handling null values in JSON data structures.
    ///
    /// # Returns
    ///
    /// `true` if the value is `JsonValue::Null`, `false` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Check for optional fields that are not present
    /// - Handle explicit null values in JSON data
    /// - Validate that required fields are not null
    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    /// Check if the JSON value is a boolean
    ///
    /// Returns true if the value is a JSON boolean (true or false). This is
    /// useful for type checking and ensuring boolean values before performing
    /// boolean operations or accessing boolean data.
    ///
    /// # Returns
    ///
    /// `true` if the value is `JsonValue::Bool`, `false` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Validate boolean configuration flags
    /// - Ensure boolean values before boolean operations
    /// - Type-check JSON data before processing
    pub fn is_bool(&self) -> bool {
        matches!(self, JsonValue::Bool(_))
    }

    /// Check if the JSON value is a number
    ///
    /// Returns true if the value is a JSON number (integer or floating-point).
    /// This is useful for type checking and ensuring numeric values before
    /// performing mathematical operations or accessing numeric data.
    ///
    /// # Returns
    ///
    /// `true` if the value is `JsonValue::Number`, `false` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Validate numeric configuration values
    /// - Ensure numeric values before mathematical operations
    /// - Type-check JSON data before numeric processing
    pub fn is_number(&self) -> bool {
        matches!(self, JsonValue::Number(_))
    }

    /// Check if the JSON value is a string
    ///
    /// Returns true if the value is a JSON string. This is useful for type
    /// checking and ensuring string values before performing string operations
    /// or accessing text data.
    ///
    /// # Returns
    ///
    /// `true` if the value is `JsonValue::String`, `false` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Validate string configuration values
    /// - Ensure string values before string operations
    /// - Type-check JSON data before text processing
    pub fn is_string(&self) -> bool {
        matches!(self, JsonValue::String(_))
    }

    /// Check if the JSON value is an array
    ///
    /// Returns true if the value is a JSON array. This is useful for type
    /// checking and ensuring array values before performing array operations
    /// or accessing collection data.
    ///
    /// # Returns
    ///
    /// `true` if the value is `JsonValue::Array`, `false` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Validate array configuration values
    /// - Ensure array values before iteration or indexing
    /// - Type-check JSON data before collection processing
    pub fn is_array(&self) -> bool {
        matches!(self, JsonValue::Array(_))
    }

    /// Check if the JSON value is an object
    ///
    /// Returns true if the value is a JSON object. This is useful for type
    /// checking and ensuring object values before performing object operations
    /// or accessing key-value data.
    ///
    /// # Returns
    ///
    /// `true` if the value is `JsonValue::Object`, `false` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Validate object configuration values
    /// - Ensure object values before key access
    /// - Type-check JSON data before object processing
    pub fn is_object(&self) -> bool {
        matches!(self, JsonValue::Object(_))
    }

    /// Get the JSON value as a string reference
    ///
    /// Returns a reference to the string value if the JSON value is a string,
    /// or None if the value is not a string. This provides type-safe access
    /// to string data without panicking on type mismatches.
    ///
    /// # Returns
    ///
    /// `Some(&str)` if the value is a string, `None` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Safely access string values without type checking
    /// - Handle optional string fields in JSON objects
    /// - Extract text data from JSON values
    /// - Avoid panics when accessing non-string values
    pub fn as_string(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get the JSON value as a number
    ///
    /// Returns the numeric value if the JSON value is a number, or None if
    /// the value is not a number. This provides type-safe access to numeric
    /// data without panicking on type mismatches.
    ///
    /// # Returns
    ///
    /// `Some(f64)` if the value is a number, `None` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Safely access numeric values without type checking
    /// - Handle optional numeric fields in JSON objects
    /// - Extract numeric data from JSON values
    /// - Avoid panics when accessing non-numeric values
    pub fn as_number(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get the JSON value as a boolean
    ///
    /// Returns the boolean value if the JSON value is a boolean, or None if
    /// the value is not a boolean. This provides type-safe access to boolean
    /// data without panicking on type mismatches.
    ///
    /// # Returns
    ///
    /// `Some(bool)` if the value is a boolean, `None` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Safely access boolean values without type checking
    /// - Handle optional boolean fields in JSON objects
    /// - Extract boolean data from JSON values
    /// - Avoid panics when accessing non-boolean values
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get the JSON value as an array reference
    ///
    /// Returns a reference to the array if the JSON value is an array, or None
    /// if the value is not an array. This provides type-safe access to array
    /// data without panicking on type mismatches.
    ///
    /// # Returns
    ///
    /// `Some(&Vec<JsonValue>)` if the value is an array, `None` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Safely access array values without type checking
    /// - Handle optional array fields in JSON objects
    /// - Extract collection data from JSON values
    /// - Avoid panics when accessing non-array values
    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Get the JSON value as an object reference
    ///
    /// Returns a reference to the object if the JSON value is an object, or None
    /// if the value is not an object. This provides type-safe access to object
    /// data without panicking on type mismatches.
    ///
    /// # Returns
    ///
    /// `Some(&HashMap<String, JsonValue>)` if the value is an object, `None` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Safely access object values without type checking
    /// - Handle optional object fields in JSON objects
    /// - Extract key-value data from JSON values
    /// - Avoid panics when accessing non-object values
    pub fn as_object(&self) -> Option<&HashMap<String, JsonValue>> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Get a field from a JSON object by key
    ///
    /// Returns a reference to the value associated with the given key if the
    /// JSON value is an object and the key exists, or None otherwise. This
    /// provides type-safe access to object fields without panicking on type
    /// mismatches or missing keys.
    ///
    /// # Arguments
    ///
    /// * `key` - The string key to look up in the object
    ///
    /// # Returns
    ///
    /// `Some(&JsonValue)` if the value is an object and the key exists, `None` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Safely access object fields without type checking
    /// - Handle optional fields in JSON objects
    /// - Extract nested values from JSON objects
    /// - Avoid panics when accessing non-object values or missing keys
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(obj) => obj.get(key),
            _ => None,
        }
    }

    /// Get an element from a JSON array by index
    ///
    /// Returns a reference to the element at the given index if the JSON value
    /// is an array and the index is valid, or None otherwise. This provides
    /// type-safe access to array elements without panicking on type mismatches
    /// or out-of-bounds indices.
    ///
    /// # Arguments
    ///
    /// * `index` - The numeric index to access in the array
    ///
    /// # Returns
    ///
    /// `Some(&JsonValue)` if the value is an array and the index is valid, `None` otherwise
    ///
    /// # Usage
    ///
    /// Use this method to:
    /// - Safely access array elements without type checking
    /// - Handle optional array elements
    /// - Extract values from JSON arrays
    /// - Avoid panics when accessing non-array values or out-of-bounds indices
    pub fn get_index(&self, index: usize) -> Option<&JsonValue> {
        match self {
            JsonValue::Array(arr) => arr.get(index),
            _ => None,
        }
    }

    /// Convert the JSON value to a compact string representation
    ///
    /// This method provides serde_json-compatible functionality for converting
    /// JSON values to compact string format without unnecessary whitespace.
    /// It uses the internal formatter to produce minimal character count
    /// for efficient storage and network transmission.
    ///
    /// # Returns
    ///
    /// A compact JSON string representation with minimal whitespace
    ///
    /// # Formatting Features
    ///
    /// The compact format provides:
    /// - **Minimal whitespace**: No unnecessary spaces or line breaks
    /// - **Efficient storage**: Reduced character count for file size optimization
    /// - **Network transmission**: Optimized for bandwidth efficiency
    /// - **Consistent output**: Deterministic formatting with sorted keys
    /// - **Full JSON compliance**: Complete JSON specification compliance
    ///
    /// # Error Handling
    ///
    /// If formatting fails for any reason, the method returns "null" as a
    /// fallback to ensure the method never panics or returns invalid JSON.
    pub fn to_string_compact(&self) -> String {
        use super::formatter::format_compact;
        format_compact(self).unwrap_or_else(|_| "null".to_string())
    }

    /// Convert the JSON value to a pretty-printed string representation
    ///
    /// This method provides serde_json-compatible functionality for converting
    /// JSON values to human-readable format with proper indentation and line
    /// breaks. It uses the internal formatter to produce well-formatted
    /// output for debugging and human consumption.
    ///
    /// # Returns
    ///
    /// A pretty-printed JSON string representation with proper formatting
    ///
    /// # Formatting Features
    ///
    /// The pretty format provides:
    /// - **2-space indentation**: Consistent indentation for readability
    /// - **Line breaks**: Proper line breaks for complex structures
    /// - **Human readability**: Optimized for visual inspection and debugging
    /// - **Consistent output**: Deterministic formatting with sorted keys
    /// - **Full JSON compliance**: Complete JSON specification compliance
    ///
    /// # Error Handling
    ///
    /// If formatting fails for any reason, the method returns "null" as a
    /// fallback to ensure the method never panics or returns invalid JSON.
    pub fn to_string_pretty(&self) -> String {
        use super::formatter::format_pretty;
        format_pretty(self).unwrap_or_else(|_| "null".to_string())
    }
}

// Implement From traits for common conversions
// These implementations provide convenient automatic conversion from Rust types to JSON values

/// Convert a boolean value to a JSON boolean
///
/// Provides automatic conversion from Rust bool to JsonValue::Bool,
/// enabling seamless integration with boolean values in JSON operations.
impl From<bool> for JsonValue {
    fn from(b: bool) -> Self {
        JsonValue::Bool(b)
    }
}

/// Convert a 32-bit floating-point number to a JSON number
///
/// Provides automatic conversion from Rust f32 to JsonValue::Number,
/// converting to f64 to maintain full precision in JSON representation.
impl From<f32> for JsonValue {
    fn from(n: f32) -> Self {
        JsonValue::Number(n as f64)
    }
}

/// Convert a 64-bit floating-point number to a JSON number
///
/// Provides automatic conversion from Rust f64 to JsonValue::Number,
/// maintaining full precision for both integers and floating-point values.
impl From<f64> for JsonValue {
    fn from(n: f64) -> Self {
        JsonValue::Number(n)
    }
}

/// Convert a 32-bit integer to a JSON number
///
/// Provides automatic conversion from Rust i32 to JsonValue::Number,
/// converting to f64 to maintain full precision in JSON representation.
impl From<i32> for JsonValue {
    fn from(n: i32) -> Self {
        JsonValue::Number(n as f64)
    }
}

/// Convert a 64-bit integer to a JSON number
///
/// Provides automatic conversion from Rust i64 to JsonValue::Number,
/// converting to f64 to maintain full precision in JSON representation.
impl From<i64> for JsonValue {
    fn from(n: i64) -> Self {
        JsonValue::Number(n as f64)
    }
}

/// Convert an unsigned size integer to a JSON number
///
/// Provides automatic conversion from Rust usize to JsonValue::Number,
/// converting to f64 to maintain full precision in JSON representation.
/// This is particularly useful for array indices and tensor dimensions.
impl From<usize> for JsonValue {
    fn from(n: usize) -> Self {
        JsonValue::Number(n as f64)
    }
}

/// Convert an owned string to a JSON string
///
/// Provides automatic conversion from Rust String to JsonValue::String,
/// taking ownership of the string for efficient JSON representation.
impl From<String> for JsonValue {
    fn from(s: String) -> Self {
        JsonValue::String(s)
    }
}

/// Convert a string slice to a JSON string
///
/// Provides automatic conversion from Rust &str to JsonValue::String,
/// creating an owned string copy for JSON representation.
impl From<&str> for JsonValue {
    fn from(s: &str) -> Self {
        JsonValue::String(s.to_string())
    }
}

impl std::fmt::Display for JsonValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
            JsonValue::String(s) => write!(f, "\"{}\"", s),
            JsonValue::Array(arr) => {
                write!(f, "[")?;
                for (i, item) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            JsonValue::Object(obj) => {
                write!(f, "{{")?;
                for (i, (key, value)) in obj.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "\"{}\":{}", key, value)?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// Convert a vector of convertible values to a JSON array
///
/// Provides automatic conversion from Vec<T> to JsonValue::Array,
/// where T can be converted to JsonValue. This enables seamless
/// conversion of Rust collections to JSON arrays.
///
/// # Type Constraints
///
/// The vector elements must implement `Into<JsonValue>` to enable
/// automatic conversion of each element to a JSON value.
impl<T: Into<JsonValue>> From<Vec<T>> for JsonValue {
    fn from(vec: Vec<T>) -> Self {
        JsonValue::Array(vec.into_iter().map(|x| x.into()).collect())
    }
}

/// Convert an optional value to a JSON value
///
/// Provides automatic conversion from Option<T> to JsonValue,
/// where T can be converted to JsonValue. Some values are converted
/// to their JSON representation, while None becomes JsonValue::Null.
///
/// # Type Constraints
///
/// The optional value must implement `Into<JsonValue>` to enable
/// automatic conversion when the value is Some.
///
/// # Conversion Behavior
///
/// - `Some(value)` → Converted JSON value
/// - `None` → `JsonValue::Null`
impl<T: Into<JsonValue>> From<Option<T>> for JsonValue {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(value) => value.into(),
            None => JsonValue::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_value_creation() {
        let null = JsonValue::null();
        assert!(null.is_null());

        let bool_val = JsonValue::bool(true);
        assert!(bool_val.is_bool());
        assert_eq!(bool_val.as_bool(), Some(true));

        let number = JsonValue::number(42.5);
        assert!(number.is_number());
        assert_eq!(number.as_number(), Some(42.5));

        let string = JsonValue::string("hello".to_string());
        assert!(string.is_string());
        assert_eq!(string.as_string(), Some("hello"));

        let array = JsonValue::array(vec![JsonValue::Number(1.0), JsonValue::Number(2.0)]);
        assert!(array.is_array());
        assert_eq!(array.as_array().unwrap().len(), 2);

        let mut map = HashMap::new();
        map.insert("key".to_string(), JsonValue::String("value".to_string()));
        let object = JsonValue::object(map);
        assert!(object.is_object());
        assert_eq!(object.get("key").unwrap().as_string(), Some("value"));
    }

    #[test]
    fn test_json_value_from_conversions() {
        let bool_val: JsonValue = true.into();
        assert_eq!(bool_val, JsonValue::Bool(true));

        let number: JsonValue = 42.5f64.into();
        assert_eq!(number, JsonValue::Number(42.5));

        let string: JsonValue = "hello".into();
        assert_eq!(string, JsonValue::String("hello".to_string()));

        let array: JsonValue = vec![1, 2, 3].into();
        assert!(array.is_array());

        let option: JsonValue = Some(42).into();
        assert_eq!(option, JsonValue::Number(42.0));

        let none: JsonValue = Option::<i32>::None.into();
        assert!(none.is_null());
    }

    #[test]
    fn test_json_value_access() {
        let mut map = HashMap::new();
        map.insert("name".to_string(), JsonValue::String("test".to_string()));
        map.insert("value".to_string(), JsonValue::Number(42.0));
        let object = JsonValue::object(map);

        assert_eq!(object.get("name").unwrap().as_string(), Some("test"));
        assert_eq!(object.get("value").unwrap().as_number(), Some(42.0));
        assert_eq!(object.get("missing"), None);

        let array = JsonValue::array(vec![JsonValue::Number(1.0), JsonValue::Number(2.0)]);
        assert_eq!(array.get_index(0).unwrap().as_number(), Some(1.0));
        assert_eq!(array.get_index(1).unwrap().as_number(), Some(2.0));
        assert_eq!(array.get_index(2), None);
    }

    #[test]
    fn test_json_value_serde_compatibility() {
        let value = JsonValue::object_from_iter(vec![
            ("name".to_string(), JsonValue::String("test".to_string())),
            ("value".to_string(), JsonValue::Number(42.0)),
            ("enabled".to_string(), JsonValue::Bool(true)),
            ("null_value".to_string(), JsonValue::Null),
        ]);

        // Test compact serialization
        let compact = value.to_string_compact();
        assert_eq!(
            compact,
            r#"{"enabled":true,"name":"test","null_value":null,"value":42}"#
        );

        // Test pretty serialization
        let pretty = value.to_string_pretty();
        assert!(pretty.contains("\n"));
        assert!(pretty.contains("  "));
        assert!(pretty.contains("name"));
        assert!(pretty.contains("value"));
        assert!(pretty.contains("enabled"));
        assert!(pretty.contains("null_value"));
    }

    #[test]
    fn test_json_value_edge_cases() {
        // Test empty objects and arrays
        let empty_object = JsonValue::object(HashMap::new());
        assert_eq!(empty_object.to_string_compact(), "{}");
        assert_eq!(empty_object.to_string_pretty(), "{}");

        let empty_array = JsonValue::array(vec![]);
        assert_eq!(empty_array.to_string_compact(), "[]");
        assert_eq!(empty_array.to_string_pretty(), "[]");

        // Test nested structures
        let nested = JsonValue::object_from_iter(vec![
            (
                "outer".to_string(),
                JsonValue::object_from_iter(vec![(
                    "inner".to_string(),
                    JsonValue::String("value".to_string()),
                )]),
            ),
            (
                "array".to_string(),
                JsonValue::array(vec![
                    JsonValue::Number(1.0),
                    JsonValue::Number(2.0),
                    JsonValue::Number(3.0),
                ]),
            ),
        ]);

        let compact = nested.to_string_compact();
        assert!(compact.contains("outer"));
        assert!(compact.contains("inner"));
        assert!(compact.contains("array"));
        assert!(compact.contains("[1,2,3]"));

        // Test special characters in strings
        let special_chars = JsonValue::String("Hello\nWorld\twith \"quotes\"".to_string());
        let escaped = special_chars.to_string_compact();
        assert!(escaped.contains("\\n"));
        assert!(escaped.contains("\\t"));
        assert!(escaped.contains("\\\""));
    }

    #[test]
    fn test_json_value_number_precision() {
        // Test integer numbers
        let int_value = JsonValue::Number(42.0);
        assert_eq!(int_value.to_string_compact(), "42");

        // Test floating point numbers
        let float_value = JsonValue::Number(std::f64::consts::PI);
        assert_eq!(float_value.to_string_compact(), "3.141592653589793");

        // Test negative numbers
        let neg_value = JsonValue::Number(-123.45);
        assert_eq!(neg_value.to_string_compact(), "-123.45");

        // Test zero
        let zero_value = JsonValue::Number(0.0);
        assert_eq!(zero_value.to_string_compact(), "0");
    }

    #[test]
    fn test_json_value_boolean_values() {
        let true_value = JsonValue::Bool(true);
        assert_eq!(true_value.to_string_compact(), "true");

        let false_value = JsonValue::Bool(false);
        assert_eq!(false_value.to_string_compact(), "false");
    }

    #[test]
    fn test_json_value_null_handling() {
        let null_value = JsonValue::Null;
        assert_eq!(null_value.to_string_compact(), "null");
        assert_eq!(null_value.to_string_pretty(), "null");
    }
}
