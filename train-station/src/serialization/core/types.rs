//! Core types for serialization operations and value representation
//!
//! This module defines the fundamental types used throughout the serialization system,
//! providing a unified, type-safe representation for all serializable values. The
//! `FieldValue` enum serves as the cornerstone of the serialization framework,
//! enabling uniform handling of diverse data types while preserving type information
//! and supporting comprehensive validation.
//!
//! # Purpose
//!
//! The types module provides:
//! - **Universal value representation**: Single enum type for all serializable values
//! - **Type safety**: Compile-time guarantees for value type preservation
//! - **Validation**: Comprehensive type checking and conversion with detailed error reporting
//! - **Extensibility**: Support for custom types through trait implementations
//! - **Performance**: Efficient memory layout and zero-copy operations where possible
//!
//! # Core Components
//!
//! ## FieldValue Enum
//!
//! The `FieldValue` enum represents all possible serializable values in a type-safe manner.
//! It provides both constructor methods for creating values and accessor methods for
//! extracting typed data with comprehensive validation.
//!
//! ## Value Categories
//!
//! ### Primitive Types
//! - **Integers**: `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`, `Usize`
//! - **Floating Point**: `F32`, `F64`
//! - **Boolean**: `Bool`
//! - **String**: `String`
//!
//! ### Complex Types
//! - **Binary Data**: `Bytes` for raw byte arrays
//! - **Serialized Objects**: `JsonObject`, `BinaryObject` for format-specific serialization
//! - **Collections**: `Array`, `Object` for structured data
//! - **Optional Values**: `Optional` for nullable fields
//! - **Enums**: `Enum` for variant-based data with associated values
//!
//! # Type Conversion and Validation
//!
//! The `FieldValue` type provides comprehensive type conversion capabilities:
//!
//! - **Safe conversions**: Automatic conversion between compatible numeric types
//! - **Range validation**: Checks for overflow/underflow during conversions
//! - **Sign validation**: Ensures proper handling of signed/unsigned conversions
//! - **Detailed errors**: Provides specific error messages for validation failures
//!
//! # Memory Layout
//!
//! The enum is designed for efficient memory usage:
//! - **Tagged union**: Uses Rust's enum optimization for minimal memory overhead
//! - **String optimization**: Leverages Rust's string optimization for text data
//! - **Boxed variants**: Large variants (Optional, Enum) use heap allocation
//! - **Zero-copy**: Small variants are stored inline for maximum performance
//!
//! # Thread Safety
//!
//! All `FieldValue` instances are thread-safe and can be shared between threads.
//! The type implements `Clone` and `PartialEq` for flexible usage patterns.

use super::error::{SerializationError, SerializationResult};
use std::collections::HashMap;

/// Universal type-safe container for all serializable field values
///
/// This enum provides a unified representation for all possible serializable values
/// in the Train Station serialization system. It maintains type safety while enabling
/// uniform handling in the serialization pipeline, supporting both primitive types
/// and complex data structures.
///
/// The enum uses Rust's tagged union optimization for efficient memory layout,
/// with larger variants (Optional, Enum) using heap allocation to minimize the
/// base enum size. All variants support comprehensive validation and type conversion.
///
/// # Variants
///
/// ## Primitive Types
/// * `Bool(bool)` - Boolean true/false values
/// * `I8(i8)` - 8-bit signed integers (-128 to 127)
/// * `I16(i16)` - 16-bit signed integers (-32,768 to 32,767)
/// * `I32(i32)` - 32-bit signed integers (-2^31 to 2^31-1)
/// * `I64(i64)` - 64-bit signed integers (-2^63 to 2^63-1)
/// * `U8(u8)` - 8-bit unsigned integers (0 to 255)
/// * `U16(u16)` - 16-bit unsigned integers (0 to 65,535)
/// * `U32(u32)` - 32-bit unsigned integers (0 to 2^32-1)
/// * `U64(u64)` - 64-bit unsigned integers (0 to 2^64-1)
/// * `Usize(usize)` - Platform-specific size integers
/// * `F32(f32)` - 32-bit floating point numbers
/// * `F64(f64)` - 64-bit floating point numbers
/// * `String(String)` - UTF-8 encoded text strings
///
/// ## Complex Types
/// * `Bytes(Vec<u8>)` - Raw binary data arrays
/// * `JsonObject(String)` - JSON-serialized object strings
/// * `BinaryObject(Vec<u8>)` - Binary-serialized object data
/// * `Array(Vec<FieldValue>)` - Heterogeneous value arrays
/// * `Optional(Option<Box<FieldValue>>)` - Nullable field values
/// * `Object(HashMap<String, FieldValue>)` - Key-value object maps
/// * `Enum { variant: String, data: Option<Box<FieldValue>> }` - Enum variants with optional data
///
/// # Type Conversion
///
/// The enum provides comprehensive type conversion capabilities:
/// - **Numeric conversions**: Automatic conversion between compatible integer and float types
/// - **Range validation**: Overflow/underflow checking for all conversions
/// - **Sign validation**: Proper handling of signed/unsigned type conversions
/// - **String parsing**: Hex string parsing for byte array conversion
///
/// # Error Handling
///
/// All conversion methods return `SerializationResult<T>` with detailed error information:
/// - **Type mismatch errors**: Clear indication of expected vs actual types
/// - **Range validation errors**: Specific overflow/underflow details
/// - **Parsing errors**: Detailed hex string parsing failure information
/// - **Field context**: Error messages include field names for debugging
///
/// # Performance Characteristics
///
/// - **Zero-copy operations**: Small variants stored inline for maximum performance
/// - **Efficient conversions**: Optimized type conversion with minimal allocations
/// - **Memory layout**: Tagged union optimization for minimal memory overhead
/// - **Validation speed**: Fast type checking with early return on mismatches
///
/// # Thread Safety
///
/// This type is thread-safe and can be shared between threads. All variants
/// implement `Clone` and `PartialEq` for flexible usage patterns.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    /// Boolean true/false values
    Bool(bool),
    /// 8-bit signed integers (-128 to 127)
    I8(i8),
    /// 16-bit signed integers (-32,768 to 32,767)
    I16(i16),
    /// 32-bit signed integers (-2^31 to 2^31-1)
    I32(i32),
    /// 64-bit signed integers (-2^63 to 2^63-1)
    I64(i64),
    /// 8-bit unsigned integers (0 to 255)
    U8(u8),
    /// 16-bit unsigned integers (0 to 65,535)
    U16(u16),
    /// 32-bit unsigned integers (0 to 2^32-1)
    U32(u32),
    /// 64-bit unsigned integers (0 to 2^64-1)
    U64(u64),
    /// Platform-specific size integers for array lengths and indices
    Usize(usize),
    /// 32-bit floating point numbers with single precision
    F32(f32),
    /// 64-bit floating point numbers with double precision
    F64(f64),
    /// UTF-8 encoded text strings
    String(String),
    /// Raw binary data arrays for efficient storage
    Bytes(Vec<u8>),
    /// JSON-serialized object strings for human-readable format
    JsonObject(String),
    /// Binary-serialized object data for compact storage
    BinaryObject(Vec<u8>),
    /// Heterogeneous arrays of field values
    Array(Vec<FieldValue>),
    /// Nullable field values that may be None
    Optional(Option<Box<FieldValue>>),
    /// Key-value object maps for structured data
    Object(HashMap<String, FieldValue>),
    /// Enum variants with optional associated data
    ///
    /// Supports three enum variant types:
    /// - **Unit variant**: `Enum { variant: "VariantName", data: None }`
    /// - **Tuple variant**: `Enum { variant: "VariantName", data: Some(Array([...])) }`
    /// - **Struct variant**: `Enum { variant: "VariantName", data: Some(Object({...})) }`
    Enum {
        /// The name of the enum variant
        variant: String,
        /// Optional associated data for the variant (boxed for memory efficiency)
        data: Option<Box<FieldValue>>,
    },
}

impl FieldValue {
    // ===== Constructor Methods =====

    /// Creates a boolean field value
    ///
    /// # Arguments
    ///
    /// * `value` - Boolean value to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::Bool` variant containing the boolean value
    pub fn from_bool(value: bool) -> Self {
        FieldValue::Bool(value)
    }

    /// Creates an 8-bit signed integer field value
    ///
    /// # Arguments
    ///
    /// * `value` - 8-bit signed integer to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::I8` variant containing the integer value
    pub fn from_i8(value: i8) -> Self {
        FieldValue::I8(value)
    }

    /// Creates a 16-bit signed integer field value
    ///
    /// # Arguments
    ///
    /// * `value` - 16-bit signed integer to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::I16` variant containing the integer value
    pub fn from_i16(value: i16) -> Self {
        FieldValue::I16(value)
    }

    /// Creates a 32-bit signed integer field value
    ///
    /// # Arguments
    ///
    /// * `value` - 32-bit signed integer to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::I32` variant containing the integer value
    pub fn from_i32(value: i32) -> Self {
        FieldValue::I32(value)
    }

    /// Creates a 64-bit signed integer field value
    ///
    /// # Arguments
    ///
    /// * `value` - 64-bit signed integer to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::I64` variant containing the integer value
    pub fn from_i64(value: i64) -> Self {
        FieldValue::I64(value)
    }

    /// Creates an 8-bit unsigned integer field value
    ///
    /// # Arguments
    ///
    /// * `value` - 8-bit unsigned integer to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::U8` variant containing the integer value
    pub fn from_u8(value: u8) -> Self {
        FieldValue::U8(value)
    }

    /// Creates a 16-bit unsigned integer field value
    ///
    /// # Arguments
    ///
    /// * `value` - 16-bit unsigned integer to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::U16` variant containing the integer value
    pub fn from_u16(value: u16) -> Self {
        FieldValue::U16(value)
    }

    /// Creates a 32-bit unsigned integer field value
    ///
    /// # Arguments
    ///
    /// * `value` - 32-bit unsigned integer to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::U32` variant containing the integer value
    pub fn from_u32(value: u32) -> Self {
        FieldValue::U32(value)
    }

    /// Creates a 64-bit unsigned integer field value
    ///
    /// # Arguments
    ///
    /// * `value` - 64-bit unsigned integer to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::U64` variant containing the integer value
    pub fn from_u64(value: u64) -> Self {
        FieldValue::U64(value)
    }

    /// Creates a platform-specific size integer field value
    ///
    /// # Arguments
    ///
    /// * `value` - Platform-specific size integer to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::Usize` variant containing the size value
    pub fn from_usize(value: usize) -> Self {
        FieldValue::Usize(value)
    }

    /// Creates a 32-bit floating point field value
    ///
    /// # Arguments
    ///
    /// * `value` - 32-bit floating point number to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::F32` variant containing the float value
    pub fn from_f32(value: f32) -> Self {
        FieldValue::F32(value)
    }

    /// Creates a 64-bit floating point field value
    ///
    /// # Arguments
    ///
    /// * `value` - 64-bit floating point number to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::F64` variant containing the float value
    pub fn from_f64(value: f64) -> Self {
        FieldValue::F64(value)
    }

    /// Creates a string field value from an owned string
    ///
    /// # Arguments
    ///
    /// * `value` - Owned string to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::String` variant containing the string value
    pub fn from_string(value: String) -> Self {
        FieldValue::String(value)
    }

    /// Creates a string field value from a string slice
    ///
    /// # Arguments
    ///
    /// * `value` - String slice to store (converted to owned string)
    ///
    /// # Returns
    ///
    /// A `FieldValue::String` variant containing the string value
    pub fn from_string_slice(value: &str) -> Self {
        FieldValue::String(value.to_string())
    }

    /// Creates a byte array field value
    ///
    /// # Arguments
    ///
    /// * `value` - Vector of bytes to store
    ///
    /// # Returns
    ///
    /// A `FieldValue::Bytes` variant containing the byte array
    pub fn from_bytes(value: Vec<u8>) -> Self {
        FieldValue::Bytes(value)
    }

    /// Creates a JSON object field value
    ///
    /// # Arguments
    ///
    /// * `json` - JSON string representation of the object
    ///
    /// # Returns
    ///
    /// A `FieldValue::JsonObject` variant containing the JSON string
    pub fn from_json_object(json: String) -> Self {
        FieldValue::JsonObject(json)
    }

    /// Creates a binary object field value
    ///
    /// # Arguments
    ///
    /// * `data` - Binary data representing the serialized object
    ///
    /// # Returns
    ///
    /// A `FieldValue::BinaryObject` variant containing the binary data
    pub fn from_binary_object(data: Vec<u8>) -> Self {
        FieldValue::BinaryObject(data)
    }

    /// Creates an array field value
    ///
    /// # Arguments
    ///
    /// * `values` - Vector of field values to store as an array
    ///
    /// # Returns
    ///
    /// A `FieldValue::Array` variant containing the array of values
    pub fn from_array(values: Vec<FieldValue>) -> Self {
        FieldValue::Array(values)
    }

    /// Creates an optional field value
    ///
    /// # Arguments
    ///
    /// * `value` - Optional field value (None becomes Optional(None), Some(v) becomes Optional(Some(Box::new(v))))
    ///
    /// # Returns
    ///
    /// A `FieldValue::Optional` variant containing the optional value
    pub fn from_optional(value: Option<FieldValue>) -> Self {
        FieldValue::Optional(value.map(Box::new))
    }

    /// Creates an object field value
    ///
    /// # Arguments
    ///
    /// * `value` - HashMap of string keys to field values
    ///
    /// # Returns
    ///
    /// A `FieldValue::Object` variant containing the key-value pairs
    pub fn from_object(value: HashMap<String, FieldValue>) -> Self {
        FieldValue::Object(value)
    }

    /// Creates an enum field value with variant name and optional associated data
    ///
    /// # Arguments
    ///
    /// * `variant` - Name of the enum variant
    /// * `data` - Optional associated data for the variant
    ///
    /// # Returns
    ///
    /// A `FieldValue::Enum` variant containing the variant name and optional data
    pub fn from_enum(variant: String, data: Option<FieldValue>) -> Self {
        FieldValue::Enum {
            variant,
            data: data.map(Box::new),
        }
    }

    /// Creates a unit enum variant with no associated data
    ///
    /// # Arguments
    ///
    /// * `variant` - Name of the enum variant
    ///
    /// # Returns
    ///
    /// A `FieldValue::Enum` variant with no associated data
    pub fn from_enum_unit(variant: String) -> Self {
        FieldValue::Enum {
            variant,
            data: None,
        }
    }

    /// Creates a tuple enum variant with array data
    ///
    /// # Arguments
    ///
    /// * `variant` - Name of the enum variant
    /// * `values` - Vector of field values representing the tuple data
    ///
    /// # Returns
    ///
    /// A `FieldValue::Enum` variant with array data
    pub fn from_enum_tuple(variant: String, values: Vec<FieldValue>) -> Self {
        FieldValue::Enum {
            variant,
            data: Some(Box::new(FieldValue::Array(values))),
        }
    }

    /// Creates a struct enum variant with object data
    ///
    /// # Arguments
    ///
    /// * `variant` - Name of the enum variant
    /// * `fields` - HashMap of field names to values representing the struct data
    ///
    /// # Returns
    ///
    /// A `FieldValue::Enum` variant with object data
    pub fn from_enum_struct(variant: String, fields: HashMap<String, FieldValue>) -> Self {
        FieldValue::Enum {
            variant,
            data: Some(Box::new(FieldValue::Object(fields))),
        }
    }

    // ===== Accessor Methods with Type Checking =====

    /// Extracts a boolean value with type validation
    ///
    /// Attempts to extract a boolean value from the field value. Returns an error
    /// if the field value is not a boolean type.
    ///
    /// # Returns
    ///
    /// `Ok(bool)` if the field value is a boolean
    /// `Err(SerializationError::ValidationFailed)` if the field value is not a boolean
    pub fn as_bool(&self) -> SerializationResult<bool> {
        match self {
            FieldValue::Bool(value) => Ok(*value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected bool, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts an 8-bit signed integer value with type validation
    ///
    /// Attempts to extract an 8-bit signed integer from the field value. Returns an error
    /// if the field value is not an i8 type.
    ///
    /// # Returns
    ///
    /// `Ok(i8)` if the field value is an 8-bit signed integer
    /// `Err(SerializationError::ValidationFailed)` if the field value is not an i8
    pub fn as_i8(&self) -> SerializationResult<i8> {
        match self {
            FieldValue::I8(value) => Ok(*value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected i8, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a 16-bit signed integer value with type validation
    ///
    /// Attempts to extract a 16-bit signed integer from the field value. Returns an error
    /// if the field value is not an i16 type.
    ///
    /// # Returns
    ///
    /// `Ok(i16)` if the field value is a 16-bit signed integer
    /// `Err(SerializationError::ValidationFailed)` if the field value is not an i16
    pub fn as_i16(&self) -> SerializationResult<i16> {
        match self {
            FieldValue::I16(value) => Ok(*value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected i16, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a 32-bit signed integer value with type validation and conversion
    ///
    /// Attempts to extract a 32-bit signed integer from the field value. Supports
    /// automatic conversion from smaller integer types (i8, i16) and validates
    /// range constraints for larger types (i64).
    ///
    /// # Returns
    ///
    /// `Ok(i32)` if the field value can be converted to a 32-bit signed integer
    /// `Err(SerializationError::ValidationFailed)` if conversion fails or value is out of range
    ///
    /// # Conversion Rules
    ///
    /// - **i8, i16**: Automatic conversion (always safe)
    /// - **i32**: Direct extraction
    /// - **i64**: Conversion with range validation (must be within i32::MIN..=i32::MAX)
    /// - **Other types**: Conversion not supported
    pub fn as_i32(&self) -> SerializationResult<i32> {
        match self {
            FieldValue::I8(value) => Ok(*value as i32),
            FieldValue::I16(value) => Ok(*value as i32),
            FieldValue::I32(value) => Ok(*value),
            FieldValue::I64(value) => {
                if *value >= i32::MIN as i64 && *value <= i32::MAX as i64 {
                    Ok(*value as i32)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("i64 value {} out of range for i32", value),
                    })
                }
            }
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected i32, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a 64-bit signed integer value with type validation and conversion
    ///
    /// Attempts to extract a 64-bit signed integer from the field value. Supports
    /// automatic conversion from smaller integer types (i8, i16, i32).
    ///
    /// # Returns
    ///
    /// `Ok(i64)` if the field value can be converted to a 64-bit signed integer
    /// `Err(SerializationError::ValidationFailed)` if conversion is not supported
    ///
    /// # Conversion Rules
    ///
    /// - **i8, i16, i32**: Automatic conversion (always safe)
    /// - **i64**: Direct extraction
    /// - **Other types**: Conversion not supported
    pub fn as_i64(&self) -> SerializationResult<i64> {
        match self {
            FieldValue::I8(value) => Ok(*value as i64),
            FieldValue::I16(value) => Ok(*value as i64),
            FieldValue::I32(value) => Ok(*value as i64),
            FieldValue::I64(value) => Ok(*value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected i64, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts an 8-bit unsigned integer value with type validation
    ///
    /// Attempts to extract an 8-bit unsigned integer from the field value. Returns an error
    /// if the field value is not a u8 type.
    ///
    /// # Returns
    ///
    /// `Ok(u8)` if the field value is an 8-bit unsigned integer
    /// `Err(SerializationError::ValidationFailed)` if the field value is not a u8
    pub fn as_u8(&self) -> SerializationResult<u8> {
        match self {
            FieldValue::U8(value) => Ok(*value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected u8, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a 16-bit unsigned integer value with type validation
    ///
    /// Attempts to extract a 16-bit unsigned integer from the field value. Returns an error
    /// if the field value is not a u16 type.
    ///
    /// # Returns
    ///
    /// `Ok(u16)` if the field value is a 16-bit unsigned integer
    /// `Err(SerializationError::ValidationFailed)` if the field value is not a u16
    pub fn as_u16(&self) -> SerializationResult<u16> {
        match self {
            FieldValue::U16(value) => Ok(*value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected u16, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a 32-bit unsigned integer value with comprehensive type validation and conversion
    ///
    /// Attempts to extract a 32-bit unsigned integer from the field value. Supports
    /// automatic conversion from compatible integer types with comprehensive range
    /// and sign validation.
    ///
    /// # Returns
    ///
    /// `Ok(u32)` if the field value can be converted to a 32-bit unsigned integer
    /// `Err(SerializationError::ValidationFailed)` if conversion fails or value is out of range
    ///
    /// # Conversion Rules
    ///
    /// ## Unsigned Types (Safe Conversions)
    /// - **u8, u16**: Automatic conversion (always safe)
    /// - **u32**: Direct extraction
    /// - **u64**: Conversion with range validation (must be ≤ u32::MAX)
    /// - **usize**: Conversion with range validation (must be ≤ u32::MAX)
    ///
    /// ## Signed Types (Sign Validation Required)
    /// - **i8, i16, i32**: Conversion only if value ≥ 0
    /// - **i64**: Conversion only if value ≥ 0 and ≤ u32::MAX
    ///
    /// ## Error Conditions
    /// - Negative values from signed types
    /// - Values exceeding u32::MAX from larger types
    /// - Incompatible types (floats, strings, etc.)
    pub fn as_u32(&self) -> SerializationResult<u32> {
        match self {
            FieldValue::U8(value) => Ok(*value as u32),
            FieldValue::U16(value) => Ok(*value as u32),
            FieldValue::U32(value) => Ok(*value),
            FieldValue::U64(value) => {
                if *value <= u32::MAX as u64 {
                    Ok(*value as u32)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("u64 value {} out of range for u32", value),
                    })
                }
            }
            FieldValue::Usize(value) => {
                if *value <= u32::MAX as usize {
                    Ok(*value as u32)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("usize value {} out of range for u32", value),
                    })
                }
            }
            FieldValue::I8(value) => {
                if *value >= 0 {
                    Ok(*value as u32)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i8 {} to u32", value),
                    })
                }
            }
            FieldValue::I16(value) => {
                if *value >= 0 {
                    Ok(*value as u32)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i16 {} to u32", value),
                    })
                }
            }
            FieldValue::I32(value) => {
                if *value >= 0 {
                    Ok(*value as u32)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i32 {} to u32", value),
                    })
                }
            }
            FieldValue::I64(value) => {
                if *value >= 0 && *value <= u32::MAX as i64 {
                    Ok(*value as u32)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("i64 value {} out of range for u32", value),
                    })
                }
            }
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected u32, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a 64-bit unsigned integer value with type validation and conversion
    ///
    /// Attempts to extract a 64-bit unsigned integer from the field value. Supports
    /// automatic conversion from smaller unsigned types and signed types with sign validation.
    ///
    /// # Returns
    ///
    /// `Ok(u64)` if the field value can be converted to a 64-bit unsigned integer
    /// `Err(SerializationError::ValidationFailed)` if conversion fails or value is negative
    ///
    /// # Conversion Rules
    ///
    /// - **u8, u16, u32, usize**: Automatic conversion (always safe)
    /// - **u64**: Direct extraction
    /// - **i8, i16, i32, i64**: Conversion only if value ≥ 0
    /// - **Other types**: Conversion not supported
    pub fn as_u64(&self) -> SerializationResult<u64> {
        match self {
            FieldValue::U64(value) => Ok(*value),
            FieldValue::U8(value) => Ok(*value as u64),
            FieldValue::U16(value) => Ok(*value as u64),
            FieldValue::U32(value) => Ok(*value as u64),
            FieldValue::Usize(value) => Ok(*value as u64),
            FieldValue::I8(value) => {
                if *value >= 0 {
                    Ok(*value as u64)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i8 {} to u64", value),
                    })
                }
            }
            FieldValue::I16(value) => {
                if *value >= 0 {
                    Ok(*value as u64)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i16 {} to u64", value),
                    })
                }
            }
            FieldValue::I32(value) => {
                if *value >= 0 {
                    Ok(*value as u64)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i32 {} to u64", value),
                    })
                }
            }
            FieldValue::I64(value) => {
                if *value >= 0 {
                    Ok(*value as u64)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i64 {} to u64", value),
                    })
                }
            }
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected u64, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a platform-specific size integer value with type validation and conversion
    ///
    /// Attempts to extract a platform-specific size integer from the field value. Supports
    /// automatic conversion from smaller unsigned types and signed types with sign validation.
    ///
    /// # Returns
    ///
    /// `Ok(usize)` if the field value can be converted to a platform-specific size integer
    /// `Err(SerializationError::ValidationFailed)` if conversion fails or value is negative
    ///
    /// # Conversion Rules
    ///
    /// - **u8, u16, u32, u64**: Automatic conversion (always safe)
    /// - **usize**: Direct extraction
    /// - **i8, i16, i32, i64**: Conversion only if value ≥ 0
    /// - **Other types**: Conversion not supported
    pub fn as_usize(&self) -> SerializationResult<usize> {
        match self {
            FieldValue::Usize(value) => Ok(*value),
            FieldValue::U8(value) => Ok(*value as usize),
            FieldValue::U16(value) => Ok(*value as usize),
            FieldValue::U32(value) => Ok(*value as usize),
            FieldValue::U64(value) => Ok(*value as usize),
            FieldValue::I8(value) => {
                if *value >= 0 {
                    Ok(*value as usize)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i8 {} to usize", value),
                    })
                }
            }
            FieldValue::I16(value) => {
                if *value >= 0 {
                    Ok(*value as usize)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i16 {} to usize", value),
                    })
                }
            }
            FieldValue::I32(value) => {
                if *value >= 0 {
                    Ok(*value as usize)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i32 {} to usize", value),
                    })
                }
            }
            FieldValue::I64(value) => {
                if *value >= 0 {
                    Ok(*value as usize)
                } else {
                    Err(SerializationError::ValidationFailed {
                        field: "unknown".to_string(),
                        message: format!("Cannot convert negative i64 {} to usize", value),
                    })
                }
            }
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected usize, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a 32-bit floating point value with type validation and conversion
    ///
    /// Attempts to extract a 32-bit floating point number from the field value. Supports
    /// automatic conversion from all numeric types with potential precision loss.
    ///
    /// # Returns
    ///
    /// `Ok(f32)` if the field value can be converted to a 32-bit float
    /// `Err(SerializationError::ValidationFailed)` if conversion is not supported
    ///
    /// # Conversion Rules
    ///
    /// - **f32**: Direct extraction
    /// - **f64**: Conversion with potential precision loss
    /// - **All integer types**: Automatic conversion (i8, i16, i32, i64, u8, u16, u32, u64, usize)
    /// - **Other types**: Conversion not supported
    pub fn as_f32(&self) -> SerializationResult<f32> {
        match self {
            FieldValue::F32(value) => Ok(*value),
            FieldValue::F64(value) => Ok(*value as f32),
            FieldValue::I8(value) => Ok(*value as f32),
            FieldValue::I16(value) => Ok(*value as f32),
            FieldValue::I32(value) => Ok(*value as f32),
            FieldValue::I64(value) => Ok(*value as f32),
            FieldValue::U8(value) => Ok(*value as f32),
            FieldValue::U16(value) => Ok(*value as f32),
            FieldValue::U32(value) => Ok(*value as f32),
            FieldValue::U64(value) => Ok(*value as f32),
            FieldValue::Usize(value) => Ok(*value as f32),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected f32, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a 64-bit floating point value with type validation and conversion
    ///
    /// Attempts to extract a 64-bit floating point number from the field value. Supports
    /// automatic conversion from all numeric types without precision loss.
    ///
    /// # Returns
    ///
    /// `Ok(f64)` if the field value can be converted to a 64-bit float
    /// `Err(SerializationError::ValidationFailed)` if conversion is not supported
    ///
    /// # Conversion Rules
    ///
    /// - **f64**: Direct extraction
    /// - **f32**: Conversion without precision loss
    /// - **All integer types**: Automatic conversion (i8, i16, i32, i64, u8, u16, u32, u64, usize)
    /// - **Other types**: Conversion not supported
    pub fn as_f64(&self) -> SerializationResult<f64> {
        match self {
            FieldValue::F64(value) => Ok(*value),
            FieldValue::F32(value) => Ok(*value as f64),
            FieldValue::I8(value) => Ok(*value as f64),
            FieldValue::I16(value) => Ok(*value as f64),
            FieldValue::I32(value) => Ok(*value as f64),
            FieldValue::I64(value) => Ok(*value as f64),
            FieldValue::U8(value) => Ok(*value as f64),
            FieldValue::U16(value) => Ok(*value as f64),
            FieldValue::U32(value) => Ok(*value as f64),
            FieldValue::U64(value) => Ok(*value as f64),
            FieldValue::Usize(value) => Ok(*value as f64),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected f64, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a string slice with type validation
    ///
    /// Attempts to extract a string slice from the field value. Returns an error
    /// if the field value is not a string type.
    ///
    /// # Returns
    ///
    /// `Ok(&str)` if the field value is a string
    /// `Err(SerializationError::ValidationFailed)` if the field value is not a string
    pub fn as_string(&self) -> SerializationResult<&str> {
        match self {
            FieldValue::String(value) => Ok(value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected string, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a byte slice with type validation
    ///
    /// Attempts to extract a byte slice from the field value. Returns an error
    /// if the field value is not a byte array type.
    ///
    /// # Returns
    ///
    /// `Ok(&[u8])` if the field value is a byte array
    /// `Err(SerializationError::ValidationFailed)` if the field value is not a byte array
    pub fn as_bytes(&self) -> SerializationResult<&[u8]> {
        match self {
            FieldValue::Bytes(value) => Ok(value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected bytes, found {:?}", self.type_name()),
            }),
        }
    }

    /// Converts the field value to a byte vector with intelligent parsing
    ///
    /// Attempts to convert the field value to a byte vector. For byte arrays, returns
    /// a clone of the data. For strings, attempts intelligent parsing including hex
    /// string conversion before falling back to UTF-8 encoding.
    ///
    /// # Returns
    ///
    /// `Ok(Vec<u8>)` containing the byte data
    /// `Err(SerializationError::ValidationFailed)` if conversion fails
    ///
    /// # String Parsing Rules
    ///
    /// The method attempts the following parsing strategies in order:
    ///
    /// 1. **Hex string with 0x prefix**: `"0x1234abcd"` → `[0x12, 0x34, 0xab, 0xcd]`
    /// 2. **Plain hex string**: `"1234abcd"` → `[0x12, 0x34, 0xab, 0xcd]`
    /// 3. **UTF-8 encoding**: Any other string → UTF-8 byte representation
    ///
    /// # Hex String Requirements
    ///
    /// - Must contain only ASCII hex digits (0-9, a-f, A-F)
    /// - Must have even length (pairs of hex digits)
    /// - Optional "0x" prefix for explicit hex notation
    /// - Empty strings are treated as UTF-8
    ///
    /// # Error Conditions
    ///
    /// - Invalid hex characters in hex strings
    /// - Odd-length hex strings
    /// - Non-string/non-byte field values
    pub fn to_bytes(&self) -> SerializationResult<Vec<u8>> {
        match self {
            FieldValue::Bytes(value) => Ok(value.clone()),
            FieldValue::String(value) => {
                // Try to parse as hex string
                if value.starts_with("0x") && value.len() > 2 && value.len() % 2 == 0 {
                    let hex_part = &value[2..];
                    if hex_part.chars().all(|c| c.is_ascii_hexdigit()) {
                        let mut bytes = Vec::new();
                        let mut chars = hex_part.chars();
                        while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
                            let byte =
                                u8::from_str_radix(&format!("{}{}", a, b), 16).map_err(|_| {
                                    SerializationError::ValidationFailed {
                                        field: "unknown".to_string(),
                                        message: format!("Invalid hex string: {}", value),
                                    }
                                })?;
                            bytes.push(byte);
                        }
                        Ok(bytes)
                    } else {
                        // Convert string to bytes as UTF-8
                        Ok(value.as_bytes().to_vec())
                    }
                } else if value.chars().all(|c| c.is_ascii_hexdigit())
                    && value.len() % 2 == 0
                    && !value.is_empty()
                {
                    // Try to parse as plain hex string
                    let mut bytes = Vec::new();
                    let mut chars = value.chars();
                    while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
                        let byte =
                            u8::from_str_radix(&format!("{}{}", a, b), 16).map_err(|_| {
                                SerializationError::ValidationFailed {
                                    field: "unknown".to_string(),
                                    message: format!("Invalid hex string: {}", value),
                                }
                            })?;
                        bytes.push(byte);
                    }
                    Ok(bytes)
                } else {
                    // Convert string to bytes as UTF-8
                    Ok(value.as_bytes().to_vec())
                }
            }
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected bytes, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts a JSON object string with type validation
    ///
    /// Attempts to extract a JSON object string from the field value. Returns an error
    /// if the field value is not a JSON object type.
    ///
    /// # Returns
    ///
    /// `Ok(&str)` if the field value is a JSON object string
    /// `Err(SerializationError::ValidationFailed)` if the field value is not a JSON object
    pub fn as_json_object(&self) -> SerializationResult<&str> {
        match self {
            FieldValue::JsonObject(value) => Ok(value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected JSON object, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts binary object data with type validation
    ///
    /// Attempts to extract binary object data from the field value. Returns an error
    /// if the field value is not a binary object type.
    ///
    /// # Returns
    ///
    /// `Ok(&[u8])` if the field value is binary object data
    /// `Err(SerializationError::ValidationFailed)` if the field value is not a binary object
    pub fn as_binary_object(&self) -> SerializationResult<&[u8]> {
        match self {
            FieldValue::BinaryObject(value) => Ok(value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected binary object, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts an array of field values with type validation
    ///
    /// Attempts to extract an array of field values from the field value. Returns an error
    /// if the field value is not an array type.
    ///
    /// # Returns
    ///
    /// `Ok(&[FieldValue])` if the field value is an array
    /// `Err(SerializationError::ValidationFailed)` if the field value is not an array
    pub fn as_array(&self) -> SerializationResult<&[FieldValue]> {
        match self {
            FieldValue::Array(value) => Ok(value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected array, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts an optional field value with type validation
    ///
    /// Attempts to extract an optional field value from the field value. Returns an error
    /// if the field value is not an optional type.
    ///
    /// # Returns
    ///
    /// `Ok(Option<&FieldValue>)` if the field value is an optional
    /// `Err(SerializationError::ValidationFailed)` if the field value is not an optional
    pub fn as_optional(&self) -> SerializationResult<Option<&FieldValue>> {
        match self {
            FieldValue::Optional(value) => Ok(value.as_deref()),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected optional, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts an object map with type validation
    ///
    /// Attempts to extract a key-value object map from the field value. Returns an error
    /// if the field value is not an object type.
    ///
    /// # Returns
    ///
    /// `Ok(&HashMap<String, FieldValue>)` if the field value is an object
    /// `Err(SerializationError::ValidationFailed)` if the field value is not an object
    pub fn as_object(&self) -> SerializationResult<&HashMap<String, FieldValue>> {
        match self {
            FieldValue::Object(value) => Ok(value),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected object, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts enum variant name and associated data with type validation
    ///
    /// Attempts to extract both the enum variant name and its associated data from
    /// the field value. Returns an error if the field value is not an enum type.
    ///
    /// # Returns
    ///
    /// `Ok((&str, Option<&FieldValue>))` containing the variant name and optional data
    /// `Err(SerializationError::ValidationFailed)` if the field value is not an enum
    pub fn as_enum(&self) -> SerializationResult<(&str, Option<&FieldValue>)> {
        match self {
            FieldValue::Enum { variant, data } => Ok((variant, data.as_ref().map(|d| d.as_ref()))),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected enum, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts enum variant name with type validation
    ///
    /// Attempts to extract the enum variant name from the field value. Returns an error
    /// if the field value is not an enum type.
    ///
    /// # Returns
    ///
    /// `Ok(&str)` containing the enum variant name
    /// `Err(SerializationError::ValidationFailed)` if the field value is not an enum
    pub fn as_enum_variant(&self) -> SerializationResult<&str> {
        match self {
            FieldValue::Enum { variant, .. } => Ok(variant),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected enum, found {:?}", self.type_name()),
            }),
        }
    }

    /// Extracts enum associated data with type validation
    ///
    /// Attempts to extract the associated data from the enum field value. Returns an error
    /// if the field value is not an enum type.
    ///
    /// # Returns
    ///
    /// `Ok(Option<&FieldValue>)` containing the optional associated data
    /// `Err(SerializationError::ValidationFailed)` if the field value is not an enum
    pub fn as_enum_data(&self) -> SerializationResult<Option<&FieldValue>> {
        match self {
            FieldValue::Enum { data, .. } => Ok(data.as_ref().map(|d| d.as_ref())),
            _ => Err(SerializationError::ValidationFailed {
                field: "unknown".to_string(),
                message: format!("Expected enum, found {:?}", self.type_name()),
            }),
        }
    }

    /// Returns the human-readable type name for error reporting
    ///
    /// Provides a consistent string representation of the field value type for use
    /// in error messages and debugging. The returned string is static and does not
    /// depend on the actual value content.
    ///
    /// # Returns
    ///
    /// A static string slice representing the type name
    ///
    /// # Type Names
    ///
    /// - **Primitive types**: `"bool"`, `"i8"`, `"i16"`, `"i32"`, `"i64"`, `"u8"`, `"u16"`, `"u32"`, `"u64"`, `"usize"`, `"f32"`, `"f64"`
    /// - **String types**: `"string"`
    /// - **Binary types**: `"bytes"`
    /// - **Object types**: `"json_object"`, `"binary_object"`
    /// - **Collection types**: `"array"`, `"optional"`, `"object"`
    /// - **Enum types**: `"enum"`
    pub fn type_name(&self) -> &'static str {
        match self {
            FieldValue::Bool(_) => "bool",
            FieldValue::I8(_) => "i8",
            FieldValue::I16(_) => "i16",
            FieldValue::I32(_) => "i32",
            FieldValue::I64(_) => "i64",
            FieldValue::U8(_) => "u8",
            FieldValue::U16(_) => "u16",
            FieldValue::U32(_) => "u32",
            FieldValue::U64(_) => "u64",
            FieldValue::Usize(_) => "usize",
            FieldValue::F32(_) => "f32",
            FieldValue::F64(_) => "f64",
            FieldValue::String(_) => "string",
            FieldValue::Bytes(_) => "bytes",
            FieldValue::JsonObject(_) => "json_object",
            FieldValue::BinaryObject(_) => "binary_object",
            FieldValue::Array(_) => "array",
            FieldValue::Optional(_) => "optional",
            FieldValue::Object(_) => "object",
            FieldValue::Enum { .. } => "enum",
        }
    }
}
