//! JSON-specific implementations for struct serialization
//!
//! This module contains the JSON-specific serialization and deserialization
//! implementations for the StructSerializer and StructDeserializer types.
//! These implementations handle the conversion between FieldValue and JSON format.

use super::{parse as parse_json, JsonValue};
use crate::serialization::core::{
    FieldValue, SerializationError, SerializationResult, StructDeserializer, StructSerializer,
};

/// Converts a StructSerializer to its JSON string representation
///
/// This function takes a StructSerializer containing field data and converts it
/// to a properly formatted JSON string. All field values are converted to their
/// appropriate JSON representations with proper escaping and formatting.
///
/// # Arguments
///
/// * `serializer` - The StructSerializer containing the field data to serialize
///
/// # Returns
///
/// `Ok(String)` containing the JSON representation on success
/// `Err(SerializationError)` if serialization fails
///
/// # Field Type Support
///
/// The function supports all FieldValue types including primitive types, strings,
/// binary data, arrays, objects, enums, and optional values. Each type is converted
/// to its appropriate JSON representation with proper formatting and escaping.
pub fn to_json_internal(serializer: StructSerializer) -> SerializationResult<String> {
    let mut json_fields = Vec::new();

    for (name, value) in serializer.fields {
        let json_value = field_value_to_json_static(value)?;
        json_fields.push(format!("\"{}\":{}", name, json_value));
    }

    Ok(format!("{{{}}}", json_fields.join(",")))
}

/// Converts a FieldValue to its JSON string representation
///
/// This function handles the conversion of individual FieldValue types to their
/// appropriate JSON string representations. It ensures proper formatting, escaping,
/// and type preservation for all supported value types.
///
/// # Arguments
///
/// * `value` - The FieldValue to convert to JSON
///
/// # Returns
///
/// `Ok(String)` containing the JSON representation on success
/// `Err(SerializationError)` if conversion fails
///
/// # Type-Specific Behavior
///
/// * **Primitive Types**: Direct string conversion with proper formatting
/// * **Strings**: JSON-escaped with quotes
/// * **Bytes**: Hex-encoded as strings
/// * **Arrays**: Comma-separated values in brackets
/// * **Objects**: Key-value pairs in braces
/// * **Enums**: Structured format with variant and data
/// * **Optional**: null for None, direct value for Some
fn field_value_to_json_static(value: FieldValue) -> SerializationResult<String> {
    match value {
        FieldValue::Bool(v) => Ok(v.to_string()),
        FieldValue::I8(v) => Ok(v.to_string()),
        FieldValue::I16(v) => Ok(v.to_string()),
        FieldValue::I32(v) => Ok(v.to_string()),
        FieldValue::I64(v) => Ok(v.to_string()),
        FieldValue::U8(v) => Ok(v.to_string()),
        FieldValue::U16(v) => Ok(v.to_string()),
        FieldValue::U32(v) => Ok(v.to_string()),
        FieldValue::U64(v) => Ok(v.to_string()),
        FieldValue::Usize(v) => Ok(v.to_string()),
        FieldValue::F32(v) => {
            // Ensure f32 values are serialized with decimal point to preserve float type
            if v.fract() == 0.0 {
                Ok(format!("{}.0", v))
            } else {
                Ok(v.to_string())
            }
        }
        FieldValue::F64(v) => {
            // Ensure f64 values are serialized with decimal point to preserve float type
            if v.fract() == 0.0 {
                Ok(format!("{}.0", v))
            } else {
                Ok(v.to_string())
            }
        }
        FieldValue::String(v) => Ok(format!("\"{}\"", super::escape::escape_string(&v))),
        FieldValue::Bytes(v) => {
            // Encode bytes as base64-like string for JSON
            let mut encoded = String::with_capacity(v.len() * 2);
            for b in v {
                encoded.push_str(&format!("{:02x}", b));
            }
            Ok(format!("\"{}\"", encoded))
        }
        FieldValue::JsonObject(v) => Ok(v),
        FieldValue::BinaryObject(v) => {
            // For JSON, convert binary objects to base64-like hex string
            let mut encoded = String::with_capacity(v.len() * 2);
            for b in v {
                encoded.push_str(&format!("{:02x}", b));
            }
            Ok(format!("\"{}\"", encoded))
        }
        FieldValue::Array(v) => {
            let elements: Result<Vec<String>, SerializationError> =
                v.into_iter().map(field_value_to_json_static).collect();
            Ok(format!("[{}]", elements?.join(",")))
        }
        FieldValue::Optional(v) => match v {
            Some(inner) => field_value_to_json_static(*inner),
            None => Ok("null".to_string()),
        },
        FieldValue::Object(v) => {
            let properties: Result<Vec<String>, SerializationError> = v
                .into_iter()
                .map(|(key, value): (String, FieldValue)| {
                    let json_value = field_value_to_json_static(value)?;
                    Ok(format!(
                        "\"{}\":{}",
                        super::escape::escape_string(&key),
                        json_value
                    ))
                })
                .collect();
            Ok(format!("{{{}}}", properties?.join(",")))
        }
        FieldValue::Enum { variant, data } => {
            // JSON format for enums: {"variant": "VariantName", "data": ... }
            // Always use full enum structure for consistency
            match data {
                Some(enum_data) => {
                    let data_json = field_value_to_json_static(*enum_data)?;
                    Ok(format!(
                        "{{\"variant\":\"{}\",\"data\":{}}}",
                        super::escape::escape_string(&variant),
                        data_json
                    ))
                }
                None => {
                    // Unit variant: use null for data
                    Ok(format!(
                        "{{\"variant\":\"{}\",\"data\":null}}",
                        super::escape::escape_string(&variant)
                    ))
                }
            }
        }
    }
}

/// Converts a JSON string to a StructDeserializer
///
/// This function parses a JSON string and creates a StructDeserializer containing
/// the field data. It handles conversion from JSON values to FieldValue types
/// with proper type detection and validation.
///
/// # Arguments
///
/// * `json` - The JSON string to parse and convert
///
/// # Returns
///
/// `Ok(StructDeserializer)` containing the parsed field data on success
/// `Err(SerializationError)` if parsing or conversion fails
///
/// # Type Detection
///
/// The function automatically detects and converts JSON types to appropriate FieldValue types.
/// Numbers are converted to the smallest fitting integer type with preference for usize
/// for positive values. Floats are preserved as f64. Strings remain as strings unless
/// they are hex-encoded with "0x" prefix, in which case they are converted to bytes.
///
/// # Error Handling
///
/// The function provides detailed error information for parsing issues including
/// invalid JSON syntax, non-object JSON structures, and type conversion failures.
pub fn from_json_internal(json: &str) -> SerializationResult<StructDeserializer> {
    let json_value = parse_json(json)?;

    let json_object = json_value.as_object().ok_or_else(|| {
        SerializationError::json_format("Expected JSON object at root".to_string(), None, None)
    })?;

    let mut fields = std::collections::HashMap::new();

    for (key, value) in json_object {
        let field_value = json_value_to_field_value(value)?;
        fields.insert(key.clone(), field_value);
    }

    Ok(StructDeserializer { fields })
}

/// Converts a JsonValue to its corresponding FieldValue representation
///
/// This function handles the conversion from JSON values to FieldValue types
/// with intelligent type detection and proper handling of special cases like
/// hex-encoded strings and enum structures.
///
/// # Arguments
///
/// * `json_value` - The JsonValue to convert to FieldValue
///
/// # Returns
///
/// `Ok(FieldValue)` containing the converted value on success
/// `Err(SerializationError)` if conversion fails
///
/// # Type Detection Rules
///
/// * **Numbers**: Integers are converted to the smallest fitting integer type,
///   with preference for usize for positive values. Floats are preserved as f64.
/// * **Strings**: Regular strings remain as strings. Hex strings with "0x" prefix
///   are converted to bytes.
/// * **Arrays**: Converted to FieldValue::Array with recursive conversion of elements
/// * **Objects**: Regular objects become FieldValue::Object. Objects with "variant"
///   and "data" fields are treated as enums.
/// * **Null**: Converted to FieldValue::Optional(None)
fn json_value_to_field_value(json_value: &JsonValue) -> SerializationResult<FieldValue> {
    match json_value {
        JsonValue::Null => Ok(FieldValue::Optional(None)),
        JsonValue::Bool(v) => Ok(FieldValue::Bool(*v)),
        JsonValue::Number(n) => {
            // Try to parse as different numeric types
            let n_val = *n;

            // Check if it's an integer
            if n_val.fract() == 0.0 {
                let i_val = n_val as i64;

                // For positive integers, prefer usize if they fit (for better compatibility with tensor shapes)
                if i_val >= 0 && i_val <= usize::MAX as i64 {
                    Ok(FieldValue::Usize(i_val as usize))
                } else if i_val >= i8::MIN as i64 && i_val <= i8::MAX as i64 {
                    Ok(FieldValue::I8(i_val as i8))
                } else if i_val >= i16::MIN as i64 && i_val <= i16::MAX as i64 {
                    Ok(FieldValue::I16(i_val as i16))
                } else if i_val >= i32::MIN as i64 && i_val <= i32::MAX as i64 {
                    Ok(FieldValue::I32(i_val as i32))
                } else {
                    Ok(FieldValue::I64(i_val))
                }
            } else {
                // It's a floating point number - always preserve as f64 to avoid precision loss
                // Users who want f32 should explicitly use f32 types
                Ok(FieldValue::F64(n_val))
            }
        }
        JsonValue::String(s) => {
            // JSON strings should remain as strings
            // Only convert to bytes if explicitly marked with a special prefix
            if s.starts_with("0x")
                && s.len() > 2
                && s[2..].len() % 2 == 0
                && s[2..].chars().all(|c| c.is_ascii_hexdigit())
            {
                // Handle explicit hex strings like "0x48656c6c6f"
                let hex_part = &s[2..];
                let mut bytes = Vec::new();
                let mut chars = hex_part.chars();
                while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
                    let byte = u8::from_str_radix(&format!("{}{}", a, b), 16).map_err(|_| {
                        SerializationError::json_format(
                            "Invalid hex string".to_string(),
                            None,
                            None,
                        )
                    })?;
                    bytes.push(byte);
                }
                Ok(FieldValue::Bytes(bytes))
            } else {
                // All other strings remain as strings
                // Note: Unit enum variants are serialized as plain strings and
                // cannot be automatically detected during deserialization without context.
                // They will need to be converted manually by the specific FromFieldValue implementation.
                Ok(FieldValue::String(s.clone()))
            }
        }
        JsonValue::Array(arr) => {
            let elements: Result<Vec<FieldValue>, SerializationError> =
                arr.iter().map(json_value_to_field_value).collect();
            Ok(FieldValue::Array(elements?))
        }
        JsonValue::Object(obj) => {
            // Check if this object represents an enum variant
            if obj.len() == 2 && obj.contains_key("variant") && obj.contains_key("data") {
                // This looks like an enum: {"variant": "VariantName", "data": ...}
                let variant = obj
                    .get("variant")
                    .and_then(|v| v.as_string())
                    .ok_or_else(|| {
                        SerializationError::json_format(
                            "Enum variant must be a string".to_string(),
                            None,
                            None,
                        )
                    })?;

                let data_json = obj.get("data").unwrap();
                let data_field_value = if data_json.is_null() {
                    // Unit variant with null data
                    None
                } else {
                    // Variant with actual data
                    Some(Box::new(json_value_to_field_value(data_json)?))
                };

                Ok(FieldValue::Enum {
                    variant: variant.to_string(),
                    data: data_field_value,
                })
            } else {
                // Regular object
                let mut object = std::collections::HashMap::new();
                for (key, value) in obj {
                    let field_value = json_value_to_field_value(value)?;
                    object.insert(key.clone(), field_value);
                }
                Ok(FieldValue::Object(object))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_serializer_to_json() {
        let serializer = StructSerializer::new()
            .field("name", &"test")
            .field("value", &42)
            .field("enabled", &true);

        let json_string = to_json_internal(serializer).unwrap();
        assert!(json_string.contains("\"name\""));
        assert!(json_string.contains("\"test\""));
        assert!(json_string.contains("\"value\""));
        assert!(json_string.contains("42"));
        assert!(json_string.contains("\"enabled\""));
        assert!(json_string.contains("true"));
    }

    #[test]
    fn test_struct_deserializer_from_json() {
        let json_string = r#"{"name": "test", "value": 42, "enabled": true}"#;
        let deserializer = from_json_internal(json_string).unwrap();

        assert!(deserializer.has_field("name"));
        assert!(deserializer.has_field("value"));
        assert!(deserializer.has_field("enabled"));
        assert!(!deserializer.has_field("missing"));
    }

    #[test]
    fn test_json_number_parsing() {
        // Test that JSON numbers are parsed as usize when appropriate
        let json_string = r#"{"value": 42}"#;
        let mut deserializer = from_json_internal(json_string).unwrap();
        let value: usize = deserializer.field("value").unwrap();
        assert_eq!(value, 42);

        // Test array of numbers
        let json_string = r#"{"values": [1, 2, 3]}"#;
        let mut deserializer = from_json_internal(json_string).unwrap();
        let values: Vec<usize> = deserializer.field("values").unwrap();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_field_value_to_json_static() {
        // Test primitive types
        assert_eq!(
            field_value_to_json_static(FieldValue::Bool(true)).unwrap(),
            "true"
        );
        assert_eq!(
            field_value_to_json_static(FieldValue::I32(42)).unwrap(),
            "42"
        );
        assert_eq!(
            field_value_to_json_static(FieldValue::String("hello".to_string())).unwrap(),
            "\"hello\""
        );

        // Test bytes encoding
        let bytes = FieldValue::Bytes(vec![0x01, 0x02, 0x03]);
        let json = field_value_to_json_static(bytes).unwrap();
        assert_eq!(json, "\"010203\"");
    }

    #[test]
    fn test_json_value_to_field_value() {
        // Test primitive types
        let bool_value = JsonValue::Bool(true);
        let field_value = json_value_to_field_value(&bool_value).unwrap();
        assert!(matches!(field_value, FieldValue::Bool(true)));

        let string_value = JsonValue::String("hello".to_string());
        let field_value = json_value_to_field_value(&string_value).unwrap();
        assert!(matches!(field_value, FieldValue::String(ref s) if s == "hello"));

        // Test hex-encoded bytes (with 0x prefix)
        let hex_string = JsonValue::String("0x010203".to_string());
        let field_value = json_value_to_field_value(&hex_string).unwrap();
        assert!(matches!(field_value, FieldValue::Bytes(ref b) if b == &vec![1, 2, 3]));

        // Test that plain hex-looking strings remain as strings
        let plain_hex = JsonValue::String("010203".to_string());
        let field_value = json_value_to_field_value(&plain_hex).unwrap();
        assert!(matches!(field_value, FieldValue::String(ref s) if s == "010203"));
    }

    #[test]
    fn test_struct_serialization_comprehensive() {
        // Test all field types
        let serializer = StructSerializer::new()
            .field("bool_field", &true)
            .field("i8_field", &42i8)
            .field("i16_field", &1234i16)
            .field("i32_field", &123456i32)
            .field("i64_field", &123456789i64)
            .field("u8_field", &255u8)
            .field("u16_field", &65535u16)
            .field("u32_field", &4294967295u32)
            .field("u64_field", &18446744073709551615u64)
            .field("usize_field", &42usize)
            .field("f32_field", &std::f32::consts::PI)
            .field("f64_field", &std::f64::consts::E)
            .field("string_field", &"Hello, World!".to_string())
            .field("bytes_field", &vec![0x01u8, 0x02u8, 0x03u8, 0x04u8])
            .field("json_field", &"{\"nested\": \"value\"}".to_string())
            .field("binary_field", &vec![0xFFu8, 0xFEu8, 0xFDu8, 0xFCu8]);

        let json_string = to_json_internal(serializer).unwrap();

        // Debug output
        println!("Generated JSON: {}", json_string);

        // Debug: Check what FieldValue types are being created
        let bytes: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04];
        let serializer_debug = StructSerializer::new().field("bytes_field", &bytes);
        println!("Bytes field value: {:?}", serializer_debug.fields[0].1);

        // Verify all fields are present
        assert!(json_string.contains("\"bool_field\":true"));
        assert!(json_string.contains("\"i8_field\":42"));
        assert!(json_string.contains("\"i16_field\":1234"));
        assert!(json_string.contains("\"i32_field\":123456"));
        assert!(json_string.contains("\"i64_field\":123456789"));
        assert!(json_string.contains("\"u8_field\":255"));
        assert!(json_string.contains("\"u16_field\":65535"));
        assert!(json_string.contains("\"u32_field\":4294967295"));
        assert!(json_string.contains("\"u64_field\":18446744073709551615"));
        assert!(json_string.contains("\"usize_field\":42"));
        assert!(json_string.contains("\"f32_field\":3.1415927"));
        assert!(json_string.contains("\"f64_field\":2.718281828459045"));
        assert!(json_string.contains("\"string_field\":\"Hello, World!\""));
        assert!(json_string.contains("\"bytes_field\":\"01020304\""));
        assert!(json_string.contains("\"json_field\":\"{\\\"nested\\\": \\\"value\\\"}\""));
        assert!(json_string.contains("\"binary_field\":\"fffefdfc\""));
    }

    #[test]
    fn test_struct_serialization_special_characters() {
        // Test string escaping
        let serializer = StructSerializer::new()
            .field("quotes", &"Hello \"World\"".to_string())
            .field("newlines", &"Line1\nLine2".to_string())
            .field("tabs", &"Tab\there".to_string())
            .field("backslashes", &"Path\\to\\file".to_string());

        let json_string = to_json_internal(serializer).unwrap();

        // Verify proper escaping
        assert!(json_string.contains("\"quotes\":\"Hello \\\"World\\\"\""));
        assert!(json_string.contains("\"newlines\":\"Line1\\nLine2\""));
        assert!(json_string.contains("\"tabs\":\"Tab\\there\""));
        assert!(json_string.contains("\"backslashes\":\"Path\\\\to\\\\file\""));
    }

    #[test]
    fn test_struct_deserialization_comprehensive() {
        let json_string = r#"{
            "bool_field": true,
            "i8_field": 42,
            "i16_field": 1234,
            "i32_field": 123456,
            "i64_field": 123456789,
            "u8_field": 255,
            "u16_field": 65535,
            "u32_field": 4294967295,
            "u64_field": 18446744073709551615,
            "usize_field": 42,
            "f32_field": 3.14159,
            "f64_field": 2.718281828,
            "string_field": "Hello, World!",
            "bytes_field": "01020304",
            "json_field": "{\"nested\": \"value\"}",
            "binary_field": "fffefdfc"
        }"#;

        let deserializer = from_json_internal(json_string).unwrap();

        // Verify all fields are present and correct
        assert!(deserializer.has_field("bool_field"));
        assert!(deserializer.has_field("i8_field"));
        assert!(deserializer.has_field("i16_field"));
        assert!(deserializer.has_field("i32_field"));
        assert!(deserializer.has_field("i64_field"));
        assert!(deserializer.has_field("u8_field"));
        assert!(deserializer.has_field("u16_field"));
        assert!(deserializer.has_field("u32_field"));
        assert!(deserializer.has_field("u64_field"));
        assert!(deserializer.has_field("usize_field"));
        assert!(deserializer.has_field("f32_field"));
        assert!(deserializer.has_field("f64_field"));
        assert!(deserializer.has_field("string_field"));
        assert!(deserializer.has_field("bytes_field"));
        assert!(deserializer.has_field("json_field"));
        assert!(deserializer.has_field("binary_field"));
    }

    #[test]
    fn test_struct_serialization_roundtrip() {
        // Test round-trip serialization and deserialization
        let original_serializer = StructSerializer::new()
            .field("name", &"test".to_string())
            .field("value", &42i32)
            .field("enabled", &true)
            .field("data", &vec![1u8, 2u8, 3u8, 4u8]);

        let json_string = to_json_internal(original_serializer).unwrap();
        let deserializer = from_json_internal(&json_string).unwrap();

        // Verify all fields are preserved
        assert!(deserializer.has_field("name"));
        assert!(deserializer.has_field("value"));
        assert!(deserializer.has_field("enabled"));
        assert!(deserializer.has_field("data"));
    }

    #[test]
    fn test_field_value_to_json_edge_cases() {
        // Test empty strings
        let empty_string = field_value_to_json_static(FieldValue::String("".to_string())).unwrap();
        assert_eq!(empty_string, "\"\"");

        // Test empty bytes
        let empty_bytes = field_value_to_json_static(FieldValue::Bytes(vec![])).unwrap();
        assert_eq!(empty_bytes, "\"\"");

        // Test zero values
        let zero_i32 = field_value_to_json_static(FieldValue::I32(0)).unwrap();
        assert_eq!(zero_i32, "0");

        let zero_f64 = field_value_to_json_static(FieldValue::F64(0.0)).unwrap();
        assert_eq!(zero_f64, "0.0");

        // Test negative values
        let neg_i32 = field_value_to_json_static(FieldValue::I32(-42)).unwrap();
        assert_eq!(neg_i32, "-42");

        let neg_f64 = field_value_to_json_static(FieldValue::F64(-std::f64::consts::PI)).unwrap();
        assert_eq!(neg_f64, "-3.141592653589793");

        // Test optional values
        let some_value = field_value_to_json_static(FieldValue::Optional(Some(Box::new(
            FieldValue::String("test".to_string()),
        ))))
        .unwrap();
        assert_eq!(some_value, "\"test\"");

        let none_value = field_value_to_json_static(FieldValue::Optional(None)).unwrap();
        assert_eq!(none_value, "null");
    }
}
