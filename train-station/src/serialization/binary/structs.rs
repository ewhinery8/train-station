//! Binary-specific implementations for struct serialization
//!
//! This module contains the binary-specific serialization and deserialization
//! implementations for the StructSerializer and StructDeserializer types.
//! These implementations handle the conversion between FieldValue and binary format
//! with proper header management and type safety.
//!
//! # Purpose
//!
//! The binary struct serialization system provides:
//! - Efficient binary format for production deployment
//! - Type-safe serialization of all FieldValue variants
//! - Proper header management with magic numbers and versioning
//! - Comprehensive error handling for malformed data
//!
//! # Implementation Details
//!
//! The binary format uses a type-tagged approach where each FieldValue variant
//! is preceded by a type tag byte that identifies the variant type. This allows
//! for efficient deserialization and proper error handling when encountering
//! unknown or malformed data.
//!
//! # Type Tags
//!
//! The following type tags are used for FieldValue variants:
//! - 0: Bool
//! - 1: I8
//! - 2: I16
//! - 3: I32
//! - 4: I64
//! - 5: U8
//! - 6: U16
//! - 7: U32
//! - 8: U64
//! - 9: Usize
//! - 10: F32
//! - 11: F64
//! - 12: String
//! - 13: Bytes
//! - 14: JsonObject
//! - 15: BinaryObject
//! - 16: Array
//! - 17: Optional
//! - 18: Object
//! - 19: Enum
//!
//! # Thread Safety
//!
//! All functions in this module are thread-safe and can be called concurrently
//! from multiple threads. The underlying binary operations use standard library
//! primitives that provide appropriate synchronization.

use super::{
    deserialize_with_header, serialize_with_header, BinaryReader, BinaryWriter, ObjectType,
};
use crate::serialization::core::{
    FieldValue, SerializationError, SerializationResult, StructDeserializer, StructSerializer,
};
use std::io::Cursor;

/// Converts a StructSerializer to binary format with proper header management
///
/// This function serializes a StructSerializer into a binary format that includes
/// a header with magic number, version information, and the serialized field data.
/// The binary format is optimized for production deployment with minimal overhead.
///
/// # Arguments
///
/// * `serializer` - The StructSerializer containing field data to serialize
///
/// # Returns
///
/// Binary data as a byte vector on success, or `SerializationError` on failure
///
/// # Implementation Details
///
/// The serialization process:
/// 1. Writes a binary header with magic number and version
/// 2. Writes the number of fields as a usize
/// 3. For each field, writes the field name and serializes the FieldValue
/// 4. Uses type-tagged serialization for each FieldValue variant
///
/// # Error Conditions
///
/// - I/O errors during writing
/// - Invalid field data that cannot be serialized
/// - Memory allocation failures for large data structures
///

pub fn to_binary_internal(serializer: StructSerializer) -> SerializationResult<Vec<u8>> {
    let fields = serializer.fields; // Move fields out first
    serialize_with_header(
        ObjectType::Tensor,
        &fields,
        |writer, data: &Vec<(String, FieldValue)>| {
            // Write number of fields
            writer.write_usize(data.len())?;

            // Write each field
            for (name, value) in data {
                writer.write_string(name)?;
                write_field_value_binary_static(writer, value)?;
            }

            Ok(())
        },
    )
}

/// Writes a FieldValue to binary format using type-tagged serialization
///
/// This function serializes a FieldValue variant to binary format by writing
/// a type tag byte followed by the variant's data. The type tag allows for
/// proper deserialization and error handling when reading the data back.
///
/// # Arguments
///
/// * `writer` - Binary writer to output the serialized data
/// * `value` - The FieldValue variant to serialize
///
/// # Returns
///
/// `Ok(())` on success, or `SerializationError` on failure
///
/// # Implementation Details
///
/// The serialization process for each variant:
/// - Writes a type tag byte (0-19) to identify the variant
/// - Serializes the variant's data using appropriate binary writers
/// - Handles nested structures (arrays, objects, enums) recursively
/// - Preserves full precision for floating-point numbers
///
/// # Type Tags
///
/// Each FieldValue variant is assigned a unique type tag:
/// - Bool: 0, I8: 1, I16: 2, I32: 3, I64: 4
/// - U8: 5, U16: 6, U32: 7, U64: 8, Usize: 9
/// - F32: 10, F64: 11, String: 12, Bytes: 13
/// - JsonObject: 14, BinaryObject: 15, Array: 16
/// - Optional: 17, Object: 18, Enum: 19
///
/// # Error Conditions
///
/// - I/O errors during writing
/// - Memory allocation failures for large strings or arrays
/// - Invalid data that cannot be serialized
fn write_field_value_binary_static(
    writer: &mut BinaryWriter<Cursor<Vec<u8>>>,
    value: &FieldValue,
) -> SerializationResult<()> {
    // Write type tag first
    match value {
        FieldValue::Bool(v) => {
            writer.write_u8(0)?;
            writer.write_bool(*v)?;
        }
        FieldValue::I8(v) => {
            writer.write_u8(1)?;
            writer.write_i8(*v)?;
        }
        FieldValue::I16(v) => {
            writer.write_u8(2)?;
            writer.write_i16(*v)?;
        }
        FieldValue::I32(v) => {
            writer.write_u8(3)?;
            writer.write_i32(*v)?;
        }
        FieldValue::I64(v) => {
            writer.write_u8(4)?;
            writer.write_i64(*v)?;
        }
        FieldValue::U8(v) => {
            writer.write_u8(5)?;
            writer.write_u8(*v)?;
        }
        FieldValue::U16(v) => {
            writer.write_u8(6)?;
            writer.write_u16(*v)?;
        }
        FieldValue::U32(v) => {
            writer.write_u8(7)?;
            writer.write_u32(*v)?;
        }
        FieldValue::U64(v) => {
            writer.write_u8(8)?;
            writer.write_u64(*v)?;
        }
        FieldValue::Usize(v) => {
            writer.write_u8(9)?;
            writer.write_usize(*v)?;
        }
        FieldValue::F32(v) => {
            writer.write_u8(10)?;
            writer.write_f32(*v)?;
        }
        FieldValue::F64(v) => {
            writer.write_u8(11)?;
            writer.write_f64(*v)?; // Preserve full f64 precision
        }
        FieldValue::String(v) => {
            writer.write_u8(12)?;
            writer.write_string(v)?;
        }
        FieldValue::Bytes(v) => {
            writer.write_u8(13)?;
            writer.write_vec_u8(v)?;
        }
        FieldValue::JsonObject(v) => {
            writer.write_u8(14)?;
            writer.write_string(v)?;
        }
        FieldValue::BinaryObject(v) => {
            writer.write_u8(15)?;
            writer.write_vec_u8(v)?;
        }
        FieldValue::Array(v) => {
            writer.write_u8(16)?;
            writer.write_usize(v.len())?;
            for elem in v {
                write_field_value_binary_static(writer, elem)?;
            }
        }
        FieldValue::Optional(v) => {
            writer.write_u8(17)?;
            match v {
                Some(inner) => {
                    writer.write_bool(true)?;
                    write_field_value_binary_static(writer, inner)?;
                }
                None => {
                    writer.write_bool(false)?;
                }
            }
        }
        FieldValue::Object(v) => {
            writer.write_u8(18)?;
            writer.write_usize(v.len())?;
            for (key, value) in v {
                writer.write_string(key)?;
                write_field_value_binary_static(writer, value)?;
            }
        }
        FieldValue::Enum { variant, data } => {
            writer.write_u8(19)?; // Type tag for enum
            writer.write_string(variant)?;
            match data {
                Some(enum_data) => {
                    writer.write_bool(true)?; // Has data
                    write_field_value_binary_static(writer, enum_data)?;
                }
                None => {
                    writer.write_bool(false)?; // No data (unit variant)
                }
            }
        }
    }
    Ok(())
}

/// Creates a StructDeserializer from binary data with header validation
///
/// This function deserializes binary data into a StructDeserializer by first
/// validating the binary header and then reading the field data. The function
/// ensures that the binary data contains valid serialized struct information.
///
/// # Arguments
///
/// * `data` - Binary data containing serialized struct information
///
/// # Returns
///
/// A StructDeserializer containing the deserialized fields on success,
/// or `SerializationError` on failure
///
/// # Implementation Details
///
/// The deserialization process:
/// 1. Validates the binary header (magic number, version)
/// 2. Reads the number of fields as a usize
/// 3. For each field, reads the field name and deserializes the FieldValue
/// 4. Uses type-tagged deserialization for each FieldValue variant
/// 5. Constructs a StructDeserializer with the loaded fields
///
/// # Error Conditions
///
/// - Invalid or corrupted binary header
/// - Malformed field data that cannot be deserialized
/// - Memory allocation failures for large data structures
/// - Unexpected end of data during deserialization
///

pub fn from_binary_internal(data: &[u8]) -> SerializationResult<StructDeserializer> {
    deserialize_with_header(data, ObjectType::Tensor, |reader| {
        // Read number of fields
        let field_count = reader.read_usize()?;

        let mut fields = std::collections::HashMap::new();

        // Read each field
        for _ in 0..field_count {
            let name = reader.read_string()?;
            let value = read_field_value_binary(reader)?;
            fields.insert(name, value);
        }

        Ok(StructDeserializer { fields })
    })
}

/// Reads a FieldValue from binary format using type-tagged deserialization
///
/// This function deserializes a FieldValue variant from binary format by first
/// reading a type tag byte to identify the variant, then deserializing the
/// variant's data. The function handles all FieldValue variants and provides
/// proper error handling for malformed data.
///
/// # Arguments
///
/// * `reader` - Binary reader containing the serialized FieldValue data
///
/// # Returns
///
/// The deserialized FieldValue variant on success, or `SerializationError` on failure
///
/// # Implementation Details
///
/// The deserialization process:
/// 1. Reads a type tag byte (0-19) to identify the FieldValue variant
/// 2. Deserializes the variant's data using appropriate binary readers
/// 3. Handles nested structures (arrays, objects, enums) recursively
/// 4. Validates data integrity and provides detailed error messages
///
/// # Type Tags
///
/// The function expects the same type tags used during serialization:
/// - Bool: 0, I8: 1, I16: 2, I32: 3, I64: 4
/// - U8: 5, U16: 6, U32: 7, U64: 8, Usize: 9
/// - F32: 10, F64: 11, String: 12, Bytes: 13
/// - JsonObject: 14, BinaryObject: 15, Array: 16
/// - Optional: 17, Object: 18, Enum: 19
///
/// # Error Conditions
///
/// - Unknown type tags (returns BinaryFormat error)
/// - Malformed or truncated data
/// - Memory allocation failures for large strings or arrays
/// - Invalid data that cannot be deserialized
fn read_field_value_binary(
    reader: &mut BinaryReader<Cursor<&[u8]>>,
) -> SerializationResult<FieldValue> {
    let type_tag = reader.read_u8()?;

    match type_tag {
        0 => {
            let value = reader.read_bool()?;
            Ok(FieldValue::Bool(value))
        }
        1 => {
            let value = reader.read_i8()?;
            Ok(FieldValue::I8(value))
        }
        2 => {
            let value = reader.read_i16()?;
            Ok(FieldValue::I16(value))
        }
        3 => {
            let value = reader.read_i32()?;
            Ok(FieldValue::I32(value))
        }
        4 => {
            let value = reader.read_i64()?;
            Ok(FieldValue::I64(value))
        }
        5 => {
            let value = reader.read_u8()?;
            Ok(FieldValue::U8(value))
        }
        6 => {
            let value = reader.read_u16()?;
            Ok(FieldValue::U16(value))
        }
        7 => {
            let value = reader.read_u32()?;
            Ok(FieldValue::U32(value))
        }
        8 => {
            let value = reader.read_u64()?;
            Ok(FieldValue::U64(value))
        }
        9 => {
            let value = reader.read_usize()?;
            Ok(FieldValue::Usize(value))
        }
        10 => {
            let value = reader.read_f32()?;
            Ok(FieldValue::F32(value))
        }
        11 => {
            let value = reader.read_f64()?;
            Ok(FieldValue::F64(value))
        }
        12 => {
            let value = reader.read_string()?;
            Ok(FieldValue::String(value))
        }
        13 => {
            let value = reader.read_vec_u8()?;
            Ok(FieldValue::Bytes(value))
        }
        14 => {
            let value = reader.read_string()?;
            Ok(FieldValue::JsonObject(value))
        }
        15 => {
            let value = reader.read_vec_u8()?;
            Ok(FieldValue::BinaryObject(value))
        }
        16 => {
            let len = reader.read_usize()?;
            let mut elements = Vec::new();
            for _ in 0..len {
                let elem = read_field_value_binary(reader)?;
                elements.push(elem);
            }
            Ok(FieldValue::Array(elements))
        }
        17 => {
            let has_value = reader.read_bool()?;
            if has_value {
                let value = read_field_value_binary(reader)?;
                Ok(FieldValue::Optional(Some(Box::new(value))))
            } else {
                Ok(FieldValue::Optional(None))
            }
        }
        18 => {
            let len = reader.read_usize()?;
            let mut object = std::collections::HashMap::new();
            for _ in 0..len {
                let key = reader.read_string()?;
                let value = read_field_value_binary(reader)?;
                object.insert(key, value);
            }
            Ok(FieldValue::Object(object))
        }
        19 => {
            let variant = reader.read_string()?;
            let has_data = reader.read_bool()?;
            let data = if has_data {
                Some(Box::new(read_field_value_binary(reader)?))
            } else {
                None
            };
            Ok(FieldValue::Enum { variant, data })
        }
        _ => Err(SerializationError::BinaryFormat {
            message: format!("Unknown type tag: {}", type_tag),
            position: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_serializer_to_binary() {
        let serializer = StructSerializer::new()
            .field("name", &"test")
            .field("value", &42)
            .field("enabled", &true);

        let binary_data = to_binary_internal(serializer).unwrap();
        assert!(!binary_data.is_empty());
        assert!(binary_data.len() > 10); // Should have some content
    }

    #[test]
    fn test_struct_deserializer_from_binary() {
        let serializer = StructSerializer::new()
            .field("name", &"test")
            .field("value", &42)
            .field("enabled", &true);

        let binary_data = to_binary_internal(serializer).unwrap();
        let deserializer = from_binary_internal(&binary_data).unwrap();

        assert!(deserializer.has_field("name"));
        assert!(deserializer.has_field("value"));
        assert!(deserializer.has_field("enabled"));
        assert!(!deserializer.has_field("missing"));
    }

    #[test]
    fn test_write_field_value_binary_static() {
        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);
        let mut writer = BinaryWriter::new(cursor);

        // Test writing different field value types
        let bool_value = FieldValue::Bool(true);
        write_field_value_binary_static(&mut writer, &bool_value).unwrap();

        let int_value = FieldValue::I32(42);
        write_field_value_binary_static(&mut writer, &int_value).unwrap();

        let string_value = FieldValue::String("hello".to_string());
        write_field_value_binary_static(&mut writer, &string_value).unwrap();

        let final_buffer = writer.writer.into_inner();
        assert!(!final_buffer.is_empty());
    }

    #[test]
    fn test_read_field_value_binary() {
        // Create test data
        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);
        let mut writer = BinaryWriter::new(cursor);

        // Write test values
        writer.write_u8(0).unwrap(); // Bool type tag
        writer.write_bool(true).unwrap();

        writer.write_u8(3).unwrap(); // I32 type tag
        writer.write_i32(42).unwrap();

        writer.write_u8(12).unwrap(); // String type tag
        writer.write_string("hello").unwrap();

        // Read back the values
        let final_buffer = writer.writer.into_inner();
        let cursor = Cursor::new(&final_buffer[..]);
        let mut reader = BinaryReader::new(cursor);

        let bool_value = read_field_value_binary(&mut reader).unwrap();
        assert!(matches!(bool_value, FieldValue::Bool(true)));

        let int_value = read_field_value_binary(&mut reader).unwrap();
        assert!(matches!(int_value, FieldValue::I32(42)));

        let string_value = read_field_value_binary(&mut reader).unwrap();
        assert!(matches!(string_value, FieldValue::String(ref s) if s == "hello"));
    }

    #[test]
    fn test_unknown_type_tag_error() {
        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);
        let mut writer = BinaryWriter::new(cursor);

        // Write an unknown type tag
        writer.write_u8(255).unwrap(); // Invalid type tag

        let final_buffer = writer.writer.into_inner();
        let cursor = Cursor::new(&final_buffer[..]);
        let mut reader = BinaryReader::new(cursor);

        let result = read_field_value_binary(&mut reader);
        assert!(result.is_err());
        if let Err(SerializationError::BinaryFormat { message, .. }) = result {
            assert!(message.contains("Unknown type tag: 255"));
        }
    }

    /// Test roundtrip serialization for all field value types
    ///
    /// Verifies that all field value types can be written and read back correctly.
    #[test]
    fn test_all_field_value_types_roundtrip() {
        let test_values = vec![
            FieldValue::Bool(true),
            FieldValue::Bool(false),
            FieldValue::I8(-42),
            FieldValue::I16(-1234),
            FieldValue::I32(-0x12345678),
            FieldValue::I64(-0x123456789ABCDEF0),
            FieldValue::U8(42),
            FieldValue::U16(1234),
            FieldValue::U32(0x12345678),
            FieldValue::U64(0x123456789ABCDEF0),
            FieldValue::Usize(12345),
            FieldValue::F32(std::f32::consts::PI),
            FieldValue::F32(std::f32::consts::E),
            FieldValue::String("Hello, World!".to_string()),
            FieldValue::Bytes(vec![1, 2, 3, 4, 5]),
            FieldValue::JsonObject(r#"{"key": "value"}"#.to_string()),
            FieldValue::BinaryObject(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            FieldValue::Array(vec![
                FieldValue::I32(1),
                FieldValue::I32(2),
                FieldValue::I32(3),
            ]),
            FieldValue::Optional(Some(Box::new(FieldValue::String("optional".to_string())))),
            FieldValue::Optional(None),
        ];

        for value in test_values {
            // Write the value
            let buffer = Vec::new();
            let cursor = Cursor::new(buffer);
            let mut writer = BinaryWriter::new(cursor);
            write_field_value_binary_static(&mut writer, &value).unwrap();
            let final_buffer = writer.writer.into_inner();

            // Read the value back
            let cursor = Cursor::new(&final_buffer[..]);
            let mut reader = BinaryReader::new(cursor);
            let read_value = read_field_value_binary(&mut reader).unwrap();

            // Compare the values
            assert_eq!(value, read_value, "Failed roundtrip for {:?}", value);
        }
    }

    /// Test complex nested structures
    ///
    /// Verifies that complex nested field values (arrays, options) work correctly.
    #[test]
    fn test_complex_nested_structures() {
        // Create a complex nested structure
        let complex_value = FieldValue::Array(vec![
            FieldValue::String("nested".to_string()),
            FieldValue::Array(vec![
                FieldValue::I32(1),
                FieldValue::I32(2),
                FieldValue::Optional(Some(Box::new(FieldValue::Bool(true)))),
            ]),
            FieldValue::Optional(Some(Box::new(FieldValue::Array(vec![
                FieldValue::F32(1.5),
                FieldValue::F32(2.5),
            ])))),
        ]);

        // Write the complex value
        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);
        let mut writer = BinaryWriter::new(cursor);
        write_field_value_binary_static(&mut writer, &complex_value).unwrap();
        let final_buffer = writer.writer.into_inner();

        // Read the complex value back
        let cursor = Cursor::new(&final_buffer[..]);
        let mut reader = BinaryReader::new(cursor);
        let read_value = read_field_value_binary(&mut reader).unwrap();

        // Compare the values
        assert_eq!(complex_value, read_value);
    }

    /// Test large data structures
    ///
    /// Verifies that large field values can be serialized and deserialized correctly.
    #[test]
    fn test_large_data_structures() {
        // Create large arrays and strings
        let large_array = FieldValue::Array((0..1000).map(FieldValue::I32).collect());

        let large_string = FieldValue::String(
            (0..1000)
                .map(|i| format!("item_{}", i))
                .collect::<Vec<_>>()
                .join(", "),
        );

        let large_bytes = FieldValue::Bytes((0..1000).map(|i| (i % 256) as u8).collect());

        let test_values = vec![large_array, large_string, large_bytes];

        for value in test_values {
            // Write the large value
            let buffer = Vec::new();
            let cursor = Cursor::new(buffer);
            let mut writer = BinaryWriter::new(cursor);
            write_field_value_binary_static(&mut writer, &value).unwrap();
            let final_buffer = writer.writer.into_inner();

            // Read the large value back
            let cursor = Cursor::new(&final_buffer[..]);
            let mut reader = BinaryReader::new(cursor);
            let read_value = read_field_value_binary(&mut reader).unwrap();

            // Compare the values
            assert_eq!(value, read_value, "Failed roundtrip for large value");
        }
    }

    /// Test edge cases and boundary values
    ///
    /// Verifies that edge cases and boundary values are handled correctly.
    #[test]
    fn test_edge_cases_and_boundaries() {
        let edge_cases = vec![
            // Numeric boundaries
            FieldValue::I8(i8::MIN),
            FieldValue::I8(i8::MAX),
            FieldValue::I16(i16::MIN),
            FieldValue::I16(i16::MAX),
            FieldValue::I32(i32::MIN),
            FieldValue::I32(i32::MAX),
            FieldValue::I64(i64::MIN),
            FieldValue::I64(i64::MAX),
            FieldValue::U8(u8::MAX),
            FieldValue::U16(u16::MAX),
            FieldValue::U32(u32::MAX),
            FieldValue::U64(u64::MAX),
            FieldValue::Usize(usize::MAX),
            // Floating point special values
            FieldValue::F32(f32::INFINITY),
            FieldValue::F32(f32::NEG_INFINITY),
            FieldValue::F32(f32::NAN),
            FieldValue::F64(f64::INFINITY),
            FieldValue::F64(f64::NEG_INFINITY),
            FieldValue::F64(f64::NAN),
            // Empty collections
            FieldValue::String("".to_string()),
            FieldValue::Bytes(vec![]),
            FieldValue::Array(vec![]),
            FieldValue::JsonObject("".to_string()),
            FieldValue::BinaryObject(vec![]),
        ];

        for value in edge_cases {
            // Write the edge case value
            let buffer = Vec::new();
            let cursor = Cursor::new(buffer);
            let mut writer = BinaryWriter::new(cursor);
            write_field_value_binary_static(&mut writer, &value).unwrap();
            let final_buffer = writer.writer.into_inner();

            // Read the edge case value back
            let cursor = Cursor::new(&final_buffer[..]);
            let mut reader = BinaryReader::new(cursor);
            let read_value = read_field_value_binary(&mut reader).unwrap();

            // For NaN values, we need special comparison
            match (&value, &read_value) {
                (FieldValue::F32(f1), FieldValue::F32(f2)) if f1.is_nan() && f2.is_nan() => {
                    // Both are NaN, which is correct
                }
                (FieldValue::F64(f1), FieldValue::F64(f2)) if f1.is_nan() && f2.is_nan() => {
                    // Both are NaN, which is correct
                }
                _ => {
                    assert_eq!(
                        value, read_value,
                        "Failed roundtrip for edge case {:?}",
                        value
                    );
                }
            }
        }
    }

    /// Test malformed data handling
    ///
    /// Verifies that the system correctly handles malformed binary data.
    #[test]
    fn test_malformed_data_handling() {
        // Test with empty data
        let empty_data = [];
        let cursor = Cursor::new(&empty_data[..]);
        let mut reader = BinaryReader::new(cursor);
        let result = read_field_value_binary(&mut reader);
        assert!(result.is_err());

        // Test with incomplete type tag
        let incomplete_data = [0x01]; // Just a type tag, no data
        let cursor = Cursor::new(&incomplete_data[..]);
        let mut reader = BinaryReader::new(cursor);
        let result = read_field_value_binary(&mut reader);
        assert!(result.is_err());

        // Test with invalid array length
        let invalid_array_data = [
            0x10, // Array type tag
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        ];
        let cursor = Cursor::new(&invalid_array_data[..]);
        let mut reader = BinaryReader::new(cursor);
        let result = read_field_value_binary(&mut reader);
        assert!(result.is_err());
    }

    /// Test complete struct serialization roundtrip
    ///
    /// Verifies that complete struct serialization and deserialization works correctly.
    #[test]
    fn test_complete_struct_roundtrip() {
        let original_serializer = StructSerializer::new()
            .field("name", &"test_struct")
            .field("value", &42)
            .field("enabled", &true)
            .field("float_value", &std::f32::consts::PI)
            .field("array_value", &vec![1, 2, 3, 4, 5])
            .field("string_value", &"Hello, World!")
            .field("optional_value", &"optional".to_string())
            .field("empty_optional", &"".to_string())
            .field("bytes_value", &vec![0xDEu8, 0xAD, 0xBE, 0xEF]);

        // Serialize to binary
        let binary_data = to_binary_internal(original_serializer).unwrap();
        assert!(!binary_data.is_empty());

        // Deserialize from binary
        let mut deserializer = from_binary_internal(&binary_data).unwrap();

        // Verify all fields are present and correct
        assert!(deserializer.has_field("name"));
        assert!(deserializer.has_field("value"));
        assert!(deserializer.has_field("enabled"));
        assert!(deserializer.has_field("float_value"));
        assert!(deserializer.has_field("array_value"));
        assert!(deserializer.has_field("string_value"));
        assert!(deserializer.has_field("optional_value"));
        assert!(deserializer.has_field("empty_optional"));
        assert!(deserializer.has_field("bytes_value"));

        // Verify field values
        assert_eq!(deserializer.field::<String>("name").unwrap(), "test_struct");
        assert_eq!(deserializer.field::<i32>("value").unwrap(), 42);
        assert!(deserializer.field::<bool>("enabled").unwrap());
        assert!(
            (deserializer.field::<f32>("float_value").unwrap() - std::f32::consts::PI).abs() < 1e-6
        );
        assert_eq!(
            deserializer.field::<Vec<i32>>("array_value").unwrap(),
            vec![1, 2, 3, 4, 5]
        );
        assert_eq!(
            deserializer.field::<String>("string_value").unwrap(),
            "Hello, World!"
        );
        assert_eq!(
            deserializer.field::<String>("optional_value").unwrap(),
            "optional".to_string()
        );
        assert_eq!(
            deserializer.field::<String>("empty_optional").unwrap(),
            "".to_string()
        );
        assert_eq!(
            deserializer.field::<Vec<u8>>("bytes_value").unwrap(),
            vec![0xDE, 0xAD, 0xBE, 0xEF]
        );
    }
}
