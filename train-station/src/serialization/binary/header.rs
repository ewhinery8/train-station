//! Binary header management for file format validation and metadata handling
//!
//! This module provides functions for writing and reading binary file headers,
//! including magic number validation, version checking, and object type verification.
//! The header system ensures data integrity and format compatibility.

use std::io::{Cursor, Read, Write};

use super::reader::BinaryReader;
use super::types::{ObjectType, FORMAT_VERSION, MAGIC_NUMBER};
use super::writer::BinaryWriter;
use crate::serialization::core::{SerializationError, SerializationResult};

/// Write binary header with magic number, version, and object type
///
/// This function writes the standard binary file header that identifies the file
/// format, version, and object type. The header includes a magic number for format
/// identification, version information for compatibility checking, and metadata
/// about the serialized object.
///
/// # Arguments
///
/// * `writer` - Binary writer to output the header
/// * `object_type` - Type identifier for the serialized object
/// * `data_length` - Length of the object data in bytes
///
/// # Returns
///
/// `Ok(())` on success, or `SerializationError` on failure
///
/// # Header Format
///
/// The header consists of:
/// - Magic number (4 bytes): 0x54535F42 ("TS_B")
/// - Version (4 bytes): Format version number
/// - Checksum (4 bytes): Placeholder for future CRC32 validation
/// - Data length (8 bytes): Size of the object data
/// - Object type (4 bytes): Type identifier
///

pub fn write_header<W: Write>(
    writer: &mut BinaryWriter<W>,
    object_type: ObjectType,
    data_length: u64,
) -> SerializationResult<()> {
    writer.write_u32(MAGIC_NUMBER)?;
    writer.write_u32(FORMAT_VERSION)?;
    writer.write_u32(0)?; // Placeholder for checksum
    writer.write_u64(data_length)?;
    writer.write_u32(object_type as u32)?;
    Ok(())
}

/// Read and validate binary header
///
/// This function reads and validates the binary file header, checking the magic
/// number and version for format compatibility. It returns the object type and
/// data length for further processing.
///
/// # Arguments
///
/// * `reader` - Binary reader containing the header data
///
/// # Returns
///
/// A tuple containing the object type and data length on success, or `SerializationError` on failure
///
/// # Validation
///
/// The function performs the following validations:
/// - Magic number must match the expected value (0x54535F42)
/// - Version must match the current format version
/// - Object type must be a recognized type identifier
///

pub fn read_header<R: Read>(
    reader: &mut BinaryReader<R>,
) -> SerializationResult<(ObjectType, u64)> {
    let magic = reader.read_u32()?;
    if magic != MAGIC_NUMBER {
        return Err(SerializationError::InvalidMagic {
            expected: MAGIC_NUMBER,
            found: magic,
        });
    }

    let version = reader.read_u32()?;
    if version != FORMAT_VERSION {
        return Err(SerializationError::VersionMismatch {
            expected: FORMAT_VERSION,
            found: version,
        });
    }

    let _checksum = reader.read_u32()?; // TODO: Implement checksum validation
    let data_length = reader.read_u64()?;
    let object_type_raw = reader.read_u32()?;
    let object_type = ObjectType::from_u32(object_type_raw)?;

    Ok((object_type, data_length))
}

/// Serialize data to binary format with header
///
/// This function provides a convenient way to serialize objects to binary format
/// with proper header information. It handles the creation of the binary header,
/// serialization of the object data, and finalization of the binary stream.
///
/// # Arguments
///
/// * `object_type` - Type identifier for the serialized object
/// * `data` - The object to serialize
/// * `serialize_fn` - Function to serialize the object data
///
/// # Returns
///
/// Binary data with header on success, or `SerializationError` on failure
///

pub fn serialize_with_header<T, F>(
    object_type: ObjectType,
    data: &T,
    serialize_fn: F,
) -> SerializationResult<Vec<u8>>
where
    F: FnOnce(&mut BinaryWriter<Cursor<Vec<u8>>>, &T) -> SerializationResult<()>,
{
    // Use a single-pass approach: write header with placeholder length, then data
    let buffer = Vec::new();
    let mut writer = BinaryWriter::new(Cursor::new(buffer));

    // Write header with placeholder data length (will be updated later)
    write_header(&mut writer, object_type, 0)?;

    // Remember the position where we need to update the data length
    let header_end_pos = writer.bytes_written();

    // Serialize the actual data
    serialize_fn(&mut writer, data)?;

    // Get the final buffer and update the data length in the header
    let mut final_buffer = writer.writer.into_inner();
    let data_length = final_buffer.len() - header_end_pos;

    // Update the data length field in the header (at position 12-19)
    // Header: magic(4) + version(4) + checksum(4) + data_length(8) + object_type(4) = 24 bytes
    let length_bytes = (data_length as u64).to_le_bytes();
    final_buffer[12..20].copy_from_slice(&length_bytes);

    Ok(final_buffer)
}

/// Deserialize data from binary format with header validation
///
/// This function provides a convenient way to deserialize objects from binary format
/// with proper header validation. It validates the binary header, checks object type
/// compatibility, and handles the deserialization of the object data.
///
/// # Arguments
///
/// * `data` - Binary data containing the serialized object
/// * `expected_type` - Expected object type for validation
/// * `deserialize_fn` - Function to deserialize the object data
///
/// # Returns
///
/// The deserialized object on success, or `SerializationError` on failure
///
/// # Validation
///
/// The function performs the following validations:
/// - Magic number validation
/// - Version compatibility checking
/// - Object type verification
/// - Data length validation
///

pub fn deserialize_with_header<T, F>(
    data: &[u8],
    expected_type: ObjectType,
    deserialize_fn: F,
) -> SerializationResult<T>
where
    F: FnOnce(&mut BinaryReader<Cursor<&[u8]>>) -> SerializationResult<T>,
{
    let mut reader = BinaryReader::new(Cursor::new(data));

    let (object_type, data_length) = read_header(&mut reader)?;

    if object_type != expected_type {
        return Err(SerializationError::BinaryFormat {
            message: format!(
                "Object type mismatch: expected {:?}, found {:?}",
                expected_type, object_type
            ),
            position: Some(reader.bytes_read()),
        });
    }

    // The data_length field represents the length of the object data only
    // We should verify that we have enough data to read the object
    if reader.bytes_read() + data_length as usize > data.len() {
        return Err(SerializationError::BinaryFormat {
            message: format!(
                "Data length mismatch: header claims {} bytes of data, but only {} bytes available",
                data_length,
                data.len() - reader.bytes_read()
            ),
            position: Some(reader.bytes_read()),
        });
    }

    deserialize_fn(&mut reader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Test binary header writing and reading functionality
    ///
    /// Verifies that binary headers can be written and read correctly with
    /// proper magic number, version, and object type handling.
    #[test]
    fn test_header_roundtrip() {
        let mut buffer = Vec::new();
        let object_type = ObjectType::Tensor;
        let data_length = 1024u64;

        {
            let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));
            write_header(&mut writer, object_type, data_length).unwrap();
        }

        let mut reader = BinaryReader::new(Cursor::new(&buffer));
        let (read_type, read_length) = read_header(&mut reader).unwrap();

        assert_eq!(read_type, object_type);
        assert_eq!(read_length, data_length);
    }

    /// Test error handling for invalid magic numbers
    ///
    /// Verifies that the binary reader correctly rejects files with invalid
    /// magic numbers and provides appropriate error information.
    #[test]
    fn test_invalid_magic_number() {
        let mut buffer = Vec::new();

        {
            let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));
            writer.write_u32(0xDEADBEEF).unwrap(); // Wrong magic number
            writer.write_u32(FORMAT_VERSION).unwrap();
            writer.write_u32(0).unwrap();
            writer.write_u64(0).unwrap();
            writer.write_u32(ObjectType::Tensor as u32).unwrap();
        }

        let mut reader = BinaryReader::new(Cursor::new(&buffer));
        let result = read_header(&mut reader);

        match result {
            Err(SerializationError::InvalidMagic { expected, found }) => {
                assert_eq!(expected, MAGIC_NUMBER);
                assert_eq!(found, 0xDEADBEEF);
            }
            _ => panic!("Expected InvalidMagic error"),
        }
    }

    /// Test error handling for version mismatches
    ///
    /// Verifies that the binary reader correctly rejects files with incompatible
    /// format versions and provides appropriate error information.
    #[test]
    fn test_version_mismatch() {
        let mut buffer = Vec::new();

        {
            let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));
            writer.write_u32(MAGIC_NUMBER).unwrap();
            writer.write_u32(999).unwrap(); // Wrong version
            writer.write_u32(0).unwrap();
            writer.write_u64(0).unwrap();
            writer.write_u32(ObjectType::Tensor as u32).unwrap();
        }

        let mut reader = BinaryReader::new(Cursor::new(&buffer));
        let result = read_header(&mut reader);

        match result {
            Err(SerializationError::VersionMismatch { expected, found }) => {
                assert_eq!(expected, FORMAT_VERSION);
                assert_eq!(found, 999);
            }
            _ => panic!("Expected VersionMismatch error"),
        }
    }

    /// Test error handling for unknown object types
    ///
    /// Verifies that the binary reader correctly rejects files with unknown
    /// object type identifiers.
    #[test]
    fn test_unknown_object_type() {
        let mut buffer = Vec::new();

        {
            let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));
            writer.write_u32(MAGIC_NUMBER).unwrap();
            writer.write_u32(FORMAT_VERSION).unwrap();
            writer.write_u32(0).unwrap();
            writer.write_u64(0).unwrap();
            writer.write_u32(0x9999).unwrap(); // Unknown object type
        }

        let mut reader = BinaryReader::new(Cursor::new(&buffer));
        let result = read_header(&mut reader);

        match result {
            Err(SerializationError::BinaryFormat { message, .. }) => {
                assert!(message.contains("Unknown object type"));
            }
            _ => panic!("Expected BinaryFormat error for unknown object type"),
        }
    }

    /// Test serialize_with_header and deserialize_with_header roundtrip
    ///
    /// Verifies that the high-level serialization functions work correctly
    /// for complete roundtrip serialization and deserialization.
    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let test_data = vec![1u8, 2, 3, 4, 5];

        // Serialize with header
        let binary_data = serialize_with_header(ObjectType::Tensor, &test_data, |writer, data| {
            writer.write_vec_u8(data)
        })
        .unwrap();

        // Deserialize with header
        let deserialized_data =
            deserialize_with_header(&binary_data, ObjectType::Tensor, |reader| {
                reader.read_vec_u8()
            })
            .unwrap();

        assert_eq!(test_data, deserialized_data);
    }

    /// Test object type mismatch error handling
    ///
    /// Verifies that deserialize_with_header correctly rejects data with
    /// mismatched object types.
    #[test]
    fn test_object_type_mismatch() {
        let test_data = vec![1u8, 2, 3, 4, 5];

        // Serialize as Tensor type
        let binary_data = serialize_with_header(ObjectType::Tensor, &test_data, |writer, data| {
            writer.write_vec_u8(data)
        })
        .unwrap();

        // Try to deserialize as Adam type (should fail)
        let result = deserialize_with_header(&binary_data, ObjectType::Adam, |reader| {
            reader.read_vec_u8()
        });

        match result {
            Err(SerializationError::BinaryFormat { message, .. }) => {
                assert!(message.contains("Object type mismatch"));
            }
            _ => panic!("Expected BinaryFormat error for object type mismatch"),
        }
    }

    /// Test data length validation
    ///
    /// Verifies that deserialize_with_header correctly validates data length
    /// and rejects truncated or corrupted data.
    #[test]
    fn test_data_length_validation() {
        let test_data = vec![1u8, 2, 3, 4, 5];

        // Serialize with header
        let mut binary_data =
            serialize_with_header(ObjectType::Tensor, &test_data, |writer, data| {
                writer.write_vec_u8(data)
            })
            .unwrap();

        // Truncate the data (remove last byte)
        binary_data.pop();

        // Try to deserialize truncated data (should fail)
        let result = deserialize_with_header(&binary_data, ObjectType::Tensor, |reader| {
            reader.read_vec_u8()
        });

        match result {
            Err(SerializationError::BinaryFormat { message, .. }) => {
                assert!(message.contains("Data length mismatch"));
            }
            _ => panic!("Expected BinaryFormat error for data length mismatch"),
        }
    }

    /// Test edge case: empty data
    ///
    /// Verifies that the header system works correctly with empty data.
    #[test]
    fn test_empty_data() {
        let empty_data: Vec<u8> = vec![];

        // Serialize empty data
        let binary_data = serialize_with_header(ObjectType::Tensor, &empty_data, |writer, data| {
            writer.write_vec_u8(data)
        })
        .unwrap();

        // Deserialize empty data
        let deserialized_data =
            deserialize_with_header(&binary_data, ObjectType::Tensor, |reader| {
                reader.read_vec_u8()
            })
            .unwrap();

        assert_eq!(empty_data, deserialized_data);
    }

    /// Test edge case: large data
    ///
    /// Verifies that the header system works correctly with large data.
    #[test]
    fn test_large_data() {
        let large_data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        // Serialize large data
        let binary_data = serialize_with_header(ObjectType::Tensor, &large_data, |writer, data| {
            writer.write_vec_u8(data)
        })
        .unwrap();

        // Deserialize large data
        let deserialized_data =
            deserialize_with_header(&binary_data, ObjectType::Tensor, |reader| {
                reader.read_vec_u8()
            })
            .unwrap();

        assert_eq!(large_data, deserialized_data);
    }

    /// Test all object types
    ///
    /// Verifies that all defined object types can be serialized and deserialized.
    #[test]
    fn test_all_object_types() {
        let test_data = vec![1u8, 2, 3, 4, 5];
        let object_types = vec![
            ObjectType::Tensor,
            ObjectType::Adam,
            ObjectType::AdamConfig,
            ObjectType::Shape,
            ObjectType::ParameterState,
        ];

        for object_type in object_types {
            // Serialize with specific object type
            let binary_data = serialize_with_header(object_type, &test_data, |writer, data| {
                writer.write_vec_u8(data)
            })
            .unwrap();

            // Deserialize with same object type
            let deserialized_data =
                deserialize_with_header(&binary_data, object_type, |reader| reader.read_vec_u8())
                    .unwrap();

            assert_eq!(test_data, deserialized_data);
        }
    }

    /// Test malformed header data
    ///
    /// Verifies that the system correctly handles various forms of malformed data.
    #[test]
    fn test_malformed_header_data() {
        // Test with insufficient data for header
        let insufficient_data = vec![0x42, 0x5F, 0x53, 0x54]; // Partial magic number
        let result = deserialize_with_header(&insufficient_data, ObjectType::Tensor, |reader| {
            reader.read_vec_u8()
        });
        assert!(result.is_err());

        // Test with completely random data
        let random_data: Vec<u8> = (0..100).map(|i| ((i * 7 + 13) % 256) as u8).collect();
        let result = deserialize_with_header(&random_data, ObjectType::Tensor, |reader| {
            reader.read_vec_u8()
        });
        assert!(result.is_err());
    }

    /// Test header size validation
    ///
    /// Verifies that the header has the expected size and structure.
    #[test]
    fn test_header_size() {
        let mut buffer = Vec::new();
        let object_type = ObjectType::Tensor;
        let data_length = 1024u64;

        {
            let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));
            write_header(&mut writer, object_type, data_length).unwrap();
        }

        // Header should be exactly 24 bytes:
        // magic(4) + version(4) + checksum(4) + data_length(8) + object_type(4) = 24 bytes
        assert_eq!(buffer.len(), 24);

        // Verify magic number is at the beginning
        let magic_bytes = &buffer[0..4];
        let magic = u32::from_le_bytes([
            magic_bytes[0],
            magic_bytes[1],
            magic_bytes[2],
            magic_bytes[3],
        ]);
        assert_eq!(magic, MAGIC_NUMBER);

        // Verify version
        let version_bytes = &buffer[4..8];
        let version = u32::from_le_bytes([
            version_bytes[0],
            version_bytes[1],
            version_bytes[2],
            version_bytes[3],
        ]);
        assert_eq!(version, FORMAT_VERSION);

        // Verify data length
        let length_bytes = &buffer[12..20];
        let length = u64::from_le_bytes([
            length_bytes[0],
            length_bytes[1],
            length_bytes[2],
            length_bytes[3],
            length_bytes[4],
            length_bytes[5],
            length_bytes[6],
            length_bytes[7],
        ]);
        assert_eq!(length, data_length);

        // Verify object type
        let type_bytes = &buffer[20..24];
        let obj_type =
            u32::from_le_bytes([type_bytes[0], type_bytes[1], type_bytes[2], type_bytes[3]]);
        assert_eq!(obj_type, object_type as u32);
    }
}
