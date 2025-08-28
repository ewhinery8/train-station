//! Binary reader with built-in endianness handling and validation
//!
//! This module provides safe binary reading capabilities with automatic
//! little-endian byte ordering, byte counting, and comprehensive validation.
//! It wraps any type implementing `Read` and provides a high-level interface
//! for deserializing primitive types and collections with bounds checking.

use std::io::Read;

use crate::serialization::core::{SerializationError, SerializationResult};

/// Binary reader with built-in endianness handling and validation
///
/// This struct provides safe binary reading capabilities with automatic
/// little-endian byte ordering, byte counting, and comprehensive validation.
/// It wraps any type implementing `Read` and provides a high-level interface
/// for deserializing primitive types and collections with bounds checking.
///
/// # Features
///
/// - **Endianness Handling**: Automatic little-endian conversion for all numeric types
/// - **Byte Counting**: Tracks total bytes read for debugging and validation
/// - **Bounds Checking**: Validates data availability before reading
/// - **Size Limits**: Prevents memory exhaustion with collection size limits
/// - **Type Safety**: Strongly typed methods for each data type
/// - **Error Handling**: Comprehensive error propagation through `SerializationResult`
/// - **Generic Design**: Works with any `Read` implementation
///

///
/// # Thread Safety
///
/// This type is not thread-safe. Access from multiple threads requires external
/// synchronization.
pub struct BinaryReader<R: Read> {
    /// The underlying reader implementation
    reader: R,
    /// Total number of bytes read
    bytes_read: usize,
}

impl<R: Read> BinaryReader<R> {
    /// Create a new BinaryReader wrapping the provided reader
    ///
    /// # Arguments
    ///
    /// * `reader` - The underlying reader implementation
    ///
    /// # Returns
    ///
    /// A new BinaryReader instance with byte counting initialized to zero
    ///

    pub fn new(reader: R) -> Self {
        Self {
            reader,
            bytes_read: 0,
        }
    }

    /// Get the total number of bytes read so far
    ///
    /// This method returns the cumulative count of all bytes read through
    /// this BinaryReader instance. This is useful for debugging, validation,
    /// and determining the position in the data stream.
    ///
    /// # Returns
    ///
    /// The total number of bytes read
    ///

    pub fn bytes_read(&self) -> usize {
        self.bytes_read
    }

    /// Read an 8-bit unsigned integer
    ///
    /// Reads a single byte representing an unsigned 8-bit integer value.
    /// No endianness conversion is needed for single-byte values.
    ///
    /// # Returns
    ///
    /// The 8-bit unsigned integer on success, or `SerializationError` on failure
    pub fn read_u8(&mut self) -> SerializationResult<u8> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf)?;
        self.bytes_read += 1;
        Ok(buf[0])
    }

    /// Read a 16-bit unsigned integer in little-endian format
    ///
    /// Reads two bytes representing an unsigned 16-bit integer value in
    /// little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 16-bit unsigned integer on success, or `SerializationError` on failure
    pub fn read_u16(&mut self) -> SerializationResult<u16> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf)?;
        self.bytes_read += 2;
        Ok(u16::from_le_bytes(buf))
    }

    /// Read a 32-bit unsigned integer in little-endian format
    ///
    /// Reads four bytes representing an unsigned 32-bit integer value in
    /// little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 32-bit unsigned integer on success, or `SerializationError` on failure
    pub fn read_u32(&mut self) -> SerializationResult<u32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        self.bytes_read += 4;
        Ok(u32::from_le_bytes(buf))
    }

    /// Read a 64-bit unsigned integer in little-endian format
    ///
    /// Reads eight bytes representing an unsigned 64-bit integer value in
    /// little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 64-bit unsigned integer on success, or `SerializationError` on failure
    pub fn read_u64(&mut self) -> SerializationResult<u64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        self.bytes_read += 8;
        Ok(u64::from_le_bytes(buf))
    }

    /// Read an 8-bit signed integer
    ///
    /// Reads a single byte representing a signed 8-bit integer value.
    /// No endianness conversion is needed for single-byte values.
    ///
    /// # Returns
    ///
    /// The 8-bit signed integer on success, or `SerializationError` on failure
    pub fn read_i8(&mut self) -> SerializationResult<i8> {
        Ok(self.read_u8()? as i8)
    }

    /// Read a 16-bit signed integer in little-endian format
    ///
    /// Reads two bytes representing a signed 16-bit integer value in
    /// little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 16-bit signed integer on success, or `SerializationError` on failure
    pub fn read_i16(&mut self) -> SerializationResult<i16> {
        Ok(self.read_u16()? as i16)
    }

    /// Read a 32-bit signed integer in little-endian format
    ///
    /// Reads four bytes representing a signed 32-bit integer value in
    /// little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 32-bit signed integer on success, or `SerializationError` on failure
    pub fn read_i32(&mut self) -> SerializationResult<i32> {
        Ok(self.read_u32()? as i32)
    }

    /// Read a 64-bit signed integer in little-endian format
    ///
    /// Reads eight bytes representing a signed 64-bit integer value in
    /// little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 64-bit signed integer on success, or `SerializationError` on failure
    pub fn read_i64(&mut self) -> SerializationResult<i64> {
        Ok(self.read_u64()? as i64)
    }

    /// Read a 32-bit floating point number in little-endian format
    ///
    /// Reads four bytes representing a 32-bit floating point value in
    /// little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 32-bit floating point number on success, or `SerializationError` on failure
    pub fn read_f32(&mut self) -> SerializationResult<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    /// Read a 64-bit floating point number in little-endian format
    ///
    /// Reads eight bytes representing a 64-bit floating point value in
    /// little-endian byte order by reading the bit representation.
    ///
    /// # Returns
    ///
    /// The 64-bit floating point number on success, or `SerializationError` on failure
    pub fn read_f64(&mut self) -> SerializationResult<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    /// Read a platform-specific size type in little-endian format
    ///
    /// Reads eight bytes representing a platform-specific size value in
    /// little-endian byte order. Validates that the value fits within the
    /// platform's size type range.
    ///
    /// # Returns
    ///
    /// The platform-specific size value on success, or `SerializationError` on failure
    pub fn read_usize(&mut self) -> SerializationResult<usize> {
        let value = self.read_u64()?;
        if value > usize::MAX as u64 {
            return Err(SerializationError::BinaryFormat {
                message: format!("usize value {} exceeds platform maximum", value),
                position: Some(self.bytes_read - 8),
            });
        }
        Ok(value as usize)
    }

    /// Read a boolean value
    ///
    /// Reads a single byte representing a boolean value (0 = false, 1 = true).
    /// Validates that the value is either 0 or 1.
    ///
    /// # Returns
    ///
    /// The boolean value on success, or `SerializationError` on failure
    pub fn read_bool(&mut self) -> SerializationResult<bool> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(SerializationError::BinaryFormat {
                message: format!("Invalid boolean value: {}", value),
                position: Some(self.bytes_read - 1),
            }),
        }
    }

    /// Read a UTF-8 string with length prefix
    ///
    /// Reads a string by first reading its length as a 32-bit unsigned integer,
    /// followed by the UTF-8 bytes of the string. The method includes validation
    /// for string length limits and UTF-8 encoding correctness.
    ///
    /// # Returns
    ///
    /// The UTF-8 string on success, or `SerializationError` on failure
    ///
    /// # Validation
    ///
    /// - String length must not exceed 1,000,000 bytes
    /// - String data must be valid UTF-8 encoding
    ///

    pub fn read_string(&mut self) -> SerializationResult<String> {
        let length = self.read_u32()? as usize;
        if length > 1_000_000 {
            return Err(SerializationError::BinaryFormat {
                message: format!("String length {} exceeds maximum allowed", length),
                position: Some(self.bytes_read - 4),
            });
        }

        let mut buf = vec![0u8; length];
        self.reader.read_exact(&mut buf)?;
        self.bytes_read += length;

        String::from_utf8(buf).map_err(|e| SerializationError::BinaryFormat {
            message: format!("Invalid UTF-8 string: {}", e),
            position: Some(self.bytes_read - length),
        })
    }

    /// Read raw bytes with specified length
    ///
    /// Reads the specified number of bytes directly from the input without
    /// any length prefix. This is useful for reading raw binary data or
    /// when the length is known from context.
    ///
    /// # Arguments
    ///
    /// * `length` - Number of bytes to read
    ///
    /// # Returns
    ///
    /// The raw bytes on success, or `SerializationError` on failure
    pub fn read_bytes(&mut self, length: usize) -> SerializationResult<Vec<u8>> {
        let mut buf = vec![0u8; length];
        self.reader.read_exact(&mut buf)?;
        self.bytes_read += length;
        Ok(buf)
    }

    /// Read a vector of 8-bit unsigned integers with length prefix
    ///
    /// Reads a vector by first reading its length as a 64-bit unsigned integer,
    /// followed by the raw bytes of the vector data.
    ///
    /// # Returns
    ///
    /// The vector of 8-bit unsigned integers on success, or `SerializationError` on failure
    pub fn read_vec_u8(&mut self) -> SerializationResult<Vec<u8>> {
        let length = self.read_u64()? as usize;
        self.validate_collection_size(length, 1)?;
        self.read_bytes(length)
    }

    /// Read a vector of platform-specific size types with length prefix
    ///
    /// Reads a vector by first reading its length as a 64-bit unsigned integer,
    /// followed by each size value in little-endian format.
    ///
    /// # Returns
    ///
    /// The vector of platform-specific size values on success, or `SerializationError` on failure
    #[allow(unused)]
    pub fn read_vec_usize(&mut self) -> SerializationResult<Vec<usize>> {
        let length = self.read_u64()? as usize;
        self.validate_collection_size(length, 8)?;

        let mut vec = Vec::with_capacity(length);
        for _ in 0..length {
            vec.push(self.read_usize()?);
        }
        Ok(vec)
    }

    /// Read a vector of 32-bit floating point numbers with length prefix
    ///
    /// Reads a vector by first reading its length as a 64-bit unsigned integer,
    /// followed by each floating point value in little-endian format.
    ///
    /// # Returns
    ///
    /// The vector of 32-bit floating point numbers on success, or `SerializationError` on failure
    #[allow(unused)]
    pub fn read_vec_f32(&mut self) -> SerializationResult<Vec<f32>> {
        let length = self.read_u64()? as usize;
        self.validate_collection_size(length, 4)?;

        let mut vec = Vec::with_capacity(length);
        for _ in 0..length {
            vec.push(self.read_f32()?);
        }
        Ok(vec)
    }

    /// Read an optional value with presence flag
    ///
    /// Reads an optional value by first reading a boolean flag indicating whether
    /// the value is present, followed by the actual value if present. This allows
    /// for efficient handling of nullable or optional data during deserialization.
    ///
    /// # Arguments
    ///
    /// * `read_fn` - Function to read the contained value if present
    ///
    /// # Returns
    ///
    /// The optional value on success, or `SerializationError` on failure
    ///

    #[allow(unused)]
    pub fn read_option<T, F>(&mut self, read_fn: F) -> SerializationResult<Option<T>>
    where
        F: FnOnce(&mut Self) -> SerializationResult<T>,
    {
        if self.read_bool()? {
            Ok(Some(read_fn(self)?))
        } else {
            Ok(None)
        }
    }

    /// Validate collection size to prevent memory exhaustion
    ///
    /// This method validates that a collection's size and memory requirements
    /// are within acceptable limits to prevent memory exhaustion attacks and
    /// ensure system stability during deserialization.
    ///
    /// # Arguments
    ///
    /// * `length` - Number of elements in the collection
    /// * `element_size` - Size of each element in bytes
    ///
    /// # Returns
    ///
    /// `Ok(())` if the collection size is valid, or `SerializationError` if limits are exceeded
    ///
    /// # Limits
    ///
    /// - Maximum collection length: 100,000,000 elements
    /// - Maximum memory size: 1,000,000,000 bytes (1GB)
    ///

    fn validate_collection_size(
        &self,
        length: usize,
        element_size: usize,
    ) -> SerializationResult<()> {
        const MAX_COLLECTION_SIZE: usize = 100_000_000; // 100M elements max
        const MAX_MEMORY_SIZE: usize = 1_000_000_000; // 1GB max

        if length > MAX_COLLECTION_SIZE {
            return Err(SerializationError::BinaryFormat {
                message: format!("Collection length {} exceeds maximum allowed", length),
                position: Some(self.bytes_read.saturating_sub(8)),
            });
        }

        let total_size = length.saturating_mul(element_size);
        if total_size > MAX_MEMORY_SIZE {
            return Err(SerializationError::BinaryFormat {
                message: format!(
                    "Collection memory size {} exceeds maximum allowed",
                    total_size
                ),
                position: Some(self.bytes_read.saturating_sub(8)),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Test roundtrip serialization for all basic data types
    ///
    /// Verifies that all primitive types can be read correctly
    /// with proper endianness handling and value preservation.
    #[test]
    fn test_basic_types_reading() {
        let data = vec![
            42, // u8
            0xD2, 0x04, // u16: 1234 in little-endian (0x04D2)
            0x78, 0x56, 0x34, 0x12, // u32: 0x12345678 in little-endian
            0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34,
            0x12, // u64: 0x123456789ABCDEF0 in little-endian
            0xDB, 0x0F, 0x49, 0x40, // f32: PI in little-endian
            1,    // bool: true
        ];

        let mut reader = BinaryReader::new(Cursor::new(&data));
        assert_eq!(reader.read_u8().unwrap(), 42);
        assert_eq!(reader.read_u16().unwrap(), 1234);
        assert_eq!(reader.read_u32().unwrap(), 0x12345678);
        assert_eq!(reader.read_u64().unwrap(), 0x123456789ABCDEF0);
        assert!((reader.read_f32().unwrap() - std::f32::consts::PI).abs() < 1e-6);
        assert!(reader.read_bool().unwrap());
    }

    /// Test string reading functionality
    ///
    /// Verifies that strings can be read correctly with proper
    /// length prefixing and UTF-8 validation.
    #[test]
    fn test_string_reading() {
        let data = vec![
            0x05, 0x00, 0x00, 0x00, // Length: 5
            0x48, 0x65, 0x6C, 0x6C, 0x6F, // "Hello"
        ];

        let mut reader = BinaryReader::new(Cursor::new(&data));
        let text = reader.read_string().unwrap();
        assert_eq!(text, "Hello");
    }

    /// Test vector reading functionality
    ///
    /// Verifies that vectors can be read correctly with proper
    /// length prefixing and data preservation.
    #[test]
    fn test_vector_reading() {
        let data = vec![
            0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Length: 5
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1
            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 2
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 3
            0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 4
            0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 5
        ];

        let mut reader = BinaryReader::new(Cursor::new(&data));
        let vec = reader.read_vec_usize().unwrap();
        assert_eq!(vec, vec![1, 2, 3, 4, 5]);
    }

    /// Test validation limits for collection sizes and memory usage
    ///
    /// Verifies that the binary reader correctly enforces size limits to prevent
    /// memory exhaustion attacks and ensures system stability.
    #[test]
    fn test_validation_limits() {
        let reader = BinaryReader::new(Cursor::new(&[]));

        // Test collection size validation
        let result = reader.validate_collection_size(usize::MAX, 1);
        assert!(result.is_err());

        let result = reader.validate_collection_size(100_000_001, 1);
        assert!(result.is_err());

        let result = reader.validate_collection_size(1000, 1_000_001);
        assert!(result.is_err());

        // Valid sizes should pass
        let result = reader.validate_collection_size(1000, 1000);
        assert!(result.is_ok());
    }

    /// Test roundtrip serialization for all data types
    ///
    /// Verifies that data written by BinaryWriter can be read back correctly
    /// by BinaryReader for all supported data types.
    #[test]
    fn test_roundtrip_serialization() {
        use crate::serialization::binary::BinaryWriter;

        // Test data
        let test_u8 = 42u8;
        let test_u16 = 1234u16;
        let test_u32 = 0x12345678u32;
        let test_u64 = 0x123456789ABCDEF0u64;
        let test_i8 = -42i8;
        let test_i16 = -1234i16;
        let test_i32 = -0x12345678i32;
        let test_i64 = -0x123456789ABCDEF0i64;
        let test_f32 = std::f32::consts::PI;
        let test_bool = true;
        let test_string = "Hello, World!".to_string();
        let test_vec_u8 = vec![1u8, 2, 3, 4, 5];
        let test_vec_f32 = vec![1.0f32, 2.5, std::f32::consts::PI, -1.5];
        let test_option_some = Some(42u32);
        let test_option_none: Option<u32> = None;

        // Write data
        let mut buffer = Vec::new();
        {
            let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));
            writer.write_u8(test_u8).unwrap();
            writer.write_u16(test_u16).unwrap();
            writer.write_u32(test_u32).unwrap();
            writer.write_u64(test_u64).unwrap();
            writer.write_i8(test_i8).unwrap();
            writer.write_i16(test_i16).unwrap();
            writer.write_i32(test_i32).unwrap();
            writer.write_i64(test_i64).unwrap();
            writer.write_f32(test_f32).unwrap();
            writer.write_bool(test_bool).unwrap();
            writer.write_string(&test_string).unwrap();
            writer.write_vec_u8(&test_vec_u8).unwrap();
            writer.write_vec_f32(&test_vec_f32).unwrap();
            writer
                .write_option(&test_option_some, |w, v| w.write_u32(*v))
                .unwrap();
            writer
                .write_option(&test_option_none, |w, v| w.write_u32(*v))
                .unwrap();
        }

        // Read data back
        let mut reader = BinaryReader::new(Cursor::new(&buffer));
        assert_eq!(reader.read_u8().unwrap(), test_u8);
        assert_eq!(reader.read_u16().unwrap(), test_u16);
        assert_eq!(reader.read_u32().unwrap(), test_u32);
        assert_eq!(reader.read_u64().unwrap(), test_u64);
        assert_eq!(reader.read_i8().unwrap(), test_i8);
        assert_eq!(reader.read_i16().unwrap(), test_i16);
        assert_eq!(reader.read_i32().unwrap(), test_i32);
        assert_eq!(reader.read_i64().unwrap(), test_i64);
        assert!((reader.read_f32().unwrap() - test_f32).abs() < 1e-6);
        assert_eq!(reader.read_bool().unwrap(), test_bool);
        assert_eq!(reader.read_string().unwrap(), test_string);
        assert_eq!(reader.read_vec_u8().unwrap(), test_vec_u8);
        assert_eq!(reader.read_vec_f32().unwrap(), test_vec_f32);
        assert_eq!(
            reader.read_option(|r| r.read_u32()).unwrap(),
            test_option_some
        );
        assert_eq!(
            reader.read_option(|r| r.read_u32()).unwrap(),
            test_option_none
        );
    }

    /// Test edge cases and error conditions
    ///
    /// Verifies that the reader correctly handles edge cases and error conditions.
    #[test]
    fn test_edge_cases_and_errors() {
        // Test reading from empty data
        let empty_data = vec![];
        let mut reader = BinaryReader::new(Cursor::new(&empty_data));
        assert!(reader.read_u8().is_err());

        // Test reading string with invalid length
        let invalid_string_data = vec![0xFF, 0xFF, 0xFF, 0xFF]; // Length: 4294967295
        let mut reader = BinaryReader::new(Cursor::new(&invalid_string_data));
        assert!(reader.read_string().is_err());

        // Test reading string with invalid UTF-8
        let invalid_utf8_data = vec![
            0x03, 0x00, 0x00, 0x00, // Length: 3
            0xFF, 0xFE, 0xFD, // Invalid UTF-8 bytes
        ];
        let mut reader = BinaryReader::new(Cursor::new(&invalid_utf8_data));
        assert!(reader.read_string().is_err());

        // Test reading boolean with invalid value
        let invalid_bool_data = vec![2]; // Invalid boolean value
        let mut reader = BinaryReader::new(Cursor::new(&invalid_bool_data));
        assert!(reader.read_bool().is_err());

        // Test reading usize that exceeds platform maximum (only on 32-bit platforms)
        if usize::MAX < u64::MAX as usize {
            let large_usize_data = vec![
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // u64::MAX
            ];
            let mut reader = BinaryReader::new(Cursor::new(&large_usize_data));
            assert!(reader.read_usize().is_err());
        }

        // Test reading vector with excessive size
        let large_vector_data = vec![
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // Length: u64::MAX
        ];
        let mut reader = BinaryReader::new(Cursor::new(&large_vector_data));
        assert!(reader.read_vec_u8().is_err());
    }

    /// Test byte counting accuracy
    ///
    /// Verifies that the byte counting mechanism accurately tracks
    /// the number of bytes read.
    #[test]
    fn test_byte_counting() {
        let data = vec![
            42, // u8: 1 byte
            0xD2, 0x04, // u16: 2 bytes
            0x78, 0x56, 0x34, 0x12, // u32: 4 bytes
            0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12, // u64: 8 bytes
            0x05, 0x00, 0x00, 0x00, // String length: 4 bytes
            0x48, 0x65, 0x6C, 0x6C, 0x6F, // String data: 5 bytes
        ];

        let mut reader = BinaryReader::new(Cursor::new(&data));

        assert_eq!(reader.bytes_read(), 0);

        reader.read_u8().unwrap();
        assert_eq!(reader.bytes_read(), 1);

        reader.read_u16().unwrap();
        assert_eq!(reader.bytes_read(), 3);

        reader.read_u32().unwrap();
        assert_eq!(reader.bytes_read(), 7);

        reader.read_u64().unwrap();
        assert_eq!(reader.bytes_read(), 15);

        reader.read_string().unwrap();
        assert_eq!(reader.bytes_read(), 24);
    }

    /// Test reading with partial data
    ///
    /// Verifies that the reader correctly handles cases where
    /// not enough data is available for a complete read.
    #[test]
    fn test_partial_data_reading() {
        // Test reading u32 with only 3 bytes available
        let partial_data = vec![0x78, 0x56, 0x34];
        let mut reader = BinaryReader::new(Cursor::new(&partial_data));
        assert!(reader.read_u32().is_err());

        // Test reading string with incomplete length
        let incomplete_string_data = vec![0x05, 0x00, 0x00]; // Incomplete length
        let mut reader = BinaryReader::new(Cursor::new(&incomplete_string_data));
        assert!(reader.read_string().is_err());

        // Test reading string with incomplete data
        let incomplete_string_data = vec![
            0x05, 0x00, 0x00, 0x00, // Length: 5
            0x48, 0x65, 0x6C, // Only 3 bytes of "Hello"
        ];
        let mut reader = BinaryReader::new(Cursor::new(&incomplete_string_data));
        assert!(reader.read_string().is_err());
    }

    /// Test reading large data efficiently
    ///
    /// Verifies that the reader can handle large amounts of data
    /// without performance issues.
    #[test]
    fn test_large_data_reading() {
        // Create large test data
        let large_data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        // Write large data
        let mut buffer = Vec::new();
        {
            let mut writer =
                crate::serialization::binary::BinaryWriter::new(Cursor::new(&mut buffer));
            writer.write_vec_u8(&large_data).unwrap();
        }

        // Read large data back
        let mut reader = BinaryReader::new(Cursor::new(&buffer));
        let read_data = reader.read_vec_u8().unwrap();

        assert_eq!(read_data, large_data);
        assert_eq!(read_data.len(), 10000);
    }

    /// Test reading zero-sized data
    ///
    /// Verifies that the reader correctly handles zero-sized
    /// collections and empty data.
    #[test]
    fn test_zero_sized_data() {
        // Test empty vector
        let empty_vector_data = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]; // Length: 0
        let mut reader = BinaryReader::new(Cursor::new(&empty_vector_data));
        let empty_vec = reader.read_vec_u8().unwrap();
        assert_eq!(empty_vec, vec![]);

        // Test empty string
        let empty_string_data = vec![0x00, 0x00, 0x00, 0x00]; // Length: 0
        let mut reader = BinaryReader::new(Cursor::new(&empty_string_data));
        let empty_string = reader.read_string().unwrap();
        assert_eq!(empty_string, "");

        // Test None option
        let none_option_data = vec![0x00]; // false
        let mut reader = BinaryReader::new(Cursor::new(&none_option_data));
        let none_option = reader.read_option(|r| r.read_u32()).unwrap();
        assert_eq!(none_option, None);
    }
}
