//! Binary writer with built-in endianness handling and byte counting
//!
//! This module provides efficient binary writing capabilities with automatic
//! little-endian byte ordering, byte counting, and comprehensive error handling.
//! It wraps any type implementing `Write` and provides a high-level interface
//! for serializing primitive types and collections.

use std::io::Write;

use crate::serialization::core::SerializationResult;

/// Binary writer with built-in endianness handling and byte counting
///
/// This struct provides efficient binary writing capabilities with automatic
/// little-endian byte ordering, byte counting, and comprehensive error handling.
/// It wraps any type implementing `Write` and provides a high-level interface
/// for serializing primitive types and collections.
///
/// # Features
///
/// - **Endianness Handling**: Automatic little-endian conversion for all numeric types
/// - **Byte Counting**: Tracks total bytes written for debugging and validation
/// - **Type Safety**: Strongly typed methods for each data type
/// - **Error Handling**: Comprehensive error propagation through `SerializationResult`
/// - **Generic Design**: Works with any `Write` implementation
///
/// # Thread Safety
///
/// This type is not thread-safe. Access from multiple threads requires external
/// synchronization.
pub struct BinaryWriter<W: Write> {
    /// The underlying writer implementation
    pub(crate) writer: W,
    /// Total number of bytes written
    bytes_written: usize,
}

impl<W: Write> BinaryWriter<W> {
    /// Create a new BinaryWriter wrapping the provided writer
    ///
    /// # Arguments
    ///
    /// * `writer` - The underlying writer implementation
    ///
    /// # Returns
    ///
    /// A new BinaryWriter instance with byte counting initialized to zero
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            bytes_written: 0,
        }
    }

    /// Get the total number of bytes written so far
    ///
    /// This method returns the cumulative count of all bytes written through
    /// this BinaryWriter instance. This is useful for debugging, validation,
    /// and determining the size of serialized data.
    ///
    /// # Returns
    ///
    /// The total number of bytes written
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }

    /// Write an 8-bit unsigned integer
    ///
    /// Writes a single byte representing the unsigned 8-bit integer value.
    /// No endianness conversion is needed for single-byte values.
    ///
    /// # Arguments
    ///
    /// * `value` - The 8-bit unsigned integer to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_u8(&mut self, value: u8) -> SerializationResult<()> {
        self.writer.write_all(&[value])?;
        self.bytes_written += 1;
        Ok(())
    }

    /// Write a 16-bit unsigned integer in little-endian format
    ///
    /// Writes two bytes representing the unsigned 16-bit integer value in
    /// little-endian byte order for cross-platform compatibility.
    ///
    /// # Arguments
    ///
    /// * `value` - The 16-bit unsigned integer to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_u16(&mut self, value: u16) -> SerializationResult<()> {
        self.writer.write_all(&value.to_le_bytes())?;
        self.bytes_written += 2;
        Ok(())
    }

    /// Write a 32-bit unsigned integer in little-endian format
    ///
    /// Writes four bytes representing the unsigned 32-bit integer value in
    /// little-endian byte order for cross-platform compatibility.
    ///
    /// # Arguments
    ///
    /// * `value` - The 32-bit unsigned integer to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    ///

    pub fn write_u32(&mut self, value: u32) -> SerializationResult<()> {
        self.writer.write_all(&value.to_le_bytes())?;
        self.bytes_written += 4;
        Ok(())
    }

    /// Write a 64-bit unsigned integer in little-endian format
    ///
    /// Writes eight bytes representing the unsigned 64-bit integer value in
    /// little-endian byte order for cross-platform compatibility.
    ///
    /// # Arguments
    ///
    /// * `value` - The 64-bit unsigned integer to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_u64(&mut self, value: u64) -> SerializationResult<()> {
        self.writer.write_all(&value.to_le_bytes())?;
        self.bytes_written += 8;
        Ok(())
    }

    /// Write an 8-bit signed integer
    ///
    /// Writes a single byte representing the signed 8-bit integer value.
    /// No endianness conversion is needed for single-byte values.
    ///
    /// # Arguments
    ///
    /// * `value` - The 8-bit signed integer to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_i8(&mut self, value: i8) -> SerializationResult<()> {
        self.write_u8(value as u8)
    }

    /// Write a 16-bit signed integer in little-endian format
    ///
    /// Writes two bytes representing the signed 16-bit integer value in
    /// little-endian byte order for cross-platform compatibility.
    ///
    /// # Arguments
    ///
    /// * `value` - The 16-bit signed integer to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_i16(&mut self, value: i16) -> SerializationResult<()> {
        self.write_u16(value as u16)
    }

    /// Write a 32-bit signed integer in little-endian format
    ///
    /// Writes four bytes representing the signed 32-bit integer value in
    /// little-endian byte order for cross-platform compatibility.
    ///
    /// # Arguments
    ///
    /// * `value` - The 32-bit signed integer to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_i32(&mut self, value: i32) -> SerializationResult<()> {
        self.write_u32(value as u32)
    }

    /// Write a 64-bit signed integer in little-endian format
    ///
    /// Writes eight bytes representing the signed 64-bit integer value in
    /// little-endian byte order for cross-platform compatibility.
    ///
    /// # Arguments
    ///
    /// * `value` - The 64-bit signed integer to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_i64(&mut self, value: i64) -> SerializationResult<()> {
        self.write_u64(value as u64)
    }

    /// Write a 32-bit floating point number in little-endian format
    ///
    /// Writes four bytes representing the 32-bit floating point value in
    /// little-endian byte order for cross-platform compatibility.
    ///
    /// # Arguments
    ///
    /// * `value` - The 32-bit floating point number to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_f32(&mut self, value: f32) -> SerializationResult<()> {
        self.write_u32(value.to_bits())
    }

    /// Write a 64-bit floating point number in little-endian format
    ///
    /// Writes eight bytes representing a 64-bit floating point value in
    /// little-endian byte order by converting to its bit representation.
    ///
    /// # Arguments
    ///
    /// * `value` - The 64-bit floating point number to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_f64(&mut self, value: f64) -> SerializationResult<()> {
        self.write_u64(value.to_bits())
    }

    /// Write a platform-specific size type in little-endian format
    ///
    /// Writes eight bytes representing the platform-specific size value in
    /// little-endian byte order. This ensures cross-platform compatibility
    /// regardless of the platform's native size type.
    ///
    /// # Arguments
    ///
    /// * `value` - The platform-specific size value to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_usize(&mut self, value: usize) -> SerializationResult<()> {
        self.write_u64(value as u64)
    }

    /// Write a boolean value
    ///
    /// Writes a single byte representing the boolean value (0 = false, 1 = true).
    /// No endianness conversion is needed for single-byte values.
    ///
    /// # Arguments
    ///
    /// * `value` - The boolean value to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_bool(&mut self, value: bool) -> SerializationResult<()> {
        self.write_u8(if value { 1 } else { 0 })
    }

    /// Write a UTF-8 string with length prefix
    ///
    /// Writes a string by first writing its length as a 32-bit unsigned integer,
    /// followed by the UTF-8 bytes of the string. The length prefix allows for
    /// efficient string reading during deserialization.
    ///
    /// # Arguments
    ///
    /// * `value` - The UTF-8 string to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    ///

    pub fn write_string(&mut self, value: &str) -> SerializationResult<()> {
        let bytes = value.as_bytes();
        self.write_u32(bytes.len() as u32)?;
        self.writer.write_all(bytes)?;
        self.bytes_written += bytes.len();
        Ok(())
    }

    /// Write raw bytes without length prefix
    ///
    /// Writes the provided bytes directly to the output without any length prefix.
    /// This is useful for writing raw binary data or when the length is known
    /// from context.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The raw bytes to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_bytes(&mut self, bytes: &[u8]) -> SerializationResult<()> {
        self.writer.write_all(bytes)?;
        self.bytes_written += bytes.len();
        Ok(())
    }

    /// Write a vector of 8-bit unsigned integers with length prefix
    ///
    /// Writes a vector by first writing its length as a 64-bit unsigned integer,
    /// followed by the raw bytes of the vector data.
    ///
    /// # Arguments
    ///
    /// * `vec` - The vector of 8-bit unsigned integers to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    pub fn write_vec_u8(&mut self, vec: &[u8]) -> SerializationResult<()> {
        self.write_u64(vec.len() as u64)?;
        self.write_bytes(vec)
    }

    /// Write a vector of platform-specific size types with length prefix
    ///
    /// Writes a vector by first writing its length as a 64-bit unsigned integer,
    /// followed by each size value in little-endian format.
    ///
    /// # Arguments
    ///
    /// * `vec` - The vector of platform-specific size values to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    #[allow(unused)]
    pub fn write_vec_usize(&mut self, vec: &[usize]) -> SerializationResult<()> {
        self.write_u64(vec.len() as u64)?;
        for &value in vec {
            self.write_usize(value)?;
        }
        Ok(())
    }

    /// Write a vector of 32-bit floating point numbers with length prefix
    ///
    /// Writes a vector by first writing its length as a 64-bit unsigned integer,
    /// followed by each floating point value in little-endian format.
    ///
    /// # Arguments
    ///
    /// * `vec` - The vector of 32-bit floating point numbers to write
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    #[allow(unused)]
    pub fn write_vec_f32(&mut self, vec: &[f32]) -> SerializationResult<()> {
        self.write_u64(vec.len() as u64)?;
        for &value in vec {
            self.write_f32(value)?;
        }
        Ok(())
    }

    /// Write an optional value with presence flag
    ///
    /// Writes an optional value by first writing a boolean flag indicating whether
    /// the value is present, followed by the actual value if present. This allows
    /// for efficient handling of nullable or optional data.
    ///
    /// # Arguments
    ///
    /// * `option` - The optional value to write
    /// * `write_fn` - Function to write the contained value if present
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    ///

    #[allow(unused)]
    pub fn write_option<T, F>(&mut self, option: &Option<T>, write_fn: F) -> SerializationResult<()>
    where
        F: FnOnce(&mut Self, &T) -> SerializationResult<()>,
    {
        match option {
            Some(value) => {
                self.write_bool(true)?;
                write_fn(self, value)
            }
            None => self.write_bool(false),
        }
    }

    /// Flush the underlying writer
    ///
    /// Ensures that all buffered data is written to the underlying writer.
    /// This is useful for ensuring data is written immediately.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `SerializationError` on failure
    #[allow(unused)]
    pub fn flush(&mut self) -> SerializationResult<()> {
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Test roundtrip serialization for all basic data types
    ///
    /// Verifies that all primitive types can be written correctly
    /// with proper endianness handling and value preservation.
    #[test]
    fn test_basic_types_writing() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        writer.write_u8(42).unwrap();
        writer.write_u16(1234).unwrap();
        writer.write_u32(0x12345678).unwrap();
        writer.write_u64(0x123456789ABCDEF0).unwrap();
        writer.write_f32(std::f32::consts::PI).unwrap();
        writer.write_bool(true).unwrap();
        writer.write_string("Hello, World!").unwrap();

        // Verify byte count
        assert_eq!(writer.bytes_written(), 1 + 2 + 4 + 8 + 4 + 1 + 4 + 13);
    }

    /// Test vector writing functionality
    ///
    /// Verifies that vectors can be written correctly with proper
    /// length prefixing and data preservation.
    #[test]
    fn test_vector_writing() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        let test_vec_usize = vec![1, 2, 3, 4, 5];
        let test_vec_f32 = vec![1.0, 2.5, std::f32::consts::PI, -1.5];

        writer.write_vec_usize(&test_vec_usize).unwrap();
        writer.write_vec_f32(&test_vec_f32).unwrap();

        // Verify byte count (length + data)
        assert_eq!(writer.bytes_written(), 8 + 5 * 8 + 8 + 4 * 4);
    }

    /// Test optional value writing
    ///
    /// Verifies that optional values can be written correctly with
    /// presence flags and proper value handling.
    #[test]
    fn test_option_writing() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        let test_option_some = Some(42usize);
        let test_option_none: Option<usize> = None;

        writer
            .write_option(&test_option_some, |w, v| w.write_usize(*v))
            .unwrap();
        writer
            .write_option(&test_option_none, |w, v: &usize| w.write_usize(*v))
            .unwrap();

        // Verify byte count (bool + usize for Some, bool for None)
        assert_eq!(writer.bytes_written(), 1 + 8 + 1);
    }

    /// Test roundtrip serialization for all data types
    ///
    /// Verifies that data written by BinaryWriter can be read back correctly
    /// by BinaryReader for all supported data types.
    #[test]
    fn test_roundtrip_serialization() {
        use crate::serialization::binary::BinaryReader;

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

    /// Test edge cases and boundary values
    ///
    /// Verifies that edge cases and boundary values are written correctly.
    #[test]
    fn test_edge_cases_and_boundaries() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        // Test numeric boundaries
        writer.write_i8(i8::MIN).unwrap();
        writer.write_i8(i8::MAX).unwrap();
        writer.write_i16(i16::MIN).unwrap();
        writer.write_i16(i16::MAX).unwrap();
        writer.write_i32(i32::MIN).unwrap();
        writer.write_i32(i32::MAX).unwrap();
        writer.write_i64(i64::MIN).unwrap();
        writer.write_i64(i64::MAX).unwrap();
        writer.write_u8(u8::MAX).unwrap();
        writer.write_u16(u16::MAX).unwrap();
        writer.write_u32(u32::MAX).unwrap();
        writer.write_u64(u64::MAX).unwrap();
        writer.write_usize(usize::MAX).unwrap();

        // Test floating point special values
        writer.write_f32(f32::INFINITY).unwrap();
        writer.write_f32(f32::NEG_INFINITY).unwrap();
        writer.write_f32(f32::NAN).unwrap();

        // Test empty collections
        writer.write_string("").unwrap();
        writer.write_vec_u8(&[]).unwrap();
        writer.write_vec_f32(&[]).unwrap();

        // Test boolean values
        writer.write_bool(false).unwrap();
        writer.write_bool(true).unwrap();

        // Verify all data was written
        assert!(writer.bytes_written() > 0);
    }

    /// Test large data writing
    ///
    /// Verifies that large amounts of data can be written efficiently.
    #[test]
    fn test_large_data_writing() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        // Create large test data
        let large_vec_u8: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let large_vec_f32: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let large_string: String = (0..1000)
            .map(|i| format!("item_{}", i))
            .collect::<Vec<_>>()
            .join(", ");

        // Write large data
        writer.write_vec_u8(&large_vec_u8).unwrap();
        writer.write_vec_f32(&large_vec_f32).unwrap();
        writer.write_string(&large_string).unwrap();

        // Verify data was written
        assert_eq!(
            writer.bytes_written(),
            8 + 10000 + 8 + 1000 * 4 + 4 + large_string.len()
        );
    }

    /// Test byte counting accuracy
    ///
    /// Verifies that the byte counting mechanism accurately tracks
    /// the number of bytes written.
    #[test]
    fn test_byte_counting() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        assert_eq!(writer.bytes_written(), 0);

        writer.write_u8(42).unwrap();
        assert_eq!(writer.bytes_written(), 1);

        writer.write_u16(1234).unwrap();
        assert_eq!(writer.bytes_written(), 3);

        writer.write_u32(0x12345678).unwrap();
        assert_eq!(writer.bytes_written(), 7);

        writer.write_u64(0x123456789ABCDEF0).unwrap();
        assert_eq!(writer.bytes_written(), 15);

        writer.write_string("Hello").unwrap();
        assert_eq!(writer.bytes_written(), 24); // 15 + 4 (length) + 5 (string)

        writer.write_vec_u8(&[1, 2, 3]).unwrap();
        assert_eq!(writer.bytes_written(), 35); // 24 + 8 (length) + 3 (data)
    }

    /// Test endianness consistency
    ///
    /// Verifies that all numeric values are written in little-endian format.
    #[test]
    fn test_endianness_consistency() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        // Write known values
        writer.write_u16(0x1234).unwrap();
        writer.write_u32(0x12345678).unwrap();
        writer.write_u64(0x123456789ABCDEF0).unwrap();

        let final_buffer = writer.writer.into_inner();

        // Verify little-endian format
        assert_eq!(final_buffer[0], 0x34); // u16: 0x1234 in little-endian
        assert_eq!(final_buffer[1], 0x12);
        assert_eq!(final_buffer[2], 0x78); // u32: 0x12345678 in little-endian
        assert_eq!(final_buffer[3], 0x56);
        assert_eq!(final_buffer[4], 0x34);
        assert_eq!(final_buffer[5], 0x12);
        assert_eq!(final_buffer[6], 0xF0); // u64: 0x123456789ABCDEF0 in little-endian
        assert_eq!(final_buffer[7], 0xDE);
        assert_eq!(final_buffer[8], 0xBC);
        assert_eq!(final_buffer[9], 0x9A);
        assert_eq!(final_buffer[10], 0x78);
        assert_eq!(final_buffer[11], 0x56);
        assert_eq!(final_buffer[12], 0x34);
        assert_eq!(final_buffer[13], 0x12);
    }

    /// Test string encoding
    ///
    /// Verifies that strings are written with proper UTF-8 encoding and length prefixing.
    #[test]
    fn test_string_encoding() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        // Test various string types
        let test_strings = vec![
            "Hello, World!",
            "Unicode: 你好世界",
            "Special chars: \n\t\r\\\"",
            "", // Empty string
        ];

        for s in &test_strings {
            writer.write_string(s).unwrap();
        }

        let final_buffer = writer.writer.into_inner();

        // Verify string encoding
        let mut offset = 0;
        for s in &test_strings {
            // Read length
            let length = u32::from_le_bytes([
                final_buffer[offset],
                final_buffer[offset + 1],
                final_buffer[offset + 2],
                final_buffer[offset + 3],
            ]);
            assert_eq!(length as usize, s.len());

            // Read string data
            let string_data = &final_buffer[offset + 4..offset + 4 + s.len()];
            let decoded_string = std::str::from_utf8(string_data).unwrap();
            assert_eq!(decoded_string, *s);

            offset += 4 + s.len();
        }
    }

    /// Test vector encoding
    ///
    /// Verifies that vectors are written with proper length prefixing and data preservation.
    #[test]
    fn test_vector_encoding() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        // Test various vector types
        let test_vec_u8 = vec![1u8, 2, 3, 4, 5];
        let test_vec_f32 = vec![1.0f32, 2.5, std::f32::consts::PI, -1.5];
        let test_vec_usize = vec![1usize, 2, 3, 4, 5];

        writer.write_vec_u8(&test_vec_u8).unwrap();
        writer.write_vec_f32(&test_vec_f32).unwrap();
        writer.write_vec_usize(&test_vec_usize).unwrap();

        let final_buffer = writer.writer.into_inner();

        // Verify vector encoding
        let mut offset = 0;

        // Check u8 vector
        let length = u64::from_le_bytes([
            final_buffer[offset],
            final_buffer[offset + 1],
            final_buffer[offset + 2],
            final_buffer[offset + 3],
            final_buffer[offset + 4],
            final_buffer[offset + 5],
            final_buffer[offset + 6],
            final_buffer[offset + 7],
        ]);
        assert_eq!(length, test_vec_u8.len() as u64);
        let vec_data = &final_buffer[offset + 8..offset + 8 + test_vec_u8.len()];
        assert_eq!(vec_data, test_vec_u8.as_slice());
        offset += 8 + test_vec_u8.len();

        // Check f32 vector
        let length = u64::from_le_bytes([
            final_buffer[offset],
            final_buffer[offset + 1],
            final_buffer[offset + 2],
            final_buffer[offset + 3],
            final_buffer[offset + 4],
            final_buffer[offset + 5],
            final_buffer[offset + 6],
            final_buffer[offset + 7],
        ]);
        assert_eq!(length, test_vec_f32.len() as u64);
        offset += 8 + test_vec_f32.len() * 4;

        // Check usize vector
        let length = u64::from_le_bytes([
            final_buffer[offset],
            final_buffer[offset + 1],
            final_buffer[offset + 2],
            final_buffer[offset + 3],
            final_buffer[offset + 4],
            final_buffer[offset + 5],
            final_buffer[offset + 6],
            final_buffer[offset + 7],
        ]);
        assert_eq!(length, test_vec_usize.len() as u64);
    }

    /// Test option encoding
    ///
    /// Verifies that optional values are written with proper presence flags.
    #[test]
    fn test_option_encoding() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        // Test Some and None values
        let some_value = Some(42u32);
        let none_value: Option<u32> = None;

        writer
            .write_option(&some_value, |w, v| w.write_u32(*v))
            .unwrap();
        writer
            .write_option(&none_value, |w, v| w.write_u32(*v))
            .unwrap();

        let final_buffer = writer.writer.into_inner();

        // Verify option encoding
        assert_eq!(final_buffer[0], 1); // Some flag
        let value = u32::from_le_bytes([
            final_buffer[1],
            final_buffer[2],
            final_buffer[3],
            final_buffer[4],
        ]);
        assert_eq!(value, 42);
        assert_eq!(final_buffer[5], 0); // None flag
    }

    /// Test flush functionality
    ///
    /// Verifies that the flush method works correctly.
    #[test]
    fn test_flush_functionality() {
        let mut buffer = Vec::new();
        let mut writer = BinaryWriter::new(Cursor::new(&mut buffer));

        // Write some data
        writer.write_u32(42).unwrap();
        writer.write_string("test").unwrap();

        // Flush should not fail
        writer.flush().unwrap();

        // Verify data is still intact
        assert_eq!(writer.bytes_written(), 4 + 4 + 4); // u32 + string length + string data
    }
}
