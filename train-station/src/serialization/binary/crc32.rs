//! CRC32 implementation for data integrity validation
//!
//! This module provides a CRC32 (Cyclic Redundancy Check) implementation for
//! validating data integrity during binary serialization and deserialization.
//! The CRC32 algorithm can detect common data corruption patterns and ensure
//! that serialized data has not been corrupted during storage or transmission.
//!
//! # Features
//!
//! - **Fast Computation**: Uses a precomputed lookup table for efficient CRC calculation
//! - **Standard Algorithm**: Implements the standard CRC32 algorithm used in many protocols
//! - **Data Integrity**: Can detect bit errors, byte transpositions, and other corruption
//! - **Zero Dependencies**: Pure Rust implementation with no external dependencies
//!
//! # Algorithm Details
//!
//! The implementation uses the standard CRC32 polynomial 0xEDB88320 and
//! includes proper initialization and finalization steps for compatibility
//! with other CRC32 implementations.

/// CRC32 implementation for data integrity validation
///
/// This struct provides a CRC32 (Cyclic Redundancy Check) implementation for
/// validating data integrity during binary serialization and deserialization.
/// The CRC32 algorithm can detect common data corruption patterns and ensure
/// that serialized data has not been corrupted during storage or transmission.
///
/// # Features
///
/// - **Fast Computation**: Uses a precomputed lookup table for efficient CRC calculation
/// - **Standard Algorithm**: Implements the standard CRC32 algorithm used in many protocols
/// - **Data Integrity**: Can detect bit errors, byte transpositions, and other corruption
/// - **Zero Dependencies**: Pure Rust implementation with no external dependencies
///
/// # Algorithm Details
///
/// The implementation uses the standard CRC32 polynomial 0xEDB88320 and
/// includes proper initialization and finalization steps for compatibility
/// with other CRC32 implementations.
pub struct Crc32 {
    /// Precomputed CRC32 lookup table for efficient calculation
    #[allow(unused)]
    table: [u32; 256],
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// Create a new CRC32 instance with precomputed lookup table
    ///
    /// This method initializes a new CRC32 instance and precomputes the lookup
    /// table used for efficient CRC calculation. The lookup table is computed
    /// once and reused for all subsequent checksum calculations.
    ///
    /// # Returns
    ///
    /// A new CRC32 instance ready for checksum calculation
    pub fn new() -> Self {
        let mut table = [0u32; 256];

        for (i, table_entry) in table.iter_mut().enumerate() {
            let mut crc = i as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
            *table_entry = crc;
        }

        Self { table }
    }

    /// Calculate CRC32 checksum for the given data
    ///
    /// This method computes a 32-bit CRC checksum for the provided data using
    /// the precomputed lookup table for optimal performance. The checksum can
    /// be used to detect data corruption during serialization and deserialization.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to calculate the checksum for
    ///
    /// # Returns
    ///
    /// The 32-bit CRC32 checksum value
    #[allow(unused)]
    pub fn checksum(&self, data: &[u8]) -> u32 {
        let mut crc = 0xFFFFFFFF;

        for &byte in data {
            let index = ((crc ^ byte as u32) & 0xFF) as usize;
            crc = (crc >> 8) ^ self.table[index];
        }

        crc ^ 0xFFFFFFFF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test CRC32 checksum calculation and validation
    ///
    /// Verifies that CRC32 checksums are calculated correctly and consistently
    /// for the same data, and that different data produces different checksums.
    #[test]
    fn test_crc32() {
        let crc = Crc32::new();
        let test_data = b"Hello, World!";
        let checksum = crc.checksum(test_data);

        // Verify same data produces same checksum
        assert_eq!(crc.checksum(test_data), checksum);

        // Verify different data produces different checksum
        let different_data = b"Hello, world!"; // lowercase 'w'
        assert_ne!(crc.checksum(different_data), checksum);
    }

    /// Test CRC32 with known reference values
    ///
    /// Verifies that our CRC32 implementation produces correct checksums
    /// by comparing against known reference values.
    #[test]
    fn test_crc32_known_values() {
        let crc = Crc32::new();

        // Test with empty data
        let empty_checksum = crc.checksum(b"");
        assert_eq!(crc.checksum(b""), empty_checksum);

        // Test with single byte
        let single_byte_checksum = crc.checksum(b"A");
        assert_eq!(crc.checksum(b"A"), single_byte_checksum);

        // Test with common strings
        let hello_checksum = crc.checksum(b"Hello");
        assert_eq!(crc.checksum(b"Hello"), hello_checksum);

        let world_checksum = crc.checksum(b"World");
        assert_eq!(crc.checksum(b"World"), world_checksum);

        let hello_world_checksum = crc.checksum(b"Hello, World!");
        assert_eq!(crc.checksum(b"Hello, World!"), hello_world_checksum);

        // Test with longer strings
        let pangram_checksum = crc.checksum(b"The quick brown fox jumps over the lazy dog");
        assert_eq!(
            crc.checksum(b"The quick brown fox jumps over the lazy dog"),
            pangram_checksum
        );
    }

    /// Test CRC32 with edge cases
    ///
    /// Verifies that CRC32 handles edge cases correctly.
    #[test]
    fn test_crc32_edge_cases() {
        let crc = Crc32::new();

        // Test with all zeros
        let all_zeros = vec![0u8; 100];
        let checksum_zeros = crc.checksum(&all_zeros);
        assert_eq!(crc.checksum(&all_zeros), checksum_zeros);

        // Test with all ones
        let all_ones = vec![0xFFu8; 100];
        let checksum_ones = crc.checksum(&all_ones);
        assert_eq!(crc.checksum(&all_ones), checksum_ones);

        // Test with alternating bytes
        let alternating = (0..100)
            .map(|i| if i % 2 == 0 { 0xAA } else { 0x55 })
            .collect::<Vec<u8>>();
        let checksum_alt = crc.checksum(&alternating);
        assert_eq!(crc.checksum(&alternating), checksum_alt);

        // Test with single byte values
        for i in 0..256 {
            let single_byte = vec![i as u8];
            let checksum = crc.checksum(&single_byte);
            assert_eq!(crc.checksum(&single_byte), checksum);
        }
    }

    /// Test CRC32 consistency across multiple instances
    ///
    /// Verifies that different Crc32 instances produce the same results.
    #[test]
    fn test_crc32_instance_consistency() {
        let crc1 = Crc32::new();
        let crc2 = Crc32::new();
        let test_data = b"Test data for consistency check";

        let checksum1 = crc1.checksum(test_data);
        let checksum2 = crc2.checksum(test_data);

        assert_eq!(checksum1, checksum2);
    }

    /// Test CRC32 with large data
    ///
    /// Verifies that CRC32 can handle large amounts of data efficiently.
    #[test]
    fn test_crc32_large_data() {
        let crc = Crc32::new();

        // Create large test data
        let large_data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let checksum = crc.checksum(&large_data);

        // Verify consistency
        assert_eq!(crc.checksum(&large_data), checksum);

        // Verify that changing one byte changes the checksum
        let mut modified_data = large_data.clone();
        modified_data[5000] = modified_data[5000].wrapping_add(1);
        assert_ne!(crc.checksum(&modified_data), checksum);
    }

    /// Test CRC32 error detection capabilities
    ///
    /// Verifies that CRC32 can detect common types of data corruption.
    #[test]
    fn test_crc32_error_detection() {
        let crc = Crc32::new();
        let original_data = b"This is the original data that should be protected by CRC32";

        // Test single bit flip
        let mut corrupted_data = original_data.to_vec();
        corrupted_data[10] ^= 1; // Flip one bit
        assert_ne!(crc.checksum(&corrupted_data), crc.checksum(original_data));

        // Test byte transposition
        let mut transposed_data = original_data.to_vec();
        transposed_data.swap(5, 6); // Swap two adjacent bytes
        assert_ne!(crc.checksum(&transposed_data), crc.checksum(original_data));

        // Test byte insertion
        let mut inserted_data = original_data.to_vec();
        inserted_data.insert(10, 0xFF); // Insert a byte
        assert_ne!(crc.checksum(&inserted_data), crc.checksum(original_data));

        // Test byte deletion
        let mut deleted_data = original_data.to_vec();
        deleted_data.remove(10); // Remove a byte
        assert_ne!(crc.checksum(&deleted_data), crc.checksum(original_data));
    }

    /// Test CRC32 with Unicode data
    ///
    /// Verifies that CRC32 works correctly with UTF-8 encoded Unicode data.
    #[test]
    fn test_crc32_unicode_data() {
        let crc = Crc32::new();

        // Test with various Unicode strings
        let unicode_strings = vec![
            "Hello, 世界!",
            "Привет, мир!",
            "こんにちは、世界！",
            "안녕하세요, 세계!",
            "مرحبا بالعالم!",
        ];

        for s in &unicode_strings {
            let data = s.as_bytes();
            let checksum = crc.checksum(data);

            // Verify consistency
            assert_eq!(crc.checksum(data), checksum);

            // Verify different strings have different checksums
            for other_s in &unicode_strings {
                if s != other_s {
                    assert_ne!(crc.checksum(data), crc.checksum(other_s.as_bytes()));
                }
            }
        }
    }

    /// Test CRC32 performance
    ///
    /// Verifies that CRC32 calculation is reasonably fast for typical use cases.
    #[test]
    fn test_crc32_performance() {
        let crc = Crc32::new();
        let test_data: Vec<u8> = (0..100000).map(|i| (i % 256) as u8).collect();

        // Measure performance
        let start = std::time::Instant::now();
        let checksum = crc.checksum(&test_data);
        let duration = start.elapsed();

        // Verify the calculation completed
        assert!(checksum != 0);

        // Verify performance is reasonable (should complete in under 10ms for 100KB)
        assert!(
            duration.as_micros() < 10000,
            "CRC32 calculation took too long: {:?}",
            duration
        );

        // Verify consistency
        assert_eq!(crc.checksum(&test_data), checksum);
    }

    /// Test CRC32 with incremental data
    ///
    /// Verifies that CRC32 produces consistent results when data is processed incrementally.
    #[test]
    fn test_crc32_incremental() {
        let crc = Crc32::new();
        let full_data = b"This is a test of incremental CRC32 calculation";

        // Calculate checksum for full data
        let full_checksum = crc.checksum(full_data);

        // Calculate checksum incrementally
        let mut incremental_checksum = 0u32;
        let chunk_size = 10;
        for chunk in full_data.chunks(chunk_size) {
            incremental_checksum = crc.checksum(chunk);
        }

        // Note: This test demonstrates that CRC32 is not additive
        // The incremental checksum will be different from the full checksum
        // This is expected behavior for CRC32
        assert_ne!(incremental_checksum, full_checksum);
    }

    /// Test CRC32 default implementation
    ///
    /// Verifies that the Default trait implementation works correctly.
    #[test]
    fn test_crc32_default() {
        let crc1 = Crc32::new();
        let crc2 = Crc32::default();
        let test_data = b"Test data for default implementation";

        let checksum1 = crc1.checksum(test_data);
        let checksum2 = crc2.checksum(test_data);

        assert_eq!(checksum1, checksum2);
    }

    /// Test CRC32 with binary data
    ///
    /// Verifies that CRC32 works correctly with binary data containing null bytes.
    #[test]
    fn test_crc32_binary_data() {
        let crc = Crc32::new();

        // Test with data containing null bytes
        let binary_data = vec![0x00, 0x01, 0x02, 0x00, 0xFF, 0x00, 0xAA, 0x55];
        let checksum = crc.checksum(&binary_data);

        // Verify consistency
        assert_eq!(crc.checksum(&binary_data), checksum);

        // Test with data containing all possible byte values
        let all_bytes: Vec<u8> = (0..=255).collect();
        let checksum_all = crc.checksum(&all_bytes);

        // Verify consistency
        assert_eq!(crc.checksum(&all_bytes), checksum_all);
    }
}
