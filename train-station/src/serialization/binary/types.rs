//! Object type identifiers and format constants for binary serialization
//!
//! This module defines the type identifiers used in the binary format to distinguish
//! between different serializable objects. Each variant has a unique 32-bit identifier
//! that is written to the binary file header for type validation during deserialization.

use crate::serialization::core::{SerializationError, SerializationResult};

/// Magic number for Train Station binary format
///
/// This constant identifies Train Station binary files. The value 0x54535F42
/// corresponds to the ASCII string "TS_B" (Train Station Binary).
pub const MAGIC_NUMBER: u32 = 0x54535F42; // "TS_B" in ASCII

/// Current binary format version
///
/// This constant defines the current version of the binary format specification.
/// Version mismatches during deserialization will result in an error to ensure
/// compatibility and prevent data corruption.
pub const FORMAT_VERSION: u32 = 1;

/// Object type identifiers for binary serialization
///
/// This enum defines the type identifiers used in the binary format to distinguish
/// between different serializable objects. Each variant has a unique 32-bit identifier
/// that is written to the binary file header for type validation during deserialization.
///
/// # Variants
///
/// * `Tensor` - Tensor data and metadata (0x1001)
/// * `Adam` - Adam optimizer state (0x1002)
/// * `AdamConfig` - Adam optimizer configuration (0x1003)
/// * `Shape` - Tensor shape information (0x1004)
/// * `ParameterState` - Individual parameter state (0x1005)
///
/// # Binary Format
///
/// Object types are stored as 32-bit little-endian values in the binary file header,
/// allowing for up to 4 billion unique object types. The current range 0x1000-0x1FFF
/// is reserved for Train Station objects.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    /// Tensor data and metadata
    Tensor = 0x1001,
    /// Adam optimizer state
    Adam = 0x1002,
    /// Adam optimizer configuration
    AdamConfig = 0x1003,
    /// Tensor shape information
    Shape = 0x1004,
    /// Individual parameter state
    ParameterState = 0x1005,
}

impl ObjectType {
    /// Convert a 32-bit value to an ObjectType
    ///
    /// This function attempts to convert a raw 32-bit value into an ObjectType
    /// variant. If the value doesn't correspond to a known object type, an error
    /// is returned with details about the unknown value.
    ///
    /// # Arguments
    ///
    /// * `value` - The 32-bit value to convert
    ///
    /// # Returns
    ///
    /// The corresponding ObjectType on success, or `SerializationError` if the
    /// value is unknown
    pub fn from_u32(value: u32) -> SerializationResult<Self> {
        match value {
            0x1001 => Ok(ObjectType::Tensor),
            0x1002 => Ok(ObjectType::Adam),
            0x1003 => Ok(ObjectType::AdamConfig),
            0x1004 => Ok(ObjectType::Shape),
            0x1005 => Ok(ObjectType::ParameterState),
            _ => Err(SerializationError::BinaryFormat {
                message: format!("Unknown object type: 0x{:08X}", value),
                position: None,
            }),
        }
    }
}
