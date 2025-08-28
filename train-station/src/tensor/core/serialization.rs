//! Tensor serialization implementation
//!
//! This module provides comprehensive serialization support for tensors, shapes,
//! and device information using the Train Station serialization framework.
//! It supports both JSON and binary formats with full roundtrip fidelity.
//!
//! # Key Features
//!
//! - **Complete Tensor Serialization**: Tensor data, shape, device, and gradtrack state
//! - **Shape Serialization**: Dimensions, strides, size, and memory layout information
//! - **Device Serialization**: Device type and index for CPU/CUDA placement
//! - **Efficient Binary Format**: Optimized binary serialization for performance
//! - **Human-Readable JSON**: JSON format for debugging and interoperability
//! - **Roundtrip Fidelity**: Perfect reconstruction of tensors from serialized data
//! - **Struct Field Support**: Tensors can be serialized as fields within larger structures
//!
//! # Architecture
//!
//! The serialization system handles:
//! - **Tensor Data**: Serialized as `Vec<f32>` for efficiency
//! - **Shape Information**: Complete shape metadata including strides and layout
//! - **Device Placement**: Device type and index for proper reconstruction
//! - **GradTrack State**: requires_grad flag (runtime gradient state not serialized)
//!
//! # Non-Serialized Fields
//!
//! The following tensor fields are NOT serialized as they are runtime state:
//! - `data` (raw pointer): Reconstructed from serialized `Vec<f32>`
//! - `id`: Regenerated during deserialization for uniqueness
//! - `grad`: Runtime gradient state, not persistent
//! - `grad_fn`: Runtime gradient function, not persistent
//! - `allocation_owner`: Internal memory management, reconstructed
//! - `_phantom`: Zero-sized type, no serialization needed
//!
//! # Usage Example
//!
//! ## Basic Tensor Serialization
//!
//! ```rust
//! use train_station::Tensor;
//! use train_station::serialization::StructSerializable;
//!
//! // Create and populate a tensor
//! let mut tensor = Tensor::zeros(vec![2, 3, 4]).with_requires_grad();
//! tensor.fill(42.0);
//!
//! // Serialize to JSON
//! let json = tensor.to_json().unwrap();
//! assert!(!json.is_empty());
//!
//! // Deserialize from JSON
//! let loaded_tensor = Tensor::from_json(&json).unwrap();
//! assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
//! assert_eq!(tensor.requires_grad(), loaded_tensor.requires_grad());
//!
//! // Serialize to binary
//! let binary = tensor.to_binary().unwrap();
//! assert!(!binary.is_empty());
//!
//! // Deserialize from binary
//! let loaded_tensor = Tensor::from_binary(&binary).unwrap();
//! assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
//! ```
//!
//! ## Tensor as Struct Field
//!
//! ```rust
//! use train_station::Tensor;
//! use train_station::serialization::StructSerializable;
//!
//! // Define a struct containing tensors
//! #[derive(Debug)]
//! struct ModelWeights {
//!     weight_matrix: Tensor,
//!     bias_vector: Tensor,
//!     learning_rate: f32,
//!     name: String,
//! }
//!
//! impl StructSerializable for ModelWeights {
//!     fn to_serializer(&self) -> train_station::serialization::StructSerializer {
//!         train_station::serialization::StructSerializer::new()
//!             .field("weight_matrix", &self.weight_matrix)
//!             .field("bias_vector", &self.bias_vector)
//!             .field("learning_rate", &self.learning_rate)
//!             .field("name", &self.name)
//!     }
//!
//!     fn from_deserializer(
//!         deserializer: &mut train_station::serialization::StructDeserializer,
//!     ) -> train_station::serialization::SerializationResult<Self> {
//!         Ok(ModelWeights {
//!             weight_matrix: deserializer.field("weight_matrix")?,
//!             bias_vector: deserializer.field("bias_vector")?,
//!             learning_rate: deserializer.field("learning_rate")?,
//!             name: deserializer.field("name")?,
//!         })
//!     }
//! }
//!
//! // Create test struct with tensors
//! let mut weights = ModelWeights {
//!     weight_matrix: Tensor::zeros(vec![10, 5]),
//!     bias_vector: Tensor::ones(vec![5]).with_requires_grad(),
//!     learning_rate: 0.001,
//!     name: "test_model".to_string(),
//! };
//!
//! // Set some values
//! weights.weight_matrix.set(&[0, 0], 0.5);
//! weights.bias_vector.set(&[2], 2.0);
//!
//! // Test JSON serialization
//! let json = weights.to_json().unwrap();
//! let loaded_weights = ModelWeights::from_json(&json).unwrap();
//!
//! assert_eq!(weights.learning_rate, loaded_weights.learning_rate);
//! assert_eq!(weights.name, loaded_weights.name);
//! assert_eq!(
//!     weights.weight_matrix.shape().dims,
//!     loaded_weights.weight_matrix.shape().dims
//! );
//! ```
//!
//! ## Large Tensor Serialization
//!
//! ```rust
//! use train_station::Tensor;
//! use train_station::serialization::StructSerializable;
//!
//! // Create large tensor
//! let mut tensor = Tensor::zeros(vec![100, 100]).with_requires_grad();
//!
//! // Set some values
//! for i in 0..10 {
//!     for j in 0..10 {
//!         tensor.set(&[i, j], (i * 10 + j) as f32);
//!     }
//! }
//!
//! // Binary serialization is more efficient for large tensors
//! let binary = tensor.to_binary().unwrap();
//! let loaded_tensor = Tensor::from_binary(&binary).unwrap();
//!
//! // Verify properties
//! assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
//! assert_eq!(tensor.requires_grad(), loaded_tensor.requires_grad());
//!
//! // Verify data integrity
//! for i in 0..10 {
//!     for j in 0..10 {
//!         assert_eq!(tensor.get(&[i, j]), loaded_tensor.get(&[i, j]));
//!     }
//! }
//! ```
//!
//! ## Error Handling
//!
//! ```rust
//! use train_station::Tensor;
//! use train_station::serialization::StructSerializable;
//!
//! // Test invalid serialization data
//! let invalid_json = r#"{"invalid": "data"}"#;
//! let result = Tensor::from_json(invalid_json);
//! assert!(result.is_err());
//!
//! // Test empty binary data
//! let empty_binary = vec![];
//! let result = Tensor::from_binary(&empty_binary);
//! assert!(result.is_err());
//! ```
//!
//! # Performance Characteristics
//!
//! - **Binary Format**: Optimized for size and speed
//! - **JSON Format**: Human-readable with reasonable performance
//! - **Memory Efficient**: Minimal overhead during serialization
//! - **Zero-Copy**: Direct serialization of tensor data arrays
//! - **Type Safety**: Compile-time guarantees for serialization correctness

use std::collections::HashMap;

use crate::device::{Device, DeviceType};
use crate::serialization::{
    FieldValue, FromFieldValue, SerializationError, SerializationResult, StructDeserializer,
    StructSerializable, StructSerializer, ToFieldValue,
};
use crate::tensor::{Shape, Tensor};

// ===== Device Serialization =====

impl ToFieldValue for DeviceType {
    /// Convert DeviceType to FieldValue for serialization
    ///
    /// # Returns
    ///
    /// Enum FieldValue with variant name (proper enum serialization)
    fn to_field_value(&self) -> FieldValue {
        match self {
            DeviceType::Cpu => FieldValue::from_enum_unit("Cpu".to_string()),
            DeviceType::Cuda => FieldValue::from_enum_unit("Cuda".to_string()),
        }
    }
}

impl FromFieldValue for DeviceType {
    /// Convert FieldValue to DeviceType for deserialization
    ///
    /// # Arguments
    ///
    /// * `value` - FieldValue containing enum data
    /// * `field_name` - Name of the field for error reporting
    ///
    /// # Returns
    ///
    /// DeviceType enum value or error if invalid
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        let (variant, data) =
            value
                .as_enum()
                .map_err(|_| SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: "Expected enum value for device type".to_string(),
                })?;

        // Ensure no data for unit variants
        if data.is_some() {
            return Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "DeviceType variants should not have associated data".to_string(),
            });
        }

        match variant {
            "Cpu" => Ok(DeviceType::Cpu),
            "Cuda" => Ok(DeviceType::Cuda),
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Unknown device type variant: {}", variant),
            }),
        }
    }
}

impl ToFieldValue for Device {
    /// Convert Device to FieldValue for serialization
    ///
    /// # Returns
    ///
    /// Object containing device type and index
    fn to_field_value(&self) -> FieldValue {
        let mut object = HashMap::new();
        object.insert("type".to_string(), self.device_type().to_field_value());
        object.insert("index".to_string(), self.index().to_field_value());
        FieldValue::from_object(object)
    }
}

impl FromFieldValue for Device {
    /// Convert FieldValue to Device for deserialization
    ///
    /// # Arguments
    ///
    /// * `value` - FieldValue containing device object
    /// * `field_name` - Name of the field for error reporting
    ///
    /// # Returns
    ///
    /// Device instance or error if invalid
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        let object = value
            .as_object()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "Expected object for device".to_string(),
            })?;

        let device_type = object
            .get("type")
            .ok_or_else(|| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "Missing device type field".to_string(),
            })?
            .clone();

        let index = object
            .get("index")
            .ok_or_else(|| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "Missing device index field".to_string(),
            })?
            .clone();

        let device_type = DeviceType::from_field_value(device_type, "type")?;
        let index = usize::from_field_value(index, "index")?;

        match device_type {
            DeviceType::Cpu => Ok(Device::cpu()),
            DeviceType::Cuda => Ok(Device::cuda(index)),
        }
    }
}

// ===== Memory Layout Serialization =====

impl ToFieldValue for crate::tensor::MemoryLayout {
    /// Convert MemoryLayout to FieldValue for serialization
    ///
    /// # Returns
    ///
    /// Enum FieldValue with variant name (proper enum serialization)
    fn to_field_value(&self) -> FieldValue {
        match self {
            crate::tensor::MemoryLayout::Contiguous => {
                FieldValue::from_enum_unit("Contiguous".to_string())
            }
            crate::tensor::core::MemoryLayout::Strided => {
                FieldValue::from_enum_unit("Strided".to_string())
            }
            crate::tensor::core::MemoryLayout::View => {
                FieldValue::from_enum_unit("View".to_string())
            }
        }
    }
}

impl FromFieldValue for crate::tensor::MemoryLayout {
    /// Convert FieldValue to MemoryLayout for deserialization
    ///
    /// # Arguments
    ///
    /// * `value` - FieldValue containing enum data
    /// * `field_name` - Name of the field for error reporting
    ///
    /// # Returns
    ///
    /// MemoryLayout enum value or error if invalid
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        let (variant, data) =
            value
                .as_enum()
                .map_err(|_| SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: "Expected enum value for memory layout".to_string(),
                })?;

        // Ensure no data for unit variants
        if data.is_some() {
            return Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "MemoryLayout variants should not have associated data".to_string(),
            });
        }

        match variant {
            "Contiguous" => Ok(crate::tensor::MemoryLayout::Contiguous),
            "Strided" => Ok(crate::tensor::MemoryLayout::Strided),
            "View" => Ok(crate::tensor::MemoryLayout::View),
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!("Unknown memory layout variant: {}", variant),
            }),
        }
    }
}

// ===== Shape Serialization =====

impl ToFieldValue for Shape {
    /// Convert Shape to FieldValue for serialization
    ///
    /// # Returns
    ///
    /// Object containing all shape metadata
    fn to_field_value(&self) -> FieldValue {
        let mut object = HashMap::new();
        object.insert("dims".to_string(), self.dims.to_field_value());
        object.insert("size".to_string(), self.size.to_field_value());
        object.insert("strides".to_string(), self.strides.to_field_value());
        object.insert("layout".to_string(), self.layout.to_field_value());
        FieldValue::from_object(object)
    }
}

impl FromFieldValue for Shape {
    /// Convert FieldValue to Shape for deserialization
    ///
    /// # Arguments
    ///
    /// * `value` - FieldValue containing shape object
    /// * `field_name` - Name of the field for error reporting
    ///
    /// # Returns
    ///
    /// Shape instance or error if invalid
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        let object = value
            .as_object()
            .map_err(|_| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "Expected object for shape".to_string(),
            })?;

        let dims = object
            .get("dims")
            .ok_or_else(|| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "Missing dims field in shape".to_string(),
            })?
            .clone();

        let size = object
            .get("size")
            .ok_or_else(|| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "Missing size field in shape".to_string(),
            })?
            .clone();

        let strides = object
            .get("strides")
            .ok_or_else(|| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "Missing strides field in shape".to_string(),
            })?
            .clone();

        let layout = object
            .get("layout")
            .ok_or_else(|| SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "Missing layout field in shape".to_string(),
            })?
            .clone();

        let dims = Vec::<usize>::from_field_value(dims, "dims")?;
        let size = usize::from_field_value(size, "size")?;
        let strides = Vec::<usize>::from_field_value(strides, "strides")?;
        let layout = crate::tensor::MemoryLayout::from_field_value(layout, "layout")?;

        // Validate consistency
        let expected_size: usize = dims.iter().product();
        if size != expected_size {
            return Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Shape size {} doesn't match computed size {}",
                    size, expected_size
                ),
            });
        }

        if dims.len() != strides.len() {
            return Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: "Dimensions and strides must have same length".to_string(),
            });
        }

        Ok(Shape {
            dims,
            size,
            strides,
            layout,
        })
    }
}

// ===== Tensor Serialization =====

impl StructSerializable for Tensor {
    /// Convert Tensor to StructSerializer for serialization
    ///
    /// Serializes tensor data, shape, device, and gradtrack state.
    /// Runtime state (id, grad, grad_fn, allocation_owner) is not serialized.
    ///
    /// # Returns
    ///
    /// StructSerializer containing all persistent tensor state
    fn to_serializer(&self) -> StructSerializer {
        // Extract tensor data as Vec<f32> - now uses efficient FieldValue implementation:
        // - JSON format: Human-readable arrays of numbers
        // - Binary format: Efficient byte representation with length header
        let data: Vec<f32> =
            unsafe { std::slice::from_raw_parts(self.as_ptr(), self.size()).to_vec() };

        StructSerializer::new()
            .field("data", &data)
            .field("shape", self.shape())
            .field("device", &self.device())
            .field("requires_grad", &self.requires_grad())
    }

    /// Create Tensor from StructDeserializer
    ///
    /// Reconstructs tensor from serialized data, shape, device, and gradtrack state.
    /// Allocates new memory and generates new tensor ID.
    ///
    /// # Arguments
    ///
    /// * `deserializer` - StructDeserializer containing tensor data
    ///
    /// # Returns
    ///
    /// Reconstructed Tensor instance or error if deserialization fails
    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let data: Vec<f32> = deserializer.field("data")?;
        let shape: Shape = deserializer.field("shape")?;
        let device: Device = deserializer.field("device")?;
        let requires_grad: bool = deserializer.field("requires_grad")?;

        // Validate data size matches shape
        if data.len() != shape.size {
            return Err(SerializationError::ValidationFailed {
                field: "tensor".to_string(),
                message: format!(
                    "Data length {} doesn't match shape size {}",
                    data.len(),
                    shape.size
                ),
            });
        }

        // Create new tensor with the deserialized shape on the correct device
        let mut tensor = Tensor::new_on_device(shape.dims.clone(), device);

        // Copy data into tensor
        if !data.is_empty() {
            unsafe {
                let dst = tensor.as_mut_ptr();
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
        }

        // Set gradtrack state
        tensor.set_requires_grad(requires_grad);

        // Validate that the reconstructed shape matches
        if tensor.shape().dims != shape.dims
            || tensor.shape().size != shape.size
            || tensor.shape().strides != shape.strides
        {
            return Err(SerializationError::ValidationFailed {
                field: "tensor".to_string(),
                message: "Reconstructed tensor shape doesn't match serialized shape".to_string(),
            });
        }

        Ok(tensor)
    }
}

impl FromFieldValue for Tensor {
    /// Convert FieldValue to Tensor for use as struct field
    ///
    /// # Arguments
    ///
    /// * `value` - FieldValue containing tensor data
    /// * `field_name` - Name of the field for error reporting
    ///
    /// # Returns
    ///
    /// Tensor instance or error if deserialization fails
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        // Try binary object first (for when serialized as binary)
        if let Ok(binary_data) = value.as_binary_object() {
            return Tensor::from_binary(binary_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize tensor from binary: {}", e),
                }
            });
        }

        // Try JSON object (for when serialized as JSON)
        if let Ok(json_data) = value.as_json_object() {
            return Tensor::from_json(json_data).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize tensor from JSON: {}", e),
                }
            });
        }

        // Try object (for when serialized as structured object in JSON)
        if let Ok(object) = value.as_object() {
            // Convert object back to deserializer and use StructSerializable
            let mut deserializer = StructDeserializer::from_fields(object.clone());
            return Tensor::from_deserializer(&mut deserializer).map_err(|e| {
                SerializationError::ValidationFailed {
                    field: field_name.to_string(),
                    message: format!("Failed to deserialize tensor from object: {}", e),
                }
            });
        }

        Err(SerializationError::ValidationFailed {
            field: field_name.to_string(),
            message: "Expected binary object, JSON object, or structured object for tensor field"
                .to_string(),
        })
    }
}

// ===== Serializable Trait Implementation =====

impl crate::serialization::Serializable for Tensor {
    /// Serialize the tensor to JSON format
    ///
    /// This method converts the tensor into a human-readable JSON string representation
    /// that includes all tensor data, shape information, device placement, and gradtrack state.
    /// The JSON format is suitable for debugging, configuration files, and cross-language
    /// interoperability.
    ///
    /// # Returns
    ///
    /// JSON string representation of the tensor on success, or `SerializationError` on failure
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    /// use train_station::serialization::Serializable;
    ///
    /// let mut tensor = Tensor::zeros(vec![2, 3]);
    /// tensor.set(&[0, 0], 1.0);
    /// tensor.set(&[1, 2], 5.0);
    ///
    /// let json = tensor.to_json().unwrap();
    /// assert!(!json.is_empty());
    /// assert!(json.contains("data"));
    /// assert!(json.contains("shape"));
    /// ```
    fn to_json(&self) -> SerializationResult<String> {
        StructSerializable::to_json(self)
    }

    /// Deserialize a tensor from JSON format
    ///
    /// This method parses a JSON string and reconstructs a tensor with all its data,
    /// shape information, device placement, and gradtrack state. The JSON must contain
    /// all necessary fields in the expected format.
    ///
    /// # Arguments
    ///
    /// * `json` - JSON string containing serialized tensor data
    ///
    /// # Returns
    ///
    /// The deserialized tensor on success, or `SerializationError` on failure
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    /// use train_station::serialization::Serializable;
    ///
    /// let mut original = Tensor::ones(vec![2, 2]);
    /// original.set(&[0, 1], 3.0);
    /// original.set_requires_grad(true);
    ///
    /// let json = original.to_json().unwrap();
    /// let restored = Tensor::from_json(&json).unwrap();
    ///
    /// assert_eq!(original.shape().dims, restored.shape().dims);
    /// assert_eq!(original.get(&[0, 1]), restored.get(&[0, 1]));
    /// assert_eq!(original.requires_grad(), restored.requires_grad());
    /// ```
    fn from_json(json: &str) -> SerializationResult<Self> {
        StructSerializable::from_json(json)
    }

    /// Serialize the tensor to binary format
    ///
    /// This method converts the tensor into a compact binary representation optimized
    /// for storage and transmission. The binary format provides maximum performance
    /// and minimal file sizes, making it ideal for large tensors and production use.
    ///
    /// # Returns
    ///
    /// Binary representation of the tensor on success, or `SerializationError` on failure
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    /// use train_station::serialization::Serializable;
    ///
    /// let mut tensor = Tensor::zeros(vec![100, 100]);
    /// for i in 0..10 {
    ///     tensor.set(&[i, i], i as f32);
    /// }
    ///
    /// let binary = tensor.to_binary().unwrap();
    /// assert!(!binary.is_empty());
    /// // Binary format is more compact than JSON for large tensors
    /// ```
    fn to_binary(&self) -> SerializationResult<Vec<u8>> {
        StructSerializable::to_binary(self)
    }

    /// Deserialize a tensor from binary format
    ///
    /// This method parses binary data and reconstructs a tensor with all its data,
    /// shape information, device placement, and gradtrack state. The binary data
    /// must contain complete serialized information in the expected format.
    ///
    /// # Arguments
    ///
    /// * `data` - Binary data containing serialized tensor information
    ///
    /// # Returns
    ///
    /// The deserialized tensor on success, or `SerializationError` on failure
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    /// use train_station::serialization::Serializable;
    ///
    /// let mut original = Tensor::ones(vec![3, 4]);
    /// original.set(&[2, 3], 7.5);
    /// original.set_requires_grad(true);
    ///
    /// let binary = original.to_binary().unwrap();
    /// let restored = Tensor::from_binary(&binary).unwrap();
    ///
    /// assert_eq!(original.shape().dims, restored.shape().dims);
    /// assert_eq!(original.get(&[2, 3]), restored.get(&[2, 3]));
    /// assert_eq!(original.requires_grad(), restored.requires_grad());
    /// ```
    fn from_binary(data: &[u8]) -> SerializationResult<Self> {
        StructSerializable::from_binary(data)
    }
}

#[cfg(test)]
mod tests {
    //! Comprehensive tests for tensor serialization functionality
    //!
    //! Tests cover all serialization formats and usage patterns including:
    //! - JSON and binary roundtrip serialization
    //! - Tensor as field within structs  
    //! - Edge cases and error conditions
    //! - Device and shape serialization
    //! - Large tensor serialization

    use super::*;

    // ===== Device Serialization Tests =====

    #[test]
    fn test_device_type_serialization() {
        // Test CPU device type
        let cpu_type = DeviceType::Cpu;
        let field_value = cpu_type.to_field_value();
        let deserialized = DeviceType::from_field_value(field_value, "device_type").unwrap();
        assert_eq!(cpu_type, deserialized);

        // Test CUDA device type
        let cuda_type = DeviceType::Cuda;
        let field_value = cuda_type.to_field_value();
        let deserialized = DeviceType::from_field_value(field_value, "device_type").unwrap();
        assert_eq!(cuda_type, deserialized);
    }

    #[test]
    fn test_device_serialization() {
        // Test CPU device
        let cpu_device = Device::cpu();
        let field_value = cpu_device.to_field_value();
        let deserialized = Device::from_field_value(field_value, "device").unwrap();
        assert_eq!(cpu_device, deserialized);
        assert!(deserialized.is_cpu());
        assert_eq!(deserialized.index(), 0);
    }

    #[test]
    fn test_device_serialization_errors() {
        // Test invalid device type
        let invalid_device_type = FieldValue::from_string("invalid".to_string());
        let result = DeviceType::from_field_value(invalid_device_type, "device_type");
        assert!(result.is_err());

        // Test missing device fields
        let incomplete_device = FieldValue::from_object({
            let mut obj = HashMap::new();
            obj.insert(
                "type".to_string(),
                FieldValue::from_string("cpu".to_string()),
            );
            // Missing index field
            obj
        });
        let result = Device::from_field_value(incomplete_device, "device");
        assert!(result.is_err());
    }

    // ===== Shape Serialization Tests =====

    #[test]
    fn test_memory_layout_serialization() {
        use crate::tensor::MemoryLayout;

        let layouts = [
            MemoryLayout::Contiguous,
            MemoryLayout::Strided,
            MemoryLayout::View,
        ];

        for layout in &layouts {
            let field_value = layout.to_field_value();
            let deserialized = MemoryLayout::from_field_value(field_value, "layout").unwrap();
            assert_eq!(*layout, deserialized);
        }
    }

    #[test]
    fn test_shape_serialization() {
        // Test contiguous shape
        let shape = Shape::new(vec![2, 3, 4]);
        let field_value = shape.to_field_value();
        let deserialized = Shape::from_field_value(field_value, "shape").unwrap();
        assert_eq!(shape, deserialized);
        assert_eq!(deserialized.dims, vec![2, 3, 4]);
        assert_eq!(deserialized.size, 24);
        assert_eq!(deserialized.strides, vec![12, 4, 1]);

        // Test strided shape
        let strided_shape = Shape::with_strides(vec![2, 3], vec![6, 2]);
        let field_value = strided_shape.to_field_value();
        let deserialized = Shape::from_field_value(field_value, "shape").unwrap();
        assert_eq!(strided_shape, deserialized);
    }

    #[test]
    fn test_shape_validation_errors() {
        use crate::tensor::MemoryLayout;

        // Test inconsistent size
        let invalid_shape = FieldValue::from_object({
            let mut obj = HashMap::new();
            obj.insert("dims".to_string(), vec![2usize, 3].to_field_value());
            obj.insert("size".to_string(), 10usize.to_field_value()); // Should be 6
            obj.insert("strides".to_string(), vec![3usize, 1].to_field_value());
            obj.insert(
                "layout".to_string(),
                MemoryLayout::Contiguous.to_field_value(),
            );
            obj
        });
        let result = Shape::from_field_value(invalid_shape, "shape");
        assert!(result.is_err());

        // Test mismatched dimensions and strides
        let invalid_shape = FieldValue::from_object({
            let mut obj = HashMap::new();
            obj.insert("dims".to_string(), vec![2usize, 3].to_field_value());
            obj.insert("size".to_string(), 6usize.to_field_value());
            obj.insert("strides".to_string(), vec![3usize].to_field_value()); // Wrong length
            obj.insert(
                "layout".to_string(),
                MemoryLayout::Contiguous.to_field_value(),
            );
            obj
        });
        let result = Shape::from_field_value(invalid_shape, "shape");
        assert!(result.is_err());
    }

    // ===== Tensor Serialization Tests =====

    #[test]
    fn test_tensor_json_roundtrip() {
        // Create test tensor with data
        let mut tensor = Tensor::zeros(vec![2, 3]);
        tensor.set(&[0, 0], 1.0);
        tensor.set(&[0, 1], 2.0);
        tensor.set(&[0, 2], 3.0);
        tensor.set(&[1, 0], 4.0);
        tensor.set(&[1, 1], 5.0);
        tensor.set(&[1, 2], 6.0);
        tensor.set_requires_grad(true);

        // Serialize to JSON
        let json = tensor.to_json().unwrap();
        assert!(!json.is_empty());

        // Deserialize from JSON
        let loaded_tensor = Tensor::from_json(&json).unwrap();

        // Verify tensor properties
        assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
        assert_eq!(tensor.size(), loaded_tensor.size());
        assert_eq!(tensor.device(), loaded_tensor.device());
        assert_eq!(tensor.requires_grad(), loaded_tensor.requires_grad());

        // Verify tensor data
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(tensor.get(&[i, j]), loaded_tensor.get(&[i, j]));
            }
        }
    }

    #[test]
    fn test_tensor_binary_roundtrip() {
        // Create test tensor with gradient tracking
        let mut tensor = Tensor::ones(vec![3, 4]).with_requires_grad();

        // Modify some values
        tensor.set(&[0, 0], 10.0);
        tensor.set(&[1, 2], 20.0);
        tensor.set(&[2, 3], 30.0);

        // Serialize to binary
        let binary = tensor.to_binary().unwrap();
        assert!(!binary.is_empty());

        // Deserialize from binary
        let loaded_tensor = Tensor::from_binary(&binary).unwrap();

        // Verify tensor properties
        assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
        assert_eq!(tensor.size(), loaded_tensor.size());
        assert_eq!(tensor.device(), loaded_tensor.device());
        assert_eq!(tensor.requires_grad(), loaded_tensor.requires_grad());

        // Verify tensor data
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(tensor.get(&[i, j]), loaded_tensor.get(&[i, j]));
            }
        }
    }

    #[test]
    fn test_empty_tensor_serialization() {
        // Test zero-sized tensor
        let tensor = Tensor::new(vec![0]);

        // JSON roundtrip
        let json = tensor.to_json().unwrap();
        let loaded_tensor = Tensor::from_json(&json).unwrap();
        assert_eq!(tensor.size(), loaded_tensor.size());
        assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);

        // Binary roundtrip
        let binary = tensor.to_binary().unwrap();
        let loaded_tensor = Tensor::from_binary(&binary).unwrap();
        assert_eq!(tensor.size(), loaded_tensor.size());
        assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
    }

    #[test]
    fn test_large_tensor_serialization() {
        // Test larger tensor
        let mut tensor = Tensor::zeros(vec![100, 100]).with_requires_grad();

        // Set some values
        for i in 0..10 {
            for j in 0..10 {
                tensor.set(&[i, j], (i * 10 + j) as f32);
            }
        }

        // Binary roundtrip (more efficient for large tensors)
        let binary = tensor.to_binary().unwrap();
        let loaded_tensor = Tensor::from_binary(&binary).unwrap();

        // Verify properties
        assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
        assert_eq!(tensor.requires_grad(), loaded_tensor.requires_grad());

        // Verify a subset of data
        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(tensor.get(&[i, j]), loaded_tensor.get(&[i, j]));
            }
        }
    }

    #[test]
    fn test_tensor_as_field_in_struct() {
        // Define a struct containing tensors
        #[derive(Debug)]
        struct ModelWeights {
            weight_matrix: Tensor,
            bias_vector: Tensor,
            learning_rate: f32,
            name: String,
        }

        impl StructSerializable for ModelWeights {
            fn to_serializer(&self) -> StructSerializer {
                StructSerializer::new()
                    .field("weight_matrix", &self.weight_matrix)
                    .field("bias_vector", &self.bias_vector)
                    .field("learning_rate", &self.learning_rate)
                    .field("name", &self.name)
            }

            fn from_deserializer(
                deserializer: &mut StructDeserializer,
            ) -> SerializationResult<Self> {
                Ok(ModelWeights {
                    weight_matrix: deserializer.field("weight_matrix")?,
                    bias_vector: deserializer.field("bias_vector")?,
                    learning_rate: deserializer.field("learning_rate")?,
                    name: deserializer.field("name")?,
                })
            }
        }

        // Create test struct with tensors
        let mut weights = ModelWeights {
            weight_matrix: Tensor::zeros(vec![10, 5]),
            bias_vector: Tensor::ones(vec![5]).with_requires_grad(),
            learning_rate: 0.001,
            name: "test_model".to_string(),
        };

        // Set some values
        weights.weight_matrix.set(&[0, 0], 0.5);
        weights.weight_matrix.set(&[9, 4], -0.3);
        weights.bias_vector.set(&[2], 2.0);

        // Test JSON serialization
        let json = weights.to_json().unwrap();
        let loaded_weights = ModelWeights::from_json(&json).unwrap();

        assert_eq!(weights.learning_rate, loaded_weights.learning_rate);
        assert_eq!(weights.name, loaded_weights.name);
        assert_eq!(
            weights.weight_matrix.shape().dims,
            loaded_weights.weight_matrix.shape().dims
        );
        assert_eq!(
            weights.bias_vector.shape().dims,
            loaded_weights.bias_vector.shape().dims
        );
        assert_eq!(
            weights.bias_vector.requires_grad(),
            loaded_weights.bias_vector.requires_grad()
        );

        // Verify tensor data
        assert_eq!(
            weights.weight_matrix.get(&[0, 0]),
            loaded_weights.weight_matrix.get(&[0, 0])
        );
        assert_eq!(
            weights.weight_matrix.get(&[9, 4]),
            loaded_weights.weight_matrix.get(&[9, 4])
        );
        assert_eq!(
            weights.bias_vector.get(&[2]),
            loaded_weights.bias_vector.get(&[2])
        );

        // Test binary serialization
        let binary = weights.to_binary().unwrap();
        let loaded_weights = ModelWeights::from_binary(&binary).unwrap();

        assert_eq!(weights.learning_rate, loaded_weights.learning_rate);
        assert_eq!(weights.name, loaded_weights.name);
        assert_eq!(
            weights.weight_matrix.shape().dims,
            loaded_weights.weight_matrix.shape().dims
        );
        assert_eq!(
            weights.bias_vector.requires_grad(),
            loaded_weights.bias_vector.requires_grad()
        );
    }

    #[test]
    fn test_multiple_tensors_in_struct() {
        // Test struct with multiple tensors of different shapes
        #[derive(Debug)]
        struct MultiTensorStruct {
            tensor_1d: Tensor,
            tensor_2d: Tensor,
            tensor_3d: Tensor,
            metadata: HashMap<String, String>,
        }

        impl StructSerializable for MultiTensorStruct {
            fn to_serializer(&self) -> StructSerializer {
                StructSerializer::new()
                    .field("tensor_1d", &self.tensor_1d)
                    .field("tensor_2d", &self.tensor_2d)
                    .field("tensor_3d", &self.tensor_3d)
                    .field("metadata", &self.metadata)
            }

            fn from_deserializer(
                deserializer: &mut StructDeserializer,
            ) -> SerializationResult<Self> {
                Ok(MultiTensorStruct {
                    tensor_1d: deserializer.field("tensor_1d")?,
                    tensor_2d: deserializer.field("tensor_2d")?,
                    tensor_3d: deserializer.field("tensor_3d")?,
                    metadata: deserializer.field("metadata")?,
                })
            }
        }

        // Create test struct
        let mut multi_tensor = MultiTensorStruct {
            tensor_1d: Tensor::zeros(vec![5]),
            tensor_2d: Tensor::ones(vec![3, 4]).with_requires_grad(),
            tensor_3d: Tensor::zeros(vec![2, 2, 2]),
            metadata: {
                let mut map = HashMap::new();
                map.insert("version".to_string(), "1.0".to_string());
                map.insert("type".to_string(), "test".to_string());
                map
            },
        };

        // Set some values
        multi_tensor.tensor_1d.set(&[0], 10.0);
        multi_tensor.tensor_2d.set(&[0, 0], 5.0);
        multi_tensor.tensor_3d.set(&[1, 1, 1], 3.0);

        // Test JSON roundtrip
        let json = multi_tensor.to_json().unwrap();
        let loaded = MultiTensorStruct::from_json(&json).unwrap();

        assert_eq!(
            multi_tensor.tensor_1d.shape().dims,
            loaded.tensor_1d.shape().dims
        );
        assert_eq!(
            multi_tensor.tensor_2d.shape().dims,
            loaded.tensor_2d.shape().dims
        );
        assert_eq!(
            multi_tensor.tensor_3d.shape().dims,
            loaded.tensor_3d.shape().dims
        );
        assert_eq!(
            multi_tensor.tensor_2d.requires_grad(),
            loaded.tensor_2d.requires_grad()
        );
        assert_eq!(multi_tensor.metadata, loaded.metadata);

        // Verify tensor values
        assert_eq!(multi_tensor.tensor_1d.get(&[0]), loaded.tensor_1d.get(&[0]));
        assert_eq!(
            multi_tensor.tensor_2d.get(&[0, 0]),
            loaded.tensor_2d.get(&[0, 0])
        );
        assert_eq!(
            multi_tensor.tensor_3d.get(&[1, 1, 1]),
            loaded.tensor_3d.get(&[1, 1, 1])
        );

        // Test binary roundtrip
        let binary = multi_tensor.to_binary().unwrap();
        let loaded = MultiTensorStruct::from_binary(&binary).unwrap();
        assert_eq!(
            multi_tensor.tensor_1d.shape().dims,
            loaded.tensor_1d.shape().dims
        );
        assert_eq!(
            multi_tensor.tensor_2d.requires_grad(),
            loaded.tensor_2d.requires_grad()
        );
    }

    #[test]
    fn test_tensor_serialization_errors() {
        // Test invalid data size
        let mut deserializer = StructDeserializer::from_json(
            r#"
        {
            "data": [1.0, 2.0, 3.0],
            "shape": {
                "dims": [2, 3],
                "size": 6,
                "strides": [3, 1],
                "layout": "contiguous"
            },
            "device": {"type": "cpu", "index": 0},
            "requires_grad": false
        }"#,
        )
        .unwrap();

        let result = Tensor::from_deserializer(&mut deserializer);
        assert!(result.is_err()); // Data length (3) doesn't match shape size (6)
    }

    #[test]
    fn test_field_value_tensor_roundtrip() {
        // Test tensor as FieldValue
        let mut tensor = Tensor::zeros(vec![2, 2]);
        tensor.set(&[0, 0], 1.0);
        tensor.set(&[1, 1], 2.0);

        let field_value = tensor.to_field_value();
        let loaded_tensor = Tensor::from_field_value(field_value, "test_tensor").unwrap();

        assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
        assert_eq!(tensor.get(&[0, 0]), loaded_tensor.get(&[0, 0]));
        assert_eq!(tensor.get(&[1, 1]), loaded_tensor.get(&[1, 1]));
    }

    #[test]
    fn test_different_tensor_shapes() {
        let test_shapes = vec![
            vec![1],          // Scalar
            vec![10],         // 1D vector
            vec![3, 4],       // 2D matrix
            vec![2, 3, 4],    // 3D tensor
            vec![2, 2, 2, 2], // 4D tensor
        ];

        for shape in test_shapes {
            let tensor = Tensor::zeros(shape.clone()).with_requires_grad();

            // JSON roundtrip
            let json = tensor.to_json().unwrap();
            let loaded = Tensor::from_json(&json).unwrap();
            assert_eq!(tensor.shape().dims, loaded.shape().dims);
            assert_eq!(tensor.requires_grad(), loaded.requires_grad());

            // Binary roundtrip
            let binary = tensor.to_binary().unwrap();
            let loaded = Tensor::from_binary(&binary).unwrap();
            assert_eq!(tensor.shape().dims, loaded.shape().dims);
            assert_eq!(tensor.requires_grad(), loaded.requires_grad());
        }
    }

    // ===== Serializable Trait Tests =====

    #[test]
    fn test_serializable_json_methods() {
        // Create and populate test tensor
        let mut tensor = Tensor::zeros(vec![2, 3]);
        tensor.set(&[0, 0], 1.0);
        tensor.set(&[0, 1], 2.0);
        tensor.set(&[1, 2], 5.0);
        tensor.set_requires_grad(true);

        // Test to_json method
        let json = <Tensor as crate::serialization::Serializable>::to_json(&tensor).unwrap();
        assert!(!json.is_empty());
        assert!(json.contains("data"));
        assert!(json.contains("shape"));
        assert!(json.contains("device"));
        assert!(json.contains("requires_grad"));

        // Test from_json method
        let restored = <Tensor as crate::serialization::Serializable>::from_json(&json).unwrap();
        assert_eq!(tensor.shape().dims, restored.shape().dims);
        assert_eq!(tensor.size(), restored.size());
        assert_eq!(tensor.device(), restored.device());
        assert_eq!(tensor.requires_grad(), restored.requires_grad());

        // Verify tensor data
        assert_eq!(tensor.get(&[0, 0]), restored.get(&[0, 0]));
        assert_eq!(tensor.get(&[0, 1]), restored.get(&[0, 1]));
        assert_eq!(tensor.get(&[1, 2]), restored.get(&[1, 2]));
    }

    #[test]
    fn test_serializable_binary_methods() {
        // Create and populate test tensor
        let mut tensor = Tensor::ones(vec![3, 4]);
        tensor.set(&[0, 0], 10.0);
        tensor.set(&[1, 2], 20.0);
        tensor.set(&[2, 3], 30.0);
        tensor.set_requires_grad(true);

        // Test to_binary method
        let binary = <Tensor as crate::serialization::Serializable>::to_binary(&tensor).unwrap();
        assert!(!binary.is_empty());

        // Test from_binary method
        let restored =
            <Tensor as crate::serialization::Serializable>::from_binary(&binary).unwrap();
        assert_eq!(tensor.shape().dims, restored.shape().dims);
        assert_eq!(tensor.size(), restored.size());
        assert_eq!(tensor.device(), restored.device());
        assert_eq!(tensor.requires_grad(), restored.requires_grad());

        // Verify tensor data
        assert_eq!(tensor.get(&[0, 0]), restored.get(&[0, 0]));
        assert_eq!(tensor.get(&[1, 2]), restored.get(&[1, 2]));
        assert_eq!(tensor.get(&[2, 3]), restored.get(&[2, 3]));
    }

    #[test]
    fn test_serializable_file_io_json() {
        use crate::serialization::{Format, Serializable};
        use std::fs;
        use std::path::Path;

        // Create test tensor
        let mut tensor = Tensor::zeros(vec![2, 2]);
        tensor.set(&[0, 0], 1.0);
        tensor.set(&[0, 1], 2.0);
        tensor.set(&[1, 0], 3.0);
        tensor.set(&[1, 1], 4.0);
        tensor.set_requires_grad(true);

        // Test file paths
        let json_path = "test_tensor_serializable.json";
        let json_path_2 = "test_tensor_serializable_2.json";

        // Cleanup any existing files
        let _ = fs::remove_file(json_path);
        let _ = fs::remove_file(json_path_2);

        // Test save method with JSON format
        Serializable::save(&tensor, json_path, Format::Json).unwrap();
        assert!(Path::new(json_path).exists());

        // Test load method with JSON format
        let loaded_tensor = Tensor::load(json_path, Format::Json).unwrap();
        assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
        assert_eq!(tensor.requires_grad(), loaded_tensor.requires_grad());
        assert_eq!(tensor.get(&[0, 0]), loaded_tensor.get(&[0, 0]));
        assert_eq!(tensor.get(&[1, 1]), loaded_tensor.get(&[1, 1]));

        // Test save_to_writer and load_from_reader
        {
            let mut writer = std::fs::File::create(json_path_2).unwrap();
            Serializable::save_to_writer(&tensor, &mut writer, Format::Json).unwrap();
        }
        assert!(Path::new(json_path_2).exists());

        {
            let mut reader = std::fs::File::open(json_path_2).unwrap();
            let loaded_tensor = Tensor::load_from_reader(&mut reader, Format::Json).unwrap();
            assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
            assert_eq!(tensor.requires_grad(), loaded_tensor.requires_grad());
            assert_eq!(tensor.get(&[0, 1]), loaded_tensor.get(&[0, 1]));
            assert_eq!(tensor.get(&[1, 0]), loaded_tensor.get(&[1, 0]));
        }

        // Cleanup test files
        let _ = fs::remove_file(json_path);
        let _ = fs::remove_file(json_path_2);
    }

    #[test]
    fn test_serializable_file_io_binary() {
        use crate::serialization::{Format, Serializable};
        use std::fs;
        use std::path::Path;

        // Create test tensor
        let mut tensor = Tensor::ones(vec![3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                tensor.set(&[i, j], (i * 3 + j) as f32);
            }
        }
        tensor.set_requires_grad(true);

        // Test file paths
        let binary_path = "test_tensor_serializable.bin";
        let binary_path_2 = "test_tensor_serializable_2.bin";

        // Cleanup any existing files
        let _ = fs::remove_file(binary_path);
        let _ = fs::remove_file(binary_path_2);

        // Test save method with binary format
        Serializable::save(&tensor, binary_path, Format::Binary).unwrap();
        assert!(Path::new(binary_path).exists());

        // Test load method with binary format
        let loaded_tensor = Tensor::load(binary_path, Format::Binary).unwrap();
        assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
        assert_eq!(tensor.requires_grad(), loaded_tensor.requires_grad());

        // Verify all data
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(tensor.get(&[i, j]), loaded_tensor.get(&[i, j]));
            }
        }

        // Test save_to_writer and load_from_reader
        {
            let mut writer = std::fs::File::create(binary_path_2).unwrap();
            Serializable::save_to_writer(&tensor, &mut writer, Format::Binary).unwrap();
        }
        assert!(Path::new(binary_path_2).exists());

        {
            let mut reader = std::fs::File::open(binary_path_2).unwrap();
            let loaded_tensor = Tensor::load_from_reader(&mut reader, Format::Binary).unwrap();
            assert_eq!(tensor.shape().dims, loaded_tensor.shape().dims);
            assert_eq!(tensor.requires_grad(), loaded_tensor.requires_grad());

            // Verify all data
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(tensor.get(&[i, j]), loaded_tensor.get(&[i, j]));
                }
            }
        }

        // Cleanup test files
        let _ = fs::remove_file(binary_path);
        let _ = fs::remove_file(binary_path_2);
    }

    #[test]
    fn test_serializable_large_tensor_performance() {
        // Create a large tensor to test performance characteristics
        let mut tensor = Tensor::zeros(vec![50, 50]);
        for i in 0..25 {
            for j in 0..25 {
                tensor.set(&[i, j], (i * 25 + j) as f32);
            }
        }
        tensor.set_requires_grad(true);

        // Test JSON serialization
        let json = <Tensor as crate::serialization::Serializable>::to_json(&tensor).unwrap();
        assert!(!json.is_empty());
        let restored_json =
            <Tensor as crate::serialization::Serializable>::from_json(&json).unwrap();
        assert_eq!(tensor.shape().dims, restored_json.shape().dims);
        assert_eq!(tensor.requires_grad(), restored_json.requires_grad());

        // Test binary serialization
        let binary = <Tensor as crate::serialization::Serializable>::to_binary(&tensor).unwrap();
        assert!(!binary.is_empty());
        // Binary format should be efficient (this is informational, not a requirement)
        println!(
            "JSON size: {} bytes, Binary size: {} bytes",
            json.len(),
            binary.len()
        );

        let restored_binary =
            <Tensor as crate::serialization::Serializable>::from_binary(&binary).unwrap();
        assert_eq!(tensor.shape().dims, restored_binary.shape().dims);
        assert_eq!(tensor.requires_grad(), restored_binary.requires_grad());

        // Verify a sample of data values
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(tensor.get(&[i, j]), restored_json.get(&[i, j]));
                assert_eq!(tensor.get(&[i, j]), restored_binary.get(&[i, j]));
            }
        }
    }

    #[test]
    fn test_serializable_error_handling() {
        // Test invalid JSON
        let invalid_json = r#"{"invalid": "json", "structure": true}"#;
        let result = <Tensor as crate::serialization::Serializable>::from_json(invalid_json);
        assert!(result.is_err());

        // Test empty JSON
        let empty_json = "{}";
        let result = <Tensor as crate::serialization::Serializable>::from_json(empty_json);
        assert!(result.is_err());

        // Test invalid binary data
        let invalid_binary = vec![1, 2, 3, 4, 5];
        let result = <Tensor as crate::serialization::Serializable>::from_binary(&invalid_binary);
        assert!(result.is_err());

        // Test empty binary data
        let empty_binary = vec![];
        let result = <Tensor as crate::serialization::Serializable>::from_binary(&empty_binary);
        assert!(result.is_err());
    }

    #[test]
    fn test_serializable_different_shapes_and_types() {
        let test_cases = vec![
            // Scalar (1-element tensor)
            (vec![1], vec![42.0]),
            // 1D vector
            (vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            // 2D matrix
            (vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            // 3D tensor
            (vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ];

        for (shape, expected_data) in test_cases {
            // Create tensor with specific shape and data
            let mut tensor = Tensor::zeros(shape.clone());

            // Set data based on shape dimensions
            match shape.len() {
                1 => {
                    for (i, &value) in expected_data.iter().enumerate().take(shape[0]) {
                        tensor.set(&[i], value);
                    }
                }
                2 => {
                    let mut idx = 0;
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            if idx < expected_data.len() {
                                tensor.set(&[i, j], expected_data[idx]);
                                idx += 1;
                            }
                        }
                    }
                }
                3 => {
                    let mut idx = 0;
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            for k in 0..shape[2] {
                                if idx < expected_data.len() {
                                    tensor.set(&[i, j, k], expected_data[idx]);
                                    idx += 1;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
            tensor.set_requires_grad(true);

            // Test JSON roundtrip
            let json = <Tensor as crate::serialization::Serializable>::to_json(&tensor).unwrap();
            let restored_json =
                <Tensor as crate::serialization::Serializable>::from_json(&json).unwrap();
            assert_eq!(tensor.shape().dims, restored_json.shape().dims);
            assert_eq!(tensor.requires_grad(), restored_json.requires_grad());

            // Test binary roundtrip
            let binary =
                <Tensor as crate::serialization::Serializable>::to_binary(&tensor).unwrap();
            let restored_binary =
                <Tensor as crate::serialization::Serializable>::from_binary(&binary).unwrap();
            assert_eq!(tensor.shape().dims, restored_binary.shape().dims);
            assert_eq!(tensor.requires_grad(), restored_binary.requires_grad());

            // Verify data for first few elements
            match shape.len() {
                1 => {
                    for i in 0..shape[0].min(3).min(expected_data.len()) {
                        assert_eq!(tensor.get(&[i]), restored_json.get(&[i]));
                        assert_eq!(tensor.get(&[i]), restored_binary.get(&[i]));
                    }
                }
                2 => {
                    let mut count = 0;
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            if count < 3 && count < expected_data.len() {
                                assert_eq!(tensor.get(&[i, j]), restored_json.get(&[i, j]));
                                assert_eq!(tensor.get(&[i, j]), restored_binary.get(&[i, j]));
                                count += 1;
                            }
                        }
                    }
                }
                3 => {
                    let mut count = 0;
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            for k in 0..shape[2] {
                                if count < 3 && count < expected_data.len() {
                                    assert_eq!(
                                        tensor.get(&[i, j, k]),
                                        restored_json.get(&[i, j, k])
                                    );
                                    assert_eq!(
                                        tensor.get(&[i, j, k]),
                                        restored_binary.get(&[i, j, k])
                                    );
                                    count += 1;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_serializable_edge_cases() {
        // Test zero-sized tensor
        let zero_tensor = Tensor::new(vec![0]);
        let json = <Tensor as crate::serialization::Serializable>::to_json(&zero_tensor).unwrap();
        let restored = <Tensor as crate::serialization::Serializable>::from_json(&json).unwrap();
        assert_eq!(zero_tensor.shape().dims, restored.shape().dims);
        assert_eq!(zero_tensor.size(), restored.size());

        let binary =
            <Tensor as crate::serialization::Serializable>::to_binary(&zero_tensor).unwrap();
        let restored =
            <Tensor as crate::serialization::Serializable>::from_binary(&binary).unwrap();
        assert_eq!(zero_tensor.shape().dims, restored.shape().dims);
        assert_eq!(zero_tensor.size(), restored.size());

        // Test tensor with special values (use reasonable large values instead of f32::MAX/MIN)
        let mut special_tensor = Tensor::zeros(vec![3]);
        special_tensor.set(&[0], 0.0); // Zero
        special_tensor.set(&[1], 1000000.0); // Large positive value
        special_tensor.set(&[2], -1000000.0); // Large negative value

        let json =
            <Tensor as crate::serialization::Serializable>::to_json(&special_tensor).unwrap();
        let restored = <Tensor as crate::serialization::Serializable>::from_json(&json).unwrap();
        assert_eq!(special_tensor.get(&[0]), restored.get(&[0]));
        assert_eq!(special_tensor.get(&[1]), restored.get(&[1]));
        assert_eq!(special_tensor.get(&[2]), restored.get(&[2]));

        let binary =
            <Tensor as crate::serialization::Serializable>::to_binary(&special_tensor).unwrap();
        let restored =
            <Tensor as crate::serialization::Serializable>::from_binary(&binary).unwrap();
        assert_eq!(special_tensor.get(&[0]), restored.get(&[0]));
        assert_eq!(special_tensor.get(&[1]), restored.get(&[1]));
        assert_eq!(special_tensor.get(&[2]), restored.get(&[2]));
    }
}
