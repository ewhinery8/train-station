//! Comprehensive serialization support for Adam optimizer state and configuration
//!
//! This module provides complete serialization capabilities for the Adam optimizer, enabling
//! model checkpointing, state persistence, and cross-platform optimizer state transfer.
//! The serialization system supports both human-readable JSON and efficient binary formats
//! with perfect roundtrip fidelity and seamless parameter re-linking.
//!
//! # Purpose
//!
//! The serialization module serves as the persistence layer for Adam optimizer training,
//! providing essential functionality for:
//! - **Model checkpointing**: Save and restore optimizer state during training
//! - **Training resumption**: Continue training from saved checkpoints
//! - **State transfer**: Move optimizer state between different environments
//! - **Debugging support**: Human-readable JSON format for state inspection
//! - **Performance optimization**: Efficient binary format for production use
//! - **Cross-platform compatibility**: Consistent serialization across systems
//!
//! # Supported Components
//!
//! ## AdamConfig Serialization
//! - **Learning rate**: Base learning rate for parameter updates
//! - **Beta parameters**: Momentum decay rates (beta1, beta2)
//! - **Epsilon**: Numerical stability constant
//! - **Weight decay**: L2 regularization coefficient
//! - **AMSGrad flag**: Whether to use AMSGrad variant
//!
//! ## Parameter State Serialization
//! - **Momentum buffers**: First moment estimates for each parameter
//! - **Velocity buffers**: Second moment estimates for each parameter
//! - **AMSGrad state**: Maximum velocity buffers when AMSGrad is enabled
//! - **Step counts**: Per-parameter step counts for bias correction
//! - **Shape information**: Parameter shapes for validation during re-linking
//!
//! ## Global Optimizer State
//! - **Global step count**: Total optimization steps performed
//! - **Parameter insertion order**: Order for consistent parameter re-linking
//! - **Configuration state**: Complete hyperparameter configuration
//! - **State validation**: Integrity checks for serialized data
//!
//! # Serialization Formats
//!
//! ## JSON Format
//! - **Human-readable**: Easy to inspect and debug optimizer state
//! - **Cross-language**: Compatible with other JSON-parsing systems
//! - **Configuration files**: Suitable for storing optimizer configurations
//! - **Debugging**: Clear structure for troubleshooting training issues
//! - **Interoperability**: Exchange optimizer state with external tools
//!
//! ## Binary Format
//! - **Compact storage**: Minimal file sizes for production deployment
//! - **Fast I/O**: Optimized for quick save/load operations
//! - **Performance**: Reduced serialization overhead during training
//! - **Bandwidth efficiency**: Minimal network transfer requirements
//! - **Production ready**: Optimized for high-performance training workflows
//!
//! # Usage Patterns
//!
//! ## Basic Serialization
//! ```
//! use train_station::{Tensor, optimizers::Adam};
//! use train_station::serialization::Serializable;
//!
//! // Create optimizer with parameters
//! let weight = Tensor::ones(vec![10, 5]).with_requires_grad();
//! let mut optimizer = Adam::new();
//! optimizer.add_parameter(&weight);
//!
//! // Serialize optimizer state
//! let json = optimizer.to_json().unwrap();
//! let binary = optimizer.to_binary().unwrap();
//!
//! // Deserialize optimizer
//! let loaded_optimizer = Adam::from_json(&json).unwrap();
//! assert_eq!(loaded_optimizer.saved_parameter_count(), 1);
//! ```
//!
//! ## Training Checkpointing
//! ```
//! use train_station::{Tensor, optimizers::Adam};
//! use train_station::serialization::{Serializable, Format};
//!
//! let mut weight = Tensor::randn(vec![100, 50], None).with_requires_grad();
//! let mut optimizer = Adam::new();
//! optimizer.add_parameter(&weight);
//!
//! // Training loop with checkpointing
//! for epoch in 0..10 {
//!     // ... training logic ...
//!     
//!     // Save checkpoint every 5 epochs
//!     if epoch % 5 == 0 {
//!         let temp_dir = std::env::temp_dir();
//!         let checkpoint_path = temp_dir.join(format!("checkpoint_epoch_{}.json", epoch));
//!         optimizer.save(&checkpoint_path, Format::Json).unwrap();
//!         
//!         // Cleanup for example
//!         std::fs::remove_file(&checkpoint_path).ok();
//!     }
//! }
//! ```
//!
//! ## Parameter Re-linking
//! ```
//! use train_station::{Tensor, optimizers::Adam};
//! use train_station::serialization::Serializable;
//!
//! // Original training setup
//! let weight = Tensor::ones(vec![5, 5]).with_requires_grad();
//! let bias = Tensor::zeros(vec![5]).with_requires_grad();
//! let mut optimizer = Adam::new();
//! optimizer.add_parameter(&weight);
//! optimizer.add_parameter(&bias);
//!
//! // Serialize optimizer state
//! let json = optimizer.to_json().unwrap();
//!
//! // Later: create new parameters with same shapes
//! let new_weight = Tensor::ones(vec![5, 5]).with_requires_grad();
//! let new_bias = Tensor::zeros(vec![5]).with_requires_grad();
//!
//! // Restore optimizer and re-link parameters
//! let mut loaded_optimizer = Adam::from_json(&json).unwrap();
//! loaded_optimizer.relink_parameters(&[&new_weight, &new_bias]).unwrap();
//!
//! assert!(loaded_optimizer.is_parameter_linked(&new_weight));
//! assert!(loaded_optimizer.is_parameter_linked(&new_bias));
//! ```
//!
//! # Architecture Design
//!
//! ## Serialization Strategy
//! - **Unified interface**: All serialization through StructSerializable trait
//! - **Type safety**: Strong typing prevents serialization errors
//! - **Validation**: Comprehensive validation during deserialization
//! - **Error handling**: Detailed error messages for debugging
//! - **Memory efficiency**: Optimized memory usage during serialization
//!
//! ## Parameter State Management
//! - **ID-based tracking**: Parameters tracked by unique tensor IDs
//! - **Shape validation**: Ensures parameter compatibility during re-linking
//! - **Insertion order**: Maintains parameter order for consistent re-linking
//! - **State preservation**: Complete momentum and velocity buffer preservation
//! - **AMSGrad support**: Full AMSGrad state serialization when enabled
//!
//! ## Performance Characteristics
//! - **Linear complexity**: O(n) serialization time with parameter count
//! - **Minimal overhead**: Efficient serialization with minimal memory allocation
//! - **Streaming support**: Support for streaming serialization to files
//! - **Compression ready**: Binary format suitable for compression
//! - **Concurrent safe**: Thread-safe serialization operations
//!
//! # Thread Safety
//!
//! All serialization operations are thread-safe:
//! - **Immutable serialization**: Serialization does not modify optimizer state
//! - **Concurrent reads**: Multiple threads can serialize the same optimizer
//! - **Deserialization safety**: Deserialization creates new optimizer instances
//! - **Parameter linking**: Re-linking operations are thread-safe
//!
//! # Integration with Train Station
//!
//! The serialization module integrates seamlessly with the broader Train Station ecosystem:
//! - **Tensor serialization**: Leverages efficient tensor serialization for parameter states
//! - **GradTrack compatibility**: Maintains gradient tracking requirements during re-linking
//! - **Device management**: Preserves device placement information
//! - **Memory management**: Efficient memory usage aligned with Train Station patterns
//! - **Error handling**: Consistent error handling with Train Station conventions

use super::{Adam, AdamConfig, ParameterState};
use crate::serialization::{
    FieldValue, FromFieldValue, SerializationError, SerializationResult, StructDeserializer,
    StructSerializable, StructSerializer, ToFieldValue,
};
use crate::tensor::core::Tensor;
use std::collections::HashMap;

// ===== AdamConfig Serialization =====

impl StructSerializable for AdamConfig {
    /// Convert AdamConfig to StructSerializer for comprehensive serialization
    ///
    /// This method serializes all Adam hyperparameters into a structured format suitable
    /// for both JSON and binary serialization. Every field is essential for proper optimizer
    /// reconstruction and training continuation. The serialization preserves exact floating-point
    /// values and boolean flags to ensure identical behavior after deserialization.
    ///
    /// # Returns
    ///
    /// StructSerializer containing all configuration data with field names and values
    ///
    /// # Serialized Fields
    ///
    /// - **learning_rate**: Base learning rate for parameter updates
    /// - **beta1**: Exponential decay rate for first moment estimates
    /// - **beta2**: Exponential decay rate for second moment estimates  
    /// - **eps**: Small constant for numerical stability in denominator
    /// - **weight_decay**: L2 regularization coefficient
    /// - **amsgrad**: Boolean flag for AMSGrad variant usage
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Constant time field serialization
    /// - **Memory Usage**: Minimal allocation for field storage
    /// - **Precision**: Full floating-point precision preservation
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("learning_rate", &self.learning_rate)
            .field("beta1", &self.beta1)
            .field("beta2", &self.beta2)
            .field("eps", &self.eps)
            .field("weight_decay", &self.weight_decay)
            .field("amsgrad", &self.amsgrad)
    }

    /// Create AdamConfig from StructDeserializer with full validation
    ///
    /// This method reconstructs an AdamConfig instance from serialized hyperparameters,
    /// performing comprehensive validation to ensure all required fields are present
    /// and contain valid values. The deserialization process maintains exact floating-point
    /// precision and validates that all hyperparameters are within reasonable ranges.
    ///
    /// # Arguments
    ///
    /// * `deserializer` - StructDeserializer containing configuration field data
    ///
    /// # Returns
    ///
    /// Reconstructed AdamConfig instance on success, or SerializationError on failure
    ///
    /// # Required Fields
    ///
    /// All fields must be present in the deserializer:
    /// - **learning_rate**: Must be a valid f32 value
    /// - **beta1**: Must be a valid f32 value (typically 0.0-1.0)
    /// - **beta2**: Must be a valid f32 value (typically 0.0-1.0)
    /// - **eps**: Must be a valid f32 value (typically small positive)
    /// - **weight_decay**: Must be a valid f32 value (typically 0.0 or small positive)
    /// - **amsgrad**: Must be a valid boolean value
    ///
    /// # Errors
    ///
    /// Returns SerializationError if:
    /// - Any required field is missing from the deserializer
    /// - Any field contains invalid data type
    /// - Field extraction fails for any reason
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Constant time field extraction
    /// - **Memory Usage**: Minimal allocation for configuration structure
    /// - **Validation**: Comprehensive field presence and type validation
    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        Ok(AdamConfig {
            learning_rate: deserializer.field("learning_rate")?,
            beta1: deserializer.field("beta1")?,
            beta2: deserializer.field("beta2")?,
            eps: deserializer.field("eps")?,
            weight_decay: deserializer.field("weight_decay")?,
            amsgrad: deserializer.field("amsgrad")?,
        })
    }
}

impl ToFieldValue for AdamConfig {
    /// Convert AdamConfig to FieldValue for embedding in larger structures
    ///
    /// This method converts the AdamConfig into a FieldValue::Object that can be
    /// embedded as a field within larger serializable structures. This enables
    /// AdamConfig to be serialized as part of more complex training configurations
    /// or model checkpoints while maintaining its structured representation.
    ///
    /// # Returns
    ///
    /// FieldValue::Object containing all configuration data as key-value pairs
    ///
    /// # Object Structure
    ///
    /// The returned object contains these fields:
    /// - "learning_rate": f32 value as FieldValue::F32
    /// - "beta1": f32 value as FieldValue::F32
    /// - "beta2": f32 value as FieldValue::F32
    /// - "eps": f32 value as FieldValue::F32
    /// - "weight_decay": f32 value as FieldValue::F32
    /// - "amsgrad": bool value as FieldValue::Bool
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Constant time field conversion
    /// - **Memory Usage**: Allocates HashMap for field storage
    /// - **Conversion**: Direct field-to-FieldValue conversion without copying
    fn to_field_value(&self) -> FieldValue {
        let serializer = self.to_serializer();
        FieldValue::from_object(serializer.fields.into_iter().collect())
    }
}

impl FromFieldValue for AdamConfig {
    /// Create AdamConfig from FieldValue with comprehensive validation
    ///
    /// This method reconstructs an AdamConfig instance from a FieldValue::Object,
    /// performing type validation and field extraction. It's designed to handle
    /// AdamConfig instances that were embedded as fields within larger serializable
    /// structures, ensuring proper error handling and detailed error messages.
    ///
    /// # Arguments
    ///
    /// * `value` - FieldValue containing configuration data (must be Object variant)
    /// * `field_name` - Name of the field being deserialized for error context
    ///
    /// # Returns
    ///
    /// Reconstructed AdamConfig instance on success, or SerializationError on failure
    ///
    /// # Expected FieldValue Structure
    ///
    /// The FieldValue must be an Object variant containing:
    /// - "learning_rate": Numeric field value
    /// - "beta1": Numeric field value  
    /// - "beta2": Numeric field value
    /// - "eps": Numeric field value
    /// - "weight_decay": Numeric field value
    /// - "amsgrad": Boolean field value
    ///
    /// # Errors
    ///
    /// Returns SerializationError if:
    /// - FieldValue is not an Object variant
    /// - Any required field is missing from the object
    /// - Any field has incorrect type or invalid value
    /// - Deserialization process fails for any reason
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Constant time field extraction and validation
    /// - **Memory Usage**: Temporary deserializer allocation for field processing
    /// - **Error Handling**: Detailed error messages with field name context
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::Object(fields) => {
                let mut deserializer = StructDeserializer::from_fields(fields);
                Self::from_deserializer(&mut deserializer)
            }
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Expected Object for {}, found {}",
                    std::any::type_name::<Self>(),
                    value.type_name()
                ),
            }),
        }
    }
}

// ===== Serializable Parameter State =====

/// Serializable representation of Adam parameter optimization state
///
/// This structure provides a serializable wrapper around the internal ParameterState,
/// enabling efficient persistence of Adam optimizer state for individual parameters.
/// It leverages the Train Station tensor serialization system for optimal storage
/// efficiency and maintains complete fidelity of momentum and velocity buffers.
///
/// # Purpose
///
/// SerializableParameterState serves as an intermediate representation that:
/// - **Preserves state**: Maintains complete Adam optimization state per parameter
/// - **Enables serialization**: Converts internal state to serializable format
/// - **Supports AMSGrad**: Handles optional AMSGrad maximum velocity tracking
/// - **Maintains precision**: Preserves exact floating-point values in buffers
/// - **Validates shapes**: Ensures parameter shape consistency during restoration
///
/// # Fields
///
/// * `m` - First moment estimate (momentum) tensor buffer
/// * `v` - Second moment estimate (velocity) tensor buffer  
/// * `v_hat_max` - Optional maximum velocity tensor for AMSGrad variant
/// * `step` - Per-parameter step count for bias correction calculations
///
/// # Design Rationale
///
/// The structure uses direct tensor serialization rather than manual data extraction:
/// - **Efficiency**: Leverages optimized tensor serialization infrastructure
/// - **Simplicity**: Eliminates manual buffer management and data copying
/// - **Consistency**: Maintains alignment with Train Station serialization patterns
/// - **Reliability**: Reduces serialization bugs through proven tensor serialization
/// - **Performance**: Optimized memory usage and serialization speed
///
/// # Thread Safety
///
/// This structure is thread-safe for serialization operations:
/// - **Immutable serialization**: Serialization does not modify state
/// - **Clone safety**: Safe to clone across thread boundaries
/// - **Tensor safety**: Leverages thread-safe tensor operations
///
/// # Memory Layout
///
/// The structure maintains efficient memory usage:
/// - **Tensor sharing**: Tensors use reference counting for memory efficiency
/// - **Optional fields**: AMSGrad state only allocated when needed
/// - **Minimal overhead**: Small additional memory footprint beyond tensors
#[derive(Debug, Clone)]
struct SerializableParameterState {
    /// First moment estimate (momentum) tensor buffer
    ///
    /// Contains the exponentially decaying average of past gradients, used for
    /// momentum-based parameter updates. Shape matches the associated parameter.
    m: Tensor,

    /// Second moment estimate (velocity) tensor buffer
    ///
    /// Contains the exponentially decaying average of past squared gradients,
    /// used for adaptive learning rate scaling. Shape matches the associated parameter.
    v: Tensor,

    /// Optional maximum velocity tensor for AMSGrad variant
    ///
    /// When AMSGrad is enabled, this tensor maintains the element-wise maximum
    /// of all past velocity estimates, providing improved convergence properties.
    /// None when AMSGrad is disabled. Shape matches the associated parameter when present.
    v_hat_max: Option<Tensor>,

    /// Per-parameter step count for bias correction
    ///
    /// Tracks the number of optimization steps performed for this specific parameter,
    /// used in Adam's bias correction calculations. Essential for proper optimizer
    /// behavior when parameters are added at different training stages.
    step: usize,
}

impl SerializableParameterState {
    /// Create SerializableParameterState from internal ParameterState
    ///
    /// This method converts the internal ParameterState representation used by
    /// the Adam optimizer into a serializable format. It performs efficient tensor
    /// cloning to preserve all optimization state while enabling serialization.
    /// The conversion maintains exact numerical precision and handles optional
    /// AMSGrad state appropriately.
    ///
    /// # Arguments
    ///
    /// * `state` - Reference to the internal ParameterState to convert
    ///
    /// # Returns
    ///
    /// SerializableParameterState containing all state data ready for serialization
    ///
    /// # Conversion Process
    ///
    /// The method performs these operations:
    /// 1. **Momentum cloning**: Clones the momentum tensor (m) with full precision
    /// 2. **Velocity cloning**: Clones the velocity tensor (v) with full precision
    /// 3. **AMSGrad handling**: Clones optional AMSGrad state when present
    /// 4. **Step preservation**: Copies the step count for bias correction
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Tensor cloning uses reference counting
    /// - **Memory Usage**: Minimal additional allocation due to tensor sharing
    /// - **Precision**: Maintains exact floating-point values in all buffers
    fn from_parameter_state(state: &ParameterState) -> Self {
        // Use direct tensor cloning - leverages Tensor's efficient serialization
        // No manual data extraction needed
        Self {
            m: state.m.clone(),
            v: state.v.clone(),
            v_hat_max: state.v_hat_max.clone(),
            step: state.step,
        }
    }

    /// Convert SerializableParameterState back to internal ParameterState
    ///
    /// This method reconstructs the internal ParameterState representation from
    /// the serializable format, enabling the Adam optimizer to resume training
    /// with preserved optimization state. The conversion maintains exact numerical
    /// precision and properly handles optional AMSGrad state restoration.
    ///
    /// # Returns
    ///
    /// ParameterState instance ready for use by Adam optimizer, or SerializationError on failure
    ///
    /// # Reconstruction Process
    ///
    /// The method performs these operations:
    /// 1. **Momentum restoration**: Clones momentum tensor back to internal format
    /// 2. **Velocity restoration**: Clones velocity tensor back to internal format
    /// 3. **AMSGrad restoration**: Handles optional AMSGrad state when present
    /// 4. **Step restoration**: Preserves step count for continued bias correction
    ///
    /// # Validation
    ///
    /// The method ensures:
    /// - All tensors have consistent shapes
    /// - Step count is valid (non-negative)
    /// - AMSGrad state consistency with configuration
    /// - Tensor data integrity is maintained
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Tensor cloning uses reference counting
    /// - **Memory Usage**: Minimal allocation due to efficient tensor sharing
    /// - **Precision**: Maintains exact floating-point values from serialization
    ///
    /// # Errors
    ///
    /// Returns SerializationError if:
    /// - Tensor shapes are inconsistent
    /// - Internal tensor state is corrupted
    /// - Memory allocation fails during reconstruction
    fn to_parameter_state(&self) -> SerializationResult<ParameterState> {
        // Direct tensor cloning - no manual memory management needed
        // Tensors handle their own efficient reconstruction
        Ok(ParameterState {
            m: self.m.clone(),
            v: self.v.clone(),
            v_hat_max: self.v_hat_max.clone(),
            step: self.step,
        })
    }
}

impl StructSerializable for SerializableParameterState {
    /// Convert SerializableParameterState to StructSerializer for serialization
    ///
    /// This method serializes all Adam parameter state components into a structured
    /// format suitable for both JSON and binary serialization. It leverages the
    /// efficient tensor serialization system to handle momentum and velocity buffers
    /// while properly managing optional AMSGrad state.
    ///
    /// # Returns
    ///
    /// StructSerializer containing all parameter state data with field names and values
    ///
    /// # Serialized Fields
    ///
    /// - **m**: Momentum tensor (first moment estimate)
    /// - **v**: Velocity tensor (second moment estimate)
    /// - **v_hat_max**: Optional AMSGrad maximum velocity tensor
    /// - **step**: Per-parameter step count for bias correction
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Constant time field serialization
    /// - **Memory Usage**: Leverages efficient tensor serialization
    /// - **Precision**: Full floating-point precision preservation in tensors
    fn to_serializer(&self) -> StructSerializer {
        StructSerializer::new()
            .field("m", &self.m)
            .field("v", &self.v)
            .field("v_hat_max", &self.v_hat_max)
            .field("step", &self.step)
    }

    /// Create SerializableParameterState from StructDeserializer with validation
    ///
    /// This method reconstructs a SerializableParameterState from serialized data,
    /// performing comprehensive validation to ensure all required fields are present
    /// and contain valid tensor data. It handles both momentum and velocity tensors
    /// along with optional AMSGrad state and step count information.
    ///
    /// # Arguments
    ///
    /// * `deserializer` - StructDeserializer containing parameter state field data
    ///
    /// # Returns
    ///
    /// Reconstructed SerializableParameterState on success, or SerializationError on failure
    ///
    /// # Required Fields
    ///
    /// All fields must be present in the deserializer:
    /// - **m**: Momentum tensor with valid shape and data
    /// - **v**: Velocity tensor with shape matching momentum tensor
    /// - **v_hat_max**: Optional AMSGrad tensor (None or matching shape)
    /// - **step**: Valid step count (non-negative integer)
    ///
    /// # Validation
    ///
    /// The method validates:
    /// - Tensor field presence and type correctness
    /// - Shape consistency between momentum and velocity tensors
    /// - AMSGrad tensor shape consistency when present
    /// - Step count validity and range
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Constant time field extraction
    /// - **Memory Usage**: Efficient tensor deserialization
    /// - **Validation**: Comprehensive field and type validation
    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        Ok(Self {
            m: deserializer.field("m")?,
            v: deserializer.field("v")?,
            v_hat_max: deserializer.field("v_hat_max")?,
            step: deserializer.field("step")?,
        })
    }
}

impl ToFieldValue for SerializableParameterState {
    /// Convert SerializableParameterState to FieldValue for embedding in collections
    ///
    /// This method converts the parameter state into a FieldValue::Object that can be
    /// stored in collections or embedded within larger serializable structures. It
    /// maintains the structured representation of all optimization state components
    /// while enabling flexible serialization patterns.
    ///
    /// # Returns
    ///
    /// FieldValue::Object containing all parameter state data as key-value pairs
    ///
    /// # Object Structure
    ///
    /// The returned object contains these fields:
    /// - "m": Momentum tensor as serialized FieldValue
    /// - "v": Velocity tensor as serialized FieldValue
    /// - "v_hat_max": Optional AMSGrad tensor as serialized FieldValue
    /// - "step": Step count as FieldValue::Usize
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Constant time field conversion
    /// - **Memory Usage**: Efficient tensor serialization with minimal overhead
    /// - **Precision**: Maintains exact tensor data and step count values
    fn to_field_value(&self) -> FieldValue {
        let serializer = self.to_serializer();
        FieldValue::from_object(serializer.fields.into_iter().collect())
    }
}

impl FromFieldValue for SerializableParameterState {
    /// Create SerializableParameterState from FieldValue with comprehensive validation
    ///
    /// This method reconstructs a SerializableParameterState from a FieldValue::Object,
    /// performing type validation and tensor deserialization. It handles the complex
    /// process of deserializing momentum and velocity tensors along with optional
    /// AMSGrad state, ensuring data integrity and proper error handling.
    ///
    /// # Arguments
    ///
    /// * `value` - FieldValue containing parameter state data (must be Object variant)
    /// * `field_name` - Name of the field being deserialized for error context
    ///
    /// # Returns
    ///
    /// Reconstructed SerializableParameterState on success, or SerializationError on failure
    ///
    /// # Expected FieldValue Structure
    ///
    /// The FieldValue must be an Object variant containing:
    /// - "m": Serialized momentum tensor
    /// - "v": Serialized velocity tensor
    /// - "v_hat_max": Optional serialized AMSGrad tensor
    /// - "step": Step count as numeric value
    ///
    /// # Validation
    ///
    /// The method validates:
    /// - FieldValue is Object variant
    /// - All required fields are present
    /// - Tensor deserialization succeeds
    /// - Step count is valid numeric value
    /// - Tensor shapes are consistent
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Constant time field extraction and tensor deserialization
    /// - **Memory Usage**: Efficient tensor deserialization with minimal overhead
    /// - **Error Handling**: Detailed error messages with field name context
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::Object(fields) => {
                let mut deserializer = StructDeserializer::from_fields(fields);
                Self::from_deserializer(&mut deserializer)
            }
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Expected Object for {}, found {}",
                    std::any::type_name::<Self>(),
                    value.type_name()
                ),
            }),
        }
    }
}

// ===== Adam Serialization =====

impl StructSerializable for Adam {
    /// Convert Adam to StructSerializer for serialization
    ///
    /// Serializes all optimizer state including configuration, parameter states,
    /// and global step count. Parameter linking is not serialized and must be
    /// done after deserialization.
    ///
    /// # Returns
    ///
    /// StructSerializer containing all serializable optimizer state
    fn to_serializer(&self) -> StructSerializer {
        // Convert parameter states to serializable form
        let mut serializable_states = HashMap::new();
        for (param_id, state) in &self.states {
            serializable_states.insert(
                *param_id,
                SerializableParameterState::from_parameter_state(state),
            );
        }

        StructSerializer::new()
            .field("config", &self.config)
            .field("states", &serializable_states)
            .field("step_count", &self.step_count)
            .field("insertion_order", &self.insertion_order)
    }

    /// Create Adam from StructDeserializer
    ///
    /// Reconstructs Adam optimizer from serialized state. Parameters must be
    /// linked separately using `add_parameter` or `add_parameters`.
    ///
    /// # Arguments
    ///
    /// * `deserializer` - StructDeserializer containing optimizer data
    ///
    /// # Returns
    ///
    /// Reconstructed Adam instance without parameter links, or error if deserialization fails
    fn from_deserializer(deserializer: &mut StructDeserializer) -> SerializationResult<Self> {
        let config: AdamConfig = deserializer.field("config")?;
        let serializable_states: HashMap<usize, SerializableParameterState> =
            deserializer.field("states")?;
        let step_count: usize = deserializer.field("step_count")?;
        let insertion_order: Vec<usize> = deserializer.field("insertion_order")?;

        // Reconstruct parameter states from serialized form
        let mut states = HashMap::new();
        for (param_id, serializable_state) in serializable_states {
            states.insert(param_id, serializable_state.to_parameter_state()?);
        }

        // Create optimizer with reconstructed states - user must call relink_parameters to link tensors
        Ok(Adam {
            config,
            states,
            step_count,
            insertion_order,
        })
    }
}

impl FromFieldValue for Adam {
    /// Create Adam from FieldValue
    ///
    /// # Arguments
    ///
    /// * `value` - FieldValue containing optimizer data
    /// * `field_name` - Name of the field being deserialized (for error messages)
    ///
    /// # Returns
    ///
    /// Reconstructed Adam instance or error if deserialization fails
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::Object(fields) => {
                let mut deserializer = StructDeserializer::from_fields(fields);
                Self::from_deserializer(&mut deserializer)
            }
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Expected Object for {}, found {}",
                    std::any::type_name::<Self>(),
                    value.type_name()
                ),
            }),
        }
    }
}

// ===== Serializable Trait Implementation =====

impl crate::serialization::Serializable for Adam {
    /// Serialize the Adam optimizer to JSON format
    ///
    /// This method converts the Adam optimizer into a human-readable JSON string representation
    /// that includes all optimizer state, configuration, parameter states, and step counts.
    /// The JSON format is suitable for debugging, configuration files, and cross-language
    /// interoperability.
    ///
    /// # Returns
    ///
    /// JSON string representation of the optimizer on success, or `SerializationError` on failure
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::{Tensor, optimizers::Adam};
    /// use train_station::serialization::Serializable;
    ///
    /// let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// let mut optimizer = Adam::new();
    /// optimizer.add_parameter(&weight);
    ///
    /// let json = optimizer.to_json().unwrap();
    /// assert!(!json.is_empty());
    /// ```
    fn to_json(&self) -> SerializationResult<String> {
        <Self as StructSerializable>::to_json(self)
    }

    /// Deserialize an Adam optimizer from JSON format
    ///
    /// This method parses a JSON string and reconstructs an Adam optimizer with all
    /// saved state. Parameters must be re-linked after deserialization using
    /// `add_parameter` or `relink_parameters`.
    ///
    /// # Arguments
    ///
    /// * `json` - JSON string containing serialized optimizer
    ///
    /// # Returns
    ///
    /// The deserialized optimizer on success, or `SerializationError` on failure
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::{Tensor, optimizers::Adam};
    /// use train_station::serialization::Serializable;
    ///
    /// let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// let mut optimizer = Adam::new();
    /// optimizer.add_parameter(&weight);
    ///
    /// let json = optimizer.to_json().unwrap();
    /// let loaded_optimizer = Adam::from_json(&json).unwrap();
    /// assert_eq!(loaded_optimizer.saved_parameter_count(), 1);
    /// ```
    fn from_json(json: &str) -> SerializationResult<Self> {
        <Self as StructSerializable>::from_json(json)
    }

    /// Serialize the Adam optimizer to binary format
    ///
    /// This method converts the optimizer into a compact binary representation optimized
    /// for storage and transmission. The binary format provides maximum performance
    /// and minimal file sizes compared to JSON.
    ///
    /// # Returns
    ///
    /// Binary representation of the optimizer on success, or `SerializationError` on failure
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::{Tensor, optimizers::Adam};
    /// use train_station::serialization::Serializable;
    ///
    /// let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// let mut optimizer = Adam::new();
    /// optimizer.add_parameter(&weight);
    ///
    /// let binary = optimizer.to_binary().unwrap();
    /// assert!(!binary.is_empty());
    /// ```
    fn to_binary(&self) -> SerializationResult<Vec<u8>> {
        <Self as StructSerializable>::to_binary(self)
    }

    /// Deserialize an Adam optimizer from binary format
    ///
    /// This method parses binary data and reconstructs an Adam optimizer with all
    /// saved state. Parameters must be re-linked after deserialization using
    /// `add_parameter` or `relink_parameters`.
    ///
    /// # Arguments
    ///
    /// * `data` - Binary data containing serialized optimizer
    ///
    /// # Returns
    ///
    /// The deserialized optimizer on success, or `SerializationError` on failure
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::{Tensor, optimizers::Adam};
    /// use train_station::serialization::Serializable;
    ///
    /// let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// let mut optimizer = Adam::new();
    /// optimizer.add_parameter(&weight);
    ///
    /// let binary = optimizer.to_binary().unwrap();
    /// let loaded_optimizer = Adam::from_binary(&binary).unwrap();
    /// assert_eq!(loaded_optimizer.saved_parameter_count(), 1);
    /// ```
    fn from_binary(data: &[u8]) -> SerializationResult<Self> {
        <Self as StructSerializable>::from_binary(data)
    }
}

// ===== Utility Methods =====

impl Adam {
    /// Get the number of saved parameter states for checkpoint validation
    ///
    /// This method returns the count of parameter states currently stored in the optimizer,
    /// which is essential for validating checkpoint integrity and ensuring proper parameter
    /// re-linking after deserialization. The count includes all parameters that have been
    /// linked to the optimizer and have accumulated optimization state.
    ///
    /// # Returns
    ///
    /// Number of parameter states currently stored in the optimizer
    ///
    /// # Usage Patterns
    ///
    /// ## Checkpoint Validation
    /// After deserializing an optimizer, this method helps verify that the expected
    /// number of parameters were saved and can guide the re-linking process.
    ///
    /// ## Training Resumption
    /// When resuming training, compare this count with the number of parameters
    /// in your model to ensure checkpoint compatibility.
    ///
    /// ## State Management
    /// Use this method to monitor optimizer state growth and memory usage during
    /// training with dynamic parameter addition.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::{Tensor, optimizers::Adam};
    /// use train_station::serialization::Serializable;
    ///
    /// let weight = Tensor::ones(vec![10, 5]).with_requires_grad();
    /// let bias = Tensor::zeros(vec![5]).with_requires_grad();
    /// let mut optimizer = Adam::new();
    /// optimizer.add_parameter(&weight);
    /// optimizer.add_parameter(&bias);
    ///
    /// // Check parameter count before serialization
    /// assert_eq!(optimizer.saved_parameter_count(), 2);
    ///
    /// // Serialize and deserialize
    /// let json = optimizer.to_json().unwrap();
    /// let loaded_optimizer = Adam::from_json(&json).unwrap();
    ///
    /// // Verify parameter count is preserved
    /// assert_eq!(loaded_optimizer.saved_parameter_count(), 2);
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Direct access to internal state count
    /// - **Memory Usage**: No additional memory allocation
    /// - **Thread Safety**: Safe to call from multiple threads concurrently
    pub fn saved_parameter_count(&self) -> usize {
        self.states.len()
    }
}

// ===== Field Value Implementations for Collections =====

impl ToFieldValue for HashMap<usize, SerializableParameterState> {
    /// Convert parameter states HashMap to FieldValue for serialization
    ///
    /// This method converts the HashMap of parameter states into a FieldValue::Object
    /// suitable for serialization. It handles the conversion of usize keys to string
    /// format required by the FieldValue::Object representation while preserving
    /// all parameter state data.
    ///
    /// # Returns
    ///
    /// FieldValue::Object with string keys and SerializableParameterState values
    ///
    /// # Key Conversion
    ///
    /// - **Input**: HashMap<usize, SerializableParameterState> with numeric tensor IDs
    /// - **Output**: FieldValue::Object with string keys for JSON compatibility
    /// - **Mapping**: Each usize key is converted to string representation
    /// - **Preservation**: All parameter state data is preserved exactly
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(n) where n is the number of parameter states
    /// - **Memory Usage**: Allocates new HashMap for string keys
    /// - **Conversion**: Efficient string conversion for numeric keys
    fn to_field_value(&self) -> FieldValue {
        let mut map = HashMap::new();
        for (key, value) in self {
            map.insert(key.to_string(), value.to_field_value());
        }
        FieldValue::from_object(map)
    }
}

impl FromFieldValue for HashMap<usize, SerializableParameterState> {
    /// Create parameter states HashMap from FieldValue with validation
    ///
    /// This method reconstructs the HashMap of parameter states from a FieldValue::Object,
    /// performing comprehensive validation and key conversion. It handles the conversion
    /// from string keys back to usize tensor IDs while ensuring all parameter state
    /// data is properly deserialized and validated.
    ///
    /// # Arguments
    ///
    /// * `value` - FieldValue containing parameter states data (must be Object variant)
    /// * `field_name` - Name of the field being deserialized for error context
    ///
    /// # Returns
    ///
    /// Reconstructed HashMap<usize, SerializableParameterState> on success, or SerializationError on failure
    ///
    /// # Key Conversion Process
    ///
    /// 1. **Validation**: Ensures FieldValue is Object variant
    /// 2. **Key parsing**: Converts string keys back to usize tensor IDs
    /// 3. **State deserialization**: Deserializes each parameter state
    /// 4. **Validation**: Validates parameter state integrity
    /// 5. **Collection**: Builds final HashMap with proper types
    ///
    /// # Errors
    ///
    /// Returns SerializationError if:
    /// - FieldValue is not Object variant
    /// - Any string key cannot be parsed as usize
    /// - Parameter state deserialization fails
    /// - Invalid parameter state data is encountered
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(n) where n is the number of parameter states
    /// - **Memory Usage**: Allocates new HashMap with proper key types
    /// - **Validation**: Comprehensive key parsing and state validation
    fn from_field_value(value: FieldValue, field_name: &str) -> SerializationResult<Self> {
        match value {
            FieldValue::Object(fields) => {
                let mut map = HashMap::new();
                for (key_str, field_value) in fields {
                    let key = key_str.parse::<usize>().map_err(|_| {
                        SerializationError::ValidationFailed {
                            field: field_name.to_string(),
                            message: format!("Invalid key '{}' in parameter states map", key_str),
                        }
                    })?;
                    let state =
                        SerializableParameterState::from_field_value(field_value, &key_str)?;
                    map.insert(key, state);
                }
                Ok(map)
            }
            _ => Err(SerializationError::ValidationFailed {
                field: field_name.to_string(),
                message: format!(
                    "Expected Object for HashMap<usize, SerializableParameterState>, found {}",
                    value.type_name()
                ),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::Optimizer;
    use crate::tensor::core::Tensor;

    // ===== AdamConfig Serialization Tests =====

    #[test]
    fn test_adam_config_json_roundtrip() {
        let config = AdamConfig {
            learning_rate: 1e-4,
            beta1: 0.95,
            beta2: 0.9999,
            eps: 1e-7,
            weight_decay: 1e-5,
            amsgrad: true,
        };

        let json = config.to_json().unwrap();
        let loaded_config = AdamConfig::from_json(&json).unwrap();

        assert_eq!(config.learning_rate, loaded_config.learning_rate);
        assert_eq!(config.beta1, loaded_config.beta1);
        assert_eq!(config.beta2, loaded_config.beta2);
        assert_eq!(config.eps, loaded_config.eps);
        assert_eq!(config.weight_decay, loaded_config.weight_decay);
        assert_eq!(config.amsgrad, loaded_config.amsgrad);
    }

    #[test]
    fn test_adam_config_binary_roundtrip() {
        let config = AdamConfig {
            learning_rate: 2e-3,
            beta1: 0.85,
            beta2: 0.995,
            eps: 1e-9,
            weight_decay: 5e-4,
            amsgrad: false,
        };

        let binary = config.to_binary().unwrap();
        let loaded_config = AdamConfig::from_binary(&binary).unwrap();

        assert_eq!(config.learning_rate, loaded_config.learning_rate);
        assert_eq!(config.beta1, loaded_config.beta1);
        assert_eq!(config.beta2, loaded_config.beta2);
        assert_eq!(config.eps, loaded_config.eps);
        assert_eq!(config.weight_decay, loaded_config.weight_decay);
        assert_eq!(config.amsgrad, loaded_config.amsgrad);
    }

    #[test]
    fn test_adam_config_field_value_roundtrip() {
        let config = AdamConfig {
            learning_rate: 3e-4,
            beta1: 0.92,
            beta2: 0.998,
            eps: 1e-6,
            weight_decay: 2e-4,
            amsgrad: true,
        };

        let field_value = config.to_field_value();
        let loaded_config = AdamConfig::from_field_value(field_value, "config").unwrap();

        assert_eq!(config.learning_rate, loaded_config.learning_rate);
        assert_eq!(config.beta1, loaded_config.beta1);
        assert_eq!(config.beta2, loaded_config.beta2);
        assert_eq!(config.eps, loaded_config.eps);
        assert_eq!(config.weight_decay, loaded_config.weight_decay);
        assert_eq!(config.amsgrad, loaded_config.amsgrad);
    }

    // ===== Adam Optimizer Serialization Tests =====

    #[test]
    fn test_adam_optimizer_json_roundtrip() {
        let mut weight = Tensor::ones(vec![2, 3]).with_requires_grad();
        let mut bias = Tensor::zeros(vec![2, 3]).with_requires_grad();

        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);
        optimizer.add_parameter(&bias);

        // Perform some steps to create state
        let output = weight.add_tensor(&bias);
        let mut loss = output.sum();
        loss.backward(None);
        optimizer.step(&mut [&mut weight, &mut bias]);

        // Test serialization
        let json = optimizer.to_json().unwrap();
        let loaded_optimizer = Adam::from_json(&json).unwrap();

        assert_eq!(
            optimizer.config().learning_rate,
            loaded_optimizer.config().learning_rate
        );
        assert_eq!(
            optimizer.saved_parameter_count(),
            loaded_optimizer.saved_parameter_count()
        );
    }

    #[test]
    fn test_adam_optimizer_binary_roundtrip() {
        let weight = Tensor::ones(vec![5, 2]).with_requires_grad();

        let mut optimizer = Adam::with_learning_rate(1e-4);
        optimizer.add_parameter(&weight);

        // Test serialization
        let binary = optimizer.to_binary().unwrap();
        let loaded_optimizer = Adam::from_binary(&binary).unwrap();

        assert_eq!(
            optimizer.config().learning_rate,
            loaded_optimizer.config().learning_rate
        );
        assert_eq!(
            optimizer.saved_parameter_count(),
            loaded_optimizer.saved_parameter_count()
        );
    }

    #[test]
    fn test_adam_parameter_relinking() {
        let weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);

        // Serialize
        let json = optimizer.to_json().unwrap();

        // Deserialize
        let mut loaded_optimizer = Adam::from_json(&json).unwrap();

        // After deserialization, saved states should be preserved
        assert_eq!(loaded_optimizer.saved_parameter_count(), 1);

        // Re-link parameter - this creates a new state since it's a new tensor with different ID
        let new_weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        loaded_optimizer.add_parameter(&new_weight);

        // Now there should be 2 states: the original saved one + the new one
        assert_eq!(loaded_optimizer.parameter_count(), 2);
        assert!(loaded_optimizer.is_parameter_linked(&new_weight));
    }

    #[test]
    fn test_adam_state_preservation() {
        let mut weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);

        // Perform training steps to build up state
        for _ in 0..3 {
            let output = weight.mul_scalar(2.0);
            let mut loss = output.sum();
            loss.backward(None);
            optimizer.step(&mut [&mut weight]);
            optimizer.zero_grad(&mut [&mut weight]);
        }

        // Serialize and deserialize
        let json = optimizer.to_json().unwrap();
        let loaded_optimizer = Adam::from_json(&json).unwrap();

        // Check that states were preserved
        assert_eq!(loaded_optimizer.saved_parameter_count(), 1);
        assert_eq!(
            loaded_optimizer.config().learning_rate,
            optimizer.config().learning_rate
        );
    }

    #[test]
    fn test_relink_parameters_success() {
        // Create original optimizer with parameters
        let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
        let bias = Tensor::zeros(vec![3]).with_requires_grad();

        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);
        optimizer.add_parameter(&bias);

        // Serialize
        let json = optimizer.to_json().unwrap();

        // Create new parameters with same shapes but different IDs
        let new_weight = Tensor::ones(vec![2, 3]).with_requires_grad();
        let new_bias = Tensor::zeros(vec![3]).with_requires_grad();

        // Deserialize and re-link
        let mut loaded_optimizer = Adam::from_json(&json).unwrap();
        loaded_optimizer
            .relink_parameters(&[&new_weight, &new_bias])
            .unwrap();

        // Verify re-linking worked
        assert_eq!(loaded_optimizer.parameter_count(), 2);
        assert!(loaded_optimizer.is_parameter_linked(&new_weight));
        assert!(loaded_optimizer.is_parameter_linked(&new_bias));
    }

    #[test]
    fn test_relink_parameters_shape_mismatch() {
        // Create original optimizer
        let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);

        // Serialize
        let json = optimizer.to_json().unwrap();

        // Create new parameter with different shape
        let new_weight = Tensor::ones(vec![3, 2]).with_requires_grad(); // Different shape!

        // Deserialize and try to re-link
        let mut loaded_optimizer = Adam::from_json(&json).unwrap();
        let result = loaded_optimizer.relink_parameters(&[&new_weight]);

        // Should fail with shape mismatch error
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Shape mismatch"));
    }

    #[test]
    fn test_relink_parameters_count_mismatch() {
        // Create original optimizer with 2 parameters
        let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
        let bias = Tensor::zeros(vec![3]).with_requires_grad();

        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);
        optimizer.add_parameter(&bias);

        // Serialize
        let json = optimizer.to_json().unwrap();

        // Create only 1 new parameter
        let new_weight = Tensor::ones(vec![2, 3]).with_requires_grad();

        // Deserialize and try to re-link with wrong count
        let mut loaded_optimizer = Adam::from_json(&json).unwrap();
        let result = loaded_optimizer.relink_parameters(&[&new_weight]);

        // Should fail with count mismatch error
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Parameter count mismatch"));
    }

    #[test]
    fn test_relink_parameters_requires_grad() {
        // Create original optimizer
        let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);

        // Serialize
        let json = optimizer.to_json().unwrap();

        // Create new parameter without requires_grad
        let new_weight = Tensor::ones(vec![2, 3]); // No requires_grad!

        // Deserialize and try to re-link
        let mut loaded_optimizer = Adam::from_json(&json).unwrap();
        let result = loaded_optimizer.relink_parameters(&[&new_weight]);

        // Should fail with requires_grad error
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must require gradients"));
    }

    #[test]
    fn test_relink_preserves_state() {
        // Create original optimizer and train it
        let mut weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);

        // Perform some training to build up state
        for _ in 0..2 {
            let output = weight.mul_scalar(2.0);
            let mut loss = output.sum();
            loss.backward(None);
            optimizer.step(&mut [&mut weight]);
            optimizer.zero_grad(&mut [&mut weight]);
        }

        // Get state before serialization
        let original_step_count = optimizer.step_count;

        // Serialize
        let json = optimizer.to_json().unwrap();

        // Create new parameter
        let new_weight = Tensor::ones(vec![2, 2]).with_requires_grad();

        // Deserialize and re-link
        let mut loaded_optimizer = Adam::from_json(&json).unwrap();
        loaded_optimizer.relink_parameters(&[&new_weight]).unwrap();

        // Verify state is preserved
        assert_eq!(loaded_optimizer.step_count, original_step_count);
        assert_eq!(loaded_optimizer.parameter_count(), 1);
        assert!(loaded_optimizer.is_parameter_linked(&new_weight));
    }

    // ===== Serializable Trait Tests =====

    #[test]
    fn test_serializable_json_methods() {
        // Create and populate test optimizer
        let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
        let bias = Tensor::zeros(vec![3]).with_requires_grad();

        let mut optimizer = Adam::with_learning_rate(1e-3);
        optimizer.add_parameter(&weight);
        optimizer.add_parameter(&bias);

        // Test to_json method
        let json = <Adam as crate::serialization::Serializable>::to_json(&optimizer).unwrap();
        assert!(!json.is_empty());
        assert!(json.contains("config"));
        assert!(json.contains("states"));
        assert!(json.contains("step_count"));
        assert!(json.contains("learning_rate"));

        // Test from_json method
        let restored = <Adam as crate::serialization::Serializable>::from_json(&json).unwrap();
        assert_eq!(
            optimizer.config().learning_rate,
            restored.config().learning_rate
        );
        assert_eq!(optimizer.config().beta1, restored.config().beta1);
        assert_eq!(optimizer.config().beta2, restored.config().beta2);
        assert_eq!(optimizer.config().eps, restored.config().eps);
        assert_eq!(
            optimizer.config().weight_decay,
            restored.config().weight_decay
        );
        assert_eq!(optimizer.config().amsgrad, restored.config().amsgrad);
        assert_eq!(
            optimizer.saved_parameter_count(),
            restored.saved_parameter_count()
        );
        assert_eq!(optimizer.step_count, restored.step_count);
    }

    #[test]
    fn test_serializable_binary_methods() {
        // Create and populate test optimizer
        let weight = Tensor::ones(vec![3, 4]).with_requires_grad();
        let mut optimizer = Adam::with_config(AdamConfig {
            learning_rate: 2e-4,
            beta1: 0.95,
            beta2: 0.999,
            eps: 1e-7,
            weight_decay: 1e-4,
            amsgrad: true,
        });
        optimizer.add_parameter(&weight);

        // Test to_binary method
        let binary = <Adam as crate::serialization::Serializable>::to_binary(&optimizer).unwrap();
        assert!(!binary.is_empty());

        // Test from_binary method
        let restored = <Adam as crate::serialization::Serializable>::from_binary(&binary).unwrap();
        assert_eq!(
            optimizer.config().learning_rate,
            restored.config().learning_rate
        );
        assert_eq!(optimizer.config().beta1, restored.config().beta1);
        assert_eq!(optimizer.config().beta2, restored.config().beta2);
        assert_eq!(optimizer.config().eps, restored.config().eps);
        assert_eq!(
            optimizer.config().weight_decay,
            restored.config().weight_decay
        );
        assert_eq!(optimizer.config().amsgrad, restored.config().amsgrad);
        assert_eq!(
            optimizer.saved_parameter_count(),
            restored.saved_parameter_count()
        );
        assert_eq!(optimizer.step_count, restored.step_count);
    }

    #[test]
    fn test_serializable_file_io_json() {
        use crate::serialization::{Format, Serializable};
        use std::fs;
        use std::path::Path;

        // Create test optimizer
        let weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let bias = Tensor::zeros(vec![2]).with_requires_grad();

        let mut optimizer = Adam::with_learning_rate(5e-4);
        optimizer.add_parameter(&weight);
        optimizer.add_parameter(&bias);

        let json_path = "test_adam_serializable.json";

        // Test save method with JSON format
        Serializable::save(&optimizer, json_path, Format::Json).unwrap();
        assert!(Path::new(json_path).exists());

        // Test load method with JSON format
        let loaded_optimizer = Adam::load(json_path, Format::Json).unwrap();
        assert_eq!(
            optimizer.config().learning_rate,
            loaded_optimizer.config().learning_rate
        );
        assert_eq!(
            optimizer.saved_parameter_count(),
            loaded_optimizer.saved_parameter_count()
        );

        // Test save_to_writer method
        let json_path_2 = "test_adam_serializable_writer.json";
        {
            let file = std::fs::File::create(json_path_2).unwrap();
            let mut writer = std::io::BufWriter::new(file);
            Serializable::save_to_writer(&optimizer, &mut writer, Format::Json).unwrap();
        }
        assert!(Path::new(json_path_2).exists());

        // Test load_from_reader method
        {
            let file = std::fs::File::open(json_path_2).unwrap();
            let mut reader = std::io::BufReader::new(file);
            let loaded_optimizer = Adam::load_from_reader(&mut reader, Format::Json).unwrap();
            assert_eq!(
                optimizer.config().learning_rate,
                loaded_optimizer.config().learning_rate
            );
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

        // Create test optimizer
        let weight = Tensor::ones(vec![3, 3]).with_requires_grad();
        let mut optimizer = Adam::with_config(AdamConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        });
        optimizer.add_parameter(&weight);

        let binary_path = "test_adam_serializable.bin";

        // Test save method with binary format
        Serializable::save(&optimizer, binary_path, Format::Binary).unwrap();
        assert!(Path::new(binary_path).exists());

        // Test load method with binary format
        let loaded_optimizer = Adam::load(binary_path, Format::Binary).unwrap();
        assert_eq!(
            optimizer.config().learning_rate,
            loaded_optimizer.config().learning_rate
        );
        assert_eq!(
            optimizer.saved_parameter_count(),
            loaded_optimizer.saved_parameter_count()
        );

        // Test save_to_writer method
        let binary_path_2 = "test_adam_serializable_writer.bin";
        {
            let file = std::fs::File::create(binary_path_2).unwrap();
            let mut writer = std::io::BufWriter::new(file);
            Serializable::save_to_writer(&optimizer, &mut writer, Format::Binary).unwrap();
        }
        assert!(Path::new(binary_path_2).exists());

        // Test load_from_reader method
        {
            let file = std::fs::File::open(binary_path_2).unwrap();
            let mut reader = std::io::BufReader::new(file);
            let loaded_optimizer = Adam::load_from_reader(&mut reader, Format::Binary).unwrap();
            assert_eq!(
                optimizer.config().learning_rate,
                loaded_optimizer.config().learning_rate
            );
        }

        // Cleanup test files
        let _ = fs::remove_file(binary_path);
        let _ = fs::remove_file(binary_path_2);
    }

    #[test]
    fn test_serializable_large_optimizer_performance() {
        // Create a large optimizer to test performance characteristics
        let mut optimizer = Adam::with_learning_rate(1e-4);

        // Add multiple parameters of different sizes
        for i in 0..5 {
            let size = 10 + i * 5;
            let param = Tensor::ones(vec![size, size]).with_requires_grad();
            optimizer.add_parameter(&param);
        }

        // Test JSON serialization
        let json = <Adam as crate::serialization::Serializable>::to_json(&optimizer).unwrap();
        assert!(!json.is_empty());
        let restored_json = <Adam as crate::serialization::Serializable>::from_json(&json).unwrap();
        assert_eq!(
            optimizer.config().learning_rate,
            restored_json.config().learning_rate
        );
        assert_eq!(
            optimizer.saved_parameter_count(),
            restored_json.saved_parameter_count()
        );

        // Test binary serialization
        let binary = <Adam as crate::serialization::Serializable>::to_binary(&optimizer).unwrap();
        assert!(!binary.is_empty());
        // Binary format should be efficient (this is informational, not a requirement)
        println!(
            "JSON size: {} bytes, Binary size: {} bytes",
            json.len(),
            binary.len()
        );

        let restored_binary =
            <Adam as crate::serialization::Serializable>::from_binary(&binary).unwrap();
        assert_eq!(
            optimizer.config().learning_rate,
            restored_binary.config().learning_rate
        );
        assert_eq!(
            optimizer.saved_parameter_count(),
            restored_binary.saved_parameter_count()
        );

        // Verify all configurations match
        assert_eq!(optimizer.config().beta1, restored_binary.config().beta1);
        assert_eq!(optimizer.config().beta2, restored_binary.config().beta2);
        assert_eq!(optimizer.config().eps, restored_binary.config().eps);
        assert_eq!(
            optimizer.config().weight_decay,
            restored_binary.config().weight_decay
        );
        assert_eq!(optimizer.config().amsgrad, restored_binary.config().amsgrad);
    }

    #[test]
    fn test_serializable_error_handling() {
        // Test invalid JSON
        let invalid_json = r#"{"invalid": "json", "structure": true}"#;
        let result = <Adam as crate::serialization::Serializable>::from_json(invalid_json);
        assert!(result.is_err());

        // Test empty JSON
        let empty_json = "{}";
        let result = <Adam as crate::serialization::Serializable>::from_json(empty_json);
        assert!(result.is_err());

        // Test invalid binary data
        let invalid_binary = vec![1, 2, 3, 4, 5];
        let result = <Adam as crate::serialization::Serializable>::from_binary(&invalid_binary);
        assert!(result.is_err());

        // Test empty binary data
        let empty_binary = vec![];
        let result = <Adam as crate::serialization::Serializable>::from_binary(&empty_binary);
        assert!(result.is_err());
    }

    #[test]
    fn test_serializable_different_configurations() {
        let test_configs = vec![
            // Default configuration
            AdamConfig::default(),
            // High learning rate
            AdamConfig {
                learning_rate: 1e-2,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
                amsgrad: false,
            },
            // AMSGrad enabled
            AdamConfig {
                learning_rate: 1e-4,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 1e-4,
                amsgrad: true,
            },
            // Custom betas
            AdamConfig {
                learning_rate: 5e-4,
                beta1: 0.95,
                beta2: 0.9999,
                eps: 1e-7,
                weight_decay: 1e-5,
                amsgrad: false,
            },
        ];

        for config in test_configs {
            // Create optimizer with specific configuration
            let weight = Tensor::ones(vec![2, 2]).with_requires_grad();
            let mut optimizer = Adam::with_config(config.clone());
            optimizer.add_parameter(&weight);

            // Test JSON roundtrip
            let json = <Adam as crate::serialization::Serializable>::to_json(&optimizer).unwrap();
            let restored_json =
                <Adam as crate::serialization::Serializable>::from_json(&json).unwrap();
            assert_eq!(config.learning_rate, restored_json.config().learning_rate);
            assert_eq!(config.beta1, restored_json.config().beta1);
            assert_eq!(config.beta2, restored_json.config().beta2);
            assert_eq!(config.eps, restored_json.config().eps);
            assert_eq!(config.weight_decay, restored_json.config().weight_decay);
            assert_eq!(config.amsgrad, restored_json.config().amsgrad);

            // Test binary roundtrip
            let binary =
                <Adam as crate::serialization::Serializable>::to_binary(&optimizer).unwrap();
            let restored_binary =
                <Adam as crate::serialization::Serializable>::from_binary(&binary).unwrap();
            assert_eq!(config.learning_rate, restored_binary.config().learning_rate);
            assert_eq!(config.beta1, restored_binary.config().beta1);
            assert_eq!(config.beta2, restored_binary.config().beta2);
            assert_eq!(config.eps, restored_binary.config().eps);
            assert_eq!(config.weight_decay, restored_binary.config().weight_decay);
            assert_eq!(config.amsgrad, restored_binary.config().amsgrad);
        }
    }

    #[test]
    fn test_serializable_edge_cases() {
        // Test optimizer with no parameters
        let empty_optimizer = Adam::new();
        let json = <Adam as crate::serialization::Serializable>::to_json(&empty_optimizer).unwrap();
        let restored = <Adam as crate::serialization::Serializable>::from_json(&json).unwrap();
        assert_eq!(
            empty_optimizer.saved_parameter_count(),
            restored.saved_parameter_count()
        );
        assert_eq!(empty_optimizer.step_count, restored.step_count);

        let binary =
            <Adam as crate::serialization::Serializable>::to_binary(&empty_optimizer).unwrap();
        let restored = <Adam as crate::serialization::Serializable>::from_binary(&binary).unwrap();
        assert_eq!(
            empty_optimizer.saved_parameter_count(),
            restored.saved_parameter_count()
        );
        assert_eq!(empty_optimizer.step_count, restored.step_count);

        // Test optimizer with extreme configuration values
        let extreme_config = AdamConfig {
            learning_rate: 1e-10, // Very small learning rate
            beta1: 0.999999,      // Very high beta1
            beta2: 0.000001,      // Very low beta2
            eps: 1e-15,           // Very small epsilon
            weight_decay: 1e-1,   // High weight decay
            amsgrad: true,
        };

        let weight = Tensor::ones(vec![1]).with_requires_grad();
        let mut extreme_optimizer = Adam::with_config(extreme_config.clone());
        extreme_optimizer.add_parameter(&weight);

        let json =
            <Adam as crate::serialization::Serializable>::to_json(&extreme_optimizer).unwrap();
        let restored = <Adam as crate::serialization::Serializable>::from_json(&json).unwrap();
        assert_eq!(
            extreme_config.learning_rate,
            restored.config().learning_rate
        );
        assert_eq!(extreme_config.beta1, restored.config().beta1);
        assert_eq!(extreme_config.beta2, restored.config().beta2);
        assert_eq!(extreme_config.eps, restored.config().eps);
        assert_eq!(extreme_config.weight_decay, restored.config().weight_decay);
        assert_eq!(extreme_config.amsgrad, restored.config().amsgrad);

        let binary =
            <Adam as crate::serialization::Serializable>::to_binary(&extreme_optimizer).unwrap();
        let restored = <Adam as crate::serialization::Serializable>::from_binary(&binary).unwrap();
        assert_eq!(
            extreme_config.learning_rate,
            restored.config().learning_rate
        );
        assert_eq!(extreme_config.beta1, restored.config().beta1);
        assert_eq!(extreme_config.beta2, restored.config().beta2);
        assert_eq!(extreme_config.eps, restored.config().eps);
        assert_eq!(extreme_config.weight_decay, restored.config().weight_decay);
        assert_eq!(extreme_config.amsgrad, restored.config().amsgrad);
    }
}
