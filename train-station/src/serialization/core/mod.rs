//! Core serialization functionality for structured data handling
//!
//! This module provides the fundamental types, traits, and functionality that form the
//! foundation for all serialization formats in the Train Station library. It implements
//! a type-safe, extensible serialization framework that supports both JSON and binary
//! formats with comprehensive error handling and validation.
//!
//! # Purpose
//!
//! The core serialization module serves as the backbone for persistent storage and data
//! exchange throughout the Train Station library. It enables:
//!
//! - **Type-safe serialization**: Compile-time guarantees for data structure preservation
//! - **Format flexibility**: Unified API supporting multiple serialization formats
//! - **Error handling**: Comprehensive validation and detailed error reporting
//! - **Extensibility**: Easy addition of new serializable types and formats
//! - **Performance**: Optimized for both human-readable and binary formats
//!
//! # Organization
//!
//! The module is organized into focused submodules that handle specific aspects of
//! serialization:
//!
//! - **`deserializer`**: Data extraction and reconstruction from serialized formats
//! - **`error`**: Comprehensive error types and handling for serialization failures
//! - **`impls`**: Standard library trait implementations for common Rust types
//! - **`serializer`**: Data structure serialization and format generation
//! - **`traits`**: Core serialization traits defining the framework interface
//! - **`types`**: Fundamental data types and value representations
//!
//! # Core Components
//!
//! ## StructSerializer
//!
//! Builder-pattern struct for constructing serializable data structures. Provides
//! fluent API for adding fields and generating multiple output formats.
//!
//! ## StructDeserializer
//!
//! Data extraction utility for reconstructing objects from serialized formats.
//! Supports field-by-field extraction with validation and error handling.
//!
//! ## FieldValue
//!
//! Universal value representation that can hold any serializable data type.
//! Provides type-safe conversion between concrete types and serialized formats.
//!
//! ## Serialization Traits
//!
//! - **`ToFieldValue`**: Convert concrete types to `FieldValue` representation
//! - **`FromFieldValue`**: Convert `FieldValue` back to concrete types
//! - **`StructSerializable`**: Define serialization behavior for custom structs
//!
//! # Usage Patterns
//!
//! The core serialization framework follows a consistent pattern:
//!
//! 1. **Define serializable types** by implementing `StructSerializable`
//! 2. **Serialize data** using `StructSerializer` with field-by-field construction
//! 3. **Deserialize data** using `StructDeserializer` with field-by-field extraction
//! 4. **Handle errors** through the comprehensive `SerializationError` type system
//!
//! # Thread Safety
//!
//! All core serialization components are thread-safe and can be used concurrently
//! across multiple threads. No shared state is maintained between serialization operations.
//!
//! # Performance Characteristics
//!
//! - **Zero-copy operations**: Where possible, data is referenced rather than copied
//! - **Efficient memory usage**: Optimized storage formats for both JSON and binary
//! - **Minimal allocations**: Reuse of buffers and efficient memory management
//! - **Fast validation**: Type checking and validation with minimal overhead

pub mod deserializer;
pub mod error;
pub mod impls;
pub mod serializer;
#[cfg(test)]
pub mod tests;
pub mod traits;
pub mod types;

// Re-export commonly used types and traits for convenient access
pub use deserializer::StructDeserializer;
pub use error::{SerializationError, SerializationResult};
pub use serializer::StructSerializer;
pub use traits::{FromFieldValue, StructSerializable, ToFieldValue};
pub use types::FieldValue;
