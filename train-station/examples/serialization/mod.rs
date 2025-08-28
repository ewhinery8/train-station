//! Serialization Examples for Train Station
//!
//! This module contains comprehensive examples demonstrating Train Station's serialization
//! system for persisting and exchanging data:
//! - Basic struct serialization patterns and best practices
//! - Custom data structures with Train Station's StructSerializable trait
//! - JSON and binary format comparison and use cases
//! - Complex nested structures and collections handling
//! - Error handling and validation patterns
//! - Real-world serialization workflows and checkpointing
//!
//! These examples are designed to be self-contained and executable, providing
//! hands-on learning for users building persistent applications with Train Station.
//!
//! # Learning Objectives
//!
//! - Understand Train Station's zero-dependency serialization system
//! - Learn to implement StructSerializable for custom types
//! - Master JSON vs binary format selection for different use cases
//! - Explore nested structures and collection serialization patterns
//! - Implement robust error handling and data validation
//! - Build complete data persistence workflows
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of file I/O operations
//! - Familiarity with structured data concepts
//! - Basic Train Station library knowledge (see getting_started examples)
//!
//! # Usage
//!
//! ```bash
//! cargo run --example basic_structs
//! cargo run --example nested_structures
//! cargo run --example json_vs_binary
//! cargo run --example real_world_workflows
//! cargo run --example error_handling
//! ```
//!
//! # Architecture Overview
//!
//! The examples demonstrate how to use Train Station's serialization system:
//!
//! - **Basic Structs**: Simple data structures with primitive fields
//! - **Nested Structures**: Complex hierarchical data with multiple levels
//! - **Format Comparison**: JSON vs binary trade-offs and selection criteria
//! - **Real-World Workflows**: Complete application patterns and best practices
//! - **Error Handling**: Robust validation and recovery strategies
//!
//! # Key Concepts Demonstrated
//!
//! - **StructSerializable Trait**: Primary interface for custom struct serialization
//! - **Format Selection**: When to use JSON vs binary serialization
//! - **Field Management**: Adding, removing, and validating struct fields
//! - **Collections Handling**: Vectors, hashmaps, and optional fields
//! - **Error Recovery**: Graceful handling of serialization failures
//! - **Performance Optimization**: Memory-efficient serialization patterns

pub mod basic_structs;
pub mod error_handling;
pub mod json_vs_binary;
pub mod nested_structures;
pub mod real_world_workflows;

pub use basic_structs::*;
pub use error_handling::*;
pub use json_vs_binary::*;
pub use nested_structures::*;
pub use real_world_workflows::*;
