//! LibTorch FFI validation for optimizers
//!
//! This module provides validation against LibTorch optimizers to ensure
//! mathematical equivalence and correctness of Train Station's optimization algorithms.

pub mod adam;

pub use adam::AdamValidator;
