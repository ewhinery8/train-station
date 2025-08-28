//! Tensor transformation validation tests
//!
//! This module contains validation tests for tensor transformation operations
//! against LibTorch reference implementation.

pub mod cat;
pub mod contiguous;
pub mod flatten;
pub mod permute;
pub mod reshape;
pub mod slice_view;
pub mod split;
pub mod squeeze;
pub mod stack;
pub mod transpose;
pub mod unsqueeze;
pub mod view;
