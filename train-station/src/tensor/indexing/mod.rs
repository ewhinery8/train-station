//! Tensor indexing operations
//!
//! This module provides comprehensive indexing and selection operations for tensors,
//! including gather/scatter style operations, masked operations, and element selection.
//! These operations enable efficient data extraction and manipulation based on indices
//! and conditions.
//!
//! # Key Features
//!
//! - **Index Selection**: Extract elements using index tensors (`index_select`)
//! - **Gather Operations**: Collect elements from specified indices (`gather`)
//! - **Masked Operations**: Apply operations based on boolean masks (`masked_fill`)
//! - **Element Selection**: Select specific elements or slices (`select`)
//! - **GradTrack Support**: Full gradient tracking for all indexing operations
//! - **Performance Optimized**: Efficient implementations with SIMD support
//!
//! # Performance Characteristics
//!
//! - **Index Selection**: O(n) where n is the number of selected elements
//! - **Gather Operations**: O(n) with optimized memory access patterns
//! - **Masked Operations**: O(n) with vectorized boolean operations
//! - **Memory Efficiency**: Minimal overhead for indexing operations
//! - **SIMD Optimization**: Vectorized operations where applicable
//!
//! # Organization
//!
//! The indexing module is organized into specialized submodules:
//!
//! - **`gather`**: Gather operations for collecting elements from specified indices
//! - **`index_select`**: Index selection operations for extracting elements by index
//! - **`masked_fill`**: Masked operations for conditional element manipulation
//! - **`select`**: Element selection and slicing operations
//!
//! # Examples
//!
//! ## Index Selection
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
//! let indices = [0, 2, 4];
//! let selected = tensor.index_select(0, &indices);
//! assert_eq!(selected.data(), &[1.0, 3.0, 5.0]);
//! ```
//!
//! ## Masked Operations
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
//! let mask = [true, false, true, false];
//! let result = tensor.masked_fill(&mask, 0.0);
//! assert_eq!(result.data(), &[0.0, 2.0, 0.0, 4.0]);
//! ```
//!
//! ## Element Selection
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let element = tensor.select(0, 1);
//! assert_eq!(element.data(), &[3.0, 4.0]);
//! ```
//!
//! # Design Principles
//!
//! - **NumPy Compatibility**: Operations follow NumPy conventions where applicable
//! - **Performance First**: Optimized implementations for common use cases
//! - **Memory Safety**: Safe operations with proper bounds checking
//! - **GradTrack Integration**: Seamless integration with the gradtrack system
//! - **Type Safety**: Strong typing for indices and masks

pub mod gather;
pub mod index_select;
pub mod masked_fill;
pub mod select;
