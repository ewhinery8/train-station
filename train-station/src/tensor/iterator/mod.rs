//! Iterator module for tensor iteration
//!
//! This module provides high-performance iterators over tensor elements, where each
//! element is represented as a view tensor of shape `[1]`. This design allows for
//! seamless integration with Rust's standard library iterator methods while
//! leveraging the existing tensor operation framework and gradient tracking.
//!
//! Implicit performance routing: iterator constructors decide at creation time whether
//! to use a no-grad fast path (borrowed contiguous traversal or one-time materialization)
//! or a grad-preserving view path based on `requires_grad` and `gradtrack::is_grad_enabled()`.
//!
//! # Key Features
//!
//! - **Standard Library Compatibility**: Full implementation of Iterator, ExactSizeIterator,
//!   DoubleEndedIterator, FusedIterator, IntoIterator, and FromIterator traits
//! - **Gradient Tracking**: Automatic gradient propagation through element operations
//! - **Performance Optimized**: True zero-copy views with shared memory
//! - **SIMD Compatible**: All operations use existing optimized tensor implementations
//! - **Memory Efficient**: Adaptive view creation based on tensor size
//! - **Zero-Copy Operations**: Element views share memory with source tensor
//! - **Full Tensor Operations**: Each element supports all tensor methods
//!
//! # Performance Characteristics
//!
//! - **View Creation**: O(1) per element with true zero-copy views
//! - **Memory Overhead**: ~64 bytes per view tensor (no data copying)
//! - **SIMD Operations**: Full utilization of existing optimizations
//! - **Gradient Tracking**: True gradient flow with element-level accumulation
//! - **Iterator Overhead**: Minimal performance impact for element access
//! - **Collection Optimization**: Efficient reconstruction from element views
//!
//! # Examples
//!
//! ## Basic Element Iteration (1D tensors)
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
//!
//! // Basic iteration over elements
//! // For 1D tensors, `iter()` yields scalar views
//! for element in tensor.iter() {
//!     println!("Element value: {}", element.value());
//! }
//!
//! // Collect elements into a new tensor
//! let collected: Tensor = tensor.iter().collect();
//! assert_eq!(collected.data(), tensor.data());
//! ```
//!
//! ## Element-Wise Transformations (1D convenience)
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
//!
//! // Apply tensor operations to each element
//! let doubled: Tensor = tensor.iter()
//!     .map(|elem| elem.mul_scalar(2.0))
//!     .collect();
//!
//! assert_eq!(doubled.data(), &[2.0, 4.0, 6.0]);
//!
//! // Chain multiple operations
//! let transformed: Tensor = tensor.iter()
//!     .map(|elem| elem.mul_scalar(2.0).add_scalar(1.0)) // 2x + 1
//!     .collect();
//!
//! assert_eq!(transformed.data(), &[3.0, 5.0, 7.0]);
//! ```
//!
//! ## Advanced Iterator Operations (1D convenience)
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
//!
//! // Filter elements based on values
//! let large_values: Tensor = tensor.iter()
//!     .filter(|elem| elem.value() > 3.0)
//!     .collect();
//!
//! assert_eq!(large_values.data(), &[4.0, 5.0]);
//!
//! // Use enumerate for indexed operations
//! let indexed: Tensor = tensor.iter()
//!     .enumerate()
//!     .map(|(i, elem)| elem.add_scalar(i as f32))
//!     .collect();
//!
//! assert_eq!(indexed.data(), &[1.0, 3.0, 5.0, 7.0, 9.0]);
//! ```
//!
//! ## Range Iteration (1D convenience)
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
//!
//! // Iterate over a specific range
//! let middle: Tensor = tensor.iter_range(1, 4)
//!     .map(|elem| elem.mul_scalar(2.0))
//!     .collect();
//!
//! assert_eq!(middle.data(), &[4.0, 6.0, 8.0]);
//! ```
//!
//! ## Double-Ended Iteration (1D convenience)
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
//!
//! // Reverse iteration
//! let reversed: Tensor = tensor.iter().rev().collect();
//! assert_eq!(reversed.data(), &[4.0, 3.0, 2.0, 1.0]);
//!
//! // Iterate from both ends
//! let mut iter = tensor.iter();
//! assert_eq!(iter.next().unwrap().value(), 1.0);
//! assert_eq!(iter.next_back().unwrap().value(), 4.0);
//! ```
//!
//! ## Gradient Tracking
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0], vec![2])
//!     .unwrap()
//!     .with_requires_grad();
//!
//! // Element operations maintain gradient tracking
//! let result: Tensor = tensor.iter()
//!     .map(|elem| elem.mul_scalar(2.0))
//!     .collect();
//!
//! assert!(result.requires_grad());
//! assert_eq!(result.data(), &[2.0, 4.0]);
//! ```
//!
//! # Design Principles and API Overview
//!
//! - Use `iter()` or `outer_iter()` to iterate outermost-dimension sub-tensors (Vec-like semantics)
//! - Use `iter_dim(dim)` to iterate sub-tensors along an arbitrary dimension
//! - Use `iter_flat()` to iterate scalar views in row-major order
//! - Use `chunks()` / `chunks_exact()` for linear chunk views
//! - Use `windows()` / `windows_step()` for overlapping linear window views
//!
//! Deprecated aliases (will be removed pre-1.0): `iter_chunks`, `iter_chunks_exact`,
//! `iter_windows`, `iter_windows_step`.
//!
//! - **Zero-Copy Views**: Element views share memory with source tensor
//! - **Full Tensor Operations**: Each element supports all tensor methods
//! - **Standard Library Integration**: Complete compatibility with Rust iterators
//! - **Performance First**: Optimized for high-performance element access
//! - **Gradient Preservation**: Maintains gradtrack functionality through operations
//! - **Memory Efficiency**: Minimal overhead for element iteration
//! - **Type Safety**: Compile-time guarantees for iterator operations

pub mod chunks;
pub mod collect;
pub mod element;
pub mod viewdim;
pub mod windows;

use crate::gradtrack::is_grad_enabled;
use crate::tensor::core::Tensor;
pub use collect::{TensorCollectExt, ValuesCollectExt};
use std::iter::FromIterator;

/// High-performance iterator over tensor elements as view tensors
///
/// Each element becomes a proper `Tensor` view of shape `[1]` that can use
/// all existing tensor operations and gradient tracking. Implements all
/// standard iterator traits for maximum compatibility with Rust's ecosystem.
///
/// This iterator provides zero-copy access to tensor elements through view
/// tensors, enabling efficient element-wise operations while maintaining
/// full compatibility with Rust's standard library iterator methods.
///
/// # Performance
///
/// - **Zero-Copy Views**: Each element is a view tensor sharing memory with source
/// - **O(1) Element Access**: Constant-time view creation for each element
/// - **Memory Efficient**: ~64 bytes overhead per element view
/// - **SIMD Compatible**: All tensor operations use existing optimizations
/// - **Gradient Tracking**: Full gradtrack support through element operations
///
/// # Implementation Details
///
/// The iterator creates lightweight view tensors on-demand, sharing the same
/// memory allocation as the source tensor. This ensures zero-copy semantics
/// while maintaining full tensor operation compatibility.
///
/// Each element view is created using `Tensor::element_view()`, which provides
/// a true view of the underlying data without any copying. The view tensors
/// support all standard tensor operations including gradient tracking.
///
/// # Standard Library Compatibility
///
/// This iterator implements all standard iterator traits:
/// - `Iterator`: Basic iteration with `next()` and `size_hint()`
/// - `ExactSizeIterator`: Precise size information with `len()`
/// - `DoubleEndedIterator`: Reverse iteration with `next_back()`
/// - `FusedIterator`: Fused iteration for better performance
/// - `IntoIterator`: Automatic conversion for `for` loops
///
/// # Examples
///
/// ## Basic Iteration
///
/// ```
/// use train_station::Tensor;
///
/// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
///
/// // Basic iteration
/// for element in tensor.iter() {
///     println!("Element value: {}", element.value());
/// }
///
/// // Standard library methods
/// let sum: f32 = tensor.iter()
///     .map(|elem| elem.value())
///     .sum();
///
/// assert_eq!(sum, 6.0);
/// ```
///
/// ## Element Operations
///
/// ```
/// use train_station::Tensor;
///
/// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
///
/// // Tensor operations on elements
/// let transformed: Tensor = tensor.iter()
///     .map(|elem| elem.mul_scalar(2.0).add_scalar(1.0)) // 2x + 1
///     .collect();
///
/// assert_eq!(transformed.data(), &[3.0, 5.0, 7.0]);
/// ```
///
/// ## Advanced Iterator Methods
///
/// ```
/// use train_station::Tensor;
///
/// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
///
/// // Filter and transform
/// let result: Tensor = tensor.iter()
///     .filter(|elem| elem.value() > 2.0)
///     .map(|elem| elem.mul_scalar(10.0))
///     .collect();
///
/// assert_eq!(result.data(), &[30.0, 40.0, 50.0]);
///
/// // Reverse iteration
/// let reversed: Tensor = tensor.iter().rev().collect();
/// assert_eq!(reversed.data(), &[5.0, 4.0, 3.0, 2.0, 1.0]);
/// ```
// Re-export iterator types from submodules for public API
// ===== IntoIterator Implementation =====
/// IntoIterator for &Tensor now iterates outermost dimension, yielding sub-tensors (views)
impl<'a> IntoIterator for &'a Tensor {
    type Item = Tensor;
    type IntoIter = crate::tensor::iterator::viewdim::TensorDimIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        // Iterate outermost dim by default
        self.iter_dim(0)
    }
}

/// IntoIterator for owned Tensor: iterate outermost dimension producing sub-tensors.
/// Enables `.into_iter().flatten()` patterns on owned tensors.
impl IntoIterator for Tensor {
    type Item = Tensor;
    type IntoIter = crate::tensor::iterator::viewdim::TensorDimOwnedIterator;

    fn into_iter(self) -> Self::IntoIter {
        crate::tensor::iterator::viewdim::TensorDimOwnedIterator::new(self, 0)
    }
}

// ===== FromIterator Implementation =====

impl FromIterator<Tensor> for Tensor {
    /// Collect element view tensors back into a single tensor
    ///
    /// This method reconstructs a tensor from an iterator of element view tensors.
    /// It includes optimizations for common patterns and maintains gradient tracking
    /// when appropriate.
    ///
    /// The collection process automatically detects whether all elements are scalar
    /// views (shape `[1]`) and uses optimized collection strategies accordingly.
    /// Gradient tracking is preserved when any input element requires gradients.
    ///
    /// # Performance
    ///
    /// - **Optimized Collection**: Specialized paths for scalar and mixed views
    /// - **Memory Efficient**: Direct memory copying without intermediate allocations
    /// - **Gradient Preservation**: Maintains gradtrack functionality when enabled
    /// - **Shape Detection**: Automatic detection of element shapes for optimization
    ///
    /// # Implementation Details
    ///
    /// The method performs the following steps:
    /// 1. **Element Collection**: Gathers all element tensors from the iterator
    /// 2. **Shape Analysis**: Determines if all elements are scalar views
    /// 3. **Optimized Path**: Uses specialized collection for scalar views
    /// 4. **General Path**: Handles mixed shapes by flattening into 1D tensor
    /// 5. **Gradient Setup**: Preserves gradient tracking when appropriate
    ///
    /// # Examples
    ///
    /// ## Basic Collection
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let original = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let doubled: Tensor = original.iter()
    ///     .map(|elem| elem.mul_scalar(2.0))
    ///     .collect();
    ///
    /// assert_eq!(doubled.data(), &[2.0, 4.0, 6.0]);
    /// ```
    ///
    /// ## Collection with Gradient Tracking
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let original = Tensor::from_slice(&[1.0, 2.0], vec![2])
    ///     .unwrap()
    ///     .with_requires_grad();
    ///
    /// let result: Tensor = original.iter()
    ///     .map(|elem| elem.mul_scalar(2.0))
    ///     .collect();
    ///
    /// assert!(result.requires_grad());
    /// assert_eq!(result.data(), &[2.0, 4.0]);
    /// ```
    ///
    /// ## Empty Iterator Handling
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let empty: Tensor = Vec::<Tensor>::new().into_iter().collect();
    /// assert_eq!(empty.size(), 0);
    /// assert_eq!(empty.shape().dims(), vec![0]);
    /// ```
    fn from_iter<I: IntoIterator<Item = Tensor>>(iter: I) -> Self {
        let elements: Vec<Tensor> = iter.into_iter().collect();

        if elements.is_empty() {
            return Tensor::new(vec![0]);
        }

        // Check if all elements are scalars (size == 1). Supports both [1] and 0-D [] shapes
        let all_scalars = elements.iter().all(|e| e.size() == 1);

        if all_scalars {
            // Optimized path for scalar element views
            Self::collect_scalar_views(elements)
        } else {
            // General path for mixed shapes
            Self::collect_mixed_views(elements)
        }
    }
}

impl Tensor {
    /// Optimized collection for scalar element views
    ///
    /// This method efficiently reconstructs a tensor from scalar element views,
    /// preserving gradient tracking and using optimized memory operations.
    ///
    /// This is the fast path for collection when all elements are scalar views
    /// (shape `[1]`). It performs direct memory copying and sets up gradient
    /// tracking when any input element requires gradients.
    ///
    /// # Arguments
    ///
    /// * `elements` - Vector of scalar element view tensors
    ///
    /// # Returns
    ///
    /// A new tensor containing all element values in a 1D layout
    ///
    /// # Performance
    ///
    /// - **Direct Memory Copy**: Single-pass copying without intermediate allocations
    /// - **Gradient Optimization**: Efficient gradient tracking setup
    /// - **Memory Efficient**: Minimal overhead for collection process
    /// - **SIMD Compatible**: Result tensor supports all optimizations
    ///
    /// # Implementation Details
    ///
    /// The method performs the following steps:
    /// 1. **Allocation**: Creates uninitialized tensor with correct size
    /// 2. **Gradient Check**: Determines if any element requires gradients
    /// 3. **Memory Copy**: Direct copying from element views to result
    /// 4. **Gradient Setup**: Configures gradient tracking when needed
    /// 5. **Operation Registration**: Registers with gradtrack engine
    fn collect_scalar_views(elements: Vec<Tensor>) -> Self {
        if elements.is_empty() {
            return Tensor::new(vec![0]);
        }
        // Fast path: if no element requires grad or gradients are disabled, copy directly
        let any_requires = elements.iter().any(|t| t.requires_grad());
        if !any_requires || !is_grad_enabled() {
            let n = elements.len();
            let mut out = Tensor::new_uninitialized(vec![n]);
            unsafe {
                let dst = out.as_mut_ptr();
                for (i, t) in elements.iter().enumerate() {
                    debug_assert_eq!(t.size(), 1);
                    std::ptr::copy_nonoverlapping(t.as_ptr(), dst.add(i), 1);
                }
            }
            return out;
        }

        // Grad-preserving path: concat along dim 0 then flatten
        let mut prepped: Vec<Tensor> = Vec::with_capacity(elements.len());
        for t in elements.into_iter() {
            if t.shape().rank() == 0 {
                prepped.push(t.unsqueeze(0)); // [] -> [1]
            } else {
                prepped.push(t);
            }
        }
        let concatenated = Tensor::cat(&prepped, 0); // shape: [N, 1]
        concatenated.flatten() // shape: [N]
    }

    /// General collection for mixed element shapes
    ///
    /// This method handles collection when elements have different shapes,
    /// flattening all elements into a 1D tensor.
    ///
    /// This is the general path for collection when elements have varying shapes.
    /// It flattens all elements into a single 1D tensor and preserves gradient
    /// tracking when any input element requires gradients.
    ///
    /// # Arguments
    ///
    /// * `elements` - Vector of element tensors with potentially different shapes
    ///
    /// # Returns
    ///
    /// A new 1D tensor containing all flattened element values
    ///
    /// # Performance
    ///
    /// - **Flattening**: Converts all elements to 1D layout
    /// - **Memory Copy**: Efficient copying with size calculation
    /// - **Gradient Preservation**: Maintains gradtrack functionality
    /// - **Mixed Shapes**: Handles elements with different dimensions
    ///
    /// # Implementation Details
    ///
    /// The method performs the following steps:
    /// 1. **Size Calculation**: Sums sizes of all elements for total size
    /// 2. **Allocation**: Creates uninitialized tensor with total size
    /// 3. **Sequential Copy**: Copies each element's data sequentially
    /// 4. **Gradient Setup**: Configures gradient tracking when needed
    /// 5. **Operation Registration**: Registers with gradtrack engine
    fn collect_mixed_views(elements: Vec<Tensor>) -> Self {
        let requires_grad = elements.iter().any(|e| e.requires_grad());
        // Concatenate then flatten to preserve gradient connections
        let concatenated = Tensor::cat(&elements, 0);
        let flattened = concatenated.flatten();
        if requires_grad && is_grad_enabled() {
            // Flags are handled by ops; return as-is
        }
        flattened
    }

    // Iterator entry points are implemented in iterator/element.rs
}

// Redundant iterator type and collection trait/impls have been moved to dedicated files.

#[cfg(test)]
mod tests {
    //! Comprehensive tests for tensor element iterator functionality
    //!
    //! These tests cover all aspects of the iterator implementation:
    //! - Basic iteration functionality
    //! - Standard library trait compliance
    //! - Gradient tracking through element operations
    //! - Performance characteristics
    //! - Edge cases and error conditions

    use super::*;

    /// Test basic iterator functionality
    #[test]
    fn test_basic_iteration() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

        let elements: Vec<Tensor> = tensor.iter_elements().collect();
        assert_eq!(elements.len(), 4);

        // Check that each element is a scalar tensor with correct value
        for (i, elem) in elements.iter().enumerate() {
            assert_eq!(elem.shape().dims(), vec![1]);
            assert_eq!(elem.size(), 1);
            assert_eq!(elem.value(), (i + 1) as f32);
        }
    }

    /// Test Iterator trait methods
    #[test]
    fn test_iterator_trait_methods() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let mut iter = tensor.iter();

        // Test next()
        let _first = iter.next().unwrap();
        assert_eq!(_first.value(), 1.0);

        // Test size_hint() after consuming one element
        assert_eq!(iter.size_hint(), (4, Some(4)));

        // Test count()
        assert_eq!(iter.count(), 4);

        // Test nth()
        let mut iter = tensor.iter();
        let third = iter.nth(2).unwrap();
        assert_eq!(third.value(), 3.0);

        // Test last()
        let mut iter = tensor.iter();
        let last = iter.next_back().unwrap();
        assert_eq!(last.value(), 5.0);
    }

    /// Test ExactSizeIterator
    #[test]
    fn test_exact_size_iterator() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let iter = tensor.iter();

        assert_eq!(iter.len(), 3);

        // Test that len() decreases as we consume the iterator
        let mut iter = tensor.iter();
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.len(), 2);
        iter.next();
        assert_eq!(iter.len(), 1);
        iter.next();
        assert_eq!(iter.len(), 0);
    }

    /// Test DoubleEndedIterator
    #[test]
    fn test_double_ended_iterator() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut iter = tensor.iter();

        // Test next_back()
        let last = iter.next_back().unwrap();
        assert_eq!(last.value(), 4.0);

        let first = iter.next().unwrap();
        assert_eq!(first.value(), 1.0);

        // Test nth_back()
        let mut iter = tensor.iter();
        let second_to_last = iter.nth_back(1).unwrap();
        assert_eq!(second_to_last.value(), 3.0);

        // Test consuming from both ends
        let mut iter = tensor.iter();
        assert_eq!(iter.next().unwrap().value(), 1.0);
        assert_eq!(iter.next_back().unwrap().value(), 4.0);
        assert_eq!(iter.next().unwrap().value(), 2.0);
        assert_eq!(iter.next_back().unwrap().value(), 3.0);
        assert!(iter.next().is_none());
    }

    /// Test IntoIterator trait
    #[test]
    fn test_into_iterator() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();

        // Test with for loop
        let mut values = Vec::new();
        for element in &tensor {
            values.push(element.value());
        }
        assert_eq!(values, vec![1.0, 2.0, 3.0]);

        // Test with into_iter() explicitly
        let values: Vec<f32> = (&tensor).into_iter().map(|elem| elem.value()).collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    /// Test FromIterator trait (collect)
    #[test]
    fn test_from_iterator() {
        let original = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

        // Test collecting back to tensor
        let collected: Tensor = original.iter().collect();
        assert_eq!(collected.shape().dims(), vec![4]);
        assert_eq!(collected.data(), original.data());

        // Test collecting with transformations
        let doubled: Tensor = original
            .iter()
            .map(|elem| {
                let val = elem.value();
                Tensor::from_slice(&[val * 2.0], vec![1]).unwrap()
            })
            .collect();

        assert_eq!(doubled.data(), &[2.0, 4.0, 6.0, 8.0]);
    }

    /// Test standard library iterator methods
    #[test]
    fn test_std_iterator_methods() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();

        // Test map
        let doubled: Vec<f32> = tensor.iter().map(|elem| elem.value() * 2.0).collect();
        assert_eq!(doubled, vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        // Test filter
        let large_values: Vec<f32> = tensor
            .iter()
            .filter(|elem| elem.value() > 3.0)
            .map(|elem| elem.value())
            .collect();
        assert_eq!(large_values, vec![4.0, 5.0]);

        // Test enumerate
        let with_indices: Vec<(usize, f32)> = tensor
            .iter()
            .enumerate()
            .map(|(i, elem)| (i, elem.value()))
            .collect();
        assert_eq!(
            with_indices,
            vec![(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0), (4, 5.0)]
        );

        // Test fold
        let sum: f32 = tensor.iter().fold(0.0, |acc, elem| acc + elem.value());
        assert_eq!(sum, 15.0);

        // Test find
        let found = tensor.iter().find(|elem| elem.value() == 3.0);
        assert!(found.is_some());
        assert_eq!(found.unwrap().value(), 3.0);

        // Test any/all
        let all_positive = tensor.iter().all(|elem| elem.value() > 0.0);
        assert!(all_positive);

        let any_large = tensor.iter().any(|elem| elem.value() > 4.0);
        assert!(any_large);
    }

    /// Test element operations with tensor methods
    #[test]
    fn test_element_tensor_operations() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();

        // Test scalar operations on elements
        let scaled: Tensor = tensor.iter().map(|elem| elem.mul_scalar(2.0)).collect();
        assert_eq!(scaled.data(), &[2.0, 4.0, 6.0]);

        let offset: Tensor = tensor.iter().map(|elem| elem.add_scalar(10.0)).collect();
        assert_eq!(offset.data(), &[11.0, 12.0, 13.0]);

        // Test chaining operations
        let complex: Tensor = tensor
            .iter()
            .map(|elem| elem.mul_scalar(2.0).add_scalar(1.0)) // 2x + 1
            .collect();
        assert_eq!(complex.data(), &[3.0, 5.0, 7.0]);
    }

    /// Test gradient tracking through element operations
    #[test]
    fn test_gradient_tracking() {
        let tensor = Tensor::from_slice(&[1.0, 2.0], vec![2])
            .unwrap()
            .with_requires_grad();

        // Perform element-wise operations
        let result: Tensor = tensor.iter().map(|elem| elem.mul_scalar(2.0)).collect();

        // The result should require gradients if any element requires gradients
        // Note: Current implementation creates copies, so gradient tracking is
        // implemented but may not propagate back to original tensor
        assert!(result.requires_grad());

        // For now, just verify the forward pass works with gradient-enabled tensors
        // Full gradient propagation would require true view implementation
        assert_eq!(result.data(), &[2.0, 4.0]);
    }

    /// Test with zero-sized tensors
    #[test]
    fn test_zero_sized_tensor() {
        let empty = Tensor::new(vec![0]);
        let iter = empty.iter();

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));

        let collected: Tensor = iter.collect();
        assert_eq!(collected.size(), 0);
    }

    /// Test range iteration
    #[test]
    fn test_range_iteration() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();

        // Test middle range
        let middle: Vec<f32> = tensor.iter_range(1, 4).map(|elem| elem.value()).collect();
        assert_eq!(middle, vec![2.0, 3.0, 4.0]);

        // Test out of bounds (should be clamped)
        let clamped: Vec<f32> = tensor.iter_range(3, 10).map(|elem| elem.value()).collect();
        assert_eq!(clamped, vec![4.0, 5.0]);

        // Test empty range
        let empty: Vec<f32> = tensor.iter_range(2, 2).map(|elem| elem.value()).collect();
        assert_eq!(empty, Vec::<f32>::new());
    }

    /// Test complex iterator chains
    #[test]
    fn test_complex_chains() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();

        // Complex chain: enumerate -> filter -> map -> collect
        let result: Tensor = tensor
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0) // Take even indices
            .map(|(i, elem)| elem.add_scalar(i as f32)) // Add index to value
            .collect();

        // Should have elements [1.0 + 0, 3.0 + 2, 5.0 + 4] = [1.0, 5.0, 9.0]
        assert_eq!(result.data(), &[1.0, 5.0, 9.0]);

        // Test with rev()
        let reversed: Tensor = tensor.iter().rev().take(3).collect();

        assert_eq!(reversed.data(), &[6.0, 5.0, 4.0]);
    }

    /// Performance test for iterator overhead
    #[test]
    fn test_performance() {
        let large_tensor =
            Tensor::from_slice(&(0..1000).map(|i| i as f32).collect::<Vec<_>>(), vec![1000])
                .unwrap();

        let start = std::time::Instant::now();

        let result: Tensor = large_tensor
            .iter()
            .map(|elem| elem.mul_scalar(2.0))
            .collect();

        let duration = start.elapsed();
        println!("Iterator performance test took: {:?}", duration);

        // Verify correctness
        assert_eq!(result.size(), 1000);
        assert_eq!(result.data()[0], 0.0);
        assert_eq!(result.data()[999], 1998.0);
    }

    /// Test chunks iterator basic behavior
    #[test]
    fn test_chunks_basic() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let chunks: Vec<Tensor> = t.chunks(2).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].data(), &[1.0, 2.0]);
        assert_eq!(chunks[1].data(), &[3.0, 4.0]);
        assert_eq!(chunks[2].data(), &[5.0]);
    }

    /// Test chunks_exact with remainder
    #[test]
    fn test_chunks_exact_with_remainder() {
        let t = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0], vec![5]).unwrap();
        let mut it = t.chunks_exact(2);
        let v0 = it.next().unwrap();
        let v1 = it.next().unwrap();
        assert!(it.next().is_none());
        assert_eq!(v0.data(), &[10.0, 20.0]);
        assert_eq!(v1.data(), &[30.0, 40.0]);
        let r = it.remainder();
        assert_eq!(r.data(), &[50.0]);
    }

    /// Test windows iterator with step 1
    #[test]
    fn test_windows_basic() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let wins: Vec<Tensor> = t.windows(3).collect();
        assert_eq!(wins.len(), 2);
        assert_eq!(wins[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(wins[1].data(), &[2.0, 3.0, 4.0]);
    }

    /// Test windows iterator with custom step
    #[test]
    fn test_windows_step() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let wins: Vec<Tensor> = t.windows_step(2, 2).collect();
        assert_eq!(wins.len(), 2);
        assert_eq!(wins[0].data(), &[1.0, 2.0]);
        assert_eq!(wins[1].data(), &[3.0, 4.0]);
    }

    /// Test collect_shape utility
    #[test]
    fn test_collect_shape() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();
        let mat = t.chunks(2).collect_shape(vec![3, 2]);
        assert_eq!(mat.shape().dims(), &[3, 2]);
        assert_eq!(mat.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// Performance comparison: Tensor iterator/view system vs Vec iteration
    ///
    /// This test compares end-to-end pipelines (creation → iteration → ops → collection)
    /// across multiple sizes and loop styles, and prints a concise summary.
    #[test]
    fn test_iterator_vs_vec_performance_summary() {
        use std::time::Instant;

        let sizes: [usize; 3] = [100, 1000, 10_000];
        let iterations: usize = 3; // exclude 1 warmup
        let chunk_size: usize = 8192;

        println!(
            "Iterator/View vs Vec performance ({} runs avg, chunk_size={})",
            iterations, chunk_size
        );

        for &n in &sizes {
            // -------- Element-wise iterator pipeline (Tensor) --------
            let mut total_elem_tensor = std::time::Duration::ZERO;
            for run in 0..(iterations + 1) {
                let t0 = Instant::now();
                let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let t = Tensor::from_slice(&data, vec![n]).unwrap();
                let out: Tensor = t
                    .iter_elements()
                    .map(|e| e.mul_scalar(2.0).add_scalar(1.0))
                    .collect();
                // Touch a value to avoid any dead-code elimination concerns
                let _ = out.get(&[0]);
                let dt = t0.elapsed();
                if run > 0 {
                    total_elem_tensor += dt;
                }
            }
            let avg_elem_tensor = total_elem_tensor / iterations as u32;

            // -------- Element-wise iterator pipeline (Vec) --------
            let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let mut total_elem_vec = std::time::Duration::ZERO;
            for run in 0..(iterations + 1) {
                let t0 = Instant::now();
                let _v_out: Vec<f32> = data.iter().map(|&x| 2.0 * x + 1.0).collect();
                let dt = t0.elapsed();
                if run > 0 {
                    total_elem_vec += dt;
                }
            }
            let avg_elem_vec = total_elem_vec / iterations as u32;

            // -------- Chunked iterator pipeline (Tensor) --------
            let mut total_chunks_tensor = std::time::Duration::ZERO;
            for run in 0..(iterations + 1) {
                let t0 = Instant::now();
                let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let t = Tensor::from_slice(&data, vec![n]).unwrap();
                let parts: Vec<Tensor> = t
                    .chunks(chunk_size)
                    .map(|c| c.mul_scalar(2.0).add_scalar(1.0))
                    .collect();
                let out = Tensor::cat(&parts, 0);
                let _ = out.get(&[out.size().saturating_sub(1)]);
                let dt = t0.elapsed();
                if run > 0 {
                    total_chunks_tensor += dt;
                }
            }
            let avg_chunks_tensor = total_chunks_tensor / iterations as u32;

            // -------- Chunked iterator pipeline (Vec) --------
            let mut total_chunks_vec = std::time::Duration::ZERO;
            for run in 0..(iterations + 1) {
                let t0 = Instant::now();
                let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let mut out: Vec<f32> = Vec::with_capacity(n);
                for chunk in data.chunks(chunk_size) {
                    for &x in chunk.iter() {
                        out.push(2.0 * x + 1.0);
                    }
                }
                let _ = out.get(out.len().saturating_sub(1)).copied().unwrap_or(0.0);
                let dt = t0.elapsed();
                if run > 0 {
                    total_chunks_vec += dt;
                }
            }
            let avg_chunks_vec = total_chunks_vec / iterations as u32;

            // -------- Value iterator (Tensor) --------
            let mut total_values_tensor = std::time::Duration::ZERO;
            let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let t = Tensor::from_slice(&data, vec![n]).unwrap();

            for run in 0..(iterations + 1) {
                let t0 = Instant::now();
                let _v_out: Tensor = t.iter().map(|e| 2.0 * e.value() + 1.0).collect();
                let dt = t0.elapsed();
                if run > 0 {
                    total_values_tensor += dt;
                }
            }
            let avg_values_tensor = total_values_tensor / iterations as u32;

            // -------- Mutable value iterator (Tensor) --------
            let mut total_values_mut_tensor = std::time::Duration::ZERO;
            for run in 0..(iterations + 1) {
                let t0 = Instant::now();
                let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let mut out = Tensor::from_slice(&data, vec![n]).unwrap();
                {
                    let d = out.data_mut();
                    for v in d.iter_mut() {
                        *v = 2.0 * *v + 1.0;
                    }
                }
                let _ = out.get(&[out.size().saturating_sub(1)]);
                let dt = t0.elapsed();
                if run > 0 {
                    total_values_mut_tensor += dt;
                }
            }
            let avg_values_mut_tensor = total_values_mut_tensor / iterations as u32;

            // -------- Summary per size --------
            let s_elem = avg_elem_vec.as_secs_f64() / avg_elem_tensor.as_secs_f64();
            let s_chunks = avg_chunks_vec.as_secs_f64() / avg_chunks_tensor.as_secs_f64();
            let s_values = avg_elem_vec.as_secs_f64() / avg_values_tensor.as_secs_f64();
            let s_values_mut = avg_elem_vec.as_secs_f64() / avg_values_mut_tensor.as_secs_f64();

            println!(
                "\n[Size: {:>9} elements]\n  - Tensor (element): {:>8.3} ms\n  - Vec   (element): {:>8.3} ms\n    Speedup (Tensor/Vec): {:>6.2}x\n  - Tensor (chunks):  {:>8.3} ms\n  - Vec   (chunks):  {:>8.3} ms\n    Speedup (Tensor/Vec): {:>6.2}x\n  - Tensor (values):  {:>8.3} ms\n    Speedup (values vs Vec element): {:>6.2}x\n  - Tensor (values_mut):  {:>8.3} ms\n    Speedup (values_mut vs Vec element): {:>6.2}x",
                n,
                avg_elem_tensor.as_secs_f64() * 1e3,
                avg_elem_vec.as_secs_f64() * 1e3,
                s_elem,
                avg_chunks_tensor.as_secs_f64() * 1e3,
                avg_chunks_vec.as_secs_f64() * 1e3,
                s_chunks,
                avg_values_tensor.as_secs_f64() * 1e3,
                s_values,
                avg_values_mut_tensor.as_secs_f64() * 1e3,
                s_values_mut,
            );
        }

        println!("\nNote: timings include creation, iteration, ops (2x+1), and collection.");
    }

    /// Replacement for previous values-only iteration: derive values via scalar views
    #[test]
    fn test_values_via_views() {
        let t =
            Tensor::from_slice(&(0..16).map(|i| i as f32).collect::<Vec<_>>(), vec![16]).unwrap();
        let vals: Vec<f32> = t.iter().map(|e| e.value()).collect();
        assert_eq!(vals, (0..16).map(|i| i as f32).collect::<Vec<_>>());
    }

    /// Replacement for previous iter_values_mut: mutate via data_mut
    #[test]
    fn test_mutation_via_data_mut() {
        let mut t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        {
            let d = t.data_mut();
            for v in d.iter_mut() {
                *v += 1.0;
            }
        }
        assert_eq!(t.data(), &[2.0, 3.0, 4.0, 5.0]);
    }
}
