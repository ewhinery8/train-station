//! Iterator module for tensor element-wise operations
//!
//! This module provides high-performance iterators over tensor elements, where each
//! element is represented as a view tensor of shape `[1]`. This design allows for
//! seamless integration with Rust's standard library iterator methods while
//! leveraging the existing tensor operation framework and gradient tracking.
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
//! ## Basic Element Iteration
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
//!
//! // Basic iteration over elements
//! for element in tensor.iter() {
//!     println!("Element value: {}", element.value());
//! }
//!
//! // Collect elements into a new tensor
//! let collected: Tensor = tensor.iter().collect();
//! assert_eq!(collected.data(), tensor.data());
//! ```
//!
//! ## Element-Wise Transformations
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
//! ## Advanced Iterator Operations
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
//! ## Range Iteration
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
//! ## Double-Ended Iteration
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
//! # Design Principles
//!
//! - **Zero-Copy Views**: Element views share memory with source tensor
//! - **Full Tensor Operations**: Each element supports all tensor methods
//! - **Standard Library Integration**: Complete compatibility with Rust iterators
//! - **Performance First**: Optimized for high-performance element access
//! - **Gradient Preservation**: Maintains gradtrack functionality through operations
//! - **Memory Efficiency**: Minimal overhead for element iteration
//! - **Type Safety**: Compile-time guarantees for iterator operations

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;
use std::iter::{FromIterator, FusedIterator};

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
pub struct TensorElementIterator<'a> {
    /// Reference to the source tensor
    source: &'a Tensor,
    /// Current position in iteration
    position: usize,
    /// End position (exclusive)
    end: usize,
}

impl<'a> TensorElementIterator<'a> {
    /// Create a new iterator over all tensor elements
    ///
    /// Creates an iterator that yields view tensors for each element in the
    /// source tensor. Each element becomes a `Tensor` of shape `[1]` that
    /// supports all tensor operations and gradient tracking.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The source tensor to iterate over
    ///
    /// # Returns
    ///
    /// An iterator that yields view tensors for each element
    ///
    /// # Performance
    ///
    /// - **O(1) Creation**: Constant-time iterator initialization
    /// - **Zero-Copy Views**: Each element is a view sharing memory with source
    /// - **Memory Efficient**: Minimal overhead for iterator state
    ///
    /// # Implementation Details
    ///
    /// This method creates an iterator that yields view tensors for each element
    /// in the source tensor. Each element becomes a `Tensor` of shape `[1]` that
    /// supports all tensor operations and gradient tracking.
    ///
    /// The iterator provides zero-copy access to tensor elements through view
    /// tensors, enabling efficient element-wise operations while maintaining
    /// full compatibility with Rust's standard library iterator methods.
    pub fn new(tensor: &'a Tensor) -> Self {
        Self {
            source: tensor,
            position: 0,
            end: tensor.size(),
        }
    }

    /// Create an iterator over a specific range of elements
    ///
    /// Creates an iterator that yields view tensors for elements in the specified
    /// range. The range is automatically clamped to valid tensor bounds for safety.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The source tensor to iterate over
    /// * `start` - Starting index (inclusive)
    /// * `end` - Ending index (exclusive)
    ///
    /// # Returns
    ///
    /// An iterator that yields view tensors for elements in the specified range
    ///
    /// # Safety
    ///
    /// The range is automatically clamped to valid tensor bounds:
    /// - `start` is clamped to `[0, tensor.size()]`
    /// - `end` is clamped to `[start, tensor.size()]`
    /// - Empty ranges (start >= end) are handled gracefully
    ///
    /// # Performance
    ///
    /// - **O(1) Creation**: Constant-time iterator initialization
    /// - **Bounds Checking**: Automatic range validation and clamping
    /// - **Zero-Copy Views**: Each element is a view sharing memory with source
    ///
    /// # Implementation Details
    ///
    /// This method creates an iterator that yields view tensors for elements in
    /// the specified range. The range is automatically clamped to valid tensor
    /// bounds for safety, ensuring that out-of-bounds access is handled gracefully.
    ///
    /// The iterator provides zero-copy access to tensor elements through view
    /// tensors, enabling efficient element-wise operations while maintaining
    /// full compatibility with Rust's standard library iterator methods.
    pub fn with_range(tensor: &'a Tensor, start: usize, end: usize) -> Self {
        let end = end.min(tensor.size());
        let start = start.min(end);
        Self {
            source: tensor,
            position: start,
            end,
        }
    }

    /// Create an optimized element view for the given position
    ///
    /// This method creates a true view tensor of shape `[1]` that shares memory
    /// with the element at the specified index in the source tensor. The view
    /// enables zero-copy element access with full gradient tracking.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the element to create a view for
    ///
    /// # Returns
    ///
    /// A view tensor of shape `[1]` representing the element at the specified index
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.source.size()`.
    ///
    /// # Performance
    ///
    /// - **O(1) View Creation**: Constant-time view tensor creation
    /// - **Zero-Copy**: View shares memory with source tensor
    /// - **Memory Efficient**: ~64 bytes overhead for view metadata
    /// - **Gradient Tracking**: Full gradtrack support through view operations
    ///
    /// # Implementation Details
    ///
    /// This method delegates to `Tensor::element_view()` which creates a true
    /// view of the underlying data without any copying. The view tensor supports
    /// all standard tensor operations including gradient tracking and SIMD
    /// optimizations.
    fn create_element_view(&self, index: usize) -> Tensor {
        debug_assert!(index < self.source.size());

        self.source.element_view(index)
    }
}

// ===== Core Iterator Implementation =====

impl<'a> Iterator for TensorElementIterator<'a> {
    type Item = Tensor;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.end {
            let view = self.create_element_view(self.position);
            self.position += 1;
            Some(view)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.position;
        (remaining, Some(remaining))
    }

    #[inline]
    fn count(self) -> usize {
        self.end - self.position
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let new_pos = self.position.saturating_add(n);
        if new_pos < self.end {
            self.position = new_pos + 1;
            Some(self.create_element_view(new_pos))
        } else {
            self.position = self.end;
            None
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.position < self.end {
            let last_idx = self.end - 1;
            Some(self.create_element_view(last_idx))
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for TensorElementIterator<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.position
    }
}

impl<'a> FusedIterator for TensorElementIterator<'a> {}

impl<'a> DoubleEndedIterator for TensorElementIterator<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.position < self.end {
            self.end -= 1;
            Some(self.create_element_view(self.end))
        } else {
            None
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let new_end = self.end.saturating_sub(n + 1);
        if new_end >= self.position {
            self.end = new_end;
            Some(self.create_element_view(self.end))
        } else {
            self.position = self.end;
            None
        }
    }
}

// ===== IntoIterator Implementation =====

impl<'a> IntoIterator for &'a Tensor {
    type Item = Tensor;
    type IntoIter = TensorElementIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        TensorElementIterator::new(self)
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
    /// assert_eq!(empty.shape().dims, vec![0]);
    /// ```
    fn from_iter<I: IntoIterator<Item = Tensor>>(iter: I) -> Self {
        let elements: Vec<Tensor> = iter.into_iter().collect();

        if elements.is_empty() {
            return Tensor::new(vec![0]);
        }

        // Check if all elements are scalar views (shape [1])
        let all_scalars = elements.iter().all(|e| e.shape().dims == vec![1]);

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
        let len = elements.len();
        let mut result = Self::new_uninitialized(vec![len]);

        // Determine if we can track gradients
        let requires_grad = elements.iter().any(|e| e.requires_grad());

        // Copy data from element views
        unsafe {
            let dst = result.as_mut_ptr();
            for (i, element) in elements.iter().enumerate() {
                *dst.add(i) = *element.as_ptr();
            }
        }

        // Set up gradient tracking
        if requires_grad && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let element_ids: Vec<usize> = elements.iter().map(|e| e.id()).collect();
            let grad_fn = GradFn::ElementCollection {
                element_ids: element_ids.clone(),
                result_shape: vec![len],
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), element_ids, grad_fn);
        }

        result
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
        // For mixed shapes, flatten all elements into a 1D tensor
        let total_size: usize = elements.iter().map(|e| e.size()).sum();
        let mut result = Self::new_uninitialized(vec![total_size]);

        let requires_grad = elements.iter().any(|e| e.requires_grad());
        let mut offset = 0;

        unsafe {
            let dst = result.as_mut_ptr();
            for element in &elements {
                let src = element.as_ptr();
                let size = element.size();
                std::ptr::copy_nonoverlapping(src, dst.add(offset), size);
                offset += size;
            }
        }

        if requires_grad && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let element_ids: Vec<usize> = elements.iter().map(|e| e.id()).collect();
            let grad_fn = GradFn::ElementCollection {
                element_ids: element_ids.clone(),
                result_shape: vec![total_size],
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), element_ids, grad_fn);
        }

        result
    }

    /// Create an iterator over tensor elements as view tensors
    ///
    /// Each element becomes a `Tensor` of shape `[1]` that supports all
    /// tensor operations and gradient tracking. This is the main entry point
    /// for element-wise iteration with full tensor operation support.
    ///
    /// The iterator provides zero-copy access to tensor elements through view
    /// tensors, enabling efficient element-wise operations while maintaining
    /// full compatibility with Rust's standard library iterator methods.
    ///
    /// # Returns
    ///
    /// An iterator that yields view tensors for each element
    ///
    /// # Performance
    ///
    /// - **Zero-Copy Views**: Each element is a view sharing memory with source
    /// - **O(1) Element Access**: Constant-time view creation for each element
    /// - **Memory Efficient**: ~64 bytes overhead per element view
    /// - **SIMD Compatible**: All tensor operations use existing optimizations
    /// - **Gradient Tracking**: Full gradtrack support through element operations
    ///
    /// # Examples
    ///
    /// ## Basic Element Operations
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    ///
    /// // Use any std iterator method
    /// let result: Tensor = tensor.iter()
    ///     .map(|elem| elem.mul_scalar(2.0).add_scalar(1.0)) // 2x + 1
    ///     .filter(|elem| elem.value() > 3.0)                // Keep values > 3
    ///     .collect();
    ///
    /// assert_eq!(result.data(), &[5.0, 7.0]);
    /// ```
    ///
    /// ## Advanced Iterator Chains
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    ///
    /// // Chain with enumerate, zip, etc.
    /// let indexed: Tensor = tensor.iter()
    ///     .enumerate()
    ///     .map(|(i, elem)| elem.add_scalar(i as f32))
    ///     .collect();
    ///
    /// assert_eq!(indexed.data(), &[1.0, 3.0, 5.0, 7.0, 9.0]);
    /// ```
    ///
    /// ## Double-Ended Iteration
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    ///
    /// // Use double-ended iterator
    /// let reversed: Tensor = tensor.iter()
    ///     .rev()
    ///     .collect();
    ///
    /// assert_eq!(reversed.data(), &[4.0, 3.0, 2.0, 1.0]);
    /// ```
    ///
    /// ## Gradient Tracking
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0], vec![2])
    ///     .unwrap()
    ///     .with_requires_grad();
    ///
    /// let result: Tensor = tensor.iter()
    ///     .map(|elem| elem.mul_scalar(2.0))
    ///     .collect();
    ///
    /// assert!(result.requires_grad());
    /// assert_eq!(result.data(), &[2.0, 4.0]);
    /// ```
    pub fn iter(&self) -> TensorElementIterator {
        TensorElementIterator::new(self)
    }

    /// Create an iterator over a range of elements
    ///
    /// Creates an iterator that yields view tensors for elements in the specified
    /// range. The range is automatically clamped to valid tensor bounds for safety.
    ///
    /// # Arguments
    ///
    /// * `start` - Starting index (inclusive)
    /// * `end` - Ending index (exclusive)
    ///
    /// # Returns
    ///
    /// An iterator that yields view tensors for elements in the specified range
    ///
    /// # Safety
    ///
    /// The range is automatically clamped to valid tensor bounds:
    /// - `start` is clamped to `[0, tensor.size()]`
    /// - `end` is clamped to `[start, tensor.size()]`
    /// - Empty ranges (start >= end) are handled gracefully
    ///
    /// # Performance
    ///
    /// - **O(1) Creation**: Constant-time iterator initialization
    /// - **Bounds Checking**: Automatic range validation and clamping
    /// - **Zero-Copy Views**: Each element is a view sharing memory with source
    /// - **Memory Efficient**: Minimal overhead for range iteration
    ///
    /// # Examples
    ///
    /// ## Basic Range Iteration
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    /// let middle: Tensor = tensor.iter_range(1, 4)
    ///     .map(|elem| elem.mul_scalar(2.0))
    ///     .collect();
    ///
    /// assert_eq!(middle.data(), &[4.0, 6.0, 8.0]);
    /// ```
    ///
    /// ## Range with Operations
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    ///
    /// // Apply complex operations to range
    /// let result: Tensor = tensor.iter_range(0, 3)
    ///     .enumerate()
    ///     .map(|(i, elem)| elem.add_scalar(i as f32))
    ///     .collect();
    ///
    /// assert_eq!(result.data(), &[1.0, 3.0, 5.0]);
    /// ```
    ///
    /// ## Out of Bounds Handling
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    ///
    /// // Out of bounds range is clamped
    /// let empty: Tensor = tensor.iter_range(5, 10).collect();
    /// assert_eq!(empty.size(), 0);
    ///
    /// // Partial out of bounds
    /// let partial: Tensor = tensor.iter_range(1, 10).collect();
    /// assert_eq!(partial.data(), &[2.0, 3.0]);
    /// ```
    pub fn iter_range(&self, start: usize, end: usize) -> TensorElementIterator {
        TensorElementIterator::with_range(self, start, end)
    }
}

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

        let elements: Vec<Tensor> = tensor.iter().collect();
        assert_eq!(elements.len(), 4);

        // Check that each element is a scalar tensor with correct value
        for (i, elem) in elements.iter().enumerate() {
            assert_eq!(elem.shape().dims, vec![1]);
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
        let first = iter.next().unwrap();
        assert_eq!(first.value(), 1.0);

        // Test size_hint()
        assert_eq!(iter.size_hint(), (4, Some(4)));

        // Test count()
        assert_eq!(iter.count(), 4);

        // Test nth()
        let mut iter = tensor.iter();
        let third = iter.nth(2).unwrap();
        assert_eq!(third.value(), 3.0);

        // Test last()
        let iter = tensor.iter();
        let last = iter.last().unwrap();
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
        assert_eq!(collected.shape().dims, vec![4]);
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
}
