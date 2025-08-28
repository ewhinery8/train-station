//! Tensor shape and memory layout management
//!
//! This module provides the `Shape` struct and related components for managing
//! tensor dimensions, memory strides, and layout information. The shape system
//! enables efficient view operations, broadcasting, and memory access optimization.
//!
//! # Architecture
//!
//! The shape system consists of:
//! - **Shape**: Main struct containing dimensions, strides, and layout information
//! - **MemoryLayout**: Enum describing memory layout types (Contiguous, Strided, View)
//! - **Stride Calculation**: Efficient computation of memory access patterns
//! - **Broadcasting**: NumPy-compatible broadcasting rules implementation
//!
//! # Key Features
//!
//! - **Memory Layout Tracking**: Contiguous, strided, and view layout types
//! - **Stride Optimization**: Efficient memory access pattern calculation
//! - **Broadcasting Support**: NumPy-compatible broadcasting rules
//! - **View Operations**: Zero-copy tensor transformations
//! - **Performance Hints**: Layout information for operation optimization
//! - **Memory Safety**: Bounds checking and validation
//!
//! # Performance Characteristics
//!
//! - **Zero-Cost Layout**: Layout information computed once and cached
//! - **Efficient Strides**: Row-major stride calculation for optimal memory access
//! - **Broadcasting**: O(rank) complexity for broadcasting compatibility checks
//! - **Memory Access**: O(1) offset calculation for multi-dimensional indices
//! - **View Efficiency**: Zero-copy view creation with minimal overhead
//!
//! # Memory Layout Types
//!
//! - **Contiguous**: Standard row-major layout with sequential memory access
//! - **Strided**: Custom stride layout for non-contiguous memory access
//! - **View**: Non-contiguous reference to existing tensor data
//!
//! # Examples
//!
//! ## Basic Shape Operations
//!
//! ```
//! use train_station::tensor::Shape;
//!
//! // Create contiguous shape
//! let shape = Shape::new(vec![2, 3, 4]);
//! assert_eq!(shape.size, 24);
//! assert!(shape.is_contiguous());
//!
//! // Create view shape
//! let view_shape = Shape::as_view(vec![2, 2], vec![4, 1]);
//! assert!(view_shape.is_view());
//!
//! // Check broadcasting compatibility
//! let shape1 = Shape::new(vec![2, 3, 4]);
//! let shape2 = Shape::new(vec![1, 3, 4]);
//! assert!(shape1.is_broadcastable_with(&shape2));
//!
//! // Calculate memory offset
//! let offset = shape.offset(&[1, 2, 3]);
//! assert_eq!(offset, 12 + 8 + 3);
//! ```
//!
//! # Design Principles
//!
//! - **Memory Efficiency**: Optimized for cache-friendly access patterns
//! - **Zero-Cost Abstractions**: Minimal overhead for shape operations
//! - **NumPy Compatibility**: Broadcasting rules match NumPy behavior
//! - **Type Safety**: Strong typing for memory layout and dimensions
//! - **Performance First**: All operations optimized for speed

/// Memory layout information for tensors
///
/// Describes how tensor data is arranged in memory for optimized access patterns
/// and view operations. This enum provides performance hints for operation
/// selection and memory access optimization.
///
/// # Variants
///
/// * `Contiguous` - Standard row-major layout with sequential memory access
/// * `Strided` - Custom stride layout for non-contiguous memory access
/// * `View` - Non-contiguous reference to existing tensor data
///
/// # Performance Characteristics
///
/// - **Contiguous**: Optimal for SIMD operations and cache efficiency
/// - **Strided**: Requires custom memory access patterns
/// - **View**: Zero-copy operations with shared memory management
///
/// # Implementation Details
///
/// This enum is used internally by the shape system to track memory layout
/// information for optimization decisions. The layout type determines which
/// operations can be used efficiently on the tensor data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Contiguous memory layout (standard row-major)
    Contiguous,
    /// Strided memory layout with custom stride information
    Strided,
    /// Non-contiguous view of another tensor
    View,
}

/// Represents the shape/dimensions of a tensor with stride tracking
///
/// This struct holds the dimensional information for a tensor, including the size
/// of each dimension, memory strides, and layout information for efficient view
/// operations and transformations. The shape system enables zero-copy tensor
/// views and optimized memory access patterns.
///
/// # Key Features
///
/// - **Dimension Management**: Multi-dimensional shape representation
/// - **Stride Calculation**: Efficient memory access pattern computation
/// - **Layout Tracking**: Contiguous, strided, and view layout types
/// - **Broadcasting**: NumPy-compatible broadcasting rules
/// - **Memory Safety**: Bounds checking and validation
///
/// # Performance Characteristics
///
/// - **Zero-Cost Layout**: Layout information computed once and cached
/// - **Efficient Strides**: Row-major stride calculation for optimal access
/// - **Memory Access**: O(1) offset calculation for multi-dimensional indices
/// - **View Efficiency**: Zero-copy view creation with minimal overhead
///
/// # Examples
///
/// ```
/// use train_station::tensor::Shape;
///
/// let shape = Shape::new(vec![2, 3, 4]);
/// assert_eq!(shape.size, 24);
/// assert!(shape.is_contiguous());
/// assert_eq!(shape.strides(), &[12, 4, 1]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// The dimensions of the tensor (e.g., [2, 3, 4] for a 2x3x4 tensor)
    pub dims: Vec<usize>,
    /// Total number of elements in the tensor
    pub size: usize,
    /// Memory strides for each dimension (elements between consecutive indices)
    /// For a contiguous tensor with shape [2, 3, 4], strides would be [12, 4, 1]
    pub strides: Vec<usize>,
    /// Memory layout type for optimization decisions
    pub layout: MemoryLayout,
}

impl Shape {
    /// Creates a new contiguous shape from a vector of dimensions
    ///
    /// Computes the total size and contiguous strides for the given dimensions.
    /// The resulting shape uses row-major memory layout optimized for cache
    /// efficiency and SIMD operations.
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector of dimension sizes defining the tensor shape
    ///
    /// # Returns
    ///
    /// A new Shape with calculated size, contiguous strides, and contiguous layout
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(rank) for stride calculation
    /// - **Memory**: Single allocation for dimensions and strides
    /// - **Optimization**: Row-major layout for cache efficiency
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.size, 24);
    /// assert!(shape.is_contiguous());
    /// assert_eq!(shape.strides(), &[12, 4, 1]);
    /// ```
    #[inline]
    pub fn new(dims: Vec<usize>) -> Self {
        let size = dims.iter().product();
        let strides = Self::compute_contiguous_strides(&dims);
        Self {
            dims,
            size,
            strides,
            layout: MemoryLayout::Contiguous,
        }
    }

    /// Creates a new shape with custom strides
    ///
    /// Creates a shape with user-defined strides for non-contiguous memory layouts.
    /// Automatically detects if the strides represent a contiguous layout and sets
    /// the appropriate layout type.
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector of dimension sizes defining the tensor shape
    /// * `strides` - Vector of memory strides for each dimension
    ///
    /// # Returns
    ///
    /// A new Shape with the given dimensions and strides, with layout type
    /// automatically determined
    ///
    /// # Panics
    ///
    /// Panics if dimensions and strides have different lengths
    ///
    /// # Performance
    ///
    /// - **Layout Detection**: O(rank) comparison with contiguous strides
    /// - **Memory**: Single allocation for shape data
    /// - **Optimization**: Automatic layout type detection
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::Shape;
    ///
    /// let shape = Shape::with_strides(vec![2, 3], vec![6, 2]);
    /// assert_eq!(shape.size, 6);
    /// assert!(!shape.is_contiguous());
    /// assert_eq!(shape.strides(), &[6, 2]);
    /// ```
    #[inline]
    pub fn with_strides(dims: Vec<usize>, strides: Vec<usize>) -> Self {
        assert_eq!(
            dims.len(),
            strides.len(),
            "Dimensions and strides must have same length"
        );
        let size = dims.iter().product();
        let layout = if strides == Self::compute_contiguous_strides(&dims) {
            MemoryLayout::Contiguous
        } else {
            MemoryLayout::Strided
        };
        Self {
            dims,
            size,
            strides,
            layout,
        }
    }

    /// Creates a view shape (non-contiguous reference to existing tensor)
    ///
    /// Creates a shape representing a view of existing tensor data with custom
    /// dimensions and strides. View shapes enable zero-copy tensor transformations
    /// by sharing memory with the original tensor.
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector of dimension sizes for the view
    /// * `strides` - Vector of memory strides for the view
    ///
    /// # Returns
    ///
    /// A new Shape marked as a view with the given dimensions and strides
    ///
    /// # Panics
    ///
    /// Panics if dimensions and strides have different lengths
    ///
    /// # Performance
    ///
    /// - **Zero-Copy**: No data copying, only metadata creation
    /// - **Memory Efficient**: Shares memory with original tensor
    /// - **View Optimization**: Enables view-specific operation optimizations
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::Shape;
    ///
    /// let view_shape = Shape::as_view(vec![2, 2], vec![4, 1]);
    /// assert!(view_shape.is_view());
    /// assert!(!view_shape.is_contiguous());
    /// ```
    #[inline]
    pub fn as_view(dims: Vec<usize>, strides: Vec<usize>) -> Self {
        assert_eq!(
            dims.len(),
            strides.len(),
            "Dimensions and strides must have same length"
        );
        let size = dims.iter().product();
        Self {
            dims,
            size,
            strides,
            layout: MemoryLayout::View,
        }
    }

    /// Computes contiguous strides for given dimensions (row-major order)
    ///
    /// Calculates the memory strides for a contiguous row-major layout.
    /// This is used internally for shape creation and layout detection.
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector of dimension sizes
    ///
    /// # Returns
    ///
    /// Vector of strides for contiguous row-major layout
    ///
    /// # Implementation Details
    ///
    /// This method is used internally by the shape system to compute
    /// contiguous strides for new shapes and to detect if custom strides
    /// represent a contiguous layout.
    fn compute_contiguous_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(dims.len());
        if dims.is_empty() {
            return strides;
        }

        let mut stride = 1;
        for &dim in dims.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();
        strides
    }

    /// Returns the number of dimensions (rank) of the tensor
    ///
    /// # Returns
    ///
    /// The number of dimensions in the tensor shape
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.rank(), 3);
    /// ```
    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Checks if the tensor has contiguous memory layout
    ///
    /// # Returns
    ///
    /// `true` if the tensor data is stored contiguously in memory
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert!(shape.is_contiguous());
    /// ```
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        matches!(self.layout, MemoryLayout::Contiguous)
    }

    /// Checks if the tensor is a view of another tensor
    ///
    /// # Returns
    ///
    /// `true` if this tensor is a view (non-contiguous reference)
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::Shape;
    ///
    /// let view_shape = Shape::as_view(vec![2, 2], vec![4, 1]);
    /// assert!(view_shape.is_view());
    /// ```
    #[inline]
    pub fn is_view(&self) -> bool {
        matches!(self.layout, MemoryLayout::View)
    }

    /// Gets the memory stride for a specific dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension index
    ///
    /// # Returns
    ///
    /// The memory stride for the given dimension
    ///
    /// # Panics
    ///
    /// Panics if `dim` is out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.stride(0), 12);
    /// assert_eq!(shape.stride(1), 4);
    /// assert_eq!(shape.stride(2), 1);
    /// ```
    #[inline]
    pub fn stride(&self, dim: usize) -> usize {
        self.strides[dim]
    }

    /// Gets all memory strides
    ///
    /// # Returns
    ///
    /// Reference to the stride vector
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.strides(), &[12, 4, 1]);
    /// ```
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Gets the memory layout type
    ///
    /// # Returns
    ///
    /// Reference to the memory layout
    ///
    /// # Implementation Details
    ///
    /// This method returns the memory layout type which can be used for
    /// optimization decisions in tensor operations.
    #[inline]
    pub fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Calculates the linear memory offset for given indices
    ///
    /// Computes the linear memory offset for multi-dimensional tensor indices
    /// using the stored stride information. This enables efficient direct memory
    /// access for tensor operations.
    ///
    /// # Arguments
    ///
    /// * `indices` - Vector of indices for each dimension
    ///
    /// # Returns
    ///
    /// Linear memory offset for the given multi-dimensional indices
    ///
    /// # Panics
    ///
    /// Panics if indices length doesn't match tensor rank
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(rank) for offset calculation
    /// - **Memory**: No allocation, uses existing stride data
    /// - **Optimization**: Efficient dot product of indices and strides
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// let offset = shape.offset(&[1, 2, 3]);
    /// assert_eq!(offset, 12 + 8 + 3); // 1*12 + 2*4 + 3*1
    /// ```
    #[inline]
    pub fn offset(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.rank(), "Indices must match tensor rank");
        indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    /// Checks if this shape is broadcastable with another shape
    ///
    /// Determines if two shapes can be broadcast together according to NumPy
    /// broadcasting rules. Broadcasting enables element-wise operations between
    /// tensors with different shapes by expanding singleton dimensions.
    ///
    /// # Arguments
    ///
    /// * `other` - The other shape to check broadcasting compatibility
    ///
    /// # Returns
    ///
    /// `true` if the shapes are broadcastable according to NumPy broadcasting rules
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(max(rank1, rank2)) for broadcasting check
    /// - **Memory**: No allocation, uses existing dimension data
    /// - **Optimization**: Right-aligned dimension comparison
    ///
    /// # Broadcasting Rules
    ///
    /// - Dimensions are compared from right to left
    /// - Dimensions are compatible if they are equal, or one is 1
    /// - Missing dimensions are treated as 1
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::Shape;
    ///
    /// let shape1 = Shape::new(vec![2, 3, 4]);
    /// let shape2 = Shape::new(vec![1, 3, 4]);
    /// assert!(shape1.is_broadcastable_with(&shape2));
    ///
    /// let shape3 = Shape::new(vec![4]);
    /// assert!(shape1.is_broadcastable_with(&shape3));
    /// ```
    pub fn is_broadcastable_with(&self, other: &Shape) -> bool {
        let max_rank = self.rank().max(other.rank());

        for i in 0..max_rank {
            let self_dim = if i < self.rank() {
                self.dims[self.rank() - 1 - i]
            } else {
                1
            };

            let other_dim = if i < other.rank() {
                other.dims[other.rank() - 1 - i]
            } else {
                1
            };

            if self_dim != other_dim && self_dim != 1 && other_dim != 1 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    //! Shape and memory layout tests
    //!
    //! Comprehensive tests for shape creation, memory layout detection, stride
    //! calculation, broadcasting compatibility, and offset computation. Tests
    //! cover all major functionality including edge cases and performance characteristics.

    use super::*;

    /// Test basic shape creation and properties
    ///
    /// Verifies that shapes are created with correct dimensions, size, rank,
    /// and layout information. Tests the fundamental shape creation functionality.
    #[test]
    fn test_shape_creation() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.size, 24);
        assert_eq!(shape.rank(), 3);
        assert!(shape.is_contiguous());
        assert!(!shape.is_view());
        assert_eq!(shape.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_shape_1d() {
        let shape = Shape::new(vec![10]);
        assert_eq!(shape.size, 10);
        assert_eq!(shape.rank(), 1);
        assert!(shape.is_contiguous());
        assert_eq!(shape.strides(), &[1]);
    }

    #[test]
    fn test_shape_2d() {
        let shape = Shape::new(vec![5, 6]);
        assert_eq!(shape.size, 30);
        assert_eq!(shape.rank(), 2);
        assert!(shape.is_contiguous());
        assert_eq!(shape.strides(), &[6, 1]);
    }

    #[test]
    fn test_zero_sized_shape() {
        let shape = Shape::new(vec![0]);
        assert_eq!(shape.size, 0);
        assert_eq!(shape.rank(), 1);
        assert_eq!(shape.strides(), &[1]);
    }

    #[test]
    fn test_shape_with_strides() {
        let shape = Shape::with_strides(vec![2, 3], vec![6, 2]);
        assert_eq!(shape.size, 6);
        assert_eq!(shape.rank(), 2);
        assert!(!shape.is_contiguous());
        assert_eq!(shape.strides(), &[6, 2]);
        assert_eq!(shape.stride(0), 6);
        assert_eq!(shape.stride(1), 2);
    }

    #[test]
    fn test_shape_as_view() {
        let shape = Shape::as_view(vec![2, 2], vec![4, 1]);
        assert_eq!(shape.size, 4);
        assert_eq!(shape.rank(), 2);
        assert!(shape.is_view());
        assert!(!shape.is_contiguous());
        assert_eq!(shape.strides(), &[4, 1]);
    }

    #[test]
    fn test_stride_calculation() {
        let dims = vec![2, 3, 4];
        let strides = Shape::compute_contiguous_strides(&dims);
        assert_eq!(strides, vec![12, 4, 1]);

        let dims = vec![5];
        let strides = Shape::compute_contiguous_strides(&dims);
        assert_eq!(strides, vec![1]);
    }

    /// Test memory offset calculation for multi-dimensional indices
    ///
    /// Verifies that linear memory offsets are correctly calculated for various
    /// multi-dimensional index combinations using stride information.
    #[test]
    fn test_offset_calculation() {
        let shape = Shape::new(vec![2, 3, 4]);

        // Test corner cases
        assert_eq!(shape.offset(&[0, 0, 0]), 0);
        assert_eq!(shape.offset(&[1, 2, 3]), 12 + 8 + 3);
        assert_eq!(shape.offset(&[1, 0, 0]), 12);
        assert_eq!(shape.offset(&[0, 1, 0]), 4);
        assert_eq!(shape.offset(&[0, 0, 1]), 1);
    }

    /// Test broadcasting compatibility rules
    ///
    /// Verifies that NumPy-compatible broadcasting rules are correctly implemented
    /// for various shape combinations including singleton dimensions and different ranks.
    #[test]
    fn test_broadcasting_compatibility() {
        let shape1 = Shape::new(vec![2, 3, 4]);
        let shape2 = Shape::new(vec![1, 3, 4]);
        let shape3 = Shape::new(vec![4]);
        let shape4 = Shape::new(vec![2, 1, 4]);
        let shape5 = Shape::new(vec![2, 2, 4]);

        assert!(shape1.is_broadcastable_with(&shape2));
        assert!(shape1.is_broadcastable_with(&shape3));
        assert!(shape1.is_broadcastable_with(&shape4));
        assert!(!shape1.is_broadcastable_with(&shape5)); // 3 != 2 and neither is 1

        assert!(shape2.is_broadcastable_with(&shape1));
        assert!(shape3.is_broadcastable_with(&shape1));
        assert!(shape4.is_broadcastable_with(&shape1));
    }

    #[test]
    fn test_contiguous_strides_detection() {
        let shape1 = Shape::with_strides(vec![2, 3, 4], vec![12, 4, 1]);
        assert!(shape1.is_contiguous());

        let shape2 = Shape::with_strides(vec![2, 3, 4], vec![12, 4, 2]);
        assert!(!shape2.is_contiguous());
    }
}
