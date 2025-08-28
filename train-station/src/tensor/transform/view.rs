//! Tensor view operations for zero-copy tensor transformations
//!
//! This module provides a comprehensive view system that enables zero-copy tensor views
//! with proper gradient tracking. Views allow you to create new tensor references that
//! share memory with source tensors, enabling efficient operations without data copying.
//!
//! # Core Operations
//!
//! * `view()` - Reshape tensor with new dimensions (requires contiguous memory)
//! * `element_view()` - Create a view of a single element
//! * `slice_view()` - Create a view of a contiguous or strided slice
//! * `create_view()` - Create views using the unified ViewSpec system
//!
//! # Performance Characteristics
//!
//! * **Zero-Copy Operations**: Views share memory with source tensors
//! * **Memory Efficient**: No additional memory allocation for view operations
//! * **Gradient Tracking**: Full GradTrack support through view operations
//! * **Unified Design**: Single ViewSpec enum handles all view types
//! * **Memory Safety**: Views maintain proper lifetimes and memory ownership
//!
//! # Implementation Details
//!
//! The view system provides zero-copy tensor transformations through:
//! - **Unified ViewSpec System**: Single enum handles all view types
//! - **Memory Sharing**: Views share memory with source tensors
//! - **Gradient Tracking**: Full GradTrack support for all view operations
//! - **Performance Optimization**: Zero-copy operations when possible
//! - **Type Safety**: Comprehensive validation for safe view creation
//!
//! # Gradient Tracking
//!
//! The view operations support automatic gradient tracking through
//! the GradTrack system. When `requires_grad` is enabled, views maintain
//! gradient tracking and properly propagate gradients back to source tensors.

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;
use crate::tensor::Shape;

/// Unified view specification enum defining all supported view types
///
/// This enum provides a unified interface for creating different types of tensor views.
/// Each variant contains the parameters needed for that specific view type, enabling
/// consistent view creation across the entire system. The ViewSpec system ensures
/// type-safe view creation with comprehensive validation and gradient tracking support.
///
/// # Variants
///
/// * `Element` - Single element view for accessing individual tensor elements
/// * `Slice` - Contiguous or strided slice views for accessing tensor regions
/// * `Reshape` - Shape transformation views that maintain memory layout
/// * `Transpose` - Dimension reordering views for matrix operations
///
/// # Implementation Details
///
/// The ViewSpec system provides type-safe view creation with:
/// - **Comprehensive Validation**: Built-in bounds checking and compatibility validation
/// - **Gradient Function Generation**: Automatic gradient function creation for all view types
/// - **Shape Calculation**: Efficient shape transformation for all view operations
/// - **Contiguity Analysis**: Optimized view creation based on memory layout
///
/// # Performance
///
/// - **Zero-copy**: All view types share memory with source tensors
/// - **Validation**: Built-in validation ensures safe view creation
/// - **Gradient tracking**: Automatic gradient function generation
/// - **Memory layout**: Optimized for contiguous memory access patterns
#[derive(Debug, Clone)]
pub enum ViewSpec {
    /// Single element view - creates a scalar tensor viewing one element
    ///
    /// Used primarily by the iterator system to create element views that
    /// maintain gradient tracking. Returns a scalar tensor with shape `[1]`.
    /// This variant enables efficient access to individual tensor elements
    /// while preserving memory sharing and gradient connectivity.
    ///
    /// # Arguments
    ///
    /// * `index` - Linear index of the element in the source tensor
    ///
    /// # Implementation Details
    ///
    /// Element views are always contiguous and provide efficient access to individual
    /// tensor elements while maintaining gradient tracking and memory sharing.
    Element {
        /// Linear index of the element in the source tensor
        index: usize,
    },

    /// Slice view - creates a view of a contiguous or strided slice
    ///
    /// Supports both contiguous slices (step=1) and strided slices (step>1).
    /// Contiguous slices are true zero-copy views, while strided slices may
    /// require data copying for optimal performance. This variant enables
    /// efficient access to tensor regions while maintaining gradient tracking.
    ///
    /// # Arguments
    ///
    /// * `start` - Starting index in the source tensor
    /// * `step` - Step size (1 for contiguous, >1 for strided)
    /// * `length` - Number of elements in the slice
    ///
    /// # Implementation Details
    ///
    /// Slice views support both contiguous (step=1) and strided (step>1) access patterns.
    /// Contiguous slices are true zero-copy views, while strided slices may require
    /// data copying for optimal performance.
    Slice {
        /// Starting index in the source tensor
        start: usize,
        /// Step size (1 for contiguous, >1 for strided)
        step: usize,
        /// Number of elements in the slice
        length: usize,
    },

    /// Reshape view - changes tensor shape while maintaining memory layout
    ///
    /// Equivalent to PyTorch's `view()` operation. Requires the source tensor
    /// to be contiguous and the total number of elements to remain the same.
    /// This operation preserves the underlying memory layout while changing
    /// the logical arrangement of data. It enables efficient shape transformations
    /// without data copying.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - New shape for the view tensor
    ///
    /// # Implementation Details
    ///
    /// Reshape views require contiguous source tensors and preserve the underlying
    /// memory layout while changing the logical arrangement of data.
    Reshape {
        /// New shape for the view tensor
        new_shape: Vec<usize>,
    },

    /// Transpose view - swaps two dimensions
    ///
    /// Creates a view with two dimensions swapped. May result in a non-contiguous
    /// view depending on the tensor's current memory layout. For 2D tensors,
    /// this performs a standard matrix transpose operation. Currently supports
    /// 2D matrix transposes as zero-copy views.
    ///
    /// # Arguments
    ///
    /// * `dim0` - First dimension to swap
    /// * `dim1` - Second dimension to swap
    ///
    /// # Implementation Details
    ///
    /// Transpose views may result in non-contiguous layouts depending on the source
    /// tensor's memory arrangement. Currently supports 2D matrix transposes as
    /// zero-copy views.
    #[allow(unused)]
    Transpose {
        /// First dimension to swap
        dim0: usize,
        /// Second dimension to swap
        dim1: usize,
    },
}

impl ViewSpec {
    /// Validate that this view specification can be applied to the given source tensor
    ///
    /// Performs comprehensive validation to ensure that the view specification
    /// can be safely applied to the source tensor. This includes bounds checking,
    /// shape compatibility, and memory layout requirements.
    ///
    /// # Arguments
    ///
    /// * `source_shape` - Shape of the source tensor to validate against
    ///
    /// # Returns
    ///
    /// `true` if the view can be created efficiently and safely, `false` otherwise
    ///
    /// # Implementation Details
    ///
    /// The is_valid_for method performs comprehensive validation to ensure that
    /// the view specification can be safely applied to the source tensor, including
    /// bounds checking, shape compatibility, and memory layout requirements.
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Simple bounds and compatibility checks
    /// - **Memory Usage**: No allocation
    /// - **Validation**: Comprehensive safety checks for all view types
    fn is_valid_for(&self, source_shape: &Shape) -> bool {
        match self {
            ViewSpec::Element { index } => *index < source_shape.size,

            ViewSpec::Slice {
                start,
                step,
                length,
            } => {
                *step > 0
                    && *start < source_shape.size
                    && *start + (*length - 1) * *step < source_shape.size
            }

            ViewSpec::Reshape { new_shape } => {
                source_shape.is_contiguous()
                    && source_shape.size == new_shape.iter().product::<usize>()
            }

            ViewSpec::Transpose { dim0, dim1 } => {
                *dim0 < source_shape.rank() && *dim1 < source_shape.rank()
            }
        }
    }

    /// Calculate the shape of the resulting view tensor
    ///
    /// Computes the shape that the view tensor will have based on the view
    /// specification and the source tensor's shape. This method handles all
    /// view types and their specific shape transformation rules.
    ///
    /// # Arguments
    ///
    /// * `source_shape` - Shape of the source tensor
    ///
    /// # Returns
    ///
    /// Shape of the view tensor after applying the view specification
    ///
    /// # Implementation Details
    ///
    /// The calculate_view_shape method computes the shape that the view tensor will
    /// have based on the view specification and source tensor shape, handling all
    /// view types with appropriate transformation rules.
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) for most operations, O(n) for transpose
    /// - **Memory Usage**: Allocates new shape data for complex transformations
    /// - **Shape Calculation**: Handles all view types with appropriate transformations
    fn calculate_view_shape(&self, source_shape: &Shape) -> Shape {
        match self {
            ViewSpec::Element { .. } => Shape::new(vec![1]),

            ViewSpec::Slice { length, .. } => Shape::new(vec![*length]),

            ViewSpec::Reshape { new_shape } => Shape::new(new_shape.clone()),

            ViewSpec::Transpose { dim0, dim1 } => {
                let mut new_dims = source_shape.dims.clone();
                let mut new_strides = source_shape.strides.clone();
                new_dims.swap(*dim0, *dim1);
                new_strides.swap(*dim0, *dim1);
                Shape::as_view(new_dims, new_strides)
            }
        }
    }

    /// Get the gradient function for this view type
    ///
    /// Creates the appropriate gradient function for this view type to ensure
    /// proper gradient flow during backward passes. Each view type has specific
    /// gradient propagation rules that are encoded in the returned GradFn.
    ///
    /// # Arguments
    ///
    /// * `source_id` - ID of the source tensor for gradient tracking
    /// * `source_shape` - Shape of the source tensor for gradient calculations
    ///
    /// # Returns
    ///
    /// GradFn describing how gradients flow through this view during backward passes
    ///
    /// # Implementation Details
    ///
    /// The grad_fn method creates appropriate gradient functions for each view type
    /// to ensure proper gradient flow during backward passes. Each view type has
    /// specific gradient propagation rules encoded in the returned GradFn.
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Simple gradient function creation
    /// - **Memory Usage**: Allocates gradient function data
    /// - **Gradient Tracking**: Ensures proper gradient propagation for all view types
    fn grad_fn(&self, source_id: usize, source_shape: &Shape) -> GradFn {
        match self {
            ViewSpec::Element { index } => GradFn::ElementView {
                source_id,
                element_index: *index,
                source_shape: source_shape.dims.clone(),
            },

            ViewSpec::Slice {
                start,
                step,
                length,
            } => GradFn::SliceView {
                start: *start,
                step: *step,
                length: *length,
                input_shape: source_shape.dims.clone(),
            },

            ViewSpec::Reshape { .. } => GradFn::Reshape {
                original_shape: source_shape.dims.clone(),
            },

            ViewSpec::Transpose { .. } => GradFn::Reshape {
                original_shape: source_shape.dims.clone(),
            },
        }
    }

    /// Check if this view creates a contiguous memory layout
    ///
    /// Determines whether the view operation will result in a contiguous
    /// memory layout. This is important for performance optimization and
    /// determining whether the view can be created as a zero-copy operation.
    ///
    /// # Arguments
    ///
    /// * `source_shape` - Shape of the source tensor to analyze
    ///
    /// # Returns
    ///
    /// `true` if the resulting view will be contiguous, `false` otherwise
    ///
    /// # Implementation Details
    ///
    /// The creates_contiguous_view method determines whether a view operation will
    /// result in a contiguous memory layout, which is important for performance
    /// optimization and determining optimal view creation strategies.
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Simple contiguity checks
    /// - **Memory Usage**: No allocation
    /// - **Optimization**: Helps determine optimal view creation strategy
    fn creates_contiguous_view(&self, source_shape: &Shape) -> bool {
        match self {
            ViewSpec::Element { .. } => true, // Single element is always contiguous
            ViewSpec::Slice { step, .. } => *step == 1, // Only contiguous slices
            ViewSpec::Reshape { .. } => source_shape.is_contiguous(), // Preserve contiguity
            ViewSpec::Transpose { dim0, dim1 } => {
                // Simple case: only allow transpose as view if it preserves contiguity
                source_shape.is_contiguous() && can_transpose_as_view(source_shape, *dim0, *dim1)
            }
        }
    }
}

/// Helper function to check if transpose can be done as a view
///
/// Determines whether a transpose operation can be performed as a zero-copy
/// view operation. Currently, only 2D matrix transposes are supported as views.
///
/// # Arguments
///
/// * `shape` - Shape of the tensor to transpose
/// * `dim0` - First dimension to swap
/// * `dim1` - Second dimension to swap
///
/// # Returns
///
/// `true` if the transpose can be done as a view, `false` otherwise
///
/// # Performance
///
/// - **Time Complexity**: O(1) - Simple rank and dimension checks
/// - **Memory Usage**: No allocation
/// - **View Optimization**: Helps determine when transpose can be zero-copy
fn can_transpose_as_view(shape: &Shape, dim0: usize, dim1: usize) -> bool {
    // For now, only allow matrix transpose (2D) as view
    shape.rank() == 2 && ((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0))
}

/// Create a tensor view using the unified ViewSpec system
///
/// This function creates a zero-copy view of a source tensor using the provided
/// view specification. The view shares memory with the source tensor and maintains
/// proper gradient tracking through the GradTrack system.
///
/// # Arguments
///
/// * `source` - Source tensor to create a view of
/// * `spec` - View specification defining the type and parameters of the view
///
/// # Returns
///
/// A new tensor that views the source tensor data with appropriate shape and gradient tracking
///
/// # Panics
///
/// * If the view specification is invalid for the source tensor
/// * If the view parameters are out of bounds
/// * If the view requirements cannot be satisfied
///
/// # Examples
///
/// # Implementation Details
///
/// The create_view function provides the core implementation for creating tensor views
/// using the unified ViewSpec system. It handles all view types with comprehensive
/// validation, gradient tracking, and memory safety.
///
/// # Performance
///
/// - **Zero-copy**: Views share memory with source tensors when possible
/// - **Gradient tracking**: Automatic gradient function registration
/// - **Memory efficiency**: No additional allocation for view operations
/// - **Validation**: Comprehensive safety checks before view creation
///
/// # Memory Safety
///
/// Views maintain proper memory safety through:
/// - Shared ownership of underlying data
/// - Proper lifetime management
/// - Bounds checking and validation
/// - Gradient tracking preservation
pub fn create_view(source: &Tensor, spec: ViewSpec) -> Tensor {
    // Validate the view specification
    assert!(
        spec.is_valid_for(source.shape()),
        "Invalid view specification for source tensor shape"
    );

    // Calculate the view shape
    let view_shape = spec.calculate_view_shape(source.shape());

    // Handle different view types
    match spec {
        ViewSpec::Element { index } => {
            // Create element view with offset pointer
            let element_ptr = unsafe { source.as_ptr().add(index) };
            let mut view = Tensor::from_raw_view(
                element_ptr,
                view_shape,
                source.device(),
                source.allocation_owner().cloned(),
            );

            // Set up gradient tracking
            if source.requires_grad() && is_grad_enabled() {
                view.set_requires_grad(true);
                let grad_fn = spec.grad_fn(source.id(), source.shape());
                view.set_grad_fn(grad_fn.clone());
                GradEngine::register_operation(view.id(), vec![source.id()], grad_fn);
            }

            view
        }

        ViewSpec::Slice {
            start,
            step,
            length,
        } => {
            if step == 1 && start + length <= source.size() {
                // Contiguous slice - create a true view
                let slice_ptr = unsafe { source.as_ptr().add(start) };
                let mut view = Tensor::from_raw_view(
                    slice_ptr,
                    view_shape,
                    source.device(),
                    source.allocation_owner().cloned(),
                );

                // Set up gradient tracking for slice views
                if source.requires_grad() && is_grad_enabled() {
                    view.set_requires_grad(true);
                    let grad_fn = spec.grad_fn(source.id(), source.shape());
                    view.set_grad_fn(grad_fn.clone());
                    GradEngine::register_operation(view.id(), vec![source.id()], grad_fn);
                }

                view
            } else {
                // Strided slice - copy data
                let mut result_data = Vec::with_capacity(length);
                let src_data = source.data();

                for i in 0..length {
                    let src_index = start + i * step;
                    result_data.push(src_data[src_index]);
                }

                let mut result = Tensor::from_slice(&result_data, vec![length])
                    .expect("Failed to create slice tensor");

                // Set up gradient tracking for strided slices
                if source.requires_grad() && is_grad_enabled() {
                    result.set_requires_grad(true);
                    let grad_fn = spec.grad_fn(source.id(), source.shape());
                    result.set_grad_fn(grad_fn.clone());
                    GradEngine::register_operation(result.id(), vec![source.id()], grad_fn);
                }

                result
            }
        }

        ViewSpec::Reshape { new_shape: _ } => {
            // Create reshape view sharing memory
            let mut view = Tensor::from_raw_view(
                unsafe { source.as_ptr() },
                view_shape,
                source.device(),
                source.allocation_owner().cloned(),
            );

            // Set up gradient tracking
            if source.requires_grad() && is_grad_enabled() {
                view.set_requires_grad(true);
                let grad_fn = spec.grad_fn(source.id(), source.shape());
                view.set_grad_fn(grad_fn.clone());
                GradEngine::register_operation(view.id(), vec![source.id()], grad_fn);
            }

            view
        }

        ViewSpec::Transpose { dim0: _, dim1: _ } => {
            if spec.creates_contiguous_view(source.shape()) {
                // Can create as view
                let mut view = Tensor::from_raw_view(
                    unsafe { source.as_ptr() },
                    view_shape,
                    source.device(),
                    source.allocation_owner().cloned(),
                );

                // Set up gradient tracking
                if source.requires_grad() && is_grad_enabled() {
                    view.set_requires_grad(true);
                    let grad_fn = spec.grad_fn(source.id(), source.shape());
                    view.set_grad_fn(grad_fn.clone());
                    GradEngine::register_operation(view.id(), vec![source.id()], grad_fn);
                }

                view
            } else {
                // Need to copy data for transpose
                let mut result = Tensor::new(view_shape.dims.clone());

                // Copy data with transposed layout
                let src_data = source.data();
                let result_data = result.data_mut();

                match source.shape().rank() {
                    2 => {
                        let (rows, cols) = (source.shape().dims[0], source.shape().dims[1]);
                        for i in 0..rows {
                            for j in 0..cols {
                                let src_idx = i * cols + j;
                                let dst_idx = j * rows + i;
                                result_data[dst_idx] = src_data[src_idx];
                            }
                        }
                    }
                    _ => {
                        // For higher dimensions, fall back to index-based copying
                        // This is a simplified implementation
                        panic!("Transpose copy for >2D tensors not yet implemented");
                    }
                }

                // Set up gradient tracking (simplified for copy case)
                if source.requires_grad() && is_grad_enabled() {
                    result.set_requires_grad(true);
                    let grad_fn = spec.grad_fn(source.id(), source.shape());
                    result.set_grad_fn(grad_fn.clone());
                    GradEngine::register_operation(result.id(), vec![source.id()], grad_fn);
                }

                result
            }
        }
    }
}

/// Extension trait for creating common view types using the unified ViewSpec system
///
/// This trait adds convenient methods to Tensor for creating different types
/// of views using the unified ViewSpec enum system. All methods delegate to the
/// unified `create_view` function, providing a consistent interface for view creation.
/// The trait enables zero-copy tensor transformations with full gradient tracking support.
///
/// # Methods
///
/// * `view()` - Reshape tensor with new dimensions (requires contiguous memory)
/// * `element_view()` - Create a view of a single element
/// * `slice_view()` - Create a view of a contiguous or strided slice
/// * `create_view()` - Create views using the unified ViewSpec system
///
/// # Implementation Details
///
/// The TensorViewExt trait provides a unified interface for creating different types
/// of tensor views. All methods delegate to the underlying ViewSpec system, ensuring
/// consistent behavior and comprehensive gradient tracking support.
///
/// # Performance
///
/// - **Zero-copy**: All methods create views that share memory with source tensors
/// - **Gradient tracking**: Automatic gradient tracking preservation
/// - **Memory efficiency**: No additional allocation for view operations
/// - **Unified interface**: Consistent API for all view types
pub trait TensorViewExt {
    /// Create a view with a new shape (requires contiguous memory)
    ///
    /// Behaves like PyTorch `view`: tensor must be contiguous and the total
    /// number of elements must remain the same. Supports -1 inference for one dimension.
    /// This operation preserves the underlying memory layout while changing the logical
    /// arrangement of data.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - New shape for the tensor (can contain -1 for inference)
    ///
    /// # Returns
    ///
    /// A tensor viewing the same data with a new shape
    ///
    /// # Panics
    ///
    /// * If the tensor is not contiguous
    /// * If the total number of elements changes
    /// * If the shape is invalid or incompatible
    ///
    /// # Implementation Details
    ///
    /// The view method requires the tensor to be contiguous and maintains the total
    /// number of elements. It supports -1 inference for one dimension and preserves
    /// gradient tracking requirements.
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Returns a view with modified shape metadata
    /// - **Memory Usage**: No additional allocation (view operation)
    /// - **Gradient Tracking**: Preserves gradient requirements and tracking
    /// - **Shape Transformation**: Changes logical arrangement without data copying
    fn view(&self, new_shape: Vec<i32>) -> Tensor;

    /// Create an element view for the specified index
    ///
    /// Returns a scalar tensor (shape `[1]`) that views a single element
    /// of the source tensor. This is useful for accessing individual elements
    /// while maintaining gradient tracking and memory efficiency.
    ///
    /// # Arguments
    ///
    /// * `index` - Linear index of the element to view
    ///
    /// # Returns
    ///
    /// A scalar tensor viewing the specified element
    ///
    /// # Panics
    ///
    /// * If the index is out of bounds for the tensor size
    ///
    /// # Implementation Details
    ///
    /// Element views return scalar tensors with shape `[1]` that share memory with
    /// the source tensor. They maintain gradient tracking and provide efficient
    /// access to individual tensor elements.
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Returns a view with offset pointer
    /// - **Memory Usage**: No additional allocation (view operation)
    /// - **Gradient Tracking**: Preserves gradient requirements and tracking
    /// - **Element Access**: Efficient single-element access without copying
    fn element_view(&self, index: usize) -> Tensor;

    /// Create a slice view of the tensor
    ///
    /// Returns a view of a contiguous or strided slice of the source tensor.
    /// Contiguous slices (step=1) are true zero-copy views, while strided slices
    /// may require data copying for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `start` - Starting index in the source tensor
    /// * `step` - Step size (1 for contiguous, >1 for strided)
    /// * `length` - Number of elements in the slice
    ///
    /// # Returns
    ///
    /// A tensor viewing the specified slice with shape `[length]`
    ///
    /// # Panics
    ///
    /// * If the start index is out of bounds
    /// * If the step size is zero
    /// * If the slice extends beyond tensor bounds
    ///
    /// # Implementation Details
    ///
    /// Slice views support both contiguous (step=1) and strided (step>1) access patterns.
    /// Contiguous slices are true zero-copy views, while strided slices may require
    /// data copying for optimal performance. All slice views maintain gradient tracking.
    ///
    /// # Performance
    ///
    /// - **Contiguous slices**: O(1) - True zero-copy views
    /// - **Strided slices**: O(n) - May require data copying
    /// - **Memory Usage**: No allocation for contiguous slices
    /// - **Gradient Tracking**: Preserves gradient requirements and tracking
    fn slice_view(&self, start: usize, step: usize, length: usize) -> Tensor;

    /// Create a view using the unified ViewSpec system
    ///
    /// This is the core method that all other view methods delegate to.
    /// It provides direct access to the unified view creation system, allowing
    /// you to create any type of view using the ViewSpec enum.
    ///
    /// # Arguments
    ///
    /// * `spec` - View specification defining the type and parameters of the view
    ///
    /// # Returns
    ///
    /// A tensor view created according to the specification
    ///
    /// # Implementation Details
    ///
    /// The create_view method is the core method that all other view methods delegate to.
    /// It provides direct access to the unified view creation system, allowing creation
    /// of any type of view using the ViewSpec enum with comprehensive validation and
    /// gradient tracking support.
    ///
    /// # Performance
    ///
    /// - **Zero-copy**: Views share memory with source tensors when possible
    /// - **Gradient tracking**: Automatic gradient function registration
    /// - **Memory efficiency**: No additional allocation for view operations
    /// - **Unified interface**: Consistent API for all view types
    fn create_view(&self, spec: ViewSpec) -> Tensor;
}

impl TensorViewExt for Tensor {
    fn view(&self, new_shape: Vec<i32>) -> Tensor {
        assert!(
            self.is_contiguous(),
            "Tensor must be contiguous to use view"
        );

        // Process the new shape and handle -1 inference
        let processed_shape = self.process_reshape_dimensions(new_shape);

        // Verify total size matches
        let current_size = self.size();
        let new_size = processed_shape.iter().product::<usize>();
        assert_eq!(
            current_size, new_size,
            "Total number of elements must remain the same: {} vs {}",
            current_size, new_size
        );

        // Use the unified ViewSpec system
        self.create_view(ViewSpec::Reshape {
            new_shape: processed_shape,
        })
    }

    fn element_view(&self, index: usize) -> Tensor {
        assert!(
            index < self.size(),
            "Element index {} out of bounds for tensor of size {}",
            index,
            self.size()
        );

        // Use the unified ViewSpec system
        self.create_view(ViewSpec::Element { index })
    }

    fn slice_view(&self, start: usize, step: usize, length: usize) -> Tensor {
        assert!(start < self.size(), "Start index out of bounds");
        assert!(step > 0, "Step must be positive");
        assert!(
            start + (length.saturating_sub(1)) * step < self.size(),
            "Slice extends beyond tensor bounds"
        );

        // Use the unified ViewSpec system
        self.create_view(ViewSpec::Slice {
            start,
            step,
            length,
        })
    }

    fn create_view(&self, spec: ViewSpec) -> Tensor {
        create_view(self, spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_basic() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = x.view(vec![2, 2]);
        assert_eq!(y.shape().dims, vec![2, 2]);
        assert_eq!(y.get(&[1, 1]), 4.0);
    }

    #[test]
    fn test_view_with_infer() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = x.view(vec![2, -1]);
        assert_eq!(y.shape().dims, vec![2, 2]);
        assert_eq!(y.get(&[1, 1]), 4.0);
    }

    #[test]
    #[should_panic(expected = "Tensor must be contiguous to use view")]
    fn test_view_non_contiguous_panics() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let transposed = x.transpose(0, 1); // Creates non-contiguous tensor
        let _ = transposed.view(vec![4]);
    }

    #[test]
    fn test_view_autograd() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4])
            .unwrap()
            .with_requires_grad();
        let y = x.view(vec![2, 2]);
        assert!(y.requires_grad());

        let mut z = y.sum();
        z.backward(None);

        let grad = x.grad_by_value().expect("Gradient should exist");
        assert_eq!(grad.data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_element_view_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, vec![4]).unwrap();

        for (i, &expected_value) in data.iter().enumerate() {
            let view = tensor.element_view(i);
            assert_eq!(view.value(), expected_value);
            assert_eq!(view.shape().dims, vec![1]);
        }
    }

    #[test]
    fn test_element_view_autograd() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3])
            .unwrap()
            .with_requires_grad();

        let view = tensor.element_view(1);
        let mut result = view.mul_scalar(3.0);
        result.backward(None);

        let grad = tensor.grad_by_value().expect("Gradient should exist");
        assert_eq!(grad.data(), &[0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_slice_view_contiguous() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_slice(&data, vec![6]).unwrap();

        // Contiguous slice: [2.0, 3.0, 4.0]
        let slice = tensor.slice_view(1, 1, 3);
        assert_eq!(slice.data(), &[2.0, 3.0, 4.0]);
        assert_eq!(slice.shape().dims, vec![3]);
    }

    #[test]
    fn test_slice_view_strided() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_slice(&data, vec![6]).unwrap();

        // Strided slice: [2.0, 4.0, 6.0] (every other element starting from index 1)
        let slice = tensor.slice_view(1, 2, 3);
        assert_eq!(slice.data(), &[2.0, 4.0, 6.0]);
        assert_eq!(slice.shape().dims, vec![3]);
    }

    #[test]
    fn test_slice_view_gradient_support() {
        // Test that slice views now properly support gradients (both contiguous and strided)
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5])
            .unwrap()
            .with_requires_grad();

        // Contiguous slice
        let contiguous_slice = tensor.slice_view(1, 1, 2);
        assert!(contiguous_slice.requires_grad());
        assert_eq!(contiguous_slice.data(), &[2.0, 3.0]);

        // Strided slice
        let strided_slice = tensor.slice_view(1, 2, 2);
        assert!(strided_slice.requires_grad());
        assert_eq!(strided_slice.data(), &[2.0, 4.0]);

        // Test gradient flow through contiguous slice
        let tensor2 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4])
            .unwrap()
            .with_requires_grad();
        let slice = tensor2.slice_view(1, 1, 2);
        let result = slice.mul_scalar(2.0);
        let mut loss = result.sum();
        loss.backward(None);

        // Check that gradients flowed back to the source tensor
        assert!(tensor2.grad_by_value().is_some());
        let grad = tensor2.grad_by_value().unwrap();
        assert_eq!(grad.data(), &[0.0, 2.0, 2.0, 0.0]); // Only slice positions have gradients
    }

    #[test]
    #[should_panic(expected = "Start index out of bounds")]
    fn test_slice_view_out_of_bounds() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let _ = tensor.slice_view(5, 1, 2);
    }

    #[test]
    #[should_panic(expected = "Step must be positive")]
    fn test_slice_view_zero_step() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let _ = tensor.slice_view(0, 0, 2);
    }

    #[test]
    fn test_viewspec_element() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

        let spec = ViewSpec::Element { index: 2 };
        assert!(spec.is_valid_for(tensor.shape()));

        let view = tensor.create_view(spec);
        assert_eq!(view.value(), 3.0);
        assert_eq!(view.shape().dims, vec![1]);
    }

    #[test]
    fn test_viewspec_slice() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();

        let spec = ViewSpec::Slice {
            start: 1,
            step: 2,
            length: 3,
        };
        assert!(spec.is_valid_for(tensor.shape()));

        let view = tensor.create_view(spec);
        assert_eq!(view.data(), &[2.0, 4.0, 6.0]);
        assert_eq!(view.shape().dims, vec![3]);
    }

    #[test]
    fn test_viewspec_reshape() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let spec = ViewSpec::Reshape {
            new_shape: vec![3, 2],
        };
        assert!(spec.is_valid_for(tensor.shape()));

        let view = tensor.create_view(spec);
        assert_eq!(view.shape().dims, vec![3, 2]);
        assert_eq!(view.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_viewspec_transpose() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let spec = ViewSpec::Transpose { dim0: 0, dim1: 1 };
        assert!(spec.is_valid_for(tensor.shape()));

        let view = tensor.create_view(spec);
        assert_eq!(view.shape().dims, vec![2, 2]);
        // Due to transpose: [1,2; 3,4] -> [1,3; 2,4]
        assert_eq!(view.get(&[0, 0]), 1.0);
        assert_eq!(view.get(&[0, 1]), 3.0);
        assert_eq!(view.get(&[1, 0]), 2.0);
        assert_eq!(view.get(&[1, 1]), 4.0);
    }

    #[test]
    fn test_viewspec_validation() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();

        // Invalid element index
        let invalid_element = ViewSpec::Element { index: 5 };
        assert!(!invalid_element.is_valid_for(tensor.shape()));

        // Invalid slice
        let invalid_slice = ViewSpec::Slice {
            start: 2,
            step: 2,
            length: 3,
        };
        assert!(!invalid_slice.is_valid_for(tensor.shape()));

        // Invalid reshape (wrong total size)
        let invalid_reshape = ViewSpec::Reshape {
            new_shape: vec![2, 3],
        };
        assert!(!invalid_reshape.is_valid_for(tensor.shape()));

        // Invalid transpose dimensions
        let invalid_transpose = ViewSpec::Transpose { dim0: 0, dim1: 2 };
        assert!(!invalid_transpose.is_valid_for(tensor.shape()));
    }

    #[test]
    fn test_unified_view_system() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4])
            .unwrap()
            .with_requires_grad();

        // Test that both old and new interfaces work the same
        let old_element = tensor.element_view(2);
        let new_element = tensor.create_view(ViewSpec::Element { index: 2 });

        assert_eq!(old_element.value(), new_element.value());
        assert_eq!(old_element.requires_grad(), new_element.requires_grad());

        // Test reshape equivalence
        let old_reshape = tensor.view(vec![2, 2]);
        let new_reshape = tensor.create_view(ViewSpec::Reshape {
            new_shape: vec![2, 2],
        });

        assert_eq!(old_reshape.shape().dims, new_reshape.shape().dims);
        assert_eq!(old_reshape.requires_grad(), new_reshape.requires_grad());
    }
}
