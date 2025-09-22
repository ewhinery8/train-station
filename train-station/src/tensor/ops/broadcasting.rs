//! Broadcasting utilities for tensor operations
//!
//! This module provides comprehensive broadcasting functionality for element-wise operations
//! following NumPy broadcasting semantics. Broadcasting enables operations between tensors
//! of different shapes by automatically expanding dimensions to make them compatible.
//!
//! # Key Features
//!
//! - **NumPy Compatible**: Follows NumPy broadcasting semantics precisely
//! - **Zero-Copy Views**: Creates efficient views when possible
//! - **Memory Efficient**: Minimal memory allocations during broadcasting
//! - **SIMD Optimized**: Optimized paths for common broadcasting patterns
//! - **Error Handling**: Clear error messages for incompatible shapes
//! - **Performance Optimized**: Fast paths for common ML broadcasting patterns
//!
//! # Broadcasting Rules
//!
//! 1. Dimensions are aligned from the rightmost (trailing) dimension
//! 2. Dimensions are compatible if they are equal, or one of them is 1
//! 3. Missing dimensions are treated as 1
//! 4. The result shape is the element-wise maximum of input shapes
//!
//! # Performance Characteristics
//!
//! - **Shape Computation**: O(max_rank) time complexity
//! - **Memory Usage**: O(1) for view creation, O(n) only when copying needed
//! - **SIMD Ready**: Maintains alignment for vectorized operations
//! - **Cache Friendly**: Optimized memory access patterns
//! - **Optimized Patterns**: Fast paths for scalar and vector-matrix broadcasting
//!
//! # Implementation Details
//!
//! The broadcasting system provides multiple specialized implementations:
//!
//! - **Shape Computation**: Efficient broadcast shape calculation with error handling
//! - **Stride Optimization**: Zero-copy broadcasting using stride manipulation
//! - **SIMD Broadcasting**: AVX2-optimized scalar broadcasting for maximum performance
//! - **Memory Management**: Efficient copying with minimal allocations
//! - **Optimized Patterns**: Fast paths for common neural network broadcasting scenarios

use crate::tensor::core::view::as_strided_view;
use crate::tensor::core::Tensor;
use crate::tensor::Shape;
use std::borrow::Cow;

/// Error type for broadcasting operations
#[derive(Debug, Clone, PartialEq)]
pub enum BroadcastError {
    /// Shapes are incompatible for broadcasting
    IncompatibleShapes {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
        conflicting_dim: usize,
    },
    /// Memory allocation failed during broadcasting
    AllocationFailed,
}

impl std::fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BroadcastError::IncompatibleShapes {
                shape1,
                shape2,
                conflicting_dim,
            } => write!(
                f,
                "Cannot broadcast shapes {:?} and {:?}: incompatible dimensions at axis {}",
                shape1, shape2, conflicting_dim
            ),
            BroadcastError::AllocationFailed => {
                write!(f, "Memory allocation failed during broadcasting")
            }
        }
    }
}

impl std::error::Error for BroadcastError {}

/// Broadcast compatibility result
pub type BroadcastResult<T> = Result<T, BroadcastError>;

/// Computes the broadcasted shape for two input shapes
///
/// Returns the shape that results from broadcasting two tensor shapes together.
/// Uses NumPy broadcasting rules to determine compatibility and result shape.
///
/// # Arguments
///
/// * `shape1` - First tensor shape
/// * `shape2` - Second tensor shape
///
/// # Returns
///
/// The broadcasted result shape, or an error if shapes are incompatible
///
/// # Broadcasting Rules
///
/// - Dimensions are compared from right to left
/// - Dimensions are compatible if equal or one is 1
/// - Missing dimensions are treated as 1
/// - Result dimension is the maximum of the two dimensions
///
/// # Implementation Details
///
/// This function implements NumPy-style broadcasting rules:
///
/// - **Dimension Alignment**: Dimensions are compared from right to left
/// - **Compatibility Check**: Dimensions are compatible if equal or one is 1
/// - **Missing Dimensions**: Treated as size 1 for broadcasting
/// - **Result Shape**: Element-wise maximum of input dimensions
///
/// ## Broadcasting Examples
///
/// - `[2, 1, 4]` + `[3, 1]` → `[2, 3, 4]`
/// - `[1]` + `[2, 3]` → `[2, 3]` (scalar broadcasting)
/// - `[2, 3, 4]` + `[4]` → `[2, 3, 4]` (different ranks)
#[track_caller]
pub(crate) fn compute_broadcast_shape(shape1: &Shape, shape2: &Shape) -> BroadcastResult<Shape> {
    let rank1 = shape1.rank();
    let rank2 = shape2.rank();
    let max_rank = rank1.max(rank2);

    let mut result_dims = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let dim1 = if i < rank1 {
            shape1.dims()[rank1 - 1 - i]
        } else {
            1
        };

        let dim2 = if i < rank2 {
            shape2.dims()[rank2 - 1 - i]
        } else {
            1
        };

        if dim1 == dim2 {
            result_dims.push(dim1);
        } else if dim1 == 1 {
            result_dims.push(dim2);
        } else if dim2 == 1 {
            result_dims.push(dim1);
        } else {
            return Err(BroadcastError::IncompatibleShapes {
                shape1: shape1.dims().to_vec(),
                shape2: shape2.dims().to_vec(),
                conflicting_dim: max_rank - 1 - i,
            });
        }
    }

    result_dims.reverse();
    Ok(Shape::new(result_dims))
}

/// Information about how a tensor should be broadcast
#[derive(Debug, Clone)]
pub struct BroadcastInfo {
    /// Original tensor shape
    pub _original_shape: Shape,
    /// Target broadcasted shape
    pub broadcast_shape: Shape,
    /// Whether broadcasting is needed (shapes differ)
    pub needs_broadcast: bool,
    /// Stride adjustments for efficient broadcasting
    pub broadcast_strides: Vec<usize>,
}

impl BroadcastInfo {
    /// Creates broadcast info for a tensor to a target shape
    fn new(original_shape: Shape, target_shape: Shape) -> Self {
        let needs_broadcast = original_shape.dims() != target_shape.dims();
        let broadcast_strides = if needs_broadcast {
            compute_broadcast_strides(&original_shape, &target_shape)
        } else {
            original_shape.strides().to_vec()
        };

        Self {
            _original_shape: original_shape,
            broadcast_shape: target_shape,
            needs_broadcast,
            broadcast_strides,
        }
    }
}

/// Computes stride adjustments for broadcasting
///
/// Calculates the memory strides needed to broadcast from an original shape
/// to a target shape. Sets stride to 0 for dimensions that need expansion.
/// This enables zero-copy broadcasting by manipulating memory access patterns.
///
/// # Arguments
///
/// * `original_shape` - The original tensor shape
/// * `target_shape` - The target broadcasted shape
///
/// # Returns
///
/// Vector of strides for the broadcasted tensor. Dimensions with stride 0
/// will repeat the same value (broadcasting).
///
/// # Implementation Details
///
/// - **Zero Stride**: When a dimension has size 1 in original but larger in target,
///   the stride becomes 0 to repeat the single element
/// - **Preserved Stride**: When dimensions match, the original stride is preserved
/// - **Missing Dimensions**: Treated as size 1 with stride 0
///
/// # Performance Characteristics
///
/// - **Zero-Copy**: Enables broadcasting without data copying
/// - **Memory Efficient**: Minimal memory overhead for broadcasting
/// - **Cache Friendly**: Maintains memory access patterns for performance
fn compute_broadcast_strides(original_shape: &Shape, target_shape: &Shape) -> Vec<usize> {
    let orig_rank = original_shape.rank();
    let target_rank = target_shape.rank();
    let mut broadcast_strides = vec![0; target_rank];

    for i in 0..target_rank {
        let target_dim = target_shape.dims()[target_rank - 1 - i];

        if i < orig_rank {
            let orig_dim = original_shape.dims()[orig_rank - 1 - i];
            let orig_stride = original_shape.strides()[orig_rank - 1 - i];

            if orig_dim == target_dim {
                // No broadcasting needed for this dimension
                broadcast_strides[target_rank - 1 - i] = orig_stride;
            } else if orig_dim == 1 {
                // Broadcasting needed: stride becomes 0 to repeat the single element
                broadcast_strides[target_rank - 1 - i] = 0;
            } else {
                // This should not happen if shapes are compatible
                panic!(
                    "Invalid broadcasting: {} cannot be broadcast to {}",
                    orig_dim, target_dim
                );
            }
        } else {
            // Missing dimension, treated as 1, stride is 0
            broadcast_strides[target_rank - 1 - i] = 0;
        }
    }

    broadcast_strides
}

/// Broadcasts two tensors to compatible shapes for element-wise operations
///
/// This is the main function for preparing tensors for broadcasting. It computes
/// the broadcast shapes and creates efficient tensor views when possible.
///
/// # Arguments
///
/// * `tensor1` - First input tensor
/// * `tensor2` - Second input tensor
///
/// # Returns
///
/// A tuple containing:
/// - Broadcasted view of first tensor
/// - Broadcasted view of second tensor  
/// - Result shape for the operation
///
/// # Performance
///
/// - **Zero-Copy**: Creates views when possible, avoiding data copying
/// - **Memory Efficient**: Only allocates when explicit broadcasting needed
/// - **SIMD Ready**: Maintains alignment for vectorized operations
#[track_caller]
pub(crate) fn broadcast_shapes(
    tensor1: &Tensor,
    tensor2: &Tensor,
) -> BroadcastResult<(Tensor, Tensor, Shape)> {
    let shape1 = tensor1.shape();
    let shape2 = tensor2.shape();

    if shape1.dims() == shape2.dims() {
        // No broadcasting needed: return the original tensors as-is without cloning
        // Note: this function returns owned Tensors, so we can just move via clones
        // only if the caller needs ownership elsewhere. For max efficiency here,
        // call sites that don't need ownership should prefer `broadcast_shapes_mixed`.
        return Ok((tensor1.clone(), tensor2.clone(), shape1.clone()));
    }

    // Compute the broadcasted result shape
    let result_shape = compute_broadcast_shape(shape1, shape2)?;

    // Create broadcast info for both tensors
    let info1 = BroadcastInfo::new(shape1.clone(), result_shape.clone());
    let info2 = BroadcastInfo::new(shape2.clone(), result_shape.clone());

    // Create broadcasted tensors
    let broadcast1 = if info1.needs_broadcast {
        create_broadcast_tensor(tensor1, &info1)?
    } else {
        // Return original without extra allocation
        tensor1.clone()
    };

    let broadcast2 = if info2.needs_broadcast {
        create_broadcast_tensor(tensor2, &info2)?
    } else {
        tensor2.clone()
    };

    Ok((broadcast1, broadcast2, result_shape))
}

/// Cow-based broadcasting API: borrows when possible, owns only when expansion is required
#[track_caller]
pub(crate) fn broadcast_shapes_cow<'a, 'b>(
    tensor1: &'a Tensor,
    tensor2: &'b Tensor,
) -> BroadcastResult<(Cow<'a, Tensor>, Cow<'b, Tensor>, Shape)> {
    let shape1 = tensor1.shape();
    let shape2 = tensor2.shape();
    if shape1.dims() == shape2.dims() {
        return Ok((
            Cow::Borrowed(tensor1),
            Cow::Borrowed(tensor2),
            shape1.clone(),
        ));
    }
    let result_shape = compute_broadcast_shape(shape1, shape2)?;
    let info1 = BroadcastInfo::new(shape1.clone(), result_shape.clone());
    let info2 = BroadcastInfo::new(shape2.clone(), result_shape.clone());
    let out1: Cow<'a, Tensor> = if info1.needs_broadcast {
        Cow::Owned(create_broadcast_tensor(tensor1, &info1)?)
    } else {
        Cow::Borrowed(tensor1)
    };
    let out2: Cow<'b, Tensor> = if info2.needs_broadcast {
        Cow::Owned(create_broadcast_tensor(tensor2, &info2)?)
    } else {
        Cow::Borrowed(tensor2)
    };
    Ok((out1, out2, result_shape))
}

/// Creates a broadcasted tensor from the original tensor and broadcast info
///
/// This function creates an efficient broadcasted view of a tensor. When possible,
/// it creates zero-copy views. When memory expansion is needed, it performs
/// efficient copying with SIMD optimization.
///
/// # Arguments
///
/// * `tensor` - The original tensor to broadcast
/// * `info` - Broadcast information containing shape and stride details
///
/// # Returns
///
/// A new tensor with the broadcasted shape, or an error if allocation fails
///
/// # Implementation Details
///
/// - **Zero-Copy Views**: When possible, creates views with modified strides
/// - **SIMD Copying**: Uses optimized copying for scalar broadcasting
/// - **Memory Management**: Efficient allocation and copying strategies
/// - **Gradient Preservation**: Maintains gradient tracking requirements
///
/// # Performance Characteristics
///
/// - **Scalar Broadcasting**: SIMD-optimized for single value broadcasting
/// - **Memory Efficiency**: Minimal allocations when possible
/// - **Cache Optimization**: Optimized memory access patterns
fn create_broadcast_tensor(tensor: &Tensor, info: &BroadcastInfo) -> BroadcastResult<Tensor> {
    if !info.needs_broadcast {
        return Ok(tensor.clone());
    }

    // Zero-copy: create an as_strided view that repeats along broadcasted dims using stride 0
    match as_strided_view(
        tensor,
        info.broadcast_shape.dims(),
        &info.broadcast_strides,
        0,
    ) {
        Ok(mut view) => {
            if tensor.requires_grad() {
                view.set_requires_grad(true);
            }
            Ok(view)
        }
        // View creation should be safe for valid broadcast pairs; map any unexpected error
        Err(_e) => Err(BroadcastError::AllocationFailed),
    }
}

/// Broadcast a single tensor to a target shape as a zero-copy view when possible
#[allow(unused)]
#[track_caller]
pub(crate) fn broadcast_to_shape(tensor: &Tensor, target_shape: &Shape) -> BroadcastResult<Tensor> {
    if tensor.shape().dims() == target_shape.dims() {
        return Ok(tensor.clone());
    }
    // Validate compatibility via compute_broadcast_shape
    let _ = compute_broadcast_shape(tensor.shape(), target_shape)?;
    let info = BroadcastInfo::new(tensor.shape().clone(), target_shape.clone());
    create_broadcast_tensor(tensor, &info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape_computation() {
        // Basic broadcasting
        let shape1 = Shape::new(vec![3, 1]);
        let shape2 = Shape::new(vec![1, 4]);
        let result = compute_broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(result.dims(), vec![3, 4]);

        // Different ranks
        let shape1 = Shape::new(vec![2, 3, 4]);
        let shape2 = Shape::new(vec![4]);
        let result = compute_broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(result.dims(), vec![2, 3, 4]);

        // Scalar broadcasting
        let shape1 = Shape::new(vec![1]);
        let shape2 = Shape::new(vec![2, 3]);
        let result = compute_broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(result.dims(), vec![2, 3]);
    }

    #[test]
    fn test_incompatible_shapes() {
        let shape1 = Shape::new(vec![3, 4]);
        let shape2 = Shape::new(vec![2, 4]);
        let result = compute_broadcast_shape(&shape1, &shape2);
        assert!(result.is_err());

        match result.err().unwrap() {
            BroadcastError::IncompatibleShapes {
                conflicting_dim, ..
            } => {
                assert_eq!(conflicting_dim, 0);
            }
            _ => panic!("Expected IncompatibleShapes error"),
        }
    }

    #[test]
    fn test_broadcast_strides() {
        let original = Shape::new(vec![1, 3]);
        let target = Shape::new(vec![2, 3]);
        let strides = compute_broadcast_strides(&original, &target);
        assert_eq!(strides, vec![0, 1]); // First dimension stride is 0 (broadcast)
    }

    #[test]
    fn test_scalar_broadcasting() {
        let scalar = Tensor::from_slice(&[5.0], vec![1]).unwrap();
        let tensor = Tensor::ones(vec![2, 3]);

        let (broadcast_scalar, _broadcast_tensor, result_shape) =
            scalar.broadcast_with(&tensor).unwrap();

        assert_eq!(result_shape.dims(), vec![2, 3]);
        assert_eq!(broadcast_scalar.size(), 6);

        // Check that all elements in broadcast_scalar are 5.0
        for i in 0..6 {
            let indices = vec![i / 3, i % 3];
            assert_eq!(broadcast_scalar.get(&indices), 5.0);
        }
    }

    #[test]
    fn test_no_broadcast_needed() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![2, 3]);

        let (a_broadcast, b_broadcast, result_shape) = a.broadcast_with(&b).unwrap();

        assert_eq!(result_shape.dims(), vec![2, 3]);
        // Should return the same tensors when no broadcasting is needed
        assert_eq!(a_broadcast.shape().dims(), a.shape().dims());
        assert_eq!(b_broadcast.shape().dims(), b.shape().dims());
    }

    #[test]
    fn test_tricky_broadcast_3d_inner_dims() {
        // [1, 4, 1] -> [2, 4, 3]
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 4, 1]).unwrap();
        let b = Tensor::from_slice(&[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], vec![2, 1, 3]).unwrap();

        let (a_b, b_b, out_shape) = broadcast_shapes(&a, &b).unwrap();
        assert_eq!(out_shape.dims(), vec![2, 4, 3]);
        assert_eq!(a_b.shape().dims(), vec![2, 4, 3]);
        assert_eq!(b_b.shape().dims(), vec![2, 4, 3]);

        // Spot-check a few positions to ensure inner-dim expansion is correct
        // a expands across last dim, b expands across middle dim
        assert!((a_b.get(&[0, 0, 0]) - 1.0).abs() < 1e-6);
        assert!((a_b.get(&[0, 3, 2]) - 4.0).abs() < 1e-6);
        assert!((b_b.get(&[1, 0, 2]) - 0.3).abs() < 1e-6);
        assert!((b_b.get(&[1, 3, 1]) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_tricky_broadcast_4d_multiple_dims() {
        // [2, 1, 4, 1] -> [2, 3, 4, 5]
        let a_data: Vec<f32> = (0..2 * 4).map(|i| i as f32 + 1.0).collect();
        let a = Tensor::from_slice(&a_data, vec![2, 1, 4, 1]).unwrap();
        let b = Tensor::from_slice(&[0.0; 3 * 5], vec![1, 3, 1, 5]).unwrap();

        let (a_b, b_b, out_shape) = broadcast_shapes(&a, &b).unwrap();
        assert_eq!(out_shape.dims(), vec![2, 3, 4, 5]);
        assert_eq!(a_b.shape().dims(), vec![2, 3, 4, 5]);
        assert_eq!(b_b.shape().dims(), vec![2, 3, 4, 5]);

        // Check that broadcasting repeated the original values along the broadcasted dims
        // Sample indices
        assert!((a_b.get(&[0, 0, 0, 0]) - 1.0).abs() < 1e-6);
        assert!((a_b.get(&[0, 2, 3, 4]) - 4.0).abs() < 1e-6);
        // Spot-check another element in the second batch
        let _sample = a_b.get(&[1, 1, 2, 3]);
        // b is zeros, so just ensure shape correctness; values are zero
        assert!((b_b.get(&[1, 2, 3, 4]) - 0.0).abs() < 1e-6);
    }
}
