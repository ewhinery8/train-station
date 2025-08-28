//! Tensor dimension permutation operations
//!
//! This module provides tensor permutation functionality that rearranges the
//! dimensions of a tensor according to a specified order. Permutation is a
//! fundamental tensor transformation operation used in machine learning for
//! reordering tensor axes, preparing data for specific operations, and
//! implementing complex tensor manipulations.
//!
//! # Operations
//!
//! * `permute()` - Rearrange tensor dimensions according to specified order
//!
//! # Performance Characteristics
//!
//! * **Zero-Copy Operation**: Returns a view with reordered strides, avoiding data copying
//! * **Memory Efficient**: Reuses existing tensor data through stride manipulation
//! * **Gradient Tracking**: Full GradTrack support for automatic differentiation
//! * **Shape Transformation**: Changes dimension order while preserving total elements
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Permute 2D tensor dimensions
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//! let permuted = tensor.permute(vec![1, 0]);
//! assert_eq!(permuted.shape().dims, vec![3, 2]);
//! assert_eq!(permuted.get(&[0, 0]), 1.0);
//! assert_eq!(permuted.get(&[1, 0]), 2.0);
//! assert_eq!(permuted.get(&[2, 1]), 6.0);
//! ```
//!
//! ```
//! use train_station::Tensor;
//!
//! // Permute 3D tensor dimensions
//! let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
//! let tensor = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();
//! let permuted = tensor.permute(vec![2, 0, 1]);
//! assert_eq!(permuted.shape().dims, vec![4, 2, 3]);
//! ```
//!
//! # Gradient Tracking
//!
//! The permute operation supports automatic gradient tracking through
//! the GradTrack system. When `requires_grad` is enabled, the operation
//! registers a gradient function that applies the inverse permutation
//! during backward passes.

use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;
use crate::tensor::Shape;

impl Tensor {
    /// Permute tensor dimensions according to specified order
    ///
    /// Rearranges the dimensions of the tensor according to the provided
    /// dimension order. This operation returns a view with reordered strides,
    /// avoiding data copying while changing the logical arrangement of the
    /// tensor's dimensions.
    ///
    /// The permutation is specified as a vector where each element represents
    /// the new position of the corresponding dimension from the original tensor.
    /// For example, `permute(vec![1, 0])` swaps the first two dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector specifying the new order of dimensions (must have length equal to tensor rank)
    ///
    /// # Returns
    ///
    /// A new tensor view with rearranged dimensions and correspondingly
    /// adjusted strides. The total number of elements remains unchanged.
    ///
    /// # Panics
    ///
    /// * If `dims` length does not equal the tensor rank
    /// * If any dimension index is out of bounds for the tensor rank
    /// * If `dims` contains duplicate dimension indices
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Permute 2D tensor (swap dimensions)
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let permuted = tensor.permute(vec![1, 0]);
    /// assert_eq!(permuted.shape().dims, vec![3, 2]);
    /// assert_eq!(permuted.get(&[0, 0]), 1.0);
    /// assert_eq!(permuted.get(&[1, 0]), 2.0);
    /// assert_eq!(permuted.get(&[2, 1]), 6.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Permute 3D tensor (reorder dimensions)
    /// let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    /// let tensor = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();
    /// let permuted = tensor.permute(vec![2, 0, 1]);
    /// assert_eq!(permuted.shape().dims, vec![4, 2, 3]);
    /// assert_eq!(permuted.size(), 24); // Total elements unchanged
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Permute with gradient tracking
    /// let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// tensor.set_requires_grad(true);
    ///
    /// let permuted = tensor.permute(vec![1, 0]);
    /// assert!(permuted.requires_grad());
    /// assert_eq!(permuted.shape().dims, vec![2, 2]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Identity permutation (no change)
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let permuted = tensor.permute(vec![0, 1]);
    /// assert_eq!(permuted.shape().dims, vec![2, 2]);
    /// assert_eq!(permuted.get(&[0, 0]), 1.0);
    /// assert_eq!(permuted.get(&[1, 1]), 4.0);
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Returns a view with reordered strides
    /// - **Memory Usage**: No additional memory allocation (view operation)
    /// - **Gradient Tracking**: Preserves gradient requirements and tracking
    ///
    /// # Relationship to Other Operations
    ///
    /// This operation is similar to `transpose()` but more general:
    /// - `transpose(dim0, dim1)` is equivalent to `permute()` with a swap of two dimensions
    /// - `permute()` can handle arbitrary dimension reordering for tensors of any rank
    ///
    /// # Memory Layout
    ///
    /// The permuted tensor maintains the same underlying data but with
    /// reordered strides. This means the tensor becomes non-contiguous
    /// unless the permutation is the identity permutation.
    pub fn permute(&self, dims: Vec<usize>) -> Tensor {
        let rank = self.shape().rank();
        assert_eq!(
            dims.len(),
            rank,
            "permute order must have length equal to rank"
        );

        // Validate dims has all unique values in range
        {
            let mut seen = vec![false; rank];
            for &d in &dims {
                assert!(
                    d < rank,
                    "permute index {} out of bounds for rank {}",
                    d,
                    rank
                );
                assert!(!seen[d], "duplicate dimension {} in permute", d);
                seen[d] = true;
            }
        }

        // Compute new dims and strides for view
        let mut new_dims = Vec::with_capacity(rank);
        for &d in &dims {
            new_dims.push(self.shape().dims[d]);
        }
        // Reorder strides accordingly
        let mut new_strides = Vec::with_capacity(rank);
        for &d in &dims {
            new_strides.push(self.stride(d));
        }

        // Create a non-copy view with strided layout
        let view_shape = Shape::as_view(new_dims, new_strides);
        let mut result = self.create_view_with_shape(view_shape);

        // GradTrack: register permute for backward (inverse permutation)
        if self.requires_grad() {
            result.set_requires_grad(true);
            let grad_fn = GradFn::Permute {
                dims: dims.clone(),
                input_shape: self.shape().dims.clone(),
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permute_basic_2d() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let y = x.permute(vec![1, 0]);
        assert_eq!(y.shape().dims, vec![3, 2]);
        assert_eq!(y.get(&[0, 0]), 1.0);
        assert_eq!(y.get(&[1, 0]), 2.0);
        assert_eq!(y.get(&[2, 1]), 6.0);
    }
}
