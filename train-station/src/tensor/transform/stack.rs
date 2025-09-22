//! Tensor stacking operations
//!
//! This module provides tensor stacking functionality that combines multiple
//! tensors along a new dimension. Stacking is a fundamental tensor transformation
//! operation used in machine learning for combining multiple feature maps,
//! creating batch dimensions, and implementing complex tensor manipulations
//! that require adding new axes to tensor data.
//!
//! # Operations
//!
//! * `stack()` - Stack multiple tensors along a new dimension
//!
//! # Performance Characteristics
//!
//! * **SIMD Optimized**: AVX2 acceleration for large block copies
//! * **Memory Efficient**: Optimized block-wise copying with minimal allocations
//! * **Contiguous Output**: Always produces a contiguous tensor for optimal performance
//! * **Gradient Tracking**: Full GradTrack support for automatic differentiation
//! * **Shape Validation**: Comprehensive error checking for compatible tensor shapes
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Stack two 1D tensors along dimension 0
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
//! let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]).unwrap();
//! let stacked = Tensor::stack(&[a, b], 0);
//! assert_eq!(stacked.shape().dims(), vec![2, 3]);
//! assert_eq!(stacked.get(&[0, 0]), 1.0);
//! assert_eq!(stacked.get(&[1, 2]), 6.0);
//! ```
//!
//! ```
//! use train_station::Tensor;
//!
//! // Stack multiple 2D tensors along dimension 1
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//! let c = Tensor::from_slice(&[9.0, 10.0, 11.0, 12.0], vec![2, 2]).unwrap();
//! let stacked = Tensor::stack(&[a, b, c], 1);
//! assert_eq!(stacked.shape().dims(), vec![2, 3, 2]);
//! ```

use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;
use crate::tensor::iterator::collect::optimized_copy;

impl Tensor {
    /// Stack a list of tensors along a new dimension
    ///
    /// Combines multiple tensors by adding a new dimension at the specified
    /// position. All input tensors must have identical shapes, and the output
    /// tensor will have a new dimension of size equal to the number of input
    /// tensors. This operation is similar to PyTorch's `torch.stack` function.
    ///
    /// The stacking operation creates a new axis in the output tensor, unlike
    /// concatenation which operates along existing dimensions. This makes
    /// stacking useful for creating batch dimensions, combining feature maps,
    /// and implementing operations that require adding new tensor axes.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Array of tensors to stack. All tensors must have identical shapes.
    /// * `dim` - Index of the new axis in the output shape (0 <= dim <= rank)
    ///
    /// # Returns
    ///
    /// A new tensor with the stacked data. The output shape is the input shape
    /// with a new dimension of size `tensors.len()` inserted at position `dim`.
    ///
    /// # Panics
    ///
    /// * If the tensor array is empty
    /// * If any tensor has a different shape than the first tensor
    /// * If `dim` is out of bounds (dim > rank of input tensors)
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Stack two 1D tensors along dimension 0
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]).unwrap();
    /// let stacked = Tensor::stack(&[a, b], 0);
    /// assert_eq!(stacked.shape().dims(), vec![2, 3]);
    /// assert_eq!(stacked.get(&[0, 0]), 1.0);
    /// assert_eq!(stacked.get(&[1, 2]), 6.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Stack multiple 2D tensors along dimension 1
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    /// let c = Tensor::from_slice(&[9.0, 10.0, 11.0, 12.0], vec![2, 2]).unwrap();
    /// let stacked = Tensor::stack(&[a, b, c], 1);
    /// assert_eq!(stacked.shape().dims(), vec![2, 3, 2]);
    /// assert_eq!(stacked.get(&[0, 0, 0]), 1.0);
    /// assert_eq!(stacked.get(&[1, 2, 1]), 12.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Stack with gradient tracking
    /// let mut a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
    /// let mut b = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
    /// a.set_requires_grad(true);
    /// b.set_requires_grad(true);
    ///
    /// let stacked = Tensor::stack(&[a, b], 0);
    /// assert!(stacked.requires_grad());
    /// assert_eq!(stacked.shape().dims(), vec![2, 2]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Stack 3D tensors along the last dimension
    /// let data1: Vec<f32> = (0..8).map(|i| i as f32).collect();
    /// let data2: Vec<f32> = (8..16).map(|i| i as f32).collect();
    /// let a = Tensor::from_slice(&data1, vec![2, 2, 2]).unwrap();
    /// let b = Tensor::from_slice(&data2, vec![2, 2, 2]).unwrap();
    /// let stacked = Tensor::stack(&[a, b], 3);
    /// assert_eq!(stacked.shape().dims(), vec![2, 2, 2, 2]);
    /// assert_eq!(stacked.get(&[0, 0, 0, 0]), 0.0);
    /// assert_eq!(stacked.get(&[1, 1, 1, 1]), 15.0);
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(n) where n is the total number of elements
    /// - **Memory Usage**: Allocates new contiguous tensor for output
    /// - **SIMD Optimization**: Uses AVX2 acceleration for large block copies
    /// - **Block-wise Copying**: Optimized copying strategy for better cache performance
    /// - **Gradient Tracking**: Preserves gradient requirements and tracking
    ///
    /// # Relationship to Other Operations
    ///
    /// This operation is related to other tensor transformations:
    /// - `cat()` - Concatenates tensors along existing dimensions
    /// - `unsqueeze()` - Adds a single dimension of size 1
    /// - `reshape()` - Changes tensor shape without adding dimensions
    ///
    /// # Memory Layout
    ///
    /// The output tensor is always contiguous, with elements arranged so that
    /// the stacked dimension is the fastest-changing index. This ensures optimal
    /// performance for subsequent operations and maintains compatibility with
    /// SIMD optimizations.
    ///
    /// # Gradient Computation
    ///
    /// During backward passes, gradients are split along the stacked dimension
    /// and distributed back to the original input tensors. This is implemented
    /// using the same gradient function as concatenation, treating the stack
    /// operation as concatenation along a new axis.
    #[track_caller]
    pub fn stack(tensors: &[Tensor], dim: usize) -> Tensor {
        assert!(!tensors.is_empty(), "stack requires at least one tensor");

        // Validate all shapes identical
        let base_dims = tensors[0].shape().dims();
        for t in tensors.iter() {
            assert_eq!(
                t.shape().dims(),
                base_dims,
                "All tensors must have identical shapes for stack"
            );
        }

        let rank = base_dims.len();
        assert!(
            dim <= rank,
            "stack dim {} out of bounds for rank {}",
            dim,
            rank
        );

        // Compute output shape by inserting new axis of size = tensors.len()
        let mut out_dims = Vec::with_capacity(rank + 1);
        out_dims.extend_from_slice(&base_dims[..dim]);
        out_dims.push(tensors.len());
        out_dims.extend_from_slice(&base_dims[dim..]);

        // Materialize into a new contiguous tensor
        let mut output = Tensor::new(out_dims.clone());

        // Copy block-wise: treat stack dim separately
        // For output shape [pre..., K=tensors.len(), post...]
        // inner = product(post...), outer = product(pre...)
        let inner: usize = base_dims[dim..].iter().product();
        let outer: usize = base_dims[..dim].iter().product();

        unsafe {
            let dst_ptr = output.as_mut_ptr();
            for outer_idx in 0..outer {
                for (k, t) in tensors.iter().enumerate() {
                    // Ensure contiguous source
                    let src = if t.is_contiguous() {
                        t.clone()
                    } else {
                        t.contiguous()
                    };
                    // Source offset: within each tensor, block size is inner
                    let src_base = outer_idx * inner;
                    let src_ptr = src.as_ptr().add(src_base);

                    // Destination offset computes with inserted axis
                    // out block along stacked axis of length K, each block is inner
                    let dst_base = outer_idx * (tensors.len() * inner) + k * inner;
                    optimized_copy(src_ptr, dst_ptr.add(dst_base), inner);
                }
            }
        }

        // GradTrack: stack is like cat with a new axis; gradient splits along that axis
        let any_requires = tensors.iter().any(|t| t.requires_grad());
        if any_requires {
            output.set_requires_grad(true);
            // For GradFn::Cat, provide sizes along concat dim and input shapes
            let mut input_ids = Vec::with_capacity(tensors.len());
            let mut input_sizes = Vec::with_capacity(tensors.len());
            let mut input_shapes = Vec::with_capacity(tensors.len());
            for t in tensors.iter() {
                if t.requires_grad() {
                    input_ids.push(t.id());
                }
                input_sizes.push(1); // each slice along new axis has length 1
                input_shapes.push(t.shape().dims().to_vec());
            }
            let grad_fn = GradFn::Cat {
                dim,
                input_sizes,
                input_shapes,
            };
            output.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(output.id(), input_ids, grad_fn);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_basic() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]).unwrap();
        let y = Tensor::stack(&[a, b], 0);
        assert_eq!(y.shape().dims(), vec![2, 3]);
        assert_eq!(y.get(&[0, 0]), 1.0);
        assert_eq!(y.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_stack_multiple_tensors() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
        let c = Tensor::from_slice(&[5.0, 6.0], vec![2]).unwrap();
        let stacked = Tensor::stack(&[a, b, c], 0);
        assert_eq!(stacked.shape().dims(), vec![3, 2]);
        assert_eq!(stacked.get(&[0, 0]), 1.0);
        assert_eq!(stacked.get(&[1, 1]), 4.0);
        assert_eq!(stacked.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_stack_2d_tensors() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let stacked = Tensor::stack(&[a, b], 1);
        assert_eq!(stacked.shape().dims(), vec![2, 2, 2]);
        assert_eq!(stacked.get(&[0, 0, 0]), 1.0);
        assert_eq!(stacked.get(&[1, 1, 1]), 8.0);
    }

    #[test]
    fn test_stack_with_gradients() {
        let mut a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
        let mut b = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        let stacked = Tensor::stack(&[a, b], 0);
        assert!(stacked.requires_grad());
        assert_eq!(stacked.shape().dims(), vec![2, 2]);
    }

    #[test]
    #[should_panic(expected = "stack requires at least one tensor")]
    fn test_stack_empty() {
        Tensor::stack(&[], 0);
    }

    #[test]
    #[should_panic(expected = "All tensors must have identical shapes")]
    fn test_stack_different_shapes() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[3.0, 4.0, 5.0], vec![3]).unwrap();
        Tensor::stack(&[a, b], 0);
    }

    #[test]
    #[should_panic(expected = "stack dim 2 out of bounds for rank 1")]
    fn test_stack_dim_out_of_bounds() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
        Tensor::stack(&[a, b], 2);
    }
}
