//! Tensor splitting operations
//!
//! This module provides tensor splitting functionality that divides a tensor
//! into multiple smaller tensors along a specified dimension. Splitting is a
//! fundamental tensor transformation operation used in machine learning for
//! dividing data into batches, creating multiple outputs from a single tensor,
//! and implementing complex tensor manipulations.
//!
//! # Operations
//!
//! * `split()` - Split tensor into chunks of equal size along a dimension
//! * `split_with_sizes()` - Split tensor into chunks with explicit sizes along a dimension
//!
//! # Performance Characteristics
//!
//! * **View Operations**: First chunk returns a view when possible (zero-copy)
//! * **Copy Operations**: Subsequent chunks require data copying for non-zero offsets
//! * **Memory Efficient**: Minimizes memory allocation through view reuse
//! * **Gradient Tracking**: Full GradTrack support for automatic differentiation
//! * **Shape Transformation**: Divides tensor along specified dimension while preserving other dimensions
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Split tensor into equal-sized chunks
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//! let parts = tensor.split(1, 1);
//! assert_eq!(parts.len(), 3);
//! assert_eq!(parts[0].shape().dims, vec![2, 1]);
//! assert_eq!(parts[1].shape().dims, vec![2, 1]);
//! assert_eq!(parts[2].shape().dims, vec![2, 1]);
//! ```
//!
//! ```
//! use train_station::Tensor;
//!
//! // Split tensor with explicit sizes
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
//! let parts = tensor.split_with_sizes(&[2, 3], 1);
//! assert_eq!(parts.len(), 2);
//! assert_eq!(parts[0].shape().dims, vec![1, 2]);
//! assert_eq!(parts[1].shape().dims, vec![1, 3]);
//! ```
//!
//! # Gradient Tracking
//!
//! The split operations support automatic gradient tracking through
//! the GradTrack system. When `requires_grad` is enabled, each split
//! piece registers a gradient function that scatters gradients back
//! to the original tensor during backward passes.

use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Split tensor into chunks of equal size along specified dimension
    ///
    /// Divides the tensor into multiple smaller tensors along the specified
    /// dimension, where each chunk (except possibly the last) has the same size.
    /// The last chunk may be smaller if the dimension size is not evenly
    /// divisible by the split size.
    ///
    /// This operation returns a vector of tensors, where each tensor is a
    /// view or copy of a portion of the original tensor. The first chunk
    /// is returned as a view when possible (zero-copy), while subsequent
    /// chunks may require data copying for non-zero base offsets.
    ///
    /// # Arguments
    ///
    /// * `split_size` - Size of each chunk along the specified dimension (must be > 0)
    /// * `dim` - Dimension along which to split the tensor (must be < tensor rank)
    ///
    /// # Returns
    ///
    /// A vector of tensors, each representing a chunk of the original tensor.
    /// The number of chunks depends on the dimension size and split size.
    ///
    /// # Panics
    ///
    /// * If tensor rank is 0 (scalar tensors cannot be split)
    /// * If `dim` is out of bounds for the tensor rank
    /// * If `split_size` is 0
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Split 2D tensor into equal chunks along dimension 1
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let parts = tensor.split(1, 1);
    /// assert_eq!(parts.len(), 3);
    /// assert_eq!(parts[0].shape().dims, vec![2, 1]);
    /// assert_eq!(parts[1].shape().dims, vec![2, 1]);
    /// assert_eq!(parts[2].shape().dims, vec![2, 1]);
    /// assert_eq!(parts[0].get(&[0, 0]), 1.0);
    /// assert_eq!(parts[1].get(&[0, 0]), 2.0);
    /// assert_eq!(parts[2].get(&[1, 0]), 6.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Split with uneven division (last chunk smaller)
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
    /// let parts = tensor.split(2, 1);
    /// assert_eq!(parts.len(), 3);
    /// assert_eq!(parts[0].shape().dims, vec![1, 2]);
    /// assert_eq!(parts[1].shape().dims, vec![1, 2]);
    /// assert_eq!(parts[2].shape().dims, vec![1, 1]); // Last chunk smaller
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Split with gradient tracking
    /// let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// tensor.set_requires_grad(true);
    ///
    /// let parts = tensor.split(1, 1);
    /// assert_eq!(parts.len(), 2);
    /// assert!(parts[0].requires_grad());
    /// assert!(parts[1].requires_grad());
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Split 1D tensor
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();
    /// let parts = tensor.split(2, 0);
    /// assert_eq!(parts.len(), 3);
    /// assert_eq!(parts[0].shape().dims, vec![2]);
    /// assert_eq!(parts[1].shape().dims, vec![2]);
    /// assert_eq!(parts[2].shape().dims, vec![2]);
    /// ```
    ///
    /// # Performance
    ///
    /// - **First Chunk**: O(1) - Returns a view when possible (zero-copy)
    /// - **Subsequent Chunks**: O(n) - May require data copying for non-zero offsets
    /// - **Memory Usage**: Minimal allocation for view operations, copying for non-zero offsets
    /// - **Gradient Tracking**: Each chunk preserves gradient requirements and tracking
    ///
    /// # Relationship to Other Operations
    ///
    /// This operation is related to other tensor transformations:
    /// - `split_with_sizes()` - More general version with explicit chunk sizes
    /// - `cat()` - Inverse operation that concatenates tensors back together
    /// - `chunk()` - Alternative splitting operation with different semantics
    ///
    /// # Memory Layout
    ///
    /// The first chunk maintains the same underlying data as a view when
    /// the base offset is zero. Subsequent chunks may require data copying
    /// to handle non-zero base offsets, ensuring proper memory layout.
    pub fn split(&self, split_size: usize, dim: usize) -> Vec<Tensor> {
        assert!(self.shape().rank() > 0, "split requires non-zero rank");
        assert!(
            dim < self.shape().rank(),
            "split dim {} out of bounds for rank {}",
            dim,
            self.shape().rank()
        );
        assert!(split_size > 0, "split_size must be > 0");
        let dim_size = self.shape().dims[dim];
        if dim_size == 0 {
            return vec![];
        }

        let mut sizes = Vec::new();
        let mut remaining = dim_size;
        while remaining > 0 {
            let len = remaining.min(split_size);
            sizes.push(len);
            remaining -= len;
        }
        self.split_with_sizes(&sizes, dim)
    }

    /// Split tensor into chunks with explicit sizes along specified dimension
    ///
    /// Divides the tensor into multiple smaller tensors along the specified
    /// dimension according to the provided size specifications. Each chunk
    /// has the exact size specified in the `split_sizes` array, and the sum
    /// of all sizes must equal the size of the specified dimension.
    ///
    /// This operation provides precise control over the size of each resulting
    /// chunk, unlike `split()` which creates equal-sized chunks. The first
    /// chunk is returned as a view when possible (zero-copy), while subsequent
    /// chunks may require data copying for non-zero base offsets.
    ///
    /// # Arguments
    ///
    /// * `split_sizes` - Array specifying the size of each chunk along the dimension
    /// * `dim` - Dimension along which to split the tensor (must be < tensor rank)
    ///
    /// # Returns
    ///
    /// A vector of tensors, each representing a chunk of the original tensor
    /// with the specified size. The number of chunks equals the length of `split_sizes`.
    ///
    /// # Panics
    ///
    /// * If tensor rank is 0 (scalar tensors cannot be split)
    /// * If `dim` is out of bounds for the tensor rank
    /// * If sum of `split_sizes` does not equal the size of the specified dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Split with explicit sizes
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
    /// let parts = tensor.split_with_sizes(&[2, 3], 1);
    /// assert_eq!(parts.len(), 2);
    /// assert_eq!(parts[0].shape().dims, vec![1, 2]);
    /// assert_eq!(parts[1].shape().dims, vec![1, 3]);
    /// assert_eq!(parts[0].get(&[0, 0]), 1.0);
    /// assert_eq!(parts[0].get(&[0, 1]), 2.0);
    /// assert_eq!(parts[1].get(&[0, 0]), 3.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Split 2D tensor with different chunk sizes
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let parts = tensor.split_with_sizes(&[1, 2], 1);
    /// assert_eq!(parts.len(), 2);
    /// assert_eq!(parts[0].shape().dims, vec![2, 1]);
    /// assert_eq!(parts[1].shape().dims, vec![2, 2]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Split with gradient tracking
    /// let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// tensor.set_requires_grad(true);
    ///
    /// let parts = tensor.split_with_sizes(&[1, 1], 1);
    /// assert_eq!(parts.len(), 2);
    /// assert!(parts[0].requires_grad());
    /// assert!(parts[1].requires_grad());
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Split 1D tensor with explicit sizes
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();
    /// let parts = tensor.split_with_sizes(&[2, 2, 2], 0);
    /// assert_eq!(parts.len(), 3);
    /// assert_eq!(parts[0].shape().dims, vec![2]);
    /// assert_eq!(parts[1].shape().dims, vec![2]);
    /// assert_eq!(parts[2].shape().dims, vec![2]);
    /// ```
    ///
    /// # Performance
    ///
    /// - **First Chunk**: O(1) - Returns a view when possible (zero-copy)
    /// - **Subsequent Chunks**: O(n) - May require data copying for non-zero offsets
    /// - **Memory Usage**: Minimal allocation for view operations, copying for non-zero offsets
    /// - **Gradient Tracking**: Each chunk preserves gradient requirements and tracking
    ///
    /// # Relationship to Other Operations
    ///
    /// This operation is related to other tensor transformations:
    /// - `split()` - Simplified version with equal-sized chunks
    /// - `cat()` - Inverse operation that concatenates tensors back together
    /// - `chunk()` - Alternative splitting operation with different semantics
    ///
    /// # Memory Layout
    ///
    /// The first chunk maintains the same underlying data as a view when
    /// the base offset is zero. Subsequent chunks may require data copying
    /// to handle non-zero base offsets, ensuring proper memory layout.
    /// Zero-sized chunks are handled by creating empty tensors with
    /// appropriate shapes.
    pub fn split_with_sizes(&self, split_sizes: &[usize], dim: usize) -> Vec<Tensor> {
        assert!(self.shape().rank() > 0, "split requires non-zero rank");
        assert!(
            dim < self.shape().rank(),
            "split dim {} out of bounds for rank {}",
            dim,
            self.shape().rank()
        );
        let dim_size = self.shape().dims[dim];
        let total: usize = split_sizes.iter().sum();
        assert!(
            total == dim_size,
            "sum of split sizes {} must equal size {} of dim {}",
            total,
            dim_size,
            dim
        );

        let mut outputs = Vec::with_capacity(split_sizes.len());
        let mut start = 0usize;
        for &len in split_sizes {
            if len == 0 {
                outputs.push(Tensor::zeros(
                    self.shape()
                        .dims
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| if i == dim { 0 } else { d })
                        .collect(),
                ));
                continue;
            }
            // Build new dims/strides with updated length along `dim`
            let mut new_dims = self.shape().dims.clone();
            new_dims[dim] = len;
            let new_strides = self.strides().to_vec();

            let base_offset = start * self.stride(dim);

            let mut piece: Tensor;
            if base_offset == 0 {
                // True view for the first chunk
                let view_shape = crate::tensor::Shape::as_view(new_dims, new_strides);
                piece = self.create_view_with_shape(view_shape);
            } else {
                // Materialize contiguous copy for non-zero base offset
                piece = Tensor::new(new_dims.clone());
                let rank = new_dims.len();
                let numel = piece.size();
                let mut coords = vec![0usize; rank];
                for lin in 0..numel {
                    let mut tmp = lin;
                    for i in (0..rank).rev() {
                        let s = new_dims[i];
                        coords[i] = if s == 0 { 0 } else { tmp % s };
                        if s != 0 {
                            tmp /= s;
                        }
                    }
                    // Map to source coords
                    let mut src_coords = coords.clone();
                    src_coords[dim] = start + coords[dim];
                    let src_off = self.shape().offset(&src_coords);
                    unsafe {
                        *piece.as_mut_ptr().add(lin) = *self.as_ptr().add(src_off);
                    }
                }
            }

            // GradTrack: register backward to scatter this piece's grad into original input range
            if self.requires_grad() {
                piece.set_requires_grad_internal(true);
                let grad_fn = GradFn::Split {
                    dim,
                    start,
                    length: len,
                    input_shape: self.shape().dims.clone(),
                };
                piece.set_grad_fn(grad_fn.clone());
                GradEngine::register_operation(piece.id(), vec![self.id()], grad_fn);
            }

            outputs.push(piece);
            start += len;
        }

        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_equal_forward() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&data, vec![2, 6]).unwrap();
        let parts = x.split(2, 1);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape().dims, vec![2, 2]);
        assert_eq!(parts[1].shape().dims, vec![2, 2]);
        assert_eq!(parts[2].shape().dims, vec![2, 2]);
        // Check a few values
        assert_eq!(parts[0].get(&[0, 0]), 0.0);
        assert_eq!(parts[1].get(&[0, 0]), 2.0);
        assert_eq!(parts[2].get(&[1, 1]), 11.0);
    }

    #[test]
    fn test_split_with_sizes_forward() {
        let data: Vec<f32> = (0..15).map(|i| (i as f32) * 0.1).collect();
        let x = Tensor::from_slice(&data, vec![3, 5]).unwrap();
        let parts = x.split_with_sizes(&[2, 1, 2], 1);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape().dims, vec![3, 2]);
        assert_eq!(parts[1].shape().dims, vec![3, 1]);
        assert_eq!(parts[2].shape().dims, vec![3, 2]);
        assert_eq!(parts[1].get(&[2, 0]), (2 * 5 + 2) as f32 * 0.1);
    }

    #[test]
    fn test_split_gradients_scatter() {
        let data: Vec<f32> = (0..10).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let x = Tensor::from_slice(&data, vec![2, 5])
            .unwrap()
            .with_requires_grad();
        let parts = x.split_with_sizes(&[2, 3], 1);
        // Reconstruct full tensor via concatenation then backward with implicit ones
        let mut full = Tensor::cat(&parts, 1);
        full.backward(None);
        let gx = x.grad_by_value().expect("grad missing");
        // All positions receive 1.0
        for i in 0..x.size() {
            assert_eq!(gx.get(&[i / 5, i % 5]), 1.0);
        }
    }

    #[test]
    fn test_split_1d_three_parts_grad() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&data, vec![6])
            .unwrap()
            .with_requires_grad();
        let parts = x.split_with_sizes(&[2, 2, 2], 0);
        // Concatenate then backward to avoid view/contig mismatches
        let mut full = Tensor::cat(&parts, 0);
        full.backward(None);
        let gx = x.grad_by_value().expect("grad missing");
        for i in 0..6 {
            assert_eq!(gx.get(&[i]), 1.0);
        }
    }
}
