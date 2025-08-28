//! Tensor squeeze operations
//!
//! This module provides tensor squeeze functionality that removes dimensions
//! of size 1 from tensors, effectively reducing the dimensionality while
//! preserving the total number of elements. Squeezing is a fundamental
//! tensor transformation operation used in machine learning for cleaning
//! up tensor shapes, preparing data for specific operations, and
//! implementing shape normalization.
//!
//! # Operations
//!
//! * `squeeze()` - Remove dimensions of size 1 from tensor
//!
//! # Performance Characteristics
//!
//! * **Zero-Copy Operation**: Returns a view when possible, avoiding data copying
//! * **Memory Efficient**: Reuses existing tensor data through reshape operations
//! * **Shape Reduction**: Reduces tensor rank by removing singleton dimensions
//! * **Gradient Tracking**: Full GradTrack support through reshape operations
//! * **Edge Case Handling**: Properly handles tensors with all size-1 dimensions
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Squeeze all size-1 dimensions
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3, 1]).unwrap();
//! let squeezed = tensor.squeeze(None);
//! assert_eq!(squeezed.shape().dims, vec![3]);
//! ```
//!
//! ```
//! use train_station::Tensor;
//!
//! // Squeeze specific dimension
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3, 1]).unwrap();
//! let squeezed = tensor.squeeze(Some(0));
//! assert_eq!(squeezed.shape().dims, vec![3, 1]);
//! ```
//!
//! # Gradient Tracking
//!
//! The squeeze operation supports automatic gradient tracking through
//! the GradTrack system via the underlying reshape operation. When
//! `requires_grad` is enabled, the operation preserves gradient
//! requirements and tracking through the transformation.

use crate::tensor::Tensor;

impl Tensor {
    /// Remove dimensions of size 1 from the tensor
    ///
    /// Removes singleton dimensions (dimensions with size 1) from the tensor,
    /// reducing its rank while preserving the total number of elements.
    /// This operation is useful for cleaning up tensor shapes and preparing
    /// data for operations that expect specific dimensionality.
    ///
    /// The squeeze operation can remove either all size-1 dimensions or a
    /// specific dimension if it has size 1. When all dimensions are size 1,
    /// the result is a scalar tensor with shape `[1]` rather than an empty
    /// tensor to maintain mathematical consistency.
    ///
    /// # Arguments
    ///
    /// * `dim` - Optional specific dimension to squeeze. If `None`, all size-1
    ///           dimensions are removed. If `Some(d)`, only dimension `d` is
    ///           removed if it has size 1.
    ///
    /// # Returns
    ///
    /// A new tensor with size-1 dimensions removed. The total number of
    /// elements remains unchanged.
    ///
    /// # Panics
    ///
    /// * If `dim` is specified but out of bounds for the tensor rank
    /// * If `dim` is specified but the dimension does not have size 1
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Squeeze all size-1 dimensions
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3, 1]).unwrap();
    /// let squeezed = tensor.squeeze(None);
    /// assert_eq!(squeezed.shape().dims, vec![3]);
    /// assert_eq!(squeezed.get(&[0]), 1.0);
    /// assert_eq!(squeezed.get(&[1]), 2.0);
    /// assert_eq!(squeezed.get(&[2]), 3.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Squeeze specific dimension
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3, 1]).unwrap();
    /// let squeezed = tensor.squeeze(Some(0));
    /// assert_eq!(squeezed.shape().dims, vec![3, 1]);
    /// assert_eq!(squeezed.get(&[0, 0]), 1.0);
    /// assert_eq!(squeezed.get(&[1, 0]), 2.0);
    /// assert_eq!(squeezed.get(&[2, 0]), 3.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Squeeze preserves data integrity
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::from_slice(&data, vec![1, 2, 1, 2]).unwrap();
    /// let squeezed = tensor.squeeze(None);
    /// assert_eq!(squeezed.shape().dims, vec![2, 2]);
    /// assert_eq!(squeezed.size(), 4);
    /// assert_eq!(squeezed.get(&[0, 0]), data[0]);
    /// assert_eq!(squeezed.get(&[0, 1]), data[1]);
    /// assert_eq!(squeezed.get(&[1, 0]), data[2]);
    /// assert_eq!(squeezed.get(&[1, 1]), data[3]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Handle edge case: all dimensions are size 1
    /// let tensor = Tensor::from_slice(&[5.0], vec![1, 1, 1]).unwrap();
    /// let squeezed = tensor.squeeze(None);
    /// assert_eq!(squeezed.shape().dims, vec![1]); // Not empty!
    /// assert_eq!(squeezed.get(&[0]), 5.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Squeeze with gradient tracking
    /// let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3, 1]).unwrap();
    /// tensor.set_requires_grad(true);
    ///
    /// let squeezed = tensor.squeeze(None);
    /// assert!(squeezed.requires_grad());
    /// assert_eq!(squeezed.shape().dims, vec![3]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Squeeze and unsqueeze roundtrip
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let unsqueezed = tensor.unsqueeze(0);
    /// assert_eq!(unsqueezed.shape().dims, vec![1, 3]);
    ///
    /// let squeezed = unsqueezed.squeeze(Some(0));
    /// assert_eq!(squeezed.shape().dims, vec![3]);
    /// assert_eq!(squeezed.get(&[0]), 1.0);
    /// assert_eq!(squeezed.get(&[2]), 3.0);
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Returns a view through reshape operation
    /// - **Memory Usage**: No additional memory allocation (view operation)
    /// - **Gradient Tracking**: Preserves gradient requirements and tracking
    /// - **Shape Transformation**: Reduces tensor rank by removing singleton dimensions
    ///
    /// # Relationship to Other Operations
    ///
    /// This operation is related to other tensor transformations:
    /// - `unsqueeze()` - Inverse operation that adds size-1 dimensions
    /// - `reshape()` - More general shape transformation operation
    /// - `flatten()` - Reduces tensor to 1D by combining all dimensions
    ///
    /// # Memory Layout
    ///
    /// The squeezed tensor maintains the same underlying data as the original
    /// tensor through the reshape operation. This ensures zero-copy behavior
    /// when the tensor is contiguous, with only the shape metadata being
    /// modified to reflect the reduced dimensionality.
    ///
    /// # Edge Cases
    ///
    /// - **All size-1 dimensions**: Returns a tensor with shape `[1]` rather than
    ///   an empty tensor to maintain mathematical consistency
    /// - **No size-1 dimensions**: Returns a tensor with the same shape as the input
    /// - **Mixed dimensions**: Only removes dimensions with size 1, preserving others
    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        let mut new_dims = Vec::new();

        if let Some(d) = dim {
            // Squeeze specific dimension
            assert!(d < self.shape().rank(), "Dimension {} out of bounds", d);
            assert_eq!(
                self.shape().dims[d],
                1,
                "Cannot squeeze dimension {} with size {}",
                d,
                self.shape().dims[d]
            );

            for (i, &size) in self.shape().dims.iter().enumerate() {
                if i != d {
                    new_dims.push(size);
                }
            }
        } else {
            // Squeeze all size-1 dimensions
            for &size in &self.shape().dims {
                if size != 1 {
                    new_dims.push(size);
                }
            }
        }

        // Handle edge case where all dimensions were size 1
        if new_dims.is_empty() {
            new_dims.push(1);
        }

        // Convert to i32 for reshape call
        let new_shape: Vec<i32> = new_dims.iter().map(|&d| d as i32).collect();
        self.reshape(new_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squeeze() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3, 1]).unwrap();

        // Squeeze all size-1 dimensions
        let squeezed = tensor.squeeze(None);
        assert_eq!(squeezed.shape().dims, vec![3]);

        // Squeeze specific dimension
        let squeezed = tensor.squeeze(Some(0));
        assert_eq!(squeezed.shape().dims, vec![3, 1]);
    }

    #[test]
    fn test_squeeze_preserves_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, vec![1, 2, 1, 2]).unwrap();
        let squeezed = tensor.squeeze(None);

        assert_eq!(squeezed.shape().dims, vec![2, 2]);
        assert_eq!(squeezed.size(), 4);

        // Verify data is preserved
        assert_eq!(squeezed.get(&[0, 0]), data[0]);
        assert_eq!(squeezed.get(&[0, 1]), data[1]);
        assert_eq!(squeezed.get(&[1, 0]), data[2]);
        assert_eq!(squeezed.get(&[1, 1]), data[3]);
    }

    #[test]
    fn test_squeeze_all_ones() {
        let tensor = Tensor::from_slice(&[5.0], vec![1, 1, 1]).unwrap();
        let squeezed = tensor.squeeze(None);

        // When all dimensions are 1, result should be [1] not []
        assert_eq!(squeezed.shape().dims, vec![1]);
        assert_eq!(squeezed.get(&[0]), 5.0);
    }

    #[test]
    fn test_squeeze_specific_dimension() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();

        let squeezed = tensor.squeeze(Some(0));
        assert_eq!(squeezed.shape().dims, vec![2, 2]);
    }

    #[test]
    #[should_panic(expected = "Dimension 3 out of bounds")]
    fn test_squeeze_out_of_bounds() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        tensor.squeeze(Some(3)); // Should panic
    }

    #[test]
    #[should_panic(expected = "Cannot squeeze dimension 0 with size 3")]
    fn test_squeeze_non_unit_dimension() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        tensor.squeeze(Some(0)); // Should panic - dimension 0 has size 3, not 1
    }

    #[test]
    fn test_squeeze_unsqueeze_roundtrip() {
        // Test that squeeze and unsqueeze are inverses
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();

        let unsqueezed = tensor.unsqueeze(0);
        assert_eq!(unsqueezed.shape().dims, vec![1, 3]);

        let squeezed = unsqueezed.squeeze(Some(0));
        assert_eq!(squeezed.shape().dims, vec![3]);

        // Verify data integrity
        assert_eq!(squeezed.get(&[0]), 1.0);
        assert_eq!(squeezed.get(&[2]), 3.0);
    }
}
