//! Tensor unsqueeze operations
//!
//! This module provides tensor unsqueeze functionality that adds dimensions
//! of size 1 to tensors, effectively increasing the dimensionality while
//! preserving the total number of elements. Unsqueezing is a fundamental
//! tensor transformation operation used in machine learning for preparing
//! data for specific layer types, implementing broadcasting operations,
//! and creating batch dimensions from single samples.
//!
//! # Operations
//!
//! * `unsqueeze()` - Add a dimension of size 1 at the specified position
//!
//! # Performance Characteristics
//!
//! * **Zero-Copy Operation**: Returns a view through reshape operation
//! * **Memory Efficient**: Reuses existing tensor data through view operations
//! * **Shape Expansion**: Increases tensor rank by adding singleton dimensions
//! * **Gradient Tracking**: Full GradTrack support through reshape operations
//! * **Edge Case Handling**: Properly handles tensors of any rank and shape
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Add dimension at the beginning
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
//! let unsqueezed = tensor.unsqueeze(0);
//! assert_eq!(unsqueezed.shape().dims, vec![1, 3]);
//! ```
//!
//! ```
//! use train_station::Tensor;
//!
//! // Add dimension at the end
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
//! let unsqueezed = tensor.unsqueeze(1);
//! assert_eq!(unsqueezed.shape().dims, vec![3, 1]);
//! ```
//!
//! # Gradient Tracking
//!
//! The unsqueeze operation supports automatic gradient tracking through
//! the GradTrack system via the underlying reshape operation. When
//! `requires_grad` is enabled, the operation preserves gradient
//! requirements and tracking through the transformation.

use crate::tensor::Tensor;

impl Tensor {
    /// Add a dimension of size 1 at the specified position
    ///
    /// Inserts a new dimension of size 1 at the specified position in the
    /// tensor's shape, increasing the rank by 1 while preserving the total
    /// number of elements. This operation is useful for preparing tensors
    /// for broadcasting, creating batch dimensions, and adapting tensor
    /// shapes for specific neural network operations.
    ///
    /// The unsqueeze operation is the inverse of `squeeze()` - unsqueezing
    /// a dimension and then squeezing it at the same position returns the
    /// original tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - Position to insert the new dimension (0 <= dim <= rank)
    ///
    /// # Returns
    ///
    /// A new tensor with an additional dimension of size 1 at the specified
    /// position. The total number of elements remains unchanged.
    ///
    /// # Panics
    ///
    /// * If `dim` is out of bounds (dim > rank of the tensor)
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Add dimension at the beginning
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let unsqueezed = tensor.unsqueeze(0);
    /// assert_eq!(unsqueezed.shape().dims, vec![1, 3]);
    /// assert_eq!(unsqueezed.get(&[0, 0]), 1.0);
    /// assert_eq!(unsqueezed.get(&[0, 1]), 2.0);
    /// assert_eq!(unsqueezed.get(&[0, 2]), 3.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Add dimension at the end
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let unsqueezed = tensor.unsqueeze(1);
    /// assert_eq!(unsqueezed.shape().dims, vec![3, 1]);
    /// assert_eq!(unsqueezed.get(&[0, 0]), 1.0);
    /// assert_eq!(unsqueezed.get(&[1, 0]), 2.0);
    /// assert_eq!(unsqueezed.get(&[2, 0]), 3.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Add dimension in the middle of 2D tensor
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let unsqueezed = tensor.unsqueeze(1);
    /// assert_eq!(unsqueezed.shape().dims, vec![2, 1, 2]);
    /// assert_eq!(unsqueezed.get(&[0, 0, 0]), 1.0);
    /// assert_eq!(unsqueezed.get(&[0, 0, 1]), 2.0);
    /// assert_eq!(unsqueezed.get(&[1, 0, 0]), 3.0);
    /// assert_eq!(unsqueezed.get(&[1, 0, 1]), 4.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Unsqueeze preserves data integrity
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::from_slice(&data, vec![4]).unwrap();
    /// let unsqueezed = tensor.unsqueeze(0);
    /// assert_eq!(unsqueezed.shape().dims, vec![1, 4]);
    /// assert_eq!(unsqueezed.size(), 4);
    /// for (i, &d) in data.iter().enumerate() {
    ///     assert_eq!(unsqueezed.get(&[0, i]), d);
    /// }
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Unsqueeze with gradient tracking
    /// let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// tensor.set_requires_grad(true);
    ///
    /// let unsqueezed = tensor.unsqueeze(0);
    /// assert!(unsqueezed.requires_grad());
    /// assert_eq!(unsqueezed.shape().dims, vec![1, 3]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Unsqueeze and squeeze roundtrip
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
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Multiple unsqueeze operations
    /// let tensor = Tensor::from_slice(&[42.0], vec![1]).unwrap();
    /// let unsqueezed1 = tensor.unsqueeze(0);
    /// assert_eq!(unsqueezed1.shape().dims, vec![1, 1]);
    ///
    /// let unsqueezed2 = unsqueezed1.unsqueeze(0);
    /// assert_eq!(unsqueezed2.shape().dims, vec![1, 1, 1]);
    /// assert_eq!(unsqueezed2.get(&[0, 0, 0]), 42.0);
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Returns a view through reshape operation
    /// - **Memory Usage**: No additional memory allocation (view operation)
    /// - **Gradient Tracking**: Preserves gradient requirements and tracking
    /// - **Shape Transformation**: Increases tensor rank by adding singleton dimensions
    ///
    /// # Relationship to Other Operations
    ///
    /// This operation is related to other tensor transformations:
    /// - `squeeze()` - Inverse operation that removes size-1 dimensions
    /// - `reshape()` - More general shape transformation operation
    /// - `expand()` - Broadcasts dimensions to larger sizes
    ///
    /// # Memory Layout
    ///
    /// The unsqueezed tensor maintains the same underlying data as the original
    /// tensor through the reshape operation. This ensures zero-copy behavior
    /// when the tensor is contiguous, with only the shape metadata being
    /// modified to reflect the increased dimensionality.
    ///
    /// # Broadcasting Applications
    ///
    /// Unsqueeze is commonly used for broadcasting operations:
    /// ```rust
    /// use train_station::Tensor;
    ///
    /// // Prepare for broadcasting: [3] -> [1, 3] for row-wise operations
    /// let vector = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let row_vector = vector.unsqueeze(0); // Shape: [1, 3]
    ///
    /// // Prepare for broadcasting: [3] -> [3, 1] for column-wise operations
    /// let column_vector = vector.unsqueeze(1); // Shape: [3, 1]
    /// ```
    ///
    /// # Neural Network Applications
    ///
    /// Unsqueeze is essential for neural network operations:
    /// ```rust
    /// use train_station::Tensor;
    ///
    /// // Single sample -> batch dimension for neural network input
    /// let sample = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let batch = sample.unsqueeze(0); // Shape: [1, 3] for batch processing
    ///
    /// // Add channel dimension for convolutional operations
    /// let feature_map = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let with_channels = feature_map.unsqueeze(0); // Shape: [1, 2, 2] for conv layers
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let mut new_dims = self.shape().dims.clone();
        assert!(dim <= new_dims.len(), "Dimension {} out of bounds", dim);
        new_dims.insert(dim, 1);

        // Convert to i32 for reshape call
        let new_shape: Vec<i32> = new_dims.iter().map(|&d| d as i32).collect();
        self.reshape(new_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsqueeze() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();

        // Unsqueeze at beginning
        let unsqueezed = tensor.unsqueeze(0);
        assert_eq!(unsqueezed.shape().dims, vec![1, 3]);

        // Unsqueeze at end
        let unsqueezed = tensor.unsqueeze(1);
        assert_eq!(unsqueezed.shape().dims, vec![3, 1]);
    }

    #[test]
    fn test_unsqueeze_2d() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // Unsqueeze at different positions
        let unsqueezed = tensor.unsqueeze(0);
        assert_eq!(unsqueezed.shape().dims, vec![1, 2, 2]);

        let unsqueezed = tensor.unsqueeze(1);
        assert_eq!(unsqueezed.shape().dims, vec![2, 1, 2]);

        let unsqueezed = tensor.unsqueeze(2);
        assert_eq!(unsqueezed.shape().dims, vec![2, 2, 1]);
    }

    #[test]
    fn test_unsqueeze_preserves_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, vec![4]).unwrap();
        let unsqueezed = tensor.unsqueeze(0);

        assert_eq!(unsqueezed.shape().dims, vec![1, 4]);
        assert_eq!(unsqueezed.size(), 4);

        // Verify data is preserved
        for (i, &d) in data.iter().enumerate() {
            assert_eq!(unsqueezed.get(&[0, i]), d);
        }
    }

    #[test]
    #[should_panic(expected = "Dimension 4 out of bounds")]
    fn test_unsqueeze_out_of_bounds() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        tensor.unsqueeze(4); // Should panic
    }

    #[test]
    fn test_unsqueeze_with_gradients() {
        let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        tensor.set_requires_grad(true);

        let unsqueezed = tensor.unsqueeze(0);
        assert!(unsqueezed.requires_grad());
        assert_eq!(unsqueezed.shape().dims, vec![1, 3]);
    }

    #[test]
    fn test_unsqueeze_squeeze_roundtrip() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let unsqueezed = tensor.unsqueeze(0);
        let squeezed = unsqueezed.squeeze(Some(0));

        assert_eq!(squeezed.shape().dims, tensor.shape().dims);
        assert_eq!(squeezed.get(&[0]), tensor.get(&[0]));
        assert_eq!(squeezed.get(&[2]), tensor.get(&[2]));
    }

    #[test]
    fn test_multiple_unsqueeze() {
        let tensor = Tensor::from_slice(&[42.0], vec![1]).unwrap();
        let unsqueezed1 = tensor.unsqueeze(0);
        let unsqueezed2 = unsqueezed1.unsqueeze(0);

        assert_eq!(unsqueezed2.shape().dims, vec![1, 1, 1]);
        assert_eq!(unsqueezed2.get(&[0, 0, 0]), 42.0);
    }
}
