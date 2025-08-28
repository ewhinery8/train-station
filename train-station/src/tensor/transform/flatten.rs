//! Tensor flattening operations
//!
//! This module provides tensor flattening functionality that transforms
//! multi-dimensional tensors into 1D representations. Flattening is a
//! fundamental tensor transformation operation used in machine learning
//! for preparing data for linear layers, feature extraction, and
//! dimensionality reduction.
//!
//! # Operations
//!
//! * `flatten()` - Flatten a tensor into a 1D representation
//!
//! # Performance Characteristics
//!
//! * **Zero-Copy Operation**: Returns a view when possible, avoiding data copying
//! * **Memory Efficient**: Reuses existing tensor data through reshape operations
//! * **Gradient Tracking**: Full GradTrack support for automatic differentiation
//! * **Shape Preservation**: Maintains the total number of elements while changing dimensions
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Flatten a 2D tensor
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let flattened = tensor.flatten();
//! assert_eq!(flattened.shape().dims, vec![4]);
//!
//! // Flatten a 3D tensor
//! let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
//! let tensor = Tensor::from_slice(&data, vec![2, 2, 3]).unwrap();
//! let flattened = tensor.flatten();
//! assert_eq!(flattened.shape().dims, vec![12]);
//! ```

use crate::tensor::Tensor;

impl Tensor {
    /// Flatten the tensor into a 1D representation
    ///
    /// Transforms a multi-dimensional tensor into a 1D tensor by reshaping
    /// all dimensions into a single dimension. This is equivalent to
    /// `reshape(vec![-1])` where `-1` automatically calculates the size
    /// based on the total number of elements.
    ///
    /// The flatten operation preserves the total number of elements while
    /// changing the tensor's shape to have a single dimension. This is
    /// commonly used in neural networks to prepare tensor data for linear
    /// layers or feature extraction.
    ///
    /// # Returns
    ///
    /// A 1D tensor containing the same data as the original tensor, with
    /// shape `[total_elements]` where `total_elements` is the product of
    /// all original dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Flatten a 2D tensor
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let flattened = tensor.flatten();
    /// assert_eq!(flattened.shape().dims, vec![4]);
    /// assert_eq!(flattened.get(&[0]), 1.0);
    /// assert_eq!(flattened.get(&[1]), 2.0);
    /// assert_eq!(flattened.get(&[2]), 3.0);
    /// assert_eq!(flattened.get(&[3]), 4.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Flatten a 3D tensor
    /// let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    /// let tensor = Tensor::from_slice(&data, vec![2, 2, 3]).unwrap();
    /// let flattened = tensor.flatten();
    /// assert_eq!(flattened.shape().dims, vec![12]);
    /// assert_eq!(flattened.size(), 12);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Flatten with gradient tracking
    /// let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// tensor.set_requires_grad(true);
    ///
    /// let flattened = tensor.flatten();
    /// assert!(flattened.requires_grad());
    /// assert_eq!(flattened.shape().dims, vec![4]);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Flatten an already 1D tensor (no change)
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let flattened = tensor.flatten();
    /// assert_eq!(flattened.shape().dims, vec![3]);
    /// assert_eq!(flattened.size(), 3);
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - Returns a view when possible
    /// - **Memory Usage**: No additional memory allocation for view operations
    /// - **Gradient Tracking**: Preserves gradient requirements and tracking
    ///
    /// # Relationship to Other Operations
    ///
    /// This operation is equivalent to:
    /// ```rust
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let flattened = tensor.reshape(vec![-1]);
    /// ```
    ///
    /// Where `-1` is a special value that automatically calculates the
    /// dimension size based on the total number of elements in the tensor.
    pub fn flatten(&self) -> Tensor {
        self.reshape(vec![-1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let flattened = tensor.flatten();

        assert_eq!(flattened.shape().dims, vec![4]);
        assert_eq!(flattened.size(), 4);
    }

    #[test]
    fn test_flatten_3d() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();
        let flattened = tensor.flatten();

        assert_eq!(flattened.shape().dims, vec![24]);
        assert_eq!(flattened.size(), 24);
    }

    #[test]
    fn test_flatten_already_1d() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let flattened = tensor.flatten();

        assert_eq!(flattened.shape().dims, vec![3]);
        assert_eq!(flattened.size(), 3);
    }

    #[test]
    fn test_flatten_preserves_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_slice(&data, vec![2, 3]).unwrap();
        let flattened = tensor.flatten();

        // Verify data is preserved
        for (i, &d) in data.iter().enumerate().take(data.len()) {
            assert_eq!(flattened.get(&[i]), d);
        }
    }
}
