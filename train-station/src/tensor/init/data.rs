//! Data-based tensor initialization methods
//!
//! This module provides methods to create tensors from existing data sources.
//! All methods validate data compatibility and perform efficient memory copying
//! with optimized performance characteristics.
//!
//! # Key Features
//!
//! - **`from_slice`**: Create tensors from slices of f32 data
//! - **Data validation**: Automatic size and compatibility checking
//! - **Efficient copying**: Optimized memory operations for performance
//! - **Error handling**: Clear error messages for validation failures
//! - **Multi-dimensional support**: Support for 1D, 2D, 3D, and higher-dimensional tensors
//! - **Zero-sized handling**: Proper handling of empty tensors
//!
//! # Performance Characteristics
//!
//! - **Memory Copy**: Efficient non-overlapping copy using SIMD when possible
//! - **Validation**: Fast size validation before allocation
//! - **Alignment**: Proper memory alignment for optimal performance
//! - **Large Data**: Optimized handling of large datasets
//! - **Zero Overhead**: Minimal validation overhead for correct data
//!
//! # Examples
//!
//! ## Basic Data Initialization
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create tensor from slice data
//! let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let tensor = Tensor::from_slice(&data, vec![2, 3]).unwrap();
//!
//! assert_eq!(tensor.size(), 6);
//! assert_eq!(tensor.shape().dims, vec![2, 3]);
//!
//! // Verify data was copied correctly
//! assert_eq!(tensor.get(&[0, 0]), 1.0);
//! assert_eq!(tensor.get(&[1, 2]), 6.0);
//! ```
//!
//! ## Multi-Dimensional Tensors
//!
//! ```
//! use train_station::Tensor;
//!
//! // 1D tensor
//! let data_1d = [1.0, 2.0, 3.0];
//! let tensor_1d = Tensor::from_slice(&data_1d, vec![3]).unwrap();
//! assert_eq!(tensor_1d.shape().dims, vec![3]);
//! assert_eq!(tensor_1d.get(&[1]), 2.0);
//!
//! // 3D tensor
//! let data_3d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let tensor_3d = Tensor::from_slice(&data_3d, vec![2, 2, 2]).unwrap();
//! assert_eq!(tensor_3d.shape().dims, vec![2, 2, 2]);
//! assert_eq!(tensor_3d.get(&[0, 0, 0]), 1.0);
//! assert_eq!(tensor_3d.get(&[1, 1, 1]), 8.0);
//! ```
//!
//! ## Error Handling
//!
//! ```
//! use train_station::Tensor;
//!
//! // Size mismatch error
//! let data = [1.0, 2.0, 3.0];
//! let result = Tensor::from_slice(&data, vec![2, 2]);
//! assert!(result.is_err());
//! let err = result.unwrap_err();
//! assert!(err.contains("Data size 3 doesn't match shape size 4"));
//! ```
//!
//! ## Zero-Sized Tensors
//!
//! ```
//! use train_station::Tensor;
//!
//! // Handle empty tensors gracefully
//! let data: [f32; 0] = [];
//! let tensor = Tensor::from_slice(&data, vec![0]).unwrap();
//! assert_eq!(tensor.size(), 0);
//! assert_eq!(tensor.shape().dims, vec![0]);
//! ```
//!
//! ## Large Data Sets
//!
//! ```
//! use train_station::Tensor;
//!
//! // Efficient handling of large datasets
//! let size = 1000;
//! let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
//! let tensor = Tensor::from_slice(&data, vec![size]).unwrap();
//!
//! assert_eq!(tensor.size(), size);
//! assert_eq!(tensor.get(&[0]), 0.0);
//! assert_eq!(tensor.get(&[100]), 100.0);
//! assert_eq!(tensor.get(&[999]), 999.0);
//! ```
//!
//! # Design Principles
//!
//! - **Data Safety**: Comprehensive validation of data compatibility
//! - **Performance First**: Optimized memory operations for maximum speed
//! - **Error Clarity**: Clear and descriptive error messages
//! - **Memory Efficiency**: Efficient copying with minimal overhead
//! - **Type Safety**: Strong typing for all data operations
//! - **Zero-Cost Validation**: Minimal overhead for correct data

use crate::tensor::core::Tensor;

impl Tensor {
    /// Creates a tensor from a slice of data
    ///
    /// Creates a new tensor with the specified shape and copies data from the
    /// provided slice. Validates that the data size matches the tensor shape
    /// before performing the copy operation.
    ///
    /// This method provides an efficient way to create tensors from existing
    /// data sources while ensuring data integrity and proper memory management.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of f32 values to copy into the tensor
    /// * `shape_dims` - Vector of dimension sizes defining the tensor shape
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor)` - Successfully created tensor with copied data
    /// * `Err(String)` - Error if data size doesn't match shape
    ///
    /// # Performance
    ///
    /// - **Memory Copy**: Efficient non-overlapping copy using SIMD when possible
    /// - **Validation**: Fast size validation before allocation
    /// - **Alignment**: Proper memory alignment for optimal performance
    /// - **Large Data**: Optimized handling of large datasets
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_slice(&data, vec![2, 3]).unwrap();
    /// assert_eq!(tensor.size(), 6);
    /// assert_eq!(tensor.get(&[0, 0]), 1.0);
    /// assert_eq!(tensor.get(&[1, 2]), 6.0);
    /// ```
    ///
    /// ## Multi-Dimensional Data
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // 1D tensor
    /// let data_1d = [1.0, 2.0, 3.0];
    /// let tensor_1d = Tensor::from_slice(&data_1d, vec![3]).unwrap();
    /// assert_eq!(tensor_1d.shape().dims, vec![3]);
    /// assert_eq!(tensor_1d.get(&[1]), 2.0);
    ///
    /// // 3D tensor
    /// let data_3d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let tensor_3d = Tensor::from_slice(&data_3d, vec![2, 2, 2]).unwrap();
    /// assert_eq!(tensor_3d.shape().dims, vec![2, 2, 2]);
    /// assert_eq!(tensor_3d.get(&[0, 0, 0]), 1.0);
    /// assert_eq!(tensor_3d.get(&[1, 1, 1]), 8.0);
    /// ```
    ///
    /// ## Error Handling
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Size mismatch error
    /// let data = [1.0, 2.0, 3.0];
    /// let result = Tensor::from_slice(&data, vec![2, 2]);
    /// assert!(result.is_err());
    /// let err = result.unwrap_err();
    /// assert!(err.contains("Data size 3 doesn't match shape size 4"));
    /// ```
    ///
    /// ## Zero-Sized Tensors
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Handle empty tensors gracefully
    /// let data: [f32; 0] = [];
    /// let tensor = Tensor::from_slice(&data, vec![0]).unwrap();
    /// assert_eq!(tensor.size(), 0);
    /// assert_eq!(tensor.shape().dims, vec![0]);
    /// ```
    ///
    /// ## Large Data Sets
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Efficient handling of large datasets
    /// let size = 1000;
    /// let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    /// let tensor = Tensor::from_slice(&data, vec![size]).unwrap();
    ///
    /// assert_eq!(tensor.size(), size);
    /// assert_eq!(tensor.get(&[0]), 0.0);
    /// assert_eq!(tensor.get(&[100]), 100.0);
    /// assert_eq!(tensor.get(&[999]), 999.0);
    /// ```
    ///
    /// # Implementation Details
    ///
    /// This method performs the following steps:
    /// 1. **Shape Validation**: Creates a Shape object and validates dimensions
    /// 2. **Size Check**: Ensures data length matches the calculated tensor size
    /// 3. **Memory Allocation**: Allocates tensor memory with proper alignment
    /// 4. **Data Copy**: Uses efficient non-overlapping memory copy operation
    /// 5. **Return**: Returns the created tensor or descriptive error message
    ///
    /// The memory copy operation uses `std::ptr::copy_nonoverlapping` for
    /// maximum performance and safety, ensuring no data corruption occurs
    /// during the copy process.
    pub fn from_slice(data: &[f32], shape_dims: Vec<usize>) -> Result<Self, String> {
        let shape = crate::tensor::Shape::new(shape_dims);

        if data.len() != shape.size {
            return Err(format!(
                "Data size {} doesn't match shape size {}",
                data.len(),
                shape.size
            ));
        }

        let mut tensor = Self::new(shape.dims.clone());

        // Copy data into tensor using efficient non-overlapping copy
        unsafe {
            let dst = tensor.as_mut_ptr();
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }

        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_slice_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_slice(&data, vec![2, 3]).unwrap();

        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.shape().dims, vec![2, 3]);

        // Verify data was copied correctly
        assert_eq!(tensor.get(&[0, 0]), 1.0);
        assert_eq!(tensor.get(&[0, 1]), 2.0);
        assert_eq!(tensor.get(&[0, 2]), 3.0);
        assert_eq!(tensor.get(&[1, 0]), 4.0);
        assert_eq!(tensor.get(&[1, 1]), 5.0);
        assert_eq!(tensor.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_from_slice_1d() {
        let data = [1.0, 2.0, 3.0];
        let tensor = Tensor::from_slice(&data, vec![3]).unwrap();

        assert_eq!(tensor.size(), 3);
        assert_eq!(tensor.shape().dims, vec![3]);

        assert_eq!(tensor.get(&[0]), 1.0);
        assert_eq!(tensor.get(&[1]), 2.0);
        assert_eq!(tensor.get(&[2]), 3.0);
    }

    #[test]
    fn test_from_slice_3d() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 7.0, 8.0];
        let tensor = Tensor::from_slice(&data, vec![2, 2, 2]).unwrap();

        assert_eq!(tensor.size(), 8);
        assert_eq!(tensor.shape().dims, vec![2, 2, 2]);

        // Verify 3D indexing
        assert_eq!(tensor.get(&[0, 0, 0]), 1.0);
        assert_eq!(tensor.get(&[0, 0, 1]), 2.0);
        assert_eq!(tensor.get(&[0, 1, 0]), 3.0);
        assert_eq!(tensor.get(&[0, 1, 1]), 4.0);
        assert_eq!(tensor.get(&[1, 0, 0]), 5.0);
        assert_eq!(tensor.get(&[1, 0, 1]), 8.0);
        assert_eq!(tensor.get(&[1, 1, 0]), 7.0);
        assert_eq!(tensor.get(&[1, 1, 1]), 8.0);
    }

    #[test]
    fn test_from_slice_size_mismatch() {
        let data = [1.0, 2.0, 3.0];
        let result = Tensor::from_slice(&data, vec![2, 2]);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Data size 3 doesn't match shape size 4"));
    }

    #[test]
    fn test_from_slice_empty() {
        let data: [f32; 0] = [];
        let tensor = Tensor::from_slice(&data, vec![0]).unwrap();

        assert_eq!(tensor.size(), 0);
        assert_eq!(tensor.shape().dims, vec![0]);
    }

    #[test]
    fn test_from_slice_large_data() {
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, vec![size]).unwrap();

        assert_eq!(tensor.size(), size);

        // Verify a few values
        assert_eq!(tensor.get(&[0]), 0.0);
        assert_eq!(tensor.get(&[100]), 100.0);
        assert_eq!(tensor.get(&[999]), 999.0);
    }
}
