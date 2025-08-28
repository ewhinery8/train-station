//! Basic tensor initialization methods
//!
//! This module provides fundamental tensor initialization operations for creating
//! tensors with specific constant values. All methods are optimized with SIMD
//! operations for maximum performance on large tensors.
//!
//! # Key Features
//!
//! - **`zeros`**: Create tensors filled with zeros
//! - **`ones`**: Create tensors filled with ones  
//! - **`fill`**: Fill existing tensors with a constant value
//! - **Device-aware initialization**: Create tensors on specific devices
//! - **SIMD optimization**: Vectorized operations for large tensors
//! - **Thread safety**: All operations are thread-safe
//!
//! # Performance Characteristics
//!
//! - **Memory Allocation**: Single allocation with optimized alignment
//! - **SIMD Operations**: AVX2-optimized filling for large tensors
//! - **Unrolled Loops**: 4x unrolling for better instruction throughput
//! - **Memory Bandwidth**: Optimized for maximum memory bandwidth utilization
//! - **Zero-sized Handling**: Efficient handling of empty tensors
//!
//! # Examples
//!
//! ## Basic Initialization
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create tensors with different constant values
//! let zeros = Tensor::zeros(vec![2, 3]);
//! let ones = Tensor::ones(vec![2, 3]);
//! let mut filled = Tensor::new(vec![2, 3]);
//! filled.fill(42.0);
//!
//! assert_eq!(zeros.shape().dims, vec![2, 3]);
//! assert_eq!(ones.shape().dims, vec![2, 3]);
//! assert_eq!(filled.shape().dims, vec![2, 3]);
//!
//! // Verify initialization values
//! assert_eq!(zeros.get(&[0, 0]), 0.0);
//! assert_eq!(ones.get(&[0, 0]), 1.0);
//! assert_eq!(filled.get(&[0, 0]), 42.0);
//! ```
//!
//! ## Device-Specific Initialization
//!
//! ```
//! use train_station::Tensor;
//! use train_station::Device;
//!
//! // Create tensors on specific devices
//! let cpu_zeros = Tensor::zeros_on_device(vec![2, 2], Device::cpu());
//! let cpu_ones = Tensor::ones_on_device(vec![2, 2], Device::cpu());
//!
//! assert_eq!(cpu_zeros.device(), Device::cpu());
//! assert_eq!(cpu_ones.device(), Device::cpu());
//! assert_eq!(cpu_zeros.size(), 4);
//! assert_eq!(cpu_ones.size(), 4);
//!
//! // Verify device-specific initialization
//! assert_eq!(cpu_zeros.get(&[0, 0]), 0.0);
//! assert_eq!(cpu_ones.get(&[0, 0]), 1.0);
//! ```
//!
//! ## Fill Operations
//!
//! ```
//! use train_station::Tensor;
//!
//! // Fill existing tensors with constant values
//! let mut tensor = Tensor::new(vec![3, 3]);
//! tensor.fill(3.14159);
//!
//! // Verify all elements are filled with the specified value
//! for i in 0..tensor.size() {
//!     assert!((tensor.get(&[i / 3, i % 3]) - 3.14159).abs() < 1e-6);
//! }
//! ```
//!
//! ## Zero-Sized Tensor Handling
//!
//! ```
//! use train_station::Tensor;
//!
//! // Handle zero-sized tensors gracefully
//! let mut empty_tensor = Tensor::new(vec![0]);
//! empty_tensor.fill(42.0); // Should not panic
//! assert_eq!(empty_tensor.size(), 0);
//! ```
//!
//! # Design Principles
//!
//! - **Performance First**: SIMD-optimized operations for maximum speed
//! - **Memory Safety**: Safe operations with proper bounds checking
//! - **Device Abstraction**: Unified interface for CPU and future GPU operations
//! - **Zero-Cost Abstractions**: Minimal overhead for initialization operations
//! - **Thread Safety**: All operations are safe for concurrent access

use crate::tensor::core::Tensor;

impl Tensor {
    /// Creates a new tensor filled with zeros
    ///
    /// Convenience constructor that creates a tensor and initializes all elements
    /// to zero. Uses optimized SIMD operations for efficient zero initialization.
    ///
    /// # Arguments
    ///
    /// * `shape_dims` - Vector of dimension sizes defining the tensor shape
    ///
    /// # Returns
    ///
    /// A new tensor with all elements initialized to zero
    ///
    /// # Performance
    ///
    /// - **Memory Allocation**: Single allocation with optimized alignment
    /// - **Initialization**: SIMD-optimized zero filling for large tensors
    /// - **Thread Safe**: Atomic ID generation for gradtrack tracking
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 3]);
    /// assert_eq!(tensor.size(), 6);
    /// assert_eq!(tensor.shape().dims, vec![2, 3]);
    ///
    /// // Verify all elements are zero
    /// assert_eq!(tensor.get(&[0, 0]), 0.0);
    /// assert_eq!(tensor.get(&[1, 2]), 0.0);
    /// ```
    #[inline]
    pub fn zeros(shape_dims: Vec<usize>) -> Self {
        let mut tensor = Self::new(shape_dims);
        tensor.fill(0.0);
        tensor
    }

    /// Creates a new tensor filled with ones
    ///
    /// Convenience constructor that creates a tensor and initializes all elements
    /// to one. Uses optimized SIMD operations for efficient initialization.
    ///
    /// # Arguments
    ///
    /// * `shape_dims` - Vector of dimension sizes defining the tensor shape
    ///
    /// # Returns
    ///
    /// A new tensor with all elements initialized to one
    ///
    /// # Performance
    ///
    /// - **Memory Allocation**: Single allocation with optimized alignment
    /// - **Initialization**: SIMD-optimized one filling for large tensors
    /// - **Thread Safe**: Atomic ID generation for gradtrack tracking
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 3]);
    /// assert_eq!(tensor.size(), 6);
    /// assert_eq!(tensor.shape().dims, vec![2, 3]);
    ///
    /// // Verify all elements are one
    /// assert_eq!(tensor.get(&[0, 0]), 1.0);
    /// assert_eq!(tensor.get(&[1, 2]), 1.0);
    /// ```
    #[inline]
    pub fn ones(shape_dims: Vec<usize>) -> Self {
        let mut tensor = Self::new(shape_dims);
        tensor.fill(1.0);
        tensor
    }

    /// Creates a new tensor filled with zeros on a specific device
    ///
    /// Convenience constructor that creates a tensor on the specified device
    /// and initializes all elements to zero. Uses optimized SIMD operations
    /// for efficient zero initialization.
    ///
    /// # Arguments
    ///
    /// * `shape_dims` - Vector of dimension sizes defining the tensor shape
    /// * `device` - The device where the tensor should be allocated
    ///
    /// # Returns
    ///
    /// A new tensor with all elements initialized to zero
    ///
    /// # Performance
    ///
    /// - **Memory Allocation**: Device-specific allocation with optimized alignment
    /// - **Initialization**: SIMD-optimized zero filling for large tensors
    /// - **Thread Safe**: Atomic ID generation for gradtrack tracking
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    /// use train_station::Device;
    ///
    /// let tensor = Tensor::zeros_on_device(vec![2, 2], Device::cpu());
    /// assert_eq!(tensor.device(), Device::cpu());
    /// assert_eq!(tensor.size(), 4);
    ///
    /// // Verify all elements are zero
    /// assert_eq!(tensor.get(&[0, 0]), 0.0);
    /// assert_eq!(tensor.get(&[1, 1]), 0.0);
    /// ```
    #[inline]
    pub fn zeros_on_device(shape_dims: Vec<usize>, device: crate::device::Device) -> Self {
        let mut tensor = Self::new_on_device(shape_dims, device);
        tensor.fill(0.0);
        tensor
    }

    /// Creates a new tensor filled with ones on a specific device
    ///
    /// Convenience constructor that creates a tensor on the specified device
    /// and initializes all elements to one. Uses optimized SIMD operations
    /// for efficient initialization.
    ///
    /// # Arguments
    ///
    /// * `shape_dims` - Vector of dimension sizes defining the tensor shape
    /// * `device` - The device where the tensor should be allocated
    ///
    /// # Returns
    ///
    /// A new tensor with all elements initialized to one
    ///
    /// # Performance
    ///
    /// - **Memory Allocation**: Device-specific allocation with optimized alignment
    /// - **Initialization**: SIMD-optimized one filling for large tensors
    /// - **Thread Safe**: Atomic ID generation for gradtrack tracking
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    /// use train_station::Device;
    ///
    /// let tensor = Tensor::ones_on_device(vec![2, 2], Device::cpu());
    /// assert_eq!(tensor.device(), Device::cpu());
    /// assert_eq!(tensor.size(), 4);
    ///
    /// // Verify all elements are one
    /// assert_eq!(tensor.get(&[0, 0]), 1.0);
    /// assert_eq!(tensor.get(&[1, 1]), 1.0);
    /// ```
    #[inline]
    pub fn ones_on_device(shape_dims: Vec<usize>, device: crate::device::Device) -> Self {
        let mut tensor = Self::new_on_device(shape_dims, device);
        tensor.fill(1.0);
        tensor
    }

    /// Fills the tensor with a constant value using SIMD optimization
    ///
    /// Efficiently initializes all elements of the tensor to the specified value.
    /// Uses SIMD operations for large tensors to maximize performance.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to fill the tensor with
    ///
    /// # Performance
    ///
    /// - **SIMD Optimization**: Uses AVX2 for large tensors when available
    /// - **Unrolled Loops**: 4x unrolling for better instruction throughput
    /// - **Memory Bandwidth**: Optimized for maximum memory bandwidth utilization
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let mut tensor = Tensor::new(vec![2, 3]);
    /// tensor.fill(42.0);
    ///
    /// // Verify all elements are 42.0
    /// assert_eq!(tensor.get(&[0, 0]), 42.0);
    /// assert_eq!(tensor.get(&[1, 2]), 42.0);
    /// ```
    ///
    /// ## Zero-Sized Tensor Handling
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let mut empty_tensor = Tensor::new(vec![0]);
    /// empty_tensor.fill(42.0); // Should not panic
    /// assert_eq!(empty_tensor.size(), 0);
    /// ```
    #[inline]
    pub fn fill(&mut self, value: f32) {
        if self.shape().size == 0 {
            return;
        }

        unsafe {
            let ptr = self.as_mut_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                // Use SIMD for better performance when available
                if is_x86_feature_detected!("avx2") {
                    self.fill_simd_avx2(ptr, value);
                    return;
                }
            }

            // Fallback to scalar operations
            for i in 0..self.shape().size {
                *ptr.add(i) = value;
            }
        }
    }

    /// Fills the tensor with a constant value using AVX2 SIMD optimization
    ///
    /// Internal method that uses AVX2 instructions to efficiently fill large tensors.
    /// Processes 32 elements per iteration with 4x unrolling for maximum memory bandwidth.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to the tensor data
    /// * `value` - The value to fill the tensor with
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// * `ptr` is a valid pointer to tensor data
    /// * The tensor size matches the allocated memory
    /// * AVX2 is available on the target architecture
    ///
    /// # Performance
    ///
    /// - **SIMD Operations**: 32 elements per iteration using AVX2
    /// - **Unrolled Loops**: 4x unrolling for better instruction throughput
    /// - **Memory Bandwidth**: Optimized for maximum memory bandwidth utilization
    /// - **Remaining Elements**: Efficient handling of non-multiple-of-32 sizes
    ///
    /// # Implementation Details
    ///
    /// This method uses AVX2 SIMD instructions to fill memory efficiently:
    /// 1. Creates a vector of 8 identical values using `_mm256_set1_ps`
    /// 2. Processes 32 elements per iteration (4x unrolled)
    /// 3. Handles remaining 8-element blocks
    /// 4. Fills final elements with scalar operations
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn fill_simd_avx2(&self, ptr: *mut f32, value: f32) {
        let mut_ptr = ptr;
        let value_vec = _mm256_set1_ps(value);
        let size = self.shape().size;
        let simd_count = size / 32; // Process 32 elements per iteration
        let mut offset = 0;

        // Unrolled SIMD fill for better memory bandwidth utilization
        for _ in 0..simd_count {
            _mm256_store_ps(mut_ptr.add(offset), value_vec);
            _mm256_store_ps(mut_ptr.add(offset + 8), value_vec);
            _mm256_store_ps(mut_ptr.add(offset + 16), value_vec);
            _mm256_store_ps(mut_ptr.add(offset + 24), value_vec);
            offset += 32;
        }

        // Handle remaining 8-element blocks
        let remaining_full_blocks = (size - offset) / 8;
        for _ in 0..remaining_full_blocks {
            _mm256_store_ps(mut_ptr.add(offset), value_vec);
            offset += 8;
        }

        // Handle final elements
        for i in offset..size {
            *mut_ptr.add(i) = value;
        }
    }
}

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_basic() {
        let tensor = Tensor::zeros(vec![2, 3]);
        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.shape().dims, vec![2, 3]);

        // Verify all elements are zero
        for i in 0..tensor.size() {
            unsafe {
                assert_eq!(*tensor.as_ptr().add(i), 0.0);
            }
        }
    }

    #[test]
    fn test_ones_basic() {
        let tensor = Tensor::ones(vec![2, 3]);
        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.shape().dims, vec![2, 3]);

        // Verify all elements are one
        for i in 0..tensor.size() {
            unsafe {
                assert_eq!(*tensor.as_ptr().add(i), 1.0);
            }
        }
    }

    #[test]
    fn test_zeros_on_device() {
        use crate::device::Device;

        let tensor = Tensor::zeros_on_device(vec![2, 2], Device::cpu());
        assert_eq!(tensor.device(), Device::cpu());
        assert_eq!(tensor.size(), 4);

        // Verify all elements are zero
        for i in 0..tensor.size() {
            unsafe {
                assert_eq!(*tensor.as_ptr().add(i), 0.0);
            }
        }
    }

    #[test]
    fn test_ones_on_device() {
        use crate::device::Device;

        let tensor = Tensor::ones_on_device(vec![2, 2], Device::cpu());
        assert_eq!(tensor.device(), Device::cpu());
        assert_eq!(tensor.size(), 4);

        // Verify all elements are one
        for i in 0..tensor.size() {
            unsafe {
                assert_eq!(*tensor.as_ptr().add(i), 1.0);
            }
        }
    }

    #[test]
    fn test_fill_basic() {
        let mut tensor = Tensor::new(vec![2, 3]);
        tensor.fill(42.0);

        // Verify all elements are 42.0
        for i in 0..tensor.size() {
            unsafe {
                assert_eq!(*tensor.as_ptr().add(i), 42.0);
            }
        }
    }

    #[test]
    fn test_fill_zero_sized() {
        let mut tensor = Tensor::new(vec![0]);
        // Should not panic
        tensor.fill(42.0);
        assert_eq!(tensor.size(), 0);
    }

    #[test]
    fn test_fill_large_tensor() {
        let mut tensor = Tensor::new(vec![100, 100]);
        tensor.fill(std::f32::consts::PI);

        // Verify all elements are 3.14159
        for i in 0..tensor.size() {
            unsafe {
                assert!((*tensor.as_ptr().add(i) - std::f32::consts::PI).abs() < 1e-6);
            }
        }
    }
}
