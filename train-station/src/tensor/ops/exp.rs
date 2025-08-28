//! Exponential operations for tensors
//!
//! Provides element-wise exponential function following PyTorch conventions with
//! comprehensive automatic differentiation support and optimized scalar computation.
//!
//! # Key Features
//!
//! - **Element-wise Exponential**: `exp()` - Computes e^x for each element (PyTorch `exp()` equivalent)
//! - **Automatic Differentiation**: Full gradtrack support with efficient gradient computation
//! - **Optimized Scalar Math**: Uses optimized scalar exponential for accuracy and simplicity
//! - **Cache Optimization**: Memory access patterns optimized for modern CPUs
//! - **Zero-copy Operations**: Efficient memory usage where possible
//! - **Mathematical Accuracy**: High-precision exponential computation
//!
//! # Mathematical Properties
//!
//! The exponential function e^x has the following properties:
//! - e^0 = 1
//! - e^1 ≈ 2.71828 (Euler's number)
//! - e^(-x) = 1/e^x
//! - e^(x+y) = e^x * e^y
//! - Gradient: d/dx(e^x) = e^x
//!
//! # Performance Characteristics
//!
//! - **Scalar Optimization**: Optimized scalar exponential computation
//! - **Unrolled Loops**: 4x unrolling for optimal instruction throughput
//! - **Cache-friendly Access**: Linear memory access patterns
//! - **Mathematical Accuracy**: High-precision floating-point exponential
//! - **Gradient Optimization**: Efficient gradtrack with NoGradTrack support

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

// Note: exp uses scalar math for accuracy and simplicity; SIMD width-load path was removed due to alignment strictness

impl Tensor {
    /// Element-wise exponential function.
    ///
    /// Computes e^x for each element: `output[i] = e^(self[i])`
    ///
    /// # Returns
    /// A new tensor with the exponential of each element
    ///
    /// # Examples
    ///
    /// ## Basic Exponential
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[0.0, 1.0, 2.0], vec![3]).unwrap();
    /// let b = a.exp();
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 1.0); // e^0 = 1
    /// assert!((b.get(&[1]) - 2.71828).abs() < 1e-5); // e^1 ≈ 2.71828
    /// assert!((b.get(&[2]) - 7.38906).abs() < 1e-5); // e^2 ≈ 7.38906
    /// ```
    ///
    /// ## Negative Values
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[-1.0, 0.0, 1.0], vec![3]).unwrap();
    /// let b = a.exp();
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert!((b.get(&[0]) - 0.36788).abs() < 1e-5); // e^(-1) ≈ 0.36788
    /// assert_eq!(b.get(&[1]), 1.0); // e^0 = 1
    /// assert!((b.get(&[2]) - 2.71828).abs() < 1e-5); // e^1 ≈ 2.71828
    /// ```
    #[inline]
    pub fn exp(&self) -> Tensor {
        let mut result = self.exp_optimized();
        if self.requires_grad() && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Exp {
                saved_output: Box::new(result.clone()),
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
        }
        result
    }
    /// Internal optimized exponential operation
    ///
    /// Performs element-wise exponential computation using optimized scalar math
    /// for maximum accuracy and performance. This is the core implementation
    /// used by `exp()`.
    ///
    /// # Returns
    ///
    /// A new tensor containing the exponential of each element
    ///
    /// # Performance Characteristics
    ///
    /// - **Scalar Optimization**: Uses optimized scalar exponential for accuracy
    /// - **Unrolled Loops**: 4x unrolling for optimal instruction throughput
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Mathematical Accuracy**: High-precision floating-point exponential
    /// - **Zero-sized Handling**: Fast return for empty tensors
    ///
    /// # Implementation Details
    ///
    /// Uses scalar exponential computation for maximum mathematical accuracy.
    /// SIMD optimization was removed due to alignment requirements and the
    /// need for high-precision mathematical operations.
    #[inline]
    pub(crate) fn exp_optimized(&self) -> Tensor {
        let mut output = Tensor::new(self.shape().dims.clone());

        // Fast return for zero-sized tensors
        if self.size() == 0 {
            return output;
        }

        unsafe {
            let src = self.as_ptr();
            let dst = output.as_mut_ptr();
            self.exp_scalar_fallback(src, dst);
        }

        output
    }

    /// Optimized scalar exponential fallback
    ///
    /// Performs element-wise exponential using optimized scalar operations with
    /// 4x unrolling for better instruction-level parallelism and cache efficiency.
    ///
    /// # Arguments
    ///
    /// * `src` - Pointer to source tensor data
    /// * `dst` - Pointer to output tensor data
    ///
    /// # Safety
    ///
    /// Requires valid pointers with sufficient memory for the tensor size.
    /// All pointers must point to valid tensor data.
    ///
    /// # Performance Characteristics
    ///
    /// - **Unrolling**: 4x unrolling for instruction-level parallelism
    /// - **Memory Access**: Linear access patterns for cache efficiency
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Mathematical Accuracy**: Uses high-precision scalar exponential
    #[inline]
    unsafe fn exp_scalar_fallback(&self, src: *const f32, dst: *mut f32) {
        let size = self.size();
        let unroll = 4;
        let mut offset = 0;
        let unroll_count = size / unroll;
        for _ in 0..unroll_count {
            *dst.add(offset) = (*src.add(offset)).exp();
            *dst.add(offset + 1) = (*src.add(offset + 1)).exp();
            *dst.add(offset + 2) = (*src.add(offset + 2)).exp();
            *dst.add(offset + 3) = (*src.add(offset + 3)).exp();
            offset += unroll;
        }
        for i in offset..size {
            *dst.add(i) = (*src.add(i)).exp();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_basic() {
        let data = [0.0, 1.0, -1.0, 2.0];
        let x = Tensor::from_slice(&data, vec![2, 2]).unwrap();
        let y = x.exp_optimized();
        unsafe {
            let yd = std::slice::from_raw_parts(y.as_ptr(), y.size());
            let xd = std::slice::from_raw_parts(x.as_ptr(), x.size());
            for i in 0..y.size() {
                let expected = xd[i].exp();
                assert!((yd[i] - expected).abs() < 1e-6);
            }
        }
    }
}
