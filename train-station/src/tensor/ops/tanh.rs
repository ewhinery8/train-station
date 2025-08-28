//! Hyperbolic tangent activation function
//!
//! Provides element-wise hyperbolic tangent activation following PyTorch conventions with
//! comprehensive GradTrack support and high-precision scalar computation.
//!
//! # Key Features
//!
//! - **Hyperbolic Tangent**: `tanh()` - Element-wise hyperbolic tangent activation
//! - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
//! - **High Precision**: Accurate scalar implementation for mathematical validation
//! - **Performance Optimization**: 4x unrolled scalar operations for better throughput
//! - **Numerical Stability**: Robust implementation for extreme input values
//!
//! # Mathematical Properties
//!
//! The hyperbolic tangent function has the following properties:
//! - **Range**: Output values are in the range (-1, 1)
//! - **Symmetry**: tanh(-x) = -tanh(x) (odd function)
//! - **Asymptotes**: Approaches ±1 as x approaches ±∞
//! - **Zero**: tanh(0) = 0
//! - **Gradient**: ∂tanh(x)/∂x = 1 - tanh²(x) = sech²(x)
//! - **Monotonic**: Strictly increasing function
//!
//! # Performance Characteristics
//!
//! - **Scalar Implementation**: High-precision scalar computation for mathematical accuracy
//! - **4x Unrolling**: Optimized scalar operations with instruction-level parallelism
//! - **Cache-friendly**: Linear memory access patterns
//! - **Numerical Stability**: Robust handling of extreme input values
//! - **GradTrack Optimization**: Efficient automatic differentiation with gradient computation

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Element-wise hyperbolic tangent activation
    ///
    /// Computes hyperbolic tangent for each element: `output[i] = tanh(self[i])`
    ///
    /// The hyperbolic tangent function maps any real number to the range (-1, 1),
    /// making it useful as an activation function in neural networks.
    ///
    /// # Returns
    ///
    /// A new tensor with tanh applied to each element, values in range (-1, 1)
    ///
    /// # Performance Characteristics
    ///
    /// - **High Precision**: Accurate scalar implementation for mathematical validation
    /// - **4x Unrolling**: Optimized scalar operations with instruction-level parallelism
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Numerical Stability**: Robust handling of extreme input values
    /// - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
    ///
    /// # Mathematical Properties
    ///
    /// - **Range**: Output values are in the range (-1, 1)
    /// - **Symmetry**: tanh(-x) = -tanh(x) (odd function)
    /// - **Zero**: tanh(0) = 0
    /// - **Gradient**: ∂tanh(x)/∂x = 1 - tanh²(x) = sech²(x)
    ///
    /// # Examples
    ///
    /// ## Basic Hyperbolic Tangent
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[-1.0, 0.0, 1.0], vec![3]).unwrap();
    /// let b = a.tanh();
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert!((b.get(&[0]) - (-0.7615942)).abs() < 1e-6); // tanh(-1.0)
    /// assert!((b.get(&[1]) - 0.0).abs() < 1e-6); // tanh(0.0)
    /// assert!((b.get(&[2]) - 0.7615942).abs() < 1e-6); // tanh(1.0)
    /// ```
    ///
    /// ## Extreme Values
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[-10.0, 10.0], vec![2]).unwrap();
    /// let b = a.tanh();
    /// assert_eq!(b.shape().dims, vec![2]);
    /// assert!((b.get(&[0]) - (-1.0)).abs() < 1e-6); // tanh(-10.0) ≈ -1
    /// assert!((b.get(&[1]) - 1.0).abs() < 1e-6); // tanh(10.0) ≈ 1
    /// ```
    pub fn tanh(&self) -> Tensor {
        let mut out = self.tanh_optimized();

        if self.requires_grad() && is_grad_enabled() {
            out.set_requires_grad_internal(true);
            let grad_fn = GradFn::Tanh {
                saved_output: Box::new(out.clone()),
            };
            out.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(out.id(), vec![self.id()], grad_fn);
        }

        out
    }

    /// Internal optimized tanh operation
    ///
    /// Performs element-wise hyperbolic tangent computation using high-precision
    /// scalar implementation for mathematical accuracy and validation.
    ///
    /// # Returns
    ///
    /// A new tensor with tanh applied to each element
    ///
    /// # Performance Characteristics
    ///
    /// - **High Precision**: Accurate scalar implementation for mathematical validation
    /// - **4x Unrolling**: Optimized scalar operations with instruction-level parallelism
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Zero-sized Handling**: Fast return for empty tensors
    /// - **Numerical Stability**: Robust handling of extreme input values
    ///
    /// # Implementation Details
    ///
    /// Uses high-precision scalar implementation rather than SIMD approximations
    /// to ensure mathematical accuracy for validation against reference implementations.
    /// Implements 4x unrolling for better instruction-level parallelism and cache utilization.
    #[inline]
    pub(crate) fn tanh_optimized(&self) -> Tensor {
        let mut output = Tensor::new(self.shape().dims.clone());

        if self.size() == 0 {
            return output;
        }

        unsafe {
            let src = self.as_ptr();
            let dst = output.as_mut_ptr();

            // Use scalar implementation for accuracy
            // SIMD approximations for tanh introduce too much error for validation
            self.tanh_scalar_optimized(src, dst);
        }

        output
    }

    /// Optimized scalar hyperbolic tangent implementation
    ///
    /// Performs element-wise hyperbolic tangent computation using optimized scalar
    /// operations with 4x unrolling for better instruction-level parallelism.
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
    /// - **High Precision**: Accurate scalar implementation for mathematical validation
    /// - **4x Unrolling**: Optimized scalar operations with instruction-level parallelism
    /// - **Memory Access**: Linear access patterns for cache efficiency
    /// - **Fallback**: Handles remaining elements with scalar operations
    /// - **Mathematical Accuracy**: High-precision hyperbolic tangent computation
    ///
    /// # Implementation Details
    ///
    /// Uses 4x unrolled scalar operations for optimal performance while maintaining
    /// high mathematical accuracy. Processes elements in groups of 4 to improve
    /// instruction-level parallelism and reduce loop overhead.
    #[inline]
    unsafe fn tanh_scalar_optimized(&self, src: *const f32, dst: *mut f32) {
        let size = self.size();
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar loop for better performance
        for _ in 0..unroll_count {
            *dst.add(offset) = (*src.add(offset)).tanh();
            *dst.add(offset + 1) = (*src.add(offset + 1)).tanh();
            *dst.add(offset + 2) = (*src.add(offset + 2)).tanh();
            *dst.add(offset + 3) = (*src.add(offset + 3)).tanh();
            offset += 4;
        }

        // Handle remaining elements
        for i in offset..size {
            *dst.add(i) = (*src.add(i)).tanh();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_forward_basic() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 1.0], vec![3]).unwrap();
        let y = x.tanh();
        unsafe {
            assert!((*y.as_ptr() + 0.7615942).abs() < 1e-6);
            assert!((*y.as_ptr().add(1) - 0.0).abs() < 1e-6);
            assert!((*y.as_ptr().add(2) - 0.7615942).abs() < 1e-6);
        }
    }
}
