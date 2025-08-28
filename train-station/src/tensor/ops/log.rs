//! Natural logarithm operations for tensors
//!
//! Provides element-wise natural logarithm function following PyTorch conventions with
//! comprehensive automatic differentiation support and optimized scalar computation.
//!
//! # Key Features
//!
//! - **Natural Logarithm**: `log()` - Computes ln(x) for each element (PyTorch `log()` equivalent)
//! - **Automatic Differentiation**: Full gradtrack support with gradient d/dx log(x) = 1/x
//! - **Optimized Scalar Math**: Uses optimized scalar logarithm for accuracy and simplicity
//! - **Domain Validation**: Automatic validation of positive input values
//! - **Cache Optimization**: Memory access patterns optimized for modern CPUs
//! - **Mathematical Accuracy**: High-precision logarithm computation
//!
//! # Mathematical Properties
//!
//! The natural logarithm function ln(x) has the following properties:
//! - ln(1) = 0
//! - ln(e) = 1 (where e ≈ 2.71828 is Euler's number)
//! - ln(x*y) = ln(x) + ln(y)
//! - ln(x^n) = n * ln(x)
//! - Domain: x > 0 (positive real numbers only)
//! - Gradient: d/dx ln(x) = 1/x
//!
//! # Performance Characteristics
//!
//! - **Scalar Optimization**: Optimized scalar logarithm computation
//! - **Unrolled Loops**: 4x unrolling for optimal instruction throughput
//! - **Cache-friendly Access**: Linear memory access patterns
//! - **Mathematical Accuracy**: High-precision floating-point logarithm
//! - **Domain Validation**: Efficient positive value checking
//! - **Gradient Optimization**: Efficient gradtrack with NoGradTrack support

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Internal optimized logarithm operation
    ///
    /// Performs element-wise natural logarithm computation using optimized scalar math
    /// for maximum accuracy and performance. This is the core implementation
    /// used by `log()`.
    ///
    /// # Returns
    ///
    /// A new tensor containing the natural logarithm of each element
    ///
    /// # Performance Characteristics
    ///
    /// - **Scalar Optimization**: Uses optimized scalar logarithm for accuracy
    /// - **Unrolled Loops**: 4x unrolling for optimal instruction throughput
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Mathematical Accuracy**: High-precision floating-point logarithm
    /// - **Zero-sized Handling**: Fast return for empty tensors
    /// - **Domain Validation**: Efficient positive value checking
    ///
    /// # Implementation Details
    ///
    /// Uses scalar logarithm computation for maximum mathematical accuracy.
    /// Implements 4x unrolled loops for optimal instruction-level parallelism.
    /// Validates that all input values are positive to ensure mathematical correctness.
    ///
    /// # Panics
    ///
    /// Panics if any element is non-positive (x <= 0) as logarithm is undefined
    /// for non-positive real numbers.
    #[inline]
    pub(crate) fn log_optimized(&self) -> Tensor {
        let mut output = Tensor::new(self.shape().dims.clone());

        if self.size() == 0 {
            return output;
        }

        unsafe {
            let src = self.as_ptr();
            let dst = output.as_mut_ptr();
            let size = self.size();
            let mut i = 0;
            // Unrolled scalar loop
            while i + 4 <= size {
                let x0 = *src.add(i);
                let x1 = *src.add(i + 1);
                let x2 = *src.add(i + 2);
                let x3 = *src.add(i + 3);
                assert!(
                    x0 > 0.0 && x1 > 0.0 && x2 > 0.0 && x3 > 0.0,
                    "log domain error: x <= 0"
                );
                *dst.add(i) = x0.ln();
                *dst.add(i + 1) = x1.ln();
                *dst.add(i + 2) = x2.ln();
                *dst.add(i + 3) = x3.ln();
                i += 4;
            }
            while i < size {
                let x = *src.add(i);
                assert!(x > 0.0, "log domain error: x <= 0");
                *dst.add(i) = x.ln();
                i += 1;
            }
        }

        output
    }

    /// Element-wise natural logarithm.
    ///
    /// Computes the natural logarithm for each element: `output[i] = ln(self[i])`
    ///
    /// # Returns
    /// A new tensor with the natural logarithm of each element
    ///
    /// # Examples
    ///
    /// ## Basic Natural Logarithm
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.71828, 7.38906], vec![3]).unwrap();
    /// let b = a.log();
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert_eq!(b.get(&[0]), 0.0); // ln(1) = 0
    /// assert!((b.get(&[1]) - 1.0).abs() < 1e-5); // ln(e) ≈ 1
    /// assert!((b.get(&[2]) - 2.0).abs() < 1e-5); // ln(e^2) ≈ 2
    /// ```
    ///
    /// ## Mathematical Properties
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[4.0, 8.0, 16.0], vec![3]).unwrap();
    /// let b = a.log();
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert!((b.get(&[0]) - 1.38629).abs() < 1e-5); // ln(4) ≈ 1.38629
    /// assert!((b.get(&[1]) - 2.07944).abs() < 1e-5); // ln(8) ≈ 2.07944
    /// assert!((b.get(&[2]) - 2.77259).abs() < 1e-5); // ln(16) ≈ 2.77259
    /// ```
    ///
    /// # Panics
    /// Panics if any element is non-positive (x <= 0)
    #[inline]
    pub fn log(&self) -> Tensor {
        let mut result = self.log_optimized();
        if self.requires_grad() && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Log {
                saved_input: Box::new(self.clone()),
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
    fn test_log_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let x = Tensor::from_slice(&data, vec![2, 2]).unwrap();
        let y = x.log_optimized();
        unsafe {
            let yd = std::slice::from_raw_parts(y.as_ptr(), y.size());
            for i in 0..y.size() {
                assert!((yd[i] - data[i].ln()).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_log_domain_panic() {
        let data = [1.0, 0.0];
        let x = Tensor::from_slice(&data, vec![2]).unwrap();
        let _ = x.log_optimized();
    }

    #[test]
    fn test_log_gradtrack() {
        let x = Tensor::from_slice(&[1.0, 2.0, 4.0], vec![3])
            .unwrap()
            .with_requires_grad();
        let mut y = x.log();
        y.backward(None);
        let gx = x.grad_by_value().expect("grad missing");
        // d/dx log(x) = 1/x
        assert!((gx.get(&[0]) - 1.0).abs() < 1e-6);
        assert!((gx.get(&[1]) - 0.5).abs() < 1e-6);
        assert!((gx.get(&[2]) - 0.25).abs() < 1e-6);
    }
}
