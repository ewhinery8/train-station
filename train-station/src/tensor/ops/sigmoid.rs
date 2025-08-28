//! Sigmoid activation function
//!
//! Provides the sigmoid activation function following PyTorch conventions with
//! comprehensive automatic differentiation support and numerically stable computation.
//!
//! # Key Features
//!
//! - **Sigmoid Activation**: `sigmoid()` - Computes 1/(1+e^(-x)) for each element (PyTorch `sigmoid()` equivalent)
//! - **Automatic Differentiation**: Full gradtrack support with efficient gradient computation
//! - **Numerical Stability**: Avoids overflow for large positive/negative values
//! - **Mathematical Accuracy**: High-precision sigmoid computation
//! - **Range Guarantee**: Output values always in range (0, 1)
//! - **Gradient Optimization**: Efficient gradtrack with NoGradTrack support
//!
//! # Mathematical Properties
//!
//! The sigmoid activation function has the following properties:
//! - **Definition**: f(x) = 1 / (1 + e^(-x))
//! - **Range**: (0, 1) - outputs are always between 0 and 1
//! - **Symmetry**: f(-x) = 1 - f(x) for all x
//! - **Monotonicity**: Strictly increasing function
//! - **Continuity**: Continuous and differentiable everywhere
//! - **Gradient**: f'(x) = f(x) * (1 - f(x)) = sigmoid(x) * (1 - sigmoid(x))
//! - **Limits**: lim(x→-∞) f(x) = 0, lim(x→+∞) f(x) = 1
//!
//! # Performance Characteristics
//!
//! - **Numerical Stability**: Avoids overflow using stable implementation
//! - **Scalar Implementation**: Optimized scalar computation for mathematical accuracy
//! - **Cache-friendly Access**: Linear memory access patterns
//! - **Mathematical Accuracy**: High-precision exponential and division operations
//! - **Gradient Optimization**: Efficient gradtrack with NoGradTrack support

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Element-wise sigmoid activation function
    ///
    /// Computes the sigmoid function for each element: `output[i] = 1 / (1 + e^(-self[i]))`
    ///
    /// Uses a numerically stable implementation that avoids overflow for large positive/negative
    /// values by using different computation paths for positive and negative inputs.
    ///
    /// # Returns
    ///
    /// A new tensor with sigmoid applied to each element, values in range (0, 1)
    ///
    /// # Performance Characteristics
    ///
    /// - **Numerical Stability**: Avoids overflow using stable implementation
    /// - **Scalar Implementation**: Optimized scalar computation for mathematical accuracy
    /// - **Cache-friendly**: Linear memory access patterns
    /// - **Mathematical Accuracy**: High-precision exponential and division operations
    /// - **Gradient Tracking**: Full gradtrack support with efficient gradient computation
    ///
    /// # Implementation Details
    ///
    /// Uses a numerically stable implementation:
    /// - For x ≥ 0: computes 1 / (1 + e^(-x)) to avoid overflow in e^x for large positive x
    /// - For x < 0: computes e^x / (1 + e^x) to avoid overflow in e^(-x) for large negative x
    ///   This ensures the result is always in the range (0, 1) without numerical overflow.
    ///
    /// # Examples
    ///
    /// ## Basic Sigmoid Activation
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[-1.0, 0.0, 1.0], vec![3]).unwrap();
    /// let b = a.sigmoid();
    /// assert_eq!(b.shape().dims, vec![3]);
    /// assert!((b.get(&[0]) - 0.26894143).abs() < 1e-6); // sigmoid(-1.0)
    /// assert!((b.get(&[1]) - 0.5).abs() < 1e-6); // sigmoid(0.0)
    /// assert!((b.get(&[2]) - 0.7310586).abs() < 1e-6); // sigmoid(1.0)
    /// ```
    ///
    /// ## Extreme Values
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[-10.0, 10.0], vec![2]).unwrap();
    /// let b = a.sigmoid();
    /// assert_eq!(b.shape().dims, vec![2]);
    /// assert!(b.get(&[0]) < 1e-4); // sigmoid(-10.0) ≈ 0
    /// assert!(b.get(&[1]) > 0.9999); // sigmoid(10.0) ≈ 1
    /// ```
    pub fn sigmoid(&self) -> Tensor {
        let mut out = Tensor::new(self.shape().dims.clone());
        unsafe {
            let src = self.as_ptr();
            let dst = out.as_mut_ptr();
            let n = self.size();
            for i in 0..n {
                let x = *src.add(i);
                // Stable sigmoid
                let y = if x >= 0.0 {
                    let z = (-x).exp();
                    1.0 / (1.0 + z)
                } else {
                    let z = x.exp();
                    z / (1.0 + z)
                };
                *dst.add(i) = y;
            }
        }

        if self.requires_grad() && is_grad_enabled() {
            let mut result = out.clone();
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Sigmoid {
                saved_output: Box::new(out.clone()),
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
            return result;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_forward_basic() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 1.0], vec![3]).unwrap();
        let y = x.sigmoid();
        unsafe {
            assert!((*y.as_ptr() - 0.26894143).abs() < 1e-6);
            assert!((*y.as_ptr().add(1) - 0.5).abs() < 1e-6);
            assert!((*y.as_ptr().add(2) - 0.7310586).abs() < 1e-6);
        }
    }
}
