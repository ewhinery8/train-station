//! Softmax activation function
//!
//! Provides the softmax activation function following PyTorch conventions with
//! comprehensive GradTrack support and numerically stable computation.
//!
//! # Key Features
//!
//! - **Softmax Activation**: `softmax(dim)` - Computes softmax along specified dimension (PyTorch `softmax()` equivalent)
//! - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
//! - **Numerical Stability**: Avoids overflow using max subtraction technique
//! - **Mathematical Accuracy**: High-precision softmax computation
//! - **Dimension Flexibility**: Supports softmax along any dimension
//! - **Probability Output**: Values sum to 1 along the specified dimension
//!
//! # Mathematical Properties
//!
//! The softmax activation function has the following properties:
//! - **Definition**: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//! - **Range**: (0, 1) - outputs are always positive and sum to 1
//! - **Numerical Stability**: Subtracts max value to prevent overflow
//! - **Monotonicity**: Preserves relative ordering of input values
//! - **Continuity**: Continuous and differentiable everywhere
//! - **Gradient**: Complex gradient computation involving the softmax output
//! - **Probability Interpretation**: Outputs can be interpreted as probabilities
//!
//! # Performance Characteristics
//!
//! - **Numerical Stability**: Avoids overflow using max subtraction technique
//! - **Scalar Implementation**: Optimized scalar computation for mathematical accuracy
//! - **Cache-friendly Access**: Optimized memory access patterns for dimension operations
//! - **Mathematical Accuracy**: High-precision exponential and division operations
//! - **GradTrack Optimization**: Efficient automatic differentiation with NoGradTrack support

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Computes softmax activation along the specified dimension
    ///
    /// Applies the softmax function along dimension `dim`, transforming values into
    /// probabilities that sum to 1 along that dimension. Uses numerically stable
    /// computation to avoid overflow: `softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension along which to compute softmax (0-based indexing)
    ///
    /// # Returns
    ///
    /// A new tensor with softmax applied along the specified dimension.
    /// Values are in range (0, 1) and sum to 1 along `dim`.
    ///
    /// # Performance Characteristics
    ///
    /// - **Numerical Stability**: Avoids overflow using max subtraction technique
    /// - **Scalar Implementation**: Optimized scalar computation for mathematical accuracy
    /// - **Cache-friendly**: Optimized memory access patterns for dimension operations
    /// - **Mathematical Accuracy**: High-precision exponential and division operations
    /// - **GradTrack Support**: Full automatic differentiation with efficient gradient computation
    ///
    /// # Implementation Details
    ///
    /// Uses a numerically stable three-pass algorithm:
    /// 1. **Max Computation**: Find the maximum value along the specified dimension
    /// 2. **Exponential Sum**: Compute exp(x - max) and sum the results
    /// 3. **Normalization**: Divide each exp(x - max) by the sum to get probabilities
    ///
    /// This approach prevents overflow by subtracting the maximum value before
    /// computing exponentials, ensuring numerical stability for any input range.
    ///
    /// # Examples
    ///
    /// ## Basic Softmax Activation
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = a.softmax(0);
    /// assert_eq!(b.shape().dims, vec![3]);
    ///
    /// // Verify probabilities sum to 1
    /// let sum = b.get(&[0]) + b.get(&[1]) + b.get(&[2]);
    /// assert!((sum - 1.0).abs() < 1e-6);
    ///
    /// // Verify relative ordering is preserved
    /// assert!(b.get(&[0]) < b.get(&[1]));
    /// assert!(b.get(&[1]) < b.get(&[2]));
    /// ```
    ///
    /// ## 2D Softmax Along Different Dimensions
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let b = a.softmax(0); // Softmax along first dimension
    /// assert_eq!(b.shape().dims, vec![2, 2]);
    ///
    /// // Each column should sum to 1
    /// let col1_sum = b.get(&[0, 0]) + b.get(&[1, 0]);
    /// let col2_sum = b.get(&[0, 1]) + b.get(&[1, 1]);
    /// assert!((col1_sum - 1.0).abs() < 1e-6);
    /// assert!((col2_sum - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// # Panics
    /// - Panics if `dim` is out of bounds for the tensor's rank
    /// - Panics if the dimension size is 0
    pub fn softmax(&self, dim: usize) -> Tensor {
        let rank = self.shape().rank();
        assert!(
            dim < rank,
            "softmax dim {} out of bounds for rank {}",
            dim,
            rank
        );
        let dims = self.shape().dims.clone();
        let reduce = dims[dim];
        assert!(reduce > 0, "cannot softmax over empty dimension");

        let inner: usize = dims[dim + 1..].iter().product();
        let outer: usize = dims[..dim].iter().product();

        let mut out = Tensor::new(dims.clone());
        unsafe {
            let xptr = self.as_ptr();
            let yptr = out.as_mut_ptr();
            // For each slice along `dim`, find max then compute exp and sum, then normalize
            for o in 0..outer {
                for i in 0..inner {
                    // 1) max
                    let mut maxv = f32::NEG_INFINITY;
                    for j in 0..reduce {
                        let off = o * (reduce * inner) + j * inner + i;
                        let v = *xptr.add(off);
                        if v > maxv {
                            maxv = v;
                        }
                    }
                    // 2) exp sum
                    let mut sum = 0.0f32;
                    for j in 0..reduce {
                        let off = o * (reduce * inner) + j * inner + i;
                        let e = (*xptr.add(off) - maxv).exp();
                        *yptr.add(off) = e;
                        sum += e;
                    }
                    // 3) normalize
                    let inv = 1.0f32 / sum;
                    for j in 0..reduce {
                        let off = o * (reduce * inner) + j * inner + i;
                        *yptr.add(off) *= inv;
                    }
                }
            }
        }

        if self.requires_grad() && is_grad_enabled() {
            let mut result = out.clone();
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Softmax {
                dim,
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
    fn test_softmax_forward_basic() {
        let x = Tensor::from_slice(&[0.0, 1.0, 2.0], vec![3]).unwrap();
        let y = x.softmax(0);
        let s = y.sum();
        unsafe {
            assert!((*s.as_ptr() - 1.0).abs() < 1e-6);
        }
    }
}
