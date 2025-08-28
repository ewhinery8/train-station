//! L2 norm reduction operations for tensors
//!
//! This module provides L2 norm (Euclidean norm) reduction operations for tensors.
//! The L2 norm computes the square root of the sum of squared elements, which is
//! commonly used in machine learning for regularization, distance calculations,
//! and gradient clipping.
//!
//! # Operations
//!
//! * `norm()` - Computes L2 norm over all elements, returning a scalar tensor
//! * `norm_dims()` - Computes L2 norm over specified dimensions with optional dimension preservation
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Compute L2 norm of all elements
//! let tensor = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
//! let norm = tensor.norm();
//! assert!((norm.get(&[0]) - 5.0).abs() < 1e-6); // sqrt(3² + 4²) = 5
//!
//! // Compute L2 norm along specific dimensions
//! let matrix = Tensor::from_slice(&[3.0, 4.0, 0.0, 5.0], vec![2, 2]).unwrap();
//! let row_norms = matrix.norm_dims(&[1], true);
//! assert_eq!(row_norms.shape().dims, vec![2, 1]);
//! ```
//!
//! # Performance
//!
//! The implementation uses optimized paths for contiguous tensors with manual loop unrolling
//! for better performance. Non-contiguous tensors use stride-aware iteration to maintain
//! correctness while preserving memory layout efficiency.
//!
//! # Gradient Tracking
//!
//! Both operations support automatic gradient tracking when `requires_grad` is enabled.
//! The gradient computation follows the mathematical derivative of the L2 norm operation.

use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Computes the L2 norm (Euclidean norm) over all elements
    ///
    /// The L2 norm is calculated as sqrt(sum(x²)) where x represents each element
    /// in the tensor. This operation reduces the tensor to a scalar value \[1\].
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the L2 norm value
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Basic L2 norm calculation
    /// let tensor = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
    /// let norm = tensor.norm();
    /// assert!((norm.get(&[0]) - 5.0).abs() < 1e-6); // sqrt(3² + 4²) = 5
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // L2 norm of a larger tensor
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let tensor = Tensor::from_slice(&data, vec![2, 2, 2]).unwrap();
    /// let norm = tensor.norm();
    /// // sqrt(1² + 2² + 3² + 4² + 5² + 6² + 7² + 8²) = sqrt(204) ≈ 14.283
    /// let expected = 204.0_f32.sqrt();
    /// assert!((norm.get(&[0]) - expected).abs() < 1e-5);
    /// ```
    ///
    /// # Performance
    ///
    /// Uses optimized contiguous tensor path with 4x loop unrolling for better
    /// performance. Non-contiguous tensors use stride-aware iteration.
    pub fn norm(&self) -> Tensor {
        // Compute sqrt(sum(x^2))
        let mut sumsq = 0.0f32;
        let n = self.size();

        if self.is_contiguous() {
            // Fast path for contiguous tensors
            unsafe {
                let src = self.as_ptr();
                let mut i = 0usize;
                while i + 4 <= n {
                    let x0 = *src.add(i);
                    let x1 = *src.add(i + 1);
                    let x2 = *src.add(i + 2);
                    let x3 = *src.add(i + 3);
                    sumsq += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
                    i += 4;
                }
                while i < n {
                    let v = *src.add(i);
                    sumsq += v * v;
                    i += 1;
                }
            }
        } else {
            // Stride-aware path for non-contiguous tensors
            let dims = self.shape().dims.clone();
            for flat_idx in 0..n {
                // Convert flat index to multi-dimensional coordinates
                let mut coords = vec![0; dims.len()];
                let mut tmp = flat_idx;
                for k in (0..dims.len()).rev() {
                    coords[k] = tmp % dims[k];
                    tmp /= dims[k];
                }

                // Get value using stride-aware offset
                let offset = self.shape().offset(&coords);
                let value = unsafe { *self.as_ptr().add(offset) };
                sumsq += value * value;
            }
        }
        let mut out = Tensor::new(vec![1]);
        unsafe {
            *out.as_mut_ptr() = sumsq.sqrt();
        }

        if self.requires_grad() {
            let mut result = out.clone();
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::ReduceNorm {
                saved_norm: Box::new(out.clone()),
                saved_input: Box::new(self.clone()),
                input_shape: self.shape().dims.clone(),
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
            return result;
        }

        out
    }

    /// Computes the L2 norm over specified dimensions
    ///
    /// Reduces the tensor along the specified dimensions by computing the L2 norm
    /// of each slice. The result maintains the original tensor structure with
    /// reduced dimensions optionally preserved as size-1 dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector of dimension indices to reduce over (must be valid for tensor rank)
    /// * `keepdim` - Whether to keep reduced dimensions as size-1 dimensions
    ///
    /// # Returns
    ///
    /// A tensor with L2 norm computed over the specified dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Norm along rows (dimension 1) with keepdim=true
    /// let matrix = Tensor::from_slice(&[3.0, 4.0, 0.0, 5.0], vec![2, 2]).unwrap();
    /// let row_norms = matrix.norm_dims(&[1], true);
    /// assert_eq!(row_norms.shape().dims, vec![2, 1]);
    /// assert!((row_norms.get(&[0, 0]) - 5.0).abs() < 1e-6); // sqrt(3² + 4²)
    /// assert!((row_norms.get(&[1, 0]) - 5.0).abs() < 1e-6); // sqrt(0² + 5²)
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Norm along columns (dimension 0) with keepdim=false
    /// let matrix = Tensor::from_slice(&[3.0, 4.0, 0.0, 5.0], vec![2, 2]).unwrap();
    /// let col_norms = matrix.norm_dims(&[0], false);
    /// assert_eq!(col_norms.shape().dims, vec![2]);
    /// assert!((col_norms.get(&[0]) - 3.0).abs() < 1e-6); // sqrt(3² + 0²)
    /// assert!((col_norms.get(&[1]) - 6.403).abs() < 1e-3); // sqrt(4² + 5²)
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Norm over multiple dimensions
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let norm_all = tensor.norm_dims(&[0, 1], false);
    /// assert_eq!(norm_all.shape().dims, vec![1]);
    /// // sqrt(1² + 2² + 3² + 4²) = sqrt(30) ≈ 5.477
    /// assert!((norm_all.get(&[0]) - 30.0_f32.sqrt()).abs() < 1e-5);
    /// ```
    ///
    /// # Panics
    ///
    /// * If `dims` is empty
    /// * If any dimension index is out of bounds for the tensor rank
    ///
    /// # Performance
    ///
    /// Uses efficient coordinate-based iteration that works correctly with
    /// both contiguous and non-contiguous tensor layouts.
    pub fn norm_dims(&self, dims: &[usize], keepdim: bool) -> Tensor {
        assert!(
            !dims.is_empty(),
            "norm_dims requires at least one dimension"
        );
        let rank = self.shape().rank();
        for &d in dims {
            assert!(
                d < rank,
                "norm_dims dim {} out of bounds for rank {}",
                d,
                rank
            );
        }

        // Build output shape
        let in_shape = self.shape().dims.clone();
        let mut out_dims = in_shape.clone();
        let mut reduced: Vec<usize> = dims.to_vec();
        reduced.sort_unstable();
        reduced.dedup();
        for &d in reduced.iter() {
            out_dims[d] = if keepdim { 1 } else { 0 };
        }
        if !keepdim {
            out_dims.retain(|&s| s != 0);
        }
        if out_dims.is_empty() {
            out_dims.push(1);
        }
        let mut out = Tensor::zeros(out_dims.clone());

        // Compute sum of squares reduced, then sqrt
        let out_rank = out.shape().rank();
        let mut coords = vec![0usize; rank];
        unsafe {
            let sptr = out.as_mut_ptr();
            for lin in 0..self.size() {
                let mut tmp = lin;
                for i in (0..rank).rev() {
                    let s = in_shape[i];
                    coords[i] = if s == 0 { 0 } else { tmp % s };
                    if s != 0 {
                        tmp /= s;
                    }
                }
                let mut out_coords: Vec<usize> = Vec::with_capacity(out_rank);
                for (i, &c) in coords.iter().enumerate().take(rank) {
                    if reduced.contains(&i) {
                        if keepdim {
                            out_coords.push(0);
                        }
                    } else {
                        out_coords.push(c);
                    }
                }
                let off = if out_coords.is_empty() {
                    0
                } else {
                    out.shape().offset(&out_coords)
                };
                // Get input value using stride-aware offset
                let in_offset = self.shape().offset(&coords);
                let v = *self.as_ptr().add(in_offset);
                *sptr.add(off) += v * v;
            }
            // sqrt in place
            for i in 0..out.size() {
                *sptr.add(i) = (*sptr.add(i)).sqrt();
            }
        }

        if self.requires_grad() {
            let mut result = out.clone();
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::ReduceNormDims {
                dims: reduced,
                keepdim,
                input_shape: self.shape().dims.clone(),
                saved_norm: Box::new(out.clone()),
                saved_input: Box::new(self.clone()),
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
    fn test_norm_forward_basic() {
        let x = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();
        let n = x.norm();
        unsafe {
            assert!((*n.as_ptr() - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_norm_dims_forward() {
        let x = Tensor::from_slice(&[3.0, 4.0, 0.0, 5.0], vec![2, 2]).unwrap();
        let n = x.norm_dims(&[1], true);
        assert_eq!(n.shape().dims, vec![2, 1]);
        assert!((n.get(&[0, 0]) - 5.0).abs() < 1e-6);
        assert!((n.get(&[1, 0]) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm_non_contiguous_transpose() {
        // Test norm on transposed tensor (non-contiguous view)
        let x = Tensor::from_slice(&[3.0, 4.0, 0.0, 12.0, 5.0, 0.0], vec![2, 3]).unwrap();
        // Original: [[3, 4, 0], [12, 5, 0]]

        let x_t = x.transpose(0, 1);
        // Transposed: [[3, 12], [4, 5], [0, 0]]
        assert!(!x_t.is_contiguous()); // Should be a view

        let norm_orig = x.norm();
        let norm_view = x_t.norm();

        // Both should give the same result
        assert!((norm_orig.get(&[0]) - norm_view.get(&[0])).abs() < 1e-6);

        // Expected norm of [3,4,0,12,5,0]: sqrt(3²+4²+0²+12²+5²+0²) = sqrt(9+16+144+25) = sqrt(194) ≈ 13.928
        let expected_norm = 194.0_f32.sqrt();
        assert!((norm_orig.get(&[0]) - expected_norm).abs() < 1e-5);
    }

    #[test]
    fn test_norm_dims_non_contiguous() {
        // Test norm_dims on non-contiguous tensor
        let x = Tensor::from_slice(&[3.0, 4.0, 0.0, 12.0, 5.0, 0.0], vec![2, 3]).unwrap();
        let x_t = x.transpose(0, 1); // [3, 2]
        assert!(!x_t.is_contiguous());

        // Norm along dim 0 of transposed tensor
        let norm_dim0 = x_t.norm_dims(&[0], false);
        assert_eq!(norm_dim0.shape().dims, vec![2]);

        // For dim 0: [3,4,0] and [12,5,0]
        // norm([3,4,0]) = sqrt(3²+4²+0²) = sqrt(25) = 5
        // norm([12,5,0]) = sqrt(12²+5²+0²) = sqrt(169) = 13
        assert!((norm_dim0.get(&[0]) - 5.0).abs() < 1e-6);
        assert!((norm_dim0.get(&[1]) - 13.0).abs() < 1e-6);

        // Norm along dim 1 of transposed tensor
        let norm_dim1 = x_t.norm_dims(&[1], false);
        assert_eq!(norm_dim1.shape().dims, vec![3]);
        // norm([3,12]) = sqrt(9+144) = sqrt(153) ≈ 12.369
        // norm([4,5]) = sqrt(16+25) = sqrt(41) ≈ 6.403
        // norm([0,0]) = sqrt(0+0) = 0
        assert!((norm_dim1.get(&[0]) - 153.0_f32.sqrt()).abs() < 1e-5);
        assert!((norm_dim1.get(&[1]) - 41.0_f32.sqrt()).abs() < 1e-5);
        assert!((norm_dim1.get(&[2]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm_permuted_tensor() {
        // Test with permuted tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = Tensor::from_slice(&data, vec![2, 2, 2]).unwrap();

        // Permute dimensions [2, 2, 2] -> [2, 2, 2] (swap first and last)
        let x_perm = x.permute(vec![2, 1, 0]);
        assert!(!x_perm.is_contiguous());

        let norm_orig = x.norm();
        let norm_perm = x_perm.norm();

        // Should give same result
        assert!((norm_orig.get(&[0]) - norm_perm.get(&[0])).abs() < 1e-6);

        // norm([1,2,3,4,5,6,7,8]) = sqrt(1+4+9+16+25+36+49+64) = sqrt(204) ≈ 14.283
        let expected_norm = 204.0_f32.sqrt();
        assert!((norm_orig.get(&[0]) - expected_norm).abs() < 1e-5);
    }
}
