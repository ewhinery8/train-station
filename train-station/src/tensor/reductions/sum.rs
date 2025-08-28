//! Sum reduction operations for tensors
//!
//! This module provides sum reduction operations that compute the sum of tensor elements.
//! These operations support both global summation and dimension-wise summation with
//! automatic gradient tracking when enabled.
//!
//! # Operations
//!
//! * `sum()` - Sum all elements into a scalar tensor
//! * `sum_dims()` - Sum elements along specified dimensions
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let total = tensor.sum();
//! assert_eq!(total.get(&[0]), 10.0); // 1 + 2 + 3 + 4 = 10
//! ```

use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Returns the sum of all elements in the tensor
    ///
    /// This operation computes the sum of all elements across all dimensions,
    /// reducing the tensor to a scalar value. The output is a tensor with shape \[1\]
    /// containing the sum as a float.
    ///
    /// When `requires_grad` is enabled, this operation supports automatic gradient
    /// tracking through the GradTrack system.
    ///
    /// # Returns
    ///
    /// A tensor with shape \[1\] containing the sum of all elements
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Basic sum calculation
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let total = tensor.sum();
    /// assert_eq!(total.shape().dims, vec![1]);
    /// assert_eq!(total.get(&[0]), 10.0); // 1 + 2 + 3 + 4 = 10
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Sum with gradient tracking
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3])
    ///     .unwrap()
    ///     .with_requires_grad();
    /// let mut total = tensor.sum();
    /// total.backward(None);
    /// let grad = tensor.grad_by_value().expect("gradient should exist");
    /// // Gradient should be [1.0, 1.0, 1.0] for each element
    /// assert_eq!(grad.get(&[0]), 1.0);
    /// assert_eq!(grad.get(&[1]), 1.0);
    /// assert_eq!(grad.get(&[2]), 1.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Sum of empty tensor
    /// let tensor = Tensor::new(vec![0]);
    /// let total = tensor.sum();
    /// assert_eq!(total.get(&[0]), 0.0); // Sum of empty tensor is 0
    /// ```
    ///
    /// # Performance
    ///
    /// Uses optimized contiguous tensor path with 4x loop unrolling for better
    /// performance. Non-contiguous tensors use stride-aware iteration.
    pub fn sum(&self) -> Tensor {
        let mut out = Tensor::new(vec![1]);
        if self.size() == 0 {
            out.fill(0.0);
        } else {
            let mut acc0 = 0.0f32;

            if self.is_contiguous() {
                // Fast path for contiguous tensors
                unsafe {
                    let src = self.as_ptr();
                    let size = self.size();
                    let mut i = 0usize;
                    // Unrolled loop for better throughput
                    while i + 4 <= size {
                        let x0 = *src.add(i);
                        let x1 = *src.add(i + 1);
                        let x2 = *src.add(i + 2);
                        let x3 = *src.add(i + 3);
                        acc0 += x0 + x1 + x2 + x3;
                        i += 4;
                    }
                    while i < size {
                        acc0 += *src.add(i);
                        i += 1;
                    }
                }
            } else {
                // Stride-aware path for non-contiguous tensors
                let dims = self.shape().dims.clone();
                for flat_idx in 0..self.size() {
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
                    acc0 += value;
                }
            }

            unsafe {
                *out.as_mut_ptr() = acc0;
            }
        }

        if self.requires_grad() {
            out.set_requires_grad_internal(true);
            let grad_fn = GradFn::ReduceSum {
                input_shape: self.shape().dims.clone(),
            };
            out.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(out.id(), vec![self.id()], grad_fn);
        }

        out
    }

    /// Returns the sum of elements along specified dimensions
    ///
    /// This operation computes the sum of elements along the specified dimensions,
    /// reducing the tensor while optionally preserving the reduced dimensions as
    /// size-1 dimensions.
    ///
    /// The output shape depends on the `keepdim` parameter:
    /// * If `keepdim` is `true`, the reduced dimensions are kept with size 1
    /// * If `keepdim` is `false`, the reduced dimensions are removed
    ///
    /// When `requires_grad` is enabled, this operation supports automatic gradient
    /// tracking through the GradTrack system.
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector of dimension indices to sum over (must be valid for tensor rank)
    /// * `keepdim` - Whether to keep reduced dimensions as size-1 dimensions
    ///
    /// # Returns
    ///
    /// A tensor with sum computed over the specified dimensions
    ///
    /// # Panics
    ///
    /// * If `dims` is empty
    /// * If any dimension index is out of bounds for the tensor rank
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Sum along rows (dimension 0) with keepdim=false
    /// let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let row_sums = matrix.sum_dims(&[0], false);
    /// assert_eq!(row_sums.shape().dims, vec![2]);
    /// assert_eq!(row_sums.get(&[0]), 4.0); // 1 + 3 = 4
    /// assert_eq!(row_sums.get(&[1]), 6.0); // 2 + 4 = 6
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Sum along columns (dimension 1) with keepdim=true
    /// let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let col_sums = matrix.sum_dims(&[1], true);
    /// assert_eq!(col_sums.shape().dims, vec![2, 1]);
    /// assert_eq!(col_sums.get(&[0, 0]), 3.0); // 1 + 2 = 3
    /// assert_eq!(col_sums.get(&[1, 0]), 7.0); // 3 + 4 = 7
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Sum over multiple dimensions
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let total = tensor.sum_dims(&[0, 1], false);
    /// assert_eq!(total.shape().dims, vec![1]);
    /// assert_eq!(total.get(&[0]), 10.0); // 1 + 2 + 3 + 4 = 10
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Sum with gradient tracking
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])
    ///     .unwrap()
    ///     .with_requires_grad();
    /// let mut row_sums = tensor.sum_dims(&[0], false);
    /// row_sums.backward(None);
    /// let grad = tensor.grad_by_value().expect("gradient should exist");
    /// // Gradient should be [1.0, 1.0, 1.0, 1.0] for each element
    /// assert_eq!(grad.get(&[0, 0]), 1.0);
    /// assert_eq!(grad.get(&[0, 1]), 1.0);
    /// assert_eq!(grad.get(&[1, 0]), 1.0);
    /// assert_eq!(grad.get(&[1, 1]), 1.0);
    /// ```
    ///
    /// # Performance
    ///
    /// Uses efficient coordinate-based iteration that works correctly with
    /// both contiguous and non-contiguous tensor layouts.
    pub fn sum_dims(&self, dims: &[usize], keepdim: bool) -> Tensor {
        assert!(!dims.is_empty(), "sum_dims requires at least one dimension");
        let rank = self.shape().rank();
        for &d in dims {
            assert!(
                d < rank,
                "sum_dims dim {} out of bounds for rank {}",
                d,
                rank
            );
        }

        // Build output shape
        let mut out_dims = self.shape().dims.clone();
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

        // Accumulate along reduced dims
        let in_shape = self.shape().dims.clone();
        let out_rank = out.shape().rank();
        let mut in_coords = vec![0usize; rank];
        unsafe {
            let dst = out.as_mut_ptr();
            // Iterate over all input elements, map to output index
            for lin in 0..self.size() {
                let mut tmp = lin;
                for i in (0..rank).rev() {
                    let s = in_shape[i];
                    in_coords[i] = if s == 0 { 0 } else { tmp % s };
                    if s != 0 {
                        tmp /= s;
                    }
                }

                // Get input value using stride-aware offset
                let in_offset = self.shape().offset(&in_coords);
                let value = *self.as_ptr().add(in_offset);

                // build output coords
                let mut out_coords: Vec<usize> = Vec::with_capacity(out_rank);
                for (i, &c) in in_coords.iter().enumerate().take(rank) {
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
                *dst.add(off) += value;
            }
        }

        if self.requires_grad() {
            out.set_requires_grad_internal(true);
            let grad_fn = GradFn::ReduceSumDims {
                dims: reduced,
                input_shape: self.shape().dims.clone(),
                keepdim,
            };
            out.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(out.id(), vec![self.id()], grad_fn);
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_forward_basic() {
        let mut x = Tensor::zeros(vec![2, 3]);
        unsafe {
            for i in 0..6 {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.5;
            }
        }
        let s = x.sum();
        assert_eq!(s.shape().dims, vec![1]);
        unsafe {
            assert!((*s.as_ptr() - 7.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sum_autograd_all_ones_grad() {
        let mut x = Tensor::zeros(vec![2, 2]).with_requires_grad();
        unsafe {
            for i in 0..4 {
                *x.as_mut_ptr().add(i) = i as f32;
            }
        }
        let mut s = x.sum();
        s.backward(None);
        let gx = x.grad_by_value().expect("grad missing");
        for i in 0..4 {
            unsafe {
                assert_eq!(*gx.as_ptr().add(i), 1.0);
            }
        }
    }

    #[test]
    fn test_sum_chain_autograd() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .unwrap()
            .with_requires_grad();
        let y = x.mul_scalar(2.0).add_scalar(1.0);
        let mut s = y.sum();
        s.backward(None);
        let gx = x.grad_by_value().expect("grad missing");
        // d/dx of sum(2x+1) = 2 for each element
        for i in 0..4 {
            unsafe {
                assert_eq!(*gx.as_ptr().add(i), 2.0);
            }
        }
    }

    #[test]
    fn test_sum_non_contiguous_transpose() {
        // Test sum on transposed tensor (non-contiguous view)
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        // Original: [[1, 2, 3], [4, 5, 6]]

        let x_t = x.transpose(0, 1);
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert!(!x_t.is_contiguous()); // Should be a view

        let sum_orig = x.sum();
        let sum_view = x_t.sum();

        // Both should give the same result: 1+2+3+4+5+6 = 21
        assert_eq!(sum_orig.get(&[0]), 21.0);
        assert_eq!(sum_view.get(&[0]), 21.0);
    }

    #[test]
    fn test_sum_dims_non_contiguous() {
        // Test sum_dims on non-contiguous tensor
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let x_t = x.transpose(0, 1); // [3, 2]
        assert!(!x_t.is_contiguous());

        // Sum along dim 0 of transposed tensor
        let sum_dim0 = x_t.sum_dims(&[0], false);
        assert_eq!(sum_dim0.shape().dims, vec![2]);
        // Should be [1+2+3, 4+5+6] = [6, 15]
        assert_eq!(sum_dim0.get(&[0]), 6.0);
        assert_eq!(sum_dim0.get(&[1]), 15.0);

        // Sum along dim 1 of transposed tensor
        let sum_dim1 = x_t.sum_dims(&[1], false);
        assert_eq!(sum_dim1.shape().dims, vec![3]);
        // Should be [1+4, 2+5, 3+6] = [5, 7, 9]
        assert_eq!(sum_dim1.get(&[0]), 5.0);
        assert_eq!(sum_dim1.get(&[1]), 7.0);
        assert_eq!(sum_dim1.get(&[2]), 9.0);
    }

    #[test]
    fn test_sum_permuted_tensor() {
        // Test with permuted tensor
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();

        // Permute dimensions [2, 3, 4] -> [4, 2, 3]
        let x_perm = x.permute(vec![2, 1, 0]);
        assert!(!x_perm.is_contiguous());

        let sum_orig = x.sum();
        let sum_perm = x_perm.sum();

        // Should give same result
        assert_eq!(sum_orig.get(&[0]), sum_perm.get(&[0]));

        // Expected sum: 0+1+2+...+23 = 23*24/2 = 276
        assert_eq!(sum_orig.get(&[0]), 276.0);
    }
}
