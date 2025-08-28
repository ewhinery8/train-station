use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Computes the minimum value over all elements in the tensor
    ///
    /// Returns a scalar tensor containing the minimum value. For empty tensors,
    /// returns positive infinity. This operation supports gradient tracking
    /// through the GradTrack system.
    ///
    /// # Returns
    ///
    /// A tensor with shape `[1]` containing the minimum value
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 5.0, 3.0, 2.0], vec![2, 2]).unwrap();
    /// let min_val = tensor.min();
    /// assert_eq!(min_val.get(&[0]), 1.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Empty tensor case
    /// let empty_tensor = Tensor::new(vec![0]);
    /// let min_val = empty_tensor.min();
    /// assert_eq!(min_val.get(&[0]), f32::INFINITY);
    /// ```
    ///
    /// # GradTrack Support
    ///
    /// When `requires_grad` is true, this operation is tracked for automatic
    /// differentiation. The gradient computation uses the saved input and output
    /// for efficient backward pass.
    pub fn min(&self) -> Tensor {
        let mut out = Tensor::new(vec![1]);
        if self.size() == 0 {
            out.fill(f32::INFINITY);
        } else {
            let mut m = f32::INFINITY;

            if self.is_contiguous() {
                // Fast path for contiguous tensors
                unsafe {
                    let src = self.as_ptr();
                    let size = self.size();
                    m = *src;
                    let mut i = 1usize;
                    while i + 4 <= size {
                        let x0 = *src.add(i);
                        let x1 = *src.add(i + 1);
                        let x2 = *src.add(i + 2);
                        let x3 = *src.add(i + 3);
                        m = m.min(x0).min(x1).min(x2).min(x3);
                        i += 4;
                    }
                    while i < size {
                        m = m.min(*src.add(i));
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
                    if flat_idx == 0 {
                        m = value;
                    } else {
                        m = m.min(value);
                    }
                }
            }

            unsafe {
                *out.as_mut_ptr() = m;
            }
        }

        if self.requires_grad() {
            let mut result = out.clone();
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::ReduceMin {
                saved_output: Box::new(out.clone()),
                saved_input: Box::new(self.clone()),
                input_shape: self.shape().dims.clone(),
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
            return result;
        }

        out
    }

    /// Computes the minimum value over specified dimensions
    ///
    /// Reduces the tensor along the specified dimensions by computing the minimum
    /// value in each reduction group. The `keepdim` parameter determines whether
    /// reduced dimensions are kept with size 1 or removed entirely.
    ///
    /// # Arguments
    ///
    /// * `dims` - Dimensions to reduce over (must be valid for the tensor's rank)
    /// * `keepdim` - If true, reduced dimensions are kept with size 1; if false, they are removed
    ///
    /// # Returns
    ///
    /// A tensor with the specified dimensions reduced
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    ///
    /// // Min over columns (dim 1), keeping dimensions
    /// let min_cols = tensor.min_dims(&[1], true);
    /// assert_eq!(min_cols.shape().dims, vec![2, 1]);
    /// assert_eq!(min_cols.get(&[0, 0]), 1.0);
    /// assert_eq!(min_cols.get(&[1, 0]), 4.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    ///
    /// // Min over rows (dim 0), removing dimensions
    /// let min_rows = tensor.min_dims(&[0], false);
    /// assert_eq!(min_rows.shape().dims, vec![3]);
    /// assert_eq!(min_rows.get(&[0]), 1.0);
    /// assert_eq!(min_rows.get(&[1]), 2.0);
    /// assert_eq!(min_rows.get(&[2]), 3.0);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    ///
    /// // Min over multiple dimensions
    /// let min_all = tensor.min_dims(&[0, 1], false);
    /// assert_eq!(min_all.shape().dims, vec![1]);
    /// assert_eq!(min_all.get(&[0]), 1.0);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `dims` is empty
    /// * Any dimension in `dims` is out of bounds for the tensor's rank
    ///
    /// # GradTrack Support
    ///
    /// When `requires_grad` is true, this operation is tracked for automatic
    /// differentiation. The gradient computation preserves the original input
    /// shape and handles broadcasting correctly.
    pub fn min_dims(&self, dims: &[usize], keepdim: bool) -> Tensor {
        assert!(!dims.is_empty(), "min_dims requires at least one dimension");
        let rank = self.shape().rank();
        for &d in dims {
            assert!(
                d < rank,
                "min_dims dim {} out of bounds for rank {}",
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

        // Compute min along reduced dims
        let in_shape = self.shape().dims.clone();
        let out_rank = out.shape().rank();
        let mut in_coords = vec![0usize; rank];
        unsafe {
            let dst = out.as_mut_ptr();
            // Initialize output with +inf
            for i in 0..out.size() {
                *dst.add(i) = f32::INFINITY;
            }
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
                let val = *self.as_ptr().add(in_offset);

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
                let cur = *dst.add(off);
                if val < cur {
                    *dst.add(off) = val;
                }
            }
        }

        if self.requires_grad() {
            let mut result = out.clone();
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::ReduceMinDims {
                dims: reduced,
                keepdim,
                input_shape: self.shape().dims.clone(),
                saved_output: Box::new(out.clone()),
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
    fn test_min_forward_basic() {
        let mut x = Tensor::zeros(vec![2, 3]);
        unsafe {
            for i in 0..6 {
                *x.as_mut_ptr().add(i) = (i as f32) - 3.0;
            }
        }
        let m = x.min();
        assert_eq!(m.shape().dims, vec![1]);
        unsafe {
            assert_eq!(*m.as_ptr(), -3.0);
        }
    }

    #[test]
    fn test_min_dims_forward() {
        let mut x = Tensor::zeros(vec![2, 3]);
        unsafe {
            for i in 0..6 {
                *x.as_mut_ptr().add(i) = (i as f32) - 3.0;
            }
        }
        let m = x.min_dims(&[1], true);
        assert_eq!(m.shape().dims, vec![2, 1]);
        assert_eq!(m.get(&[0, 0]), -3.0);
        assert_eq!(m.get(&[1, 0]), 0.0);
    }

    #[test]
    fn test_min_non_contiguous_transpose() {
        // Test min on transposed tensor (non-contiguous view)
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        // Original: [[1, 2, 3], [4, 5, 6]]

        let x_t = x.transpose(0, 1);
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert!(!x_t.is_contiguous()); // Should be a view

        let min_orig = x.min();
        let min_view = x_t.min();

        // Both should give the same result: min(1,2,3,4,5,6) = 1
        assert_eq!(min_orig.get(&[0]), 1.0);
        assert_eq!(min_view.get(&[0]), 1.0);
    }

    #[test]
    fn test_min_dims_non_contiguous() {
        // Test min_dims on non-contiguous tensor
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let x_t = x.transpose(0, 1); // [3, 2]
        assert!(!x_t.is_contiguous());

        // Min along dim 0 of transposed tensor
        let min_dim0 = x_t.min_dims(&[0], false);
        assert_eq!(min_dim0.shape().dims, vec![2]);
        // Should be [min(1,2,3), min(4,5,6)] = [1, 4]
        assert_eq!(min_dim0.get(&[0]), 1.0);
        assert_eq!(min_dim0.get(&[1]), 4.0);

        // Min along dim 1 of transposed tensor
        let min_dim1 = x_t.min_dims(&[1], false);
        assert_eq!(min_dim1.shape().dims, vec![3]);
        // Should be [min(1,4), min(2,5), min(3,6)] = [1, 2, 3]
        assert_eq!(min_dim1.get(&[0]), 1.0);
        assert_eq!(min_dim1.get(&[1]), 2.0);
        assert_eq!(min_dim1.get(&[2]), 3.0);
    }

    #[test]
    fn test_min_permuted_tensor() {
        // Test with permuted tensor
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();

        // Permute dimensions [2, 3, 4] -> [4, 2, 3]
        let x_perm = x.permute(vec![2, 1, 0]);
        assert!(!x_perm.is_contiguous());

        let min_orig = x.min();
        let min_perm = x_perm.min();

        // Should give same result
        assert_eq!(min_orig.get(&[0]), min_perm.get(&[0]));

        // Expected min: min(0,1,2,...,23) = 0
        assert_eq!(min_orig.get(&[0]), 0.0);
    }

    #[test]
    fn test_min_with_negative_values() {
        // Test min with negative values on non-contiguous tensor
        let x = Tensor::from_slice(&[-5.0, -2.0, -8.0, -1.0, -3.0, -6.0], vec![2, 3]).unwrap();
        let x_t = x.transpose(0, 1);
        assert!(!x_t.is_contiguous());

        let min_orig = x.min();
        let min_view = x_t.min();

        // Both should give the same result: min(-5,-2,-8,-1,-3,-6) = -8
        assert_eq!(min_orig.get(&[0]), -8.0);
        assert_eq!(min_view.get(&[0]), -8.0);
    }
}
