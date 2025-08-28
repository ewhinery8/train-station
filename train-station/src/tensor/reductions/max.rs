use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Computes the maximum value over all elements in the tensor
    ///
    /// Returns a scalar tensor containing the maximum value. For empty tensors,
    /// returns negative infinity. This operation supports gradient tracking
    /// through the GradTrack system.
    ///
    /// # Returns
    ///
    /// A tensor with shape `[1]` containing the maximum value
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 5.0, 3.0, 2.0], vec![2, 2]).unwrap();
    /// let max_val = tensor.max();
    /// assert_eq!(max_val.get(&[0]), 5.0);
    /// ```
    ///
    /// # GradTrack Support
    ///
    /// When `requires_grad` is true, this operation is tracked for automatic
    /// differentiation. The gradient computation uses the saved input and output
    /// for efficient backward pass.
    pub fn max(&self) -> Tensor {
        let mut out = Tensor::new(vec![1]);
        if self.size() == 0 {
            out.fill(f32::NEG_INFINITY);
        } else {
            let mut m = f32::NEG_INFINITY;

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
                        m = m.max(x0).max(x1).max(x2).max(x3);
                        i += 4;
                    }
                    while i < size {
                        m = m.max(*src.add(i));
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
                        m = m.max(value);
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
            let grad_fn = GradFn::ReduceMax {
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

    /// Computes the maximum value over specified dimensions
    ///
    /// Reduces the tensor along the specified dimensions by computing the maximum
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
    /// // Max over columns (dim 1), keeping dimensions
    /// let max_cols = tensor.max_dims(&[1], true);
    /// assert_eq!(max_cols.shape().dims, vec![2, 1]);
    /// assert_eq!(max_cols.get(&[0, 0]), 3.0);
    /// assert_eq!(max_cols.get(&[1, 0]), 6.0);
    ///
    /// // Max over rows (dim 0), removing dimensions
    /// let max_rows = tensor.max_dims(&[0], false);
    /// assert_eq!(max_rows.shape().dims, vec![3]);
    /// assert_eq!(max_rows.get(&[0]), 4.0);
    /// assert_eq!(max_rows.get(&[1]), 5.0);
    /// assert_eq!(max_rows.get(&[2]), 6.0);
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
    pub fn max_dims(&self, dims: &[usize], keepdim: bool) -> Tensor {
        assert!(!dims.is_empty(), "max_dims requires at least one dimension");
        let rank = self.shape().rank();
        for &d in dims {
            assert!(
                d < rank,
                "max_dims dim {} out of bounds for rank {}",
                d,
                rank
            );
        }

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

        let in_shape = self.shape().dims.clone();
        let out_rank = out.shape().rank();
        let mut in_coords = vec![0usize; rank];
        unsafe {
            let dst = out.as_mut_ptr();
            for i in 0..out.size() {
                *dst.add(i) = f32::NEG_INFINITY;
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
                if val > cur {
                    *dst.add(off) = val;
                }
            }
        }

        if self.requires_grad() {
            let mut result = out.clone();
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::ReduceMaxDims {
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
    fn test_max_forward_basic() {
        let mut x = Tensor::zeros(vec![2, 3]);
        unsafe {
            for i in 0..6 {
                *x.as_mut_ptr().add(i) = (i as f32) - 3.0;
            }
        }
        let m = x.max();
        assert_eq!(m.shape().dims, vec![1]);
        unsafe {
            assert_eq!(*m.as_ptr(), 2.0);
        }
    }

    #[test]
    fn test_max_dims_forward() {
        let mut x = Tensor::zeros(vec![2, 3]);
        unsafe {
            for i in 0..6 {
                *x.as_mut_ptr().add(i) = (i as f32) - 3.0;
            }
        }
        let m = x.max_dims(&[1], true);
        assert_eq!(m.shape().dims, vec![2, 1]);
        assert_eq!(m.get(&[0, 0]), -1.0);
        assert_eq!(m.get(&[1, 0]), 2.0);
    }

    #[test]
    fn test_max_non_contiguous_transpose() {
        // Test max on transposed tensor (non-contiguous view)
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        // Original: [[1, 2, 3], [4, 5, 6]]

        let x_t = x.transpose(0, 1);
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert!(!x_t.is_contiguous()); // Should be a view

        let max_orig = x.max();
        let max_view = x_t.max();

        // Both should give the same result: max(1,2,3,4,5,6) = 6
        assert_eq!(max_orig.get(&[0]), 6.0);
        assert_eq!(max_view.get(&[0]), 6.0);
    }

    #[test]
    fn test_max_dims_non_contiguous() {
        // Test max_dims on non-contiguous tensor
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let x_t = x.transpose(0, 1); // [3, 2]
        assert!(!x_t.is_contiguous());

        // Max along dim 0 of transposed tensor
        let max_dim0 = x_t.max_dims(&[0], false);
        assert_eq!(max_dim0.shape().dims, vec![2]);
        // Should be [max(1,2,3), max(4,5,6)] = [3, 6]
        assert_eq!(max_dim0.get(&[0]), 3.0);
        assert_eq!(max_dim0.get(&[1]), 6.0);

        // Max along dim 1 of transposed tensor
        let max_dim1 = x_t.max_dims(&[1], false);
        assert_eq!(max_dim1.shape().dims, vec![3]);
        // Should be [max(1,4), max(2,5), max(3,6)] = [4, 5, 6]
        assert_eq!(max_dim1.get(&[0]), 4.0);
        assert_eq!(max_dim1.get(&[1]), 5.0);
        assert_eq!(max_dim1.get(&[2]), 6.0);
    }

    #[test]
    fn test_max_permuted_tensor() {
        // Test with permuted tensor
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();

        // Permute dimensions [2, 3, 4] -> [4, 2, 3]
        let x_perm = x.permute(vec![2, 1, 0]);
        assert!(!x_perm.is_contiguous());

        let max_orig = x.max();
        let max_perm = x_perm.max();

        // Should give same result
        assert_eq!(max_orig.get(&[0]), max_perm.get(&[0]));

        // Expected max: max(0,1,2,...,23) = 23
        assert_eq!(max_orig.get(&[0]), 23.0);
    }

    #[test]
    fn test_max_with_negative_values() {
        // Test max with negative values on non-contiguous tensor
        let x = Tensor::from_slice(&[-5.0, -2.0, -8.0, -1.0, -3.0, -6.0], vec![2, 3]).unwrap();
        let x_t = x.transpose(0, 1);
        assert!(!x_t.is_contiguous());

        let max_orig = x.max();
        let max_view = x_t.max();

        // Both should give the same result: max(-5,-2,-8,-1,-3,-6) = -1
        assert_eq!(max_orig.get(&[0]), -1.0);
        assert_eq!(max_view.get(&[0]), -1.0);
    }
}
