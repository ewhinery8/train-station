use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Computes the arithmetic mean of all elements in the tensor
    ///
    /// This method calculates the average value across all tensor elements by summing
    /// all values and dividing by the total number of elements. The result is a scalar
    /// tensor containing the mean value. This operation supports gradient tracking
    /// through the GradTrack system.
    ///
    /// # Returns
    ///
    /// A tensor with shape `[1]` containing the arithmetic mean of all elements.
    /// For empty tensors, returns `0.0` as a safe default.
    ///
    /// # Performance Characteristics
    ///
    /// - **Linear Time**: O(n) complexity for computing the sum
    /// - **Memory Efficient**: Single pass through tensor data with SIMD-optimized accumulation
    /// - **Numerical Stability**: Uses direct accumulation for typical tensor sizes
    /// - **Edge Case Handling**: Returns 0.0 for empty tensors
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let mean_val = tensor.mean();
    /// assert_eq!(mean_val.get(&[0]), 2.5); // (1+2+3+4)/4 = 2.5
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Empty tensor case
    /// let empty_tensor = Tensor::new(vec![0]);
    /// let mean_val = empty_tensor.mean();
    /// assert_eq!(mean_val.get(&[0]), 0.0);
    /// ```
    ///
    /// # GradTrack Support
    ///
    /// When `requires_grad` is true, this operation is tracked for automatic
    /// differentiation. The gradient computation distributes the gradient equally
    /// across all input elements.
    pub fn mean(&self) -> Tensor {
        let mut out = Tensor::new(vec![1]);
        if self.size() == 0 {
            // Convention: mean over empty returns 0.0 (aligns with safe behavior for now)
            out.fill(0.0);
        } else {
            let mut acc0 = 0.0f32;

            if self.is_contiguous() {
                // Fast path for contiguous tensors
                unsafe {
                    let src = self.as_ptr();
                    let size = self.size();
                    let mut i = 0usize;
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
                *out.as_mut_ptr() = acc0 / (self.size() as f32);
            }
        }

        if self.requires_grad() {
            out.set_requires_grad_internal(true);
            let grad_fn = GradFn::ReduceMean {
                input_shape: self.shape().dims.clone(),
                numel: self.size(),
            };
            out.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(out.id(), vec![self.id()], grad_fn);
        }
        out
    }

    /// Computes the arithmetic mean over specified dimensions
    ///
    /// This method calculates the mean value along the specified dimensions by first
    /// computing the sum over those dimensions and then dividing by the product of
    /// the reduced dimension sizes. The `keepdim` parameter determines whether
    /// reduced dimensions are kept with size 1 or removed entirely.
    ///
    /// # Arguments
    ///
    /// * `dims` - Dimensions to reduce over (must be valid for the tensor's rank)
    /// * `keepdim` - If true, reduced dimensions are kept with size 1; if false, they are removed
    ///
    /// # Returns
    ///
    /// A tensor with the specified dimensions reduced by computing the mean.
    /// The output shape depends on `keepdim`:
    /// - If `keepdim` is `true`, reduced dimensions have size 1
    /// - If `keepdim` is `false`, reduced dimensions are removed
    ///
    /// # Performance Characteristics
    ///
    /// - **Efficient Implementation**: Uses `sum_dims` followed by scalar multiplication
    /// - **Memory Optimized**: Leverages existing sum reduction for optimal performance
    /// - **Shape Computation**: Fast output shape calculation with dimension preservation
    /// - **Numerical Stability**: Maintains precision through direct computation
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    ///
    /// // Mean over columns (dim 1), keeping dimensions
    /// let mean_cols = tensor.mean_dims(&[1], true);
    /// assert_eq!(mean_cols.shape().dims, vec![2, 1]);
    /// assert_eq!(mean_cols.get(&[0, 0]), 2.0); // (1+2+3)/3 = 2.0
    /// assert_eq!(mean_cols.get(&[1, 0]), 5.0); // (4+5+6)/3 = 5.0
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    ///
    /// // Mean over rows (dim 0), removing dimensions
    /// let mean_rows = tensor.mean_dims(&[0], false);
    /// assert_eq!(mean_rows.shape().dims, vec![3]);
    /// assert_eq!(mean_rows.get(&[0]), 2.5); // (1+4)/2 = 2.5
    /// assert_eq!(mean_rows.get(&[1]), 3.5); // (2+5)/2 = 3.5
    /// assert_eq!(mean_rows.get(&[2]), 4.5); // (3+6)/2 = 4.5
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    ///
    /// // Mean over multiple dimensions
    /// let mean_all = tensor.mean_dims(&[0, 1], false);
    /// assert_eq!(mean_all.shape().dims, vec![1]);
    /// assert_eq!(mean_all.get(&[0]), 2.5); // (1+2+3+4)/4 = 2.5
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
    /// shape and handles broadcasting correctly through the ReduceMeanDims gradient function.
    pub fn mean_dims(&self, dims: &[usize], keepdim: bool) -> Tensor {
        assert!(
            !dims.is_empty(),
            "mean_dims requires at least one dimension"
        );
        let rank = self.shape().rank();
        for &d in dims {
            assert!(
                d < rank,
                "mean_dims dim {} out of bounds for rank {}",
                d,
                rank
            );
        }

        // Compute sum over dims first, then divide by product of reduced sizes
        let sum = self.sum_dims(dims, keepdim);
        let factor: usize = dims.iter().map(|&d| self.shape().dims[d]).product();
        let scale = if factor > 0 {
            1.0f32 / (factor as f32)
        } else {
            0.0
        };
        let out = sum.mul_scalar(scale);

        if self.requires_grad() {
            // Override autograd of mul to a single ReduceMeanDims node for correctness and clarity
            // Re-register operation for out to use ReduceMeanDims
            let mut reg = out.clone();
            reg.set_requires_grad_internal(true);
            let mut reduced: Vec<usize> = dims.to_vec();
            reduced.sort_unstable();
            reduced.dedup();
            let grad_fn = GradFn::ReduceMeanDims {
                dims: reduced,
                input_shape: self.shape().dims.clone(),
                keepdim,
            };
            reg.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(reg.id(), vec![self.id()], grad_fn);
            return reg;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_forward_basic() {
        let mut x = Tensor::zeros(vec![2, 3]);
        unsafe {
            for i in 0..6 {
                *x.as_mut_ptr().add(i) = i as f32;
            }
        }
        let m = x.mean();
        assert_eq!(m.shape().dims, vec![1]);
        unsafe {
            assert!((*m.as_ptr() - (0.0 + 1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mean_autograd_all_equal() {
        let x = Tensor::from_slice(&[1.0, 3.0, 5.0, 7.0], vec![4])
            .unwrap()
            .with_requires_grad();
        let mut m = x.mean();
        m.backward(None);
        let gx = x.grad_by_value().expect("grad missing");
        for i in 0..4 {
            unsafe {
                assert_eq!(*gx.as_ptr().add(i), 0.25);
            }
        }
    }

    #[test]
    fn test_mean_non_contiguous_transpose() {
        // Test mean on transposed tensor (non-contiguous view)
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        // Original: [[1, 2, 3], [4, 5, 6]]

        let x_t = x.transpose(0, 1);
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert!(!x_t.is_contiguous()); // Should be a view

        let mean_orig = x.mean();
        let mean_view = x_t.mean();

        // Both should give the same result: (1+2+3+4+5+6)/6 = 3.5
        assert!((mean_orig.get(&[0]) - 3.5).abs() < 1e-6);
        assert!((mean_view.get(&[0]) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_mean_dims_non_contiguous() {
        // Test mean_dims on non-contiguous tensor
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let x_t = x.transpose(0, 1); // [3, 2]
        assert!(!x_t.is_contiguous());

        // Mean along dim 0 of transposed tensor
        let mean_dim0 = x_t.mean_dims(&[0], false);
        assert_eq!(mean_dim0.shape().dims, vec![2]);
        // Should be [(1+2+3)/3, (4+5+6)/3] = [2.0, 5.0]
        assert!((mean_dim0.get(&[0]) - 2.0).abs() < 1e-6);
        assert!((mean_dim0.get(&[1]) - 5.0).abs() < 1e-6);

        // Mean along dim 1 of transposed tensor
        let mean_dim1 = x_t.mean_dims(&[1], false);
        assert_eq!(mean_dim1.shape().dims, vec![3]);
        // Should be [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
        assert!((mean_dim1.get(&[0]) - 2.5).abs() < 1e-6);
        assert!((mean_dim1.get(&[1]) - 3.5).abs() < 1e-6);
        assert!((mean_dim1.get(&[2]) - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_mean_permuted_tensor() {
        // Test with permuted tensor
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();

        // Permute dimensions [2, 3, 4] -> [4, 2, 3]
        let x_perm = x.permute(vec![2, 1, 0]);
        assert!(!x_perm.is_contiguous());

        let mean_orig = x.mean();
        let mean_perm = x_perm.mean();

        // Should give same result
        assert!((mean_orig.get(&[0]) - mean_perm.get(&[0])).abs() < 1e-6);

        // Expected mean: (0+1+2+...+23)/24 = 23*24/2/24 = 11.5
        assert!((mean_orig.get(&[0]) - 11.5).abs() < 1e-6);
    }
}
