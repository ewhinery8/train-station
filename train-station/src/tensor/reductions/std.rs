//! Standard deviation reduction operations for tensors
//!
//! This module provides standard deviation reduction operations for tensors.
//! The standard deviation measures the dispersion of values around the mean,
//! calculated as the square root of the variance. This is commonly used in
//! statistics, data analysis, and machine learning for understanding data
//! variability and normalization.
//!
//! # Operations
//!
//! * `std()` - Computes standard deviation over all elements, returning a scalar tensor
//! * `std_dims()` - Computes standard deviation over specified dimensions with optional dimension preservation
//!
//! # Statistical Details
//!
//! The implementation uses population standard deviation (unbiased=false), which
//! divides by n rather than n-1. This matches PyTorch's default behavior for
//! consistency with the reference implementation.
//!
//! # Examples
//!
//! ```
//! use train_station::Tensor;
//!
//! // Compute standard deviation of all elements
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
//! let std_dev = tensor.std();
//! assert!((std_dev.get(&[0]) - 1.118_034).abs() < 1e-5);
//!
//! // Compute standard deviation along specific dimensions
//! let matrix = Tensor::from_slice(&[1.0, 3.0, 2.0, 2.0], vec![2, 2]).unwrap();
//! let row_stds = matrix.std_dims(&[1], true);
//! assert_eq!(row_stds.shape().dims, vec![2, 1]);
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
//! The gradient computation follows the mathematical derivative of the standard deviation operation.

use crate::gradtrack::{GradEngine, GradFn};
use crate::tensor::core::Tensor;

impl Tensor {
    /// Computes the standard deviation over all elements
    ///
    /// The standard deviation is calculated as sqrt(variance) where variance is
    /// the mean of squared differences from the mean. This operation reduces the
    /// tensor to a scalar value \[1\].
    ///
    /// The implementation uses population standard deviation (divides by n rather
    /// than n-1) to match PyTorch's default behavior.
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the standard deviation value
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Basic standard deviation calculation
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    /// let std_dev = tensor.std();
    /// assert!((std_dev.get(&[0]) - 1.118_034).abs() < 1e-5);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Standard deviation of a larger dataset
    /// let data = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
    /// let tensor = Tensor::from_slice(&data, vec![2, 2, 2]).unwrap();
    /// let std_dev = tensor.std();
    /// // mean=4.5, var=5.25, std=sqrt(5.25)≈2.291
    /// let expected = 5.25_f32.sqrt();
    /// assert!((std_dev.get(&[0]) - expected).abs() < 1e-5);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Standard deviation of constant values (should be 0)
    /// let tensor = Tensor::from_slice(&[5.0, 5.0, 5.0, 5.0], vec![4]).unwrap();
    /// let std_dev = tensor.std();
    /// assert!((std_dev.get(&[0]) - 0.0).abs() < 1e-6);
    /// ```
    ///
    /// # Performance
    ///
    /// Uses optimized contiguous tensor path with 4x loop unrolling for better
    /// performance. Non-contiguous tensors use stride-aware iteration.
    /// The algorithm performs two passes: first to compute the mean, then to
    /// compute the variance.
    pub fn std(&self) -> Tensor {
        let mut out = Tensor::new(vec![1]);
        if self.size() == 0 {
            out.fill(0.0);
        } else {
            // First pass: mean
            let mut mean_val = 0.0f32;
            let n = self.size() as f32;

            if self.is_contiguous() {
                // Fast path for contiguous tensors
                unsafe {
                    let src = self.as_ptr();
                    let mut i = 0usize;
                    while i + 4 <= self.size() {
                        mean_val +=
                            *src.add(i) + *src.add(i + 1) + *src.add(i + 2) + *src.add(i + 3);
                        i += 4;
                    }
                    while i < self.size() {
                        mean_val += *src.add(i);
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
                    mean_val += value;
                }
            }
            mean_val /= n;

            // Second pass: variance
            let mut var_val = 0.0f32;

            if self.is_contiguous() {
                // Fast path for contiguous tensors
                unsafe {
                    let src = self.as_ptr();
                    let mut i = 0usize;
                    while i + 4 <= self.size() {
                        let x0 = *src.add(i) - mean_val;
                        let x1 = *src.add(i + 1) - mean_val;
                        let x2 = *src.add(i + 2) - mean_val;
                        let x3 = *src.add(i + 3) - mean_val;
                        var_val += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
                        i += 4;
                    }
                    while i < self.size() {
                        let d = *src.add(i) - mean_val;
                        var_val += d * d;
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
                    let d = value - mean_val;
                    var_val += d * d;
                }
            }
            var_val /= n;

            let std_val = var_val.sqrt();
            unsafe {
                *out.as_mut_ptr() = std_val;
            }
        }

        if self.requires_grad() {
            // Save tensors for backward
            let mut result = out.clone();
            result.set_requires_grad_internal(true);
            let mean_tensor = {
                let mut t = Tensor::new(vec![1]);
                if self.size() == 0 {
                    t.fill(0.0);
                } else {
                    let mut acc = 0.0f32;

                    if self.is_contiguous() {
                        unsafe {
                            for i in 0..self.size() {
                                acc += *self.as_ptr().add(i);
                            }
                        }
                    } else {
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
                            acc += value;
                        }
                    }

                    unsafe {
                        *t.as_mut_ptr() = acc / (self.size() as f32);
                    }
                }
                t
            };
            let std_saved = out.clone();
            let grad_fn = GradFn::ReduceStd {
                saved_mean: Box::new(mean_tensor),
                saved_std: Box::new(std_saved),
                saved_input: Box::new(self.clone()),
                input_shape: self.shape().dims.clone(),
            };
            result.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(result.id(), vec![self.id()], grad_fn);
            return result;
        }

        out
    }

    /// Computes the standard deviation over specified dimensions
    ///
    /// Reduces the tensor along the specified dimensions by computing the standard
    /// deviation of each slice. The result maintains the original tensor structure
    /// with reduced dimensions optionally preserved as size-1 dimensions.
    ///
    /// Uses population standard deviation (divides by n rather than n-1) to match
    /// PyTorch's default behavior.
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector of dimension indices to reduce over (must be valid for tensor rank)
    /// * `keepdim` - Whether to keep reduced dimensions as size-1 dimensions
    ///
    /// # Returns
    ///
    /// A tensor with standard deviation computed over the specified dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Standard deviation along rows (dimension 1) with keepdim=true
    /// let matrix = Tensor::from_slice(&[1.0, 3.0, 2.0, 2.0], vec![2, 2]).unwrap();
    /// let row_stds = matrix.std_dims(&[1], true);
    /// assert_eq!(row_stds.shape().dims, vec![2, 1]);
    /// assert!((row_stds.get(&[0, 0]) - 1.0).abs() < 1e-6); // std([1, 3]) = 1.0
    /// assert!((row_stds.get(&[1, 0]) - 0.0).abs() < 1e-6); // std([2, 2]) = 0.0
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Standard deviation along columns (dimension 0) with keepdim=false
    /// let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let col_stds = matrix.std_dims(&[0], false);
    /// assert_eq!(col_stds.shape().dims, vec![2]);
    /// // std([1, 3]) = 1.0, std([2, 4]) = 1.0
    /// assert!((col_stds.get(&[0]) - 1.0).abs() < 1e-6);
    /// assert!((col_stds.get(&[1]) - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Standard deviation over multiple dimensions
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let std_all = tensor.std_dims(&[0, 1], false);
    /// assert_eq!(std_all.shape().dims, vec![1]);
    /// // std([1, 2, 3, 4]) = sqrt(1.25) ≈ 1.118
    /// assert!((std_all.get(&[0]) - 1.25_f32.sqrt()).abs() < 1e-5);
    /// ```
    ///
    /// # Panics
    ///
    /// * If `dims` is empty
    /// * If any dimension index is out of bounds for the tensor rank
    /// * If the reduced size is 0 (invalid for standard deviation calculation)
    ///
    /// # Performance
    ///
    /// Uses efficient coordinate-based iteration that works correctly with
    /// both contiguous and non-contiguous tensor layouts. The algorithm performs
    /// two passes: first to compute means, then to compute variances.
    pub fn std_dims(&self, dims: &[usize], keepdim: bool) -> Tensor {
        assert!(!dims.is_empty(), "std_dims requires at least one dimension");
        let rank = self.shape().rank();
        for &d in dims {
            assert!(
                d < rank,
                "std_dims dim {} out of bounds for rank {}",
                d,
                rank
            );
        }

        // Output shape
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
        let mut mean = Tensor::zeros(out_dims.clone());
        let mut var = Tensor::zeros(out_dims.clone());

        let in_shape = self.shape().dims.clone();
        let out_rank = mean.shape().rank();
        let mut in_coords = vec![0usize; rank];
        let n_reduced: usize = reduced.iter().map(|&d| in_shape[d]).product();
        assert!(n_reduced > 0, "reduced size must be > 0");
        unsafe {
            let mptr = mean.as_mut_ptr();
            let vptr = var.as_mut_ptr();

            // First pass: sum to compute means
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
                    mean.shape().offset(&out_coords)
                };
                *mptr.add(off) += value;
            }
            let msize = mean.size();
            for i in 0..msize {
                *mptr.add(i) /= n_reduced as f32;
            }

            // Second pass: accumulate squared diffs
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
                let x = *self.as_ptr().add(in_offset);

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
                    var.shape().offset(&out_coords)
                };
                let diff = x - *mptr.add(off);
                *vptr.add(off) += diff * diff;
            }
            let vsize = var.size();
            for i in 0..vsize {
                *vptr.add(i) /= n_reduced as f32;
            }
        }
        // std = sqrt(var)
        let mut out = Tensor::zeros(out_dims.clone());
        unsafe {
            let d = out.as_mut_ptr();
            let v = var.as_ptr();
            for i in 0..out.size() {
                *d.add(i) = (*v.add(i)).sqrt();
            }
        }

        if self.requires_grad() {
            let mut result = out.clone();
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::ReduceStdDims {
                dims: reduced,
                keepdim,
                input_shape: self.shape().dims.clone(),
                saved_mean: Box::new(mean),
                saved_std: Box::new(out.clone()),
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
    fn test_std_forward_basic() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let s = x.std();
        unsafe {
            let v = *s.as_ptr();
            assert!((v - 1.118_034).abs() < 1e-5);
        }
    }

    #[test]
    fn test_std_dims_forward() {
        let x = Tensor::from_slice(&[1.0, 3.0, 2.0, 2.0], vec![2, 2]).unwrap();
        let s = x.std_dims(&[1], true);
        assert_eq!(s.shape().dims, vec![2, 1]);
        assert!((s.get(&[0, 0]) - 1.0).abs() < 1e-6);
        assert!((s.get(&[1, 0]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_std_non_contiguous_transpose() {
        // Test std on transposed tensor (non-contiguous view)
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        // Original: [[1, 2, 3], [4, 5, 6]]

        let x_t = x.transpose(0, 1);
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert!(!x_t.is_contiguous()); // Should be a view

        let std_orig = x.std();
        let std_view = x_t.std();

        // Both should give the same result
        assert!((std_orig.get(&[0]) - std_view.get(&[0])).abs() < 1e-6);

        // Expected std of [1,2,3,4,5,6]: mean=3.5, var=mean([2.5^2,1.5^2,0.5^2,0.5^2,1.5^2,2.5^2])=2.9167
        let expected_std = (2.9166667_f32).sqrt(); // ≈ 1.708
        assert!((std_orig.get(&[0]) - expected_std).abs() < 1e-5);
    }

    #[test]
    fn test_std_dims_non_contiguous() {
        // Test std_dims on non-contiguous tensor
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let x_t = x.transpose(0, 1); // [3, 2]
        assert!(!x_t.is_contiguous());

        // Std along dim 0 of transposed tensor
        let std_dim0 = x_t.std_dims(&[0], false);
        assert_eq!(std_dim0.shape().dims, vec![2]);

        // For dim 0: [1,2,3] and [4,5,6]
        // [1,2,3]: mean=2, var=((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3, std=sqrt(2/3)≈0.816
        // [4,5,6]: mean=5, var=((4-5)^2 + (5-5)^2 + (6-5)^2)/3 = 2/3, std=sqrt(2/3)≈0.816
        let expected_std = (2.0 / 3.0_f32).sqrt();
        assert!((std_dim0.get(&[0]) - expected_std).abs() < 1e-5);
        assert!((std_dim0.get(&[1]) - expected_std).abs() < 1e-5);
    }

    #[test]
    fn test_std_permuted_tensor() {
        // Test with permuted tensor - simple case with known std
        let data = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        let x = Tensor::from_slice(&data, vec![2, 2, 2]).unwrap();

        // Permute dimensions [2, 2, 2] -> [2, 2, 2] (swap first and last)
        let x_perm = x.permute(vec![2, 1, 0]);
        assert!(!x_perm.is_contiguous());

        let std_orig = x.std();
        let std_perm = x_perm.std();

        // Should give same result
        assert!((std_orig.get(&[0]) - std_perm.get(&[0])).abs() < 1e-6);

        // Data is [1,3,5,7,2,4,6,8], mean=4.5
        // var = mean([3.5^2, 1.5^2, 0.5^2, 2.5^2, 2.5^2, 0.5^2, 1.5^2, 3.5^2]) = 5.25
        let expected_std = 5.25_f32.sqrt(); // ≈ 2.291
        assert!((std_orig.get(&[0]) - expected_std).abs() < 1e-5);
    }
}
