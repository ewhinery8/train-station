//! Core validation utilities and comparison framework
//!
//! Provides fundamental comparison functionality for validating tensor operations
//! against LibTorch reference implementation.

use crate::ffi::LibTorchTensor;
use train_station::Tensor;

/// Result of a tensor operation comparison
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub passed: bool,
    pub max_diff: f32,
    pub mean_diff: f32,
    pub relative_error: f32,
    pub details: String,
}

impl ComparisonResult {
    pub fn success() -> Self {
        ComparisonResult {
            passed: true,
            max_diff: 0.0,
            mean_diff: 0.0,
            relative_error: 0.0,
            details: "All checks passed".to_string(),
        }
    }

    pub fn failure(details: String) -> Self {
        ComparisonResult {
            passed: false,
            max_diff: f32::INFINITY,
            mean_diff: f32::INFINITY,
            relative_error: f32::INFINITY,
            details,
        }
    }
}

/// Core tensor validation and comparison utilities
pub struct TensorValidator {
    /// Relative tolerance for floating point comparisons
    pub rtol: f64,
    /// Absolute tolerance for floating point comparisons
    pub atol: f64,
}

impl Default for TensorValidator {
    fn default() -> Self {
        TensorValidator {
            rtol: 1e-5,
            atol: 1e-8,
        }
    }
}

impl TensorValidator {
    pub fn new(rtol: f64, atol: f64) -> Self {
        TensorValidator { rtol, atol }
    }

    /// Compare our tensor implementation with libtorch for basic properties
    pub fn compare_tensors(
        &self,
        our_tensor: &Tensor,
        torch_tensor: &LibTorchTensor,
    ) -> ComparisonResult {
        // Check shapes match
        if our_tensor.shape().dims != torch_tensor.shape() {
            return ComparisonResult::failure(format!(
                "Shape mismatch: our={:?}, torch={:?}",
                our_tensor.shape().dims,
                torch_tensor.shape()
            ));
        }

        // Check data. If our tensor is non-contiguous (e.g., view/permute),
        // iterate in logical order using strides to build a linear buffer.
        let torch_data = torch_tensor.data();
        let our_linear: Vec<f32> = if our_tensor.is_contiguous() {
            unsafe { std::slice::from_raw_parts(our_tensor.as_ptr(), our_tensor.size()) }.to_vec()
        } else {
            let rank = our_tensor.shape().rank();
            let mut out = Vec::with_capacity(our_tensor.size());
            for linear_idx in 0..our_tensor.size() {
                let mut coords = vec![0usize; rank];
                let mut tmp = linear_idx;
                for i in (0..rank).rev() {
                    let dim_size = our_tensor.shape().dims[i];
                    coords[i] = tmp % dim_size;
                    tmp /= dim_size;
                }
                out.push(our_tensor.get(&coords));
            }
            out
        };

        if our_linear.len() != torch_data.len() {
            return ComparisonResult::failure(format!(
                "Data length mismatch: our={}, torch={}",
                our_linear.len(),
                torch_data.len()
            ));
        }

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        let mut sum_our = 0.0f64;
        let mut sum_torch = 0.0f64;

        for (i, (&our_val, &torch_val)) in our_linear.iter().zip(torch_data.iter()).enumerate() {
            // Fast path: exact match, including equal infinities
            if our_val == torch_val {
                continue;
            }

            // Treat both NaN as equal for validation purposes
            if our_val.is_nan() && torch_val.is_nan() {
                continue;
            }

            // If either is infinite here, the other is not equal → mismatch
            if !our_val.is_finite() || !torch_val.is_finite() {
                return ComparisonResult::failure(format!(
                    "Element {} differs (non-finite): our={}, torch={}",
                    i, our_val, torch_val
                ));
            }

            let diff = (our_val - torch_val).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff as f64;
            sum_our += our_val.abs() as f64;
            sum_torch += torch_val.abs() as f64;

            // Check individual element tolerance
            let rtol_check = diff <= (self.rtol as f32) * torch_val.abs();
            let atol_check = diff <= self.atol as f32;

            if !rtol_check && !atol_check {
                return ComparisonResult::failure(format!(
                    "Element {} differs too much: our={}, torch={}, diff={}, rtol_bound={}, atol_bound={}",
                    i, our_val, torch_val, diff,
                    (self.rtol as f32) * torch_val.abs(),
                    self.atol as f32
                ));
            }
        }

        let mean_diff = (sum_diff / our_linear.len() as f64) as f32;
        let mean_magnitude = ((sum_our + sum_torch) / (2.0 * our_linear.len() as f64)) as f32;
        let relative_error = if mean_magnitude > 0.0 {
            mean_diff / mean_magnitude
        } else {
            0.0
        };

        ComparisonResult {
            passed: true,
            max_diff,
            mean_diff,
            relative_error,
            details: format!(
                "Comparison passed: max_diff={:.2e}, mean_diff={:.2e}, rel_error={:.2e}",
                max_diff, mean_diff, relative_error
            ),
        }
    }
}

#[cfg(test)]
mod core_validation_tests {
    use super::*;

    #[test]
    fn test_comparison_result_creation() {
        let success = ComparisonResult::success();
        assert!(success.passed);
        assert_eq!(success.max_diff, 0.0);

        let failure = ComparisonResult::failure("Test failure".to_string());
        assert!(!failure.passed);
        assert_eq!(failure.details, "Test failure");
    }

    #[test]
    fn test_tensor_validator_creation() {
        let default_validator = TensorValidator::default();
        assert_eq!(default_validator.rtol, 1e-5);
        assert_eq!(default_validator.atol, 1e-8);

        let custom_validator = TensorValidator::new(1e-6, 1e-9);
        assert_eq!(custom_validator.rtol, 1e-6);
        assert_eq!(custom_validator.atol, 1e-9);
    }
}
