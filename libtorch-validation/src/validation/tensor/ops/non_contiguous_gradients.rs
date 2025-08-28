//! Validation tests for gradient accumulation with non-contiguous tensors
//!
//! This module validates that operations on non-contiguous tensors (views, permuted,
//! strided) correctly compute and accumulate gradients compared to LibTorch.

use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Test gradient accumulation with non-contiguous tensors from permute operations
    pub fn test_non_contiguous_permute_gradients(&self, shape: &[usize]) -> ComparisonResult {
        assert!(
            shape.len() >= 2,
            "Shape must have at least 2 dimensions for permute"
        );

        // Our implementation with permute creating non-contiguous tensor
        let our_source = Tensor::randn(shape.to_vec(), Some(42)).with_requires_grad();
        let our_permuted = our_source.permute(vec![1, 0]); // Create non-contiguous view

        // Verify non-contiguous
        assert!(
            !our_permuted.is_contiguous(),
            "Permuted tensor should be non-contiguous"
        );

        // Perform operations on non-contiguous tensor
        let our_result = our_permuted.mul_scalar(2.0).add_scalar(1.0);
        let mut our_loss = our_result.sum();
        our_loss.backward(None);

        let our_grad = match our_source.grad_by_value() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("Our source tensor has no gradient".to_string())
            }
        };

        // LibTorch reference
        let data = our_source.data().to_vec();
        let torch_source = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!("Failed to set requires_grad: {}", e))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch tensor creation failed: {}", e))
            }
        };

        let torch_permuted = match torch_source.permute(&[1, 0]) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch permute failed: {}", e)),
        };

        let torch_result = match torch_permuted
            .mul_scalar(2.0)
            .and_then(|t| t.add_scalar(1.0))
        {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch operations failed: {}", e))
            }
        };

        let torch_loss = match torch_result.sum() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch sum failed: {}", e)),
        };

        let grad_ones = match LibTorchTensor::ones(&torch_loss.shape()) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };

        if let Err(e) = torch_loss.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }

        let torch_grad = match torch_source.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure(
                    "LibTorch source tensor has no gradient".to_string(),
                )
            }
        };

        self.compare_tensors(&our_grad, &torch_grad)
    }

    /// Test gradient accumulation with non-contiguous tensors from reshape operations
    pub fn test_non_contiguous_reshape_gradients(&self, shape: &[usize]) -> ComparisonResult {
        assert!(
            shape.len() >= 2,
            "Shape must have at least 2 dimensions for reshape"
        );
        let total_size: usize = shape.iter().product();

        // Create compatible reshape dimensions
        let new_shape = if shape.len() == 2 {
            vec![total_size, 1] // Flatten to column vector
        } else {
            vec![shape[0], total_size / shape[0]] // Reshape to 2D
        };

        // Our implementation with reshape creating a view
        let our_source = Tensor::randn(shape.to_vec(), Some(43)).with_requires_grad();
        let our_reshaped = our_source.view(new_shape.iter().map(|&x| x as i32).collect());

        // Chain multiple operations on reshaped tensor
        let our_intermediate = our_reshaped.div_scalar(2.0);
        let our_result = our_intermediate.sub_scalar(0.5);
        let mut our_loss = our_result.sum(); // Use sum instead of mean for simpler gradients
        our_loss.backward(None);

        let our_grad = match our_source.grad_by_value() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("Our source tensor has no gradient".to_string())
            }
        };

        // LibTorch reference
        let data = our_source.data().to_vec();
        let torch_source = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!("Failed to set requires_grad: {}", e))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch tensor creation failed: {}", e))
            }
        };

        let torch_reshaped = match torch_source.view(&new_shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch view failed: {}", e)),
        };

        let torch_intermediate = match torch_reshaped.div_scalar(2.0) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch div_scalar failed: {}", e))
            }
        };

        let torch_result = match torch_intermediate.sub_scalar(0.5) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch sub_scalar failed: {}", e))
            }
        };

        let torch_loss = match torch_result.sum() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch sum failed: {}", e)),
        };

        let grad_ones = match LibTorchTensor::ones(&torch_loss.shape()) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };

        if let Err(e) = torch_loss.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }

        let torch_grad = match torch_source.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure(
                    "LibTorch source tensor has no gradient".to_string(),
                )
            }
        };

        self.compare_tensors(&our_grad, &torch_grad)
    }

    /// Test gradient accumulation with mixed contiguous and non-contiguous operations
    pub fn test_mixed_contiguous_non_contiguous_gradients(
        &self,
        shape: &[usize],
    ) -> ComparisonResult {
        assert!(shape.len() >= 2, "Shape must have at least 2 dimensions");

        // Our implementation
        let our_x = Tensor::randn(shape.to_vec(), Some(44)).with_requires_grad();
        let our_y = Tensor::randn(shape.to_vec(), Some(45)).with_requires_grad();

        // Mix of contiguous and non-contiguous operations using view/reshape
        let our_x_flat = our_x.view(vec![shape.iter().product::<usize>() as i32]); // Flatten to 1D
        let our_y_flat = our_y.view(vec![shape.iter().product::<usize>() as i32]); // Flatten to 1D
        let our_contiguous_result = our_y_flat.mul_scalar(3.0); // Contiguous

        // Operations mixing reshaped tensors
        let our_result = our_x_flat.add_tensor(&our_contiguous_result);
        let mut our_loss = our_result.sum();
        our_loss.backward(None);

        let our_grad_x = match our_x.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor x has no gradient".to_string()),
        };
        let our_grad_y = match our_y.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor y has no gradient".to_string()),
        };

        // LibTorch reference
        let data_x = our_x.data().to_vec();
        let data_y = our_y.data().to_vec();

        let torch_x = match LibTorchTensor::from_data(&data_x, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad x: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor x creation failed: {}",
                    e
                ))
            }
        };

        let torch_y = match LibTorchTensor::from_data(&data_y, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad y: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor y creation failed: {}",
                    e
                ))
            }
        };

        let torch_x_flat = match torch_x.view(&[shape.iter().product::<usize>()]) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch view x failed: {}", e)),
        };

        let torch_y_flat = match torch_y.view(&[shape.iter().product::<usize>()]) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch view y failed: {}", e)),
        };

        let torch_contiguous_result = match torch_y_flat.mul_scalar(3.0) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_scalar failed: {}", e))
            }
        };

        let torch_result = match torch_x_flat.add_tensor(&torch_contiguous_result) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_tensor failed: {}", e))
            }
        };

        let torch_loss = match torch_result.sum() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch sum failed: {}", e)),
        };

        let grad_ones = match LibTorchTensor::ones(&torch_loss.shape()) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };

        if let Err(e) = torch_loss.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }

        let torch_grad_x = match torch_x.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor x has no gradient".to_string())
            }
        };
        let torch_grad_y = match torch_y.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor y has no gradient".to_string())
            }
        };

        // Compare both gradients
        let x_comparison = self.compare_tensors(&our_grad_x, &torch_grad_x);
        if !x_comparison.passed {
            return ComparisonResult::failure(format!(
                "X gradient mismatch: {}",
                x_comparison.details
            ));
        }

        self.compare_tensors(&our_grad_y, &torch_grad_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_contiguous_permute_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_shapes = vec![vec![3, 4], vec![2, 5], vec![4, 3]];

        for shape in test_shapes {
            let result = validator.test_non_contiguous_permute_gradients(&shape);
            assert!(
                result.passed,
                "Non-contiguous permute gradient validation failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_non_contiguous_reshape_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_shapes = vec![vec![3, 4], vec![2, 6], vec![4, 3], vec![2, 5]];

        for shape in test_shapes {
            let result = validator.test_non_contiguous_reshape_gradients(&shape);
            assert!(
                result.passed,
                "Non-contiguous reshape gradient validation failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_mixed_contiguous_non_contiguous_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_shapes = vec![vec![3, 4], vec![2, 5], vec![4, 3]];

        for shape in test_shapes {
            let result = validator.test_mixed_contiguous_non_contiguous_gradients(&shape);
            assert!(
                result.passed,
                "Mixed contiguous/non-contiguous gradient validation failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }
}
