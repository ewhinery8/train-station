//! Validation tests for transpose operations
//!
//! This module validates that transpose operations correctly compute
//! and accumulate gradients compared to LibTorch.

use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Test transpose gradient accumulation
    pub fn test_transpose_gradients(
        &self,
        shape: &[usize],
        dim0: usize,
        dim1: usize,
    ) -> ComparisonResult {
        assert!(
            shape.len() >= 2,
            "Shape must have at least 2 dimensions for transpose"
        );
        assert!(dim0 < shape.len(), "dim0 out of bounds");
        assert!(dim1 < shape.len(), "dim1 out of bounds");

        // Our implementation
        let our_source = Tensor::randn(shape.to_vec(), Some(100)).with_requires_grad();
        let our_transposed = our_source.transpose(dim0, dim1);

        // Chain operations on transposed tensor
        let our_result = our_transposed.mul_scalar(2.0).add_scalar(1.0);
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

        // Create permutation array for transpose
        let mut perm = (0..shape.len()).collect::<Vec<_>>();
        perm.swap(dim0, dim1);

        let torch_transposed = match torch_source.permute(&perm) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch permute (transpose) failed: {}",
                    e
                ))
            }
        };

        let torch_result = match torch_transposed
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_gradient_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_cases = vec![
            (vec![3, 4], 0, 1),    // Basic 2D transpose
            (vec![2, 3, 4], 0, 2), // 3D transpose first and last
            (vec![2, 3, 4], 1, 2), // 3D transpose last two
        ];

        for (shape, dim0, dim1) in test_cases {
            let result = validator.test_transpose_gradients(&shape, dim0, dim1);
            assert!(
                result.passed,
                "Transpose gradient validation failed for shape {:?}, dims ({}, {}): {}",
                shape, dim0, dim1, result.details
            );
        }
    }
}
