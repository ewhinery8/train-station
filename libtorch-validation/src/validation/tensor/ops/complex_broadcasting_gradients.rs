//! Validation tests for complex broadcasting gradient accumulation
//!
//! This module validates that complex broadcasting scenarios correctly compute
//! and accumulate gradients compared to LibTorch.

use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Test gradient accumulation with complex broadcasting patterns
    pub fn test_complex_broadcasting_gradients(
        &self,
        base_shape: &[usize],
        broadcast_shape: &[usize],
    ) -> ComparisonResult {
        // Our implementation with complex broadcasting
        let our_base = Tensor::randn(base_shape.to_vec(), Some(50)).with_requires_grad();
        let our_broadcast = Tensor::randn(broadcast_shape.to_vec(), Some(51)).with_requires_grad();

        // Chain of operations that involve broadcasting
        let our_added = our_base.add_tensor(&our_broadcast); // Broadcasting happens here
        let our_multiplied = our_added.mul_scalar(2.5);
        let our_final = our_multiplied.sub_scalar(1.0);
        let mut our_loss = our_final.sum();
        our_loss.backward(None);

        let our_grad_base = match our_base.grad_by_value() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("Our base tensor has no gradient".to_string())
            }
        };
        let our_grad_broadcast = match our_broadcast.grad_by_value() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure(
                    "Our broadcast tensor has no gradient".to_string(),
                )
            }
        };

        // LibTorch reference
        let data_base = our_base.data().to_vec();
        let data_broadcast = our_broadcast.data().to_vec();

        let torch_base = match LibTorchTensor::from_data(&data_base, base_shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad base: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch base tensor creation failed: {}",
                    e
                ))
            }
        };

        let torch_broadcast = match LibTorchTensor::from_data(&data_broadcast, broadcast_shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad broadcast: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch broadcast tensor creation failed: {}",
                    e
                ))
            }
        };

        let torch_added = match torch_base.add_tensor(&torch_broadcast) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_tensor failed: {}", e))
            }
        };

        let torch_multiplied = match torch_added.mul_scalar(2.5) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_scalar failed: {}", e))
            }
        };

        let torch_final = match torch_multiplied.sub_scalar(1.0) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch sub_scalar failed: {}", e))
            }
        };

        let torch_loss = match torch_final.sum() {
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

        let torch_grad_base = match torch_base.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure(
                    "LibTorch base tensor has no gradient".to_string(),
                )
            }
        };
        let torch_grad_broadcast = match torch_broadcast.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure(
                    "LibTorch broadcast tensor has no gradient".to_string(),
                )
            }
        };

        // Compare both gradients
        let base_comparison = self.compare_tensors(&our_grad_base, &torch_grad_base);
        if !base_comparison.passed {
            return ComparisonResult::failure(format!(
                "Base gradient mismatch: {}",
                base_comparison.details
            ));
        }

        self.compare_tensors(&our_grad_broadcast, &torch_grad_broadcast)
    }

    /// Test gradient accumulation with reduction broadcasting patterns
    pub fn test_reduction_broadcasting_gradients(
        &self,
        input_shape: &[usize],
        dim: usize,
    ) -> ComparisonResult {
        // Our implementation
        let our_input = Tensor::randn(input_shape.to_vec(), Some(52)).with_requires_grad();

        // Operations that change shape through reduction and then broadcasting
        let our_summed = our_input.sum_dims(&[dim], true); // Keep dims for broadcasting
        let our_broadcasted_back = our_input.add_tensor(&our_summed); // Broadcasting back to original shape
        let our_final = our_broadcasted_back.mul_scalar(0.5);
        let mut our_loss = our_final.mean();
        our_loss.backward(None);

        let our_grad = match our_input.grad_by_value() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("Our input tensor has no gradient".to_string())
            }
        };

        // LibTorch reference
        let data = our_input.data().to_vec();
        let torch_input = match LibTorchTensor::from_data(&data, input_shape) {
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

        let torch_summed = match torch_input.sum_dims(&[dim], true) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch sum_dims failed: {}", e)),
        };

        let torch_broadcasted_back = match torch_input.add_tensor(&torch_summed) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_tensor failed: {}", e))
            }
        };

        let torch_final = match torch_broadcasted_back.mul_scalar(0.5) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_scalar failed: {}", e))
            }
        };

        let torch_loss = match torch_final.mean() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch mean failed: {}", e)),
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

        let torch_grad = match torch_input.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor has no gradient".to_string())
            }
        };

        self.compare_tensors(&our_grad, &torch_grad)
    }

    /// Test gradient accumulation with multi-dimensional broadcasting
    pub fn test_multidimensional_broadcasting_gradients(&self) -> ComparisonResult {
        // Our implementation with multiple broadcasting patterns
        let our_a = Tensor::randn(vec![3, 1, 4], Some(53)).with_requires_grad(); // [3, 1, 4]
        let our_b = Tensor::randn(vec![1, 5, 1], Some(54)).with_requires_grad(); // [1, 5, 1]
        let our_c = Tensor::randn(vec![1, 1, 4], Some(55)).with_requires_grad(); // [1, 1, 4]

        // Complex operations with multiple broadcasting patterns
        let our_ab = our_a.mul_tensor(&our_b); // Should broadcast to [3, 5, 4]
        let our_abc = our_ab.add_tensor(&our_c); // Should maintain [3, 5, 4]
        let our_result = our_abc.div_scalar(3.0);
        let mut our_loss = our_result.sum();
        our_loss.backward(None);

        let our_grad_a = match our_a.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor a has no gradient".to_string()),
        };
        let our_grad_b = match our_b.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor b has no gradient".to_string()),
        };
        let our_grad_c = match our_c.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor c has no gradient".to_string()),
        };

        // LibTorch reference
        let data_a = our_a.data().to_vec();
        let data_b = our_b.data().to_vec();
        let data_c = our_c.data().to_vec();

        let torch_a = match LibTorchTensor::from_data(&data_a, &[3, 1, 4]) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad a: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor a creation failed: {}",
                    e
                ))
            }
        };

        let torch_b = match LibTorchTensor::from_data(&data_b, &[1, 5, 1]) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad b: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor b creation failed: {}",
                    e
                ))
            }
        };

        let torch_c = match LibTorchTensor::from_data(&data_c, &[1, 1, 4]) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad c: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor c creation failed: {}",
                    e
                ))
            }
        };

        let torch_ab = match torch_a.mul_tensor(&torch_b) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_tensor failed: {}", e))
            }
        };

        let torch_abc = match torch_ab.add_tensor(&torch_c) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_tensor failed: {}", e))
            }
        };

        let torch_result = match torch_abc.div_scalar(3.0) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch div_scalar failed: {}", e))
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

        let torch_grad_a = match torch_a.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor a has no gradient".to_string())
            }
        };
        let torch_grad_b = match torch_b.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor b has no gradient".to_string())
            }
        };
        let torch_grad_c = match torch_c.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor c has no gradient".to_string())
            }
        };

        // Compare all gradients
        let a_comparison = self.compare_tensors(&our_grad_a, &torch_grad_a);
        if !a_comparison.passed {
            return ComparisonResult::failure(format!(
                "Tensor A gradient mismatch: {}",
                a_comparison.details
            ));
        }

        let b_comparison = self.compare_tensors(&our_grad_b, &torch_grad_b);
        if !b_comparison.passed {
            return ComparisonResult::failure(format!(
                "Tensor B gradient mismatch: {}",
                b_comparison.details
            ));
        }

        self.compare_tensors(&our_grad_c, &torch_grad_c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_broadcasting_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_cases = vec![
            (vec![3, 1], vec![1, 4]),       // Simple broadcasting
            (vec![2, 3, 1], vec![1, 1, 4]), // 3D broadcasting
            (vec![1, 5], vec![3, 1]),       // Different order
        ];

        for (base_shape, broadcast_shape) in test_cases {
            let result =
                validator.test_complex_broadcasting_gradients(&base_shape, &broadcast_shape);
            assert!(
                result.passed,
                "Complex broadcasting gradient validation failed for shapes {:?} and {:?}: {}",
                base_shape, broadcast_shape, result.details
            );
        }
    }

    #[test]
    fn test_reduction_broadcasting_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_cases = vec![
            (vec![3, 4], 0),    // Reduce first dimension
            (vec![3, 4], 1),    // Reduce second dimension
            (vec![2, 3, 4], 1), // Reduce middle dimension
        ];

        for (shape, dim) in test_cases {
            let result = validator.test_reduction_broadcasting_gradients(&shape, dim);
            assert!(
                result.passed,
                "Reduction broadcasting gradient validation failed for shape {:?} dim {}: {}",
                shape, dim, result.details
            );
        }
    }

    #[test]
    fn test_multidimensional_broadcasting_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let result = validator.test_multidimensional_broadcasting_gradients();
        assert!(
            result.passed,
            "Multidimensional broadcasting gradient validation failed: {}",
            result.details
        );
    }
}
