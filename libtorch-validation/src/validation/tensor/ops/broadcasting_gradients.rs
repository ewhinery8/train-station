//! Broadcasting gradient validation tests against LibTorch
//!
//! This module provides comprehensive validation of broadcasting gradient computation
//! for tensor operations (add, mul, div, sub) against LibTorch reference implementation.

use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Test broadcasting addition gradients
    pub fn test_add_broadcasting_gradients(
        &self,
        shape_a: &[usize],
        shape_b: &[usize],
    ) -> ComparisonResult {
        // Our implementation
        let our_a = Tensor::ones(shape_a.to_vec()).with_requires_grad();
        let our_b = Tensor::from_slice(
            &(0..shape_b.iter().product::<usize>())
                .map(|i| (i as f32) * 0.1)
                .collect::<Vec<_>>(),
            shape_b.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let mut our_result = our_a.add_tensor(&our_b);
        our_result.backward(None);

        let our_grad_a = match our_a.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor A has no gradient".to_string()),
        };
        let our_grad_b = match our_b.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor B has no gradient".to_string()),
        };

        // LibTorch reference
        let torch_a = match LibTorchTensor::ones(shape_a) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch A requires_grad failed: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch A creation failed: {}", e))
            }
        };

        let torch_b_data: Vec<f32> = (0..shape_b.iter().product::<usize>())
            .map(|i| (i as f32) * 0.1)
            .collect();
        let torch_b = match LibTorchTensor::from_data(&torch_b_data, shape_b) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch B requires_grad failed: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch B creation failed: {}", e))
            }
        };

        let torch_result = match torch_a.add_tensor(&torch_b) {
            Ok(r) => r,
            Err(e) => return ComparisonResult::failure(format!("LibTorch add failed: {}", e)),
        };

        // Create gradient tensor with same shape as result
        let grad_shape = torch_result.shape();
        let ones_data = vec![1.0f32; grad_shape.iter().product()];
        let grad_tensor = match LibTorchTensor::from_data(&ones_data, &grad_shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };

        if let Err(e) = torch_result.backward(Some(&grad_tensor)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }

        let torch_grad_a = match torch_a.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor A has no gradient".to_string())
            }
        };

        let torch_grad_b = match torch_b.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor B has no gradient".to_string())
            }
        };

        // Compare shapes first
        if our_grad_a.shape().dims != torch_grad_a.shape() {
            return ComparisonResult::failure(format!(
                "Gradient A shape mismatch: our {:?} vs torch {:?}",
                our_grad_a.shape().dims,
                torch_grad_a.shape()
            ));
        }

        if our_grad_b.shape().dims != torch_grad_b.shape() {
            return ComparisonResult::failure(format!(
                "Gradient B shape mismatch: our {:?} vs torch {:?}",
                our_grad_b.shape().dims,
                torch_grad_b.shape()
            ));
        }

        // Compare gradient values
        let grad_a_comparison = self.compare_tensors(&our_grad_a, &torch_grad_a);
        if !grad_a_comparison.passed {
            return ComparisonResult::failure(format!(
                "Gradient A values mismatch: {}",
                grad_a_comparison.details
            ));
        }

        let grad_b_comparison = self.compare_tensors(&our_grad_b, &torch_grad_b);
        if !grad_b_comparison.passed {
            return ComparisonResult::failure(format!(
                "Gradient B values mismatch: {}",
                grad_b_comparison.details
            ));
        }

        ComparisonResult::success()
    }

    /// Test broadcasting multiplication gradients
    pub fn test_mul_broadcasting_gradients(
        &self,
        shape_a: &[usize],
        shape_b: &[usize],
    ) -> ComparisonResult {
        // Our implementation
        let our_a = Tensor::from_slice(
            &(1..=shape_a.iter().product::<usize>())
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
            shape_a.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let our_b = Tensor::from_slice(
            &(0..shape_b.iter().product::<usize>())
                .map(|i| (i as f32) * 0.2 + 1.0)
                .collect::<Vec<_>>(),
            shape_b.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let mut our_result = our_a.mul_tensor(&our_b);
        our_result.backward(None);

        let our_grad_a = match our_a.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor A has no gradient".to_string()),
        };
        let our_grad_b = match our_b.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor B has no gradient".to_string()),
        };

        // LibTorch reference
        let torch_a_data: Vec<f32> = (1..=shape_a.iter().product::<usize>())
            .map(|i| i as f32)
            .collect();
        let torch_a = match LibTorchTensor::from_data(&torch_a_data, shape_a) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch A requires_grad failed: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch A creation failed: {}", e))
            }
        };

        let torch_b_data: Vec<f32> = (0..shape_b.iter().product::<usize>())
            .map(|i| (i as f32) * 0.2 + 1.0)
            .collect();
        let torch_b = match LibTorchTensor::from_data(&torch_b_data, shape_b) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch B requires_grad failed: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch B creation failed: {}", e))
            }
        };

        let torch_result = match torch_a.mul_tensor(&torch_b) {
            Ok(r) => r,
            Err(e) => return ComparisonResult::failure(format!("LibTorch mul failed: {}", e)),
        };

        // Create gradient tensor
        let grad_shape = torch_result.shape();
        let ones_data = vec![1.0f32; grad_shape.iter().product()];
        let grad_tensor = match LibTorchTensor::from_data(&ones_data, &grad_shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };

        if let Err(e) = torch_result.backward(Some(&grad_tensor)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }

        let torch_grad_a = match torch_a.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor A has no gradient".to_string())
            }
        };

        let torch_grad_b = match torch_b.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor B has no gradient".to_string())
            }
        };

        // Compare gradients
        let grad_a_comparison = self.compare_tensors(&our_grad_a, &torch_grad_a);
        if !grad_a_comparison.passed {
            return ComparisonResult::failure(format!(
                "Gradient A values mismatch: {}",
                grad_a_comparison.details
            ));
        }

        let grad_b_comparison = self.compare_tensors(&our_grad_b, &torch_grad_b);
        if !grad_b_comparison.passed {
            return ComparisonResult::failure(format!(
                "Gradient B values mismatch: {}",
                grad_b_comparison.details
            ));
        }

        ComparisonResult::success()
    }

    /// Test linear layer pattern: matmul + bias broadcasting
    pub fn test_linear_layer_broadcasting_gradients(
        &self,
        input_shape: &[usize],  // e.g., [batch_size, input_features]
        weight_shape: &[usize], // e.g., [input_features, output_features]
        bias_shape: &[usize],   // e.g., [output_features]
    ) -> ComparisonResult {
        // Our implementation
        let our_input = Tensor::from_slice(
            &(0..input_shape.iter().product::<usize>())
                .map(|i| (i as f32) * 0.1)
                .collect::<Vec<_>>(),
            input_shape.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let our_weight = Tensor::from_slice(
            &(0..weight_shape.iter().product::<usize>())
                .map(|i| (i as f32) * 0.05)
                .collect::<Vec<_>>(),
            weight_shape.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let our_bias = Tensor::from_slice(
            &(0..bias_shape.iter().product::<usize>())
                .map(|i| (i as f32) * 0.02)
                .collect::<Vec<_>>(),
            bias_shape.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        // Forward pass: input @ weight + bias
        let matmul_result = our_input.matmul(&our_weight);
        let linear_result = matmul_result.add_tensor(&our_bias);

        // Loss: sum of all outputs
        let mut loss = linear_result.sum();
        loss.backward(None);

        let _our_input_grad = match our_input.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our input has no gradient".to_string()),
        };
        let _our_weight_grad = match our_weight.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our weight has no gradient".to_string()),
        };
        let our_bias_grad = match our_bias.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our bias has no gradient".to_string()),
        };

        // Check bias gradient shape (most important for broadcasting)
        if our_bias_grad.shape().dims != bias_shape {
            return ComparisonResult::failure(format!(
                "Bias gradient shape mismatch: expected {:?}, got {:?}",
                bias_shape,
                our_bias_grad.shape().dims
            ));
        }

        // Bias gradient should be the sum over the batch dimension
        // For batch_size = input_shape[0], each bias element should have gradient = batch_size
        let expected_bias_grad_value = input_shape[0] as f32;
        for i in 0..our_bias_grad.size() {
            let val = unsafe { *our_bias_grad.as_ptr().add(i) };
            if (val - expected_bias_grad_value).abs() > self.atol as f32 {
                return ComparisonResult::failure(format!(
                    "Bias gradient[{}] = {}, expected {}",
                    i, val, expected_bias_grad_value
                ));
            }
        }

        ComparisonResult::success()
    }

    /// Test broadcasting subtraction gradients
    pub fn test_sub_broadcasting_gradients(
        &self,
        shape_a: &[usize],
        shape_b: &[usize],
    ) -> ComparisonResult {
        // Our implementation
        let our_a = Tensor::ones(shape_a.to_vec()).with_requires_grad();
        let our_b = Tensor::from_slice(
            &(0..shape_b.iter().product::<usize>())
                .map(|i| (i as f32) * 0.1)
                .collect::<Vec<_>>(),
            shape_b.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let mut our_result = our_a.sub_tensor(&our_b);
        our_result.backward(None);

        let our_grad_a = match our_a.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor A has no gradient".to_string()),
        };
        let our_grad_b = match our_b.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor B has no gradient".to_string()),
        };

        // LibTorch reference
        let torch_a = match LibTorchTensor::ones(shape_a) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch A requires_grad failed: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch A creation failed: {}", e))
            }
        };

        let torch_b_data = (0..shape_b.iter().product::<usize>())
            .map(|i| (i as f32) * 0.1)
            .collect::<Vec<_>>();
        let torch_b = match LibTorchTensor::from_data(&torch_b_data, shape_b) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch B requires_grad failed: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch B creation failed: {}", e))
            }
        };

        let torch_result = match torch_a.sub_tensor(&torch_b) {
            Ok(r) => r,
            Err(e) => return ComparisonResult::failure(format!("LibTorch sub failed: {}", e)),
        };

        // Create gradient tensor with same shape as result
        let grad_shape = torch_result.shape();
        let ones_data = vec![1.0; grad_shape.iter().product::<usize>()];
        let grad_tensor = match LibTorchTensor::from_data(&ones_data, &grad_shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };

        if let Err(e) = torch_result.backward(Some(&grad_tensor)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }

        let torch_grad_a = match torch_a.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor A has no gradient".to_string())
            }
        };

        let torch_grad_b = match torch_b.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor B has no gradient".to_string())
            }
        };

        // Compare shapes first
        if our_grad_a.shape().dims != torch_grad_a.shape() {
            return ComparisonResult::failure(format!(
                "Gradient A shape mismatch: our {:?} vs torch {:?}",
                our_grad_a.shape().dims,
                torch_grad_a.shape()
            ));
        }

        if our_grad_b.shape().dims != torch_grad_b.shape() {
            return ComparisonResult::failure(format!(
                "Gradient B shape mismatch: our {:?} vs torch {:?}",
                our_grad_b.shape().dims,
                torch_grad_b.shape()
            ));
        }

        // Compare gradient values
        let grad_a_comparison = self.compare_tensors(&our_grad_a, &torch_grad_a);
        if !grad_a_comparison.passed {
            return ComparisonResult::failure(format!(
                "Gradient A values mismatch: {}",
                grad_a_comparison.details
            ));
        }

        let grad_b_comparison = self.compare_tensors(&our_grad_b, &torch_grad_b);
        if !grad_b_comparison.passed {
            return ComparisonResult::failure(format!(
                "Gradient B values mismatch: {}",
                grad_b_comparison.details
            ));
        }

        ComparisonResult::success()
    }

    /// Test broadcasting division gradients
    pub fn test_div_broadcasting_gradients(
        &self,
        shape_a: &[usize],
        shape_b: &[usize],
    ) -> ComparisonResult {
        // Our implementation
        let our_a = Tensor::from_slice(
            &(1..=shape_a.iter().product::<usize>())
                .map(|i| (i as f32) * 0.5 + 1.0)
                .collect::<Vec<_>>(),
            shape_a.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let our_b = Tensor::from_slice(
            &(0..shape_b.iter().product::<usize>())
                .map(|i| (i as f32) * 0.2 + 2.0)
                .collect::<Vec<_>>(),
            shape_b.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let mut our_result = our_a.div_tensor(&our_b);
        our_result.backward(None);

        let our_grad_a = match our_a.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor A has no gradient".to_string()),
        };
        let our_grad_b = match our_b.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor B has no gradient".to_string()),
        };

        // LibTorch reference
        let torch_a_data = (1..=shape_a.iter().product::<usize>())
            .map(|i| (i as f32) * 0.5 + 1.0)
            .collect::<Vec<_>>();
        let torch_a = match LibTorchTensor::from_data(&torch_a_data, shape_a) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch A requires_grad failed: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch A creation failed: {}", e))
            }
        };

        let torch_b_data = (0..shape_b.iter().product::<usize>())
            .map(|i| (i as f32) * 0.2 + 2.0)
            .collect::<Vec<_>>();
        let torch_b = match LibTorchTensor::from_data(&torch_b_data, shape_b) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch B requires_grad failed: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch B creation failed: {}", e))
            }
        };

        let torch_result = match torch_a.div_tensor(&torch_b) {
            Ok(r) => r,
            Err(e) => return ComparisonResult::failure(format!("LibTorch div failed: {}", e)),
        };

        // Create gradient tensor
        let grad_shape = torch_result.shape();
        let ones_data = vec![1.0; grad_shape.iter().product::<usize>()];
        let grad_tensor = match LibTorchTensor::from_data(&ones_data, &grad_shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };

        if let Err(e) = torch_result.backward(Some(&grad_tensor)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }

        let torch_grad_a = match torch_a.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor A has no gradient".to_string())
            }
        };

        let torch_grad_b = match torch_b.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor B has no gradient".to_string())
            }
        };

        // Compare gradients
        let grad_a_comparison = self.compare_tensors(&our_grad_a, &torch_grad_a);
        if !grad_a_comparison.passed {
            return ComparisonResult::failure(format!(
                "Gradient A values mismatch: {}",
                grad_a_comparison.details
            ));
        }

        let grad_b_comparison = self.compare_tensors(&our_grad_b, &torch_grad_b);
        if !grad_b_comparison.passed {
            return ComparisonResult::failure(format!(
                "Gradient B values mismatch: {}",
                grad_b_comparison.details
            ));
        }

        ComparisonResult::success()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_broadcasting_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-6);

        // Test various broadcasting patterns
        let test_cases = vec![
            (vec![2, 3], vec![1, 3]), // [2,3] + [1,3]
            (vec![3, 1], vec![3, 4]), // [3,1] + [3,4]
            (vec![2, 3, 4], vec![4]), // [2,3,4] + [4]
            (vec![1], vec![2, 3]),    // [1] + [2,3] (scalar)
        ];

        for (shape_a, shape_b) in test_cases {
            let result = validator.test_add_broadcasting_gradients(&shape_a, &shape_b);
            assert!(
                result.passed,
                "Add broadcasting gradients failed for shapes {:?} + {:?}: {}",
                shape_a, shape_b, result.details
            );
        }
    }

    #[test]
    fn test_mul_broadcasting_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-6);

        // Test various broadcasting patterns
        let test_cases = vec![
            (vec![2, 3], vec![1, 3]), // [2,3] * [1,3]
            (vec![3, 1], vec![3, 4]), // [3,1] * [3,4]
            (vec![2, 3, 4], vec![4]), // [2,3,4] * [4]
        ];

        for (shape_a, shape_b) in test_cases {
            let result = validator.test_mul_broadcasting_gradients(&shape_a, &shape_b);
            assert!(
                result.passed,
                "Mul broadcasting gradients failed for shapes {:?} * {:?}: {}",
                shape_a, shape_b, result.details
            );
        }
    }

    #[test]
    fn test_linear_layer_pattern_validation() {
        let validator = TensorValidator::new(1e-6, 1e-6);

        // Test linear layer patterns
        let test_cases = vec![
            (vec![2, 4], vec![4, 3], vec![3]),   // Simple linear layer
            (vec![8, 10], vec![10, 5], vec![5]), // Larger layer
            (vec![1, 2], vec![2, 1], vec![1]),   // Single sample
        ];

        for (input_shape, weight_shape, bias_shape) in test_cases {
            let result = validator.test_linear_layer_broadcasting_gradients(
                &input_shape,
                &weight_shape,
                &bias_shape,
            );
            assert!(
                result.passed,
                "Linear layer pattern failed for input {:?}, weight {:?}, bias {:?}: {}",
                input_shape, weight_shape, bias_shape, result.details
            );
        }
    }

    #[test]
    fn test_sub_broadcasting_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-6);

        // Test various broadcasting patterns
        let test_cases = vec![
            (vec![2, 3], vec![1, 3]), // [2,3] - [1,3]
            (vec![3, 1], vec![3, 4]), // [3,1] - [3,4]
            (vec![2, 3, 4], vec![4]), // [2,3,4] - [4]
            (vec![1], vec![2, 3]),    // [1] - [2,3] (scalar)
        ];

        for (shape_a, shape_b) in test_cases {
            let result = validator.test_sub_broadcasting_gradients(&shape_a, &shape_b);
            assert!(
                result.passed,
                "Subtraction broadcasting gradient test failed for shapes {:?} - {:?}: {}",
                shape_a, shape_b, result.details
            );
        }
    }

    #[test]
    fn test_div_broadcasting_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-6);

        // Test various broadcasting patterns
        let test_cases = vec![
            (vec![2, 3], vec![1, 3]), // [2,3] / [1,3]
            (vec![3, 1], vec![3, 4]), // [3,1] / [3,4]
            (vec![2, 3, 4], vec![4]), // [2,3,4] / [4]
        ];

        for (shape_a, shape_b) in test_cases {
            let result = validator.test_div_broadcasting_gradients(&shape_a, &shape_b);
            assert!(
                result.passed,
                "Division broadcasting gradient test failed for shapes {:?} / {:?}: {}",
                shape_a, shape_b, result.details
            );
        }
    }
}
