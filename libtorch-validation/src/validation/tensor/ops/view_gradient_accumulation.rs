//! Validation tests for gradient accumulation through tensor views
//!
//! This module validates that tensor views, element views, and iterator operations
//! correctly accumulate gradients back to the source tensor compared to LibTorch.

use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Test gradient accumulation through element views created by tensor iteration
    pub fn test_element_view_gradient_accumulation(&self, shape: &[usize]) -> ComparisonResult {
        let total_size = shape.iter().product::<usize>();
        assert!(total_size >= 3, "Need at least 3 elements for this test");

        // Our implementation using element views
        let our_source = Tensor::randn(shape.to_vec(), Some(60)).with_requires_grad();

        // Create element views and perform operations
        let our_elem_0 = our_source.element_view(0);
        let our_elem_1 = our_source.element_view(1);
        let our_elem_2 = our_source.element_view(2);

        // Operations on element views
        let our_result_0 = our_elem_0.mul_scalar(2.0);
        let our_result_1 = our_elem_1.add_scalar(1.0);
        let our_result_2 = our_elem_2.div_scalar(3.0);

        // Sum all results to create a single loss
        let our_combined = our_result_0
            .add_tensor(&our_result_1)
            .add_tensor(&our_result_2);
        let mut our_loss = our_combined.sum(); // Should be a scalar
        our_loss.backward(None);

        let our_grad = match our_source.grad_by_value() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("Our source tensor has no gradient".to_string())
            }
        };

        // LibTorch reference - simulate element operations using indexing
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

        // Create individual scalar tensors for each element operation
        let flat_data = torch_source.data();
        let torch_elem_0 = match LibTorchTensor::from_data(&[flat_data[0]], &[1]) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad elem_0: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch elem_0 creation failed: {}", e))
            }
        };

        let torch_elem_1 = match LibTorchTensor::from_data(&[flat_data[1]], &[1]) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad elem_1: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch elem_1 creation failed: {}", e))
            }
        };

        let torch_elem_2 = match LibTorchTensor::from_data(&[flat_data[2]], &[1]) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad elem_2: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch elem_2 creation failed: {}", e))
            }
        };

        // Operations on individual elements
        let torch_result_0 = match torch_elem_0.mul_scalar(2.0) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_scalar failed: {}", e))
            }
        };

        let torch_result_1 = match torch_elem_1.add_scalar(1.0) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_scalar failed: {}", e))
            }
        };

        let torch_result_2 = match torch_elem_2.div_scalar(3.0) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch div_scalar failed: {}", e))
            }
        };

        // Combine results
        let torch_combined_01 = match torch_result_0.add_tensor(&torch_result_1) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_tensor 0+1 failed: {}", e))
            }
        };

        let torch_combined = match torch_combined_01.add_tensor(&torch_result_2) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_tensor +2 failed: {}", e))
            }
        };

        let torch_loss = match torch_combined.sum() {
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

        // Manually construct expected gradient for LibTorch elements
        let torch_grad_0 = match torch_elem_0.grad() {
            Some(grad) => grad.data()[0],
            None => {
                return ComparisonResult::failure("LibTorch elem_0 has no gradient".to_string())
            }
        };

        let torch_grad_1 = match torch_elem_1.grad() {
            Some(grad) => grad.data()[0],
            None => {
                return ComparisonResult::failure("LibTorch elem_1 has no gradient".to_string())
            }
        };

        let torch_grad_2 = match torch_elem_2.grad() {
            Some(grad) => grad.data()[0],
            None => {
                return ComparisonResult::failure("LibTorch elem_2 has no gradient".to_string())
            }
        };

        // Create expected gradient tensor with the computed gradients
        let mut expected_grad_data = vec![0.0; total_size];
        expected_grad_data[0] = torch_grad_0;
        expected_grad_data[1] = torch_grad_1;
        expected_grad_data[2] = torch_grad_2;

        let expected_grad = match LibTorchTensor::from_data(&expected_grad_data, shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "Expected gradient tensor creation failed: {}",
                    e
                ))
            }
        };

        self.compare_tensors(&our_grad, &expected_grad)
    }

    /// Test gradient accumulation through tensor view operations
    pub fn test_view_operation_gradient_accumulation(&self, shape: &[usize]) -> ComparisonResult {
        assert!(
            shape.len() >= 2,
            "Shape must have at least 2 dimensions for view operations"
        );
        let total_size = shape.iter().product::<usize>();

        // Our implementation using view operations
        let our_source = Tensor::randn(shape.to_vec(), Some(61)).with_requires_grad();

        // Create views and perform operations
        let our_flattened = our_source.view(vec![total_size as i32]);
        let our_reshaped = our_flattened.view(vec![total_size as i32, 1]);

        // Operations on views
        let our_scaled = our_reshaped.mul_scalar(3.0);
        let our_final = our_scaled.add_scalar(0.5);
        let mut our_loss = our_final.sum();
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

        let torch_flattened = match torch_source.view(&[total_size]) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch view flatten failed: {}", e))
            }
        };

        let torch_reshaped = match torch_flattened.view(&[total_size, 1]) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch view reshape failed: {}", e))
            }
        };

        let torch_scaled = match torch_reshaped.mul_scalar(3.0) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_scalar failed: {}", e))
            }
        };

        let torch_final = match torch_scaled.add_scalar(0.5) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_scalar failed: {}", e))
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

    /// Test gradient accumulation through tensor iterator operations
    pub fn test_iterator_gradient_accumulation(&self, shape: &[usize]) -> ComparisonResult {
        let total_size = shape.iter().product::<usize>();
        assert!(
            total_size >= 4,
            "Need at least 4 elements for iterator test"
        );

        // Our implementation using tensor iterator
        let our_source = Tensor::randn(shape.to_vec(), Some(62)).with_requires_grad();

        // Use iterator to process elements and collect results
        let our_processed: Tensor = our_source
            .iter()
            .take(4) // Take first 4 elements
            .map(|elem| elem.mul_scalar(2.0).add_scalar(1.0))
            .collect();

        let mut our_loss = our_processed.sum();
        our_loss.backward(None);

        let our_grad = match our_source.grad_by_value() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("Our source tensor has no gradient".to_string())
            }
        };

        // LibTorch reference - manually process first 4 elements
        let data = our_source.data().to_vec();
        let _torch_source = match LibTorchTensor::from_data(&data, shape) {
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

        // Process first 4 elements manually
        let mut processed_data = Vec::new();
        for (i, elem) in data.iter().enumerate().take(4) {
            let elem_data = [*elem];
            let torch_elem = match LibTorchTensor::from_data(&elem_data, &[1]) {
                Ok(t) => match t.requires_grad_(true) {
                    Ok(t) => t,
                    Err(e) => {
                        return ComparisonResult::failure(format!(
                            "Failed to set requires_grad elem_{}: {}",
                            i, e
                        ))
                    }
                },
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch elem_{} creation failed: {}",
                        i, e
                    ))
                }
            };

            let torch_processed = match torch_elem.mul_scalar(2.0).and_then(|t| t.add_scalar(1.0)) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch processing elem_{} failed: {}",
                        i, e
                    ))
                }
            };

            processed_data.push(torch_processed.data()[0]);
        }

        let torch_processed_tensor = match LibTorchTensor::from_data(&processed_data, &[4]) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch processed tensor creation failed: {}",
                    e
                ))
            }
        };

        let _torch_loss = match torch_processed_tensor.sum() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch sum failed: {}", e)),
        };

        // For expected gradient, the first 4 elements should have gradient 2.0 (from mul_scalar(2.0))
        // and the rest should have gradient 0.0
        let mut expected_grad_data = vec![0.0; total_size];
        for (i, _) in data.iter().enumerate().take(4) {
            expected_grad_data[i] = 2.0; // Gradient from mul_scalar(2.0)
        }

        let expected_grad = match LibTorchTensor::from_data(&expected_grad_data, shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "Expected gradient tensor creation failed: {}",
                    e
                ))
            }
        };

        self.compare_tensors(&our_grad, &expected_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_view_gradient_accumulation_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_shapes = vec![
            vec![6],    // 1D tensor
            vec![2, 3], // 2D tensor
            vec![3, 4], // 2D tensor
        ];

        for shape in test_shapes {
            let result = validator.test_element_view_gradient_accumulation(&shape);
            assert!(
                result.passed,
                "Element view gradient accumulation validation failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_view_operation_gradient_accumulation_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_shapes = vec![
            vec![2, 3],    // 2D tensor
            vec![3, 4],    // 2D tensor
            vec![2, 2, 3], // 3D tensor
        ];

        for shape in test_shapes {
            let result = validator.test_view_operation_gradient_accumulation(&shape);
            assert!(
                result.passed,
                "View operation gradient accumulation validation failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_iterator_gradient_accumulation_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_shapes = vec![
            vec![8],    // 1D tensor
            vec![2, 4], // 2D tensor
            vec![3, 3], // 2D tensor
        ];

        for shape in test_shapes {
            let result = validator.test_iterator_gradient_accumulation(&shape);
            assert!(
                result.passed,
                "Iterator gradient accumulation validation failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }
}
