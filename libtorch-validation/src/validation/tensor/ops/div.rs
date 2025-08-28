//! FFI validation tests for tensor division operations
//!
//! This module provides comprehensive validation of division operations
//! against LibTorch reference implementation with 0.00e0 error tolerance.

// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

/// FFI validation tests for tensor division operations
pub struct DivisionValidator {
    validator: TensorValidator,
}

impl DivisionValidator {
    /// Create a new division validator with specified tolerances
    pub fn new(relative_tolerance: f64, absolute_tolerance: f64) -> Self {
        Self {
            validator: TensorValidator::new(relative_tolerance, absolute_tolerance),
        }
    }

    /// Test scalar division validation against LibTorch
    pub fn test_div_scalar(&self, shape: &[usize], scalar: f32) -> ComparisonResult {
        // Create our tensor
        let our_tensor = Tensor::ones(shape.to_vec());

        // Create LibTorch tensor
        let torch_tensor = match LibTorchTensor::ones(shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch tensor creation failed: {}", e))
            }
        };

        // Perform division
        let our_result = our_tensor.div_scalar(scalar);
        let torch_result = match torch_tensor.div_scalar(scalar) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch div_scalar failed: {}", e))
            }
        };

        // Compare results
        self.validator.compare_tensors(&our_result, &torch_result)
    }

    /// Test tensor division validation against LibTorch
    pub fn test_div_tensor(&self, shape: &[usize]) -> ComparisonResult {
        // Create our tensors
        let our_tensor_a = Tensor::ones(shape.to_vec());
        let mut our_tensor_b = Tensor::ones(shape.to_vec());
        our_tensor_b.fill(2.0);

        // Create LibTorch tensors
        let torch_tensor_a = match LibTorchTensor::ones(shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor A creation failed: {}",
                    e
                ))
            }
        };

        let torch_tensor_b = match LibTorchTensor::ones(shape) {
            Ok(t) => match t.mul_scalar(2.0) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch tensor B creation failed: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor B creation failed: {}",
                    e
                ))
            }
        };

        // Perform division
        let our_result = our_tensor_a.div_tensor(&our_tensor_b);
        let torch_result = match torch_tensor_a.div_tensor(&torch_tensor_b) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch div_tensor failed: {}", e))
            }
        };

        // Compare results
        self.validator.compare_tensors(&our_result, &torch_result)
    }

    /// Test division operations with various shapes and values
    pub fn test_div_operations(&self) -> Vec<ComparisonResult> {
        let mut results = Vec::new();

        // Test shapes for comprehensive coverage
        let test_shapes = vec![
            vec![1],       // Scalar tensors
            vec![3],       // 1D vectors
            vec![2, 2],    // 2D matrices
            vec![1, 4],    // Broadcasting candidates
            vec![2, 3],    // Rectangular matrices
            vec![1, 1, 3], // 3D tensors
            vec![10, 10],  // Medium tensors
        ];

        // Test scalar division with various values (avoiding zero)
        let test_scalars = vec![1.0, 2.0, -1.0, 0.5, -0.5, 3.0, 0.25];

        for shape in &test_shapes {
            for &scalar in &test_scalars {
                results.push(self.test_div_scalar(shape, scalar));
            }
        }

        // Test tensor division with various shapes
        for shape in &test_shapes {
            results.push(self.test_div_tensor(shape));
        }

        results
    }

    /// Test broadcasting division operations
    pub fn test_div_tensor_broadcasting(
        &self,
        shape1: &[usize],
        shape2: &[usize],
    ) -> ComparisonResult {
        // Create our tensors with different values (avoiding zeros for division)
        let our_tensor_a = Tensor::from_slice(
            &(0..shape1.iter().product::<usize>())
                .map(|i| (i as f32) * 0.1 + 2.0)
                .collect::<Vec<_>>(),
            shape1.to_vec(),
        )
        .unwrap();
        let our_tensor_b = Tensor::from_slice(
            &(0..shape2.iter().product::<usize>())
                .map(|i| (i as f32) * 0.2 + 1.0)
                .collect::<Vec<_>>(),
            shape2.to_vec(),
        )
        .unwrap();

        let our_result = our_tensor_a.div_tensor(&our_tensor_b);

        // Create LibTorch tensors
        let data_a: Vec<f32> = (0..shape1.iter().product::<usize>())
            .map(|i| (i as f32) * 0.1 + 2.0)
            .collect();
        let torch_tensor_a = match LibTorchTensor::from_data(&data_a, shape1) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor A creation failed: {}",
                    e
                ))
            }
        };

        let data_b: Vec<f32> = (0..shape2.iter().product::<usize>())
            .map(|i| (i as f32) * 0.2 + 1.0)
            .collect();
        let torch_tensor_b = match LibTorchTensor::from_data(&data_b, shape2) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor B creation failed: {}",
                    e
                ))
            }
        };

        let torch_result = match torch_tensor_a.div_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch div_tensor failed: {}", e))
            }
        };

        self.validator.compare_tensors(&our_result, &torch_result)
    }

    /// Test gradient operations for DivisionValidator
    pub fn test_div_gradient_operations(&self) -> Vec<ComparisonResult> {
        let mut results = Vec::new();

        // Use the TensorValidator methods through the inner validator
        let validator = &self.validator;

        // Scalar gradient tests (avoiding division by zero)
        let scalar_grad_cases = vec![
            (vec![1], 1.0),
            (vec![1], 2.0),
            (vec![1], -1.0),
            (vec![3], 2.5),
            (vec![5], 0.5),
            (vec![2, 3], 1.5),
            (vec![4, 5], 0.25),
        ];

        for (shape, scalar) in scalar_grad_cases {
            let result = validator.test_div_scalar_gradients(&shape, scalar);
            results.push(result);
        }

        // Tensor gradient tests (same shape)
        let tensor_grad_shapes = vec![
            vec![1],
            vec![2],
            vec![5],
            vec![2, 3],
            vec![4, 5],
            vec![1, 1],
        ];

        for shape in tensor_grad_shapes {
            let result = validator.test_div_tensor_gradients(&shape);
            results.push(result);
        }

        // Key broadcasting gradient tests
        let broadcasting_grad_cases = vec![
            (vec![1], vec![3]),
            (vec![3], vec![1]),
            (vec![1], vec![2, 3]),
            (vec![2, 3], vec![1]),
            (vec![3], vec![2, 3]),
            (vec![1, 3], vec![2, 3]),
            (vec![2, 1], vec![2, 3]),
        ];

        for (shape1, shape2) in broadcasting_grad_cases {
            let result = validator.test_div_tensor_broadcasting_gradients(&shape1, &shape2);
            results.push(result);
        }

        results
    }
}

impl TensorValidator {
    /// Test gradient computation for scalar division against LibTorch
    pub fn test_div_scalar_gradients(&self, shape: &[usize], scalar: f32) -> ComparisonResult {
        let our_tensor = Tensor::ones(shape.to_vec()).with_requires_grad();
        let mut our_result = our_tensor.div_scalar(scalar);
        our_result.backward(None);

        let our_grad = match our_tensor.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor has no gradient".to_string()),
        };

        let torch_tensor = match LibTorchTensor::ones(shape) {
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
        let torch_result = match torch_tensor.div_scalar(scalar) {
            Ok(r) => r,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch div_scalar failed: {}", e))
            }
        };
        let grad_ones = match LibTorchTensor::ones(&torch_result.shape()) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };
        if let Err(e) = torch_result.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }
        let torch_grad = match torch_tensor.grad() {
            Some(g) => g,
            None => {
                return ComparisonResult::failure("LibTorch tensor has no gradient".to_string())
            }
        };
        self.compare_tensors(&our_grad, &torch_grad)
    }

    /// Test gradient computation for tensor division against LibTorch
    pub fn test_div_tensor_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let our_tensor_a = Tensor::ones(shape.to_vec()).with_requires_grad();
        let mut our_tensor_b = Tensor::ones(shape.to_vec()).with_requires_grad();
        our_tensor_b.fill(2.0);
        let mut our_result = our_tensor_a.div_tensor(&our_tensor_b);
        our_result.backward(None);

        let our_grad_a = match our_tensor_a.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor A has no gradient".to_string()),
        };
        let our_grad_b = match our_tensor_b.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor B has no gradient".to_string()),
        };

        let torch_tensor_a = match LibTorchTensor::ones(shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad on A: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor A creation failed: {}",
                    e
                ))
            }
        };
        let torch_tensor_b = match LibTorchTensor::ones(shape) {
            Ok(t) => match t.mul_scalar(2.0) {
                Ok(t) => match t.requires_grad_(true) {
                    Ok(t) => t,
                    Err(e) => {
                        return ComparisonResult::failure(format!(
                            "Failed to set requires_grad on B: {}",
                            e
                        ))
                    }
                },
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "LibTorch tensor B creation failed: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor B creation failed: {}",
                    e
                ))
            }
        };

        let torch_result = match torch_tensor_a.div_tensor(&torch_tensor_b) {
            Ok(r) => r,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch div_tensor failed: {}", e))
            }
        };
        let grad_ones = match LibTorchTensor::ones(&torch_result.shape()) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };
        if let Err(e) = torch_result.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }
        let torch_grad_a = match torch_tensor_a.grad() {
            Some(g) => g,
            None => {
                return ComparisonResult::failure("LibTorch tensor A has no gradient".to_string())
            }
        };
        let torch_grad_b = match torch_tensor_b.grad() {
            Some(g) => g,
            None => {
                return ComparisonResult::failure("LibTorch tensor B has no gradient".to_string())
            }
        };

        let cmp_a = self.compare_tensors(&our_grad_a, &torch_grad_a);
        if !cmp_a.passed {
            return ComparisonResult::failure(format!(
                "Gradient A comparison failed: {}",
                cmp_a.details
            ));
        }
        let cmp_b = self.compare_tensors(&our_grad_b, &torch_grad_b);
        if !cmp_b.passed {
            return ComparisonResult::failure(format!(
                "Gradient B comparison failed: {}",
                cmp_b.details
            ));
        }
        ComparisonResult::success()
    }

    /// Test broadcasting division operations against LibTorch
    pub fn test_div_tensor_broadcasting(
        &self,
        shape1: &[usize],
        shape2: &[usize],
    ) -> ComparisonResult {
        // Create our tensors with different values (avoiding zeros for division)
        let our_tensor_a = Tensor::from_slice(
            &(0..shape1.iter().product::<usize>())
                .map(|i| (i as f32) * 0.1 + 2.0)
                .collect::<Vec<_>>(),
            shape1.to_vec(),
        )
        .unwrap();
        let our_tensor_b = Tensor::from_slice(
            &(0..shape2.iter().product::<usize>())
                .map(|i| (i as f32) * 0.2 + 1.0)
                .collect::<Vec<_>>(),
            shape2.to_vec(),
        )
        .unwrap();

        let our_result = our_tensor_a.div_tensor(&our_tensor_b);

        // Create LibTorch tensors
        let data_a: Vec<f32> = (0..shape1.iter().product::<usize>())
            .map(|i| (i as f32) * 0.1 + 2.0)
            .collect();
        let torch_tensor_a = match LibTorchTensor::from_data(&data_a, shape1) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor A creation failed: {}",
                    e
                ))
            }
        };

        let data_b: Vec<f32> = (0..shape2.iter().product::<usize>())
            .map(|i| (i as f32) * 0.2 + 1.0)
            .collect();
        let torch_tensor_b = match LibTorchTensor::from_data(&data_b, shape2) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor B creation failed: {}",
                    e
                ))
            }
        };

        let torch_result = match torch_tensor_a.div_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch div_tensor failed: {}", e))
            }
        };

        self.compare_tensors(&our_result, &torch_result)
    }

    /// Test broadcasting division with gradient computation
    pub fn test_div_tensor_broadcasting_gradients(
        &self,
        shape1: &[usize],
        shape2: &[usize],
    ) -> ComparisonResult {
        // Our implementation with gradient tracking
        let our_tensor_a = Tensor::from_slice(
            &(0..shape1.iter().product::<usize>())
                .map(|i| (i as f32) * 0.1 + 2.0)
                .collect::<Vec<_>>(),
            shape1.to_vec(),
        )
        .unwrap()
        .with_requires_grad();
        let our_tensor_b = Tensor::from_slice(
            &(0..shape2.iter().product::<usize>())
                .map(|i| (i as f32) * 0.2 + 1.0)
                .collect::<Vec<_>>(),
            shape2.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let mut our_result = our_tensor_a.div_tensor(&our_tensor_b);
        our_result.backward(None);

        let our_grad_a = match our_tensor_a.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor A has no gradient".to_string()),
        };
        let our_grad_b = match our_tensor_b.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor B has no gradient".to_string()),
        };

        // LibTorch reference with gradient tracking
        let data_a: Vec<f32> = (0..shape1.iter().product::<usize>())
            .map(|i| (i as f32) * 0.1 + 2.0)
            .collect();
        let torch_tensor_a = match LibTorchTensor::from_data(&data_a, shape1) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad A: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor A creation failed: {}",
                    e
                ))
            }
        };

        let data_b: Vec<f32> = (0..shape2.iter().product::<usize>())
            .map(|i| (i as f32) * 0.2 + 1.0)
            .collect();
        let torch_tensor_b = match LibTorchTensor::from_data(&data_b, shape2) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad B: {}",
                        e
                    ))
                }
            },
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor B creation failed: {}",
                    e
                ))
            }
        };

        let torch_result = match torch_tensor_a.div_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch div_tensor failed: {}", e))
            }
        };

        let grad_ones = match LibTorchTensor::ones(&torch_result.shape()) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };

        if let Err(e) = torch_result.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }

        let torch_grad_a = match torch_tensor_a.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor A has no gradient".to_string())
            }
        };
        let torch_grad_b = match torch_tensor_b.grad() {
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor B has no gradient".to_string())
            }
        };

        let result_a = self.compare_tensors(&our_grad_a, &torch_grad_a);
        if !result_a.passed {
            return ComparisonResult::failure(format!(
                "Gradient A comparison failed: {}",
                result_a.details
            ));
        }
        let result_b = self.compare_tensors(&our_grad_b, &torch_grad_b);
        if !result_b.passed {
            return ComparisonResult::failure(format!(
                "Gradient B comparison failed: {}",
                result_b.details
            ));
        }
        ComparisonResult::success()
    }

    /// Comprehensive gradient validation test suite for division operations
    pub fn test_div_gradient_operations(&self) -> Vec<(String, ComparisonResult)> {
        let mut results = Vec::new();

        // Scalar gradient tests (avoiding division by zero)
        let scalar_grad_cases = vec![
            (vec![1], 1.0),
            (vec![1], 2.0),
            (vec![1], -1.0),
            (vec![3], 2.5),
            (vec![5], 0.5),
            (vec![2, 3], 1.5),
            (vec![4, 5], 0.25),
            (vec![3, 4, 5], 10.0),
            (vec![2, 3, 4, 5], 0.1),
        ];

        for (shape, scalar) in scalar_grad_cases {
            let test_name = format!("div_scalar_grad_{:?}_{}", shape, scalar);
            let result = self.test_div_scalar_gradients(&shape, scalar);
            results.push((test_name, result));
        }

        // Tensor gradient tests (same shape)
        let tensor_grad_shapes = vec![
            vec![1],
            vec![2],
            vec![5],
            vec![2, 3],
            vec![4, 5],
            vec![1, 1],
            vec![3, 4, 5],
            vec![2, 3, 4, 5],
        ];

        for shape in tensor_grad_shapes {
            let test_name = format!("div_tensor_grad_{:?}", shape);
            let result = self.test_div_tensor_gradients(&shape);
            results.push((test_name, result));
        }

        // Broadcasting gradient tests
        let broadcasting_grad_cases = vec![
            // Basic broadcasting patterns
            (vec![1], vec![3]),
            (vec![3], vec![1]),
            (vec![1], vec![2, 3]),
            (vec![2, 3], vec![1]),
            // Vector-matrix broadcasting
            (vec![3], vec![2, 3]),
            (vec![2, 3], vec![3]),
            (vec![1, 3], vec![2, 3]),
            (vec![2, 1], vec![2, 3]),
            (vec![3, 1], vec![3, 4]),
            (vec![1, 4], vec![3, 4]),
            // Multi-dimensional broadcasting
            (vec![1, 1, 3], vec![2, 4, 3]),
            (vec![2, 4, 1], vec![1, 1, 3]),
            (vec![1, 4, 1], vec![2, 1, 3]),
            (vec![2, 1, 3], vec![1, 4, 1]),
            // Higher rank broadcasting
            (vec![1], vec![2, 3, 4]),
            (vec![4], vec![2, 3, 4]),
            (vec![3, 4], vec![2, 3, 4]),
            (vec![1, 3, 4], vec![2, 3, 4]),
            (vec![2, 1, 4], vec![2, 3, 4]),
            (vec![2, 3, 1], vec![2, 3, 4]),
            // Complex broadcasting patterns
            (vec![1, 1, 1, 1], vec![2, 3, 4, 5]),
            (vec![2, 1, 4, 1], vec![1, 3, 1, 5]),
            (vec![1, 3, 1, 5], vec![2, 1, 4, 1]),
        ];

        for (shape1, shape2) in broadcasting_grad_cases {
            let test_name = format!("div_broadcast_grad_{:?}_{:?}", shape1, shape2);
            let result = self.test_div_tensor_broadcasting_gradients(&shape1, &shape2);
            results.push((test_name, result));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_scalar_validation() {
        let validator = DivisionValidator::new(1e-6, 1e-8);
        let result = validator.test_div_scalar(&[2, 3], 2.0);
        assert!(
            result.passed,
            "Scalar division validation failed: {}",
            result.details
        );
    }

    #[test]
    fn test_div_tensor_validation() {
        let validator = DivisionValidator::new(1e-6, 1e-8);
        let result = validator.test_div_tensor(&[2, 3]);
        assert!(
            result.passed,
            "Tensor division validation failed: {}",
            result.details
        );
    }

    #[test]
    fn test_div_operations_suite() {
        let validator = DivisionValidator::new(1e-6, 1e-8);
        let results = validator.test_div_operations();

        // Verify all tests passed
        for result in &results {
            assert!(
                result.passed,
                "Division validation failed: {}",
                result.details
            );
        }

        println!("All {} division validation tests passed", results.len());
    }

    #[test]
    fn test_comprehensive_div_gradient_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let results = validator.test_div_gradient_operations();

        let mut passed = 0;
        let mut failed = 0;

        for (test_name, result) in &results {
            if result.passed {
                passed += 1;
            } else {
                failed += 1;
                println!("FAILED div gradient test {}: {}", test_name, result.details);
            }
        }

        println!(
            "Div gradient validation: {} passed, {} failed",
            passed, failed
        );
        assert_eq!(failed, 0, "Some division gradient tests failed");
    }

    #[test]
    fn test_div_broadcasting_operations() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test various broadcasting patterns
        let broadcasting_cases = vec![
            (vec![1], vec![3]),
            (vec![3], vec![1]),
            (vec![1], vec![2, 3]),
            (vec![2, 3], vec![1]),
            (vec![3], vec![2, 3]),
            (vec![1, 3], vec![2, 3]),
            (vec![2, 1], vec![2, 3]),
        ];

        for (shape1, shape2) in broadcasting_cases {
            let result = validator.test_div_tensor_broadcasting(&shape1, &shape2);
            assert!(
                result.passed,
                "Broadcasting div failed for {:?} / {:?}: {}",
                shape1, shape2, result.details
            );
        }
    }

    #[test]
    fn test_div_edge_cases() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test edge cases for division (avoiding division by zero)
        let edge_cases = vec![
            (vec![2], 1.0),
            (vec![2], 0.1),
            (vec![3], 10.0),
            (vec![3], -2.0),
        ];

        for (shape, scalar) in edge_cases {
            let result = validator.test_div_scalar_gradients(&shape, scalar);
            assert!(
                result.passed,
                "Edge case div gradient failed for shape {:?}, scalar {}: {}",
                shape, scalar, result.details
            );
        }
    }

    #[test]
    fn test_neural_network_div_patterns() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Division patterns common in neural networks
        let neural_patterns = vec![
            // Normalization operations
            (vec![32], vec![8, 32]), // Per-feature normalization
            (vec![1, 64, 1, 1], vec![8, 64, 28, 28]), // Channel-wise normalization
            (vec![1], vec![8, 64, 28, 28]), // Global normalization
            // Learning rate scaling
            (vec![10], vec![32, 10]),  // Per-class scaling
            (vec![1], vec![128, 256]), // Weight scaling
        ];

        for (shape1, shape2) in neural_patterns {
            let result = validator.test_div_tensor_broadcasting_gradients(&shape1, &shape2);
            assert!(
                result.passed,
                "Neural network div pattern failed for {:?} / {:?}: {}",
                shape1, shape2, result.details
            );
        }
    }
}
