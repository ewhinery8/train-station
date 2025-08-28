//! Addition operation validation methods
//!
//! Provides specialized validation methods for scalar and tensor addition operations
//! against LibTorch reference implementation.

// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Test gradient computation for scalar addition against LibTorch
    pub fn test_add_scalar_gradients(&self, shape: &[usize], scalar: f32) -> ComparisonResult {
        // Our implementation with gradient tracking
        let our_tensor = Tensor::ones(shape.to_vec()).with_requires_grad();
        let mut our_result = our_tensor.add_scalar(scalar);
        our_result.backward(None);

        let our_grad = match our_tensor.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor has no gradient".to_string()),
        };

        // LibTorch reference with gradient tracking
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

        let torch_result = match torch_tensor.add_scalar(scalar) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_scalar failed: {}", e))
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
            Some(grad) => grad,
            None => {
                return ComparisonResult::failure("LibTorch tensor has no gradient".to_string())
            }
        };

        self.compare_tensors(&our_grad, &torch_grad)
    }

    /// Test gradient computation for tensor addition against LibTorch
    pub fn test_add_tensor_gradients(&self, shape: &[usize]) -> ComparisonResult {
        // Our implementation with gradient tracking
        let our_tensor_a = Tensor::ones(shape.to_vec()).with_requires_grad();
        let our_tensor_b = Tensor::ones(shape.to_vec()).with_requires_grad();
        let mut our_result = our_tensor_a.add_tensor(&our_tensor_b);
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
        let torch_tensor_a = match LibTorchTensor::ones(shape) {
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
        let torch_tensor_b = match LibTorchTensor::ones(shape) {
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

        let torch_result = match torch_tensor_a.add_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_tensor failed: {}", e))
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
    /// Test add_scalar operation against LibTorch
    pub fn test_add_scalar(&self, shape: &[usize], scalar: f32) -> ComparisonResult {
        // Create our tensor
        let our_tensor = Tensor::ones(shape.to_vec());
        let our_result = our_tensor.add_scalar(scalar);

        // Create LibTorch tensor and perform same operation
        let torch_tensor = match LibTorchTensor::ones(shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch tensor creation failed: {}", e))
            }
        };

        let torch_result = match torch_tensor.add_scalar(scalar) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_scalar failed: {}", e))
            }
        };

        self.compare_tensors(&our_result, &torch_result)
    }

    /// Test add_tensor operation against LibTorch  
    pub fn test_add_tensor(&self, shape: &[usize]) -> ComparisonResult {
        // Create our tensors
        let our_tensor_a = Tensor::ones(shape.to_vec());
        let our_tensor_b = Tensor::ones(shape.to_vec());
        let our_result = our_tensor_a.add_tensor(&our_tensor_b);

        // Create LibTorch tensors and perform same operation
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
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor B creation failed: {}",
                    e
                ))
            }
        };

        let torch_result = match torch_tensor_a.add_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_tensor failed: {}", e))
            }
        };

        self.compare_tensors(&our_result, &torch_result)
    }

    /// Test broadcasting addition operations against LibTorch
    ///
    /// Tests various broadcasting patterns to ensure our implementation
    /// matches LibTorch's broadcasting semantics exactly.
    pub fn test_add_tensor_broadcasting(
        &self,
        shape1: &[usize],
        shape2: &[usize],
    ) -> ComparisonResult {
        // Create our tensors with different values to test broadcasting
        let our_tensor_a = Tensor::ones(shape1.to_vec());
        let our_tensor_b = Tensor::from_slice(
            &(0..shape2.iter().product::<usize>())
                .map(|i| (i as f32) * 0.1)
                .collect::<Vec<_>>(),
            shape2.to_vec(),
        )
        .unwrap();

        let our_result = our_tensor_a.add_tensor(&our_tensor_b);

        // Create LibTorch tensors and perform same operation
        let torch_tensor_a = match LibTorchTensor::ones(shape1) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!(
                    "LibTorch tensor A creation failed: {}",
                    e
                ))
            }
        };

        let data_b: Vec<f32> = (0..shape2.iter().product::<usize>())
            .map(|i| (i as f32) * 0.1)
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

        let torch_result = match torch_tensor_a.add_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_tensor failed: {}", e))
            }
        };

        self.compare_tensors(&our_result, &torch_result)
    }

    /// Test broadcasting addition with gradient computation
    pub fn test_add_tensor_broadcasting_gradients(
        &self,
        shape1: &[usize],
        shape2: &[usize],
    ) -> ComparisonResult {
        // Our implementation with gradient tracking
        let our_tensor_a = Tensor::ones(shape1.to_vec()).with_requires_grad();
        let our_tensor_b = Tensor::from_slice(
            &(0..shape2.iter().product::<usize>())
                .map(|i| (i as f32) * 0.1)
                .collect::<Vec<_>>(),
            shape2.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let mut our_result = our_tensor_a.add_tensor(&our_tensor_b);
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
        let torch_tensor_a = match LibTorchTensor::ones(shape1) {
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
            .map(|i| (i as f32) * 0.1)
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

        let torch_result = match torch_tensor_a.add_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch add_tensor failed: {}", e))
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

    /// Comprehensive test suite for addition operations including broadcasting
    pub fn test_add_operations(&self) -> Vec<(String, ComparisonResult)> {
        let mut results = Vec::new();

        // Scalar addition tests across various shapes
        let scalar_test_cases = vec![
            (vec![1], 1.0),
            (vec![5], 2.5),
            (vec![2, 3], -1.5),
            (vec![4, 5], 0.0),
            (vec![2, 3, 4], 10.0),
        ];

        for (shape, scalar) in scalar_test_cases {
            let test_name = format!("add_scalar_{:?}_{}", shape, scalar);
            let result = self.test_add_scalar(&shape, scalar);
            results.push((test_name, result));
        }

        // Tensor addition tests across various shapes (same shape)
        let tensor_test_shapes = vec![vec![1], vec![3], vec![2, 3], vec![4, 5], vec![2, 3, 4]];

        for shape in tensor_test_shapes {
            let test_name = format!("add_tensor_{:?}", shape);
            let result = self.test_add_tensor(&shape);
            results.push((test_name, result));
        }

        // Broadcasting addition tests - comprehensive test cases
        let broadcasting_test_cases = vec![
            // Basic scalar-like broadcasting
            (vec![1], vec![3]),
            (vec![3], vec![1]),
            (vec![1], vec![2, 3]),
            (vec![2, 3], vec![1]),
            // Vector-matrix broadcasting (common in neural networks)
            (vec![3], vec![2, 3]),
            (vec![2, 3], vec![3]),
            (vec![1, 3], vec![2, 3]),
            (vec![2, 1], vec![2, 3]),
            // Multi-dimensional broadcasting
            (vec![1, 1, 3], vec![2, 4, 3]),
            (vec![2, 4, 1], vec![1, 1, 3]),
            (vec![1, 4, 1], vec![2, 1, 3]),
            // Higher rank broadcasting
            (vec![1], vec![2, 3, 4]),
            (vec![2, 3, 4], vec![1]),
            (vec![4], vec![2, 3, 4]),
            (vec![3, 4], vec![2, 3, 4]),
            (vec![1, 3, 4], vec![2, 3, 4]),
            (vec![2, 1, 4], vec![2, 3, 4]),
            (vec![2, 3, 1], vec![2, 3, 4]),
            // Complex broadcasting patterns
            (vec![1, 1, 1, 1], vec![2, 3, 4, 5]),
            (vec![2, 1, 4, 1], vec![1, 3, 1, 5]),
        ];

        for (shape1, shape2) in broadcasting_test_cases {
            let test_name = format!("add_tensor_broadcast_{:?}_{:?}", shape1, shape2);
            let result = self.test_add_tensor_broadcasting(&shape1, &shape2);
            results.push((test_name, result));
        }

        results
    }

    /// Comprehensive gradient validation test suite for addition operations
    pub fn test_add_gradient_operations(&self) -> Vec<(String, ComparisonResult)> {
        let mut results = Vec::new();

        // Scalar gradient tests
        let scalar_grad_cases = vec![
            (vec![1], 0.0),
            (vec![1], 1.0),
            (vec![1], -1.0),
            (vec![3], 2.5),
            (vec![5], -3.7),
            (vec![2, 3], 1.5),
            (vec![4, 5], 0.0),
            (vec![3, 4, 5], 10.0),
            (vec![2, 3, 4, 5], -5.0),
        ];

        for (shape, scalar) in scalar_grad_cases {
            let test_name = format!("add_scalar_grad_{:?}_{}", shape, scalar);
            let result = self.test_add_scalar_gradients(&shape, scalar);
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
            let test_name = format!("add_tensor_grad_{:?}", shape);
            let result = self.test_add_tensor_gradients(&shape);
            results.push((test_name, result));
        }

        // Broadcasting gradient tests - comprehensive patterns
        let broadcasting_grad_cases = vec![
            // Basic scalar-like broadcasting
            (vec![1], vec![3]),
            (vec![3], vec![1]),
            (vec![1], vec![2, 3]),
            (vec![2, 3], vec![1]),
            // Vector-matrix broadcasting (neural network patterns)
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
            // Higher rank broadcasting (common in deep learning)
            (vec![1], vec![2, 3, 4]),
            (vec![4], vec![2, 3, 4]),
            (vec![3, 4], vec![2, 3, 4]),
            (vec![1, 3, 4], vec![2, 3, 4]),
            (vec![2, 1, 4], vec![2, 3, 4]),
            (vec![2, 3, 1], vec![2, 3, 4]),
            // Convolutional layer patterns
            (vec![1, 1, 1, 1], vec![2, 3, 4, 5]),
            (vec![2, 1, 4, 1], vec![1, 3, 1, 5]),
            (vec![1, 3, 1, 5], vec![2, 1, 4, 1]),
            // Attention mechanism patterns
            (vec![1, 1, 16], vec![8, 64, 16]),
            (vec![8, 1, 1], vec![1, 64, 16]),
            (vec![1, 64, 1], vec![8, 1, 16]),
        ];

        for (shape1, shape2) in broadcasting_grad_cases {
            let test_name = format!("add_broadcast_grad_{:?}_{:?}", shape1, shape2);
            let result = self.test_add_tensor_broadcasting_gradients(&shape1, &shape2);
            results.push((test_name, result));
        }

        results
    }

    /// Comprehensive test suite for broadcasting operations with gradients
    pub fn test_add_broadcasting_operations(&self) -> Vec<(String, ComparisonResult)> {
        let mut results = Vec::new();

        // Broadcasting addition tests with gradients
        let broadcasting_test_cases = vec![
            // Essential neural network patterns
            (vec![1], vec![3]),       // scalar to vector
            (vec![3], vec![2, 3]),    // bias addition
            (vec![1, 3], vec![2, 3]), // row broadcast
            (vec![2, 1], vec![2, 3]), // column broadcast
            // Multi-dimensional patterns
            (vec![1, 1, 3], vec![2, 4, 3]),
            (vec![1], vec![2, 3, 4]),
            (vec![4], vec![2, 3, 4]),
            (vec![1, 3, 4], vec![2, 3, 4]),
        ];

        for (shape1, shape2) in broadcasting_test_cases {
            let test_name = format!("add_tensor_broadcast_grad_{:?}_{:?}", shape1, shape2);
            let result = self.test_add_tensor_broadcasting_gradients(&shape1, &shape2);
            results.push((test_name, result));
        }

        results
    }
}

#[cfg(test)]
mod add_validation_tests {
    use super::*;

    #[test]
    fn test_add_scalar_validation_basic() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let result = validator.test_add_scalar(&[2, 2], 5.0);
        assert!(
            result.passed,
            "Basic add_scalar validation failed: {}",
            result.details
        );
    }

    #[test]
    fn test_add_tensor_validation_basic() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let result = validator.test_add_tensor(&[2, 2]);
        assert!(
            result.passed,
            "Basic add_tensor validation failed: {}",
            result.details
        );
    }

    #[test]
    fn test_add_operations_comprehensive() {
        let validator = TensorValidator::default();
        let results = validator.test_add_operations();

        let mut passed = 0;
        let mut failed = 0;

        for (test_name, result) in &results {
            if result.passed {
                passed += 1;
            } else {
                failed += 1;
                println!("FAILED: {}: {}", test_name, result.details);
            }
        }

        assert_eq!(
            failed, 0,
            "Some add operations failed validation: {} passed, {} failed",
            passed, failed
        );
    }

    #[test]
    fn test_comprehensive_add_operations() {
        let validator = TensorValidator::default();
        let results = validator.test_add_operations();

        let mut passed = 0;
        let mut failed = 0;

        for (test_name, result) in &results {
            if result.passed {
                passed += 1;
                println!("{}: {}", test_name, result.details);
            } else {
                failed += 1;
                println!("{}: {}", test_name, result.details);
            }
        }

        println!("\nTest Summary: {} passed, {} failed", passed, failed);

        // Fail the test if any individual test failed
        if failed > 0 {
            panic!("Some tensor operations failed validation against libtorch");
        }
    }

    #[test]
    fn test_specific_add_scalar_cases() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test edge cases
        let edge_cases = vec![
            (vec![1], 0.0),
            (vec![1], f32::MIN_POSITIVE),
            (vec![1], f32::MAX),
            (vec![100], 1.0),
            (vec![10, 10], -5.0),
        ];

        for (shape, scalar) in edge_cases {
            let result = validator.test_add_scalar(&shape, scalar);
            assert!(
                result.passed,
                "Failed for shape {:?}, scalar {}: {}",
                shape, scalar, result.details
            );
        }
    }

    #[test]
    fn test_specific_add_tensor_cases() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test various shapes
        let shapes = vec![
            vec![1],
            vec![2],
            vec![1, 1],
            vec![2, 2],
            vec![3, 4, 5],
            vec![1, 50],
            vec![50, 1],
        ];

        for shape in shapes {
            let result = validator.test_add_tensor(&shape);
            assert!(
                result.passed,
                "Failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_gradient_scalar_addition_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_cases = vec![
            (vec![1], 1.0),
            (vec![5], 2.5),
            (vec![2, 3], -1.5),
            (vec![4, 5], 0.0),
            (vec![2, 3, 4], 10.0),
        ];

        for (shape, scalar) in test_cases {
            let result = validator.test_add_scalar(&shape, scalar);
            assert!(
                result.passed,
                "Gradient scalar add shape {:?}, scalar {}: {}",
                shape, scalar, result.details
            );
            println!(
                "Gradient scalar add shape {:?}, scalar {}: {}",
                shape, scalar, result.details
            );
        }
    }

    #[test]
    fn test_gradient_tensor_addition_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_shapes = vec![vec![1], vec![3], vec![2, 3], vec![4, 5], vec![2, 3, 4]];

        for shape in test_shapes {
            let result = validator.test_add_tensor(&shape);
            assert!(
                result.passed,
                "Gradient tensor add shape {:?}: {}",
                shape, result.details
            );
            println!("Gradient tensor add shape {:?}: {}", shape, result.details);
        }
    }

    #[test]
    fn test_broadcasting_addition_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Essential broadcasting patterns for neural networks
        let broadcasting_cases = vec![
            // Scalar broadcasting
            (vec![1], vec![3]),
            (vec![1], vec![2, 3]),
            // Bias addition patterns
            (vec![3], vec![2, 3]),    // bias to matrix
            (vec![1, 3], vec![2, 3]), // row vector to matrix
            (vec![2, 1], vec![2, 3]), // column vector to matrix
            // Multi-dimensional patterns
            (vec![1], vec![2, 3, 4]),
            (vec![4], vec![2, 3, 4]),
            (vec![1, 3, 4], vec![2, 3, 4]),
        ];

        for (shape1, shape2) in broadcasting_cases {
            let result = validator.test_add_tensor_broadcasting(&shape1, &shape2);
            assert!(
                result.passed,
                "Broadcasting add {:?} + {:?}: {}",
                shape1, shape2, result.details
            );
            println!("Broadcasting add {:?} + {:?}: PASSED", shape1, shape2);
        }
    }

    #[test]
    fn test_broadcasting_addition_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Key broadcasting patterns that need correct gradient computation
        let grad_broadcasting_cases = vec![
            (vec![1], vec![3]),       // scalar to vector
            (vec![3], vec![2, 3]),    // bias addition - most common in neural networks
            (vec![1, 3], vec![2, 3]), // row broadcast
            (vec![2, 1], vec![2, 3]), // column broadcast
        ];

        for (shape1, shape2) in grad_broadcasting_cases {
            let result = validator.test_add_tensor_broadcasting_gradients(&shape1, &shape2);
            assert!(
                result.passed,
                "Broadcasting add gradients {:?} + {:?}: {}",
                shape1, shape2, result.details
            );
            println!(
                "Broadcasting add gradients {:?} + {:?}: PASSED",
                shape1, shape2
            );
        }
    }

    #[test]
    fn test_comprehensive_broadcasting_operations() {
        let validator = TensorValidator::default();
        let results = validator.test_add_operations(); // This now includes broadcasting tests

        let mut passed = 0;
        let mut failed = 0;
        let mut broadcasting_tests = 0;

        for (test_name, result) in &results {
            if test_name.contains("broadcast") {
                broadcasting_tests += 1;
            }

            if result.passed {
                passed += 1;
                if test_name.contains("broadcast") {
                    println!("✓ BROADCASTING: {}", test_name);
                }
            } else {
                failed += 1;
                println!("✗ FAILED: {}: {}", test_name, result.details);
            }
        }

        println!("\nBroadcasting Test Summary:");
        println!("  Total broadcasting tests: {}", broadcasting_tests);
        println!("  Overall: {} passed, {} failed", passed, failed);

        assert_eq!(
            failed, 0,
            "Some broadcasting operations failed validation: {} passed, {} failed",
            passed, failed
        );
    }

    #[test]
    fn test_neural_network_broadcasting_patterns() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Real-world neural network broadcasting patterns
        let neural_net_cases = vec![
            // Linear layer bias addition
            (vec![32], vec![8, 32]), // bias [features] + activations [batch, features]
            (vec![64], vec![16, 64]), // bias [hidden] + hidden layer [batch, hidden]
            (vec![10], vec![32, 10]), // bias [classes] + logits [batch, classes]
            // Convolutional layer bias addition (properly shaped)
            (vec![1, 16, 1, 1], vec![8, 16, 32, 32]), // bias [1, channels, 1, 1] + feature maps [batch, channels, h, w]
            (vec![1, 64, 1, 1], vec![8, 64, 28, 28]), // bias [1, channels, 1, 1] + feature maps
            // Attention mechanisms
            (vec![1, 1, 512], vec![8, 64, 512]), // positional encoding
            (vec![1, 8, 1], vec![16, 8, 64]),    // attention weights
        ];

        for (shape1, shape2) in neural_net_cases {
            let result = validator.test_add_tensor_broadcasting(&shape1, &shape2);
            assert!(
                result.passed,
                "Neural network broadcasting {:?} + {:?}: {}",
                shape1, shape2, result.details
            );
            println!("✓ Neural net pattern {:?} + {:?}: PASSED", shape1, shape2);
        }
    }

    #[test]
    fn test_comprehensive_add_gradient_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let results = validator.test_add_gradient_operations();

        let mut passed = 0;
        let mut failed = 0;

        for (test_name, result) in &results {
            if result.passed {
                passed += 1;
            } else {
                failed += 1;
                println!("FAILED: {}: {}", test_name, result.details);
            }
        }

        println!(
            "Add gradient validation: {} passed, {} failed",
            passed, failed
        );
        assert_eq!(failed, 0, "Some add gradient tests failed");
    }

    #[test]
    fn test_add_gradient_edge_cases() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Edge cases for gradient computation
        let edge_cases = vec![
            // Zero values
            (vec![1], 0.0),
            (vec![5], 0.0),
            // Very small values
            (vec![3], f32::MIN_POSITIVE),
            // Large values
            (vec![2, 3], 1000.0),
            // Negative values
            (vec![4, 5], -100.0),
        ];

        for (shape, scalar) in edge_cases {
            let result = validator.test_add_scalar_gradients(&shape, scalar);
            assert!(
                result.passed,
                "Add scalar gradient edge case {:?}, {}: {}",
                shape, scalar, result.details
            );
        }
    }

    #[test]
    fn test_add_gradient_shapes_comprehensive() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test various tensor shapes for gradient accuracy
        let shapes = vec![
            vec![1],
            vec![2],
            vec![10],
            vec![1, 1],
            vec![2, 3],
            vec![5, 4],
            vec![1, 5, 1],
            vec![3, 4, 5],
            vec![2, 3, 4, 5],
        ];

        for shape in shapes {
            let result = validator.test_add_tensor_gradients(&shape);
            assert!(
                result.passed,
                "Add tensor gradient shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_add_broadcasting_gradient_neural_patterns() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Neural network specific broadcasting patterns for gradients
        let neural_patterns = vec![
            // Bias addition patterns
            (vec![128], vec![32, 128]), // bias to hidden layer
            (vec![10], vec![32, 10]),   // bias to output layer
            (vec![512], vec![64, 512]), // bias to transformer layer
            // CNN bias patterns (properly shaped)
            (vec![1, 64, 1, 1], vec![8, 64, 28, 28]), // conv bias broadcasting
            (vec![1, 32, 1, 1], vec![4, 32, 14, 14]), // another conv bias pattern
            // Attention patterns
            (vec![1, 1, 512], vec![8, 64, 512]), // positional encoding
            (vec![1, 8, 1], vec![16, 8, 64]),    // attention weights
            (vec![8, 1, 64], vec![1, 16, 1]),    // attention values
        ];

        for (shape1, shape2) in neural_patterns {
            let result = validator.test_add_tensor_broadcasting_gradients(&shape1, &shape2);
            assert!(
                result.passed,
                "Neural broadcasting gradient {:?} + {:?}: {}",
                shape1, shape2, result.details
            );
        }
    }
}
