//! Subtraction operation validation methods
//!
//! Provides specialized validation methods for scalar and tensor subtraction operations
//! against LibTorch reference implementation.

// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Test sub_scalar operation against LibTorch
    pub fn test_sub_scalar(&self, shape: &[usize], scalar: f32) -> ComparisonResult {
        // Create our tensor
        let our_tensor = Tensor::ones(shape.to_vec());
        let our_result = our_tensor.sub_scalar(scalar);

        // Create LibTorch tensor and perform same operation
        let torch_tensor = match LibTorchTensor::ones(shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch tensor creation failed: {}", e))
            }
        };

        let torch_result = match torch_tensor.sub_scalar(scalar) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch sub_scalar failed: {}", e))
            }
        };

        self.compare_tensors(&our_result, &torch_result)
    }

    /// Test sub_tensor operation against LibTorch  
    pub fn test_sub_tensor(&self, shape: &[usize]) -> ComparisonResult {
        // Create our tensors
        let our_tensor_a = Tensor::ones(shape.to_vec());
        let our_tensor_b = Tensor::ones(shape.to_vec());
        let our_result = our_tensor_a.sub_tensor(&our_tensor_b);

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

        let torch_result = match torch_tensor_a.sub_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch sub_tensor failed: {}", e))
            }
        };

        self.compare_tensors(&our_result, &torch_result)
    }

    /// Comprehensive subtraction operation testing
    pub fn test_sub_operations(&self) -> Vec<(String, ComparisonResult)> {
        let mut results = Vec::new();

        // Test shapes for comprehensive coverage
        let test_shapes = vec![
            vec![1],
            vec![3],
            vec![2, 2],
            vec![1, 4],
            vec![2, 3],
            vec![1, 1, 3],
            vec![100, 100],
        ];

        // Test scalar values including edge cases
        let test_scalars = vec![0.0, 1.0, -1.0, 5.5, -3.2, 100.0, -100.0];

        // Test scalar subtraction across all shapes and scalars
        for shape in &test_shapes {
            for &scalar in &test_scalars {
                let test_name = format!("sub_scalar_shape_{:?}_scalar_{}", shape, scalar);
                let result = self.test_sub_scalar(shape, scalar);
                results.push((test_name, result));
            }
        }

        // Test tensor subtraction across all shapes
        for shape in &test_shapes {
            let test_name = format!("sub_tensor_shape_{:?}", shape);
            let result = self.test_sub_tensor(shape);
            results.push((test_name, result));
        }

        results
    }

    /// Test gradient computation for scalar subtraction
    pub fn test_sub_scalar_gradients(&self, shape: &[usize], scalar: f32) -> ComparisonResult {
        // Create our tensor with gradient tracking
        let our_tensor = Tensor::ones(shape.to_vec()).with_requires_grad();
        let mut our_result = our_tensor.sub_scalar(scalar);
        our_result.backward(None);

        let our_grad = match our_tensor.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor has no gradient".to_string()),
        };

        // Create LibTorch tensor with gradient tracking
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

        let torch_result = match torch_tensor.sub_scalar(scalar) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch sub_scalar failed: {}", e))
            }
        };

        // Create gradient tensor for backward pass (required for non-scalar outputs)
        let grad_tensor = match LibTorchTensor::ones(&torch_result.shape()) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };

        if let Err(e) = torch_result.backward(Some(&grad_tensor)) {
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

    /// Test gradient computation for tensor subtraction
    pub fn test_sub_tensor_gradients(&self, shape: &[usize]) -> ComparisonResult {
        // Create our tensors with gradient tracking
        let our_tensor_a = Tensor::ones(shape.to_vec()).with_requires_grad();
        let our_tensor_b = Tensor::ones(shape.to_vec()).with_requires_grad();
        let mut our_result = our_tensor_a.sub_tensor(&our_tensor_b);
        our_result.backward(None);

        let our_grad_a = match our_tensor_a.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor A has no gradient".to_string()),
        };

        let our_grad_b = match our_tensor_b.grad_by_value() {
            Some(grad) => grad,
            None => return ComparisonResult::failure("Our tensor B has no gradient".to_string()),
        };

        // Create LibTorch tensors with gradient tracking
        let torch_tensor_a = match LibTorchTensor::ones(shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => {
                    return ComparisonResult::failure(format!(
                        "Failed to set requires_grad on tensor A: {}",
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
                        "Failed to set requires_grad on tensor B: {}",
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

        let torch_result = match torch_tensor_a.sub_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch sub_tensor failed: {}", e))
            }
        };

        // Create gradient tensor for backward pass (required for non-scalar outputs)
        let grad_tensor = match LibTorchTensor::ones(&torch_result.shape()) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("Gradient tensor creation failed: {}", e))
            }
        };

        if let Err(e) = torch_result.backward(Some(&grad_tensor)) {
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

        // Compare gradients for both operands
        let grad_a_comparison = self.compare_tensors(&our_grad_a, &torch_grad_a);
        if !grad_a_comparison.passed {
            return ComparisonResult::failure(format!(
                "Gradient A comparison failed: {}",
                grad_a_comparison.details
            ));
        }

        let grad_b_comparison = self.compare_tensors(&our_grad_b, &torch_grad_b);
        if !grad_b_comparison.passed {
            return ComparisonResult::failure(format!(
                "Gradient B comparison failed: {}",
                grad_b_comparison.details
            ));
        }

        ComparisonResult::success()
    }

    /// Test broadcasting subtraction operations against LibTorch
    pub fn test_sub_tensor_broadcasting(
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

        let our_result = our_tensor_a.sub_tensor(&our_tensor_b);

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

        let torch_result = match torch_tensor_a.sub_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch sub_tensor failed: {}", e))
            }
        };

        self.compare_tensors(&our_result, &torch_result)
    }

    /// Test broadcasting subtraction with gradient computation
    pub fn test_sub_tensor_broadcasting_gradients(
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

        let mut our_result = our_tensor_a.sub_tensor(&our_tensor_b);
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

        let torch_result = match torch_tensor_a.sub_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch sub_tensor failed: {}", e))
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

    /// Comprehensive gradient validation test suite for subtraction operations
    pub fn test_sub_gradient_operations(&self) -> Vec<(String, ComparisonResult)> {
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
            let test_name = format!("sub_scalar_grad_{:?}_{}", shape, scalar);
            let result = self.test_sub_scalar_gradients(&shape, scalar);
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
            let test_name = format!("sub_tensor_grad_{:?}", shape);
            let result = self.test_sub_tensor_gradients(&shape);
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
            let test_name = format!("sub_broadcast_grad_{:?}_{:?}", shape1, shape2);
            let result = self.test_sub_tensor_broadcasting_gradients(&shape1, &shape2);
            results.push((test_name, result));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_scalar_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let result = validator.test_sub_scalar(&[3, 3], 2.5);
        assert!(
            result.passed,
            "Sub scalar validation failed: {}",
            result.details
        );
    }

    #[test]
    fn test_sub_tensor_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let result = validator.test_sub_tensor(&[2, 3]);
        assert!(
            result.passed,
            "Sub tensor validation failed: {}",
            result.details
        );
    }

    #[test]
    fn test_comprehensive_sub_operations() {
        let validator = TensorValidator::default();
        let results = validator.test_sub_operations();

        let failed_tests: Vec<_> = results
            .iter()
            .filter(|(_, result)| !result.passed)
            .collect();

        if !failed_tests.is_empty() {
            panic!("Some subtraction tests failed: {:?}", failed_tests);
        }
    }

    #[test]
    fn test_specific_sub_scalar_cases() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test edge cases for scalar subtraction
        let edge_cases = vec![
            (vec![1], 0.0),
            (vec![1], f32::MIN_POSITIVE),
            (vec![1], f32::MAX),
            (vec![1], -5.0),
            (vec![100], 1.0),
            (vec![10, 10], -5.0),
            (vec![2, 3, 4], 10.5),
        ];

        for (shape, scalar) in edge_cases {
            let result = validator.test_sub_scalar(&shape, scalar);
            assert!(
                result.passed,
                "Failed for shape {:?}, scalar {}: {}",
                shape, scalar, result.details
            );
        }
    }

    #[test]
    fn test_specific_sub_tensor_cases() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test various shapes for tensor subtraction
        let shapes = vec![
            vec![1],
            vec![2],
            vec![1, 1],
            vec![2, 2],
            vec![3, 4, 5],
            vec![1, 50],
            vec![50, 1],
            vec![2, 3, 4],
        ];

        for shape in shapes {
            let result = validator.test_sub_tensor(&shape);
            assert!(
                result.passed,
                "Failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_gradient_scalar_subtraction_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_cases = vec![
            (vec![1], 1.0),
            (vec![5], 2.5),
            (vec![2, 3], -1.5),
            (vec![4, 5], 0.0),
            (vec![2, 3, 4], 10.0),
            (vec![3], -5.0),
            (vec![10], 100.0),
        ];

        for (shape, scalar) in test_cases {
            let result = validator.test_sub_scalar(&shape, scalar);
            assert!(
                result.passed,
                "Gradient scalar sub shape {:?}, scalar {}: {}",
                shape, scalar, result.details
            );
            println!(
                "Gradient scalar sub shape {:?}, scalar {}: {}",
                shape, scalar, result.details
            );
        }
    }

    #[test]
    fn test_gradient_tensor_subtraction_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        let test_shapes = vec![
            vec![1],
            vec![3],
            vec![2, 3],
            vec![4, 5],
            vec![2, 3, 4],
            vec![10],
            vec![5, 6],
        ];

        for shape in test_shapes {
            let result = validator.test_sub_tensor(&shape);
            assert!(
                result.passed,
                "Gradient tensor sub shape {:?}: {}",
                shape, result.details
            );
            println!("Gradient tensor sub shape {:?}: {}", shape, result.details);
        }
    }

    #[test]
    fn test_subtraction_negative_results() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test cases that will produce negative results
        let test_cases = vec![
            (vec![3], 10.0),   // Small tensor - large scalar
            (vec![2, 2], 5.0), // 2D tensor - scalar
            (vec![1, 5], 3.0), // Different shapes
        ];

        for (shape, scalar) in test_cases {
            let result = validator.test_sub_scalar(&shape, scalar);
            assert!(
                result.passed,
                "Failed negative result test for shape {:?}, scalar {}: {}",
                shape, scalar, result.details
            );
        }
    }

    #[test]
    fn test_large_tensor_subtraction_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test large tensors to verify SIMD optimizations
        let large_shapes = vec![
            vec![1000],     // Large 1D
            vec![100, 100], // Large 2D
            vec![50, 50],   // Medium 2D
        ];

        for shape in large_shapes {
            // Test scalar subtraction
            let scalar_result = validator.test_sub_scalar(&shape, 2.5);
            assert!(
                scalar_result.passed,
                "Large tensor scalar sub failed for shape {:?}: {}",
                shape, scalar_result.details
            );

            // Test tensor subtraction
            let tensor_result = validator.test_sub_tensor(&shape);
            assert!(
                tensor_result.passed,
                "Large tensor sub failed for shape {:?}: {}",
                shape, tensor_result.details
            );
        }
    }

    #[test]
    fn test_edge_case_subtraction_values() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test extreme values
        let edge_cases = vec![
            (vec![2], 0.0),                // Zero subtraction
            (vec![2], f32::MIN_POSITIVE),  // Tiny positive
            (vec![2], -f32::MIN_POSITIVE), // Tiny negative
            (vec![3], 1e-10),              // Very small positive
            (vec![3], -1e-10),             // Very small negative
        ];

        for (shape, scalar) in edge_cases {
            let result = validator.test_sub_scalar(&shape, scalar);
            assert!(
                result.passed,
                "Edge case failed for shape {:?}, scalar {}: {}",
                shape, scalar, result.details
            );
        }
    }

    #[test]
    fn test_comprehensive_sub_gradient_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let results = validator.test_sub_gradient_operations();

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
            "Sub gradient validation: {} passed, {} failed",
            passed, failed
        );
        assert_eq!(failed, 0, "Some sub gradient tests failed");
    }

    #[test]
    fn test_sub_broadcasting_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Broadcasting patterns for subtraction
        let broadcasting_cases = vec![
            // Basic patterns
            (vec![1], vec![3]),
            (vec![3], vec![1]),
            (vec![1], vec![2, 3]),
            (vec![2, 3], vec![1]),
            // Neural network patterns
            (vec![128], vec![32, 128]),
            (vec![10], vec![32, 10]),
            (vec![3], vec![2, 3]),
            (vec![1, 3], vec![2, 3]),
            (vec![2, 1], vec![2, 3]),
            // Multi-dimensional
            (vec![1, 1, 3], vec![2, 4, 3]),
            (vec![1], vec![2, 3, 4]),
            (vec![4], vec![2, 3, 4]),
        ];

        for (shape1, shape2) in broadcasting_cases {
            let result = validator.test_sub_tensor_broadcasting(&shape1, &shape2);
            assert!(
                result.passed,
                "Broadcasting sub {:?} - {:?}: {}",
                shape1, shape2, result.details
            );
        }
    }

    #[test]
    fn test_sub_broadcasting_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Key broadcasting patterns for gradient testing
        let grad_broadcasting_cases = vec![
            (vec![1], vec![3]),                // scalar to vector
            (vec![3], vec![2, 3]),             // bias subtraction pattern
            (vec![1, 3], vec![2, 3]),          // row broadcast
            (vec![2, 1], vec![2, 3]),          // column broadcast
            (vec![128], vec![8, 128]),         // hidden layer pattern
            (vec![1, 1, 16], vec![8, 64, 16]), // attention pattern
        ];

        for (shape1, shape2) in grad_broadcasting_cases {
            let result = validator.test_sub_tensor_broadcasting_gradients(&shape1, &shape2);
            assert!(
                result.passed,
                "Broadcasting sub gradients {:?} - {:?}: {}",
                shape1, shape2, result.details
            );
        }
    }

    #[test]
    fn test_sub_gradient_edge_cases() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Edge cases for gradient computation
        let edge_cases = vec![
            // Zero cases
            (vec![1], 0.0),
            (vec![5], 0.0),
            // Small values
            (vec![3], f32::MIN_POSITIVE),
            (vec![2, 3], 1e-10),
            // Large values
            (vec![4, 5], 1000.0),
            // Negative values
            (vec![3, 4], -100.0),
        ];

        for (shape, scalar) in edge_cases {
            let result = validator.test_sub_scalar_gradients(&shape, scalar);
            assert!(
                result.passed,
                "Sub scalar gradient edge case {:?}, {}: {}",
                shape, scalar, result.details
            );
        }
    }

    #[test]
    fn test_sub_gradient_shapes_comprehensive() {
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
            let result = validator.test_sub_tensor_gradients(&shape);
            assert!(
                result.passed,
                "Sub tensor gradient shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_sub_neural_network_patterns() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Neural network specific patterns for subtraction (less common than add but still used)
        let neural_patterns = vec![
            // Residual connections (a - residual)
            (vec![128], vec![32, 128]),
            (vec![512], vec![64, 512]),
            // Feature normalization patterns
            (vec![1, 64], vec![8, 64]),
            (vec![64, 1], vec![64, 28]),
            // Attention difference patterns
            (vec![1, 8, 64], vec![16, 1, 1]),
            (vec![8, 1, 1], vec![1, 64, 512]),
        ];

        for (shape1, shape2) in neural_patterns {
            // Test both forward and gradient computation
            let forward_result = validator.test_sub_tensor_broadcasting(&shape1, &shape2);
            assert!(
                forward_result.passed,
                "Neural sub forward {:?} - {:?}: {}",
                shape1, shape2, forward_result.details
            );

            let gradient_result =
                validator.test_sub_tensor_broadcasting_gradients(&shape1, &shape2);
            assert!(
                gradient_result.passed,
                "Neural sub gradient {:?} - {:?}: {}",
                shape1, shape2, gradient_result.details
            );
        }
    }
}
