//! FFI validation tests for tensor multiplication operations
//!
//! This module provides comprehensive validation of multiplication operations
//! against LibTorch reference implementation with 0.00e0 error tolerance.

// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

/// FFI validation tests for tensor multiplication operations
pub struct MultiplicationValidator {
    validator: TensorValidator,
}

impl MultiplicationValidator {
    /// Create a new multiplication validator with specified tolerances
    pub fn new(relative_tolerance: f64, absolute_tolerance: f64) -> Self {
        Self {
            validator: TensorValidator::new(relative_tolerance, absolute_tolerance),
        }
    }

    /// Test scalar multiplication validation against LibTorch
    pub fn test_mul_scalar(&self, shape: &[usize], scalar: f32) -> ComparisonResult {
        // Create our tensor
        let our_tensor = Tensor::ones(shape.to_vec());

        // Create LibTorch tensor
        let torch_tensor = match LibTorchTensor::ones(shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch tensor creation failed: {}", e))
            }
        };

        // Perform multiplication
        let our_result = our_tensor.mul_scalar(scalar);
        let torch_result = match torch_tensor.mul_scalar(scalar) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_scalar failed: {}", e))
            }
        };

        // Compare results
        self.validator.compare_tensors(&our_result, &torch_result)
    }

    /// Test tensor multiplication validation against LibTorch
    pub fn test_mul_tensor(&self, shape: &[usize]) -> ComparisonResult {
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

        // Perform multiplication
        let our_result = our_tensor_a.mul_tensor(&our_tensor_b);
        let torch_result = match torch_tensor_a.mul_tensor(&torch_tensor_b) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_tensor failed: {}", e))
            }
        };

        // Compare results
        self.validator.compare_tensors(&our_result, &torch_result)
    }

    /// Test multiplication operations with various shapes and values
    pub fn test_mul_operations(&self) -> Vec<ComparisonResult> {
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

        // Test scalar multiplication with various values
        let test_scalars = vec![0.0, 1.0, 2.0, -1.0, 0.5, -0.5];

        for shape in &test_shapes {
            for &scalar in &test_scalars {
                results.push(self.test_mul_scalar(shape, scalar));
            }
        }

        // Test tensor multiplication with various shapes
        for shape in &test_shapes {
            results.push(self.test_mul_tensor(shape));
        }

        results
    }
}

impl TensorValidator {
    /// Test gradient computation for scalar multiplication against LibTorch
    pub fn test_mul_scalar_gradients(&self, shape: &[usize], scalar: f32) -> ComparisonResult {
        let our_tensor = Tensor::ones(shape.to_vec()).with_requires_grad();
        let mut our_result = our_tensor.mul_scalar(scalar);
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

        let torch_result = match torch_tensor.mul_scalar(scalar) {
            Ok(r) => r,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_scalar failed: {}", e))
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

    /// Test gradient computation for tensor multiplication against LibTorch
    pub fn test_mul_tensor_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let our_tensor_a = Tensor::ones(shape.to_vec()).with_requires_grad();
        let mut our_tensor_b = Tensor::ones(shape.to_vec()).with_requires_grad();
        our_tensor_b.fill(2.0);
        let mut our_result = our_tensor_a.mul_tensor(&our_tensor_b);
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

        let torch_result = match torch_tensor_a.mul_tensor(&torch_tensor_b) {
            Ok(r) => r,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_tensor failed: {}", e))
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
}

impl TensorValidator {
    /// Test broadcasting multiplication operations against LibTorch
    pub fn test_mul_tensor_broadcasting(
        &self,
        shape1: &[usize],
        shape2: &[usize],
    ) -> ComparisonResult {
        // Create our tensors with different values
        let our_tensor_a = Tensor::from_slice(
            &(0..shape1.iter().product::<usize>())
                .map(|i| (i as f32) * 0.1 + 1.0)
                .collect::<Vec<_>>(),
            shape1.to_vec(),
        )
        .unwrap();
        let our_tensor_b = Tensor::from_slice(
            &(0..shape2.iter().product::<usize>())
                .map(|i| (i as f32) * 0.2 + 0.5)
                .collect::<Vec<_>>(),
            shape2.to_vec(),
        )
        .unwrap();

        let our_result = our_tensor_a.mul_tensor(&our_tensor_b);

        // Create LibTorch tensors
        let data_a: Vec<f32> = (0..shape1.iter().product::<usize>())
            .map(|i| (i as f32) * 0.1 + 1.0)
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
            .map(|i| (i as f32) * 0.2 + 0.5)
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

        let torch_result = match torch_tensor_a.mul_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_tensor failed: {}", e))
            }
        };

        self.compare_tensors(&our_result, &torch_result)
    }

    /// Test broadcasting multiplication with gradient computation
    pub fn test_mul_tensor_broadcasting_gradients(
        &self,
        shape1: &[usize],
        shape2: &[usize],
    ) -> ComparisonResult {
        // Our implementation with gradient tracking
        let our_tensor_a = Tensor::from_slice(
            &(0..shape1.iter().product::<usize>())
                .map(|i| (i as f32) * 0.1 + 1.0)
                .collect::<Vec<_>>(),
            shape1.to_vec(),
        )
        .unwrap()
        .with_requires_grad();
        let our_tensor_b = Tensor::from_slice(
            &(0..shape2.iter().product::<usize>())
                .map(|i| (i as f32) * 0.2 + 0.5)
                .collect::<Vec<_>>(),
            shape2.to_vec(),
        )
        .unwrap()
        .with_requires_grad();

        let mut our_result = our_tensor_a.mul_tensor(&our_tensor_b);
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
            .map(|i| (i as f32) * 0.1 + 1.0)
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
            .map(|i| (i as f32) * 0.2 + 0.5)
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

        let torch_result = match torch_tensor_a.mul_tensor(&torch_tensor_b) {
            Ok(result) => result,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch mul_tensor failed: {}", e))
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

    /// Comprehensive gradient validation test suite for multiplication operations
    pub fn test_mul_gradient_operations(&self) -> Vec<ComparisonResult> {
        let mut results = Vec::new();

        // Scalar gradient tests
        let scalar_grad_cases = vec![
            (vec![1], 0.0),
            (vec![1], 1.0),
            (vec![1], -1.0),
            (vec![3], 2.5),
            (vec![5], -3.7),
            (vec![2, 3], 1.5),
            (vec![4, 5], 0.5),
            (vec![3, 4, 5], 10.0),
            (vec![2, 3, 4, 5], -5.0),
        ];

        for (shape, scalar) in scalar_grad_cases {
            let result = self.test_mul_scalar_gradients(&shape, scalar);
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
            vec![3, 4, 5],
            vec![2, 3, 4, 5],
        ];

        for shape in tensor_grad_shapes {
            let result = self.test_mul_tensor_gradients(&shape);
            results.push(result);
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
            let result = self.test_mul_tensor_broadcasting_gradients(&shape1, &shape2);
            results.push(result);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_scalar_validation() {
        let validator = MultiplicationValidator::new(1e-6, 1e-8);
        let result = validator.test_mul_scalar(&[2, 3], 3.0);
        assert!(
            result.passed,
            "Scalar multiplication validation failed: {}",
            result.details
        );
    }

    #[test]
    fn test_mul_tensor_validation() {
        let validator = MultiplicationValidator::new(1e-6, 1e-8);
        let result = validator.test_mul_tensor(&[2, 3]);
        assert!(
            result.passed,
            "Tensor multiplication validation failed: {}",
            result.details
        );
    }

    #[test]
    fn test_mul_operations_suite() {
        let validator = MultiplicationValidator::new(1e-6, 1e-8);
        let results = validator.test_mul_operations();

        // Verify all tests passed
        for result in &results {
            assert!(
                result.passed,
                "Multiplication validation failed: {}",
                result.details
            );
        }

        println!(
            "All {} multiplication validation tests passed",
            results.len()
        );
    }

    #[test]
    fn test_comprehensive_mul_gradient_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let results = validator.test_mul_gradient_operations();

        let mut passed = 0;
        let mut failed = 0;

        for result in &results {
            if result.passed {
                passed += 1;
            } else {
                failed += 1;
                println!("FAILED mul gradient test: {}", result.details);
            }
        }

        println!(
            "Mul gradient validation: {} passed, {} failed",
            passed, failed
        );
        assert_eq!(failed, 0, "Some mul gradient tests failed");
    }

    #[test]
    fn test_mul_broadcasting_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Broadcasting patterns for multiplication
        let broadcasting_cases = vec![
            // Basic patterns
            (vec![1], vec![3]),
            (vec![3], vec![1]),
            (vec![1], vec![2, 3]),
            (vec![2, 3], vec![1]),
            // Neural network patterns (element-wise gating)
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
            let result = validator.test_mul_tensor_broadcasting(&shape1, &shape2);
            assert!(
                result.passed,
                "Broadcasting mul {:?} * {:?}: {}",
                shape1, shape2, result.details
            );
        }
    }

    #[test]
    fn test_mul_broadcasting_gradients_validation() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Key broadcasting patterns for gradient testing
        let grad_broadcasting_cases = vec![
            (vec![1], vec![3]),                // scalar to vector
            (vec![3], vec![2, 3]),             // element-wise scaling pattern
            (vec![1, 3], vec![2, 3]),          // row broadcast
            (vec![2, 1], vec![2, 3]),          // column broadcast
            (vec![128], vec![8, 128]),         // feature scaling pattern
            (vec![1, 1, 16], vec![8, 64, 16]), // 3D scaling pattern
        ];

        for (shape1, shape2) in grad_broadcasting_cases {
            let result = validator.test_mul_tensor_broadcasting_gradients(&shape1, &shape2);
            assert!(
                result.passed,
                "Broadcasting mul gradients {:?} * {:?}: {}",
                shape1, shape2, result.details
            );
        }
    }

    #[test]
    fn test_mul_gradient_edge_cases() {
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
            let result = validator.test_mul_scalar_gradients(&shape, scalar);
            assert!(
                result.passed,
                "Mul scalar gradient edge case {:?}, {}: {}",
                shape, scalar, result.details
            );
        }
    }

    #[test]
    fn test_mul_gradient_shapes_comprehensive() {
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
            let result = validator.test_mul_tensor_gradients(&shape);
            assert!(
                result.passed,
                "Mul tensor gradient shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_mul_neural_network_patterns() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Neural network specific patterns for multiplication (gating, scaling)
        let neural_patterns = vec![
            // Gating mechanisms
            (vec![128], vec![32, 128]), // gate values
            (vec![512], vec![64, 512]), // LSTM gates
            // Attention scaling
            (vec![1, 8, 64], vec![16, 1, 1]),
            (vec![8, 1, 1], vec![1, 64, 512]),
            // Feature scaling
            (vec![1, 64], vec![8, 64]),
            (vec![64, 1], vec![64, 28]),
            // Channel-wise scaling (CNN) - properly shaped
            (vec![1, 64, 1, 1], vec![8, 64, 28, 28]),
            (vec![1, 32, 1, 1], vec![4, 32, 14, 14]),
        ];

        for (shape1, shape2) in neural_patterns {
            // Test both forward and gradient computation
            let forward_result = validator.test_mul_tensor_broadcasting(&shape1, &shape2);
            assert!(
                forward_result.passed,
                "Neural mul forward {:?} * {:?}: {}",
                shape1, shape2, forward_result.details
            );

            let gradient_result =
                validator.test_mul_tensor_broadcasting_gradients(&shape1, &shape2);
            assert!(
                gradient_result.passed,
                "Neural mul gradient {:?} * {:?}: {}",
                shape1, shape2, gradient_result.details
            );
        }
    }

    #[test]
    fn test_mul_zero_gradient_handling() {
        let validator = TensorValidator::new(1e-6, 1e-8);

        // Test multiplication by zero and near-zero values
        let zero_cases = vec![
            (vec![2], 0.0),
            (vec![3, 3], 0.0),
            (vec![2, 3], 1e-15),
            (vec![4], -1e-15),
        ];

        for (shape, scalar) in zero_cases {
            let result = validator.test_mul_scalar_gradients(&shape, scalar);
            assert!(
                result.passed,
                "Mul zero gradient handling {:?}, {}: {}",
                shape, scalar, result.details
            );
        }
    }
}
