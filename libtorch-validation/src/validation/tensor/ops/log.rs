//! Natural log operation validation methods
//!
//! Validates element-wise log against LibTorch, including autograd.

use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Test log operation correctness against LibTorch
    pub fn test_log(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();

        // Input must be strictly positive
        let mut our = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *our.as_mut_ptr().add(i) = (i as f32) * 0.1 + 0.5; // >= 0.5
            }
        }
        let our_result = our.log();

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let torch = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_result = match torch.log() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };

        self.compare_tensors(&our_result, &torch_result)
    }

    /// Validate gradients of log: d/dx log(x) = 1/x
    pub fn test_log_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();

        // Our autograd
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.05 + 0.2; // > 0
            }
        }
        let mut y = x.log();
        y.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our log produced no gradient".to_string()),
        };

        // LibTorch autograd
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.05 + 0.2).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.log() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let grad_ones = match LibTorchTensor::ones(&torch_y.shape()) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_y.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(format!("LibTorch backward failed: {}", e));
        }
        let torch_grad = match torch_x.grad() {
            Some(g) => g,
            None => return ComparisonResult::failure("LibTorch gradient missing".to_string()),
        };

        self.compare_tensors(&our_grad, &torch_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_validation_basic() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let result = validator.test_log(&[3, 4]);
        assert!(result.passed, "log validation failed: {}", result.details);
    }

    #[test]
    fn test_log_validation_shapes_and_grads() {
        let validator = TensorValidator::default();
        let shapes = vec![
            vec![1],
            vec![3],
            vec![2, 2],
            vec![1, 4],
            vec![2, 3],
            vec![1, 1, 3],
            vec![50, 50],
        ];
        for shape in shapes {
            let r = validator.test_log(&shape);
            assert!(r.passed, "log op mismatch for {:?}: {}", shape, r.details);

            let g = validator.test_log_gradients(&shape);
            assert!(g.passed, "log grad mismatch for {:?}: {}", shape, g.details);
        }
    }
}
