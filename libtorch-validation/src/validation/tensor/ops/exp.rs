//! Exponential operation validation methods
//!
//! Provides validation for element-wise exp against LibTorch reference implementation,
//! including gradient checks and multi-op chains.

// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Test exp operation correctness against LibTorch
    pub fn test_exp(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();

        // Our implementation
        let mut our = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *our.as_mut_ptr().add(i) = ((i as f32) * 0.1) - 2.5;
            }
        }
        let our_result = our.exp();

        // LibTorch reference
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 2.5).collect();
        let torch = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch tensor creation failed: {}", e))
            }
        };
        let torch_result = match torch.exp() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch exp failed: {}", e)),
        };

        self.compare_tensors(&our_result, &torch_result)
    }

    /// Validate gradients of exp: d/dx exp(x) = exp(x)
    pub fn test_exp_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();

        // Our autograd
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = ((i as f32) * 0.05) - 1.0;
            }
        }
        let mut y = x.exp();
        y.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our exp produced no gradient".to_string()),
        };

        // LibTorch autograd
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch tensor creation failed: {}", e))
            }
        };
        let torch_y = match torch_x.exp() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch exp failed: {}", e)),
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

    /// Multi-op gradient chain validation involving exp
    /// Example: z = exp(x) * x + exp(x); dz/dx = exp(x)*x' + exp(x)' + exp(x) = exp(x)*(x + 2)
    pub fn test_exp_chain_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();

        // Our chain: z = exp(x) * x + exp(x)
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.01 - 0.5;
            }
        }
        let expx = x.exp();
        let mut z = expx.mul_tensor(&x).add_tensor(&expx);
        z.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our chain produced no gradient".to_string()),
        };

        // Torch chain
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => {
                return ComparisonResult::failure(format!("LibTorch tensor creation failed: {}", e))
            }
        };
        let torch_expx = match torch_x.exp() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch exp failed: {}", e)),
        };
        let torch_z = match torch_expx.mul_tensor(&torch_x) {
            Ok(t) => match t.add_tensor(&torch_expx) {
                Ok(s) => s,
                Err(e) => return ComparisonResult::failure(format!("LibTorch add failed: {}", e)),
            },
            Err(e) => return ComparisonResult::failure(format!("LibTorch mul failed: {}", e)),
        };
        let grad_ones = match LibTorchTensor::ones(&torch_z.shape()) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_z.backward(Some(&grad_ones)) {
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
    fn test_exp_validation_basic() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let result = validator.test_exp(&[3, 4]);
        assert!(result.passed, "exp validation failed: {}", result.details);
    }

    #[test]
    fn test_exp_validation_shapes_and_grads() {
        let validator = TensorValidator::default();
        let shapes = vec![
            vec![1],
            vec![3],
            vec![2, 2],
            vec![1, 4],
            vec![2, 3],
            vec![1, 1, 3],
            vec![100, 100],
        ];
        for shape in shapes {
            let r = validator.test_exp(&shape);
            assert!(r.passed, "exp op mismatch for {:?}: {}", shape, r.details);

            let g = validator.test_exp_gradients(&shape);
            assert!(g.passed, "exp grad mismatch for {:?}: {}", shape, g.details);

            let c = validator.test_exp_chain_gradients(&shape);
            assert!(
                c.passed,
                "exp chain grad mismatch for {:?}: {}",
                shape, c.details
            );
        }
    }
}
