//! Square root operation validation

// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Validate sqrt against LibTorch
    pub fn test_sqrt(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.1 + 0.01;
            }
        }
        let our = x.sqrt();

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 0.01).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.sqrt() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };

        self.compare_tensors(&our, &torch)
    }

    /// Validate sqrt gradients: d/dx sqrt(x) = 0.5 / sqrt(x)
    pub fn test_sqrt_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.03 + 0.2;
            }
        }
        let mut y = x.sqrt();
        y.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.03 + 0.2).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.sqrt() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let grad_ones = match LibTorchTensor::ones(&torch_y.shape()) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_y.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(e);
        }
        let torch_grad = match torch_x.grad() {
            Some(g) => g,
            None => return ComparisonResult::failure("Torch grad missing".to_string()),
        };

        self.compare_tensors(&our_grad, &torch_grad)
    }

    /// Chain test: z = sqrt(x) * x + sqrt(x)
    pub fn test_sqrt_chain_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.01 + 0.5;
            }
        }
        let s = x.sqrt();
        let mut z = s.mul_tensor(&x).add_tensor(&s);
        z.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our chain grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 + 0.5).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_s = match torch_x.sqrt() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_z = match torch_s.mul_tensor(&torch_x) {
            Ok(t) => match t.add_tensor(&torch_s) {
                Ok(u) => u,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let grad_ones = match LibTorchTensor::ones(&torch_z.shape()) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_z.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(e);
        }
        let torch_grad = match torch_x.grad() {
            Some(g) => g,
            None => return ComparisonResult::failure("Torch chain grad missing".to_string()),
        };

        self.compare_tensors(&our_grad, &torch_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_validation_basic() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let result = validator.test_sqrt(&[3, 4]);
        assert!(result.passed, "sqrt validation failed: {}", result.details);
    }

    #[test]
    fn test_sqrt_validation_shapes_and_grads() {
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
            let r = validator.test_sqrt(&shape);
            assert!(r.passed, "sqrt op mismatch for {:?}: {}", shape, r.details);
            let g = validator.test_sqrt_gradients(&shape);
            assert!(
                g.passed,
                "sqrt grad mismatch for {:?}: {}",
                shape, g.details
            );
            let c = validator.test_sqrt_chain_gradients(&shape);
            assert!(
                c.passed,
                "sqrt chain grad mismatch for {:?}: {}",
                shape, c.details
            );
        }
    }
}
