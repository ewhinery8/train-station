//! Permute transform validation

// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Validate permute forward
    pub fn test_permute(&self, shape: &[usize], dims: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.1 + 1.0;
            }
        }
        let our = x.permute(dims.to_vec());

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.permute(dims) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };

        self.compare_tensors(&our, &torch)
    }

    /// Validate gradients through permute by comparing to LibTorch
    pub fn test_permute_gradients(&self, shape: &[usize], dims: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.02 - 0.3;
            }
        }
        let mut y = x.permute(dims.to_vec());
        y.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.permute(dims) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permute_validation_basic() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let result = validator.test_permute(&[2, 3], &[1, 0]);
        assert!(
            result.passed,
            "permute validation failed: {}",
            result.details
        );
    }

    #[test]
    fn test_permute_validation_suites() {
        let validator = TensorValidator::default();
        let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
            (vec![2, 3], vec![1, 0]),
            (vec![2, 3, 4], vec![2, 0, 1]),
            (vec![1, 2, 3, 4], vec![3, 1, 2, 0]),
        ];
        for (shape, dims) in cases {
            let r = validator.test_permute(&shape, &dims);
            assert!(
                r.passed,
                "permute mismatch for {:?} -> {:?}: {}",
                shape, dims, r.details
            );
            let g = validator.test_permute_gradients(&shape, &dims);
            assert!(
                g.passed,
                "permute grad mismatch for {:?} -> {:?}: {}",
                shape, dims, g.details
            );
        }
    }
}
