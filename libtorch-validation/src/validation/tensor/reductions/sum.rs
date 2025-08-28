use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Validate sum forward vs LibTorch
    pub fn test_sum_forward(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.01 - 0.2;
            }
        }
        let our = x.sum();

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.sum() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };

        self.compare_tensors(&our, &torch)
    }

    /// Validate sum gradients vs LibTorch using a simple chain (mul + add + sum)
    pub fn test_sum_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.02 + 0.3;
            }
        }
        let y = x.mul_scalar(1.7).add_scalar(-0.5);
        let mut s = y.sum();
        s.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.02 + 0.3).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.mul_scalar(1.7) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_y.add_scalar(-0.5) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_s = match torch_y.sum() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_s.backward(None) {
            return ComparisonResult::failure(e);
        }
        let torch_grad = match torch_x.grad() {
            Some(g) => g,
            None => return ComparisonResult::failure("Torch grad missing".to_string()),
        };

        self.compare_tensors(&our_grad, &torch_grad)
    }

    /// Validate dim-reduction sum forward
    pub fn test_sum_dims_forward(
        &self,
        shape: &[usize],
        dims: &[usize],
        keepdim: bool,
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.01 - 0.2;
            }
        }
        let our = x.sum_dims(dims, keepdim);

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.sum_dims(dims, keepdim) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        self.compare_tensors(&our, &torch)
    }

    /// Validate dim-reduction sum gradients
    pub fn test_sum_dims_gradients(
        &self,
        shape: &[usize],
        dims: &[usize],
        keepdim: bool,
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.02 + 0.3;
            }
        }
        let mut s = x.sum_dims(dims, keepdim);
        s.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.02 + 0.3).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_s = match torch_x.sum_dims(dims, keepdim) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let grad_ones = match LibTorchTensor::ones(&torch_s.shape()) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_s.backward(Some(&grad_ones)) {
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
    fn test_sum_validation_forward() {
        let v = TensorValidator::new(1e-6, 1e-8);
        let r = v.test_sum_forward(&[2, 3, 4]);
        assert!(r.passed, "sum forward validation failed: {}", r.details);
    }

    #[test]
    fn test_sum_validation_gradients_suites() {
        let v = TensorValidator::default();
        let shapes = vec![vec![6usize], vec![2, 3], vec![2, 2, 2]];
        for shape in shapes {
            let r = v.test_sum_forward(&shape);
            assert!(
                r.passed,
                "sum forward mismatch for {:?}: {}",
                shape, r.details
            );
            let g = v.test_sum_gradients(&shape);
            assert!(g.passed, "sum grad mismatch for {:?}: {}", shape, g.details);
        }
    }

    #[test]
    fn test_sum_dims_validation_suites() {
        let v = TensorValidator::default();
        let cases = vec![
            (vec![2, 3], vec![1usize], true),
            (vec![2, 3], vec![1usize], false),
            (vec![2, 3, 4], vec![0usize, 2usize], true),
            (vec![2, 3, 4], vec![0usize], false),
        ];
        for (shape, dims, keepdim) in cases {
            let r = v.test_sum_dims_forward(&shape, &dims, keepdim);
            assert!(
                r.passed,
                "sum_dims forward mismatch for {:?} dims {:?} keepdim {}: {}",
                shape, dims, keepdim, r.details
            );
            let g = v.test_sum_dims_gradients(&shape, &dims, keepdim);
            assert!(
                g.passed,
                "sum_dims grad mismatch for {:?} dims {:?} keepdim {}: {}",
                shape, dims, keepdim, g.details
            );
        }
    }
}
