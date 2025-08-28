use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    pub fn test_max_forward(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.1 - 3.0;
            }
        }
        let our = x.max();
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.max() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        self.compare_tensors(&our, &torch)
    }

    pub fn test_max_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = ((i % 11) as f32) * 0.1 - 0.5;
            }
        }
        let y = x.mul_scalar(1.7).add_scalar(0.3);
        let mut m = y.max();
        m.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();
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
        let torch_y = match torch_y.add_scalar(0.3) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_m = match torch_y.max() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_m.backward(None) {
            return ComparisonResult::failure(e);
        }
        let torch_grad = match torch_x.grad() {
            Some(g) => g,
            None => return ComparisonResult::failure("Torch grad missing".to_string()),
        };
        self.compare_tensors(&our_grad, &torch_grad)
    }

    pub fn test_max_dims_forward(
        &self,
        shape: &[usize],
        dims: &[usize],
        keepdim: bool,
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = ((i % 7) as f32) - 3.0;
            }
        }
        let our = x.max_dims(dims, keepdim);
        let data: Vec<f32> = (0..size).map(|i| ((i % 7) as f32) - 3.0).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.max_dims(dims, keepdim) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        self.compare_tensors(&our, &torch)
    }

    pub fn test_max_dims_gradients(
        &self,
        shape: &[usize],
        dims: &[usize],
        keepdim: bool,
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = ((i % 5) as f32) - 2.0;
            }
        }
        let mut m = x.max_dims(dims, keepdim);
        m.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| ((i % 5) as f32) - 2.0).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_m = match torch_x.max_dims(dims, keepdim) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let grad_ones = match LibTorchTensor::ones(&torch_m.shape()) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_m.backward(Some(&grad_ones)) {
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
    fn test_max_validation() {
        let v = TensorValidator::default();
        let r = v.test_max_forward(&[2, 3]);
        assert!(r.passed, "max forward failed: {}", r.details);
        let g = v.test_max_gradients(&[2, 3]);
        assert!(g.passed, "max gradients failed: {}", g.details);
    }

    #[test]
    fn test_max_dims_validation_suites() {
        let v = TensorValidator::default();
        let cases = vec![
            (vec![2, 3], vec![1usize], true),
            (vec![2, 3], vec![1usize], false),
            (vec![2, 3, 4], vec![0usize, 2usize], true),
        ];
        for (shape, dims, keepdim) in cases {
            let r = v.test_max_dims_forward(&shape, &dims, keepdim);
            assert!(
                r.passed,
                "max_dims forward mismatch for {:?} dims {:?} keepdim {}: {}",
                shape, dims, keepdim, r.details
            );
            let g = v.test_max_dims_gradients(&shape, &dims, keepdim);
            assert!(
                g.passed,
                "max_dims grad mismatch for {:?} dims {:?} keepdim {}: {}",
                shape, dims, keepdim, g.details
            );
        }
    }
}
