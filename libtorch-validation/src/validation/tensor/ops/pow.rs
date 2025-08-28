use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    pub fn test_pow_scalar_forward(&self, shape: &[usize], exponent: f32) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.1 + 0.1;
            }
        }
        let our = x.pow_scalar(exponent);
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.pow_scalar(exponent) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        self.compare_tensors(&our, &torch)
    }

    pub fn test_pow_scalar_gradients(&self, shape: &[usize], exponent: f32) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.05 + 0.2;
            }
        }
        let y = x.pow_scalar(exponent);
        let mut s = y.sum();
        s.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.05 + 0.2).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.pow_scalar(exponent) {
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

    pub fn test_pow_tensor_forward(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        let mut a = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.1 + 0.5;
                *a.as_mut_ptr().add(i) = 1.0 + ((i % 3) as f32) * 0.5;
            }
        }
        let our = x.pow_tensor(&a);
        let data_x: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let data_a: Vec<f32> = (0..size).map(|i| 1.0 + ((i % 3) as f32) * 0.5).collect();
        let torch_x = match LibTorchTensor::from_data(&data_x, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_a = match LibTorchTensor::from_data(&data_a, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.pow_tensor(&torch_a) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        self.compare_tensors(&our, &torch)
    }

    pub fn test_pow_tensor_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        let mut a = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.1 + 0.5;
                *a.as_mut_ptr().add(i) = 0.5 + ((i % 3) as f32) * 0.25;
            }
        }
        let y = x.pow_tensor(&a);
        let mut s = y.sum();
        s.backward(None);
        let our_gx = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad X missing".to_string()),
        };
        let our_ga = match a.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad A missing".to_string()),
        };

        let data_x: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let data_a: Vec<f32> = (0..size).map(|i| 0.5 + ((i % 3) as f32) * 0.25).collect();
        let torch_x = match LibTorchTensor::from_data(&data_x, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_a = match LibTorchTensor::from_data(&data_a, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.pow_tensor(&torch_a) {
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
        let torch_gx = match torch_x.grad() {
            Some(g) => g,
            None => return ComparisonResult::failure("Torch grad X missing".to_string()),
        };
        let torch_ga = match torch_a.grad() {
            Some(g) => g,
            None => return ComparisonResult::failure("Torch grad A missing".to_string()),
        };
        let rx = self.compare_tensors(&our_gx, &torch_gx);
        if !rx.passed {
            return rx;
        }
        self.compare_tensors(&our_ga, &torch_ga)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_scalar_validation() {
        let v = TensorValidator::default();
        for &e in &[0.5f32, 1.0, 2.0, 3.0] {
            let r = v.test_pow_scalar_forward(&[2, 3], e);
            assert!(
                r.passed,
                "pow scalar forward failed (e={}): {}",
                e, r.details
            );
            let g = v.test_pow_scalar_gradients(&[2, 3], e);
            assert!(g.passed, "pow scalar grad failed (e={}): {}", e, g.details);
        }
    }

    #[test]
    fn test_pow_tensor_validation() {
        let v = TensorValidator::default();
        let r = v.test_pow_tensor_forward(&[2, 3]);
        assert!(r.passed, "pow tensor forward failed: {}", r.details);
        let g = v.test_pow_tensor_gradients(&[2, 3]);
        assert!(g.passed, "pow tensor grad failed: {}", g.details);
    }
}
