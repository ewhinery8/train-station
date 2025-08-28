use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    pub fn test_tanh_forward(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.2 - 1.0;
            }
        }
        let our = x.tanh();
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.tanh() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        self.compare_tensors(&our, &torch)
    }

    pub fn test_tanh_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.2 - 1.0;
            }
        }
        let y = x.tanh();
        let mut s = y.sum();
        s.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.tanh() {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_validation_suites() {
        let v = TensorValidator::default();
        let shapes = vec![vec![3], vec![2, 3], vec![2, 3, 4]];
        for shape in shapes {
            let r = v.test_tanh_forward(&shape);
            assert!(
                r.passed,
                "tanh forward mismatch for {:?}: {}",
                shape, r.details
            );
            let g = v.test_tanh_gradients(&shape);
            assert!(
                g.passed,
                "tanh grad mismatch for {:?}: {}",
                shape, g.details
            );
        }
    }
}
