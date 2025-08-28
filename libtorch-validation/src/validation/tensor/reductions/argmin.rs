use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    pub fn test_argmin_forward(&self, shape: &[usize]) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = ((i as i32 % 11) - 5) as f32;
            }
        }
        let our = x.argmin();
        let data: Vec<f32> = (0..size).map(|i| ((i as i32 % 11) - 5) as f32).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.argmin() {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        self.compare_tensors(&our, &torch)
    }

    pub fn test_argmin_dim_forward(
        &self,
        shape: &[usize],
        dim: usize,
        keepdim: bool,
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (((i * 7) % 13) as i32 - 6) as f32;
            }
        }
        let our = x.argmin_dim(dim, keepdim);
        let data: Vec<f32> = (0..size)
            .map(|i| (((i * 7) % 13) as i32 - 6) as f32)
            .collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.argmin_dim(dim, keepdim) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        self.compare_tensors(&our, &torch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmin_validation() {
        let v = TensorValidator::default();
        let r = v.test_argmin_forward(&[2, 3]);
        assert!(r.passed, "argmin forward failed: {}", r.details);
    }

    #[test]
    fn test_argmin_dim_validation_suites() {
        let v = TensorValidator::default();
        let cases = vec![
            (vec![2, 3], 1usize, true),
            (vec![2, 3], 1usize, false),
            (vec![2, 3, 4], 2usize, true),
        ];
        for (shape, dim, keepdim) in cases {
            let r = v.test_argmin_dim_forward(&shape, dim, keepdim);
            assert!(
                r.passed,
                "argmin_dim mismatch for {:?} dim {} keepdim {}: {}",
                shape, dim, keepdim, r.details
            );
        }
    }
}
