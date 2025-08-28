// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    pub fn test_masked_fill(&self, shape: &[usize], mask: &[bool], value: f32) -> ComparisonResult {
        let size: usize = shape.iter().product();
        assert_eq!(mask.len(), size, "mask length must equal numel");
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.05 - 0.4;
            }
        }
        let our = x.masked_fill(mask, value);

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.05 - 0.4).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.masked_fill(mask, value) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        self.compare_tensors(&our, &torch)
    }

    pub fn test_masked_fill_gradients(
        &self,
        shape: &[usize],
        mask: &[bool],
        value: f32,
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        assert_eq!(mask.len(), size, "mask length must equal numel");
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.02 + 0.1;
            }
        }
        let mut y = x.masked_fill(mask, value);
        y.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.02 + 0.1).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.masked_fill(mask, value) {
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
    fn test_masked_fill_validation_basic() {
        let v = TensorValidator::new(1e-6, 1e-8);
        let shape = vec![2, 3];
        let mask = vec![false, true, false, true, false, true];
        let r = v.test_masked_fill(&shape, &mask, -2.0);
        assert!(r.passed, "masked_fill validation failed: {}", r.details);
    }

    #[test]
    fn test_masked_fill_validation_suites() {
        let v = TensorValidator::default();
        let cases = vec![
            (vec![3usize], vec![true, false, true], 1.23f32),
            (
                vec![2, 3],
                vec![false, true, false, true, false, false],
                -1.0f32,
            ),
            (vec![1, 2, 3], vec![false; 6], 0.0f32),
        ];
        for (shape, mask, val) in cases {
            let r = v.test_masked_fill(&shape, &mask, val);
            assert!(
                r.passed,
                "masked_fill forward mismatch for {:?}: {}",
                shape, r.details
            );
            let g = v.test_masked_fill_gradients(&shape, &mask, val);
            assert!(
                g.passed,
                "masked_fill grad mismatch for {:?}: {}",
                shape, g.details
            );
        }
    }
}
