//! View transform validation
//!
//! Validates `Tensor::view` behavior against LibTorch expectations (requires contiguous).

// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Validate that view matches reshape semantics for contiguous tensors
    pub fn test_view(&self, original_shape: &[usize], new_shape: &[i32]) -> ComparisonResult {
        // Our side
        let size: usize = original_shape.iter().product();
        let mut x = Tensor::zeros(original_shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.1 + 1.0;
            }
        }
        let y = x.view(new_shape.to_vec());

        // Torch: use real view via FFI wrapper (ensuring contiguity internally)
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let torch_x = match LibTorchTensor::from_data(&data, original_shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.view(&y.shape().dims) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };

        self.compare_tensors(&y, &torch_y)
    }

    /// Validate gradients flow through view (contiguous case)
    pub fn test_view_gradients(
        &self,
        original_shape: &[usize],
        new_shape: &[i32],
    ) -> ComparisonResult {
        let size: usize = original_shape.iter().product();

        // Our autograd
        let mut x = Tensor::zeros(original_shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.05 - 1.0;
            }
        }
        let mut y = x.view(new_shape.to_vec());
        y.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Missing our grad".to_string()),
        };

        // Torch autograd: use real view via FFI wrapper
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let torch_x = match LibTorchTensor::from_data(&data, original_shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.view(&y.shape().dims) {
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
            None => return ComparisonResult::failure("Missing torch grad".to_string()),
        };

        self.compare_tensors(&our_grad, &torch_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_validation_basic() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let result = validator.test_view(&[2, 3], &[3, 2]);
        assert!(result.passed, "view validation failed: {}", result.details);
    }

    #[test]
    fn test_view_validation_suites() {
        let validator = TensorValidator::default();
        let cases: Vec<(Vec<usize>, Vec<i32>)> = vec![
            (vec![3], vec![1, 3]),
            (vec![2, 2], vec![4]),
            (vec![2, 3, 4], vec![4, 3, 2]),
            (vec![2, 3, 4], vec![6, -1]),
        ];
        for (orig, newv) in cases {
            let r = validator.test_view(&orig, &newv);
            assert!(
                r.passed,
                "view mismatch for {:?} -> {:?}: {}",
                orig, newv, r.details
            );
            let g = validator.test_view_gradients(&orig, &newv);
            assert!(
                g.passed,
                "view grad mismatch for {:?} -> {:?}: {}",
                orig, newv, g.details
            );
        }
    }
}
