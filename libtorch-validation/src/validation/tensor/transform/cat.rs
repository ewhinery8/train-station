//! Concatenate transform validation methods
//!
//! Validates Tensor::cat against LibTorch including gradient checks and chain tests.

// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Validate cat forward against LibTorch
    pub fn test_cat(&self, shapes: &[Vec<usize>], dim: usize) -> ComparisonResult {
        assert!(!shapes.is_empty(), "need at least one shape");
        let tensors: Vec<Tensor> = shapes
            .iter()
            .map(|s| {
                let size: usize = s.iter().product();
                let mut t = Tensor::zeros(s.clone());
                unsafe {
                    for i in 0..size {
                        *t.as_mut_ptr().add(i) = (i as f32) * 0.01 + 0.1;
                    }
                }
                t
            })
            .collect();
        let our = Tensor::cat(&tensors, dim);

        let torch_tensors: Vec<LibTorchTensor> = shapes
            .iter()
            .map(|s| {
                let size: usize = s.iter().product();
                let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 + 0.1).collect();
                LibTorchTensor::from_data(&data, s).expect("ffi make")
            })
            .collect();
        let torch = match LibTorchTensor::cat(&torch_tensors, dim) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch cat failed: {}", e)),
        };

        self.compare_tensors(&our, &torch)
    }

    /// Validate cat gradients by comparing to LibTorch autograd
    pub fn test_cat_gradients(&self, shapes: &[Vec<usize>], dim: usize) -> ComparisonResult {
        let mut our_inputs: Vec<Tensor> = Vec::new();
        let mut torch_inputs: Vec<LibTorchTensor> = Vec::new();

        for s in shapes.iter() {
            let size: usize = s.iter().product();
            let mut t = Tensor::zeros(s.clone()).with_requires_grad();
            let mut data: Vec<f32> = vec![0.0; size];
            for (i, d) in data.iter_mut().enumerate().take(size) {
                *d = (i as f32) * 0.02 - 0.3;
            }
            unsafe {
                for (i, d) in data.iter().enumerate().take(size) {
                    *t.as_mut_ptr().add(i) = *d;
                }
            }
            our_inputs.push(t);

            let torch_t = match LibTorchTensor::from_data(&data, s) {
                Ok(t) => match t.requires_grad_(true) {
                    Ok(t) => t,
                    Err(e) => return ComparisonResult::failure(e),
                },
                Err(e) => return ComparisonResult::failure(format!("ffi tensor failed: {}", e)),
            };
            torch_inputs.push(torch_t);
        }

        let mut our_out = Tensor::cat(&our_inputs, dim);
        our_out.backward(None);

        let torch_out = match LibTorchTensor::cat(&torch_inputs, dim) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("LibTorch cat failed: {}", e)),
        };
        let grad_ones = match LibTorchTensor::ones(&torch_out.shape()) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_out.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(format!("backward failed: {}", e));
        }

        // Compare all input grads
        for (i, s) in shapes.iter().enumerate() {
            let our_g = match our_inputs[i].grad_by_value() {
                Some(g) => g,
                None => {
                    return ComparisonResult::failure(format!("our grad missing for input {}", i))
                }
            };
            let torch_g = match torch_inputs[i].grad() {
                Some(g) => g,
                None => {
                    return ComparisonResult::failure(format!("torch grad missing for input {}", i))
                }
            };
            let cmp = self.compare_tensors(&our_g, &torch_g);
            if !cmp.passed {
                return ComparisonResult::failure(format!(
                    "grad mismatch input {}: {}",
                    i, cmp.details
                ));
            }
            if our_g.shape().dims != *s {
                return ComparisonResult::failure(format!("grad shape mismatch input {}", i));
            }
        }
        ComparisonResult::success()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cat_validation_basic() {
        let validator = TensorValidator::new(1e-6, 1e-8);
        let result = validator.test_cat(&[vec![2, 2], vec![2, 1]], 1);
        assert!(result.passed, "cat validation failed: {}", result.details);
    }

    #[test]
    fn test_cat_validation_suites() {
        let validator = TensorValidator::default();
        let cases: Vec<(Vec<Vec<usize>>, usize)> = vec![
            (vec![vec![3], vec![2]], 0),
            (vec![vec![2, 2], vec![2, 3]], 1),
            (vec![vec![2, 1, 3], vec![2, 2, 3], vec![2, 4, 3]], 1),
        ];
        for (shapes, dim) in cases {
            let r = validator.test_cat(&shapes, dim);
            assert!(
                r.passed,
                "cat mismatch for {:?} along {}: {}",
                shapes, dim, r.details
            );
            let g = validator.test_cat_gradients(&shapes, dim);
            assert!(
                g.passed,
                "cat grad mismatch for {:?} along {}: {}",
                shapes, dim, g.details
            );
        }
    }
}
