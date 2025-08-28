// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Validate stack forward
    pub fn test_stack(&self, shape: &[usize], dim: usize, count: usize) -> ComparisonResult {
        assert!(count > 0);
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 0.5).collect();

        let ours: Vec<Tensor> = (0..count)
            .map(|k| {
                let mut t = Tensor::zeros(shape.to_vec());
                unsafe {
                    for (i, d) in data.iter().enumerate().take(size) {
                        *t.as_mut_ptr().add(i) = *d + (k as f32);
                    }
                }
                t
            })
            .collect();
        let our = Tensor::stack(&ours, dim);

        let torches: Vec<LibTorchTensor> = (0..count)
            .map(|k| {
                let shifted: Vec<f32> = data.iter().map(|v| v + k as f32).collect();
                LibTorchTensor::from_data(&shifted, shape).unwrap()
            })
            .collect();
        let torch = LibTorchTensor::stack(&torches, dim).unwrap();

        self.compare_tensors(&our, &torch)
    }

    /// Validate gradients through stack by comparing with LibTorch
    pub fn test_stack_gradients(
        &self,
        shape: &[usize],
        dim: usize,
        count: usize,
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let base: Vec<f32> = (0..size).map(|i| (i as f32) * 0.02 + 0.1).collect();

        let ours: Vec<Tensor> = (0..count)
            .map(|k| {
                let mut t = Tensor::zeros(shape.to_vec()).with_requires_grad();
                unsafe {
                    for (i, d) in base.iter().enumerate().take(size) {
                        *t.as_mut_ptr().add(i) = *d + (k as f32) * 0.5;
                    }
                }
                t
            })
            .collect();
        let mut y = Tensor::stack(&ours, dim);
        y.backward(None);
        let our_grads: Vec<Tensor> = ours
            .iter()
            .map(|t| t.grad_by_value().expect("Our grad missing"))
            .collect();

        let torches: Vec<LibTorchTensor> = (0..count)
            .map(|k| {
                let shifted: Vec<f32> = base.iter().map(|v| v + (k as f32) * 0.5).collect();
                LibTorchTensor::from_data(&shifted, shape)
                    .and_then(|t| t.requires_grad_(true))
                    .unwrap()
            })
            .collect();
        let torch_y = LibTorchTensor::stack(&torches, dim).unwrap();
        // For non-scalar outputs, supply an explicit ones gradient
        let grad_ones = match LibTorchTensor::ones(&torch_y.shape()) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_y.backward(Some(&grad_ones)) {
            return ComparisonResult::failure(e);
        }
        let torch_grads: Vec<LibTorchTensor> = torches
            .iter()
            .map(|t| t.grad().expect("Torch grad missing"))
            .collect();

        // Compare grads pairwise
        for (i, (og, tg)) in our_grads.iter().zip(torch_grads.iter()).enumerate() {
            let r = self.compare_tensors(og, tg);
            if !r.passed {
                return ComparisonResult::failure(format!(
                    "stack grad {} mismatch: {}",
                    i, r.details
                ));
            }
        }

        ComparisonResult::success()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_validation_basic() {
        let v = TensorValidator::new(1e-6, 1e-8);
        let r = v.test_stack(&[3], 0, 2);
        assert!(r.passed, "stack validation failed: {}", r.details);
    }

    #[test]
    fn test_stack_validation_suites() {
        let v = TensorValidator::default();
        let cases = vec![
            (vec![3usize], 0usize, 3usize),
            (vec![2, 3], 0, 2),
            (vec![2, 3], 1, 4),
            (vec![1, 2, 3], 2, 2),
        ];
        for (shape, dim, count) in cases {
            let r = v.test_stack(&shape, dim, count);
            assert!(
                r.passed,
                "stack mismatch for {:?} dim {} count {}: {}",
                shape, dim, count, r.details
            );
            let g = v.test_stack_gradients(&shape, dim, count);
            assert!(
                g.passed,
                "stack grad mismatch for {:?} dim {} count {}: {}",
                shape, dim, count, g.details
            );
        }
    }
}
