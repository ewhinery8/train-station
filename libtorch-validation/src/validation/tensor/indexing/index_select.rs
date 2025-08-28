// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Validate index_select forward
    pub fn test_index_select(
        &self,
        shape: &[usize],
        dim: usize,
        indices: &[usize],
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.1 + 0.3;
            }
        }
        let our = x.index_select(dim, indices);

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 0.3).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.index_select(dim, indices) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };

        self.compare_tensors(&our, &torch)
    }

    /// Validate gradients for index_select by comparing to LibTorch
    pub fn test_index_select_gradients(
        &self,
        shape: &[usize],
        dim: usize,
        indices: &[usize],
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.02 - 0.25;
            }
        }
        let mut y = x.index_select(dim, indices);
        // Upstream gradient ones, same as torch default for ones passed explicitly
        y.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.02 - 0.25).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.index_select(dim, indices) {
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
    fn test_index_select_validation_basic() {
        let v = TensorValidator::new(1e-6, 1e-8);
        let r = v.test_index_select(&[2, 3], 1, &[2, 0]);
        assert!(r.passed, "index_select validation failed: {}", r.details);
    }

    #[test]
    fn test_index_select_validation_suites() {
        let v = TensorValidator::default();
        let cases = vec![
            (vec![3usize], 0usize, vec![2usize, 1usize, 0usize]),
            (vec![2, 3], 0, vec![1, 1, 0]),
            (vec![2, 3], 1, vec![2, 0]),
            (vec![1, 2, 3], 2, vec![0, 2, 1, 2]),
        ];
        for (shape, dim, idx) in cases {
            let r = v.test_index_select(&shape, dim, &idx);
            assert!(
                r.passed,
                "index_select mismatch for {:?} dim {} idx {:?}: {}",
                shape, dim, idx, r.details
            );
            let g = v.test_index_select_gradients(&shape, dim, &idx);
            assert!(
                g.passed,
                "index_select grad mismatch for {:?} dim {} idx {:?}: {}",
                shape, dim, idx, g.details
            );
        }
    }
}
