// Autograd now uses inherent Tensor methods; no trait import needed
use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Validate gather forward
    pub fn test_gather(
        &self,
        shape: &[usize],
        dim: usize,
        index_shape: &[usize],
        indices: &[usize],
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.1 - 0.2;
            }
        }
        let our = x.gather(dim, indices, index_shape);

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 0.2).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch = match torch_x.gather(dim, index_shape, indices) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };

        self.compare_tensors(&our, &torch)
    }

    /// Validate gradients for gather by comparing to LibTorch
    pub fn test_gather_gradients(
        &self,
        shape: &[usize],
        dim: usize,
        index_shape: &[usize],
        indices: &[usize],
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.03 + 0.15;
            }
        }
        let mut y = x.gather(dim, indices, index_shape);
        y.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.03 + 0.15).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let torch_y = match torch_x.gather(dim, index_shape, indices) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        // Provide explicit ones gradient when output not scalar
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
    fn test_gather_validation_basic() {
        let v = TensorValidator::new(1e-6, 1e-8);
        let shape = vec![2, 3];
        let dim = 1usize;
        let index_shape = vec![2, 2];
        let indices = vec![2usize, 0usize, 1usize, 1usize];
        let r = v.test_gather(&shape, dim, &index_shape, &indices);
        assert!(r.passed, "gather validation failed: {}", r.details);
    }

    #[test]
    fn test_gather_validation_suites() {
        let v = TensorValidator::default();
        let cases = vec![
            (
                vec![3usize],
                0usize,
                vec![3usize],
                vec![2usize, 1usize, 0usize],
            ),
            (vec![2, 3], 1, vec![2, 2], vec![2, 0, 1, 1]),
            (vec![2, 3], 0, vec![1, 3], vec![1, 1, 1]),
            (
                vec![1, 2, 3],
                2,
                vec![1, 2, 4],
                vec![0, 2, 1, 2, 0, 1, 2, 1],
            ),
        ];
        for (shape, dim, ishape, idx) in cases {
            let r = v.test_gather(&shape, dim, &ishape, &idx);
            assert!(
                r.passed,
                "gather mismatch for {:?} dim {} idx {:?}: {}",
                shape, dim, idx, r.details
            );
            let g = v.test_gather_gradients(&shape, dim, &ishape, &idx);
            assert!(
                g.passed,
                "gather grad mismatch for {:?} dim {} idx {:?}: {}",
                shape, dim, idx, g.details
            );
        }
    }
}
