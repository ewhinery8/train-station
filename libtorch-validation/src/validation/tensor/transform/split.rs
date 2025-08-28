use crate::ffi::LibTorchTensor;
use crate::validation::core::{ComparisonResult, TensorValidator};
use train_station::Tensor;

impl TensorValidator {
    /// Validate split_with_sizes forward against LibTorch by reconstructing parts with index_select
    pub fn test_split_with_sizes(
        &self,
        shape: &[usize],
        split_sizes: &[usize],
        dim: usize,
    ) -> ComparisonResult {
        // Prepare our tensor
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec());
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.01 - 0.2;
            }
        }
        let our_parts = x.split_with_sizes(split_sizes, dim);

        // Torch tensor
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };

        // For each part, build indices range and compare
        let mut start = 0usize;
        for (i, &len) in split_sizes.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let idx: Vec<usize> = (start..start + len).collect();
            let torch_part = match torch_x.index_select(dim, &idx) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            };
            let r = self.compare_tensors(&our_parts[i], &torch_part);
            if !r.passed {
                return r;
            }
            start += len;
        }
        ComparisonResult::success()
    }

    /// Validate gradients for split by summing all parts and comparing input grads
    pub fn test_split_with_sizes_gradients(
        &self,
        shape: &[usize],
        split_sizes: &[usize],
        dim: usize,
    ) -> ComparisonResult {
        let size: usize = shape.iter().product();
        let mut x = Tensor::zeros(shape.to_vec()).with_requires_grad();
        unsafe {
            for i in 0..size {
                *x.as_mut_ptr().add(i) = (i as f32) * 0.02 + 0.3;
            }
        }
        let parts = x.split_with_sizes(split_sizes, dim);
        // Reconstruct full by concatenation to avoid shape/layout issues
        let mut full = Tensor::cat(&parts, dim);
        full.backward(None);
        let our_grad = match x.grad_by_value() {
            Some(g) => g,
            None => return ComparisonResult::failure("Our grad missing".to_string()),
        };

        // Torch reference
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.02 + 0.3).collect();
        let torch_x = match LibTorchTensor::from_data(&data, shape) {
            Ok(t) => match t.requires_grad_(true) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(e),
            },
            Err(e) => return ComparisonResult::failure(e),
        };
        let mut start = 0usize;
        let mut torch_parts: Vec<LibTorchTensor> = Vec::new();
        for &len in split_sizes {
            if len == 0 {
                continue;
            }
            let idx: Vec<usize> = (start..start + len).collect();
            match torch_x.index_select(dim, &idx) {
                Ok(p) => torch_parts.push(p),
                Err(e) => return ComparisonResult::failure(e),
            };
            start += len;
        }
        // Concatenate torch parts to reconstruct full tensor
        let torch_full = match LibTorchTensor::cat(&torch_parts, dim) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        // Upstream ones matching full
        let grad_ones = match LibTorchTensor::ones(&torch_full.shape()) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(e),
        };
        if let Err(e) = torch_full.backward(Some(&grad_ones)) {
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
    fn test_split_with_sizes_validation() {
        let v = TensorValidator::new(1e-6, 1e-8);
        let r = v.test_split_with_sizes(&[2, 7], &[2, 3, 2], 1);
        assert!(
            r.passed,
            "split_with_sizes validation failed: {}",
            r.details
        );
    }

    #[test]
    fn test_split_with_sizes_validation_suites() {
        let v = TensorValidator::default();
        let cases = vec![
            (vec![6usize], vec![2usize, 2usize, 2usize], 0usize),
            (vec![2, 5], vec![1, 4], 1),
            (vec![3, 4, 5], vec![2, 2, 1], 2),
        ];
        for (shape, sizes, dim) in cases {
            let r = v.test_split_with_sizes(&shape, &sizes, dim);
            assert!(
                r.passed,
                "split_with_sizes mismatch for {:?} sizes {:?} dim {}: {}",
                shape, sizes, dim, r.details
            );
            let g = v.test_split_with_sizes_gradients(&shape, &sizes, dim);
            assert!(
                g.passed,
                "split_with_sizes grad mismatch for {:?} sizes {:?} dim {}: {}",
                shape, sizes, dim, g.details
            );
        }
    }
}
