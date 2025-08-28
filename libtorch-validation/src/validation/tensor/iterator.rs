//! Comprehensive validation tests for tensor element iterator functionality
//!
//! This module provides validation of the tensor element iterator against LibTorch
//! to ensure correctness of both forward operations and gradient computations.

#[cfg(test)]
mod tests {
    use crate::ffi::LibTorchTensor;
    use crate::validation::core::TensorValidator;
    use train_station::Tensor;

    /// Test element iterator basic functionality
    #[test]
    fn test_element_iterator_basic_validation() {
        // Create test tensor
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

        // Test basic iteration and collection
        let collected: Tensor = tensor.iter().collect();

        // Should be identical to original
        assert_eq!(tensor.data(), collected.data());
        assert_eq!(tensor.shape().dims, collected.shape().dims);

        // Test element access
        let elements: Vec<f32> = tensor.iter().map(|elem| elem.value()).collect();

        assert_eq!(elements, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Test iterator map operation
    #[test]
    fn test_iterator_map_operation() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();

        let mapped: Tensor = tensor.iter().map(|elem| elem.mul_scalar(2.0)).collect();

        let expected = vec![2.0, 4.0, 6.0];
        assert_eq!(mapped.data(), &expected);
    }

    /// Test iterator filter operation
    #[test]
    fn test_iterator_filter_operation() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();

        let filtered: Tensor = tensor.iter().filter(|elem| elem.value() > 2.5).collect();

        let expected = vec![3.0, 4.0, 5.0];
        assert_eq!(filtered.data(), &expected);
    }

    /// Test iterator with gradient tracking
    #[test]
    fn test_iterator_gradient_tracking() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3])
            .unwrap()
            .with_requires_grad();

        let result: Tensor = tensor.iter().map(|elem| elem.mul_scalar(2.0)).collect();

        let mut loss = result.sum();
        loss.backward(None);

        let grad = tensor.grad_by_value().expect("Gradient should exist");
        let expected_grad = vec![2.0, 2.0, 2.0]; // d/dx(2x) = 2 for each element
        assert_eq!(grad.data(), &expected_grad);
    }

    /// Test iterator actual backward pass with accumulation
    #[test]
    fn test_iterator_actual_backward_pass() {
        let source = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3])
            .unwrap()
            .with_requires_grad();

        // Use iterator collect directly (this preserves gradient graph)
        let result: Tensor = source
            .iter()
            .enumerate()
            .map(|(i, elem)| elem.mul_scalar((i + 1) as f32))
            .collect();

        // Forward pass check
        assert_eq!(result.data(), &[1.0, 4.0, 9.0]); // [1*1, 2*2, 3*3]

        // Backward pass
        let mut loss = result.sum(); // sum = 14.0
        loss.backward(None);

        // Check gradients accumulated properly
        if let Some(grad) = source.grad_by_value() {
            assert_eq!(grad.data(), &[1.0, 2.0, 3.0]); // Each gradient = index
        } else {
            panic!("Expected gradient on source tensor");
        }
    }

    /// Test iterator vs direct operations for gradient equivalence
    #[test]
    fn test_iterator_vs_direct_gradients() {
        // Iterator path
        let iter_source = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3])
            .unwrap()
            .with_requires_grad();
        let iter_result: Tensor = iter_source
            .iter()
            .map(|elem| elem.mul_scalar(3.0))
            .collect();
        let mut iter_loss = iter_result.sum();
        iter_loss.backward(None);
        let iter_grad = iter_source
            .grad_by_value()
            .expect("Iterator gradient should exist");

        // Direct path
        let direct_source = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3])
            .unwrap()
            .with_requires_grad();
        let direct_result = direct_source.mul_scalar(3.0);
        let mut direct_loss = direct_result.sum();
        direct_loss.backward(None);
        let direct_grad = direct_source
            .grad_by_value()
            .expect("Direct gradient should exist");

        // Gradients should be identical
        assert_eq!(iter_grad.data(), direct_grad.data());
        assert_eq!(iter_grad.data(), &[3.0, 3.0, 3.0]);
    }

    /// Validate basic iterator element operations against LibTorch
    #[test]
    fn test_iterator_vs_libtorch_element_operations() {
        let validator = TensorValidator::new(1e-6, 1e-9);

        // Test data
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];

        // Our implementation
        let our_tensor = Tensor::from_slice(&data, shape.clone()).unwrap();
        let our_result: Tensor = our_tensor.iter().map(|elem| elem.mul_scalar(2.0)).collect();

        // LibTorch reference
        let torch_tensor = LibTorchTensor::from_data(&data, &shape).unwrap();
        let torch_result = torch_tensor.mul_scalar(2.0).unwrap();

        // Compare results
        let comparison = validator.compare_tensors(&our_result, &torch_result);
        if comparison.passed {
            println!("✓ Iterator element operations match LibTorch");
        } else {
            panic!("Iterator vs LibTorch mismatch: {}", comparison.details);
        }
    }

    /// Validate iterator gradients against LibTorch
    #[test]
    fn test_iterator_gradients_vs_libtorch() {
        let validator = TensorValidator::new(1e-6, 1e-9);

        // Test data
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![3];

        // Our implementation with gradients
        let our_tensor = Tensor::from_slice(&data, shape.clone())
            .unwrap()
            .with_requires_grad();
        let our_result: Tensor = our_tensor.iter().map(|elem| elem.mul_scalar(2.0)).collect();
        let mut our_loss = our_result.sum();
        our_loss.backward(None);
        let our_grad = our_tensor
            .grad_by_value()
            .expect("Our gradient should exist");

        // LibTorch reference with gradients
        let torch_tensor = LibTorchTensor::from_data(&data, &shape)
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        let torch_result = torch_tensor.mul_scalar(2.0).unwrap();
        let torch_loss = torch_result.sum().unwrap();
        let grad_ones = LibTorchTensor::ones(&torch_loss.shape()).unwrap();
        torch_loss.backward(Some(&grad_ones)).unwrap();
        let torch_grad = torch_tensor.grad().expect("LibTorch gradient should exist");

        // Compare gradients
        let comparison = validator.compare_tensors(&our_grad, &torch_grad);
        if comparison.passed {
            println!("✓ Iterator gradients match LibTorch");
        } else {
            panic!(
                "Iterator gradient vs LibTorch mismatch: {}",
                comparison.details
            );
        }
    }

    /// Test multi-step iterator chain with LibTorch validation
    #[test]
    fn test_iterator_chain_vs_libtorch() {
        let validator = TensorValidator::new(1e-6, 1e-9);

        // Test data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![5];

        // Our implementation: filter > 2, then multiply by 3
        let our_tensor = Tensor::from_slice(&data, shape.clone())
            .unwrap()
            .with_requires_grad();
        let our_filtered: Vec<Tensor> = our_tensor
            .iter()
            .filter(|elem| elem.value() > 2.0)
            .map(|elem| elem.mul_scalar(3.0))
            .collect();

        // Convert to tensor manually for comparison
        let filtered_data: Vec<f32> = our_filtered.iter().map(|t| t.value()).collect();
        let our_result = Tensor::from_slice(&filtered_data, vec![filtered_data.len()]).unwrap();

        // LibTorch reference: multiply all by 3, then manually filter
        let torch_tensor = LibTorchTensor::from_data(&data, &shape)
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        let torch_multiplied = torch_tensor.mul_scalar(3.0).unwrap();

        // Get values and manually filter (since LibTorch API is limited)
        let torch_data = torch_multiplied.data();
        let torch_filtered: Vec<f32> = torch_data
            .iter()
            .zip(data.iter())
            .filter(|(_, &orig)| orig > 2.0)
            .map(|(val, _)| *val)
            .collect();
        let torch_result =
            LibTorchTensor::from_data(&torch_filtered, &[torch_filtered.len()]).unwrap();

        // Compare forward results
        let comparison = validator.compare_tensors(&our_result, &torch_result);
        if comparison.passed {
            println!("✓ Iterator chain forward pass matches LibTorch");
        } else {
            panic!(
                "Iterator chain vs LibTorch mismatch: {}",
                comparison.details
            );
        }
    }

    /// Test iterator element view creation and access
    #[test]
    fn test_element_view_values_vs_libtorch() {
        let _validator = TensorValidator::new(1e-6, 1e-9);

        // Test data
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];

        // Our implementation - element views
        let our_tensor = Tensor::from_slice(&data, shape.clone()).unwrap();

        // LibTorch reference
        let torch_tensor = LibTorchTensor::from_data(&data, &shape).unwrap();

        // Compare each element view
        for (i, _) in data.iter().enumerate().take(4) {
            let our_elem = our_tensor.element_view(i);
            let our_value = our_elem.value();

            // LibTorch: access element at index
            let torch_value = torch_tensor.data()[i];

            assert!(
                (our_value - torch_value).abs() < 1e-6,
                "Element {} mismatch: our={}, torch={}",
                i,
                our_value,
                torch_value
            );
        }

        println!("✓ Element views match LibTorch element access");
    }
}
