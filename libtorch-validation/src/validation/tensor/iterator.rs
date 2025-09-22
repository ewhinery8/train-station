//! Comprehensive validation tests for tensor iterator collect functionality
//!
//! This module provides validation of tensor iterator and collect operations against LibTorch
//! to ensure correctness of both forward operations and gradient computations. Tests cover:
//! - Simple iterator transformations with collect
//! - Complex iterator chains with multiple operations
//! - Gradient computation validation
//! - Memory layout and shape validation

#[cfg(test)]
mod tests {
    use crate::ffi::LibTorchTensor;
    use crate::validation::core::TensorValidator;
    use train_station::tensor::TensorCollectExt;
    use train_station::Tensor;
    // Bring the public trait into scope to test trait-based collect_shape on chained iterators

    /// Test simple iterator map with collect - forward pass accuracy
    #[test]
    fn test_simple_iterator_collect_forward() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create test tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let our_tensor = Tensor::from_slice(&data, vec![6]).unwrap();

        // Simple iterator map: multiply each element by 2
        let collected: Vec<Tensor> = our_tensor
            .iter_elements()
            .map(|elem| elem.mul_scalar(2.0))
            .collect();

        // Collect into a new tensor with original shape
        let our_result = Tensor::collect_into_shape(collected, vec![6]);

        // LibTorch equivalent operations
        let torch_tensor = LibTorchTensor::from_data(&data, &[6]).unwrap();
        let torch_result = torch_tensor.mul_scalar(2.0).unwrap();

        // Compare forward results
        let forward_comparison = validator.compare_tensors(&our_result, &torch_result);
        assert!(
            forward_comparison.passed,
            "Forward pass failed: {}",
            forward_comparison.details
        );
    }

    /// Test simple iterator map with collect - gradient validation
    #[test]
    fn test_simple_iterator_collect_gradient() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create test tensor with gradient tracking
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut our_tensor = Tensor::from_slice(&data, vec![6]).unwrap();
        our_tensor.set_requires_grad(true);

        // Simple iterator map: multiply each element by 2
        let collected: Vec<Tensor> = our_tensor
            .iter_elements()
            .map(|elem| elem.mul_scalar(2.0))
            .collect();

        // Collect into a new tensor
        let our_result = Tensor::collect_into_shape(collected, vec![6]);

        // Compute loss (sum all elements)
        let mut loss = our_result.sum();
        loss.backward(None);

        // LibTorch equivalent operations with gradients
        let torch_tensor = LibTorchTensor::from_data(&data, &[6])
            .unwrap()
            .requires_grad_(true)
            .unwrap();

        let torch_result = torch_tensor.mul_scalar(2.0).unwrap();
        let torch_loss = torch_result.sum().unwrap();

        // Compute torch gradients
        torch_loss.backward_scalar().unwrap();

        // Compare gradients
        if let Some(our_grad) = our_tensor.grad_owned() {
            if let Some(torch_grad) = torch_tensor.grad() {
                let grad_comparison = validator.compare_tensors(&our_grad, &torch_grad);
                assert!(
                    grad_comparison.passed,
                    "Gradient validation failed: {}",
                    grad_comparison.details
                );
            } else {
                panic!("LibTorch gradient is None");
            }
        } else {
            panic!("Our gradient is None");
        }
    }

    /// Test complex iterator chain with multiple operations
    #[test]
    fn test_complex_iterator_collect_forward() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create test tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let our_tensor = Tensor::from_slice(&data, vec![8]).unwrap();

        // Complex iterator chain: chunk -> transform -> flatten -> collect
        let transformed: Vec<Tensor> = our_tensor
            .iter_chunks(2)
            .enumerate()
            .map(|(i, chunk)| {
                if i % 2 == 0 {
                    // Even chunks: multiply by 2, then exp
                    let temp = chunk.mul_scalar(2.0);
                    temp.exp()
                } else {
                    // Odd chunks: add 1, then sqrt
                    let temp = chunk.add_scalar(1.0);
                    temp.sqrt()
                }
            })
            .flat_map(|chunk| chunk.iter_elements().collect::<Vec<_>>())
            .collect();

        let our_result = Tensor::collect_into_shape(transformed, vec![8]);

        // LibTorch equivalent operations
        // Since LibTorch doesn't have direct equivalents for our iterator operations,
        // we need to manually replicate the chunk-wise transformations

        let torch_tensor = LibTorchTensor::from_data(&data, &[8]).unwrap();

        // Process each chunk manually using indexing operations
        // Chunk 0: [1, 2] -> mul_scalar(2.0) -> exp()
        let chunk0 = torch_tensor
            .index_select(0, &[0, 1])
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .exp()
            .unwrap();

        // Chunk 1: [3, 4] -> add_scalar(1.0) -> sqrt()
        let chunk1 = torch_tensor
            .index_select(0, &[2, 3])
            .unwrap()
            .add_scalar(1.0)
            .unwrap()
            .sqrt()
            .unwrap();

        // Chunk 2: [5, 6] -> mul_scalar(2.0) -> exp()
        let chunk2 = torch_tensor
            .index_select(0, &[4, 5])
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .exp()
            .unwrap();

        // Chunk 3: [7, 8] -> add_scalar(1.0) -> sqrt()
        let chunk3 = torch_tensor
            .index_select(0, &[6, 7])
            .unwrap()
            .add_scalar(1.0)
            .unwrap()
            .sqrt()
            .unwrap();

        // Concatenate all processed chunks
        let torch_result = LibTorchTensor::cat(&[chunk0, chunk1, chunk2, chunk3], 0).unwrap();

        // Compare forward results
        let forward_comparison = validator.compare_tensors(&our_result, &torch_result);
        assert!(
            forward_comparison.passed,
            "Complex forward pass failed: {}",
            forward_comparison.details
        );
    }

    /// Test complex iterator chain with gradients
    #[test]
    fn test_complex_iterator_collect_gradient() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create test tensor with gradient tracking
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut our_tensor = Tensor::from_slice(&data, vec![8]).unwrap();
        our_tensor.set_requires_grad(true);

        // Original complex iterator chain that works with gradients
        let transformed: Vec<Tensor> = our_tensor
            .iter_chunks(2)
            .enumerate()
            .map(|(i, chunk)| {
                if i % 2 == 0 {
                    let temp = chunk.mul_scalar(2.0);
                    temp.exp()
                } else {
                    let temp = chunk.add_scalar(1.0);
                    temp.sqrt()
                }
            })
            .collect();

        let our_result = Tensor::cat(&transformed, 0);

        // Compute loss and gradients
        let mut loss = our_result.sum();
        loss.backward(None);

        // LibTorch equivalent with gradients
        // Complex chunking with alternating transformations (exp for even, sqrt for odd)
        let torch_tensor = LibTorchTensor::from_data(&data, &[8])
            .unwrap()
            .requires_grad_(true)
            .unwrap();

        // Process each chunk with alternating transformations matching our iterator
        let chunk0 = torch_tensor
            .index_select(0, &[0, 1])
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .exp()
            .unwrap(); // even: mul by 2, then exp
        let chunk1 = torch_tensor
            .index_select(0, &[2, 3])
            .unwrap()
            .add_scalar(1.0)
            .unwrap()
            .sqrt()
            .unwrap(); // odd: add 1, then sqrt
        let chunk2 = torch_tensor
            .index_select(0, &[4, 5])
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .exp()
            .unwrap(); // even: mul by 2, then exp
        let chunk3 = torch_tensor
            .index_select(0, &[6, 7])
            .unwrap()
            .add_scalar(1.0)
            .unwrap()
            .sqrt()
            .unwrap(); // odd: add 1, then sqrt

        // Concatenate all transformed chunks
        let torch_result = LibTorchTensor::cat(&[chunk0, chunk1, chunk2, chunk3], 0).unwrap();

        let torch_loss = torch_result.sum().unwrap();
        torch_loss.backward_scalar().unwrap();

        // Compare gradients (retrieve ours by value to ensure access regardless of internal caching)

        if let Some(our_grad) = our_tensor.grad_owned() {
            if let Some(torch_grad) = torch_tensor.grad() {
                let grad_comparison = validator.compare_tensors(&our_grad, &torch_grad);
                assert!(
                    grad_comparison.passed,
                    "Complex gradient validation failed: {}",
                    grad_comparison.details
                );
            } else {
                panic!("LibTorch gradient is None");
            }
        } else {
            panic!("Our gradient is None");
        }
    }

    /// Test chunk-based iterator collect operations
    #[test]
    fn test_chunk_iterator_collect() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create 2D tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let our_tensor = Tensor::from_slice(&data, vec![2, 3]).unwrap();

        // Process chunks and collect
        let processed: Vec<Tensor> = our_tensor
            .iter_chunks(2)
            .map(|chunk| chunk.mul_scalar(3.0).add_scalar(1.0))
            .collect();

        let our_result = Tensor::collect_into_shape(processed, vec![2, 3]);

        // LibTorch equivalent
        let torch_tensor = LibTorchTensor::from_data(&data, &[2, 3]).unwrap();
        let torch_result = torch_tensor
            .mul_scalar(3.0)
            .unwrap()
            .add_scalar(1.0)
            .unwrap();

        // Compare results
        let comparison = validator.compare_tensors(&our_result, &torch_result);
        assert!(
            comparison.passed,
            "Chunk iterator collect failed: {}",
            comparison.details
        );
    }

    /// Test window-based iterator collect operations
    #[test]
    fn test_window_iterator_collect() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create 1D tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let our_tensor = Tensor::from_slice(&data, vec![5]).unwrap();

        // Process windows and collect
        let processed: Vec<Tensor> = our_tensor
            .iter_windows(3)
            .map(|window| window.sum())
            .collect();

        let our_result = Tensor::collect_into_shape(processed, vec![3]);

        // LibTorch equivalent using convolution-like operation
        let torch_tensor = LibTorchTensor::from_data(&data, &[5]).unwrap();

        // Manual window sum computation for LibTorch
        let w0 = torch_tensor
            .select(0, 0)
            .unwrap()
            .add_tensor(&torch_tensor.select(0, 1).unwrap())
            .unwrap()
            .add_tensor(&torch_tensor.select(0, 2).unwrap())
            .unwrap();
        let w1 = torch_tensor
            .select(0, 1)
            .unwrap()
            .add_tensor(&torch_tensor.select(0, 2).unwrap())
            .unwrap()
            .add_tensor(&torch_tensor.select(0, 3).unwrap())
            .unwrap();
        let w2 = torch_tensor
            .select(0, 2)
            .unwrap()
            .add_tensor(&torch_tensor.select(0, 3).unwrap())
            .unwrap()
            .add_tensor(&torch_tensor.select(0, 4).unwrap())
            .unwrap();

        let torch_result = LibTorchTensor::stack(&[w0, w1, w2], 0).unwrap();

        // Compare results
        let comparison = validator.compare_tensors(&our_result, &torch_result);
        assert!(
            comparison.passed,
            "Window iterator collect failed: {}",
            comparison.details
        );
    }

    /// Test dimension-based iterator collect operations
    #[test]
    fn test_dimension_iterator_collect() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create 2D tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let our_tensor = Tensor::from_slice(&data, vec![2, 3]).unwrap();

        // Process along dimension 0 and collect
        let processed: Vec<Tensor> = our_tensor
            .iter_dim(0)
            .map(|row| row.mul_scalar(2.0))
            .collect();

        let our_result = Tensor::collect_into_shape(processed, vec![2, 3]);

        // LibTorch equivalent
        let torch_tensor = LibTorchTensor::from_data(&data, &[2, 3]).unwrap();
        let torch_result = torch_tensor.mul_scalar(2.0).unwrap();

        // Compare results
        let comparison = validator.compare_tensors(&our_result, &torch_result);
        assert!(
            comparison.passed,
            "Dimension iterator collect failed: {}",
            comparison.details
        );
    }

    /// Test iterator collect with shape transformation
    #[test]
    fn test_iterator_collect_shape_transformation() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create 1D tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let our_tensor = Tensor::from_slice(&data, vec![6]).unwrap();

        // Split into chunks and collect into different shape
        let chunks: Vec<Tensor> = our_tensor.iter_chunks(2).collect();
        let our_result = Tensor::collect_into_shape(chunks, vec![3, 2]);

        // LibTorch equivalent: reshape
        let torch_tensor = LibTorchTensor::from_data(&data, &[6]).unwrap();
        let torch_result = torch_tensor.view(&[3, 2]).unwrap();

        // Compare results
        let comparison = validator.compare_tensors(&our_result, &torch_result);
        assert!(
            comparison.passed,
            "Shape transformation collect failed: {}",
            comparison.details
        );
    }

    /// Test iterator collect with gradient accumulation
    #[test]
    fn test_iterator_collect_gradient_accumulation() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create test tensor with gradient tracking
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut our_tensor = Tensor::from_slice(&data, vec![4]).unwrap();
        our_tensor.set_requires_grad(true);

        // Create multiple transformations that depend on the same tensor
        let chunk1 = our_tensor.iter_chunks(2).next().unwrap().mul_scalar(2.0);
        let chunk2 = our_tensor.iter_chunks(2).nth(1).unwrap().mul_scalar(3.0);

        let chunks = vec![chunk1, chunk2];
        let our_result = Tensor::collect_into_shape(chunks, vec![4]);

        // Compute loss
        let mut loss = our_result.sum();
        loss.backward(None);

        // LibTorch equivalent
        let torch_tensor = LibTorchTensor::from_data(&data, &[4])
            .unwrap()
            .requires_grad_(true)
            .unwrap();

        let torch_chunk1 = torch_tensor
            .index_select(0, &[0, 1])
            .unwrap()
            .mul_scalar(2.0)
            .unwrap();
        let torch_chunk2 = torch_tensor
            .index_select(0, &[2, 3])
            .unwrap()
            .mul_scalar(3.0)
            .unwrap();

        let torch_result = LibTorchTensor::cat(&[torch_chunk1, torch_chunk2], 0).unwrap();
        let torch_loss = torch_result.sum().unwrap();

        torch_loss.backward_scalar().unwrap();

        // Compare gradients (use by-value access for our side)
        if let Some(our_grad) = our_tensor.grad_owned() {
            if let Some(torch_grad) = torch_tensor.grad() {
                let grad_comparison = validator.compare_tensors(&our_grad, &torch_grad);
                assert!(
                    grad_comparison.passed,
                    "Gradient accumulation failed: {}",
                    grad_comparison.details
                );
            }
        }
    }

    /// Test iterator collect with mixed element operations
    #[test]
    fn test_iterator_collect_mixed_operations() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create test tensor
        let data = vec![0.5, 1.0, 1.5, 2.0];
        let our_tensor = Tensor::from_slice(&data, vec![4]).unwrap();

        // Mixed operations: some elements get exp, others get sqrt
        let processed: Vec<Tensor> = our_tensor
            .iter_elements()
            .enumerate()
            .map(
                |(i, elem)| {
                    if i % 2 == 0 {
                        elem.exp()
                    } else {
                        elem.sqrt()
                    }
                },
            )
            .collect();

        let our_result = Tensor::collect_into_shape(processed, vec![4]);

        // LibTorch equivalent: process elements individually and concatenate to preserve order
        let torch_tensor = LibTorchTensor::from_data(&data, &[4]).unwrap();

        let p0 = torch_tensor.select(0, 0).unwrap().exp().unwrap();
        let p1 = torch_tensor.select(0, 1).unwrap().sqrt().unwrap();
        let p2 = torch_tensor.select(0, 2).unwrap().exp().unwrap();
        let p3 = torch_tensor.select(0, 3).unwrap().sqrt().unwrap();

        let torch_result = LibTorchTensor::stack(&[p0, p1, p2, p3], 0).unwrap();

        // Compare results
        let comparison = validator.compare_tensors(&our_result, &torch_result);
        assert!(
            comparison.passed,
            "Mixed operations collect failed: {}",
            comparison.details
        );
    }

    /// Test iterator collect with values iterator
    #[test]
    fn test_values_iterator_collect() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Create test tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let our_tensor = Tensor::from_slice(&data, vec![6]).unwrap();

        // Process values and collect into new shape
        let processed_values: Vec<f32> = our_tensor
            .iter_values()
            .map(|val| val * 2.0 + 1.0)
            .collect();

        let our_result = Tensor::from_slice(&processed_values, vec![2, 3]).unwrap();

        // LibTorch equivalent
        let torch_tensor = LibTorchTensor::from_data(&data, &[6]).unwrap();
        let torch_result = torch_tensor
            .mul_scalar(2.0)
            .unwrap()
            .add_scalar(1.0)
            .unwrap()
            .view(&[2, 3])
            .unwrap();

        // Compare results
        let comparison = validator.compare_tensors(&our_result, &torch_result);
        assert!(
            comparison.passed,
            "Values iterator collect failed: {}",
            comparison.details
        );
    }

    /// Chunks iterator: forward/backward with both inherent and trait collect methods
    #[test]
    fn test_chunks_iterator_collect_trait_and_inherent_with_grad() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Base contiguous tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut ours = Tensor::from_slice(&data, vec![8]).unwrap();
        ours.set_requires_grad(true);

        // Path A: Inherent collect_shape on chunks iterator
        let chunks_a = ours
            .iter_chunks(2)
            .map(|c| c.mul_scalar(2.0).exp())
            .collect_shape(vec![8]);

        // Path B: Trait collect_shape on an equivalent map chain
        let parts_b: Vec<Tensor> = ours
            .iter_chunks(2)
            .map(|c| c.mul_scalar(2.0))
            .map(|c| c.exp())
            .collect();
        let chunks_b = parts_b.into_iter().collect_shape(vec![8]);

        // Torch baseline via explicit chunk processing
        let torch_base = LibTorchTensor::from_data(&data, &[8])
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        let c0 = torch_base
            .index_select(0, &[0, 1])
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .exp()
            .unwrap();
        let c1 = torch_base
            .index_select(0, &[2, 3])
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .exp()
            .unwrap();
        let c2 = torch_base
            .index_select(0, &[4, 5])
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .exp()
            .unwrap();
        let c3 = torch_base
            .index_select(0, &[6, 7])
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .exp()
            .unwrap();
        let torch_result = LibTorchTensor::cat(&[c0, c1, c2, c3], 0).unwrap();

        // Forward: both variants should match torch_result
        let cmp_a = TensorValidator::new(1e-5, 1e-8).compare_tensors(&chunks_a, &torch_result);
        assert!(
            cmp_a.passed,
            "Chunks (inherent) forward failed: {}",
            cmp_a.details
        );
        let cmp_b = TensorValidator::new(1e-5, 1e-8).compare_tensors(&chunks_b, &torch_result);
        assert!(
            cmp_b.passed,
            "Chunks (trait) forward failed: {}",
            cmp_b.details
        );

        // Backward
        let mut loss_a = chunks_a.sum();
        loss_a.backward(None);
        let g_ours = ours.grad_owned().expect("our gradient");

        let torch_loss = torch_result.sum().unwrap();
        torch_loss.backward_scalar().unwrap();
        let g_torch = torch_base.grad().expect("torch gradient");

        let cmp_g = validator.compare_tensors(&g_ours, &g_torch);
        assert!(cmp_g.passed, "Chunks gradient failed: {}", cmp_g.details);
    }

    /// Windows iterator: forward/backward, iterator chaining and collect methods
    #[test]
    fn test_windows_iterator_collect_forward_backward() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Base tensor length 6 -> windows of 3 produce 4 windows
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ours = Tensor::from_slice(&data, vec![6])
            .unwrap()
            .with_requires_grad();

        // Chain: windows -> sum -> affine
        let windows: Vec<Tensor> = ours
            .iter_windows(3)
            .map(|w| w.sum())
            .map(|s| s.mul_scalar(2.0).add_scalar(1.0))
            .collect();
        let collected = Tensor::collect_into_shape(windows, vec![4]);

        // Torch equivalent via manual sliding window
        let t = LibTorchTensor::from_data(&data, &[6])
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        let w0 = t
            .select(0, 0)
            .unwrap()
            .add_tensor(&t.select(0, 1).unwrap())
            .unwrap()
            .add_tensor(&t.select(0, 2).unwrap())
            .unwrap();
        let w1 = t
            .select(0, 1)
            .unwrap()
            .add_tensor(&t.select(0, 2).unwrap())
            .unwrap()
            .add_tensor(&t.select(0, 3).unwrap())
            .unwrap();
        let w2 = t
            .select(0, 2)
            .unwrap()
            .add_tensor(&t.select(0, 3).unwrap())
            .unwrap()
            .add_tensor(&t.select(0, 4).unwrap())
            .unwrap();
        let w3 = t
            .select(0, 3)
            .unwrap()
            .add_tensor(&t.select(0, 4).unwrap())
            .unwrap()
            .add_tensor(&t.select(0, 5).unwrap())
            .unwrap();
        let a0 = w0.mul_scalar(2.0).unwrap().add_scalar(1.0).unwrap();
        let a1 = w1.mul_scalar(2.0).unwrap().add_scalar(1.0).unwrap();
        let a2 = w2.mul_scalar(2.0).unwrap().add_scalar(1.0).unwrap();
        let a3 = w3.mul_scalar(2.0).unwrap().add_scalar(1.0).unwrap();
        let torch_result = LibTorchTensor::stack(&[a0, a1, a2, a3], 0).unwrap();

        let cmp = validator.compare_tensors(&collected, &torch_result);
        assert!(cmp.passed, "Windows forward failed: {}", cmp.details);

        let mut loss = collected.sum();
        loss.backward(None);
        let g_ours = ours.grad_owned().unwrap();

        let torch_loss = torch_result.sum().unwrap();
        torch_loss.backward_scalar().unwrap();
        let g_torch = t.grad().unwrap();

        let cmpg = validator.compare_tensors(&g_ours, &g_torch);
        assert!(cmpg.passed, "Windows gradient failed: {}", cmpg.details);
    }

    /// Complex map chain over chunks + elements with non-trivial origin (transpose)
    #[test]
    fn test_iterator_chain_on_transposed_tensor_forward_backward() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        // Build 2D then transpose to simulate non-trivial layout
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let base = Tensor::from_slice(&data, vec![2, 4])
            .unwrap()
            .with_requires_grad();
        // Ensure contiguous layout after transpose so iterator linearization matches logical order
        let ours = base.transpose(0, 1).contiguous().mul_scalar(1.0);

        // Chain: chunks -> transform -> flatten to elements -> collect
        let parts: Vec<Tensor> = ours
            .iter_chunks(2)
            .map(|c| c.mul_scalar(0.5))
            .flat_map(|c| c.iter_elements().collect::<Vec<_>>())
            .collect();
        let out = Tensor::collect_into_shape(parts, vec![8]);

        // Torch: replicate using index_select on transposed view
        let t2d = LibTorchTensor::from_data(&data, &[2, 4])
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        // Transpose via permute([1,0])
        let t_tr = t2d.permute(&[1, 0]).unwrap().requires_grad_(true).unwrap();
        let s0 = t_tr
            .index_select(0, &[0, 1])
            .unwrap()
            .mul_scalar(0.5)
            .unwrap();
        let s1 = t_tr
            .index_select(0, &[2, 3])
            .unwrap()
            .mul_scalar(0.5)
            .unwrap();
        // Flatten by viewing as 1D using view with inferred total elements
        let s0f = s0.view(&[4]).unwrap();
        let s1f = s1.view(&[4]).unwrap();
        let torch_result = LibTorchTensor::cat(&[s0f, s1f], 0).unwrap();

        let cmp = validator.compare_tensors(&out, &torch_result);
        assert!(cmp.passed, "Chain forward failed: {}", cmp.details);

        let mut loss = out.sum();
        loss.backward(None);
        // Compare gradients with respect to the original leaf (base)
        let g_ours = base.grad_owned().unwrap();

        let torch_loss = torch_result.sum().unwrap();
        torch_loss.backward_scalar().unwrap();
        // In LibTorch, gradients populate the leaf `t2d`; compare directly in base layout
        let g_torch = t2d.grad().unwrap();

        let cmpg = validator.compare_tensors(&g_ours, &g_torch);
        assert!(cmpg.passed, "Chain gradient failed: {}", cmpg.details);
    }

    /// Validate collect on contiguous and transposed tensors for multiple collect styles
    #[test]
    fn test_collect_variants_contiguous_and_transposed() {
        let validator = TensorValidator::new(1e-5, 1e-8);

        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let contig = Tensor::from_slice(&data, vec![12]).unwrap();
        let nontrivial = Tensor::from_slice(&data, vec![3, 4])
            .unwrap()
            .transpose(0, 1)
            .contiguous()
            .view(vec![12]);

        // Inherent collect_shape on elements iterator
        let a = contig
            .iter_elements()
            .map(|e| e.mul_scalar(2.0))
            .collect_shape(vec![12]);
        let b = nontrivial
            .iter_elements()
            .map(|e| e.mul_scalar(2.0))
            .collect_shape(vec![12]);

        // Generic helper based collect
        let c =
            Tensor::collect_shape_from(contig.iter_chunks(3).map(|c| c.add_scalar(1.0)), vec![12]);
        let d = Tensor::collect_shape_from(
            nontrivial.iter_chunks(3).map(|c| c.add_scalar(1.0)),
            vec![12],
        );

        // Torch baselines
        let t = LibTorchTensor::from_data(&data, &[12]).unwrap();
        let ta = t.mul_scalar(2.0).unwrap();
        let tb = t
            .view(&[3, 4])
            .unwrap()
            .permute(&[1, 0])
            .unwrap()
            .view(&[12])
            .unwrap()
            .mul_scalar(2.0)
            .unwrap();
        let tc = t.add_scalar(1.0).unwrap();
        // For the nontrivial + iter_chunks(3) path, our iterator chunks the flattened transposed tensor.
        // The equivalent LibTorch baseline is adding 1.0 to the flattened transposed view of `t`.
        let td = t
            .view(&[3, 4])
            .unwrap()
            .permute(&[1, 0])
            .unwrap()
            .view(&[12])
            .unwrap()
            .add_scalar(1.0)
            .unwrap();

        assert!(validator.compare_tensors(&a, &ta).passed);
        assert!(validator.compare_tensors(&b, &tb).passed);
        assert!(validator.compare_tensors(&c, &tc).passed);
        assert!(validator.compare_tensors(&d, &td).passed);
    }
}
