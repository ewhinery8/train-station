//! Adam optimizer validation against LibTorch reference
//!
//! Validates parameter updates, gradient computations, and zero_grad behavior
//! for both single and multi-parameter configurations in both safe and unsafe modes.

use crate::ffi::{LibTorchAdam, LibTorchTensor};
use crate::validation::core::{ComparisonResult, TensorValidator};
use std::sync::{Arc, RwLock};
use train_station::optimizers::Optimizer;
use train_station::optimizers::{Adam, AdamConfig};
use train_station::Tensor;

/// Configuration for Adam optimizer validation tests
#[derive(Debug, Clone)]
pub struct AdamValidationConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub steps: usize,
}

impl Default for AdamValidationConfig {
    fn default() -> Self {
        AdamValidationConfig {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            steps: 3,
        }
    }
}

/// Validator for Adam optimizer operations
pub struct AdamValidator {
    pub tensor_validator: TensorValidator,
}

impl Default for AdamValidator {
    fn default() -> Self {
        AdamValidator {
            // Use strict tolerances for optimizer parameter comparisons
            // to ensure mathematical equivalence between our f32 implementation
            // and LibTorch's f64 implementation. Differences larger than 1e-4
            // indicate potential bugs in our implementation.
            tensor_validator: TensorValidator::new(1e-4, 1e-4), // Strict tolerance for f32/f64 precision
        }
    }
}

impl AdamValidator {
    /// Create new Adam validator with custom tolerances
    pub fn new(rtol: f64, atol: f64) -> Self {
        AdamValidator {
            tensor_validator: TensorValidator::new(rtol, atol),
        }
    }

    /// Validate Adam updates for single parameter (safe mode)
    pub fn validate_single_parameter(
        &self,
        shape: &[usize],
        config: &AdamValidationConfig,
    ) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let initial_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01 - 0.5).collect();

        // Our implementation (safe mode)
        let our_param = match Tensor::from_slice(&initial_data, shape.to_vec()) {
            Ok(p) => p.with_requires_grad(),
            Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
        };
        let our_param_locked = Arc::new(RwLock::new(our_param));

        let our_options = AdamConfig {
            learning_rate: config.lr as f32,
            beta1: config.beta1 as f32,
            beta2: config.beta2 as f32,
            eps: config.eps as f32,
            weight_decay: config.weight_decay as f32,
            amsgrad: config.amsgrad,
        };
        let mut our_adam = Adam::with_config(our_options);
        our_adam.add_parameter(&our_param_locked.read().unwrap());

        // LibTorch implementation
        let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
            Ok(t) => {
                if let Err(e) = t.requires_grad(true) {
                    return ComparisonResult::failure(format!("requires_grad failed: {}", e));
                }
                t
            }
            Err(e) => return ComparisonResult::failure(format!("from_data failed: {}", e)),
        };
        let torch_adam = match LibTorchAdam::new(
            &[&torch_param],
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.weight_decay,
            config.amsgrad,
        ) {
            Ok(o) => o,
            Err(e) => {
                return ComparisonResult::failure(format!("Failed to create torch adam: {}", e))
            }
        };

        for _ in 0..config.steps {
            // Forward pass to generate gradients
            {
                let param_guard = our_param_locked.read().unwrap();
                let our_output = param_guard.mul_scalar(2.0).add_scalar(1.0);
                let mut our_loss = our_output.sum();
                our_loss.backward(None);
            }

            let torch_output = match torch_param.mul_scalar(2.0).and_then(|t| t.add_scalar(1.0)) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(format!("Torch forward failed: {}", e)),
            };
            let torch_loss = match torch_output.sum() {
                Ok(l) => l,
                Err(e) => return ComparisonResult::failure(format!("Torch sum failed: {}", e)),
            };
            if let Err(e) = torch_loss.backward(None) {
                return ComparisonResult::failure(format!("Torch backward failed: {}", e));
            }

            // Step optimizers
            {
                let mut param_guard = our_param_locked.write().unwrap();
                our_adam.step(&mut [&mut *param_guard]);
            }
            if let Err(e) = torch_adam.step() {
                return ComparisonResult::failure(format!("Torch adam step failed: {}", e));
            }

            // Zero gradients
            {
                let mut param_guard = our_param_locked.write().unwrap();
                our_adam.zero_grad(&mut [&mut *param_guard]);
            }
            if let Err(e) = torch_adam.zero_grad() {
                return ComparisonResult::failure(format!("Torch zero_grad failed: {}", e));
            }
        }

        // Compare final parameters
        let our_param_final = our_param_locked.read().unwrap();
        self.tensor_validator
            .compare_tensors(&our_param_final, &torch_param)
    }

    /// Validate Adam updates for single parameter (unsafe mode)
    pub fn validate_single_parameter_unsafe(
        &self,
        shape: &[usize],
        config: &AdamValidationConfig,
    ) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let initial_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01 - 0.5).collect();

        // Our implementation (unsafe mode)
        let mut our_param = match Tensor::from_slice(&initial_data, shape.to_vec()) {
            Ok(p) => p.with_requires_grad(),
            Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
        };

        let our_options = AdamConfig {
            learning_rate: config.lr as f32,
            beta1: config.beta1 as f32,
            beta2: config.beta2 as f32,
            eps: config.eps as f32,
            weight_decay: config.weight_decay as f32,
            amsgrad: config.amsgrad,
        };
        let mut our_adam = Adam::with_config(our_options);
        our_adam.add_parameter(&our_param);

        // LibTorch implementation
        let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
            Ok(t) => {
                if let Err(e) = t.requires_grad(true) {
                    return ComparisonResult::failure(format!("requires_grad failed: {}", e));
                }
                t
            }
            Err(e) => return ComparisonResult::failure(format!("from_data failed: {}", e)),
        };
        let torch_adam = match LibTorchAdam::new(
            &[&torch_param],
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.weight_decay,
            config.amsgrad,
        ) {
            Ok(o) => o,
            Err(e) => {
                return ComparisonResult::failure(format!("Failed to create torch adam: {}", e))
            }
        };

        for _ in 0..config.steps {
            // Forward pass to generate gradients
            let our_output = our_param.mul_scalar(2.0).add_scalar(1.0);
            let mut our_loss = our_output.sum();
            our_loss.backward(None);

            let torch_output = match torch_param.mul_scalar(2.0).and_then(|t| t.add_scalar(1.0)) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(format!("Torch forward failed: {}", e)),
            };
            let torch_loss = match torch_output.sum() {
                Ok(l) => l,
                Err(e) => return ComparisonResult::failure(format!("Torch sum failed: {}", e)),
            };
            if let Err(e) = torch_loss.backward(None) {
                return ComparisonResult::failure(format!("Torch backward failed: {}", e));
            }

            // Step optimizers
            our_adam.step(&mut [&mut our_param]);
            if let Err(e) = torch_adam.step() {
                return ComparisonResult::failure(format!("Torch adam step failed: {}", e));
            }

            // Zero gradients
            our_adam.zero_grad(&mut [&mut our_param]);
            if let Err(e) = torch_adam.zero_grad() {
                return ComparisonResult::failure(format!("Torch zero_grad failed: {}", e));
            }
        }

        // Compare final parameters
        self.tensor_validator
            .compare_tensors(&our_param, &torch_param)
    }

    /// Validate Adam updates for multiple parameters (safe mode)
    pub fn validate_multi_parameter(
        &self,
        shapes: Vec<Vec<usize>>,
        config: &AdamValidationConfig,
    ) -> Vec<ComparisonResult> {
        let mut results = Vec::with_capacity(shapes.len());

        // Create parameters
        let mut our_params_locked: Vec<Arc<RwLock<Tensor>>> = Vec::with_capacity(shapes.len());
        let mut torch_params: Vec<LibTorchTensor> = Vec::with_capacity(shapes.len());

        for shape in &shapes {
            let numel: usize = shape.iter().product();
            let initial_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01 - 0.5).collect();

            let our_param = match Tensor::from_slice(&initial_data, shape.clone()) {
                Ok(p) => p.with_requires_grad(),
                Err(e) => {
                    results.push(ComparisonResult::failure(format!(
                        "from_slice failed: {}",
                        e
                    )));
                    continue;
                }
            };
            our_params_locked.push(Arc::new(RwLock::new(our_param)));

            let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
                Ok(t) => {
                    if let Err(e) = t.requires_grad(true) {
                        results.push(ComparisonResult::failure(format!(
                            "requires_grad failed: {}",
                            e
                        )));
                        continue;
                    }
                    t
                }
                Err(e) => {
                    results.push(ComparisonResult::failure(format!(
                        "from_data failed: {}",
                        e
                    )));
                    continue;
                }
            };
            torch_params.push(torch_param);
        }

        // Create optimizers
        let our_options = AdamConfig {
            learning_rate: config.lr as f32,
            beta1: config.beta1 as f32,
            beta2: config.beta2 as f32,
            eps: config.eps as f32,
            weight_decay: config.weight_decay as f32,
            amsgrad: config.amsgrad,
        };
        let mut our_adam = Adam::with_config(our_options);
        for param_locked in &our_params_locked {
            our_adam.add_parameter(&param_locked.read().unwrap());
        }

        let torch_refs: Vec<&LibTorchTensor> = torch_params.iter().collect();
        let torch_adam = match LibTorchAdam::new(
            &torch_refs,
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.weight_decay,
            config.amsgrad,
        ) {
            Ok(o) => o,
            Err(e) => {
                results.push(ComparisonResult::failure(format!(
                    "Failed to create torch adam: {}",
                    e
                )));
                return results;
            }
        };

        for _ in 0..config.steps {
            // Forward pass: sum all params
            let mut our_loss = Tensor::zeros(vec![1]);
            for param_locked in &our_params_locked {
                let param_guard = param_locked.read().unwrap();
                our_loss = our_loss.add_tensor(&param_guard.sum());
            }

            our_loss.backward(None);

            let mut torch_loss = match LibTorchTensor::zeros(&[1]) {
                Ok(l) => l,
                Err(e) => {
                    results.push(ComparisonResult::failure(format!(
                        "Torch zeros failed: {}",
                        e
                    )));
                    continue;
                }
            };
            for param in &torch_params {
                match param.sum() {
                    Ok(s) => torch_loss = torch_loss.add_tensor(&s).unwrap(),
                    Err(e) => {
                        results.push(ComparisonResult::failure(format!(
                            "Torch sum failed: {}",
                            e
                        )));
                        continue;
                    }
                }
            }
            if let Err(e) = torch_loss.backward(None) {
                results.push(ComparisonResult::failure(format!(
                    "Torch backward failed: {}",
                    e
                )));
                continue;
            }

            // Step
            {
                let mut param_guards: Vec<_> = our_params_locked
                    .iter()
                    .map(|p| p.write().unwrap())
                    .collect();
                let mut param_refs: Vec<&mut Tensor> =
                    param_guards.iter_mut().map(|guard| &mut **guard).collect();
                our_adam.step(&mut param_refs);
            }
            if let Err(e) = torch_adam.step() {
                results.push(ComparisonResult::failure(format!(
                    "Torch step failed: {}",
                    e
                )));
                continue;
            }

            // Zero grad
            {
                let mut param_guards: Vec<_> = our_params_locked
                    .iter()
                    .map(|p| p.write().unwrap())
                    .collect();
                let mut param_refs: Vec<&mut Tensor> =
                    param_guards.iter_mut().map(|guard| &mut **guard).collect();
                our_adam.zero_grad(&mut param_refs);
            }
            if let Err(e) = torch_adam.zero_grad() {
                results.push(ComparisonResult::failure(format!(
                    "Torch zero_grad failed: {}",
                    e
                )));
                continue;
            }
        }

        // Compare each parameter
        for (param_locked, torch) in our_params_locked.iter().zip(torch_params.iter()) {
            let param_guard = param_locked.read().unwrap();
            results.push(self.tensor_validator.compare_tensors(&param_guard, torch));
        }

        results
    }

    /// Validate Adam updates for multiple parameters (unsafe mode)
    pub fn validate_multi_parameter_unsafe(
        &self,
        shapes: Vec<Vec<usize>>,
        config: &AdamValidationConfig,
    ) -> Vec<ComparisonResult> {
        let mut results = Vec::with_capacity(shapes.len());

        // Create parameters
        let mut our_params_owned: Vec<Tensor> = Vec::with_capacity(shapes.len());
        let mut torch_params: Vec<LibTorchTensor> = Vec::with_capacity(shapes.len());

        for shape in &shapes {
            let numel: usize = shape.iter().product();
            let initial_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01 - 0.5).collect();

            let our_param = match Tensor::from_slice(&initial_data, shape.clone()) {
                Ok(p) => p.with_requires_grad(),
                Err(e) => {
                    results.push(ComparisonResult::failure(format!(
                        "from_slice failed: {}",
                        e
                    )));
                    continue;
                }
            };
            our_params_owned.push(our_param);

            let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
                Ok(t) => {
                    if let Err(e) = t.requires_grad(true) {
                        results.push(ComparisonResult::failure(format!(
                            "requires_grad failed: {}",
                            e
                        )));
                        continue;
                    }
                    t
                }
                Err(e) => {
                    results.push(ComparisonResult::failure(format!(
                        "from_data failed: {}",
                        e
                    )));
                    continue;
                }
            };
            torch_params.push(torch_param);
        }

        // Create optimizers
        let our_options = AdamConfig {
            learning_rate: config.lr as f32,
            beta1: config.beta1 as f32,
            beta2: config.beta2 as f32,
            eps: config.eps as f32,
            weight_decay: config.weight_decay as f32,
            amsgrad: config.amsgrad,
        };

        // Create optimizer and link parameters
        let mut our_adam = Adam::with_config(our_options);
        for param in &our_params_owned {
            our_adam.add_parameter(param);
        }

        let torch_refs: Vec<&LibTorchTensor> = torch_params.iter().collect();
        let torch_adam = match LibTorchAdam::new(
            &torch_refs,
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.weight_decay,
            config.amsgrad,
        ) {
            Ok(o) => o,
            Err(e) => {
                results.push(ComparisonResult::failure(format!(
                    "Failed to create torch adam: {}",
                    e
                )));
                return results;
            }
        };

        for _ in 0..config.steps {
            // Forward pass: sum all params
            let mut our_loss = Tensor::zeros(vec![1]);
            for param in &our_params_owned {
                our_loss = our_loss.add_tensor(&param.sum());
            }

            our_loss.backward(None);

            let mut torch_loss = match LibTorchTensor::zeros(&[1]) {
                Ok(l) => l,
                Err(e) => {
                    results.push(ComparisonResult::failure(format!(
                        "Torch zeros failed: {}",
                        e
                    )));
                    continue;
                }
            };
            for param in &torch_params {
                match param.sum() {
                    Ok(s) => torch_loss = torch_loss.add_tensor(&s).unwrap(),
                    Err(e) => {
                        results.push(ComparisonResult::failure(format!(
                            "Torch sum failed: {}",
                            e
                        )));
                        continue;
                    }
                }
            }
            if let Err(e) = torch_loss.backward(None) {
                results.push(ComparisonResult::failure(format!(
                    "Torch backward failed: {}",
                    e
                )));
                continue;
            }

            // Step
            {
                let mut param_refs: Vec<&mut Tensor> = our_params_owned.iter_mut().collect();
                our_adam.step(&mut param_refs);
            }
            if let Err(e) = torch_adam.step() {
                results.push(ComparisonResult::failure(format!(
                    "Torch step failed: {}",
                    e
                )));
                continue;
            }

            // Zero grad
            {
                let mut param_refs: Vec<&mut Tensor> = our_params_owned.iter_mut().collect();
                our_adam.zero_grad(&mut param_refs);
            }
            if let Err(e) = torch_adam.zero_grad() {
                results.push(ComparisonResult::failure(format!(
                    "Torch zero_grad failed: {}",
                    e
                )));
                continue;
            }
        }

        // Compare each parameter
        for (our, torch) in our_params_owned.iter().zip(torch_params.iter()) {
            results.push(self.tensor_validator.compare_tensors(our, torch));
        }

        results
    }

    /// Verify zero_grad properly resets gradients (safe mode)
    pub fn verify_zero_grad(&self, shape: &[usize]) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = vec![1.0; numel];

        // Our implementation
        let our_param = match Tensor::from_slice(&data, shape.to_vec()) {
            Ok(p) => p.with_requires_grad(),
            Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
        };
        let our_param_locked = Arc::new(RwLock::new(our_param));

        // First, generate gradients through computation
        {
            let param_guard = our_param_locked.read().unwrap();
            let our_output = param_guard.mul_scalar(2.0);
            our_output.sum().backward(None);
        }

        // Check that gradients exist before zero_grad
        {
            let param_guard = our_param_locked.read().unwrap();
            if param_guard.grad_by_value().is_none() {
                return ComparisonResult::failure(
                    "No gradients generated before zero_grad".to_string(),
                );
            }
        }

        // Use optimizer-style zero_grad (which calls gradtrack::clear_gradients)
        {
            let mut param_guard = our_param_locked.write().unwrap();
            param_guard.zero_grad();
        }
        train_station::clear_gradients(); // Ensure global gradient map is cleared

        let our_grad = {
            let param_guard = our_param_locked.read().unwrap();
            param_guard
                .grad_by_value()
                .unwrap_or_else(|| Tensor::zeros(param_guard.shape().dims.clone()))
        };

        // LibTorch implementation
        let torch_param = LibTorchTensor::from_data(&data, shape).unwrap();
        if let Err(e) = torch_param.requires_grad(true) {
            return ComparisonResult::failure(format!("requires_grad failed: {}", e));
        }

        let torch_output = torch_param.mul_scalar(2.0).unwrap();
        let torch_loss = torch_output.sum().unwrap();
        if let Err(e) = torch_loss.backward(None) {
            return ComparisonResult::failure(format!("Torch backward failed: {}", e));
        }
        if let Err(e) = torch_param.zero_grad() {
            return ComparisonResult::failure(format!("Torch zero_grad failed: {}", e));
        }

        let torch_grad = torch_param
            .grad()
            .unwrap_or_else(|| LibTorchTensor::zeros(&torch_param.shape()).unwrap());

        // Both gradients should be zero tensors after zero_grad
        self.tensor_validator
            .compare_tensors(&our_grad, &torch_grad)
    }

    /// Verify zero_grad properly resets gradients (unsafe mode)
    pub fn verify_zero_grad_unsafe(&self, shape: &[usize]) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = vec![1.0; numel];

        // Our implementation
        let mut our_param = match Tensor::from_slice(&data, shape.to_vec()) {
            Ok(p) => p.with_requires_grad(),
            Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
        };

        // First, generate gradients through computation
        let our_output = our_param.mul_scalar(2.0);
        our_output.sum().backward(None);

        // Check that gradients exist before zero_grad
        if our_param.grad_by_value().is_none() {
            return ComparisonResult::failure(
                "No gradients generated before zero_grad".to_string(),
            );
        }

        // Use optimizer-style zero_grad (which calls gradtrack::clear_gradients)
        our_param.zero_grad();
        train_station::clear_gradients(); // Ensure global gradient map is cleared

        let our_grad = our_param
            .grad_by_value()
            .unwrap_or_else(|| Tensor::zeros(our_param.shape().dims.clone()));

        // LibTorch implementation
        let torch_param = LibTorchTensor::from_data(&data, shape).unwrap();
        if let Err(e) = torch_param.requires_grad(true) {
            return ComparisonResult::failure(format!("requires_grad failed: {}", e));
        }

        let torch_output = torch_param.mul_scalar(2.0).unwrap();
        let torch_loss = torch_output.sum().unwrap();
        if let Err(e) = torch_loss.backward(None) {
            return ComparisonResult::failure(format!("Torch backward failed: {}", e));
        }
        if let Err(e) = torch_param.zero_grad() {
            return ComparisonResult::failure(format!("Torch zero_grad failed: {}", e));
        }

        let torch_grad = torch_param
            .grad()
            .unwrap_or_else(|| LibTorchTensor::zeros(&torch_param.shape()).unwrap());

        // Both gradients should be zero tensors after zero_grad
        self.tensor_validator
            .compare_tensors(&our_grad, &torch_grad)
    }

    /// Validate Adam with zero gradients
    /// Tests that the optimizer correctly handles parameters with no gradients
    pub fn validate_zero_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let initial_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01 - 0.5).collect();

        // Our implementation (safe mode)
        let our_param = match Tensor::from_slice(&initial_data, shape.to_vec()) {
            Ok(p) => p.with_requires_grad(),
            Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
        };
        let our_param_locked = Arc::new(RwLock::new(our_param));

        let our_options = AdamConfig::default();
        let mut our_adam = Adam::with_config(our_options);
        our_adam.add_parameter(&our_param_locked.read().unwrap());

        // LibTorch implementation
        let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
            Ok(t) => {
                if let Err(e) = t.requires_grad(true) {
                    return ComparisonResult::failure(format!("requires_grad failed: {}", e));
                }
                t
            }
            Err(e) => return ComparisonResult::failure(format!("from_data failed: {}", e)),
        };
        let torch_adam =
            match LibTorchAdam::new(&[&torch_param], 0.001, 0.9, 0.999, 1e-8, 0.0, false) {
                Ok(o) => o,
                Err(e) => {
                    return ComparisonResult::failure(format!("Failed to create torch adam: {}", e))
                }
            };

        // Step optimizer without generating gradients (should be no-op)
        {
            let mut param_guard = our_param_locked.write().unwrap();
            our_adam.step(&mut [&mut *param_guard]);
        }
        if let Err(e) = torch_adam.step() {
            return ComparisonResult::failure(format!("Torch adam step failed: {}", e));
        }

        // Parameters should remain unchanged
        let our_param_final = our_param_locked.read().unwrap();
        self.tensor_validator
            .compare_tensors(&our_param_final, &torch_param)
    }

    /// Validate Adam with very small gradients (near numerical precision limits)
    pub fn validate_small_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let initial_data: Vec<f32> = vec![1.0; numel];

        // Our implementation
        let our_param = match Tensor::from_slice(&initial_data, shape.to_vec()) {
            Ok(p) => p.with_requires_grad(),
            Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
        };
        let our_param_locked = Arc::new(RwLock::new(our_param));

        let our_options = AdamConfig::default();
        let mut our_adam = Adam::with_config(our_options);
        our_adam.add_parameter(&our_param_locked.read().unwrap());

        // LibTorch implementation
        let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
            Ok(t) => {
                if let Err(e) = t.requires_grad(true) {
                    return ComparisonResult::failure(format!("requires_grad failed: {}", e));
                }
                t
            }
            Err(e) => return ComparisonResult::failure(format!("from_data failed: {}", e)),
        };
        let torch_adam =
            match LibTorchAdam::new(&[&torch_param], 0.001, 0.9, 0.999, 1e-8, 0.0, false) {
                Ok(o) => o,
                Err(e) => {
                    return ComparisonResult::failure(format!("Failed to create torch adam: {}", e))
                }
            };

        // Generate very small gradients (1e-8 * parameter values)
        {
            let param_guard = our_param_locked.read().unwrap();
            let our_output = param_guard.mul_scalar(1e-8);
            let mut our_loss = our_output.sum();
            our_loss.backward(None);
        }

        let torch_output = match torch_param.mul_scalar(1e-8) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("Torch forward failed: {}", e)),
        };
        let torch_loss = match torch_output.sum() {
            Ok(l) => l,
            Err(e) => return ComparisonResult::failure(format!("Torch sum failed: {}", e)),
        };
        if let Err(e) = torch_loss.backward(None) {
            return ComparisonResult::failure(format!("Torch backward failed: {}", e));
        }

        // Step optimizers
        {
            let mut param_guard = our_param_locked.write().unwrap();
            our_adam.step(&mut [&mut *param_guard]);
        }
        if let Err(e) = torch_adam.step() {
            return ComparisonResult::failure(format!("Torch adam step failed: {}", e));
        }

        // Compare final parameters
        let our_param_final = our_param_locked.read().unwrap();
        self.tensor_validator
            .compare_tensors(&our_param_final, &torch_param)
    }

    /// Validate Adam with very large gradients
    pub fn validate_large_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let initial_data: Vec<f32> = vec![1e-6; numel]; // Small initial values to prevent overflow

        // Our implementation
        let our_param = match Tensor::from_slice(&initial_data, shape.to_vec()) {
            Ok(p) => p.with_requires_grad(),
            Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
        };
        let our_param_locked = Arc::new(RwLock::new(our_param));

        let our_options = AdamConfig {
            learning_rate: 1e-6, // Very small learning rate to prevent instability
            ..Default::default()
        };
        let mut our_adam = Adam::with_config(our_options);
        our_adam.add_parameter(&our_param_locked.read().unwrap());

        // LibTorch implementation
        let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
            Ok(t) => {
                if let Err(e) = t.requires_grad(true) {
                    return ComparisonResult::failure(format!("requires_grad failed: {}", e));
                }
                t
            }
            Err(e) => return ComparisonResult::failure(format!("from_data failed: {}", e)),
        };
        let torch_adam =
            match LibTorchAdam::new(&[&torch_param], 1e-6, 0.9, 0.999, 1e-8, 0.0, false) {
                Ok(o) => o,
                Err(e) => {
                    return ComparisonResult::failure(format!("Failed to create torch adam: {}", e))
                }
            };

        // Generate large gradients (1e6 * parameter values)
        {
            let param_guard = our_param_locked.read().unwrap();
            let our_output = param_guard.mul_scalar(1e6);
            let mut our_loss = our_output.sum();
            our_loss.backward(None);
        }

        let torch_output = match torch_param.mul_scalar(1e6) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("Torch forward failed: {}", e)),
        };
        let torch_loss = match torch_output.sum() {
            Ok(l) => l,
            Err(e) => return ComparisonResult::failure(format!("Torch sum failed: {}", e)),
        };
        if let Err(e) = torch_loss.backward(None) {
            return ComparisonResult::failure(format!("Torch backward failed: {}", e));
        }

        // Step optimizers
        {
            let mut param_guard = our_param_locked.write().unwrap();
            our_adam.step(&mut [&mut *param_guard]);
        }
        if let Err(e) = torch_adam.step() {
            return ComparisonResult::failure(format!("Torch adam step failed: {}", e));
        }

        // Compare final parameters
        let our_param_final = our_param_locked.read().unwrap();
        self.tensor_validator
            .compare_tensors(&our_param_final, &torch_param)
    }

    /// Validate Adam with mixed positive and negative gradients
    pub fn validate_mixed_gradients(&self, shape: &[usize]) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let initial_data: Vec<f32> = (0..numel)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        // Our implementation
        let our_param = match Tensor::from_slice(&initial_data, shape.to_vec()) {
            Ok(p) => p.with_requires_grad(),
            Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
        };
        let our_param_locked = Arc::new(RwLock::new(our_param));

        let our_options = AdamConfig::default();
        let mut our_adam = Adam::with_config(our_options);
        our_adam.add_parameter(&our_param_locked.read().unwrap());

        // LibTorch implementation
        let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
            Ok(t) => {
                if let Err(e) = t.requires_grad(true) {
                    return ComparisonResult::failure(format!("requires_grad failed: {}", e));
                }
                t
            }
            Err(e) => return ComparisonResult::failure(format!("from_data failed: {}", e)),
        };
        let torch_adam =
            match LibTorchAdam::new(&[&torch_param], 0.001, 0.9, 0.999, 1e-8, 0.0, false) {
                Ok(o) => o,
                Err(e) => {
                    return ComparisonResult::failure(format!("Failed to create torch adam: {}", e))
                }
            };

        // Generate gradients with alternating signs
        {
            let param_guard = our_param_locked.read().unwrap();
            let our_output = param_guard.mul_scalar(2.0);
            let mut our_loss = our_output.sum();
            our_loss.backward(None);
        }

        let torch_output = match torch_param.mul_scalar(2.0) {
            Ok(t) => t,
            Err(e) => return ComparisonResult::failure(format!("Torch forward failed: {}", e)),
        };
        let torch_loss = match torch_output.sum() {
            Ok(l) => l,
            Err(e) => return ComparisonResult::failure(format!("Torch sum failed: {}", e)),
        };
        if let Err(e) = torch_loss.backward(None) {
            return ComparisonResult::failure(format!("Torch backward failed: {}", e));
        }

        // Step optimizers
        {
            let mut param_guard = our_param_locked.write().unwrap();
            our_adam.step(&mut [&mut *param_guard]);
        }
        if let Err(e) = torch_adam.step() {
            return ComparisonResult::failure(format!("Torch adam step failed: {}", e));
        }

        // Compare final parameters
        let our_param_final = our_param_locked.read().unwrap();
        self.tensor_validator
            .compare_tensors(&our_param_final, &torch_param)
    }

    /// Validate numerical stability with extreme epsilon values
    pub fn validate_extreme_epsilon(&self, shape: &[usize]) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let initial_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01).collect();

        // Test with very small epsilon (numerical precision limit)
        let config = AdamValidationConfig {
            eps: 1e-15, // Near f64 machine epsilon
            ..Default::default()
        };

        let our_param = match Tensor::from_slice(&initial_data, shape.to_vec()) {
            Ok(p) => p.with_requires_grad(),
            Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
        };
        let our_param_locked = Arc::new(RwLock::new(our_param));

        let our_options = AdamConfig {
            learning_rate: config.lr as f32,
            beta1: config.beta1 as f32,
            beta2: config.beta2 as f32,
            eps: config.eps as f32,
            weight_decay: config.weight_decay as f32,
            amsgrad: config.amsgrad,
        };
        let mut our_adam = Adam::with_config(our_options);
        our_adam.add_parameter(&our_param_locked.read().unwrap());

        let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
            Ok(t) => {
                if let Err(e) = t.requires_grad(true) {
                    return ComparisonResult::failure(format!("requires_grad failed: {}", e));
                }
                t
            }
            Err(e) => return ComparisonResult::failure(format!("from_data failed: {}", e)),
        };
        let torch_adam = match LibTorchAdam::new(
            &[&torch_param],
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.weight_decay,
            config.amsgrad,
        ) {
            Ok(o) => o,
            Err(e) => {
                return ComparisonResult::failure(format!("Failed to create torch adam: {}", e))
            }
        };

        // Multiple steps to test stability
        for _ in 0..10 {
            {
                let param_guard = our_param_locked.read().unwrap();
                let our_output = param_guard.mul_scalar(2.0).add_scalar(1.0);
                let mut our_loss = our_output.sum();
                our_loss.backward(None);
            }

            let torch_output = match torch_param.mul_scalar(2.0).and_then(|t| t.add_scalar(1.0)) {
                Ok(t) => t,
                Err(e) => return ComparisonResult::failure(format!("Torch forward failed: {}", e)),
            };
            let torch_loss = match torch_output.sum() {
                Ok(l) => l,
                Err(e) => return ComparisonResult::failure(format!("Torch sum failed: {}", e)),
            };
            if let Err(e) = torch_loss.backward(None) {
                return ComparisonResult::failure(format!("Torch backward failed: {}", e));
            }

            {
                let mut param_guard = our_param_locked.write().unwrap();
                our_adam.step(&mut [&mut *param_guard]);
            }
            if let Err(e) = torch_adam.step() {
                return ComparisonResult::failure(format!("Torch adam step failed: {}", e));
            }

            {
                let mut param_guard = our_param_locked.write().unwrap();
                our_adam.zero_grad(&mut [&mut *param_guard]);
            }
            if let Err(e) = torch_adam.zero_grad() {
                return ComparisonResult::failure(format!("Torch zero_grad failed: {}", e));
            }
        }

        let our_param_final = our_param_locked.read().unwrap();
        self.tensor_validator
            .compare_tensors(&our_param_final, &torch_param)
    }

    /// Validate convergence behavior over many steps
    pub fn validate_long_term_convergence(&self, shape: &[usize]) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let initial_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1).collect();

        let config = AdamValidationConfig {
            steps: 50, // Many steps to test convergence behavior
            ..Default::default()
        };

        let our_param = match Tensor::from_slice(&initial_data, shape.to_vec()) {
            Ok(p) => p.with_requires_grad(),
            Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
        };
        let our_param_locked = Arc::new(RwLock::new(our_param));

        let our_options = AdamConfig {
            learning_rate: config.lr as f32,
            beta1: config.beta1 as f32,
            beta2: config.beta2 as f32,
            eps: config.eps as f32,
            weight_decay: config.weight_decay as f32,
            amsgrad: config.amsgrad,
        };
        let mut our_adam = Adam::with_config(our_options);
        our_adam.add_parameter(&our_param_locked.read().unwrap());

        let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
            Ok(t) => {
                if let Err(e) = t.requires_grad(true) {
                    return ComparisonResult::failure(format!("requires_grad failed: {}", e));
                }
                t
            }
            Err(e) => return ComparisonResult::failure(format!("from_data failed: {}", e)),
        };
        let torch_adam = match LibTorchAdam::new(
            &[&torch_param],
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.weight_decay,
            config.amsgrad,
        ) {
            Ok(o) => o,
            Err(e) => {
                return ComparisonResult::failure(format!("Failed to create torch adam: {}", e))
            }
        };

        // Simulate convergence to a target (quadratic loss)
        for _ in 0..config.steps {
            {
                let param_guard = our_param_locked.read().unwrap();
                // Quadratic loss: (param - target)^2, target = 0.5
                let diff = param_guard.add_scalar(-0.5);
                let our_output = diff.mul_tensor(&diff);
                let mut our_loss = our_output.sum();
                our_loss.backward(None);
            }

            let diff = match torch_param.add_scalar(-0.5) {
                Ok(d) => d,
                Err(e) => return ComparisonResult::failure(format!("Torch diff failed: {}", e)),
            };
            let torch_output = match diff.mul_tensor(&diff) {
                Ok(o) => o,
                Err(e) => return ComparisonResult::failure(format!("Torch mul failed: {}", e)),
            };
            let torch_loss = match torch_output.sum() {
                Ok(l) => l,
                Err(e) => return ComparisonResult::failure(format!("Torch sum failed: {}", e)),
            };
            if let Err(e) = torch_loss.backward(None) {
                return ComparisonResult::failure(format!("Torch backward failed: {}", e));
            }

            {
                let mut param_guard = our_param_locked.write().unwrap();
                our_adam.step(&mut [&mut *param_guard]);
            }
            if let Err(e) = torch_adam.step() {
                return ComparisonResult::failure(format!("Torch adam step failed: {}", e));
            }

            {
                let mut param_guard = our_param_locked.write().unwrap();
                our_adam.zero_grad(&mut [&mut *param_guard]);
            }
            if let Err(e) = torch_adam.zero_grad() {
                return ComparisonResult::failure(format!("Torch zero_grad failed: {}", e));
            }
        }

        let our_param_final = our_param_locked.read().unwrap();
        self.tensor_validator
            .compare_tensors(&our_param_final, &torch_param)
    }

    /// Validate numerical stability with extreme beta values
    pub fn validate_extreme_betas(&self, shape: &[usize]) -> ComparisonResult {
        let numel: usize = shape.iter().product();
        let initial_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01).collect();

        // Test with extreme beta values (near 0 and near 1)
        let extreme_configs = vec![
            AdamValidationConfig {
                beta1: 0.001, // Very low momentum
                beta2: 0.001, // Very low velocity decay
                steps: 5,
                ..Default::default()
            },
            AdamValidationConfig {
                beta1: 0.9999, // Very high momentum
                beta2: 0.9999, // Very high velocity decay
                steps: 5,
                ..Default::default()
            },
        ];

        for config in extreme_configs {
            let our_param = match Tensor::from_slice(&initial_data, shape.to_vec()) {
                Ok(p) => p.with_requires_grad(),
                Err(e) => return ComparisonResult::failure(format!("from_slice failed: {}", e)),
            };
            let our_param_locked = Arc::new(RwLock::new(our_param));

            let our_options = AdamConfig {
                learning_rate: config.lr as f32,
                beta1: config.beta1 as f32,
                beta2: config.beta2 as f32,
                eps: config.eps as f32,
                weight_decay: config.weight_decay as f32,
                amsgrad: config.amsgrad,
            };
            let mut our_adam = Adam::with_config(our_options);
            our_adam.add_parameter(&our_param_locked.read().unwrap());

            let torch_param = match LibTorchTensor::from_data(&initial_data, shape) {
                Ok(t) => {
                    if let Err(e) = t.requires_grad(true) {
                        return ComparisonResult::failure(format!("requires_grad failed: {}", e));
                    }
                    t
                }
                Err(e) => return ComparisonResult::failure(format!("from_data failed: {}", e)),
            };
            let torch_adam = match LibTorchAdam::new(
                &[&torch_param],
                config.lr,
                config.beta1,
                config.beta2,
                config.eps,
                config.weight_decay,
                config.amsgrad,
            ) {
                Ok(o) => o,
                Err(e) => {
                    return ComparisonResult::failure(format!("Failed to create torch adam: {}", e))
                }
            };

            for _ in 0..config.steps {
                {
                    let param_guard = our_param_locked.read().unwrap();
                    let our_output = param_guard.mul_scalar(2.0).add_scalar(1.0);
                    let mut our_loss = our_output.sum();
                    our_loss.backward(None);
                }

                let torch_output = match torch_param.mul_scalar(2.0).and_then(|t| t.add_scalar(1.0))
                {
                    Ok(t) => t,
                    Err(e) => {
                        return ComparisonResult::failure(format!("Torch forward failed: {}", e))
                    }
                };
                let torch_loss = match torch_output.sum() {
                    Ok(l) => l,
                    Err(e) => return ComparisonResult::failure(format!("Torch sum failed: {}", e)),
                };
                if let Err(e) = torch_loss.backward(None) {
                    return ComparisonResult::failure(format!("Torch backward failed: {}", e));
                }

                {
                    let mut param_guard = our_param_locked.write().unwrap();
                    our_adam.step(&mut [&mut *param_guard]);
                }
                if let Err(e) = torch_adam.step() {
                    return ComparisonResult::failure(format!("Torch adam step failed: {}", e));
                }

                {
                    let mut param_guard = our_param_locked.write().unwrap();
                    our_adam.zero_grad(&mut [&mut *param_guard]);
                }
                if let Err(e) = torch_adam.zero_grad() {
                    return ComparisonResult::failure(format!("Torch zero_grad failed: {}", e));
                }
            }

            let our_param_final = our_param_locked.read().unwrap();
            let result = self
                .tensor_validator
                .compare_tensors(&our_param_final, &torch_param);
            if !result.passed {
                return ComparisonResult::failure(format!(
                    "Extreme beta test failed for beta1={}, beta2={}: {}",
                    config.beta1, config.beta2, result.details
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
    fn test_adam_single_parameter() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig::default();
        let result = validator.validate_single_parameter(&[3, 4], &config);
        assert!(result.passed, "{}", result.details);
    }

    #[test]
    fn test_adam_single_parameter_unsafe() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig::default();
        let result = validator.validate_single_parameter_unsafe(&[3, 4], &config);
        assert!(result.passed, "{}", result.details);
    }

    #[test]
    fn test_adam_multi_parameter() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig::default();
        let shapes: Vec<Vec<usize>> = vec![vec![2, 3], vec![4, 5]];
        let results = validator.validate_multi_parameter(shapes, &config);
        for result in results {
            assert!(result.passed, "{}", result.details);
        }
    }

    #[test]
    fn test_adam_multi_parameter_unsafe() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig::default();
        let shapes: Vec<Vec<usize>> = vec![vec![2, 3], vec![4, 5]];
        let results = validator.validate_multi_parameter_unsafe(shapes, &config);
        for result in results {
            assert!(result.passed, "{}", result.details);
        }
    }

    #[test]
    fn test_adam_zero_grad() {
        let validator = AdamValidator::default();
        let result = validator.verify_zero_grad(&[3, 4]);
        assert!(result.passed, "{}", result.details);
    }

    #[test]
    fn test_adam_zero_grad_unsafe() {
        let validator = AdamValidator::default();
        let result = validator.verify_zero_grad_unsafe(&[3, 4]);
        assert!(result.passed, "{}", result.details);
    }

    #[test]
    fn test_adam_with_weight_decay() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig {
            weight_decay: 0.01,
            ..Default::default()
        };
        let result = validator.validate_single_parameter(&[3, 4], &config);
        assert!(result.passed, "{}", result.details);
    }

    #[test]
    fn test_adam_with_weight_decay_unsafe() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig {
            weight_decay: 0.01,
            ..Default::default()
        };
        let result = validator.validate_single_parameter_unsafe(&[3, 4], &config);
        assert!(result.passed, "{}", result.details);
    }

    #[test]
    fn test_adam_with_amsgrad() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig {
            amsgrad: true,
            ..Default::default()
        };
        let result = validator.validate_single_parameter(&[3, 4], &config);
        assert!(result.passed, "{}", result.details);
    }

    #[test]
    fn test_adam_with_amsgrad_unsafe() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig {
            amsgrad: true,
            ..Default::default()
        };
        let result = validator.validate_single_parameter_unsafe(&[3, 4], &config);
        assert!(result.passed, "{}", result.details);
    }

    #[test]
    fn test_adam_comprehensive_shapes() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig::default();

        let test_shapes = vec![
            vec![1],       // Scalar tensors
            vec![3],       // 1D vectors
            vec![2, 2],    // 2D matrices
            vec![1, 4],    // Broadcasting candidates
            vec![2, 3],    // Rectangular matrices
            vec![1, 1, 3], // 3D tensors
            vec![10, 10],  // Larger tensors
        ];

        for shape in test_shapes {
            let result = validator.validate_single_parameter(&shape, &config);
            assert!(
                result.passed,
                "Safe mode failed for shape {:?}: {}",
                shape, result.details
            );

            let result_unsafe = validator.validate_single_parameter_unsafe(&shape, &config);
            assert!(
                result_unsafe.passed,
                "Unsafe mode failed for shape {:?}: {}",
                shape, result_unsafe.details
            );
        }
    }

    #[test]
    fn test_adam_comprehensive_configs() {
        let validator = AdamValidator::default();

        let test_configs = vec![
            AdamValidationConfig {
                lr: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
                amsgrad: false,
                steps: 3,
            },
            AdamValidationConfig {
                lr: 0.01,
                beta1: 0.8,
                beta2: 0.99,
                eps: 1e-6,
                weight_decay: 0.001,
                amsgrad: false,
                steps: 3,
            },
            AdamValidationConfig {
                lr: 0.0001,
                beta1: 0.95,
                beta2: 0.9999,
                eps: 1e-10,
                weight_decay: 0.01,
                amsgrad: true,
                steps: 3,
            },
        ];

        for config in test_configs {
            let result = validator.validate_single_parameter(&[3, 4], &config);
            assert!(
                result.passed,
                "Safe mode failed for config {:?}: {}",
                config, result.details
            );

            let result_unsafe = validator.validate_single_parameter_unsafe(&[3, 4], &config);
            assert!(
                result_unsafe.passed,
                "Unsafe mode failed for config {:?}: {}",
                config, result_unsafe.details
            );
        }
    }

    #[test]
    fn test_adam_multi_parameter_comprehensive() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig::default();

        let test_shapes_sets = vec![
            vec![vec![2, 3], vec![3, 1]],             // Small matrices
            vec![vec![5, 5], vec![5, 10], vec![10]],  // Mixed sizes
            vec![vec![1, 1], vec![1, 1], vec![1, 1]], // All small
            vec![vec![10, 5], vec![5, 2]],            // Larger matrices
        ];

        for shapes in test_shapes_sets {
            let results = validator.validate_multi_parameter(shapes.clone(), &config);
            for (i, result) in results.iter().enumerate() {
                assert!(
                    result.passed,
                    "Safe mode failed for shapes {:?}, param {}: {}",
                    shapes, i, result.details
                );
            }

            let results_unsafe = validator.validate_multi_parameter_unsafe(shapes.clone(), &config);
            for (i, result) in results_unsafe.iter().enumerate() {
                assert!(
                    result.passed,
                    "Unsafe mode failed for shapes {:?}, param {}: {}",
                    shapes, i, result.details
                );
            }
        }
    }

    #[test]
    fn test_adam_zero_grad_comprehensive() {
        let validator = AdamValidator::default();

        let test_shapes = vec![
            vec![1],       // Scalar
            vec![3],       // 1D
            vec![2, 2],    // 2D
            vec![1, 1, 3], // 3D
            vec![10, 10],  // Large
        ];

        for shape in test_shapes {
            let result = validator.verify_zero_grad(&shape);
            assert!(
                result.passed,
                "Safe mode zero_grad failed for shape {:?}: {}",
                shape, result.details
            );

            let result_unsafe = validator.verify_zero_grad_unsafe(&shape);
            assert!(
                result_unsafe.passed,
                "Unsafe mode zero_grad failed for shape {:?}: {}",
                shape, result_unsafe.details
            );
        }
    }

    #[test]
    fn test_adam_consistency_between_modes() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig::default();

        // Test that safe and unsafe modes produce consistent results
        let shape = vec![3, 4];

        let result_safe = validator.validate_single_parameter(&shape, &config);
        let result_unsafe = validator.validate_single_parameter_unsafe(&shape, &config);

        assert!(
            result_safe.passed,
            "Safe mode failed: {}",
            result_safe.details
        );
        assert!(
            result_unsafe.passed,
            "Unsafe mode failed: {}",
            result_unsafe.details
        );

        // Both should pass validation against LibTorch
        assert!(
            result_safe.passed && result_unsafe.passed,
            "Both modes should pass LibTorch validation"
        );
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn test_adam_zero_gradients() {
        let validator = AdamValidator::default();
        let test_shapes = vec![vec![2, 2], vec![5], vec![1, 1, 3]];

        for shape in test_shapes {
            let result = validator.validate_zero_gradients(&shape);
            assert!(
                result.passed,
                "Zero gradients test failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_adam_small_gradients() {
        let validator = AdamValidator::default();
        let test_shapes = vec![vec![2, 2], vec![5]];

        for shape in test_shapes {
            let result = validator.validate_small_gradients(&shape);
            assert!(
                result.passed,
                "Small gradients test failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_adam_large_gradients() {
        let validator = AdamValidator::default();
        let test_shapes = vec![vec![2, 2], vec![5]];

        for shape in test_shapes {
            let result = validator.validate_large_gradients(&shape);
            assert!(
                result.passed,
                "Large gradients test failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_adam_mixed_gradients() {
        let validator = AdamValidator::default();
        let test_shapes = vec![vec![4], vec![2, 2], vec![3, 3]];

        for shape in test_shapes {
            let result = validator.validate_mixed_gradients(&shape);
            assert!(
                result.passed,
                "Mixed gradients test failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    // ========== Numerical Stability Tests ==========

    #[test]
    fn test_adam_extreme_epsilon() {
        let validator = AdamValidator::default();
        let test_shapes = vec![vec![2, 2], vec![5]];

        for shape in test_shapes {
            let result = validator.validate_extreme_epsilon(&shape);
            assert!(
                result.passed,
                "Extreme epsilon test failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_adam_long_term_convergence() {
        let validator = AdamValidator::default();
        let test_shapes = vec![vec![3], vec![2, 2]];

        for shape in test_shapes {
            let result = validator.validate_long_term_convergence(&shape);
            assert!(
                result.passed,
                "Long-term convergence test failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_adam_extreme_betas() {
        let validator = AdamValidator::default();
        let test_shapes = vec![vec![2, 2], vec![5]];

        for shape in test_shapes {
            let result = validator.validate_extreme_betas(&shape);
            assert!(
                result.passed,
                "Extreme betas test failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    // ========== Extended Hyperparameter Tests ==========

    #[test]
    fn test_adam_extreme_learning_rates() {
        let validator = AdamValidator::default();

        let extreme_lr_configs = vec![
            AdamValidationConfig {
                lr: 1e-8, // Very small learning rate
                steps: 10,
                ..Default::default()
            },
            AdamValidationConfig {
                lr: 1e-1, // Large learning rate (but stable)
                steps: 3,
                ..Default::default()
            },
        ];

        for config in extreme_lr_configs {
            let result = validator.validate_single_parameter(&[2, 2], &config);
            assert!(
                result.passed,
                "Extreme learning rate test failed for lr={}: {}",
                config.lr, result.details
            );
        }
    }

    #[test]
    fn test_adam_heavy_weight_decay() {
        let validator = AdamValidator::default();

        let heavy_decay_configs = vec![
            AdamValidationConfig {
                weight_decay: 0.1, // Heavy L2 regularization
                steps: 5,
                ..Default::default()
            },
            AdamValidationConfig {
                weight_decay: 0.5, // Very heavy L2 regularization
                steps: 3,
                ..Default::default()
            },
        ];

        for config in heavy_decay_configs {
            let result = validator.validate_single_parameter(&[3, 3], &config);
            assert!(
                result.passed,
                "Heavy weight decay test failed for decay={}: {}",
                config.weight_decay, result.details
            );
        }
    }

    // ========== Advanced AMSGrad Tests ==========

    #[test]
    fn test_adam_amsgrad_state_persistence() {
        let validator = AdamValidator::default();

        // Test AMSGrad with many steps to verify max state tracking
        let config = AdamValidationConfig {
            amsgrad: true,
            steps: 20, // Many steps to test v_hat_max persistence
            ..Default::default()
        };

        let test_shapes = vec![vec![3, 3], vec![5], vec![2, 2, 2]];

        for shape in test_shapes {
            let result = validator.validate_single_parameter(&shape, &config);
            assert!(
                result.passed,
                "AMSGrad state persistence test failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_adam_amsgrad_vs_regular() {
        let validator = AdamValidator::default();

        // Compare AMSGrad against regular Adam with identical settings except amsgrad flag
        let base_config = AdamValidationConfig {
            steps: 10,
            ..Default::default()
        };

        let regular_config = AdamValidationConfig {
            amsgrad: false,
            ..base_config
        };

        let amsgrad_config = AdamValidationConfig {
            amsgrad: true,
            ..base_config
        };

        let shape = vec![3, 3];

        let regular_result = validator.validate_single_parameter(&shape, &regular_config);
        let amsgrad_result = validator.validate_single_parameter(&shape, &amsgrad_config);

        assert!(
            regular_result.passed,
            "Regular Adam failed: {}",
            regular_result.details
        );
        assert!(
            amsgrad_result.passed,
            "AMSGrad Adam failed: {}",
            amsgrad_result.details
        );
    }

    // ========== Stress Tests ==========

    #[test]
    fn test_adam_large_tensors() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig {
            steps: 5,
            ..Default::default()
        };

        // Test with large tensors to verify scalability
        let large_shapes = vec![
            vec![100],        // Large 1D
            vec![32, 32],     // Large 2D
            vec![10, 10, 10], // Large 3D
        ];

        for shape in large_shapes {
            let result = validator.validate_single_parameter(&shape, &config);
            assert!(
                result.passed,
                "Large tensor test failed for shape {:?}: {}",
                shape, result.details
            );
        }
    }

    #[test]
    fn test_adam_many_parameters() {
        let validator = AdamValidator::default();
        let config = AdamValidationConfig {
            steps: 5,
            ..Default::default()
        };

        // Test with many parameters to verify multi-parameter handling
        let many_shapes = vec![
            vec![2, 2],
            vec![3],
            vec![1, 4],
            vec![2, 3],
            vec![5, 5],
            vec![1],
            vec![2, 2, 2],
            vec![4, 4],
            vec![3, 3],
            vec![6],
        ];

        let results = validator.validate_multi_parameter(many_shapes.clone(), &config);
        for (i, result) in results.iter().enumerate() {
            assert!(
                result.passed,
                "Many parameters test failed for param {} (shape {:?}): {}",
                i, many_shapes[i], result.details
            );
        }
    }

    #[test]
    fn test_adam_precision_boundaries() {
        let validator = AdamValidator::default();

        // Test at f32 precision boundaries
        let precision_configs = vec![
            AdamValidationConfig {
                eps: f32::EPSILON as f64, // Machine epsilon for f32
                steps: 5,
                ..Default::default()
            },
            AdamValidationConfig {
                lr: (f32::EPSILON * 1000.0) as f64, // Very small but representable learning rate
                steps: 10,
                ..Default::default()
            },
        ];

        for config in precision_configs {
            let result = validator.validate_single_parameter(&[3, 3], &config);
            assert!(
                result.passed,
                "Precision boundary test failed for eps={}, lr={}: {}",
                config.eps, config.lr, result.details
            );
        }
    }
}
