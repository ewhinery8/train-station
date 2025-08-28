//! Adam optimizer implementation for neural network training
//!
//! This module provides the Adam optimization algorithm with PyTorch-compatible interface.
//! Adam combines the benefits of AdaGrad and RMSprop by using adaptive learning rates
//! with momentum for efficient training of neural networks.
//!
//! # Features
//!
//! - **Adaptive Learning Rates**: Per-parameter learning rates based on gradient history
//! - **Momentum Integration**: Combines momentum with adaptive learning rates
//! - **Bias Correction**: Corrects for initialization bias in moment estimates
//! - **Weight Decay**: Optional L2 regularization support
//! - **AMSGrad Variant**: Optional AMSGrad for improved convergence stability
//! - **SIMD Optimization**: AVX2-optimized parameter updates for maximum performance
//! - **Thread Safety**: Send + Sync implementation for multi-threaded training
//! - **Hybrid API**: Both safe (RwLock-based) and unsafe (direct pointer) access patterns
//!
//! # Thread Safety
//!
//! The optimizer provides two usage patterns:
//!
//! **Safe Multi-threaded Usage (Default)**:
//! - Uses `Arc<RwLock<Tensor>>` for thread-safe parameter access
//! - Multiple threads can read tensors simultaneously
//! - Optimizer steps acquire write locks only during parameter updates
//! - Recommended for most use cases
//!
//! **Unsafe Single-threaded Usage (Performance)**:
//! - Uses raw pointers for maximum performance
//! - No locking overhead during optimizer steps
//! - Caller must ensure exclusive access during optimization
//! - Use only when you can guarantee no concurrent tensor access
//!
//! # Algorithm
//!
//! Adam implements the following update rule for each parameter theta:
//!
//! ```text
//! m_t = beta1 * m_{t-1} + (1 - beta1) * grad_theta_t
//! v_t = beta2 * v_{t-1} + (1 - beta2) * (grad_theta_t)^2
//! m_hat_t = m_t / (1 - beta1^t)
//! v_hat_t = v_t / (1 - beta2^t)
//! theta_{t+1} = theta_t - lr * m_hat_t / (sqrt(v_hat_t) + eps)
//! ```
//!
//! Where:
//! - lr is the learning rate
//! - beta1, beta2 are exponential decay rates for moment estimates
//! - eps is a small constant for numerical stability
//! - m_t, v_t are biased first and second moment estimates
//! - m_hat_t, v_hat_t are bias-corrected moment estimates
//!
//! # Performance Characteristics
//!
//! - **SIMD Optimization**: Uses AVX2 instructions for 8x vectorization when available
//! - **Memory Efficiency**: In-place updates with minimal temporary allocations
//! - **Cache-Friendly**: Optimized memory access patterns for large parameter tensors
//! - **Zero-Cost Abstractions**: Compile-time optimization with minimal runtime overhead
//! - **Lock-Free Reads**: RwLock allows concurrent tensor reads during training
//!
//! # References
//!
//! - Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
//! - PyTorch Adam implementation: <https://pytorch.org/docs/stable/generated/torch.optim.Adam.html>

pub mod serialization;

use super::Optimizer;
use crate::tensor::core::Tensor;
use std::collections::HashMap;

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Configuration for the Adam optimization algorithm
///
/// Contains all hyperparameters that control the behavior of Adam optimization.
/// Default values follow PyTorch conventions for maximum compatibility and
/// optimal convergence across a wide range of neural network architectures.
///
/// # Fields
///
/// * `learning_rate` - Step size for parameter updates (default: 1e-3)
/// * `beta1` - Exponential decay rate for first moment estimates (default: 0.9)
/// * `beta2` - Exponential decay rate for second moment estimates (default: 0.999)
/// * `eps` - Small constant for numerical stability (default: 1e-8)
/// * `weight_decay` - L2 regularization coefficient (default: 0.0)
/// * `amsgrad` - Whether to use AMSGrad variant for improved stability (default: false)
#[derive(Debug, Clone)]
pub struct AdamConfig {
    /// Learning rate for parameter updates (default: 1e-3)
    pub learning_rate: f32,
    /// Exponential decay rate for first moment estimates (default: 0.9)  
    pub beta1: f32,
    /// Exponential decay rate for second moment estimates (default: 0.999)
    pub beta2: f32,
    /// Small constant for numerical stability (default: 1e-8)
    pub eps: f32,
    /// Weight decay coefficient for L2 regularization (default: 0.0)
    pub weight_decay: f32,
    /// Whether to use AMSGrad variant (default: false)
    pub amsgrad: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        }
    }
}

/// Internal state tracking for a single parameter during Adam optimization
///
/// Stores the momentum and velocity buffers needed for Adam optimization,
/// along with step count for bias correction and optional AMSGrad state.
/// Memory layout is optimized for cache efficiency and SIMD operations.
///
/// # Fields
///
/// * `m` - First moment estimate (momentum buffer)
/// * `v` - Second moment estimate (velocity buffer)
/// * `v_hat_max` - Maximum of second moment estimates (for AMSGrad variant)
/// * `step` - Step count for bias correction calculations
#[derive(Debug)]
struct ParameterState {
    /// First moment estimate (momentum)
    m: Tensor,
    /// Second moment estimate (velocity)  
    v: Tensor,
    /// Maximum of second moment estimates (for AMSGrad)
    v_hat_max: Option<Tensor>,
    /// Step count for bias correction
    step: usize,
}

impl ParameterState {
    fn new(param_shape: &[usize]) -> Self {
        Self {
            m: Tensor::zeros(param_shape.to_vec()),
            v: Tensor::zeros(param_shape.to_vec()),
            v_hat_max: None,
            step: 0,
        }
    }
}

/// Adam optimizer for neural network parameter optimization
///
/// Implements the Adam optimization algorithm with PyTorch-compatible interface.
/// Provides adaptive learning rates with momentum for efficient training of neural networks.
/// The optimizer maintains per-parameter state for momentum and velocity estimates,
/// enabling adaptive learning rates that improve convergence across diverse architectures.
///
/// # Usage Pattern
///
/// The optimizer uses ID-based parameter linking for maximum flexibility and thread safety:
/// - Parameters are linked to the optimizer via `add_parameter` or `add_parameters`
/// - The `step` method takes mutable references to parameters for thread-safe updates
/// - Parameter states are maintained by tensor ID, allowing for dynamic parameter management
/// - Supports serialization and deserialization with parameter re-linking
///
/// # Dynamic Parameter Management
///
/// Parameters can be added, removed, or re-linked at runtime:
/// - `add_parameter`: Link a single parameter
/// - `add_parameters`: Link multiple parameters at once
/// - `unlink_parameter`: Remove parameter state by ID
/// - `clear_states`: Remove all parameter states
/// - `is_parameter_linked`: Check if a parameter is linked
///
/// # Serialization Support
///
/// The optimizer supports full serialization and deserialization with state preservation:
/// - Parameter states are saved with their shapes and insertion order for validation
/// - After deserialization, use `relink_parameters` to restore saved states to new tensors
/// - Parameters must be re-linked in the same chronological order they were originally added
/// - Shape validation ensures consistency between saved and current parameters
///
/// # Features
///
/// - **ID-Based Parameter Linking**: Dynamic parameter management via tensor IDs
/// - **Thread-Safe Step Method**: Takes mutable references for safe concurrent access
/// - **Per-Parameter State**: Each parameter maintains its own momentum and velocity buffers
/// - **Bias Correction**: Automatically corrects initialization bias in moment estimates
/// - **Weight Decay**: Optional L2 regularization with efficient implementation
/// - **AMSGrad Support**: Optional AMSGrad variant for improved convergence stability
/// - **SIMD Optimization**: AVX2-optimized updates for maximum performance
/// - **Full Serialization**: Complete state persistence and restoration
///
/// # Thread Safety
///
/// This type is thread-safe and can be shared between threads. The step method
/// takes mutable references to parameters, ensuring exclusive access during updates.
pub struct Adam {
    /// Optimizer configuration
    config: AdamConfig,
    /// Parameter states indexed by tensor ID
    states: HashMap<usize, ParameterState>,
    /// Current global step count
    step_count: usize,
    /// Order in which parameters were added (for serialization/re-linking)
    insertion_order: Vec<usize>,
}

impl Default for Adam {
    fn default() -> Self {
        Self::with_config(AdamConfig::default())
    }
}

impl Adam {
    /// Create a new Adam optimizer with default configuration
    ///
    /// Initializes an Adam optimizer with PyTorch-compatible default hyperparameters.
    /// Parameters must be linked separately using `add_parameter` or `add_parameters`.
    ///
    /// # Returns
    ///
    /// A new Adam optimizer instance with default hyperparameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new Adam optimizer with custom configuration
    ///
    /// Allows full control over all Adam hyperparameters for specialized training
    /// scenarios such as fine-tuning, transfer learning, or research applications.
    /// Parameters must be linked separately using `add_parameter` or `add_parameters`.
    ///
    /// # Arguments
    ///
    /// * `config` - Adam configuration with custom hyperparameters
    ///
    /// # Returns
    ///
    /// A new Adam optimizer instance with the specified configuration
    pub fn with_config(config: AdamConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
            step_count: 0,
            insertion_order: Vec::new(),
        }
    }

    /// Create a new Adam optimizer with custom learning rate
    ///
    /// A convenience constructor that allows setting only the learning rate while
    /// using default values for all other hyperparameters. Parameters must be
    /// linked separately using `add_parameter` or `add_parameters`.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for optimization
    ///
    /// # Returns
    ///
    /// A new Adam optimizer instance with the specified learning rate and default
    /// values for all other hyperparameters
    pub fn with_learning_rate(learning_rate: f32) -> Self {
        let config = AdamConfig {
            learning_rate,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Add a single parameter to the optimizer
    ///
    /// Links a parameter to the optimizer by creating a new parameter state
    /// indexed by the tensor's ID. The parameter must have `requires_grad` set to true.
    ///
    /// # Arguments
    ///
    /// * `parameter` - Reference to the tensor to link
    ///
    /// # Panics
    ///
    /// Panics if the parameter does not have `requires_grad` set to true
    pub fn add_parameter(&mut self, parameter: &Tensor) {
        assert!(
            parameter.requires_grad(),
            "Parameter must require gradients"
        );

        let param_id = parameter.id();
        let param_shape = parameter.shape().dims.clone();

        // Initialize state for this parameter if not already present
        use std::collections::hash_map::Entry;
        if let Entry::Vacant(entry) = self.states.entry(param_id) {
            entry.insert(ParameterState::new(&param_shape));
            self.insertion_order.push(param_id);
        }
    }

    /// Add multiple parameters to the optimizer
    ///
    /// Links multiple parameters to the optimizer by creating parameter states
    /// indexed by each tensor's ID. All parameters must have `requires_grad` set to true.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Slice of references to tensors to link
    ///
    /// # Panics
    ///
    /// Panics if any parameter does not have `requires_grad` set to true
    pub fn add_parameters(&mut self, parameters: &[&Tensor]) {
        for parameter in parameters {
            self.add_parameter(parameter);
        }
    }

    /// Remove a parameter from the optimizer
    ///
    /// Unlinks a parameter by removing its state from the optimizer.
    /// The parameter ID is used for identification.
    ///
    /// # Arguments
    ///
    /// * `parameter` - Reference to the tensor to unlink
    ///
    /// # Returns
    ///
    /// True if the parameter was linked and removed, false if it was not linked
    pub fn unlink_parameter(&mut self, parameter: &Tensor) -> bool {
        let param_id = parameter.id();
        let was_linked = self.states.remove(&param_id).is_some();
        if was_linked {
            self.insertion_order.retain(|&id| id != param_id);
        }
        was_linked
    }

    /// Remove all parameter states from the optimizer
    ///
    /// Clears all parameter states, effectively unlinking all parameters.
    /// This is useful for resetting the optimizer or preparing for parameter re-linking.
    pub fn clear_states(&mut self) {
        self.states.clear();
        self.insertion_order.clear();
    }

    /// Check if a parameter is linked to the optimizer
    ///
    /// Returns true if the parameter has an associated state in the optimizer.
    ///
    /// # Arguments
    ///
    /// * `parameter` - Reference to the tensor to check
    ///
    /// # Returns
    ///
    /// True if the parameter is linked, false otherwise
    pub fn is_parameter_linked(&self, parameter: &Tensor) -> bool {
        let param_id = parameter.id();
        self.states.contains_key(&param_id)
    }

    /// Get the number of linked parameters
    ///
    /// Returns the count of parameters currently linked to the optimizer.
    ///
    /// # Returns
    ///
    /// Number of linked parameters
    pub fn parameter_count(&self) -> usize {
        self.states.len()
    }

    /// Re-link parameters to saved optimizer states in chronological order
    ///
    /// After deserializing an optimizer, use this method to restore saved parameter states
    /// to new tensors. Parameters must be provided in the same chronological order they
    /// were originally added to the optimizer. Shape validation ensures parameter compatibility.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Slice of parameter references in chronological order
    ///
    /// # Returns
    ///
    /// Result indicating success or failure with detailed error message
    ///
    /// # Panics
    ///
    /// Panics if any parameter does not have `requires_grad` set to true
    pub fn relink_parameters(&mut self, parameters: &[&Tensor]) -> Result<(), String> {
        // Validate all parameters have requires_grad first
        for (i, param) in parameters.iter().enumerate() {
            if !param.requires_grad() {
                return Err(format!("Parameter at index {} must require gradients", i));
            }
        }

        // Check parameter count matches saved states
        if parameters.len() != self.insertion_order.len() {
            return Err(format!(
                "Parameter count mismatch: expected {} parameters, got {}",
                self.insertion_order.len(),
                parameters.len()
            ));
        }

        // Create new states map with parameter IDs mapped to saved states in chronological order
        let mut new_states = HashMap::new();
        let mut new_insertion_order = Vec::new();

        for (i, param) in parameters.iter().enumerate() {
            let new_param_id = param.id();
            let old_param_id = self.insertion_order[i];

            // Get the saved state for this position
            let saved_state = self
                .states
                .get(&old_param_id)
                .ok_or_else(|| format!("No saved state found for parameter at position {}", i))?;

            // Validate shape matches
            let param_shape = &param.shape().dims;
            let saved_shape = &saved_state.m.shape().dims;
            if param_shape != saved_shape {
                return Err(format!(
                    "Shape mismatch for parameter at position {}: expected {:?}, got {:?}",
                    i, saved_shape, param_shape
                ));
            }

            // Create new state for this parameter
            let new_state = ParameterState {
                m: saved_state.m.clone(),
                v: saved_state.v.clone(),
                v_hat_max: saved_state.v_hat_max.clone(),
                step: saved_state.step,
            };

            new_states.insert(new_param_id, new_state);
            new_insertion_order.push(new_param_id);
        }

        // Replace the states and insertion order
        self.states = new_states;
        self.insertion_order = new_insertion_order;

        Ok(())
    }

    /// Get the current optimizer configuration
    ///
    /// Returns a reference to the current configuration, allowing inspection
    /// of all hyperparameters without modification.
    ///
    /// # Returns
    ///
    /// Reference to the current Adam configuration
    pub fn config(&self) -> &AdamConfig {
        &self.config
    }

    /// Update a single parameter using Adam algorithm
    ///
    /// Implements the core Adam update rule with bias correction and optional AMSGrad.
    /// Uses SIMD optimization when available for improved performance.
    /// The parameter must be linked to the optimizer before calling this method.
    ///
    /// # Arguments
    ///
    /// * `param` - Mutable reference to the parameter tensor to update
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the parameter update
    ///
    /// # Panics
    ///
    /// Panics if the parameter is not linked to the optimizer
    fn update_parameter(&mut self, param: &mut Tensor) -> Result<(), String> {
        let param_id = param.id();

        // Ensure parameter is linked
        assert!(
            self.states.contains_key(&param_id),
            "Parameter must be linked to optimizer before stepping. Use add_parameter() first."
        );

        // Get parameter gradient
        let grad = param
            .grad_by_value()
            .ok_or_else(|| format!("Parameter {} has no gradient", param_id))?;

        // Get parameter state
        let state = self
            .states
            .get_mut(&param_id)
            .expect("Parameter state should exist after link check");

        // Increment step count
        state.step += 1;
        let step = self.step_count as f32; // Use global step count for bias correction

        // Apply weight decay if enabled
        let effective_grad = if self.config.weight_decay > 0.0 {
            // L2 regularization: grad + weight_decay * param
            let mut grad_with_decay = grad.clone();
            Self::add_weight_decay(&mut grad_with_decay, param, self.config.weight_decay);
            grad_with_decay
        } else {
            grad
        };

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
        Self::update_momentum(&mut state.m, &effective_grad, self.config.beta1);

        // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
        Self::update_velocity(&mut state.v, &effective_grad, self.config.beta2);

        // Compute bias-corrected first moment estimate
        let bias_correction1 = 1.0 - (self.config.beta1 as f64).powf(step as f64);
        let m_hat = Self::bias_correct(&state.m, bias_correction1 as f32);

        // Compute bias-corrected second moment estimate
        let bias_correction2 = 1.0 - (self.config.beta2 as f64).powf(step as f64);
        let mut v_hat = Self::bias_correct(&state.v, bias_correction2 as f32);

        // AMSGrad: use maximum of v_hat over time
        if self.config.amsgrad {
            if state.v_hat_max.is_none() {
                state.v_hat_max = Some(v_hat.clone());
            }
            let v_hat_max = state.v_hat_max.as_mut().unwrap();
            Self::element_wise_max(v_hat_max, &v_hat);
            v_hat = v_hat_max.clone();
        }

        // Compute parameter update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
        Self::apply_adam_update(
            param,
            &m_hat,
            &v_hat,
            self.config.learning_rate,
            self.config.eps,
        );

        Ok(())
    }

    /// Apply weight decay (L2 regularization) to gradient
    ///
    /// Adds `weight_decay * param` to the gradient in-place for memory efficiency.
    /// This implements L2 regularization by modifying the gradient before the
    /// Adam update step, equivalent to adding a regularization term to the loss.
    ///
    /// # Arguments
    ///
    /// * `grad` - Gradient tensor to modify in-place
    /// * `param` - Parameter tensor for weight decay calculation
    /// * `weight_decay` - Weight decay coefficient
    #[inline]
    fn add_weight_decay(grad: &mut Tensor, param: &Tensor, weight_decay: f32) {
        assert_eq!(
            grad.size(),
            param.size(),
            "Gradient and parameter size mismatch"
        );

        unsafe {
            let grad_ptr = grad.as_mut_ptr();
            let param_ptr = param.as_ptr();
            let size = grad.size();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2")
                    && grad.is_simd_aligned()
                    && param.is_simd_aligned()
                {
                    Self::add_weight_decay_simd_avx2(grad_ptr, param_ptr, weight_decay, size);
                    return;
                }
            }

            // Scalar fallback
            for i in 0..size {
                *grad_ptr.add(i) += weight_decay * *param_ptr.add(i);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn add_weight_decay_simd_avx2(
        grad_ptr: *mut f32,
        param_ptr: *const f32,
        weight_decay: f32,
        size: usize,
    ) {
        let decay_vec = _mm256_set1_ps(weight_decay);
        let simd_count = size / 8;

        for i in 0..simd_count {
            let offset = i * 8;
            let grad_vec = _mm256_loadu_ps(grad_ptr.add(offset));
            let param_vec = _mm256_loadu_ps(param_ptr.add(offset));
            let decay_term = _mm256_mul_ps(decay_vec, param_vec);
            let result = _mm256_add_ps(grad_vec, decay_term);
            _mm256_storeu_ps(grad_ptr.add(offset), result);
        }

        // Handle remaining elements
        for i in (simd_count * 8)..size {
            *grad_ptr.add(i) += weight_decay * *param_ptr.add(i);
        }
    }

    /// Update momentum (first moment estimate)
    ///
    /// Implements the momentum update rule: `m = beta1 * m + (1 - beta1) * grad`
    /// This computes the exponentially decaying average of gradients, providing
    /// momentum-like behavior that helps accelerate convergence in relevant directions.
    ///
    /// # Arguments
    ///
    /// * `momentum` - Momentum buffer to update in-place
    /// * `grad` - Current gradient tensor
    /// * `beta1` - Exponential decay rate for momentum
    #[inline]
    fn update_momentum(momentum: &mut Tensor, grad: &Tensor, beta1: f32) {
        assert_eq!(
            momentum.size(),
            grad.size(),
            "Momentum and gradient size mismatch"
        );

        let beta1_complement = 1.0 - beta1;

        unsafe {
            let m_ptr = momentum.as_mut_ptr();
            let grad_ptr = grad.as_ptr();
            let size = momentum.size();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2")
                    && momentum.is_simd_aligned()
                    && grad.is_simd_aligned()
                {
                    Self::update_momentum_simd_avx2(m_ptr, grad_ptr, beta1, beta1_complement, size);
                    return;
                }
            }

            // Scalar fallback
            for i in 0..size {
                *m_ptr.add(i) = beta1 * *m_ptr.add(i) + beta1_complement * *grad_ptr.add(i);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn update_momentum_simd_avx2(
        m_ptr: *mut f32,
        grad_ptr: *const f32,
        beta1: f32,
        beta1_complement: f32,
        size: usize,
    ) {
        let beta1_vec = _mm256_set1_ps(beta1);
        let beta1_comp_vec = _mm256_set1_ps(beta1_complement);
        let simd_count = size / 8;

        for i in 0..simd_count {
            let offset = i * 8;
            let m_vec = _mm256_loadu_ps(m_ptr.add(offset));
            let grad_vec = _mm256_loadu_ps(grad_ptr.add(offset));

            let momentum_term = _mm256_mul_ps(beta1_vec, m_vec);
            let gradient_term = _mm256_mul_ps(beta1_comp_vec, grad_vec);
            let result = _mm256_add_ps(momentum_term, gradient_term);

            _mm256_storeu_ps(m_ptr.add(offset), result);
        }

        // Handle remaining elements
        for i in (simd_count * 8)..size {
            *m_ptr.add(i) = beta1 * *m_ptr.add(i) + beta1_complement * *grad_ptr.add(i);
        }
    }

    /// Update velocity (second moment estimate)
    ///
    /// Implements the velocity update rule: `v = beta2 * v + (1 - beta2) * grad^2`
    /// This computes the exponentially decaying average of squared gradients,
    /// providing adaptive learning rates that scale inversely with gradient magnitude.
    ///
    /// # Arguments
    ///
    /// * `velocity` - Velocity buffer to update in-place
    /// * `grad` - Current gradient tensor
    /// * `beta2` - Exponential decay rate for velocity
    #[inline]
    fn update_velocity(velocity: &mut Tensor, grad: &Tensor, beta2: f32) {
        assert_eq!(
            velocity.size(),
            grad.size(),
            "Velocity and gradient size mismatch"
        );

        let beta2_complement = 1.0 - beta2;

        unsafe {
            let v_ptr = velocity.as_mut_ptr();
            let grad_ptr = grad.as_ptr();
            let size = velocity.size();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2")
                    && velocity.is_simd_aligned()
                    && grad.is_simd_aligned()
                {
                    Self::update_velocity_simd_avx2(v_ptr, grad_ptr, beta2, beta2_complement, size);
                    return;
                }
            }

            // Scalar fallback
            for i in 0..size {
                let grad_val = *grad_ptr.add(i);
                *v_ptr.add(i) = beta2 * *v_ptr.add(i) + beta2_complement * grad_val * grad_val;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn update_velocity_simd_avx2(
        v_ptr: *mut f32,
        grad_ptr: *const f32,
        beta2: f32,
        beta2_complement: f32,
        size: usize,
    ) {
        let beta2_vec = _mm256_set1_ps(beta2);
        let beta2_comp_vec = _mm256_set1_ps(beta2_complement);
        let simd_count = size / 8;

        for i in 0..simd_count {
            let offset = i * 8;
            let v_vec = _mm256_loadu_ps(v_ptr.add(offset));
            let grad_vec = _mm256_loadu_ps(grad_ptr.add(offset));

            let velocity_term = _mm256_mul_ps(beta2_vec, v_vec);
            let grad_squared = _mm256_mul_ps(grad_vec, grad_vec);
            let gradient_term = _mm256_mul_ps(beta2_comp_vec, grad_squared);
            let result = _mm256_add_ps(velocity_term, gradient_term);

            _mm256_storeu_ps(v_ptr.add(offset), result);
        }

        // Handle remaining elements
        for i in (simd_count * 8)..size {
            let grad_val = *grad_ptr.add(i);
            *v_ptr.add(i) = beta2 * *v_ptr.add(i) + beta2_complement * grad_val * grad_val;
        }
    }

    /// Apply bias correction to moment estimates
    ///
    /// Returns `tensor / (1 - beta^step)` for unbiasing the moment estimates.
    /// This correction is necessary because the moment estimates are initialized
    /// to zero, creating a bias towards zero that becomes negligible as training progresses.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Moment estimate tensor to correct
    /// * `bias_correction` - Bias correction factor (1 - beta^step)
    ///
    /// # Returns
    ///
    /// Bias-corrected tensor with the same shape as input
    #[inline]
    fn bias_correct(tensor: &Tensor, bias_correction: f32) -> Tensor {
        let mut result = tensor.clone();
        let correction_factor = 1.0 / bias_correction;

        unsafe {
            let ptr = result.as_mut_ptr();
            let size = result.size();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && result.is_simd_aligned() {
                    Self::bias_correct_simd_avx2(ptr, correction_factor, size);
                    return result;
                }
            }

            // Scalar fallback
            for i in 0..size {
                *ptr.add(i) *= correction_factor;
            }
        }

        result
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn bias_correct_simd_avx2(ptr: *mut f32, correction_factor: f32, size: usize) {
        let factor_vec = _mm256_set1_ps(correction_factor);
        let simd_count = size / 8;

        for i in 0..simd_count {
            let offset = i * 8;
            let data_vec = _mm256_loadu_ps(ptr.add(offset));
            let result = _mm256_mul_ps(data_vec, factor_vec);
            _mm256_storeu_ps(ptr.add(offset), result);
        }

        // Handle remaining elements
        for i in (simd_count * 8)..size {
            *ptr.add(i) *= correction_factor;
        }
    }

    /// Element-wise maximum for AMSGrad
    ///
    /// Updates first tensor in-place with `max(first, second)` for AMSGrad variant.
    /// This maintains a running maximum of the second moment estimates, preventing
    /// the learning rate from increasing over time and improving convergence stability.
    ///
    /// # Arguments
    ///
    /// * `first` - Tensor to update in-place with maximum values
    /// * `second` - Tensor to compare against for maximum calculation
    #[inline]
    fn element_wise_max(first: &mut Tensor, second: &Tensor) {
        assert_eq!(
            first.size(),
            second.size(),
            "Tensor size mismatch for element-wise max"
        );

        unsafe {
            let first_ptr = first.as_mut_ptr();
            let second_ptr = second.as_ptr();
            let size = first.size();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2")
                    && first.is_simd_aligned()
                    && second.is_simd_aligned()
                {
                    Self::element_wise_max_simd_avx2(first_ptr, second_ptr, size);
                    return;
                }
            }

            // Scalar fallback
            for i in 0..size {
                let a = *first_ptr.add(i);
                let b = *second_ptr.add(i);
                *first_ptr.add(i) = if a > b { a } else { b };
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn element_wise_max_simd_avx2(first_ptr: *mut f32, second_ptr: *const f32, size: usize) {
        let simd_count = size / 8;

        for i in 0..simd_count {
            let offset = i * 8;
            let first_vec = _mm256_loadu_ps(first_ptr.add(offset));
            let second_vec = _mm256_loadu_ps(second_ptr.add(offset));
            let result = _mm256_max_ps(first_vec, second_vec);
            _mm256_storeu_ps(first_ptr.add(offset), result);
        }

        // Handle remaining elements
        for i in (simd_count * 8)..size {
            let a = *first_ptr.add(i);
            let b = *second_ptr.add(i);
            *first_ptr.add(i) = if a > b { a } else { b };
        }
    }

    /// Apply the final Adam parameter update
    ///
    /// Implements the core Adam update rule: `param = param - lr * m_hat / (sqrt(v_hat) + eps)`
    /// This applies the bias-corrected momentum and velocity estimates to update
    /// the parameter values, with adaptive learning rates that scale inversely
    /// with the square root of the velocity estimates.
    ///
    /// # Arguments
    ///
    /// * `param` - Parameter tensor to update in-place
    /// * `m_hat` - Bias-corrected first moment estimate
    /// * `v_hat` - Bias-corrected second moment estimate
    /// * `learning_rate` - Learning rate for the update
    /// * `eps` - Small constant for numerical stability
    #[inline]
    fn apply_adam_update(
        param: &mut Tensor,
        m_hat: &Tensor,
        v_hat: &Tensor,
        learning_rate: f32,
        eps: f32,
    ) {
        assert_eq!(
            param.size(),
            m_hat.size(),
            "Parameter and momentum size mismatch"
        );
        assert_eq!(
            param.size(),
            v_hat.size(),
            "Parameter and velocity size mismatch"
        );

        unsafe {
            let param_ptr = param.as_mut_ptr();
            let m_ptr = m_hat.as_ptr();
            let v_ptr = v_hat.as_ptr();
            let size = param.size();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2")
                    && param.is_simd_aligned()
                    && m_hat.is_simd_aligned()
                    && v_hat.is_simd_aligned()
                {
                    Self::apply_adam_update_simd_avx2(
                        param_ptr,
                        m_ptr,
                        v_ptr,
                        learning_rate,
                        eps,
                        size,
                    );
                    return;
                }
            }

            // Scalar fallback
            for i in 0..size {
                let m_val = *m_ptr.add(i);
                let v_val = *v_ptr.add(i);
                let denominator = v_val.sqrt() + eps;
                *param_ptr.add(i) -= learning_rate * m_val / denominator;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn apply_adam_update_simd_avx2(
        param_ptr: *mut f32,
        m_ptr: *const f32,
        v_ptr: *const f32,
        learning_rate: f32,
        eps: f32,
        size: usize,
    ) {
        let lr_vec = _mm256_set1_ps(learning_rate);
        let eps_vec = _mm256_set1_ps(eps);
        let simd_count = size / 8;

        for i in 0..simd_count {
            let offset = i * 8;
            let param_vec = _mm256_loadu_ps(param_ptr.add(offset));
            let m_vec = _mm256_loadu_ps(m_ptr.add(offset));
            let v_vec = _mm256_loadu_ps(v_ptr.add(offset));

            // sqrt(v_hat) + eps
            let sqrt_v = _mm256_sqrt_ps(v_vec);
            let denominator = _mm256_add_ps(sqrt_v, eps_vec);

            // lr * m_hat / denominator
            let lr_m = _mm256_mul_ps(lr_vec, m_vec);
            let update = _mm256_div_ps(lr_m, denominator);

            // param - update
            let result = _mm256_sub_ps(param_vec, update);
            _mm256_storeu_ps(param_ptr.add(offset), result);
        }

        // Handle remaining elements
        for i in (simd_count * 8)..size {
            let m_val = *m_ptr.add(i);
            let v_val = *v_ptr.add(i);
            let denominator = v_val.sqrt() + eps;
            *param_ptr.add(i) -= learning_rate * m_val / denominator;
        }
    }
}

impl Optimizer for Adam {
    /// Perform a single optimization step
    ///
    /// Updates all provided parameters based on their accumulated gradients using the Adam algorithm.
    /// Each parameter is updated according to the Adam update rule with bias correction
    /// and optional AMSGrad variant if enabled. All parameters must be linked to the optimizer
    /// before calling this method.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Mutable slice of parameter references to update
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe as it takes mutable references to parameters,
    /// ensuring exclusive access during updates.
    ///
    /// # Performance
    ///
    /// - Uses SIMD optimization (AVX2) when available for 8x vectorization
    /// - Processes parameters in sequence for optimal cache usage
    /// - Maintains per-parameter state for momentum and velocity estimates
    ///
    /// # Panics
    ///
    /// Panics if any parameter is not linked to the optimizer
    fn step(&mut self, parameters: &mut [&mut Tensor]) {
        self.step_count += 1;

        for param in parameters {
            if let Err(e) = self.update_parameter(param) {
                eprintln!("Warning: Failed to update parameter: {}", e);
            }
        }
    }

    /// Zero out all parameter gradients
    ///
    /// Clears accumulated gradients for all provided parameters. This should be called
    /// before each backward pass to prevent gradient accumulation across multiple
    /// forward/backward passes. Also clears the global autograd gradient map.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Mutable slice of parameter references to clear gradients for
    ///
    /// # Performance
    ///
    /// - Efficiently clears gradients using optimized tensor operations
    /// - Clears both per-tensor gradients and global autograd state
    /// - Thread-safe as it takes mutable references to parameters
    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) {
        for param in parameters {
            param.zero_grad();
        }

        // Also clear gradient tracking gradient map
        crate::gradtrack::clear_gradients();
    }

    /// Get the current learning rate
    ///
    /// Returns the current learning rate used for parameter updates.
    ///
    /// # Returns
    ///
    /// Current learning rate as f32
    fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    /// Set the learning rate for all parameters
    ///
    /// Updates the learning rate for all parameters in the optimizer.
    /// This allows dynamic learning rate scheduling during training.
    ///
    /// # Arguments
    ///
    /// * `lr` - New learning rate value
    fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

// Adam is automatically Send + Sync since it no longer contains raw pointers
// and all fields are Send + Sync (HashMap, usize, AdamConfig)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::core::Tensor;

    /// Test Adam optimizer creation with default configuration
    ///
    /// Verifies that the optimizer is created correctly with default hyperparameters
    /// and that parameter states are properly initialized.
    #[test]
    fn test_adam_creation() {
        let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
        let bias = Tensor::zeros(vec![3, 1]).with_requires_grad();

        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);
        optimizer.add_parameter(&bias);

        assert_eq!(optimizer.learning_rate(), 1e-3);
        assert_eq!(optimizer.config.beta1, 0.9);
        assert_eq!(optimizer.config.beta2, 0.999);
        assert_eq!(optimizer.parameter_count(), 2);
    }

    /// Test Adam optimizer creation with custom configuration
    ///
    /// Verifies that custom hyperparameters are properly set and that
    /// the optimizer configuration matches the provided values.
    #[test]
    fn test_adam_with_config() {
        let weight = Tensor::ones(vec![5, 5]).with_requires_grad();

        let config = AdamConfig {
            learning_rate: 1e-4,
            beta1: 0.95,
            beta2: 0.9999,
            weight_decay: 1e-5,
            amsgrad: true,
            ..Default::default()
        };

        let mut optimizer = Adam::with_config(config);
        optimizer.add_parameter(&weight);

        assert_eq!(optimizer.learning_rate(), 1e-4);
        assert_eq!(optimizer.config.beta1, 0.95);
        assert_eq!(optimizer.config.beta2, 0.9999);
        assert_eq!(optimizer.config.weight_decay, 1e-5);
        assert!(optimizer.config.amsgrad);
    }

    /// Test Adam optimizer creation with custom learning rate
    ///
    /// Verifies that the convenience constructor properly sets the learning rate
    /// while maintaining default values for other hyperparameters.
    #[test]
    fn test_adam_with_learning_rate() {
        let weight = Tensor::ones(vec![3, 3]).with_requires_grad();
        let mut optimizer = Adam::with_learning_rate(5e-4);
        optimizer.add_parameter(&weight);

        assert_eq!(optimizer.learning_rate(), 5e-4);
        assert_eq!(optimizer.config.beta1, 0.9); // Should use defaults for other params
    }

    /// Test Adam step without gradients
    ///
    /// Verifies that the optimizer handles the case where parameters have no
    /// gradients gracefully, leaving parameters unchanged.
    #[test]
    fn test_adam_step_without_gradients() {
        let mut weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let original_data = weight.clone();

        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);

        optimizer.step(&mut [&mut weight]); // Should not update without gradients

        // Parameters should remain unchanged without gradients
        for i in 0..weight.size() {
            unsafe {
                assert_eq!(*weight.as_ptr().add(i), *original_data.as_ptr().add(i));
            }
        }
    }

    /// Test learning rate update functionality
    ///
    /// Verifies that the learning rate can be dynamically updated during training
    /// and that the optimizer uses the new learning rate for subsequent steps.
    #[test]
    fn test_learning_rate_update() {
        let weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);

        assert_eq!(optimizer.learning_rate(), 1e-3);

        optimizer.set_learning_rate(1e-2);
        assert_eq!(optimizer.learning_rate(), 1e-2);
    }

    /// Test gradient zeroing functionality
    ///
    /// Verifies that zero_grad properly clears accumulated gradients for all
    /// parameters and the global autograd state.
    #[test]
    fn test_zero_grad() {
        let mut weight = Tensor::ones(vec![2, 2]).with_requires_grad();

        // Check that zero_grad clears gradients
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);
        optimizer.zero_grad(&mut [&mut weight]);

        // After zero_grad, there should be no accumulated gradients
        assert!(weight.grad_by_value().is_none());
    }

    /// Test requires_grad assertion
    ///
    /// Verifies that the optimizer correctly panics when parameters do not
    /// have requires_grad set to true, ensuring proper gradient tracking.
    #[test]
    #[should_panic(expected = "Parameter must require gradients")]
    fn test_adam_requires_grad_assertion() {
        let weight = Tensor::ones(vec![2, 2]); // No requires_grad
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);
    }

    /// Test AdamConfig default values
    ///
    /// Verifies that the default configuration matches PyTorch conventions
    /// and provides optimal settings for most training scenarios.
    #[test]
    fn test_adam_config_default() {
        let config = AdamConfig::default();

        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.eps, 1e-8);
        assert_eq!(config.weight_decay, 0.0);
        assert!(!config.amsgrad);
    }

    /// Test ParameterState creation and initialization
    ///
    /// Verifies that parameter states are properly initialized with zero tensors
    /// and that the step count starts at zero.
    #[test]
    fn test_parameter_state_creation() {
        let state = ParameterState::new(&[3, 4]);

        assert_eq!(state.m.shape().dims, vec![3, 4]);
        assert_eq!(state.v.shape().dims, vec![3, 4]);
        assert!(state.v_hat_max.is_none());
        assert_eq!(state.step, 0);

        // Verify tensors are zero-initialized
        for i in 0..state.m.size() {
            unsafe {
                assert_eq!(*state.m.as_ptr().add(i), 0.0);
                assert_eq!(*state.v.as_ptr().add(i), 0.0);
            }
        }
    }

    /// Test parameter linking functionality
    ///
    /// Verifies that parameters can be linked and unlinked from the optimizer.
    #[test]
    fn test_parameter_linking() {
        let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
        let bias = Tensor::zeros(vec![3]).with_requires_grad();

        let mut optimizer = Adam::new();

        // Initially no parameters linked
        assert_eq!(optimizer.parameter_count(), 0);
        assert!(!optimizer.is_parameter_linked(&weight));
        assert!(!optimizer.is_parameter_linked(&bias));

        // Link weight
        optimizer.add_parameter(&weight);
        assert_eq!(optimizer.parameter_count(), 1);
        assert!(optimizer.is_parameter_linked(&weight));
        assert!(!optimizer.is_parameter_linked(&bias));

        // Link bias
        optimizer.add_parameter(&bias);
        assert_eq!(optimizer.parameter_count(), 2);
        assert!(optimizer.is_parameter_linked(&weight));
        assert!(optimizer.is_parameter_linked(&bias));

        // Unlink weight
        let was_linked = optimizer.unlink_parameter(&weight);
        assert!(was_linked);
        assert_eq!(optimizer.parameter_count(), 1);
        assert!(!optimizer.is_parameter_linked(&weight));
        assert!(optimizer.is_parameter_linked(&bias));

        // Clear all states
        optimizer.clear_states();
        assert_eq!(optimizer.parameter_count(), 0);
        assert!(!optimizer.is_parameter_linked(&weight));
        assert!(!optimizer.is_parameter_linked(&bias));
    }

    /// Test parameter linking with multiple parameters at once
    ///
    /// Verifies that multiple parameters can be linked simultaneously.
    #[test]
    fn test_add_multiple_parameters() {
        let weight = Tensor::ones(vec![2, 3]).with_requires_grad();
        let bias = Tensor::zeros(vec![3]).with_requires_grad();
        let weight2 = Tensor::ones(vec![3, 2]).with_requires_grad();

        let mut optimizer = Adam::new();

        // Link multiple parameters at once
        optimizer.add_parameters(&[&weight, &bias, &weight2]);

        assert_eq!(optimizer.parameter_count(), 3);
        assert!(optimizer.is_parameter_linked(&weight));
        assert!(optimizer.is_parameter_linked(&bias));
        assert!(optimizer.is_parameter_linked(&weight2));
    }

    /// Test stepping with unlinked parameter
    ///
    /// Verifies that the optimizer panics when trying to step with an unlinked parameter.
    #[test]
    #[should_panic(expected = "Parameter must be linked to optimizer before stepping")]
    fn test_step_with_unlinked_parameter() {
        let mut weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut optimizer = Adam::new();

        // Don't link the parameter
        optimizer.step(&mut [&mut weight]); // Should panic
    }

    /// Test optimizer with actual gradients
    ///
    /// Verifies that the optimizer properly updates parameters when gradients are present.
    #[test]
    fn test_optimizer_with_gradients() {
        let mut weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let original_data = weight.clone();

        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);

        // Generate some gradients
        let output = weight.mul_scalar(2.0);
        let mut loss = output.sum();
        loss.backward(None);

        // Step should update parameters
        optimizer.step(&mut [&mut weight]);

        // Parameters should have changed
        let mut changed = false;
        for i in 0..weight.size() {
            unsafe {
                if (*weight.as_ptr().add(i) - *original_data.as_ptr().add(i)).abs() > 1e-6 {
                    changed = true;
                    break;
                }
            }
        }
        assert!(
            changed,
            "Parameters should have been updated by optimizer step"
        );
    }

    /// Test optimizer with multiple parameters and gradients
    ///
    /// Verifies that the optimizer works correctly with multiple parameters.
    #[test]
    fn test_optimizer_multiple_parameters() {
        let mut weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut bias = Tensor::zeros(vec![2, 2]).with_requires_grad(); // Same shape as weight

        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight);
        optimizer.add_parameter(&bias);

        // Generate gradients for both parameters
        let output = weight.mul_scalar(2.0).add_tensor(&bias);
        let mut loss = output.sum();
        loss.backward(None);

        // Step should update both parameters
        optimizer.step(&mut [&mut weight, &mut bias]);

        // Both parameters should have gradients
        assert!(weight.grad_by_value().is_some());
        assert!(bias.grad_by_value().is_some());
    }

    /// Test optimizer with custom configuration and multiple steps
    ///
    /// Verifies that the optimizer works correctly with custom configuration over multiple steps.
    #[test]
    fn test_optimizer_custom_config_multiple_steps() {
        let mut weight = Tensor::ones(vec![2, 2]).with_requires_grad();

        let config = AdamConfig {
            learning_rate: 1e-4,
            beta1: 0.95,
            beta2: 0.9999,
            weight_decay: 1e-5,
            amsgrad: true,
            ..Default::default()
        };

        let mut optimizer = Adam::with_config(config);
        optimizer.add_parameter(&weight);

        // Multiple training steps
        for _ in 0..5 {
            // Generate gradients
            let output = weight.mul_scalar(2.0);
            let mut loss = output.sum();
            loss.backward(None);

            // Step
            optimizer.step(&mut [&mut weight]);
            optimizer.zero_grad(&mut [&mut weight]);
        }

        // Should complete without errors
        assert_eq!(optimizer.parameter_count(), 1);
        assert!(optimizer.is_parameter_linked(&weight));
    }

    /// Test optimizer with different tensor shapes
    ///
    /// Verifies that the optimizer works correctly with various tensor shapes.
    #[test]
    fn test_optimizer_different_shapes() {
        let shapes = vec![
            vec![1],       // Scalar
            vec![3],       // 1D
            vec![2, 2],    // 2D square
            vec![2, 3],    // 2D rectangular
            vec![1, 1, 3], // 3D
            vec![2, 2, 2], // 3D cube
        ];

        for shape in shapes {
            let mut tensor = Tensor::ones(shape.clone()).with_requires_grad();
            let mut optimizer = Adam::new();
            optimizer.add_parameter(&tensor);

            // Generate gradients
            let output = tensor.mul_scalar(2.0);
            let mut loss = output.sum();
            loss.backward(None);

            // Step should work for all shapes
            optimizer.step(&mut [&mut tensor]);

            // Verify tensor is still valid
            assert_eq!(tensor.shape().dims, shape);
            assert!(tensor.requires_grad());
        }
    }

    /// Test double parameter linking
    ///
    /// Verifies that linking the same parameter twice doesn't create duplicate states.
    #[test]
    fn test_double_parameter_linking() {
        let weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut optimizer = Adam::new();

        // Link parameter twice
        optimizer.add_parameter(&weight);
        optimizer.add_parameter(&weight);

        // Should only have one state
        assert_eq!(optimizer.parameter_count(), 1);
        assert!(optimizer.is_parameter_linked(&weight));
    }

    /// Test unlink non-linked parameter
    ///
    /// Verifies that unlinking a non-linked parameter returns false.
    #[test]
    fn test_unlink_non_linked_parameter() {
        let weight = Tensor::ones(vec![2, 2]).with_requires_grad();
        let mut optimizer = Adam::new();

        // Try to unlink a parameter that was never linked
        let was_linked = optimizer.unlink_parameter(&weight);
        assert!(!was_linked);
    }

    /// Test parameter shape validation
    ///
    /// Verifies that parameters with different shapes can be linked correctly
    /// and maintain their shape information in the optimizer state.
    #[test]
    fn test_parameter_shape_validation() {
        let weight_1d = Tensor::ones(vec![5]).with_requires_grad();
        let weight_2d = Tensor::ones(vec![3, 4]).with_requires_grad();
        let weight_3d = Tensor::ones(vec![2, 3, 4]).with_requires_grad();

        let mut optimizer = Adam::new();

        // Test linking parameters with different shapes
        optimizer.add_parameter(&weight_1d);
        optimizer.add_parameter(&weight_2d);
        optimizer.add_parameter(&weight_3d);

        assert_eq!(optimizer.parameter_count(), 3);
        assert!(optimizer.is_parameter_linked(&weight_1d));
        assert!(optimizer.is_parameter_linked(&weight_2d));
        assert!(optimizer.is_parameter_linked(&weight_3d));

        // Test that each parameter maintains its shape after linking
        let state_1d = &optimizer.states[&weight_1d.id()];
        let state_2d = &optimizer.states[&weight_2d.id()];
        let state_3d = &optimizer.states[&weight_3d.id()];

        assert_eq!(state_1d.m.shape().dims, vec![5]);
        assert_eq!(state_2d.m.shape().dims, vec![3, 4]);
        assert_eq!(state_3d.m.shape().dims, vec![2, 3, 4]);
    }

    /// Test parameter relinking with shape consistency
    ///
    /// Verifies that when re-linking parameters (e.g., after deserialization),
    /// shape information is correctly maintained.
    #[test]
    fn test_parameter_relinking_shape_consistency() {
        // Create initial parameter and optimizer
        let mut weight_original = Tensor::ones(vec![3, 3]).with_requires_grad();
        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight_original);

        // Perform a step to create state
        let output = weight_original.mul_scalar(2.0);
        let mut loss = output.sum();
        loss.backward(None);
        optimizer.step(&mut [&mut weight_original]);

        // Verify state was created with correct shape
        let original_state = &optimizer.states[&weight_original.id()];
        assert_eq!(original_state.m.shape().dims, vec![3, 3]);
        assert_eq!(original_state.v.shape().dims, vec![3, 3]);

        // Create new parameter with same shape (will get different ID)
        let weight_new = Tensor::ones(vec![3, 3]).with_requires_grad();

        // This should create a new state since it's a different parameter
        optimizer.add_parameter(&weight_new);

        // Should now have 2 states
        assert_eq!(optimizer.parameter_count(), 2);

        // Both should be linked with correct shapes
        assert!(optimizer.is_parameter_linked(&weight_original));
        assert!(optimizer.is_parameter_linked(&weight_new));

        let new_state = &optimizer.states[&weight_new.id()];
        assert_eq!(new_state.m.shape().dims, vec![3, 3]);
        assert_eq!(new_state.v.shape().dims, vec![3, 3]);
    }

    /// Test large parameter count handling
    ///
    /// Verifies that the optimizer can handle many parameters efficiently
    /// and correctly manage linking/unlinking operations.
    #[test]
    fn test_large_parameter_count() {
        let mut optimizer = Adam::new();
        let mut params = Vec::new();

        // Create 50 parameters of different shapes
        for i in 1..=50 {
            let param = Tensor::ones(vec![i]).with_requires_grad();
            optimizer.add_parameter(&param);
            params.push(param);
        }

        assert_eq!(optimizer.parameter_count(), 50);

        // Verify all parameters are linked
        for param in &params {
            assert!(optimizer.is_parameter_linked(param));
        }

        // Test unlinking some parameters
        for param in params.iter().take(25).step_by(2) {
            assert!(optimizer.unlink_parameter(param));
        }

        assert_eq!(optimizer.parameter_count(), 37); // 50 - 13 = 37

        // Verify correct parameters are unlinked
        for (i, param) in params.iter().enumerate() {
            if i < 25 && i % 2 == 0 {
                assert!(!optimizer.is_parameter_linked(param));
            } else {
                assert!(optimizer.is_parameter_linked(param));
            }
        }
    }

    /// Test clear_states functionality
    ///
    /// Verifies that clearing all states works correctly and allows
    /// re-adding parameters afterwards.
    #[test]
    fn test_clear_states_functionality() {
        let weight1 = Tensor::ones(vec![2, 2]).with_requires_grad();
        let weight2 = Tensor::ones(vec![3, 3]).with_requires_grad();
        let weight3 = Tensor::ones(vec![4, 4]).with_requires_grad();

        let mut optimizer = Adam::new();
        optimizer.add_parameter(&weight1);
        optimizer.add_parameter(&weight2);
        optimizer.add_parameter(&weight3);

        assert_eq!(optimizer.parameter_count(), 3);

        // Clear all states
        optimizer.clear_states();

        assert_eq!(optimizer.parameter_count(), 0);
        assert!(!optimizer.is_parameter_linked(&weight1));
        assert!(!optimizer.is_parameter_linked(&weight2));
        assert!(!optimizer.is_parameter_linked(&weight3));

        // Should be able to re-add parameters after clearing
        optimizer.add_parameter(&weight1);
        assert_eq!(optimizer.parameter_count(), 1);
        assert!(optimizer.is_parameter_linked(&weight1));
    }
}
