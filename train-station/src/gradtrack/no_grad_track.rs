//! NoGradTrack for temporarily disabling gradient tracking
//!
//! This module provides functionality similar to PyTorch's `torch.no_grad()` context manager,
//! allowing users to temporarily disable gradient computation for performance optimization
//! during inference or validation phases.
//!
//! # Features
//! - Thread-local gradient context management
//! - RAII pattern for automatic restoration
//! - Nested context support with proper stack management
//! - Zero-cost abstraction when gradients are already disabled
//! - Thread-safe design for concurrent usage

use std::cell::RefCell;

thread_local! {
    /// Thread-local storage for gradient tracking state
    ///
    /// Uses a stack to support nested NoGradTrack contexts.
    /// Each element represents whether gradients were enabled at that nesting level.
    static GRAD_ENABLED_STACK: RefCell<Vec<bool>> = RefCell::new(vec![true]);
}

/// A RAII guard that temporarily disables gradient tracking
///
/// Similar to PyTorch's `torch.no_grad()`, this guard disables gradient computation
/// within its scope and automatically restores the previous gradient tracking state
/// when it goes out of scope.
///
/// # Performance Benefits
/// - Prevents computation graph construction during inference
/// - Reduces memory usage by not storing intermediate values for backpropagation
/// - Improves computation speed by skipping gradient-related operations
///
/// # Examples
///
/// ```rust
/// use train_station::{NoGradTrack, Tensor};
///
/// let x = Tensor::ones(vec![3, 3]).with_requires_grad();
/// let y = Tensor::ones(vec![3, 3]).with_requires_grad();
///
/// // Normal computation with gradients
/// let z1 = x.add_tensor(&y);
/// assert!(z1.requires_grad());
///
/// // Computation without gradients
/// {
///     let _guard = NoGradTrack::new();
///     let z2 = x.add_tensor(&y);
///     assert!(!z2.requires_grad()); // Gradients disabled
/// } // Guard drops here, gradients restored
///
/// // Gradients are automatically restored
/// let z3 = x.add_tensor(&y);
/// assert!(z3.requires_grad());
/// ```
///
/// # Nested Contexts
///
/// ```rust
/// use train_station::{NoGradTrack, is_grad_enabled, Tensor};
///
/// assert!(is_grad_enabled());
///
/// {
///     let _guard1 = NoGradTrack::new();
///     assert!(!is_grad_enabled());
///
///     {
///         let _guard2 = NoGradTrack::new();
///         assert!(!is_grad_enabled());
///     } // guard2 drops
///
///     assert!(!is_grad_enabled()); // Still disabled
/// } // guard1 drops
///
/// assert!(is_grad_enabled()); // Restored
/// ```
pub struct NoGradTrack {
    // Marker to ensure the guard cannot be constructed outside this module
    // without using the `new()` method
    _private: (),
}

impl NoGradTrack {
    /// Create a new NoGradTrack that disables gradient tracking
    ///
    /// This function pushes the current gradient state onto the stack and
    /// disables gradient tracking. When the guard is dropped, the previous
    /// state is automatically restored.
    ///
    /// # Returns
    ///
    /// A new `NoGradTrack` that will restore gradient state when dropped
    pub fn new() -> Self {
        GRAD_ENABLED_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.push(false); // Disable gradients
        });

        NoGradTrack { _private: () }
    }
}

impl Default for NoGradTrack {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradTrack {
    /// Automatically restore the previous gradient tracking state
    ///
    /// This ensures that gradient tracking is properly restored even if
    /// the guard goes out of scope due to early returns or panics.
    fn drop(&mut self) {
        GRAD_ENABLED_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if stack.len() > 1 {
                stack.pop(); // Remove the current (disabled) state
                             // The previous state is now at the top of the stack
            } else {
                // This should not happen in normal usage, but we handle it gracefully
                // by ensuring at least one state remains (the default true state)
                *stack = vec![true];
            }
        });
    }
}

/// Check if gradient computation is currently enabled
///
/// This function returns the current gradient tracking state for the current thread.
/// It respects any active `NoGradTrack` contexts.
///
/// # Returns
///
/// `true` if gradient computation is enabled, `false` otherwise
///
/// # Examples
///
/// ```rust
/// use train_station::{NoGradTrack, is_grad_enabled};
///
/// assert!(is_grad_enabled()); // Default state
///
/// {
///     let _guard = NoGradTrack::new();
///     assert!(!is_grad_enabled()); // Disabled by guard
/// }
///
/// assert!(is_grad_enabled()); // Restored after guard drops
/// ```
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED_STACK.with(|stack| {
        let stack = stack.borrow();
        *stack.last().unwrap_or(&true)
    })
}

/// Manually set the gradient tracking state
///
/// This function allows manual control over gradient tracking state.
/// It's primarily intended for internal use and testing. In most cases,
/// using `NoGradTrack` is preferred as it provides automatic restoration.
///
/// # Arguments
///
/// * `enabled` - Whether to enable or disable gradient tracking
///
/// # Warning
///
/// This function modifies the current gradient state without automatic restoration.
/// Use `NoGradTrack` for RAII-style management in most cases.
///
/// # Examples
///
/// ```rust
/// use train_station::{set_grad_enabled, is_grad_enabled};
///
/// assert!(is_grad_enabled());
/// set_grad_enabled(false);
/// assert!(!is_grad_enabled());
/// set_grad_enabled(true);
/// assert!(is_grad_enabled());
/// ```
pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if let Some(last) = stack.last_mut() {
            *last = enabled;
        } else {
            // Fallback: ensure stack has at least one element
            *stack = vec![enabled];
        }
    });
}

/// Convenience function to execute a closure with gradients disabled
///
/// This function provides a convenient way to execute code with gradients
/// disabled without explicitly managing a `NoGradTrack`.
///
/// # Arguments
///
/// * `f` - The closure to execute with gradients disabled
///
/// # Returns
///
/// The result of the closure
///
/// # Examples
///
/// ```rust
/// use train_station::{Tensor, with_no_grad, is_grad_enabled};
///
/// let x = Tensor::ones(vec![2, 2]).with_requires_grad();
/// let y = Tensor::ones(vec![2, 2]).with_requires_grad();
///
/// let result = with_no_grad(|| {
///     assert!(!is_grad_enabled());
///     x.add_tensor(&y)
/// });
///
/// assert!(!result.requires_grad());
/// assert!(is_grad_enabled()); // Restored after closure
/// ```
pub fn with_no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradTrack::new();
    f()
}

/// Reset the gradient tracking state to the default (enabled)
///
/// This function is primarily for testing and debugging purposes.
/// It clears the entire gradient state stack and resets to the default state.
///
/// # Warning
///
/// This function will disrupt any active `NoGradTrack` contexts and should
/// only be used in test cleanup or exceptional circumstances.
#[cfg(test)]
pub fn reset_grad_state() {
    GRAD_ENABLED_STACK.with(|stack| {
        *stack.borrow_mut() = vec![true];
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_grad_enabled() {
        reset_grad_state();
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_guard_basic() {
        reset_grad_state();
        assert!(is_grad_enabled());

        {
            let _guard = NoGradTrack::new();
            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_nested_no_grad_guards() {
        reset_grad_state();
        assert!(is_grad_enabled());

        {
            let _guard1 = NoGradTrack::new();
            assert!(!is_grad_enabled());

            {
                let _guard2 = NoGradTrack::new();
                assert!(!is_grad_enabled());

                {
                    let _guard3 = NoGradTrack::new();
                    assert!(!is_grad_enabled());
                }

                assert!(!is_grad_enabled());
            }

            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_set_grad_enabled() {
        reset_grad_state();
        assert!(is_grad_enabled());

        set_grad_enabled(false);
        assert!(!is_grad_enabled());

        set_grad_enabled(true);
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_with_no_grad_function() {
        reset_grad_state();
        assert!(is_grad_enabled());

        let result = with_no_grad(|| {
            assert!(!is_grad_enabled());
            42
        });

        assert_eq!(result, 42);
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_with_no_grad_nested() {
        reset_grad_state();
        assert!(is_grad_enabled());

        with_no_grad(|| {
            assert!(!is_grad_enabled());

            with_no_grad(|| {
                assert!(!is_grad_enabled());
            });

            assert!(!is_grad_enabled());
        });

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_multiple_guards_same_scope() {
        reset_grad_state();
        assert!(is_grad_enabled());

        let _guard1 = NoGradTrack::new();
        assert!(!is_grad_enabled());

        let _guard2 = NoGradTrack::new();
        assert!(!is_grad_enabled());

        drop(_guard1);
        assert!(!is_grad_enabled()); // Still disabled due to guard2

        drop(_guard2);
        assert!(is_grad_enabled()); // Now restored
    }

    #[test]
    fn test_early_return_with_guard() {
        fn test_function() -> i32 {
            reset_grad_state();
            assert!(is_grad_enabled());

            let _guard = NoGradTrack::new();
            assert!(!is_grad_enabled());

            if true {
                return 42; // Early return should still restore state
            }

            unreachable!()
        }

        let result = test_function();
        assert_eq!(result, 42);
        assert!(is_grad_enabled()); // Should be restored even with early return
    }

    #[test]
    fn test_thread_local_isolation() {
        reset_grad_state();
        assert!(is_grad_enabled());

        let handle = std::thread::spawn(|| {
            // Each thread should start with gradients enabled
            assert!(is_grad_enabled());

            let _guard = NoGradTrack::new();
            assert!(!is_grad_enabled());

            // Return the state that should be isolated to this thread
            is_grad_enabled()
        });

        // Main thread should still have gradients enabled
        assert!(is_grad_enabled());

        let other_thread_state = handle.join().unwrap();
        assert!(!other_thread_state); // Other thread had gradients disabled

        // Main thread should still be unaffected
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_panic_safety() {
        reset_grad_state();
        assert!(is_grad_enabled());

        let result = std::panic::catch_unwind(|| {
            let _guard = NoGradTrack::new();
            assert!(!is_grad_enabled());
            panic!("Test panic");
        });

        assert!(result.is_err());

        // State should be restored even after panic
        // Note: This might not work in all test runners due to panic handling
        // but the RAII pattern should ensure cleanup in normal Rust programs
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_grad_state_stack_integrity() {
        reset_grad_state();

        // Test that the stack maintains integrity through complex operations
        assert!(is_grad_enabled());

        {
            let _g1 = NoGradTrack::new();
            assert!(!is_grad_enabled());

            set_grad_enabled(true); // Manual override
            assert!(is_grad_enabled());

            {
                let _g2 = NoGradTrack::new();
                assert!(!is_grad_enabled());
            }

            assert!(is_grad_enabled()); // Should restore to manually set state
        }

        assert!(is_grad_enabled()); // Should restore to original state
    }
}
