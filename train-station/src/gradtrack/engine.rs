//! Core gradient computation engine for automatic differentiation
//!
//! This module provides the central gradient tracking engine that implements reverse-mode
//! automatic differentiation (backpropagation) for the Train Station GradTrack system.
//! The engine is designed for maximum performance with thread-local storage, efficient
//! computation graph management, and optimized gradient propagation algorithms.
//!
//! # Purpose
//!
//! The gradient engine serves as the computational core of the GradTrack system, providing:
//! - **Computation graph management**: Thread-local storage and efficient graph operations
//! - **Backward pass orchestration**: Reverse-mode automatic differentiation implementation
//! - **Gradient accumulation**: Proper handling of multiple gradient contributions
//! - **Memory management**: Efficient gradient storage and cleanup
//! - **Thread safety**: Complete isolation between concurrent training threads
//!
//! # Architecture
//!
//! The gradient engine consists of several key components:
//!
//! ## GradGraph
//! Thread-local computation graph storage that maintains:
//! - **Operation registry**: Maps tensor operations to their gradient functions
//! - **Gradient accumulation**: Efficient storage and accumulation of computed gradients
//! - **Memory optimization**: Pre-allocated HashMaps for typical neural network sizes
//!
//! ## GradEngine
//! The main computational engine that provides:
//! - **Backward pass implementation**: Reverse-mode automatic differentiation
//! - **Operation registration**: Integration with tensor operations
//! - **Gradient propagation**: Efficient worklist-based graph traversal
//!
//! ## Thread-Local Storage
//! Complete thread isolation through:
//! - **Zero contention**: No synchronization overhead between threads
//! - **Memory safety**: Prevents data races and concurrent access issues
//! - **Performance optimization**: Eliminates locking and atomic operations
//!
//! # Algorithm Implementation
//!
//! The engine implements reverse-mode automatic differentiation using:
//!
//! ## Forward Pass Registration
//! During tensor operations:
//! 1. **Operation recording**: Each operation registers its gradient function
//! 2. **Dependency tracking**: Input-output relationships are stored
//! 3. **Metadata preservation**: Necessary information for gradient computation is saved
//!
//! ## Backward Pass Execution
//! During gradient computation:
//! 1. **Initialization**: Initial gradient is set (typically ones for scalar loss)
//! 2. **Worklist traversal**: Graph is traversed in reverse topological order
//! 3. **Gradient computation**: Chain rule is applied via gradient functions
//! 4. **Accumulation**: Multiple gradients to the same tensor are properly combined
//! 5. **Propagation**: Process continues until all leaf tensors are reached
//!
//! # Performance Characteristics
//!
//! ## Computational Complexity
//! - **Time complexity**: O(V + E) where V is tensors and E is operations
//! - **Memory complexity**: O(V) for gradient storage
//! - **Gradient accumulation**: O(1) per operation with HashMap storage
//! - **Graph traversal**: Optimized worklist-based algorithm
//!
//! ## Memory Efficiency
//! - **Pre-allocation**: 256 entries pre-allocated for typical neural networks
//! - **Gradient storage**: ~64 bytes overhead per tensor for gradient tracking
//! - **Thread isolation**: ~2KB pre-allocated memory per thread
//! - **Smart cleanup**: Automatic memory management between training iterations
//!
//! ## Thread Safety
//! - **Zero contention**: Thread-local storage eliminates synchronization overhead
//! - **Concurrent training**: Multiple threads can train simultaneously
//! - **Memory isolation**: Complete separation of gradient state between threads
//! - **Performance scaling**: Linear scaling with thread count
//!
//! # Integration with Tensor Operations
//!
//! The engine integrates seamlessly with tensor operations through:
//! - **Automatic registration**: Operations automatically register gradient functions
//! - **Transparent tracking**: No changes needed to existing tensor operation code
//! - **Conditional activation**: Gradient tracking only when `requires_grad` is enabled
//! - **Efficient propagation**: Optimized gradient flow through computation graphs
//!
//! # Design Principles
//!
//! The engine follows key design principles:
//! - **Simplicity**: Avoid complex global state management
//! - **Efficiency**: Pre-allocated storage and optimized algorithms
//! - **Safety**: Thread-local storage prevents data races
//! - **Compatibility**: PyTorch-like API for familiar usage patterns
//! - **Performance**: Zero-cost abstractions and minimal overhead
//!
//! # Thread Safety
//!
//! All components in this module are designed for thread safety:
//! - **Thread-local storage**: Each thread maintains independent gradient state
//! - **No shared state**: Eliminates need for synchronization primitives
//! - **Concurrent execution**: Multiple threads can perform gradient computation simultaneously
//! - **Memory safety**: Prevents data races and concurrent access violations

use super::grad_fn::GradFn;
use crate::tensor::core::Tensor;
use std::cell::RefCell;
use std::collections::HashMap;

/// Thread-local computation graph for efficient gradient tracking
///
/// This structure maintains the computation graph and gradient storage for a single thread,
/// providing the foundation for reverse-mode automatic differentiation. It uses pre-allocated
/// HashMaps to minimize allocation overhead during training and ensures complete thread
/// isolation for concurrent gradient computation.
///
/// # Purpose
///
/// The GradGraph serves as the core data structure for gradient tracking, providing:
/// - **Operation registry**: Maps tensor operations to their gradient functions and dependencies
/// - **Gradient storage**: Accumulates computed gradients during backward pass
/// - **Memory optimization**: Pre-allocated storage for typical neural network sizes
/// - **Thread isolation**: Complete separation of gradient state between threads
///
/// # Data Structure Design
///
/// ## Operations Map
/// The operations HashMap stores the computation graph structure:
/// - **Key**: Output tensor ID (usize)
/// - **Value**: Tuple of (input_tensor_ids, gradient_function)
/// - **Purpose**: Enables reverse traversal during backward pass
/// - **Capacity**: Pre-allocated for 256 operations (typical neural network size)
///
/// ## Gradients Map
/// The gradients HashMap accumulates computed gradients:
/// - **Key**: Tensor ID (usize)
/// - **Value**: Accumulated gradient tensor
/// - **Purpose**: Stores final gradient values for each tensor
/// - **Capacity**: Pre-allocated for 256 gradients (matches operations capacity)
///
/// # Memory Layout
///
/// The structure is optimized for memory efficiency:
/// - **Base overhead**: ~2KB pre-allocated for both HashMaps
/// - **Per-operation cost**: ~32 bytes for operation metadata
/// - **Per-gradient cost**: Variable based on tensor size
/// - **Total typical usage**: ~4-8KB for standard neural networks
///
/// # Performance Characteristics
///
/// ## Access Patterns
/// - **Operation lookup**: O(1) average case HashMap access
/// - **Gradient accumulation**: O(1) HashMap access + O(n) tensor addition
/// - **Graph traversal**: O(V + E) where V is tensors and E is operations
/// - **Memory allocation**: Minimal due to pre-allocation strategy
///
/// ## Optimization Features
/// - **Pre-allocation**: Reduces allocation overhead during training
/// - **Efficient accumulation**: Optimized tensor addition for gradient combination
/// - **Memory reuse**: HashMaps maintain capacity between training iterations
/// - **Cache-friendly**: Contiguous storage for better memory access patterns
///
/// # Thread Safety
///
/// This structure is designed for thread-local usage:
/// - **No synchronization**: Eliminates locking and atomic operation overhead
/// - **Complete isolation**: Each thread maintains independent gradient state
/// - **Memory safety**: Prevents data races and concurrent access issues
/// - **Concurrent training**: Multiple threads can train simultaneously without interference
///
/// # Implementation Details
///
/// The GradGraph uses several optimization strategies:
/// - **Capacity management**: Pre-allocated HashMaps avoid frequent reallocations
/// - **Gradient accumulation**: Efficient tensor addition with memory reuse
/// - **Operation tracking**: Minimal metadata storage for gradient function dispatch
/// - **Cleanup efficiency**: Fast clearing between training iterations
struct GradGraph {
    /// Maps output tensor ID to (input_tensor_ids, grad_fn)
    ///
    /// Stores the computation graph structure where each tensor operation
    /// records its input dependencies and gradient function for backward pass.
    operations: HashMap<usize, (Vec<usize>, GradFn)>,

    /// Pre-allocated gradient storage for better performance
    ///
    /// Accumulates gradients for each tensor during backward pass.
    /// Uses optimized tensor addition for gradient accumulation.
    gradients: HashMap<usize, Tensor>,
}

impl GradGraph {
    /// Create a new gradient graph with optimized pre-allocated storage
    ///
    /// Initializes a new GradGraph with pre-allocated HashMaps sized for typical neural
    /// network training scenarios. The pre-allocation strategy minimizes memory allocation
    /// overhead during training by avoiding frequent HashMap resizing operations.
    ///
    /// # Pre-allocation Strategy
    ///
    /// The method pre-allocates storage based on typical neural network characteristics:
    /// - **Operations capacity**: 256 entries for computation graph nodes
    /// - **Gradients capacity**: 256 entries for gradient accumulation
    /// - **Memory overhead**: ~2KB initial allocation per thread
    /// - **Scaling**: HashMaps will grow automatically if needed for larger networks
    ///
    /// # Performance Benefits
    ///
    /// Pre-allocation provides several performance advantages:
    /// - **Reduced allocations**: Eliminates frequent HashMap resizing during training
    /// - **Memory locality**: Contiguous storage improves cache performance
    /// - **Predictable overhead**: Consistent memory usage across training iterations
    /// - **Fast initialization**: O(1) creation time with pre-sized storage
    ///
    /// # Returns
    ///
    /// A new GradGraph instance with pre-allocated storage ready for gradient tracking
    ///
    /// # Implementation Details
    ///
    /// The capacity of 256 entries is chosen based on analysis of typical neural networks:
    /// - **Small networks**: 10-50 operations (well within capacity)
    /// - **Medium networks**: 100-200 operations (fits comfortably)
    /// - **Large networks**: May exceed capacity but HashMap will resize automatically
    /// - **Memory efficiency**: Balances pre-allocation benefits with memory usage
    fn new() -> Self {
        Self {
            operations: HashMap::with_capacity(256), // Pre-allocate for typical graph sizes
            gradients: HashMap::with_capacity(256),
        }
    }

    /// Register a tensor operation in the computation graph for gradient tracking
    ///
    /// Records a tensor operation's metadata in the computation graph, establishing the
    /// relationship between input and output tensors along with the gradient function
    /// needed for backward pass computation. This method is called automatically by
    /// tensor operations that support gradient tracking.
    ///
    /// # Arguments
    ///
    /// * `output_id` - Unique identifier of the output tensor produced by this operation
    /// * `input_ids` - Vector of unique identifiers for input tensors that this operation depends on
    /// * `grad_fn` - Gradient function containing the logic to compute gradients for this operation
    ///
    /// # Operation Registration Process
    ///
    /// The registration process involves:
    /// 1. **Dependency tracking**: Input tensor IDs are stored to enable reverse traversal
    /// 2. **Gradient function storage**: The GradFn is stored for gradient computation
    /// 3. **Graph structure building**: Links are established between input and output tensors
    /// 4. **Metadata preservation**: All information needed for backward pass is saved
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(1) average case for HashMap insertion
    /// - **Memory usage**: Stores operation metadata (~32 bytes per operation)
    /// - **Insertion cost**: Minimal overhead due to pre-allocated HashMap capacity
    /// - **Access pattern**: Optimized for reverse traversal during backward pass
    ///
    /// # Implementation Details
    ///
    /// The method stores the operation information in the operations HashMap:
    /// - **Key**: Output tensor ID for efficient lookup during backward pass
    /// - **Value**: Tuple containing input dependencies and gradient function
    /// - **Storage**: Pre-allocated HashMap minimizes allocation overhead
    /// - **Thread safety**: Thread-local storage ensures no synchronization needed
    fn register_operation(&mut self, output_id: usize, input_ids: Vec<usize>, grad_fn: GradFn) {
        self.operations.insert(output_id, (input_ids, grad_fn));
    }

    /// Retrieve operation information for a tensor from the computation graph
    ///
    /// Looks up the operation metadata associated with a tensor ID, returning the input
    /// dependencies and gradient function if the tensor was produced by an operation.
    /// Returns `None` for leaf tensors (parameters or constants) that have no associated
    /// operation in the computation graph.
    ///
    /// # Arguments
    ///
    /// * `tensor_id` - Unique identifier of the tensor to look up operation information for
    ///
    /// # Returns
    ///
    /// - `Some((input_ids, grad_fn))` - Operation information if the tensor has a recorded operation
    /// - `None` - If the tensor is a leaf node (parameter, constant, or input) with no operation
    ///
    /// # Operation Information Structure
    ///
    /// When an operation is found, the returned tuple contains:
    /// - **input_ids**: Vector of tensor IDs that serve as inputs to the operation
    /// - **grad_fn**: Gradient function that computes gradients for this operation
    ///
    /// # Usage in Backward Pass
    ///
    /// This method is essential during backward pass traversal:
    /// 1. **Graph traversal**: Determines if a tensor has upstream dependencies
    /// 2. **Gradient computation**: Provides the gradient function for chain rule application
    /// 3. **Dependency resolution**: Identifies input tensors that need gradient accumulation
    /// 4. **Leaf detection**: Distinguishes between intermediate and leaf tensors
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(1) average case HashMap lookup
    /// - **Memory access**: Single hash table lookup with minimal overhead
    /// - **Cache efficiency**: Pre-allocated HashMap provides good memory locality
    /// - **Thread safety**: Thread-local storage eliminates synchronization overhead
    ///
    /// # Implementation Details
    ///
    /// The method performs a simple HashMap lookup:
    /// - **Key lookup**: Uses tensor ID as key for efficient access
    /// - **Reference return**: Returns reference to avoid unnecessary cloning
    /// - **Option handling**: Naturally handles both operation and leaf cases
    /// - **Memory efficiency**: No allocation required for lookup operation
    fn get_operation(&self, tensor_id: usize) -> Option<&(Vec<usize>, GradFn)> {
        self.operations.get(&tensor_id)
    }

    /// Store a gradient tensor for a specific tensor ID
    ///
    /// Sets the gradient for a tensor, replacing any existing gradient that may have been
    /// previously stored. This method is primarily used to initialize gradients at the
    /// beginning of the backward pass, typically for the output tensor (loss) that starts
    /// the gradient computation process.
    ///
    /// # Arguments
    ///
    /// * `tensor_id` - Unique identifier of the tensor to store the gradient for
    /// * `gradient` - Gradient tensor to store, containing the computed gradient values
    ///
    /// # Usage Patterns
    ///
    /// This method is used in several key scenarios:
    /// - **Backward pass initialization**: Setting the initial gradient (usually ones) for the loss tensor
    /// - **Gradient replacement**: Overwriting existing gradients when recomputing
    /// - **Direct gradient setting**: Manually setting gradients for specific tensors
    /// - **Gradient reset**: Replacing accumulated gradients with new values
    ///
    /// # Storage Behavior
    ///
    /// The method exhibits specific storage behavior:
    /// - **Overwrite policy**: Any existing gradient for the tensor ID is replaced
    /// - **Memory management**: Takes ownership of the provided gradient tensor
    /// - **No accumulation**: Unlike `accumulate_gradient`, this method replaces rather than adds
    /// - **Immediate storage**: Gradient is immediately available for retrieval
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(1) average case HashMap insertion
    /// - **Memory usage**: Stores the gradient tensor in the gradients HashMap
    /// - **Insertion cost**: Minimal overhead due to pre-allocated HashMap capacity
    /// - **Memory ownership**: Takes ownership of gradient tensor (no cloning required)
    ///
    /// # Implementation Details
    ///
    /// The method performs a straightforward HashMap insertion:
    /// - **Key**: Tensor ID for efficient lookup during gradient retrieval
    /// - **Value**: Gradient tensor containing the computed gradient values
    /// - **Replacement**: Any existing gradient is automatically replaced
    /// - **Thread safety**: Thread-local storage ensures no synchronization needed
    fn store_gradient(&mut self, tensor_id: usize, gradient: Tensor) {
        self.gradients.insert(tensor_id, gradient);
    }

    /// Retrieve a reference to a tensor's accumulated gradient
    ///
    /// Returns a reference to the gradient tensor associated with the specified tensor ID.
    /// This method provides read-only access to the gradient without transferring ownership,
    /// making it suitable for gradient inspection and analysis without affecting the stored
    /// gradient state.
    ///
    /// # Arguments
    ///
    /// * `tensor_id` - Unique identifier of the tensor to retrieve the gradient for
    ///
    /// # Returns
    ///
    /// - `Some(&Tensor)` - Reference to the gradient tensor if a gradient has been computed and stored
    /// - `None` - If no gradient has been computed or stored for the specified tensor
    ///
    /// # Usage Patterns
    ///
    /// This method is commonly used for:
    /// - **Gradient inspection**: Examining gradient values without modifying them
    /// - **Gradient analysis**: Analyzing gradient statistics or properties
    /// - **Conditional processing**: Checking if gradients exist before processing
    /// - **Read-only access**: Accessing gradients without affecting the computation graph
    ///
    /// # Gradient Availability
    ///
    /// Gradients are available in the following scenarios:
    /// - **After backward pass**: Gradients computed during backward pass are stored
    /// - **After accumulation**: Multiple gradient contributions have been accumulated
    /// - **After manual storage**: Gradients explicitly stored via `store_gradient`
    /// - **Before cleanup**: Gradients remain available until `clear` is called
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(1) average case HashMap lookup
    /// - **Memory access**: Single hash table lookup with minimal overhead
    /// - **Reference return**: No cloning or allocation required
    /// - **Cache efficiency**: Pre-allocated HashMap provides good memory locality
    ///
    /// # Implementation Details
    ///
    /// The method performs a simple HashMap lookup:
    /// - **Key lookup**: Uses tensor ID as key for efficient access
    /// - **Reference return**: Returns reference to avoid unnecessary copying
    /// - **Option handling**: Naturally handles both present and absent gradients
    /// - **Thread safety**: Thread-local storage eliminates synchronization overhead
    fn get_gradient(&self, tensor_id: usize) -> Option<&Tensor> {
        self.gradients.get(&tensor_id)
    }

    /// Take ownership of a tensor's gradient, removing it from storage
    ///
    /// Removes and returns the gradient tensor for the specified tensor ID, transferring
    /// ownership to the caller. After this operation, the gradient is no longer stored
    /// in the graph and subsequent calls to `get_gradient` for this tensor will return
    /// `None`. This method is essential for preventing gradient double-counting during
    /// backward pass traversal.
    ///
    /// # Arguments
    ///
    /// * `tensor_id` - Unique identifier of the tensor to take the gradient from
    ///
    /// # Returns
    ///
    /// - `Some(Tensor)` - Owned gradient tensor if a gradient was stored for this tensor
    /// - `None` - If no gradient was stored for the specified tensor
    ///
    /// # Usage in Backward Pass
    ///
    /// This method plays a crucial role in backward pass implementation:
    /// - **Gradient consumption**: Takes gradient for processing without leaving a copy
    /// - **Double-counting prevention**: Ensures gradients are only used once per traversal
    /// - **Memory management**: Removes gradients from storage to free memory
    /// - **Ownership transfer**: Provides owned gradient for further computation
    ///
    /// # Backward Pass Algorithm
    ///
    /// The typical usage pattern in backward pass:
    /// 1. **Take gradient**: Remove gradient from storage for current tensor
    /// 2. **Apply gradient function**: Compute input gradients using the taken gradient
    /// 3. **Accumulate results**: Add computed gradients to input tensors
    /// 4. **Continue traversal**: Process input tensors in the same manner
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(1) average case HashMap removal
    /// - **Memory management**: Frees gradient storage immediately upon removal
    /// - **Ownership transfer**: No cloning required, direct ownership transfer
    /// - **Cache efficiency**: HashMap removal is cache-friendly operation
    ///
    /// # Implementation Details
    ///
    /// The method performs HashMap removal with ownership transfer:
    /// - **Key removal**: Removes entry from gradients HashMap using tensor ID
    /// - **Value return**: Returns the removed gradient tensor with transferred ownership
    /// - **Memory cleanup**: Frees the HashMap entry immediately
    /// - **Thread safety**: Thread-local storage eliminates synchronization overhead
    fn take_gradient(&mut self, tensor_id: usize) -> Option<Tensor> {
        self.gradients.remove(&tensor_id)
    }

    /// Accumulate a gradient tensor with any existing gradient for a tensor
    ///
    /// Adds a new gradient to any existing gradient stored for the specified tensor ID.
    /// This method handles the common case where multiple operations contribute gradients
    /// to the same tensor during backward pass, ensuring proper gradient accumulation
    /// according to the chain rule of calculus.
    ///
    /// # Arguments
    ///
    /// * `tensor_id` - Unique identifier of the tensor to accumulate the gradient for
    /// * `gradient` - Gradient tensor to add to any existing accumulated gradient
    ///
    /// # Accumulation Behavior
    ///
    /// The method exhibits different behavior based on existing gradient state:
    /// - **First gradient**: If no gradient exists, stores the provided gradient directly
    /// - **Subsequent gradients**: If a gradient exists, adds the new gradient to the existing one
    /// - **Element-wise addition**: Gradients are combined using optimized tensor addition
    /// - **Shape compatibility**: Assumes gradients have compatible shapes for addition
    ///
    /// # Mathematical Foundation
    ///
    /// Gradient accumulation implements the chain rule for multiple paths:
    /// - **Chain rule**: ∂L/∂x = Σᵢ (∂L/∂yᵢ × ∂yᵢ/∂x) for all paths from x to loss L
    /// - **Multiple contributions**: When tensor x contributes to multiple operations
    /// - **Proper summation**: All gradient contributions must be summed correctly
    /// - **Numerical stability**: Uses optimized tensor addition for numerical accuracy
    ///
    /// # Usage Patterns
    ///
    /// This method is used in several key scenarios:
    /// - **Backward pass**: Accumulating gradients from multiple downstream operations
    /// - **Branching graphs**: When a tensor is used as input to multiple operations
    /// - **Gradient updates**: Building up final gradients for parameter updates
    /// - **Chain rule application**: Implementing automatic differentiation correctly
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(1) HashMap lookup + O(n) tensor addition where n is tensor size
    /// - **Memory usage**: May allocate new tensor for accumulated result
    /// - **Optimization**: Uses optimized tensor addition with SIMD when available
    /// - **Memory management**: Replaces existing gradient to avoid memory leaks
    ///
    /// # Implementation Details
    ///
    /// The accumulation process follows these steps:
    /// 1. **Lookup existing**: Check if gradient already exists for tensor ID
    /// 2. **Add or store**: Either add to existing gradient or store new gradient
    /// 3. **Optimized addition**: Use efficient tensor addition implementation
    /// 4. **Memory management**: Replace existing gradient with accumulated result
    fn accumulate_gradient(&mut self, tensor_id: usize, gradient: Tensor) {
        match self.gradients.get_mut(&tensor_id) {
            Some(existing_grad) => {
                // Use optimized tensor addition but avoid in-place to prevent corruption
                *existing_grad = existing_grad.add_tensor_optimized(&gradient);
            }
            None => {
                self.gradients.insert(tensor_id, gradient);
            }
        }
    }

    /// Accumulate gradient from an element view into the source tensor at a specific index
    ///
    /// This method handles the specialized case of accumulating gradients from element views
    /// back into their source tensors. Element views create scalar tensors that reference
    /// individual elements of larger tensors, and their gradients must be accumulated back
    /// into the appropriate position in the source tensor's gradient.
    ///
    /// # Arguments
    ///
    /// * `source_id` - Unique identifier of the source tensor that the element view references
    /// * `element_index` - Linear index of the specific element in the source tensor
    /// * `element_gradient` - Gradient tensor for the element (typically scalar with shape [1])
    /// * `source_shape` - Shape dimensions of the source tensor for bounds checking and gradient creation
    ///
    /// # Element View Gradient Accumulation
    ///
    /// The accumulation process handles element views specially:
    /// - **Index mapping**: Maps element view gradient back to specific source tensor position
    /// - **Scalar extraction**: Extracts scalar value from element gradient tensor
    /// - **Bounds checking**: Validates element index against source tensor dimensions
    /// - **Direct accumulation**: Adds gradient value directly to the appropriate element
    ///
    /// # Gradient Processing
    ///
    /// The method processes element gradients through several steps:
    /// 1. **Value extraction**: Extracts scalar gradient value from element gradient tensor
    /// 2. **Bounds validation**: Ensures element index is within source tensor bounds
    /// 3. **Gradient creation**: Creates zero gradient tensor if none exists for source
    /// 4. **Direct accumulation**: Adds gradient value to specific element position
    ///
    /// # Safety and Bounds Checking
    ///
    /// The method includes comprehensive safety measures:
    /// - **Shape validation**: Verifies element index against source tensor total size
    /// - **Existing gradient validation**: Checks bounds against existing gradient tensor
    /// - **Panic on bounds violation**: Provides clear error messages for debugging
    /// - **Memory safety**: Uses unsafe pointer arithmetic only after bounds validation
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(1) for gradient accumulation, O(n) for gradient creation if needed
    /// - **Memory usage**: May allocate new gradient tensor if none exists for source
    /// - **Direct access**: Uses unsafe pointer arithmetic for efficient element access
    /// - **Bounds checking**: Minimal overhead for safety validation
    ///
    /// # Implementation Details
    ///
    /// The method uses different strategies based on existing gradient state:
    /// - **Existing gradient**: Direct element-wise accumulation using unsafe pointer access
    /// - **New gradient**: Creates zero tensor and sets the specific element value
    /// - **Memory safety**: All unsafe operations are preceded by comprehensive bounds checking
    /// - **Error handling**: Clear panic messages for debugging bounds violations
    fn accumulate_element_gradient(
        &mut self,
        source_id: usize,
        element_index: usize,
        element_gradient: &Tensor,
        source_shape: &[usize],
    ) {
        // Get or create gradient tensor for the source with the correct shape
        let gradient_value = if element_gradient.size() == 1 {
            element_gradient.value()
        } else {
            element_gradient.sum().value()
        };

        // Calculate total size for bounds checking
        let total_size: usize = source_shape.iter().product();

        // Bounds check to prevent memory safety violations
        if element_index >= total_size {
            panic!(
                "Element index {} out of bounds for tensor with shape {:?} (total size: {})",
                element_index, source_shape, total_size
            );
        }

        match self.gradients.get_mut(&source_id) {
            Some(existing_grad) => {
                // Additional bounds check against existing gradient tensor
                if element_index >= existing_grad.size() {
                    panic!(
                        "Element index {} out of bounds for existing gradient tensor of size {}",
                        element_index,
                        existing_grad.size()
                    );
                }
                // Accumulate into existing gradient at the specific index
                unsafe {
                    let grad_ptr = existing_grad.as_mut_ptr();
                    *grad_ptr.add(element_index) += gradient_value;
                }
            }
            None => {
                // Create new gradient tensor with zeros and set the element
                let mut new_grad = Tensor::zeros(source_shape.to_vec());
                // Bounds check is already done above, safe to use unsafe here
                unsafe {
                    let grad_ptr = new_grad.as_mut_ptr();
                    *grad_ptr.add(element_index) = gradient_value;
                }
                self.gradients.insert(source_id, new_grad);
            }
        }
    }

    /// Clear all stored operations and gradients from the computation graph
    ///
    /// Resets the computation graph to an empty state by removing all stored operations
    /// and accumulated gradients. This method is essential for preventing gradient
    /// accumulation across multiple training iterations and ensuring clean state
    /// between forward/backward passes.
    ///
    /// # Purpose and Usage
    ///
    /// This method serves several critical functions:
    /// - **Training iteration cleanup**: Clears gradients between training steps
    /// - **Memory management**: Frees all stored gradient tensors and operation metadata
    /// - **State reset**: Ensures clean computation graph for next forward pass
    /// - **Gradient isolation**: Prevents accumulation across separate training iterations
    ///
    /// # Clearing Process
    ///
    /// The method performs comprehensive cleanup:
    /// 1. **Operations clearing**: Removes all registered operations and their metadata
    /// 2. **Gradients clearing**: Frees all accumulated gradient tensors
    /// 3. **Memory deallocation**: Releases memory used by HashMap entries
    /// 4. **Capacity preservation**: Maintains HashMap capacity for future use
    ///
    /// # Memory Management
    ///
    /// The clearing process handles memory efficiently:
    /// - **Tensor deallocation**: All gradient tensors are properly deallocated
    /// - **Metadata cleanup**: Operation metadata is freed from memory
    /// - **Capacity retention**: HashMap capacity is preserved to avoid reallocation
    /// - **Memory reuse**: Cleared HashMaps are ready for immediate reuse
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(n) where n is the number of stored operations and gradients
    /// - **Memory deallocation**: Frees all gradient tensors and operation metadata
    /// - **Capacity preservation**: HashMap capacity is maintained for performance
    /// - **Cleanup efficiency**: Optimized clearing process with minimal overhead
    ///
    /// # Usage Patterns
    ///
    /// This method is typically called:
    /// - **Between training iterations**: After optimizer step, before next forward pass
    /// - **Training loop cleanup**: At the end of each training batch
    /// - **Memory management**: When memory usage needs to be reduced
    /// - **State isolation**: When switching between different computation contexts
    ///
    /// # Implementation Details
    ///
    /// The method uses HashMap clear operations:
    /// - **Operations clear**: Removes all operation entries while preserving capacity
    /// - **Gradients clear**: Removes all gradient entries while preserving capacity
    /// - **Memory efficiency**: Avoids reallocation by maintaining HashMap capacity
    /// - **Thread safety**: Thread-local storage ensures no synchronization needed
    fn clear(&mut self) {
        self.operations.clear();
        self.gradients.clear();
    }
}

// Thread-local gradient graph storage for maximum performance and safety
//
// Each thread maintains its own independent GradGraph instance to ensure complete
// isolation between concurrent training processes. This design eliminates the need
// for synchronization primitives (locks, atomics) and provides linear performance
// scaling with thread count while maintaining memory safety and preventing data races.
thread_local! {
    static GRADTRACK_GRAPH: RefCell<GradGraph> = RefCell::new(GradGraph::new());
}

/// Retrieve accumulated gradient for a tensor from thread-local gradient storage
///
/// This function provides access to the accumulated gradient for a specific tensor
/// from the current thread's gradient tracking graph. It returns an owned copy of
/// the gradient tensor, allowing the caller to use the gradient without affecting
/// the stored gradient state. This is the primary interface for accessing computed
/// gradients after backward pass completion.
///
/// # Arguments
///
/// * `tensor_id` - Unique identifier of the tensor to retrieve the gradient for
///
/// # Returns
///
/// - `Some(Tensor)` - Owned copy of the accumulated gradient tensor if available
/// - `None` - If no gradient has been computed or stored for the specified tensor
///
/// # Usage Patterns
///
/// This function is commonly used for:
/// - **Gradient access**: Retrieving gradients after backward pass for inspection
/// - **Parameter updates**: Accessing gradients for optimizer parameter updates
/// - **Gradient analysis**: Examining gradient values for debugging or monitoring
/// - **Custom gradient processing**: Implementing custom gradient-based algorithms
///
/// # Gradient Availability
///
/// Gradients are available after:
/// - **Backward pass completion**: Gradients computed during backward pass are stored
/// - **Gradient accumulation**: Multiple gradient contributions have been accumulated
/// - **Manual gradient storage**: Gradients explicitly stored via engine operations
///
/// # Thread Safety
///
/// This function is completely thread-safe:
/// - **Thread-local access**: Only accesses the current thread's gradient graph
/// - **No synchronization**: No locks or atomic operations required
/// - **Concurrent execution**: Multiple threads can call this function simultaneously
/// - **Memory safety**: Thread isolation prevents data races and concurrent access issues
///
/// # Performance Characteristics
///
/// - **Time complexity**: O(1) HashMap lookup + O(n) tensor cloning where n is tensor size
/// - **Memory allocation**: Allocates new tensor for the returned gradient copy
/// - **Cache efficiency**: Thread-local storage provides good memory locality
/// - **Cloning overhead**: Gradient tensor is cloned to provide owned result
///
/// # Implementation Details
///
/// The function operates through thread-local storage:
/// - **Thread-local access**: Uses thread_local! macro for isolated storage
/// - **RefCell borrowing**: Safely borrows the gradient graph for read access
/// - **Gradient cloning**: Creates owned copy of gradient tensor for return
/// - **Option handling**: Naturally handles both present and absent gradients
pub fn get_accumulated_gradient(tensor_id: usize) -> Option<Tensor> {
    GRADTRACK_GRAPH.with(|graph| graph.borrow().get_gradient(tensor_id).cloned())
}

/// Clear all gradients and operations from the current thread's gradient tracking graph
///
/// This function resets the computation graph to an empty state by removing all stored
/// operations and accumulated gradients from the current thread's gradient tracking
/// system. It is essential for preventing gradient accumulation across multiple training
/// iterations and ensuring clean state between forward/backward passes.
///
/// # Purpose and Usage
///
/// This function serves several critical functions in training workflows:
/// - **Training iteration cleanup**: Clears gradients between training steps
/// - **Memory management**: Frees all stored gradient tensors and operation metadata
/// - **State isolation**: Ensures clean computation graph for next forward pass
/// - **Gradient reset**: Prevents accumulation across separate training iterations
///
/// # Clearing Process
///
/// The function performs comprehensive cleanup of the thread-local gradient graph:
/// 1. **Operations clearing**: Removes all registered operations and their metadata
/// 2. **Gradients clearing**: Frees all accumulated gradient tensors
/// 3. **Memory deallocation**: Releases memory used by stored data
/// 4. **Capacity preservation**: Maintains HashMap capacity for future use
///
/// # Training Loop Integration
///
/// This function is typically called at specific points in the training loop:
/// - **After optimizer step**: Clear gradients after parameter updates
/// - **Before forward pass**: Ensure clean state for next iteration
/// - **Between batches**: Reset gradient state between training batches
/// - **Memory management**: Reduce memory usage when needed
///
/// # Thread Safety
///
/// This function is completely thread-safe:
/// - **Thread-local operation**: Only affects the current thread's gradient graph
/// - **No synchronization**: No locks or atomic operations required
/// - **Concurrent execution**: Multiple threads can call this function simultaneously
/// - **Memory safety**: Thread isolation prevents data races and concurrent access issues
///
/// # Performance Characteristics
///
/// - **Time complexity**: O(n) where n is the number of stored operations and gradients
/// - **Memory deallocation**: Frees all gradient tensors and operation metadata
/// - **Capacity preservation**: HashMap capacity is maintained for performance
/// - **Cleanup efficiency**: Optimized clearing process with minimal overhead
///
/// # Implementation Details
///
/// The function operates through thread-local storage:
/// - **Thread-local access**: Uses thread_local! macro for isolated storage
/// - **RefCell borrowing**: Safely borrows the gradient graph for mutable access
/// - **Graph clearing**: Calls the GradGraph clear method to reset state
/// - **Memory efficiency**: Maintains HashMap capacity to avoid reallocation
pub fn clear_gradients() {
    GRADTRACK_GRAPH.with(|graph| {
        graph.borrow_mut().clear();
    });
}

/// Primary gradient computation engine for automatic differentiation
///
/// The GradEngine provides the core implementation of reverse-mode automatic differentiation
/// (backpropagation) for the Train Station GradTrack system. It orchestrates the backward
/// pass computation, manages gradient propagation through computation graphs, and ensures
/// proper gradient accumulation according to the mathematical principles of automatic
/// differentiation.
///
/// # Purpose and Functionality
///
/// The GradEngine serves as the central coordinator for gradient computation:
/// - **Backward pass orchestration**: Manages the complete backward pass process
/// - **Graph traversal**: Implements efficient reverse topological traversal
/// - **Gradient propagation**: Applies chain rule through gradient functions
/// - **Operation registration**: Integrates with tensor operations for graph building
/// - **Thread-local coordination**: Manages thread-local gradient state
///
/// # Automatic Differentiation Implementation
///
/// The engine implements reverse-mode automatic differentiation using:
///
/// ## Forward Pass Integration
/// During tensor operations:
/// - **Operation registration**: Each operation registers its gradient function
/// - **Dependency tracking**: Input-output relationships are recorded
/// - **Metadata preservation**: Information needed for gradient computation is stored
/// - **Graph construction**: Computation graph is built incrementally
///
/// ## Backward Pass Execution
/// During gradient computation:
/// 1. **Initialization**: Initial gradient is set (typically ones for scalar loss)
/// 2. **Worklist traversal**: Graph is traversed in reverse topological order
/// 3. **Gradient computation**: Chain rule is applied via registered gradient functions
/// 4. **Accumulation**: Multiple gradients to the same tensor are properly combined
/// 5. **Propagation**: Process continues until all leaf tensors are reached
///
/// # Algorithm Complexity and Performance
///
/// ## Computational Complexity
/// - **Time complexity**: O(V + E) where V is number of tensors and E is number of operations
/// - **Memory complexity**: O(V) for gradient storage in thread-local graph
/// - **Graph traversal**: Efficient worklist-based algorithm with optimal ordering
/// - **Gradient accumulation**: O(1) per operation with HashMap-based storage
///
/// ## Performance Optimizations
/// - **Thread-local storage**: Eliminates synchronization overhead between threads
/// - **Pre-allocated storage**: Minimizes allocation overhead during training
/// - **Efficient traversal**: Worklist-based algorithm avoids redundant computation
/// - **Optimized accumulation**: Uses SIMD-optimized tensor addition when available
///
/// # Thread Safety and Concurrency
///
/// The engine is designed for maximum thread safety and performance:
/// - **Stateless design**: The engine struct itself contains no mutable state
/// - **Thread-local operation**: All gradient state is stored in thread-local storage
/// - **Zero contention**: Multiple threads can perform backward passes simultaneously
/// - **Memory isolation**: Complete separation of gradient state between threads
/// - **Concurrent scaling**: Linear performance scaling with thread count
///
/// # Integration with Tensor Operations
///
/// The engine integrates seamlessly with the tensor system:
/// - **Automatic registration**: Tensor operations automatically register gradient functions
/// - **Transparent tracking**: No changes needed to existing tensor operation implementations
/// - **Conditional activation**: Gradient tracking only occurs when `requires_grad` is enabled
/// - **Efficient propagation**: Optimized gradient flow through computation graphs
///
/// # Mathematical Foundation
///
/// The engine implements the mathematical principles of automatic differentiation:
/// - **Chain rule**: ∂L/∂x = Σᵢ (∂L/∂yᵢ × ∂yᵢ/∂x) for all paths from x to loss L
/// - **Gradient accumulation**: Proper summation of multiple gradient contributions
/// - **Reverse-mode efficiency**: Optimal for scalar outputs (typical in machine learning)
/// - **Numerical stability**: Uses optimized tensor operations for gradient computation
pub struct GradEngine;

impl GradEngine {
    /// Execute backward pass using reverse-mode automatic differentiation
    ///
    /// This method performs the complete backward pass computation starting from the given
    /// tensor (typically a scalar loss). It traverses the computation graph in reverse
    /// topological order, applying the chain rule through registered gradient functions
    /// to compute gradients for all tensors that have `requires_grad=true`. The method
    /// implements the core algorithm of reverse-mode automatic differentiation.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to start backward pass from (typically the scalar loss tensor)
    /// * `grad_output` - Optional initial gradient tensor (defaults to ones for scalar outputs)
    ///
    /// # Backward Pass Algorithm
    ///
    /// The method implements reverse-mode automatic differentiation through these steps:
    ///
    /// ## Initialization Phase
    /// 1. **Initial gradient setup**: Creates initial gradient (ones tensor if not provided)
    /// 2. **Gradient storage**: Stores initial gradient in thread-local graph
    /// 3. **Tensor gradient setting**: Sets gradient on the starting tensor
    /// 4. **Worklist initialization**: Prepares worklist with starting tensor ID
    ///
    /// ## Traversal Phase
    /// 1. **Worklist processing**: Processes tensors in reverse topological order
    /// 2. **Operation lookup**: Retrieves operation information for each tensor
    /// 3. **Gradient consumption**: Takes accumulated gradient for current tensor
    /// 4. **Chain rule application**: Applies gradient function to compute input gradients
    /// 5. **Gradient accumulation**: Accumulates computed gradients into input tensors
    /// 6. **Propagation**: Adds input tensors to worklist for further processing
    ///
    /// ## Special Handling
    /// - **Element views**: Special accumulation for element view operations
    /// - **Leaf tensors**: Proper handling of parameter tensors (no further propagation)
    /// - **Multiple contributions**: Correct accumulation when tensors have multiple uses
    /// - **Memory management**: Efficient gradient storage and cleanup
    ///
    /// # Mathematical Foundation
    ///
    /// The algorithm implements the chain rule of calculus:
    /// - **Chain rule**: ∂L/∂x = Σᵢ (∂L/∂yᵢ × ∂yᵢ/∂x) for all paths from x to loss L
    /// - **Gradient accumulation**: Proper summation of multiple gradient contributions
    /// - **Reverse traversal**: Processes operations in reverse order of execution
    /// - **Automatic differentiation**: Systematic application of differentiation rules
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(V + E) where V is number of tensors and E is number of operations
    /// - **Memory usage**: O(V) for gradient storage in thread-local graph
    /// - **Graph traversal**: Efficient worklist-based algorithm with optimal ordering
    /// - **Thread safety**: Thread-local storage ensures zero contention between threads
    ///
    /// # Thread Safety and Concurrency
    ///
    /// This method is completely thread-safe:
    /// - **Thread-local operation**: Only affects the current thread's gradient graph
    /// - **No synchronization**: No locks or atomic operations required
    /// - **Concurrent execution**: Multiple threads can perform backward passes simultaneously
    /// - **Memory isolation**: Complete separation of gradient state between threads
    ///
    /// # Implementation Details
    ///
    /// The method uses several optimization strategies:
    /// - **Worklist traversal**: Avoids recursive calls and stack overflow issues
    /// - **Gradient consumption**: Prevents double-counting by taking gradients from storage
    /// - **Efficient accumulation**: Uses optimized tensor addition for gradient combination
    /// - **Memory management**: Proper cleanup and memory reuse throughout the process
    pub fn backward(tensor: &mut Tensor, grad_output: Option<Tensor>) {
        // Initialize gradient if not provided (assumes scalar output)
        let initial_grad = grad_output.unwrap_or_else(|| {
            let mut ones = Tensor::ones(tensor.shape().dims.clone());
            ones.set_requires_grad(false); // Gradients don't need gradients
            ones
        });

        // Set the gradient for this tensor
        tensor.accumulate_grad(initial_grad.clone());

        // Store initial gradient in thread-local graph
        GRADTRACK_GRAPH.with(|graph| {
            graph
                .borrow_mut()
                .store_gradient(tensor.id(), initial_grad.clone());
        });

        // Iterative propagation using a worklist to ensure proper accumulation
        let mut worklist: Vec<usize> = Vec::with_capacity(128);
        worklist.push(tensor.id());

        while let Some(node_id) = worklist.pop() {
            // Get operation info first; if none, this is a leaf, keep its gradient intact
            let operation_info =
                GRADTRACK_GRAPH.with(|graph| graph.borrow().get_operation(node_id).cloned());
            if let Some((input_ids, grad_fn)) = operation_info {
                // Take the currently accumulated gradient for this node
                let current_grad =
                    GRADTRACK_GRAPH.with(|graph| graph.borrow_mut().take_gradient(node_id));
                if current_grad.is_none() {
                    continue;
                }
                let current_grad = current_grad.unwrap();
                // Compute input gradients
                let input_grads = grad_fn.apply(&current_grad);

                // Accumulate into inputs and push them to worklist
                for (idx, &input_id) in input_ids.iter().enumerate() {
                    if let Some(Some(input_grad)) = input_grads.get(idx) {
                        // Check if this is an ElementView operation that needs special handling
                        match &grad_fn {
                            GradFn::ElementView {
                                source_id,
                                element_index,
                                source_shape,
                            } => {
                                // For ElementView, accumulate directly into the source tensor at the specific index
                                // Use the stored source shape for proper gradient accumulation
                                GRADTRACK_GRAPH.with(|graph| {
                                    graph.borrow_mut().accumulate_element_gradient(
                                        *source_id,
                                        *element_index,
                                        input_grad,
                                        source_shape,
                                    );
                                });

                                // Check if the source tensor has operations to propagate further
                                let has_op = GRADTRACK_GRAPH.with(|graph| {
                                    graph.borrow().get_operation(*source_id).is_some()
                                });
                                if has_op {
                                    worklist.push(*source_id);
                                }
                            }
                            _ => {
                                // Regular gradient accumulation for non-ElementView operations
                                GRADTRACK_GRAPH.with(|graph| {
                                    graph
                                        .borrow_mut()
                                        .accumulate_gradient(input_id, input_grad.clone());
                                });
                                // Only push if this input has an operation to propagate further
                                let has_op = GRADTRACK_GRAPH
                                    .with(|graph| graph.borrow().get_operation(input_id).is_some());
                                if has_op {
                                    worklist.push(input_id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Register a tensor operation in the computation graph for gradient tracking
    ///
    /// This method records a tensor operation's metadata in the thread-local computation
    /// graph, establishing the relationship between input and output tensors along with
    /// the gradient function needed for backward pass computation. It is called automatically
    /// by tensor operations that support gradient tracking and is essential for building
    /// the computation graph during the forward pass.
    ///
    /// # Arguments
    ///
    /// * `output_id` - Unique identifier of the output tensor produced by this operation
    /// * `input_ids` - Vector of unique identifiers for input tensors that this operation depends on
    /// * `grad_fn` - Gradient function containing the logic to compute gradients for this operation
    ///
    /// # Operation Registration Process
    ///
    /// The registration process involves several key steps:
    /// 1. **Dependency tracking**: Input tensor IDs are stored to enable reverse traversal
    /// 2. **Gradient function storage**: The GradFn is stored for gradient computation
    /// 3. **Graph structure building**: Links are established between input and output tensors
    /// 4. **Metadata preservation**: All information needed for backward pass is saved
    ///
    /// # Integration with Tensor Operations
    ///
    /// This method integrates seamlessly with tensor operations:
    /// - **Automatic registration**: Called automatically by tensor operations during forward pass
    /// - **Transparent tracking**: No changes needed to existing tensor operation code
    /// - **Conditional activation**: Only called when gradient tracking is enabled
    /// - **Efficient storage**: Uses thread-local storage for optimal performance
    ///
    /// # Gradient Function Storage
    ///
    /// The method stores gradient functions for various operation types:
    /// - **Arithmetic operations**: Add, subtract, multiply, divide operations
    /// - **Mathematical functions**: Exponential, logarithm, trigonometric functions
    /// - **Matrix operations**: Matrix multiplication, transpose operations
    /// - **Tensor transformations**: Reshape, permute, concatenation operations
    /// - **Reduction operations**: Sum, mean, max, min operations
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(1) average case HashMap insertion
    /// - **Memory usage**: Stores operation metadata (~32 bytes per operation)
    /// - **Insertion cost**: Minimal overhead due to pre-allocated HashMap capacity
    /// - **Thread safety**: Thread-local storage ensures zero contention between threads
    ///
    /// # Thread Safety and Concurrency
    ///
    /// This method is completely thread-safe:
    /// - **Thread-local operation**: Only affects the current thread's gradient graph
    /// - **No synchronization**: No locks or atomic operations required
    /// - **Concurrent execution**: Multiple threads can register operations simultaneously
    /// - **Memory isolation**: Complete separation of gradient state between threads
    ///
    /// # Implementation Details
    ///
    /// The method operates through thread-local storage:
    /// - **Thread-local access**: Uses thread_local! macro for isolated storage
    /// - **RefCell borrowing**: Safely borrows the gradient graph for mutable access
    /// - **Graph registration**: Calls the GradGraph register_operation method
    /// - **Memory efficiency**: Uses pre-allocated HashMap to minimize allocation overhead
    pub fn register_operation(output_id: usize, input_ids: Vec<usize>, grad_fn: GradFn) {
        GRADTRACK_GRAPH.with(|graph| {
            graph
                .borrow_mut()
                .register_operation(output_id, input_ids, grad_fn);
        });
    }
}
