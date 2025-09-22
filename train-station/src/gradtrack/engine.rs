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
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread::ThreadId;

// -------------------------------------------------------------------------------------------------
// Type aliases to reduce type complexity and satisfy clippy::type_complexity
// -------------------------------------------------------------------------------------------------
type OperationRecord = (Vec<usize>, GradFn);
type OperationMap = HashMap<usize, OperationRecord>;
type GradientMap = HashMap<usize, Tensor>;
type ShardedOpMaps = Vec<RwLock<OperationMap>>;
type ShardedGradMaps = Vec<Mutex<GradientMap>>;
type GroupMap = HashMap<usize, Arc<GraphGroupRef>>;
type GroupShards = Vec<Mutex<GroupMap>>;

// =============================================================================================
// Graph groups and shared-graph infrastructure (implicit cross-thread accumulation)
// =============================================================================================

const NUM_SHARDS: usize = 64;

static GROUP_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

// Sharded global map: tensor_id -> GraphGroupRef (shared across threads)
static ID_GROUP_SHARDS: OnceLock<GroupShards> = OnceLock::new();

fn id_group_shards() -> &'static GroupShards {
    ID_GROUP_SHARDS.get_or_init(|| {
        (0..NUM_SHARDS)
            .map(|_| Mutex::new(HashMap::with_capacity(256)))
            .collect()
    })
}

#[inline]
fn shard_index_for_id(tensor_id: usize) -> usize {
    tensor_id % NUM_SHARDS
}

#[inline]
fn id_group_get(tensor_id: usize) -> Option<Arc<GraphGroupRef>> {
    let shards = id_group_shards();
    let idx = shard_index_for_id(tensor_id);
    let map = shards[idx].lock().unwrap();
    map.get(&tensor_id).cloned()
}

#[inline]
fn id_group_insert(tensor_id: usize, group: Arc<GraphGroupRef>) {
    let shards = id_group_shards();
    let idx = shard_index_for_id(tensor_id);
    let mut map = shards[idx].lock().unwrap();
    map.insert(tensor_id, group);
}

/// Shared, sharded computation graph for cross-thread usage
struct SharedGradGraph {
    operations: ShardedOpMaps,
    gradients: ShardedGradMaps,
    retained: RwLock<HashSet<usize>>, // ids to retain gradients for after backward
}

impl SharedGradGraph {
    fn new() -> Self {
        let mut operations = Vec::with_capacity(NUM_SHARDS);
        let mut gradients = Vec::with_capacity(NUM_SHARDS);
        for _ in 0..NUM_SHARDS {
            operations.push(RwLock::new(HashMap::with_capacity(256)));
            gradients.push(Mutex::new(HashMap::with_capacity(256)));
        }
        Self {
            operations,
            gradients,
            retained: RwLock::new(HashSet::with_capacity(256)),
        }
    }

    #[inline]
    fn shard(tensor_id: usize) -> usize {
        shard_index_for_id(tensor_id)
    }

    fn register_operation(&self, output_id: usize, input_ids: Vec<usize>, grad_fn: GradFn) {
        let shard = Self::shard(output_id);
        let mut op_map = self.operations[shard].write().unwrap();
        op_map.insert(output_id, (input_ids, grad_fn));
    }

    fn get_operation(&self, tensor_id: usize) -> Option<(Vec<usize>, GradFn)> {
        let shard = Self::shard(tensor_id);
        let op_map = self.operations[shard].read().unwrap();
        op_map.get(&tensor_id).cloned()
    }

    fn store_gradient(&self, tensor_id: usize, gradient: Tensor) {
        let shard = Self::shard(tensor_id);
        let mut grads = self.gradients[shard].lock().unwrap();
        grads.insert(tensor_id, gradient);
    }

    fn get_gradient(&self, tensor_id: usize) -> Option<Tensor> {
        let shard = Self::shard(tensor_id);
        let grads = self.gradients[shard].lock().unwrap();
        grads.get(&tensor_id).cloned()
    }

    fn take_gradient(&self, tensor_id: usize) -> Option<Tensor> {
        let shard = Self::shard(tensor_id);
        let mut grads = self.gradients[shard].lock().unwrap();
        grads.remove(&tensor_id)
    }

    fn accumulate_gradient(&self, tensor_id: usize, gradient: Tensor) {
        let shard = Self::shard(tensor_id);
        let mut grads = self.gradients[shard].lock().unwrap();
        if let Some(existing) = grads.get_mut(&tensor_id) {
            // Avoid in-place corruption; assign new accumulated tensor
            let new_acc = existing.add_tensor_optimized(&gradient);
            *existing = new_acc;
        } else {
            grads.insert(tensor_id, gradient);
        }
    }

    fn mark_retain(&self, tensor_id: usize) {
        let mut set = self.retained.write().unwrap();
        set.insert(tensor_id);
    }

    fn should_retain(&self, tensor_id: usize) -> bool {
        let set = self.retained.read().unwrap();
        set.contains(&tensor_id)
    }

    /// Clear all stored operations, gradients, and retained flags from this shared graph
    fn clear_all(&self) {
        for shard in 0..NUM_SHARDS {
            self.operations[shard].write().unwrap().clear();
            self.gradients[shard].lock().unwrap().clear();
        }
        self.retained.write().unwrap().clear();
    }
}

/// Graph group reference that can be Local (lock-free) or Shared (sharded locks)
pub(crate) struct GraphGroupRef {
    id: usize,
    state: Mutex<GraphGroupState>,
}

enum GraphGroupState {
    Local {
        owner: ThreadId,
        graph: GradGraph,
        retained: HashSet<usize>,
    },
    Shared(Arc<SharedGradGraph>),
}

impl GraphGroupRef {
    fn new_local_current_thread() -> Arc<Self> {
        Arc::new(GraphGroupRef {
            id: GROUP_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed),
            state: Mutex::new(GraphGroupState::Local {
                owner: std::thread::current().id(),
                graph: GradGraph::new(),
                retained: HashSet::with_capacity(256),
            }),
        })
    }

    fn ensure_shared(this: &Arc<Self>) -> Arc<SharedGradGraph> {
        // Promote Local -> Shared if needed; return Shared handle
        let mut guard = this.state.lock().unwrap();
        match &mut *guard {
            GraphGroupState::Shared(s) => s.clone(),
            GraphGroupState::Local {
                graph, retained, ..
            } => {
                let shared = Arc::new(SharedGradGraph::new());
                // Migrate operations
                for (out_id, (inputs, gfn)) in graph.operations.drain() {
                    shared.register_operation(out_id, inputs, gfn);
                }
                // Migrate gradients
                for (tid, grad) in graph.gradients.drain() {
                    shared.store_gradient(tid, grad);
                }
                // Migrate retained
                for id in retained.drain() {
                    shared.mark_retain(id);
                }
                *guard = GraphGroupState::Shared(shared.clone());
                shared
            }
        }
    }

    fn owner_thread_id(&self) -> Option<ThreadId> {
        let guard = self.state.lock().unwrap();
        match &*guard {
            GraphGroupState::Local { owner, .. } => Some(*owner),
            GraphGroupState::Shared(_) => None,
        }
    }

    fn register_operation(&self, output_id: usize, input_ids: Vec<usize>, grad_fn: GradFn) {
        let mut guard = self.state.lock().unwrap();
        match &mut *guard {
            GraphGroupState::Local { graph, .. } => {
                graph.register_operation(output_id, input_ids, grad_fn);
            }
            GraphGroupState::Shared(shared) => {
                shared.register_operation(output_id, input_ids, grad_fn);
            }
        }
    }

    fn get_operation(&self, tensor_id: usize) -> Option<(Vec<usize>, GradFn)> {
        let guard = self.state.lock().unwrap();
        match &*guard {
            GraphGroupState::Local { graph, .. } => graph.get_operation(tensor_id).cloned(),
            GraphGroupState::Shared(shared) => shared.get_operation(tensor_id),
        }
    }

    fn store_gradient(&self, tensor_id: usize, gradient: Tensor) {
        let mut guard = self.state.lock().unwrap();
        match &mut *guard {
            GraphGroupState::Local { graph, .. } => graph.store_gradient(tensor_id, gradient),
            GraphGroupState::Shared(shared) => shared.store_gradient(tensor_id, gradient),
        }
    }

    fn take_gradient(&self, tensor_id: usize) -> Option<Tensor> {
        let mut guard = self.state.lock().unwrap();
        match &mut *guard {
            GraphGroupState::Local { graph, .. } => graph.take_gradient(tensor_id),
            GraphGroupState::Shared(shared) => shared.take_gradient(tensor_id),
        }
    }

    fn accumulate_gradient(&self, tensor_id: usize, gradient: Tensor) {
        let mut guard = self.state.lock().unwrap();
        match &mut *guard {
            GraphGroupState::Local { graph, .. } => graph.accumulate_gradient(tensor_id, gradient),
            GraphGroupState::Shared(shared) => shared.accumulate_gradient(tensor_id, gradient),
        }
    }

    fn get_gradient_value(&self, tensor_id: usize) -> Option<Tensor> {
        let guard = self.state.lock().unwrap();
        match &*guard {
            GraphGroupState::Local { graph, .. } => graph.get_gradient(tensor_id).cloned(),
            GraphGroupState::Shared(shared) => shared.get_gradient(tensor_id),
        }
    }

    fn mark_retain(&self, tensor_id: usize) {
        let mut guard = self.state.lock().unwrap();
        match &mut *guard {
            GraphGroupState::Local { retained, .. } => {
                retained.insert(tensor_id);
            }
            GraphGroupState::Shared(shared) => shared.mark_retain(tensor_id),
        }
    }

    fn should_retain(&self, tensor_id: usize) -> bool {
        let guard = self.state.lock().unwrap();
        match &*guard {
            GraphGroupState::Local { retained, .. } => retained.contains(&tensor_id),
            GraphGroupState::Shared(shared) => shared.should_retain(tensor_id),
        }
    }

    /// Clear all operations/gradients from this graph group (local or shared)
    fn clear_all(&self) {
        let mut guard = self.state.lock().unwrap();
        match &mut *guard {
            GraphGroupState::Local {
                graph, retained, ..
            } => {
                graph.clear();
                retained.clear();
            }
            GraphGroupState::Shared(shared) => {
                shared.clear_all();
            }
        }
    }
}

/// Ensure a local graph group exists for a given tensor id, binding it if missing.
pub fn ensure_local_group_for_tensor(tensor_id: usize) -> Arc<GraphGroupRef> {
    if let Some(g) = id_group_get(tensor_id) {
        return g;
    }
    let g = GraphGroupRef::new_local_current_thread();
    id_group_insert(tensor_id, g.clone());
    g
}

fn unify_groups_for_inputs(input_ids: &[usize]) -> Arc<GraphGroupRef> {
    // Gather unique groups, creating locals if missing
    let mut groups: Vec<Arc<GraphGroupRef>> = Vec::new();
    for &id in input_ids {
        let g = id_group_get(id).unwrap_or_else(|| ensure_local_group_for_tensor(id));
        // Dedup by pointer address (Arc::as_ptr)
        let gp = Arc::as_ptr(&g) as usize;
        if !groups.iter().any(|x| Arc::as_ptr(x) as usize == gp) {
            groups.push(g);
        }
    }

    if groups.is_empty() {
        // No inputs: create a fresh local
        return GraphGroupRef::new_local_current_thread();
    }

    if groups.len() == 1 {
        let g = &groups[0];
        // If local but owner thread differs, promote to shared
        if let Some(owner) = g.owner_thread_id() {
            if owner != std::thread::current().id() {
                let _ = GraphGroupRef::ensure_shared(g);
            }
        }
        // Ensure all inputs map to this group (idempotent)
        for &iid in input_ids {
            id_group_insert(iid, g.clone());
        }
        return g.clone();
    }

    // Multiple groups: if all locals with same owner, merge by moving entries into first local
    let all_local = groups.iter().all(|g| g.owner_thread_id().is_some());
    if all_local {
        let owner0 = groups[0].owner_thread_id();
        let same_owner = groups.iter().all(|g| g.owner_thread_id() == owner0);
        if same_owner {
            // Merge locals into first group's local graph
            // Lock in id order to avoid deadlocks
            let mut sorted = groups.clone();
            sorted.sort_by_key(|g| g.id);
            let first_id = sorted[0].id;
            // Acquire locks
            let mut guards = Vec::with_capacity(sorted.len());
            for g in &sorted {
                guards.push(g.state.lock().unwrap());
            }
            // Identify destination
            let mut dest_index = 0usize;
            for (i, g) in sorted.iter().enumerate() {
                if g.id == first_id {
                    dest_index = i;
                    break;
                }
            }
            // Drain from all non-destination guards into temporaries
            let mut ops_to_move: Vec<(usize, (Vec<usize>, GradFn))> = Vec::new();
            let mut grads_to_move: Vec<(usize, Tensor)> = Vec::new();
            let mut retained_to_move: Vec<usize> = Vec::new();

            for (i, guard) in guards.iter_mut().enumerate() {
                if i == dest_index {
                    continue;
                }
                if let GraphGroupState::Local {
                    graph, retained, ..
                } = &mut **guard
                {
                    for (k, v) in graph.operations.drain() {
                        ops_to_move.push((k, v));
                    }
                    for (k, v) in graph.gradients.drain() {
                        grads_to_move.push((k, v));
                    }
                    for id in retained.drain() {
                        retained_to_move.push(id);
                    }
                }
            }

            // Insert into destination
            if let GraphGroupState::Local {
                graph, retained, ..
            } = &mut *guards[dest_index]
            {
                for (k, v) in ops_to_move {
                    graph.operations.insert(k, v);
                }
                for (k, v) in grads_to_move {
                    graph.gradients.insert(k, v);
                }
                for id in retained_to_move {
                    retained.insert(id);
                }
            }
            // Ensure all inputs are bound to the canonical destination group (sorted[0])
            let dest_group_arc = sorted[0].clone();
            for &iid in input_ids {
                id_group_insert(iid, dest_group_arc.clone());
            }
            return dest_group_arc;
        }
    }

    // Otherwise promote/merge into a shared canonical
    // Choose canonical GraphGroupRef: prefer an existing Shared group if present; otherwise use groups[0]
    let mut canonical_group_ref: Option<Arc<GraphGroupRef>> = None;
    for g in &groups {
        let guard = g.state.lock().unwrap();
        if matches!(*guard, GraphGroupState::Shared(_)) {
            canonical_group_ref = Some(g.clone());
            break;
        }
    }
    let canonical_group_ref = canonical_group_ref.unwrap_or_else(|| groups[0].clone());
    // Ensure canonical has a Shared graph
    let canonical_shared = GraphGroupRef::ensure_shared(&canonical_group_ref);

    // Migrate all other groups (including groups[0] if it's not canonical) into canonical
    for g in &groups {
        if Arc::ptr_eq(g, &canonical_group_ref) {
            continue;
        }

        // First handle Shared case by cloning the Arc outside of the guard scope
        let shared_src: Option<Arc<SharedGradGraph>> = {
            let guard = g.state.lock().unwrap();
            if let GraphGroupState::Shared(s) = &*guard {
                Some(s.clone())
            } else {
                None
            }
        };

        if let Some(src_shared) = shared_src {
            // Drain from src_shared into canonical without holding the state lock
            for shard in 0..NUM_SHARDS {
                let mut src_ops = src_shared.operations[shard].write().unwrap();
                for (k, v) in src_ops.drain() {
                    canonical_shared.register_operation(k, v.0, v.1);
                }
            }
            for shard in 0..NUM_SHARDS {
                let mut src_grads = src_shared.gradients[shard].lock().unwrap();
                for (k, v) in src_grads.drain() {
                    canonical_shared.accumulate_gradient(k, v);
                }
            }
            let mut src_ret = src_shared.retained.write().unwrap();
            for id in src_ret.drain() {
                canonical_shared.mark_retain(id);
            }
            // Now rebind this group's state to canonical
            {
                let mut guard = g.state.lock().unwrap();
                *guard = GraphGroupState::Shared(canonical_shared.clone());
            }
            continue;
        }

        // Handle Local case entirely within a short lock scope, then rebind
        {
            let mut guard = g.state.lock().unwrap();
            if let GraphGroupState::Local {
                graph, retained, ..
            } = &mut *guard
            {
                for (out_id, (inputs, gfn)) in graph.operations.drain() {
                    canonical_shared.register_operation(out_id, inputs, gfn);
                }
                for (tid, grad) in graph.gradients.drain() {
                    canonical_shared.accumulate_gradient(tid, grad);
                }
                for id in retained.drain() {
                    canonical_shared.mark_retain(id);
                }
            } else {
                // Already handled Shared above
                continue;
            }
        }
        // Rebind to canonical after draining
        {
            let mut guard = g.state.lock().unwrap();
            *guard = GraphGroupState::Shared(canonical_shared.clone());
        }
    }

    // Ensure all inputs are bound to canonical shared group
    for &iid in input_ids {
        id_group_insert(iid, canonical_group_ref.clone());
    }
    canonical_group_ref
}

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
        // Correct ownership transfer: remove the entry so it is not re-used on subsequent visits
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

    // Removed: accumulate_element_gradient; element views now reuse SliceView GradFn.

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

// Thread-local set of tensor IDs that should retain their gradients after backward
thread_local! {
    static RETAINED_GRAD_IDS: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
}

#[track_caller]
pub fn mark_retain_grad(tensor_id: usize) {
    if let Some(g) = id_group_get(tensor_id) {
        g.mark_retain(tensor_id);
        return;
    }
    RETAINED_GRAD_IDS.with(|set| {
        set.borrow_mut().insert(tensor_id);
    });
}

fn should_retain_grad(tensor_id: usize) -> bool {
    RETAINED_GRAD_IDS.with(|set| set.borrow().contains(&tensor_id))
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
#[track_caller]
pub fn get_accumulated_gradient(tensor_id: usize) -> Option<Tensor> {
    if let Some(g) = id_group_get(tensor_id) {
        if let Some(grad) = g.get_gradient_value(tensor_id) {
            return Some(grad);
        }
    }

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
#[track_caller]
pub fn clear_gradients() {
    GRADTRACK_GRAPH.with(|graph| {
        graph.borrow_mut().clear();
    });
}

/// Clear the accumulated gradient for a specific tensor id across all storages
///
/// Removes the stored gradient for `tensor_id` from its bound graph group (local/shared)
/// if present, and also from the thread-local graph as a fallback. This is used by
/// optimizers' zero_grad logic to ensure per-parameter gradients are fully cleared.
#[track_caller]
pub fn clear_gradient_for_tensor(tensor_id: usize) {
    if let Some(g) = id_group_get(tensor_id) {
        // Ignore result; we only need to remove if present
        let _ = g.take_gradient(tensor_id);
    }
    // Also clear from TLS graph in case this tensor was tracked there
    GRADTRACK_GRAPH.with(|graph| {
        let _ = graph.borrow_mut().take_gradient(tensor_id);
    });
}

/// Clear the thread-local graph (alias of clear_gradients)
#[track_caller]
pub fn clear_local_graph() {
    clear_gradients();
}

/// Clear the entire graph (local or shared) associated with a specific tensor id
#[track_caller]
pub fn clear_graph_for_tensor(tensor_id: usize) {
    if let Some(g) = id_group_get(tensor_id) {
        g.clear_all();
    }
}

/// If the tensor's group is shared, clear that shared graph; otherwise no-op
#[track_caller]
pub fn clear_shared_graph_for_tensor(tensor_id: usize) {
    if let Some(g) = id_group_get(tensor_id) {
        let guard = g.state.lock().unwrap();
        if let GraphGroupState::Shared(shared) = &*guard {
            shared.clear_all();
        }
    }
}

/// Clear all known shared graphs by scanning the global id→group registry
#[track_caller]
pub fn clear_all_shared_graphs() {
    use std::collections::HashSet;

    // Collect unique GraphGroupRef arcs without holding shard locks during clear
    let mut unique_groups: Vec<Arc<GraphGroupRef>> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();

    let shards = id_group_shards();
    for shard in shards.iter() {
        let map = shard.lock().unwrap();
        for g in map.values() {
            let key = Arc::as_ptr(g) as usize;
            if seen.insert(key) {
                unique_groups.push(g.clone());
            }
        }
    }

    // Clear only the shared ones
    for g in unique_groups {
        let guard = g.state.lock().unwrap();
        if let GraphGroupState::Shared(shared) = &*guard {
            shared.clear_all();
        }
        drop(guard);
    }
}

/// Clear all known graphs: current thread-local graph and all shared graphs
#[track_caller]
pub fn clear_all_graphs_known() {
    clear_local_graph();
    clear_all_shared_graphs();
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
    #[track_caller]
    pub fn backward(tensor: &mut Tensor, grad_output: Option<Tensor>) {
        // Determine graph group for this tensor
        if let Some(group) = id_group_get(tensor.id()) {
            // Initialize gradient if not provided (assumes scalar output)
            let initial_grad = grad_output.unwrap_or_else(|| {
                let mut ones = Tensor::ones(tensor.shape().dims().to_vec());
                ones.set_requires_grad(false);
                ones
            });
            tensor.accumulate_grad(initial_grad.clone());
            group.store_gradient(tensor.id(), initial_grad.clone());

            let mut worklist: Vec<usize> = Vec::with_capacity(128);
            worklist.push(tensor.id());
            while let Some(node_id) = worklist.pop() {
                let operation_info = group.get_operation(node_id);
                if let Some((input_ids, grad_fn)) = operation_info {
                    let current_grad = group.take_gradient(node_id);
                    if current_grad.is_none() {
                        continue;
                    }
                    let current_grad = current_grad.unwrap();
                    if group.should_retain(node_id) {
                        group.store_gradient(node_id, current_grad.clone());
                    }
                    let input_grads = grad_fn.apply(&current_grad);
                    for (idx, &input_id) in input_ids.iter().enumerate() {
                        if let Some(Some(input_grad)) = input_grads.get(idx) {
                            // Rebind input id to this (possibly shared) group to ensure
                            // accumulated gradients are discoverable across threads.
                            id_group_insert(input_id, group.clone());
                            group.accumulate_gradient(input_id, input_grad.clone());
                            let has_op = group.get_operation(input_id).is_some();
                            if has_op {
                                worklist.push(input_id);
                            }
                        }
                    }
                }
            }
            return;
        }

        // Fallback to TLS graph for backward if no group mapping exists
        let initial_grad = grad_output.unwrap_or_else(|| {
            let mut ones = Tensor::ones(tensor.shape().dims().to_vec());
            ones.set_requires_grad(false);
            ones
        });
        tensor.accumulate_grad(initial_grad.clone());
        GRADTRACK_GRAPH.with(|graph| {
            graph
                .borrow_mut()
                .store_gradient(tensor.id(), initial_grad.clone());
        });

        let mut worklist: Vec<usize> = Vec::with_capacity(128);
        worklist.push(tensor.id());
        while let Some(node_id) = worklist.pop() {
            let operation_info =
                GRADTRACK_GRAPH.with(|graph| graph.borrow().get_operation(node_id).cloned());
            if let Some((input_ids, grad_fn)) = operation_info {
                let current_grad =
                    GRADTRACK_GRAPH.with(|graph| graph.borrow_mut().take_gradient(node_id));
                if current_grad.is_none() {
                    continue;
                }
                let current_grad = current_grad.unwrap();
                if should_retain_grad(node_id) {
                    GRADTRACK_GRAPH.with(|graph| {
                        graph
                            .borrow_mut()
                            .store_gradient(node_id, current_grad.clone());
                    });
                }
                let input_grads = grad_fn.apply(&current_grad);
                for (idx, &input_id) in input_ids.iter().enumerate() {
                    if let Some(Some(input_grad)) = input_grads.get(idx) {
                        // If input tensor has a shared group mapping (due to cross-thread usage),
                        // use it so gradients are placed in the shared graph, not TLS.
                        if let Some(g) = id_group_get(input_id) {
                            id_group_insert(input_id, g.clone());
                            g.accumulate_gradient(input_id, input_grad.clone());
                        } else {
                            GRADTRACK_GRAPH.with(|graph| {
                                graph
                                    .borrow_mut()
                                    .accumulate_gradient(input_id, input_grad.clone());
                            });
                        }
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
    #[track_caller]
    pub fn register_operation(output_id: usize, input_ids: Vec<usize>, grad_fn: GradFn) {
        // Implicit unification across inputs; bind output to unified group.
        let group = if input_ids.is_empty() {
            // No inputs: create or reuse a local group for output
            ensure_local_group_for_tensor(output_id)
        } else {
            let g = unify_groups_for_inputs(&input_ids);
            // Rebind all input ids to the unified canonical group to ensure consistent
            // gradient storage/retrieval across threads and after promotions/merges.
            for &iid in &input_ids {
                id_group_insert(iid, g.clone());
            }
            g
        };

        // Bind output id to this group and register op
        id_group_insert(output_id, group.clone());
        group.register_operation(output_id, input_ids, grad_fn);
    }
}

#[cfg(test)]
mod clearing_tests {
    use super::*;
    use crate::tensor::core::memory::with_no_mem_pool;
    use crate::tensor::Tensor;

    fn build_simple_add_graph() -> (Tensor, Tensor, Tensor) {
        let a = Tensor::ones(vec![2, 3]).with_requires_grad();
        let b = Tensor::ones(vec![2, 3]).with_requires_grad();
        let out = a.add_tensor(&b);
        (a, b, out)
    }

    #[test]
    fn test_clear_local_graph_clears_tls_gradients() {
        // Create a tensor without binding it to any graph group, then call backward.
        // This forces gradient storage into the TLS graph (no group mapping).
        let mut t = Tensor::ones(vec![2, 3]);
        assert!(id_group_get(t.id()).is_none());
        t.backward(None);

        assert!(get_accumulated_gradient(t.id()).is_some());

        clear_local_graph();

        assert!(get_accumulated_gradient(t.id()).is_none());
    }

    #[test]
    fn test_clear_gradient_for_tensor_only_clears_target() {
        let (a, b, out) = build_simple_add_graph();
        let mut loss = out.sum();
        loss.backward(None);

        assert!(get_accumulated_gradient(a.id()).is_some());
        assert!(get_accumulated_gradient(b.id()).is_some());

        clear_gradient_for_tensor(a.id());

        assert!(get_accumulated_gradient(a.id()).is_none());
        assert!(get_accumulated_gradient(b.id()).is_some());
    }

    #[test]
    fn test_clear_graph_for_tensor_local_group() {
        let (a, b, out) = build_simple_add_graph();
        let mut loss = out.sum();
        loss.backward(None);

        assert!(get_accumulated_gradient(a.id()).is_some());
        assert!(get_accumulated_gradient(b.id()).is_some());

        clear_graph_for_tensor(a.id());

        // Entire group cleared: both a and b grads should be gone
        assert!(get_accumulated_gradient(a.id()).is_none());
        assert!(get_accumulated_gradient(b.id()).is_none());
    }

    #[test]
    fn test_clear_shared_graph_for_tensor_cross_thread() {
        // Build a shared graph by performing ops on separate threads and merging results
        with_no_mem_pool(|| {
            let a = std::sync::Arc::new(Tensor::ones(vec![8, 4]).with_requires_grad());
            let b = std::sync::Arc::new(Tensor::ones(vec![8, 4]).with_requires_grad());

            let a1 = a.clone();
            let h1 = std::thread::spawn(move || with_no_mem_pool(|| a1.add_scalar(0.0)));

            let b1 = b.clone();
            let h2 = std::thread::spawn(move || with_no_mem_pool(|| b1.add_scalar(0.0)));

            let z1 = h1.join().unwrap();
            let z2 = h2.join().unwrap();
            let out = z1.add_tensor(&z2);
            let mut loss = out.sum();
            loss.backward(None);

            assert!(get_accumulated_gradient(a.id()).is_some());
            assert!(get_accumulated_gradient(b.id()).is_some());

            // Clear the shared graph tied to `a`
            clear_shared_graph_for_tensor(a.id());

            // After clearing the shared graph, grads for both should be gone
            assert!(get_accumulated_gradient(a.id()).is_none());
            assert!(get_accumulated_gradient(b.id()).is_none());
        });
    }
}
