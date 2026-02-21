//! Arena allocator for temporary rule-match structures.
//!
//! This module implements Phase 13 memory optimization for rule matching,
//! reducing allocation pressure in hot paths by using arena allocation.

use smallvec::SmallVec;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::mem;
use std::ptr::NonNull;
use std::sync::Arc;

use super::graph::{EdgeId, NodeId};

/// Default arena chunk size (64KB)
const DEFAULT_CHUNK_SIZE: usize = 65536;

/// Maximum number of cached arenas
const MAX_CACHED_ARENAS: usize = 4;

// Thread-local arena cache for reuse
thread_local! {
    static ARENA_CACHE: RefCell<Vec<Arena>> = RefCell::new(Vec::with_capacity(MAX_CACHED_ARENAS));
}

/// Arena allocator for temporary allocations during rule matching.
///
/// Provides fast bump-pointer allocation with bulk deallocation.
/// All allocations are freed when the arena is dropped.
pub struct Arena {
    /// Current chunk being allocated from
    current_chunk: RefCell<Chunk>,
    /// Previous chunks that are full
    chunks: RefCell<Vec<Chunk>>,
    /// Total bytes allocated
    bytes_allocated: Cell<usize>,
    /// Number of allocations made
    allocation_count: Cell<usize>,
}

/// A chunk of memory in the arena
struct Chunk {
    /// Raw memory buffer
    data: Vec<u8>,
    /// Current position in the buffer
    position: usize,
}

impl Chunk {
    fn new(size: usize) -> Self {
        Self {
            data: Vec::with_capacity(size),
            position: 0,
        }
    }

    fn allocate(&mut self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Align the position
        let aligned_pos = (self.position + align - 1) & !(align - 1);
        let end = aligned_pos + size;

        if end > self.data.capacity() {
            return None;
        }

        // Ensure Vec has enough initialized bytes
        if end > self.data.len() {
            self.data.resize(end, 0);
        }

        let ptr = unsafe { self.data.as_mut_ptr().add(aligned_pos) };
        self.position = end;

        NonNull::new(ptr)
    }

    fn reset(&mut self) {
        self.position = 0;
        self.data.clear();
    }
}

impl Arena {
    /// Creates a new arena with the default chunk size.
    pub fn new() -> Self {
        Self::with_chunk_size(DEFAULT_CHUNK_SIZE)
    }

    /// Creates a new arena with a specified chunk size.
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            current_chunk: RefCell::new(Chunk::new(chunk_size)),
            chunks: RefCell::new(Vec::new()),
            bytes_allocated: Cell::new(0),
            allocation_count: Cell::new(0),
        }
    }

    /// Gets an arena from the thread-local cache or creates a new one.
    pub fn get_cached() -> Self {
        ARENA_CACHE.with(|cache| cache.borrow_mut().pop().unwrap_or_else(Arena::new))
    }

    /// Returns the arena to the thread-local cache for reuse.
    pub fn return_to_cache(mut self) {
        self.reset();
        ARENA_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() < MAX_CACHED_ARENAS {
                cache.push(self);
            }
        });
    }

    /// Allocates memory for a value of type T.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc<T>(&self, value: T) -> &mut T {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();

        let ptr = self.alloc_raw(size, align);
        let typed_ptr = ptr.cast::<T>();

        unsafe {
            typed_ptr.as_ptr().write(value);
            &mut *typed_ptr.as_ptr()
        }
    }

    /// Allocates raw memory with the given size and alignment.
    fn alloc_raw(&self, size: usize, align: usize) -> NonNull<u8> {
        // Try to allocate from current chunk
        if let Some(ptr) = self.current_chunk.borrow_mut().allocate(size, align) {
            self.bytes_allocated.set(self.bytes_allocated.get() + size);
            self.allocation_count.set(self.allocation_count.get() + 1);
            return ptr;
        }

        // Current chunk is full, need a new one
        let mut chunks = self.chunks.borrow_mut();
        let old_chunk = self
            .current_chunk
            .replace(Chunk::new(DEFAULT_CHUNK_SIZE.max(size + align)));
        chunks.push(old_chunk);

        // Allocate from new chunk
        self.current_chunk
            .borrow_mut()
            .allocate(size, align)
            .expect("New chunk should have enough space")
    }

    /// Resets the arena for reuse, keeping allocated memory.
    pub fn reset(&mut self) {
        self.current_chunk.borrow_mut().reset();
        for chunk in self.chunks.borrow_mut().iter_mut() {
            chunk.reset();
        }
        self.bytes_allocated.set(0);
        self.allocation_count.set(0);
    }

    /// Gets statistics about the arena.
    pub fn stats(&self) -> ArenaStats {
        ArenaStats {
            bytes_allocated: self.bytes_allocated.get(),
            allocation_count: self.allocation_count.get(),
            chunk_count: self.chunks.borrow().len() + 1,
        }
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about arena usage
#[derive(Debug, Clone)]
pub struct ArenaStats {
    pub bytes_allocated: usize,
    pub allocation_count: usize,
    pub chunk_count: usize,
}

/// Arena-allocated match bindings for rule execution.
///
/// Uses inline storage for small numbers of bindings to avoid allocations entirely.
pub struct ArenaMatchBindings<'a> {
    /// Node variable bindings (inline storage for small counts)
    node_vars: SmallVec<[(&'a str, NodeId); 4]>,
    /// Edge variable bindings (inline storage for small counts)
    edge_vars: SmallVec<[(&'a str, EdgeId); 2]>,
    /// Arena for string allocations
    arena: &'a Arena,
}

impl<'a> ArenaMatchBindings<'a> {
    /// Creates new bindings using the given arena.
    pub fn new(arena: &'a Arena) -> Self {
        Self {
            node_vars: SmallVec::new(),
            edge_vars: SmallVec::new(),
            arena,
        }
    }

    /// Creates bindings with pre-allocated capacity.
    pub fn with_capacity(arena: &'a Arena, node_count: usize, edge_count: usize) -> Self {
        Self {
            node_vars: SmallVec::with_capacity(node_count),
            edge_vars: SmallVec::with_capacity(edge_count),
            arena,
        }
    }

    /// Inserts a node variable binding.
    pub fn insert_node(&mut self, var: &str, node: NodeId) {
        // Check if variable already exists
        for (existing_var, existing_node) in &mut self.node_vars {
            if *existing_var == var {
                *existing_node = node;
                return;
            }
        }

        // Allocate string in arena
        let var_str = self.allocate_str(var);
        self.node_vars.push((var_str, node));
    }

    /// Inserts an edge variable binding.
    pub fn insert_edge(&mut self, var: &str, edge: EdgeId) {
        // Check if variable already exists
        for (existing_var, existing_edge) in &mut self.edge_vars {
            if *existing_var == var {
                *existing_edge = edge;
                return;
            }
        }

        // Allocate string in arena
        let var_str = self.allocate_str(var);
        self.edge_vars.push((var_str, edge));
    }

    /// Gets a node binding by variable name.
    pub fn get_node(&self, var: &str) -> Option<NodeId> {
        self.node_vars
            .iter()
            .find(|(v, _)| *v == var)
            .map(|(_, id)| *id)
    }

    /// Gets an edge binding by variable name.
    pub fn get_edge(&self, var: &str) -> Option<EdgeId> {
        self.edge_vars
            .iter()
            .find(|(v, _)| *v == var)
            .map(|(_, id)| *id)
    }

    /// Clones the bindings into a new ArenaMatchBindings using a different arena.
    pub fn clone_to(&self, arena: &'a Arena) -> ArenaMatchBindings<'a> {
        let mut new_bindings = ArenaMatchBindings::new(arena);

        for (var, node) in &self.node_vars {
            new_bindings.insert_node(var, *node);
        }

        for (var, edge) in &self.edge_vars {
            new_bindings.insert_edge(var, *edge);
        }

        new_bindings
    }

    /// Converts to a regular HashMap-based bindings (for compatibility).
    pub fn to_hashmap(&self) -> (HashMap<String, NodeId>, HashMap<String, EdgeId>) {
        let mut node_map = HashMap::with_capacity(self.node_vars.len());
        let mut edge_map = HashMap::with_capacity(self.edge_vars.len());

        for (var, node) in &self.node_vars {
            node_map.insert((*var).to_string(), *node);
        }

        for (var, edge) in &self.edge_vars {
            edge_map.insert((*var).to_string(), *edge);
        }

        (node_map, edge_map)
    }

    /// Allocates a string in the arena.
    fn allocate_str(&self, s: &str) -> &'a str {
        let bytes = s.as_bytes();
        let size = bytes.len();
        let align = mem::align_of::<u8>();

        let ptr = self.arena.alloc_raw(size, align);

        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), size);
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr.as_ptr(), size))
        }
    }
}

/// Pool of reusable arenas for rule matching.
pub struct ArenaPool {
    arenas: Arc<parking_lot::Mutex<Vec<Arena>>>,
    max_arenas: usize,
}

impl ArenaPool {
    /// Creates a new arena pool with a maximum number of cached arenas.
    pub fn new(max_arenas: usize) -> Self {
        Self {
            arenas: Arc::new(parking_lot::Mutex::new(Vec::with_capacity(max_arenas))),
            max_arenas,
        }
    }

    /// Gets an arena from the pool or creates a new one.
    pub fn get(&self) -> Arena {
        self.arenas.lock().pop().unwrap_or_default()
    }

    /// Returns an arena to the pool for reuse.
    pub fn put(&self, mut arena: Arena) {
        arena.reset();
        let mut arenas = self.arenas.lock();
        if arenas.len() < self.max_arenas {
            arenas.push(arena);
        }
    }
}

impl Default for ArenaPool {
    fn default() -> Self {
        Self::new(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation() {
        let arena = Arena::new();

        let value1 = arena.alloc(42u32);
        let value2 = arena.alloc(100u64);

        assert_eq!(*value1, 42);
        assert_eq!(*value2, 100);

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 2);
    }

    #[test]
    fn test_arena_match_bindings() {
        let arena = Arena::new();
        let mut bindings = ArenaMatchBindings::new(&arena);

        bindings.insert_node("A", NodeId(1));
        bindings.insert_node("B", NodeId(2));
        bindings.insert_edge("e", EdgeId(1));

        assert_eq!(bindings.get_node("A"), Some(NodeId(1)));
        assert_eq!(bindings.get_node("B"), Some(NodeId(2)));
        assert_eq!(bindings.get_edge("e"), Some(EdgeId(1)));
        assert_eq!(bindings.get_node("C"), None);
    }

    #[test]
    fn test_arena_reuse() {
        let mut arena = Arena::new();

        // First use
        let value = arena.alloc(100u32);
        assert_eq!(*value, 100);
        let stats1 = arena.stats();

        // Reset and reuse
        arena.reset();
        let value2 = arena.alloc(200u32);
        assert_eq!(*value2, 200);
        let stats2 = arena.stats();

        // Should reuse memory
        assert_eq!(stats1.chunk_count, stats2.chunk_count);
    }

    #[test]
    fn test_arena_pool() {
        let pool = ArenaPool::new(2);

        let arena1 = pool.get();
        let arena2 = pool.get();

        pool.put(arena1);
        pool.put(arena2);

        // Should reuse arenas
        let arena3 = pool.get();
        let _arena4 = pool.get();
        pool.put(arena3);
    }

    #[test]
    fn test_inline_storage() {
        let arena = Arena::new();
        let bindings = ArenaMatchBindings::with_capacity(&arena, 2, 1);

        // Should use inline storage without heap allocation
        assert_eq!(bindings.node_vars.capacity(), 4); // SmallVec inline capacity
        assert_eq!(bindings.edge_vars.capacity(), 2); // SmallVec inline capacity
    }
}
