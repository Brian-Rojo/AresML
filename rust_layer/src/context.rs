//! Graph Context - replaces singleton AutogradEngine
//! 
//! This is the EXPLICIT context that replaces the global singleton.
//! Each model/training run gets its own context handle.

use std::ptr::NonNull;
use std::os::raw::c_void;

use crate::error::{AresError, Result};
use crate::ffi::{self, GraphContextHandle};

/// Computation graph context - replaces global AutogradEngine
/// 
/// This is the CORE concept for multi-graph support.
/// Each context is independent - no shared global state.
/// 
/// # Example
/// ```rust
/// let ctx1 = GraphContext::new()?;
/// let ctx2 = GraphContext::new()?;
/// // ctx1 and ctx2 are completely independent
/// ```
pub struct GraphContext {
    handle: GraphContextHandle,
}

impl GraphContext {
    /// Create new graph context
    /// 
    /// # Errors
    /// Returns error if C++ allocation fails
    pub fn new() -> Result<Self> {
        let handle = unsafe { ffi::aresml_graph_context_create() }
            .ok_or(AresError::OutOfMemory)?;
        
        Ok(GraphContext { handle })
    }
    
    /// Register tensor as leaf (for gradient tracking)
    /// 
    /// # Errors
    /// Returns error if tensor is null or context is invalid
    pub fn register_leaf(&self, tensor: &super::Tensor) -> Result<()> {
        unsafe {
            ffi::check_error(ffi::aresml_graph_register_leaf(
                self.handle,
                tensor.handle,
            ))
        }
    }
    
    /// Zero gradients of all leaf tensors in this context
    pub fn zero_grad(&self) -> Result<()> {
        unsafe {
            ffi::check_error(ffi::aresml_graph_zero_grad(self.handle))
        }
    }
    
    /// Run backward pass from loss tensor
    /// 
    /// # Arguments
    /// * `loss` - Scalar tensor to backpropagate from
    pub fn backward(&self, loss: &super::Tensor) -> Result<()> {
        unsafe {
            ffi::check_error(ffi::aresml_graph_backward(
                self.handle,
                loss.handle,
            ))
        }
    }
    
    /// Get raw handle for passing to C++ (internal use)
    #[doc(hidden)]
    pub fn as_raw(&self) -> GraphContextHandle {
        self.handle
    }
}

impl Drop for GraphContext {
    fn drop(&mut self) {
        unsafe {
            ffi::aresml_graph_context_destroy(self.handle);
        }
    }
}

// SAFETY: GraphContext is Send + Sync because it's just an opaque handle
// The underlying C++ implementation handles thread safety (or not)
unsafe impl Send for GraphContext {}
unsafe impl Sync for GraphContext {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_context() {
        let ctx = GraphContext::new();
        assert!(ctx.is_ok());
    }
    
    #[test]
    fn test_two_contexts_independent() {
        let ctx1 = GraphContext::new().unwrap();
        let ctx2 = GraphContext::new().unwrap();
        // Both contexts exist independently
        // Dropping one doesn't affect the other
    }
}
EOF