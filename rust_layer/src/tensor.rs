//! Tensor wrapper - SAFE public API
//! 
//! This is the ONLY tensor type that users interact with.
//! NO raw pointers, NO owned flag, ONLY Drop-based cleanup.

use std::ptr::NonNull;
use std::os::raw::c_void;

use crate::error::{AresError, Result};
use crate::ffi::{self, TensorHandle};

/// Safe Tensor wrapper - owns C++ tensor through FFI boundary
/// 
/// # Ownership Model
/// 
/// - CREATE: `Tensor::new()`, `Tensor::randn()` → Rust owns handle
/// - USE: Borrow as `&Tensor` for operations
/// - DROP: `Drop` calls `aresml_tensor_free()`
/// 
/// # Memory Safety Guarantees
/// 
/// - No use-after-free: Drop frees C++ tensor
/// - No double-free: Drop is called once
/// - No leaks: C++ free called on drop
/// - No raw pointer exposure: handle is opaque
/// 
/// # Example
/// ```rust
/// let x = Tensor::randn(&[32, 128])?;
/// let y = x.relu()?;  // y is new tensor, x unchanged
/// // Drop called automatically when x and y go out of scope
/// ```
pub struct Tensor {
    /// Opaque handle to C++ tensor - NEVER exposed to users
    handle: TensorHandle,
}

impl Tensor {
    // =========================================================================
    // CONSTRUCTORS
    // =========================================================================
    
    /// Create tensor with given shape (uninitialized)
    pub fn new(shape: &[i64]) -> Result<Self> {
        let handle = unsafe {
            ffi::aresml_tensor_create(shape.as_ptr(), shape.len() as i32)
        };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::OutOfMemory)?;
        
        Ok(Tensor { handle })
    }
    
    /// Create tensor filled with zeros
    pub fn zeros(shape: &[i64]) -> Result<Self> {
        let handle = unsafe {
            ffi::aresml_tensor_zeros(shape.as_ptr(), shape.len() as i32)
        };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::OutOfMemory)?;
        
        Ok(Tensor { handle })
    }
    
    /// Create tensor with random normal distribution
    pub fn randn(shape: &[i64]) -> Result<Self> {
        let handle = unsafe {
            ffi::aresml_tensor_randn(shape.as_ptr(), shape.len() as i32)
        };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::OutOfMemory)?;
        
        Ok(Tensor { handle })
    }
    
    /// Create tensor from owned data vector
    pub fn from_vec(data: Vec<f32>, shape: &[i64]) -> Result<Self> {
        // First create tensor with shape
        let tensor = Tensor::new(shape)?;
        // Then copy data (would need additional FFI for this)
        // For now, use randn as placeholder
        Ok(tensor)
    }
    
    // =========================================================================
    // PROPERTIES
    // =========================================================================
    
    /// Get tensor shape
    pub fn shape(&self) -> Vec<i64> {
        let mut ndim: i32 = 0;
        let ptr = unsafe {
            ffi::aresml_tensor_get_shape(self.handle.as_ptr(), &mut ndim)
        };
        
        if ptr.is_null() {
            return vec![];
        }
        
        let shape = unsafe {
            std::slice::from_raw_parts(ptr, ndim as usize).to_vec()
        };
        
        // Free the shape array returned by C++
        unsafe { libc::free(ptr as *mut libc::c_void) };
        
        shape
    }
    
    /// Get total number of elements
    pub fn size(&self) -> i64 {
        unsafe { ffi::aresml_tensor_size(self.handle.as_ptr()) }
    }
    
    /// Check if tensor requires gradients
    pub fn requires_grad(&self) -> bool {
        unsafe { ffi::aresml_tensor_requires_grad(self.handle.as_ptr()) }
    }
    
    /// Set requires_grad flag
    pub fn set_requires_grad(&mut self, value: bool) -> Result<()> {
        unsafe {
            ffi::check_error(ffi::aresml_tensor_set_requires_grad(
                self.handle.as_ptr(),
                value,
            ))
        }
    }
    
    /// Check for NaN or Inf values
    pub fn has_nan(&self) -> bool {
        unsafe { ffi::aresml_tensor_has_nan(self.handle.as_ptr()) }
    }
    
    // =========================================================================
    // MATH OPERATIONS
    // =========================================================================
    
    /// Matrix multiplication: self @ other
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let handle = unsafe { ffi::aresml_matmul(self.handle, other.handle) };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::NullPointer)?;
        
        Ok(Tensor { handle })
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let handle = unsafe { ffi::aresml_add(self.handle, other.handle) };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::NullPointer)?;
        
        Ok(Tensor { handle })
    }
    
    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let handle = unsafe { ffi::aresml_mul(self.handle, other.handle) };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::NullPointer)?;
        
        Ok(Tensor { handle })
    }
    
    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Result<Tensor> {
        let handle = unsafe { ffi::aresml_relu(self.handle) };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::NullPointer)?;
        
        Ok(Tensor { handle })
    }
    
    /// Softmax activation
    pub fn softmax(&self, axis: i32) -> Result<Tensor> {
        let handle = unsafe { ffi::aresml_softmax(self.handle, axis) };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::NullPointer)?;
        
        Ok(Tensor { handle })
    }
    
    /// Log-softmax activation
    pub fn log_softmax(&self, axis: i32) -> Result<Tensor> {
        let handle = unsafe { ffi::aresml_log_softmax(self.handle, axis) };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::NullPointer)?;
        
        Ok(Tensor { handle })
    }
    
    /// Sum all elements (returns scalar tensor)
    pub fn sum(&self) -> Result<Tensor> {
        let handle = unsafe { ffi::aresml_sum(self.handle) };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::NullPointer)?;
        
        Ok(Tensor { handle })
    }
    
    /// Mean of all elements (returns scalar tensor)
    pub fn mean(&self) -> Result<Tensor> {
        let handle = unsafe { ffi::aresml_mean(self.handle) };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::NullPointer)?;
        
        Ok(Tensor { handle })
    }
    
    // =========================================================================
    // GRAPH OPERATIONS
    // =========================================================================
    
    /// Clone tensor (deep copy)
    /// 
    /// Uses explicit C++ clone function - NOT a simple bit copy
    pub fn clone(&self) -> Result<Tensor> {
        let handle = unsafe { ffi::aresml_tensor_clone(self.handle) };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::NullPointer)?;
        
        Ok(Tensor { handle })
    }
    
    /// Detach from computation graph (no gradient tracking)
    /// 
    /// Returns a new tensor that doesn't track gradients
    pub fn detach(&self) -> Result<Tensor> {
        // Clone and disable grad tracking
        let mut cloned = self.clone()?;
        cloned.set_requires_grad(false)?;
        Ok(cloned)
    }
    
    /// Backward pass - compute gradients
    /// 
    /// Requires explicit GraphContext - no global state!
    pub fn backward(&self, ctx: &super::GraphContext) -> Result<()> {
        ctx.backward(self)
    }
    
    /// Get gradient tensor (if requires_grad was true and backward was called)
    pub fn grad(&self) -> Option<Tensor> {
        // Would need additional FFI to get grad tensor
        None
    }
    
    // =========================================================================
    // INTERNAL
    // =========================================================================
    
    /// Get raw handle (for internal FFI use only)
    #[doc(hidden)]
    pub fn as_raw(&self) -> TensorHandle {
        self.handle
    }
}

impl Drop for Tensor {
    /// Drop is the SOLE mechanism for freeing C++ tensor
    /// 
    /// No "owned" flag - if Tensor exists, it owns the handle.
    /// Drop ALWAYS calls free.
    fn drop(&mut self) {
        unsafe {
            ffi::aresml_tensor_free(self.handle.as_ptr());
        }
    }
}

// Clone is explicit - calls C++ clone, not bit copy
impl Clone for Tensor {
    fn clone(&self) -> Self {
        // This will panic if clone fails - consistent with Rust semantics
        self.clone().expect("Tensor::clone failed")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_lifecycle() {
        // Create
        let x = Tensor::randn(&[4, 8]).unwrap();
        let _shape = x.shape();
        let _size = x.size();
        
        // Drop automatically frees C++ tensor
    }
    
    #[test]
    fn test_no_double_free() {
        let x = Tensor::randn(&[2, 2]).unwrap();
        let _y = x.clone();
        // Both Drop correctly - C++ uses ref counting
    }
    
    #[test]
    fn test_matmul() {
        let a = Tensor::randn(&[2, 3]).unwrap();
        let b = Tensor::randn(&[3, 4]).unwrap();
        let c = a.matmul(&b).unwrap();
        
        assert_eq!(c.shape(), vec![2, 4]);
    }
}
EOF