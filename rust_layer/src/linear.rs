//! Linear (Dense) layer wrapper

use std::ptr::NonNull;
use std::os::raw::c_void;

use crate::error::{AresError, Result};
use crate::ffi::{self, LinearHandle};
use crate::Tensor;

/// Linear (Fully Connected) layer: y = xW^T + b
/// 
/// # Ownership
/// 
/// - Linear owns its weight and bias tensors
/// - Drop frees the layer and its tensors
/// - forward() doesn't consume the input tensor
/// 
/// # Example
/// ```rust
/// let linear = Linear::new(128, 256, true)?;
/// let y = linear.forward(&x)?;
/// ```
pub struct Linear {
    handle: LinearHandle,
}

impl Linear {
    /// Create new linear layer
    /// 
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `bias` - Whether to include bias term
    pub fn new(in_features: i32, out_features: i32, bias: bool) -> Result<Self> {
        let handle = unsafe {
            ffi::aresml_linear_create(in_features, out_features, bias)
        };
        let handle = ffi::to_linear_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::OutOfMemory)?;
        
        Ok(Linear { handle })
    }
    
    /// Forward pass: y = xW^T + b
    /// 
    /// # Arguments
    /// * `input` - Input tensor of shape [..., in_features]
    /// 
    /// # Returns
    /// Output tensor of shape [..., out_features]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let handle = unsafe {
            ffi::aresml_linear_forward(self.handle, input.handle)
        };
        let handle = ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .ok_or(AresError::NullPointer)?;
        
        Ok(Tensor { handle })
    }
    
    /// Get weight tensor (read-only)
    /// 
    /// Shape: [out_features, in_features]
    pub fn weight(&self) -> Option<Tensor> {
        let handle = unsafe { ffi::aresml_linear_get_weight(self.handle) };
        ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .map(|h| Tensor { handle: h })
    }
    
    /// Get bias tensor (read-only)
    /// 
    /// Shape: [out_features]
    /// Returns None if bias was not used
    pub fn bias(&self) -> Option<Tensor> {
        let handle = unsafe { ffi::aresml_linear_get_bias(self.handle) };
        if handle.as_ptr().is_null() {
            return None;
        }
        ffi::to_tensor_handle(handle.as_ptr() as *mut c_void)
            .map(|h| Tensor { handle: h })
    }
}

impl Drop for Linear {
    fn drop(&mut self) {
        unsafe {
            ffi::aresml_linear_free(self.handle.as_ptr());
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(128, 256, true).unwrap();
        let x = Tensor::randn(&[32, 128]).unwrap();
        let y = linear.forward(&x).unwrap();
        
        assert_eq!(y.shape(), vec![32, 256]);
    }
    
    #[test]
    fn test_linear_weights() {
        let linear = Linear::new(10, 20, true).unwrap();
        let w = linear.weight();
        assert!(w.is_some());
        assert_eq!(w.unwrap().shape(), vec![20, 10]);
    }
    
    #[test]
    fn test_linear_no_bias() {
        let linear = Linear::new(5, 5, false).unwrap();
        assert!(linear.bias().is_none());
    }
}
EOF