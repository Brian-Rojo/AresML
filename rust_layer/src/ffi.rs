//! FFI bindings to C++ AresML core
//! 
//! THIS MODULE IS UNSAFE - contains raw FFI calls
//! All public APIs wrap these to provide safety

use std::os::raw::c_char;
use std::os::raw::c_void;
use std::ptr::NonNull;

// ============================================================================
// C ABI ERROR CODES (must match aresml_ffi.h)
// ============================================================================

#[repr(C)]
pub enum AresmlError {
    Success = 0,
    NullPtr = 1,
    InvalidShape = 2,
    OutOfMemory = 3,
    DimensionMismatch = 4,
    InvalidOperation = 5,
    GraphError = 6,
    BackwardFailed = 7,
    OptimizerError = 8,
    Custom = 9,
}

// Convert C error to Rust error
impl From<AresmlError> for super::error::AresError {
    fn from(err: AresmlError) -> Self {
        match err {
            AresmlError::Success => unreachable!(),
            AresmlError::NullPtr => super::error::AresError::NullPointer,
            AresmlError::InvalidShape => super::error::AresError::InvalidShape,
            AresmlError::OutOfMemory => super::error::AresError::OutOfMemory,
            AresmlError::DimensionMismatch => super::error::AresError::DimensionMismatch,
            AresmlError::InvalidOperation => super::error::AresError::InvalidOperation,
            AresmlError::GraphError => super::error::AresError::GraphError,
            AresmlError::BackwardFailed => super::error::AresError::BackwardFailed,
            AresmlError::OptimizerError => super::error::AresError::OptimizerError,
            AresmlError::Custom => super::error::AresError::Custom("Unknown".to_string()),
        }
    }
}

// Check if error is success
pub fn is_success(err: AresmlError) -> bool {
    matches!(err, AresmlError::Success)
}

// ============================================================================
// OPAQUE HANDLE TYPES
// ============================================================================

// These match the C typedefs but are used as markers in Rust
pub type GraphContextHandle = NonNull<c_void>;
pub type TensorHandle = NonNull<c_void>;
pub type LinearHandle = NonNull<c_void>;
pub type SGDHandle = NonNull<c_void>;
pub type AdamHandle = NonNull<c_void>;

// ============================================================================
// GRAPH CONTEXT FFI
// ============================================================================

extern "C" {
    // Create new graph context
    pub fn aresml_graph_context_create() -> GraphContextHandle;
    
    // Destroy graph context
    pub fn aresml_graph_context_destroy(ctx: GraphContextHandle);
    
    // Register leaf tensor
    pub fn aresml_graph_register_leaf(ctx: GraphContextHandle, tensor: TensorHandle) -> AresmlError;
    
    // Zero gradients in context
    pub fn aresml_graph_zero_grad(ctx: GraphContextHandle) -> AresmlError;
    
    // Backward pass
    pub fn aresml_graph_backward(ctx: GraphContextHandle, loss: TensorHandle) -> AresmlError;
}

// ============================================================================
// TENSOR FFI
// ============================================================================

extern "C" {
    // Create tensor from shape
    pub fn aresml_tensor_create(shape: *const i64, ndim: i32) -> TensorHandle;
    
    // Create random normal tensor
    pub fn aresml_tensor_randn(shape: *const i64, ndim: i32) -> TensorHandle;
    
    // Create zeros tensor
    pub fn aresml_tensor_zeros(shape: *const i64, ndim: i32) -> TensorHandle;
    
    // Clone tensor
    pub fn aresml_tensor_clone(tensor: TensorHandle) -> TensorHandle;
    
    // Free tensor
    pub fn aresml_tensor_free(tensor: TensorHandle);
    
    // Get shape (caller must free)
    pub fn aresml_tensor_get_shape(tensor: TensorHandle, out_ndim: *mut i32) -> *mut i64;
    
    // Get total size
    pub fn aresml_tensor_size(tensor: TensorHandle) -> i64;
    
    // Get data pointer (read-only)
    pub fn aresml_tensor_data(tensor: TensorHandle) -> *const f32;
    
    // Set requires_grad
    pub fn aresml_tensor_set_requires_grad(tensor: TensorHandle, value: bool) -> AresmlError;
    
    // Get requires_grad
    pub fn aresml_tensor_requires_grad(tensor: TensorHandle) -> bool;
    
    // Check for NaN
    pub fn aresml_tensor_has_nan(tensor: TensorHandle) -> bool;
    
    // Print (debug)
    pub fn aresml_tensor_print(tensor: TensorHandle);
}

// ============================================================================
// MATH OPERATIONS FFI
// ============================================================================

extern "C" {
    pub fn aresml_add(a: TensorHandle, b: TensorHandle) -> TensorHandle;
    pub fn aresml_mul(a: TensorHandle, b: TensorHandle) -> TensorHandle;
    pub fn aresml_matmul(a: TensorHandle, b: TensorHandle) -> TensorHandle;
    pub fn aresml_relu(x: TensorHandle) -> TensorHandle;
    pub fn aresml_softmax(x: TensorHandle, axis: i32) -> TensorHandle;
    pub fn aresml_log_softmax(x: TensorHandle, axis: i32) -> TensorHandle;
    pub fn aresml_sum(x: TensorHandle) -> TensorHandle;
    pub fn aresml_mean(x: TensorHandle) -> TensorHandle;
}

// ============================================================================
// LINEAR LAYER FFI
// ============================================================================

extern "C" {
    pub fn aresml_linear_create(in_features: i32, out_features: i32, bias: bool) -> LinearHandle;
    pub fn aresml_linear_forward(layer: LinearHandle, input: TensorHandle) -> TensorHandle;
    pub fn aresml_linear_get_weight(layer: LinearHandle) -> TensorHandle;
    pub fn aresml_linear_get_bias(layer: LinearHandle) -> TensorHandle;
    pub fn aresml_linear_free(layer: LinearHandle);
}

// ============================================================================
// LOSS FUNCTIONS FFI
// ============================================================================

extern "C" {
    pub fn aresml_mse_loss(pred: TensorHandle, target: TensorHandle) -> TensorHandle;
    pub fn aresml_cross_entropy(logits: TensorHandle, targets: TensorHandle) -> TensorHandle;
}

// ============================================================================
// OPTIMIZERS FFI
// ============================================================================

extern "C" {
    pub fn aresml_sgd_create(params: *const TensorHandle, num_params: i32, lr: f32) -> SGDHandle;
    pub fn aresml_optimizer_step_sgd(optimizer: SGDHandle) -> AresmlError;
    pub fn aresml_optimizer_zero_grad_sgd(optimizer: SGDHandle) -> AresmlError;
    pub fn aresml_optimizer_free_sgd(optimizer: SGDHandle);
    
    pub fn aresml_adam_create(params: *const TensorHandle, num_params: i32, lr: f32, beta1: f32, beta2: f32) -> AdamHandle;
    pub fn aresml_optimizer_step_adam(optimizer: AdamHandle) -> AresmlError;
    pub fn aresml_optimizer_zero_grad_adam(optimizer: AdamHandle) -> AresmlError;
    pub fn aresml_optimizer_free_adam(optimizer: AdamHandle);
}

// ============================================================================
// ERROR HANDLING FFI
// ============================================================================

extern "C" {
    pub fn aresml_get_last_error() -> *const c_char;
    pub fn aresml_clear_last_error();
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Validate a non-null pointer from C++
pub fn validate_ptr<T>(ptr: *mut T) -> Option<NonNull<T>> {
    NonNull::new(ptr)
}

/// Check error code and convert to Result
pub fn check_error(err: AresmlError) -> Result<(), super::error::AresError> {
    if is_success(err) {
        Ok(())
    } else {
        Err(err.into())
    }
}

/// Convert raw pointer to handle type
pub fn to_tensor_handle(ptr: *mut c_void) -> Option<TensorHandle> {
    validate_ptr(ptr)
}

pub fn to_linear_handle(ptr: *mut c_void) -> Option<LinearHandle> {
    validate_ptr(ptr)
}

pub fn to_context_handle(ptr: *mut c_void) -> Option<GraphContextHandle> {
    validate_ptr(ptr)
}
EOF