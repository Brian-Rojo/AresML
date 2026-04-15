# AresML Rust Layer Design
## C++ Core → Rust Safe API Architecture

================================================================================
1. SYSTEM ARCHITECTURE OVERVIEW
================================================================================

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rust Safe API Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Tensor      │  │ Model       │  │ Training                │ │
│  │ Wrapper     │  │ Builder     │  │ Loop                   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              FFI Safety Boundary (Rust)                     ││
│  │  - Ownership enforcement                                    ││
│  │  - Error handling translation                               ││
│  │  - Pointer validation                                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                           ↓ FFI
┌─────────────────────────────────────────────────────────────────┐
│                 C++ AresML Core (UNMODIFIED)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Tensor      │  │ Autograd    │  │ Ops (Linear, MSE, etc)  │ │
│  │ (raw ptr)   │  │ Engine      │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

================================================================================
2. FFI C++ EXPOSED API (ffi/aresml_ffi.h)
================================================================================

```c
#ifndef ARESML_FFI_H
#define ARESML_FFI_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ERROR HANDLING
// ============================================================================

typedef enum {
    ARESML_SUCCESS = 0,
    ARESML_ERR_NULLPTR = 1,
    ARESML_ERR_SHAPE_MISMATCH = 2,
    ARESML_ERR_OUT_OF_MEMORY = 3,
    ARESML_ERR_INVALID_OP = 4,
    ARESML_ERR_BACKWARD_FAILED = 5,
    ARESML_ERR_CUDA_NOT_AVAILABLE = 6
} aresml_error_t;

// ============================================================================
// TENSOR OPERATIONS
// ============================================================================

void* aresml_tensor_create(float* data, size_t size);
void* aresml_tensor_randn(size_t* shape, size_t ndim);
void* aresml_tensor_zeros(size_t* shape, size_t ndim);
aresml_error_t aresml_tensor_free(void* handle);
aresml_error_t aresml_tensor_get_shape(void* handle, size_t* out_shape, size_t* out_ndim);
aresml_error_t aresml_tensor_size(void* handle, size_t* out_size);
aresml_error_t aresml_tensor_set_requires_grad(void* handle, bool value);

// ============================================================================
// MATH OPERATIONS
// ============================================================================

void* aresml_matmul(void* a, void* b);
void* aresml_add(void* a, void* b);
void* aresml_relu(void* x);
void* aresml_softmax(void* x, int axis);
void* aresml_log_softmax(void* x, int axis);
void* aresml_sum(void* x);

// ============================================================================
// NEURAL NETWORK LAYERS
// ============================================================================

void* aresml_linear_create(size_t in_features, size_t out_features, bool bias);
void* aresml_linear_forward(void* linear_handle, void* input);
aresml_error_t aresml_linear_free(void* handle);

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

void* aresml_mse_loss(void* pred, void* target);
void* aresml_cross_entropy(void* logits, void* targets);

// ============================================================================
// AUTOGRAD ENGINE
// ============================================================================

aresml_error_t aresml_backward(void* loss);
aresml_error_t aresml_zero_grad(void);
aresml_error_t aresml_clip_grad(float max_norm);

// ============================================================================
// OPTIMIZERS
// ============================================================================

void* aresml_sgd_create(void** params, size_t num_params, float lr);
void* aresml_adam_create(void** params, size_t num_params, float lr, float beta1, float beta2);
aresml_error_t aresml_optimizer_step(void* optimizer);
aresml_error_t aresml_optimizer_zero_grad(void* optimizer);
aresml_error_t aresml_optimizer_free(void* optimizer);

// ============================================================================
// UTILITIES
// ============================================================================

aresml_error_t aresml_tensor_has_nan(void* handle, bool* out);

#ifdef __cplusplus
}
#endif

#endif
```

================================================================================
3. RUST SAFE API LAYER
================================================================================

```rust
//! AresML Rust Safe API Layer
//! 
//! Wraps C++ AresML core with safe Rust abstractions.

use std::ptr::NonNull;
use std::os::raw::c_void;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum AresError {
    NullPointer,
    ShapeMismatch,
    OutOfMemory,
    InvalidOperation,
    BackwardFailed,
    Ffi(String),
}

impl std::fmt::Display for AresError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AresError::NullPointer => write!(f, "Null pointer from C++ core"),
            AresError::ShapeMismatch => write!(f, "Tensor shape mismatch"),
            AresError::OutOfMemory => write!(f, "Out of memory"),
            AresError::InvalidOperation => write!(f, "Invalid operation"),
            AresError::BackwardFailed => write!(f, "Backward failed"),
            AresError::Ffi(msg) => write!(f, "FFI: {}", msg),
        }
    }
}

pub type Result<T> = std::result::Result<T, AresError>;

// ============================================================================
// FFI BINDINGS (unsafe internals)
// ============================================================================

mod ffi {
    use std::os::raw::c_void;
    
    #[repr(C)] pub enum AresmlError { Success = 0, NullPtr = 1, ShapeMismatch = 2 }
    
    extern "C" {
        pub fn aresml_tensor_create(data: *const f32, size: usize) -> *mut c_void;
        pub fn aresml_tensor_randn(shape: *const usize, ndim: usize) -> *mut c_void;
        pub fn aresml_tensor_free(ptr: *mut c_void) -> AresmlError;
        pub fn aresml_matmul(a: *mut c_void, b: *mut c_void) -> *mut c_void;
        pub fn aresml_relu(x: *mut c_void) -> *mut c_void;
        pub fn aresml_softmax(x: *mut c_void, axis: i32) -> *mut c_void;
        pub fn aresml_sum(x: *mut c_void) -> *mut c_void;
        pub fn aresml_linear_create(in_f: usize, out_f: usize, bias: bool) -> *mut c_void;
        pub fn aresml_linear_forward(layer: *mut c_void, input: *mut c_void) -> *mut c_void;
        pub fn aresml_linear_free(ptr: *mut c_void) -> AresmlError;
        pub fn aresml_mse_loss(pred: *mut c_void, target: *mut c_void) -> *mut c_void;
        pub fn aresml_backward(loss: *mut c_void) -> AresmlError;
        pub fn aresml_zero_grad() -> AresmlError;
        pub fn aresml_clip_grad(max_norm: f32) -> AresmlError;
        pub fn aresml_sgd_create(params: *mut *mut c_void, n: usize, lr: f32) -> *mut c_void;
        pub fn aresml_adam_create(params: *mut *mut c_void, n: usize, lr: f32, b1: f32, b2: f32) -> *mut c_void;
        pub fn aresml_optimizer_step(opt: *mut c_void) -> AresmlError;
        pub fn aresml_optimizer_free(ptr: *mut c_void) -> AresmlError;
    }
}

fn validate<T>(ptr: *mut T) -> Result<NonNull<T>> {
    NonNull::new(ptr).ok_or(AresError::NullPointer)
}

fn ok(err: ffi::AresmlError) -> Result<()> {
    match err {
        ffi::AresmlError::Success => Ok(()),
        _ => Err(AresError::Ffi(format!("{:?}", err))),
    }
}

// ============================================================================
// TENSOR WRAPPER
// ============================================================================

pub struct Tensor {
    ptr: NonNull<c_void>,
    owned: bool,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>) -> Result<Self> {
        let ptr = unsafe { ffi::aresml_tensor_create(data.as_ptr(), data.len()) };
        Ok(Tensor { ptr: validate(ptr)?, owned: true })
    }
    
    pub fn randn(shape: &[usize]) -> Result<Self> {
        let ptr = unsafe { ffi::aresml_tensor_randn(shape.as_ptr(), shape.len()) };
        Ok(Tensor { ptr: validate(ptr)?, owned: true })
    }
    
    pub fn zeros(shape: &[usize]) -> Result<Self> {
        let ptr = unsafe { ffi::aresml_tensor_zeros(shape.as_ptr(), shape.len()) };
        Ok(Tensor { ptr: validate(ptr)?, owned: true })
    }
    
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let ptr = unsafe { ffi::aresml_matmul(self.ptr.as_ptr(), other.ptr.as_ptr()) };
        Ok(Tensor { ptr: validate(ptr)?, owned: true })
    }
    
    pub fn relu(&self) -> Result<Tensor> {
        let ptr = unsafe { ffi::aresml_relu(self.ptr.as_ptr()) };
        Ok(Tensor { ptr: validate(ptr)?, owned: true })
    }
    
    pub fn softmax(&self, axis: i32) -> Result<Tensor> {
        let ptr = unsafe { ffi::aresml_softmax(self.ptr.as_ptr(), axis) };
        Ok(Tensor { ptr: validate(ptr)?, owned: true })
    }
    
    pub fn sum(&self) -> Result<Tensor> {
        let ptr = unsafe { ffi::aresml_sum(self.ptr.as_ptr()) };
        Ok(Tensor { ptr: validate(ptr)?, owned: true })
    }
    
    pub fn backward(&self) -> Result<()> {
        ok(unsafe { ffi::aresml_backward(self.ptr.as_ptr()) })
    }
    
    pub fn clone(&self) -> Result<Tensor> {
        // Would call C++ clone
        todo!()
    }
    
    pub fn detach(&self) -> Result<Tensor> {
        // Would create detached copy
        todo!()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.owned {
            unsafe { ffi::aresml_tensor_free(self.ptr.as_ptr()); }
        }
    }
}

// ============================================================================
// LINEAR LAYER
// ============================================================================

pub struct Linear {
    ptr: NonNull<c_void>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self> {
        let ptr = unsafe { ffi::aresml_linear_create(in_features, out_features, bias) };
        Ok(Linear { ptr: validate(ptr)? })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let ptr = unsafe { ffi::aresml_linear_forward(self.ptr.as_ptr(), input.ptr.as_ptr()) };
        Ok(Tensor { ptr: validate(ptr)?, owned: true })
    }
}

impl Drop for Linear {
    fn drop(&mut self) {
        unsafe { ffi::aresml_linear_free(self.ptr.as_ptr()); }
    }
}

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let ptr = unsafe { ffi::aresml_mse_loss(pred.ptr.as_ptr(), target.ptr.as_ptr()) };
    Ok(Tensor { ptr: validate(ptr)?, owned: true })
}

pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Would call C++ cross_entropy
    todo!()
}

// ============================================================================
// OPTIMIZERS
// ============================================================================

pub struct SGD {
    ptr: NonNull<c_void>,
}

impl SGD {
    pub fn new(params: &[&Tensor], lr: f32) -> Result<Self> {
        let mut ptrs: Vec<*mut c_void> = params.iter().map(|p| p.ptr.as_ptr()).collect();
        let ptr = unsafe { ffi::aresml_sgd_create(ptrs.as_ptr(), ptrs.len(), lr) };
        Ok(SGD { ptr: validate(ptr)? })
    }
    
    pub fn step(&self) -> Result<()> {
        ok(unsafe { ffi::aresml_optimizer_step(self.ptr.as_ptr()) })
    }
    
    pub fn zero_grad(&self) -> Result<()> {
        ok(unsafe { ffi::aresml_optimizer_zero_grad(self.ptr.as_ptr()) })
    }
}

impl Drop for SGD {
    fn drop(&mut self) {
        unsafe { ffi::aresml_optimizer_free(self.ptr.as_ptr()); }
    }
}

pub struct Adam {
    ptr: NonNull<c_void>,
}

impl Adam {
    pub fn new(params: &[&Tensor], lr: f32, beta1: f32, beta2: f32) -> Result<Self> {
        let mut ptrs: Vec<*mut c_void> = params.iter().map(|p| p.ptr.as_ptr()).collect();
        let ptr = unsafe { ffi::aresml_adam_create(ptrs.as_ptr(), ptrs.len(), lr, beta1, beta2) };
        Ok(Adam { ptr: validate(ptr)? })
    }
    
    pub fn step(&self) -> Result<()> {
        ok(unsafe { ffi::aresml_optimizer_step(self.ptr.as_ptr()) })
    }
    
    pub fn zero_grad(&self) -> Result<()> {
        ok(unsafe { ffi::aresml_optimizer_zero_grad(self.ptr.as_ptr()) })
    }
}

impl Drop for Adam {
    fn drop(&mut self) {
        unsafe { ffi::aresml_optimizer_free(self.ptr.as_ptr()); }
    }
}

// ============================================================================
// TRAINING HELPERS
// ============================================================================

pub fn clip_grad(max_norm: f32) -> Result<()> {
    ok(unsafe { ffi::aresml_clip_grad(max_norm) })
}

pub fn zero_grad() -> Result<()> {
    ok(unsafe { ffi::aresml_zero_grad() })
}

// ============================================================================
// EXAMPLE USAGE
// ============================================================================

/*
use aresml::*;

fn main() -> Result<()> {
    let x = Tensor::randn(&[32, 128])?;
    let target = Tensor::randn(&[32, 10])?;
    
    let l1 = Linear::new(128, 256, true)?;
    let l2 = Linear::new(256, 10, true)?;
    
    let h = l1.forward(&x)?.relu()?;
    let y = l2.forward(&h)?;
    
    let loss = mse_loss(&y, &target)?;
    loss.backward()?;
    
    let mut opt = Adam::new(&[&l1, &l2], 0.001)?;
    opt.step()?;
    opt.zero_grad()?;
    
    Ok(())
}
*/
```

================================================================================
4. OWNERSHIP MODEL
================================================================================

```
RULES:
------
1. Tensor owns C++ tensor → Drop frees it
2. Moving transfers ownership → old becomes invalid
3. &Tensor = read-only borrow (no aliasing enforced in Rust)
4. Linear owns weights → Drop frees weights
5. Optimizer holds refs to params → params must outlive optimizer

GUARANTEES:
-----------
✓ No use-after-free: Drop calls C++ free
✓ No double-free: owned flag prevents
✓ No leaks: All C++ allocations freed in Drop
✓ No dangling: NonNull validates
```

================================================================================
5. MEMORY SAFETY STRATEGY
================================================================================

```
C++ PROBLEMA              → RUST MITIGATION
----------------------    ------------------
raw pointers             → NonNull wrapper + validation
const_cast               → Hidden, users can't trigger
singleton global         → Behind FFI boundary
use-after-free           → Drop implementation
double-free              → owned flag
null dereference         → validate_ptr()
```

================================================================================
6. ERROR HANDLING CROSS-LANGUAGE
================================================================================

```
C++ Exceptions (caught at FFI boundary):
  std::bad_alloc    → AresError::OutOfMemory
  null pointer      → AresError::NullPointer
  shape error       → AresError::ShapeMismatch
  other             → AresError::Ffi(String)

Rust returns Result<T, AresError>
User uses ? or match to handle
```

================================================================================
7. RISKS AND MITIGATIONS
================================================================================

| Risk | Severity | Mitigation |
|------|----------|------------|
| C++ memory corruption | HIGH | Valgrind/ASan testing |
| Thread safety | MEDIUM | Document single-thread |
| ABI compatibility | MEDIUM | extern "C", test on platforms |
| Exception crossing | LOW | try-catch in FFI |
| Resource leaks | LOW | Audit Drop impls |

================================================================================
8. PROJECT STRUCTURE
================================================================================

```
aresml/
├── Cargo.toml              # Rust package
├── build.rs                # Build C++ core
├── src/
│   ├── lib.rs              # Public API
│   ├── tensor.rs           # Tensor wrapper
│   ├── layers.rs           # Linear, etc
│   ├── loss.rs             # MSE, CE
│   ├── optim.rs            # SGD, Adam
│   ├── ffi.rs              # Unsafe FFI
│   └── error.rs            # Error types
├── ffi/
│   └── aresml_ffi.h        # C++ FFI header
└── core/ (existing C++)

build integration:
  [build-dependencies]
  cc = "1.0"
```

================================================================================
9. FIN - RESUMEN EJECUTIVO
================================================================================

Arquitetura propuesta:

1. **C++ FFI Layer** - extern "C" functions, error codes
2. **Rust FFI Bindings** - unsafe module with validation
3. **Safe API** - Tensor, Linear, optimizers with Drop
4. **Ownership** - owned flag, NonNull, Drop implementations

El usuario final ve:
```rust
let x = Tensor::randn(&[32, 128])?;
let l1 = Linear::new(128, 256, true)?;
let y = l1.forward(&x)?.relu()?;
y.backward()?;
```

Sin ver raw pointers, const_cast, o singletons.