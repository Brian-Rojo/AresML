//! AresML Rust Safe API Layer
//! 
//! A memory-safe Rust API over C++ AresML core.
//! 
//! # Architecture
//! 
//! ```text
//! User Code (Rust)
//!         ↓
//! Safe API (this crate)
//!         ↓
//! FFI Boundary (ffi.rs - unsafe)
//!         ↓
//! C++ Core (aresml_ffi.h)
//! ```
//! 
//! # Key Guarantees
//! 
//! - **No raw pointer exposure**: All C++ pointers wrapped in NonNull
//! - **Explicit ownership**: Drop is the only destructor
//! - **No global state**: GraphContext replaces singleton
//! - **Deterministic lifecycle**: CREATE → USE → DROP
//! - **Multi-graph support**: Independent contexts, no interference

pub mod error;
pub mod ffi;
pub mod context;
pub mod tensor;
pub mod linear;

// Re-export for convenience
pub use error::{AresError, Result};
pub use context::GraphContext;
pub use tensor::Tensor;
pub use linear::Linear;

// ============================================================================
// EXAMPLE USAGE
// ============================================================================

/*
use aresml::*;

fn main() -> Result<()> {
    // Create explicit context (replaces global singleton)
    let ctx = GraphContext::new()?;
    
    // Create tensors
    let x = Tensor::randn(&[32, 128])?;
    x.set_requires_grad(true)?;
    ctx.register_leaf(&x)?;
    
    // Build model
    let l1 = Linear::new(128, 256, true)?;
    let l2 = Linear::new(256, 10, true)?;
    
    // Forward pass
    let h = l1.forward(&x)?.relu()?;
    let y = l2.forward(&h)?;
    
    // Compute loss
    let target = Tensor::randn(&[32, 10])?;
    let loss = mse_loss(&y, &target)?;
    
    // Backward pass (using explicit context - NO GLOBAL STATE!)
    ctx.register_leaf(l1.weight().unwrap())?;
    ctx.register_leaf(l2.weight().unwrap())?;
    ctx.backward(&loss)?;
    
    // Optimize
    let mut opt = SGD::new(&[l1.weight().unwrap(), l2.weight().unwrap()], 0.01)?;
    opt.step()?;
    opt.zero_grad()?;
    
    Ok(())
}
*/

// ============================================================================
// CARGO.TOML TEMPLATE
// ============================================================================

/*
[package]
name = "aresml"
version = "0.1.0"
edition = "2021"

[dependencies]
libc = "0.2"

[build-dependencies]
cc = "1.0"

[features]
default = ["cpu"]
cpu = []
*/

// ============================================================================
// BUILD.RS TEMPLATE
// ============================================================================

/*
fn main() {
    cc::Build::new()
        .file("path/to/aresml_ffi.c")
        .include("path/to/aresml/include")
        .flag("-std=c++17")
        .compile("aresml_core");
        
    println!("cargo:rustc-link-lib=aresml_core");
}
*/
EOF