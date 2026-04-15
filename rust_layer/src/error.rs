//! Error types for AresML Rust API

use std::fmt;

/// AresML error types - all mapped from C++ FFI error codes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AresError {
    /// Null pointer from C++ - indicates memory corruption or invalid handle
    NullPointer,
    /// Invalid tensor shape provided
    InvalidShape,
    /// Out of memory in C++ core
    OutOfMemory,
    /// Dimension mismatch between tensors
    DimensionMismatch,
    /// Invalid operation attempted
    InvalidOperation,
    /// Graph-related error (context issue, etc)
    GraphError,
    /// Backward pass failed
    BackwardFailed,
    /// Optimizer error
    OptimizerError,
    /// Custom error with message from C++
    Custom(String),
}

impl fmt::Display for AresError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AresError::NullPointer => write!(f, "Null pointer from C++ core"),
            AresError::InvalidShape => write!(f, "Invalid tensor shape"),
            AresError::OutOfMemory => write!(f, "Out of memory in C++ core"),
            AresError::DimensionMismatch => write!(f, "Tensor dimension mismatch"),
            AresError::InvalidOperation => write!(f, "Invalid operation"),
            AresError::GraphError => write!(f, "Graph context error"),
            AresError::BackwardFailed => write!(f, "Backward pass failed"),
            AresError::OptimizerError => write!(f, "Optimizer error"),
            AresError::Custom(msg) => write!(f, "Custom error: {}", msg),
        }
    }
}

impl std::error::Error for AresError {}

/// Result type for all AresML operations
pub type Result<T> = std::result::Result<T, AresError>;
EOF