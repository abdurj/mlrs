//! Tensor module containing tensor operations and kernels
//! 
//! This module provides the core `Tensor` type and all operations on tensors,
//! including basic arithmetic, activations, reductions, and matrix operations.

mod core;
pub mod ops;
pub mod kernels;

// Re-export the main Tensor type and GraphNode for convenience
pub use core::{Tensor, GraphNode};
