//! Tensor module containing tensor operations and kernels
//!
//! This module provides the core `Tensor` type and all operations on tensors,
//! including basic arithmetic, activations, reductions, and matrix operations.

mod core;
pub mod kernels;
pub mod ops;

// Re-export the main Tensor type and GraphNode for convenience
pub use core::{GraphNode, Tensor};
