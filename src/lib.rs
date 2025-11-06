//! A simple neural network library in Rust
//!
//! This library provides basic tensor operations, neural network layers,
//! and optimization algorithms for building and training neural networks.

pub mod nn;
pub mod tensor;

// Re-export commonly used types for convenience
pub use nn::*;
pub use tensor::Tensor;
