//! A simple neural network library in Rust
//! 
//! This library provides basic tensor operations, neural network layers,
//! and optimization algorithms for building and training neural networks.

pub mod tensor;
pub mod nn;
pub mod optim;

// Re-export commonly used types for convenience
pub use tensor::Tensor;
pub use nn::{Linear, SimpleNet};
pub use optim::SGD;
