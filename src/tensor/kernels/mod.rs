//! Kernel implementations for tensor operations
//! 
//! This module contains optimized kernel implementations for various
//! tensor operations, particularly matrix operations.

pub mod gemm;

// Re-export commonly used kernel functions
pub use gemm::{matmul, matmul_backward_left, matmul_backward_right};
