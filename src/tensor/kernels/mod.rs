//! Kernel implementations for tensor operations
//!
//! This module contains optimized kernel implementations for various
//! tensor operations, particularly matrix operations.

/// This module contains the core matrix multiplication implementations
///
/// Internal core GEMM operation with transpose support
/// Computes: C += alpha * op(A) @ op(B)
/// where op(X) is either X or X^T depending on the transpose flag
///
/// # Arguments
/// * `a_data` - Flattened data of matrix A (row-major)
/// * `a_shape` - Shape of matrix A as [rows, cols]
/// * `transpose_left` - Whether to transpose A
/// * `b_data` - Flattened data of matrix B (row-major)
/// * `b_shape` - Shape of matrix B as [rows, cols]
/// * `transpose_right` - Whether to transpose B
/// * `c_data` - Output buffer to accumulate into
/// * `alpha` - Scaling factor (usually 1.0)
pub(crate) struct GemmParams<'a> {
    pub a_data: &'a [f32],
    pub a_shape: [usize; 2],
    pub transpose_left: bool,
    pub b_data: &'a [f32],
    pub b_shape: [usize; 2],
    pub transpose_right: bool,
    pub c_data: &'a mut [f32],
    pub alpha: f32,
}

pub mod cpu_gemm;
pub mod metal_gemm;
pub mod amx_gemm;
pub mod backend;

// Re-export the unified backend API for easy access
pub use backend::{
    matmul, matmul_backward_left, matmul_backward_right, matmul_with_config, Backend,
    MatmulConfig, available_backends, print_backend_info,
};
