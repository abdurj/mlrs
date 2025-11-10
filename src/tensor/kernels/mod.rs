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
struct GemmParams<'a> {
    a_data: &'a [f32],
    a_shape: [usize; 2],
    transpose_left: bool,
    b_data: &'a [f32],
    b_shape: [usize; 2],
    transpose_right: bool,
    c_data: &'a mut [f32],
    alpha: f32,
}

pub mod cpu_gemm;
pub mod metal_gemm;
