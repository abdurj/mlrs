/// General Matrix Multiply (GEMM) operations
/// This module contains the core matrix multiplication implementations
/// 
/// Internal core GEMM operation using explicit strides
/// Computes: C += alpha * A @ B
/// 
/// # Arguments
/// * `a_data` - Flattened data of matrix A
/// * `m` - Number of rows in the result (and rows in A)
/// * `k` - Inner dimension (cols in A, rows in B)
/// * `a_row_stride` - Stride between consecutive rows of A
/// * `a_col_stride` - Stride between consecutive cols of A
/// * `b_data` - Flattened data of matrix B
/// * `n` - Number of cols in the result (and cols in B)
/// * `b_row_stride` - Stride between consecutive rows of B
/// * `b_col_stride` - Stride between consecutive cols of B
/// * `c_data` - Output buffer to accumulate into (shape: [m, n])
/// * `alpha` - Scaling factor (usually 1.0)
struct GemmParams<'a> {
    a_data: &'a [f32],
    m: usize,
    k: usize,
    a_row_stride: usize,
    a_col_stride: usize,
    b_data: &'a [f32],
    n: usize,
    b_row_stride: usize,
    b_col_stride: usize,
    c_data: &'a mut [f32],
    alpha: f32,
}

fn gemm_core(params: GemmParams) {
    let GemmParams {
        a_data,
        m,
        k,
        a_row_stride,
        a_col_stride,
        b_data,
        n,
        b_row_stride,
        b_col_stride,
        c_data,
        alpha,
    } = params;

    assert_eq!(c_data.len(), m * n, "Output buffer size mismatch");

    // Naive triple-loop implementation
    // TODO: Optimize with blocking, vectorization, or BLAS
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                let a_idx = i * a_row_stride + p * a_col_stride;
                let b_idx = p * b_row_stride + j * b_col_stride;
                sum += a_data[a_idx] * b_data[b_idx];
            }
            c_data[i * n + j] += alpha * sum;
        }
    }
}

/// Performs matrix multiplication: C = A @ B
/// 
/// # Arguments
/// * `a_data` - Flattened data of matrix A (row-major)
/// * `a_shape` - Shape of matrix A as [rows, cols]
/// * `b_data` - Flattened data of matrix B (row-major)
/// * `b_shape` - Shape of matrix B as [rows, cols]
/// 
/// # Returns
/// * Flattened result matrix C with shape [a_shape[0], b_shape[1]]
/// 
/// # Panics
/// * If matrices have incompatible dimensions (a_shape[1] != b_shape[0])
pub fn matmul(a_data: &[f32], a_shape: &[usize], b_data: &[f32], b_shape: &[usize]) -> Vec<f32> {
    assert_eq!(a_shape.len(), 2, "Matrix A must be 2D");
    assert_eq!(b_shape.len(), 2, "Matrix B must be 2D");
    assert_eq!(
        a_shape[1], b_shape[0],
        "Incompatible dimensions: A has {} columns but B has {} rows",
        a_shape[1], b_shape[0]
    );

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];
    let mut result = vec![0.0; m * n];

    // Row-major strides: row_stride = num_cols, col_stride = 1
    gemm_core(GemmParams {
        a_data,
        m,
        k,
        a_row_stride: a_shape[1],
        a_col_stride: 1,
        b_data,
        n,
        b_row_stride: b_shape[1],
        b_col_stride: 1,
        c_data: &mut result,
        alpha: 1.0,
    });

    result
}

/// Computes gradient with respect to the left matrix (A) in matrix multiplication
/// Given grad_output (dL/dC) and B, computes dL/dA = grad_output @ B^T
/// 
/// # Arguments
/// * `grad_output` - Gradient of output (shape: [m, n])
/// * `grad_shape` - Shape of grad_output as [m, n]
/// * `b_data` - Right matrix from forward pass (shape: [k, n])
/// * `b_shape` - Shape of B as [k, n]
/// * `a_grad` - Gradient buffer to accumulate into (shape: [m, k])
pub fn matmul_backward_left(
    grad_output: &[f32],
    grad_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    a_grad: &mut [f32],
) {
    assert_eq!(b_shape[1], grad_shape[1], "Dimension mismatch in backward left");
    assert_eq!(a_grad.len(), grad_shape[0] * b_shape[0], "Gradient buffer size mismatch");

    let m = grad_shape[0];
    let n = grad_shape[1];
    let k = b_shape[0];

    // dL/dA = grad_output @ B^T
    // B^T has strides: row_stride = 1 (column becomes row), col_stride = b_shape[1] (row becomes column)
    gemm_core(GemmParams {
        a_data: grad_output,
        m,
        k: n,
        a_row_stride: grad_shape[1],
        a_col_stride: 1,
        b_data,
        n: k,
        b_row_stride: 1,
        b_col_stride: b_shape[1],
        c_data: a_grad,
        alpha: 1.0,
    });
}

/// Computes gradient with respect to the right matrix (B) in matrix multiplication
/// Given A and grad_output (dL/dC), computes dL/dB = A^T @ grad_output
/// 
/// # Arguments
/// * `a_data` - Left matrix from forward pass (shape: [m, k])
/// * `a_shape` - Shape of A as [m, k]
/// * `grad_output` - Gradient of output (shape: [m, n])
/// * `grad_shape` - Shape of grad_output as [m, n]
/// * `b_grad` - Gradient buffer to accumulate into (shape: [k, n])
pub fn matmul_backward_right(
    a_data: &[f32],
    a_shape: &[usize],
    grad_output: &[f32],
    grad_shape: &[usize],
    b_grad: &mut [f32],
) {
    assert_eq!(grad_shape[0], a_shape[0], "Dimension mismatch in backward right");
    assert_eq!(b_grad.len(), a_shape[1] * grad_shape[1], "Gradient buffer size mismatch");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = grad_shape[1];

    // dL/dB = A^T @ grad_output
    // A^T has strides: row_stride = 1 (column becomes row), col_stride = a_shape[1] (row becomes column)
    gemm_core(GemmParams {
        a_data,
        m: k,
        k: m,
        a_row_stride: 1,
        a_col_stride: a_shape[1],
        b_data: grad_output,
        n,
        b_row_stride: grad_shape[1],
        b_col_stride: 1,
        c_data: b_grad,
        alpha: 1.0,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_basic() {
        // Test: [2x3] @ [3x2] = [2x2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = vec![2, 3];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_shape = vec![3, 2];

        let result = matmul(&a, &a_shape, &b, &b_shape);

        // Expected:
        // [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6]   = [22, 28]
        // [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6]   = [49, 64]
        assert_eq!(result, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_matmul_identity() {
        // Test multiplication with identity matrix
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let a_shape = vec![2, 2];
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let i_shape = vec![2, 2];

        let result = matmul(&a, &a_shape, &identity, &i_shape);
        assert_eq!(result, a);
    }

    #[test]
    fn test_matmul_backward() {
        // Forward: C = A @ B
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let a_shape = vec![2, 2];
        let b = vec![2.0, 0.0, 0.0, 2.0];
        let b_shape = vec![2, 2];

        // Gradient from output
        let grad_output = vec![1.0, 1.0, 1.0, 1.0];
        let grad_shape = vec![2, 2];

        // Test backward for A
        let mut a_grad = vec![0.0; 4];
        matmul_backward_left(&grad_output, &grad_shape, &b, &b_shape, &mut a_grad);
        // dL/dA = grad_output @ B^T = [[1,1],[1,1]] @ [[2,0],[0,2]] = [[2,2],[2,2]]
        assert_eq!(a_grad, vec![2.0, 2.0, 2.0, 2.0]);

        // Test backward for B
        let mut b_grad = vec![0.0; 4];
        matmul_backward_right(&a, &a_shape, &grad_output, &grad_shape, &mut b_grad);
        // dL/dB = A^T @ grad_output = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        assert_eq!(b_grad, vec![4.0, 4.0, 6.0, 6.0]);
    }
}
