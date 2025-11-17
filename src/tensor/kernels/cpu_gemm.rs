use crate::tensor::kernels::GemmParams;

/// General Matrix Multiply (GEMM) operations
pub(crate) fn gemm_core(params: GemmParams) {
    let GemmParams {
        a_data,
        a_shape,
        transpose_left,
        b_data,
        b_shape,
        transpose_right,
        c_data,
        alpha,
    } = params;

    // Calculate dimensions based on transpose flags
    // For op(A): if transposed, [k x m], otherwise [m x k]
    // For op(B): if transposed, [n x k], otherwise [k x n]
    let m = if transpose_left {
        a_shape[1]
    } else {
        a_shape[0]
    };
    let k_a = if transpose_left {
        a_shape[0]
    } else {
        a_shape[1]
    };
    let k_b = if transpose_right {
        b_shape[1]
    } else {
        b_shape[0]
    };
    let n = if transpose_right {
        b_shape[0]
    } else {
        b_shape[1]
    };

    assert_eq!(k_a, k_b, "Inner dimension mismatch: {} != {}", k_a, k_b);
    assert_eq!(c_data.len(), m * n, "Output buffer size mismatch");

    let k = k_a;

    // Calculate strides based on transpose flags
    // For row-major storage:
    // - Normal: row_stride = num_cols, col_stride = 1
    // - Transposed: row_stride = 1, col_stride = num_cols
    let (a_row_stride, a_col_stride) = if transpose_left {
        (1, a_shape[1]) // Column becomes row, row becomes column
    } else {
        (a_shape[1], 1) // Standard row-major
    };

    let (b_row_stride, b_col_stride) = if transpose_right {
        (1, b_shape[1]) // Column becomes row, row becomes column
    } else {
        (b_shape[1], 1) // Standard row-major
    };

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

    let m = a_shape[0];
    let n = b_shape[1];
    let mut result = vec![0.0; m * n];

    gemm_core(GemmParams {
        a_data,
        a_shape: [a_shape[0], a_shape[1]],
        transpose_left: false,
        b_data,
        b_shape: [b_shape[0], b_shape[1]],
        transpose_right: false,
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
    assert_eq!(
        b_shape[1], grad_shape[1],
        "Dimension mismatch in backward left"
    );
    assert_eq!(
        a_grad.len(),
        grad_shape[0] * b_shape[0],
        "Gradient buffer size mismatch"
    );

    // dL/dA = grad_output @ B^T
    gemm_core(GemmParams {
        a_data: grad_output,
        a_shape: [grad_shape[0], grad_shape[1]],
        transpose_left: false,
        b_data,
        b_shape: [b_shape[0], b_shape[1]],
        transpose_right: true,
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
    assert_eq!(
        grad_shape[0], a_shape[0],
        "Dimension mismatch in backward right"
    );
    assert_eq!(
        b_grad.len(),
        a_shape[1] * grad_shape[1],
        "Gradient buffer size mismatch"
    );

    // dL/dB = A^T @ grad_output
    gemm_core(GemmParams {
        a_data,
        a_shape: [a_shape[0], a_shape[1]],
        transpose_left: true,
        b_data: grad_output,
        b_shape: [grad_shape[0], grad_shape[1]],
        transpose_right: false,
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
