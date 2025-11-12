use crate::tensor::kernels::{cpu_gemm, GemmParams};
use tracing::{debug_span, instrument};

// ============================================================================
// Apple Accelerate Backend (macOS only, feature-gated)
// ============================================================================
// Uses the Accelerate framework's vecLib BLAS routines, which leverage
// Apple's AMX (Apple Matrix coprocessor) on M1/M2/M3 chips for optimal
// matrix multiplication performance.
// ============================================================================

#[cfg(all(target_os = "macos", feature = "accelerate"))]
pub(crate) mod accelerate_backend {
    use super::*;

    // External FFI bindings to Apple's Accelerate framework BLAS
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        /// Single-precision General Matrix Multiply (SGEMM)
        /// Computes: C := alpha * op(A) * op(B) + beta * C
        ///
        /// # Safety
        /// All pointers must be valid and properly aligned.
        /// Matrix dimensions must be consistent with the memory layouts.
        fn cblas_sgemm(
            order: i32,        // CblasRowMajor (101) or CblasColMajor (102)
            trans_a: i32,      // CblasNoTrans (111) or CblasTrans (112)
            trans_b: i32,      // CblasNoTrans (111) or CblasTrans (112)
            m: i32,            // Number of rows in op(A) and C
            n: i32,            // Number of columns in op(B) and C
            k: i32,            // Number of columns in op(A) and rows in op(B)
            alpha: f32,        // Scaling factor for A*B
            a: *const f32,     // Matrix A
            lda: i32,          // Leading dimension of A
            b: *const f32,     // Matrix B
            ldb: i32,          // Leading dimension of B
            beta: f32,         // Scaling factor for C
            c: *mut f32,       // Matrix C (output)
            ldc: i32,          // Leading dimension of C
        );
    }

    // CBLAS constants
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_TRANS: i32 = 112;

    #[instrument(skip_all, fields(
        op = "gemm_accelerate",
        m = if transpose_left { a_shape[1] } else { a_shape[0] },
        n = if transpose_right { b_shape[0] } else { b_shape[1] },
        k = if transpose_left { a_shape[0] } else { a_shape[1] },
        transpose_left,
        transpose_right,
        alpha,
        a_size = a_data.len(),
        b_size = b_data.len(),
        c_size = c_data.len()
    ))]
    pub fn gemm_accelerate(
        GemmParams {
            a_data,
            a_shape,
            transpose_left,
            b_data,
            b_shape,
            transpose_right,
            c_data,
            alpha,
        }: GemmParams,
    ) -> Result<(), String> {
        // Calculate result dimensions
        let (m, k, n) = debug_span!("CalculateDimensions").in_scope(|| {
            let m = if transpose_left {
                a_shape[1]
            } else {
                a_shape[0]
            };
            let k = if transpose_left {
                a_shape[0]
            } else {
                a_shape[1]
            };
            let n = if transpose_right {
                b_shape[0]
            } else {
                b_shape[1]
            };
            (m, k, n)
        });

        // Validate dimensions
        if c_data.len() != m * n {
            return Err(format!(
                "Output buffer size mismatch: expected {}, got {}",
                m * n,
                c_data.len()
            ));
        }

        // Validate input dimensions
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
        if k_a != k_b {
            return Err(format!(
                "Inner dimension mismatch: A has {}, B has {}",
                k_a, k_b
            ));
        }

        debug_span!("CallAccelerateBLAS", m, n, k, alpha).in_scope(|| {
            // Convert transpose flags to CBLAS constants
            let trans_a = if transpose_left {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            };
            let trans_b = if transpose_right {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            };

            // Leading dimensions (stride in row-major layout)
            // For row-major: lda = number of columns in the matrix as stored
            let lda = a_shape[1] as i32;
            let ldb = b_shape[1] as i32;
            let ldc = n as i32;

            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    trans_a,
                    trans_b,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha,
                    a_data.as_ptr(),
                    lda,
                    b_data.as_ptr(),
                    ldb,
                    0.0, // beta = 0 means overwrite C (not accumulate)
                    c_data.as_mut_ptr(),
                    ldc,
                );
            }
        });

        Ok(())
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Configuration for matrix multiplication backend selection
#[derive(Debug, Clone, Copy)]
pub struct MatmulConfig {
    /// Minimum number of operations (M*N*K) to use Accelerate
    pub accelerate_threshold: usize,
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self {
            // Accelerate is very efficient even for smaller matrices
            // due to AMX acceleration, so use a lower threshold than Metal
            accelerate_threshold: 1000,
        }
    }
}

/// Matrix multiplication: C = A @ B
pub fn matmul(a_data: &[f32], a_shape: &[usize], b_data: &[f32], b_shape: &[usize]) -> Vec<f32> {
    matmul_with_config(a_data, a_shape, b_data, b_shape, &MatmulConfig::default())
}

/// Matrix multiplication with configuration
pub fn matmul_with_config(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    config: &MatmulConfig,
) -> Vec<f32> {
    // Validate inputs
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
    let ops = m * k * n;

    // Try Accelerate for matrices above threshold
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    if ops >= config.accelerate_threshold {
        let mut result = vec![0.0; m * n];

        if let Err(e) = accelerate_backend::gemm_accelerate(GemmParams {
            a_data,
            a_shape: [a_shape[0], a_shape[1]],
            transpose_left: false,
            b_data,
            b_shape: [b_shape[0], b_shape[1]],
            transpose_right: false,
            c_data: &mut result,
            alpha: 1.0,
        }) {
            // Fall through to CPU if Accelerate fails
            #[cfg(debug_assertions)]
            eprintln!(
                "Accelerate matmul failed ({}), falling back to CPU",
                e
            );
        } else {
            return result;
        }
    }

    // CPU fallback
    cpu_gemm::matmul(a_data, a_shape, b_data, b_shape)
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

    let m = grad_shape[0];
    let k = b_shape[0];
    let n = grad_shape[1];
    let ops = m * k * n;

    // Try Accelerate for large matrices
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    if ops >= MatmulConfig::default().accelerate_threshold {
        if let Ok(()) = accelerate_backend::gemm_accelerate(GemmParams {
            a_data: grad_output,
            a_shape: [grad_shape[0], grad_shape[1]],
            transpose_left: false,
            b_data,
            b_shape: [b_shape[0], b_shape[1]],
            transpose_right: true,
            c_data: a_grad,
            alpha: 1.0,
        }) {
            return;
        }
    }

    // CPU fallback
    cpu_gemm::matmul_backward_left(grad_output, grad_shape, b_data, b_shape, a_grad);
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

    let m = a_shape[0];
    let k = a_shape[1];
    let n = grad_shape[1];
    let ops = m * k * n;

    // Try Accelerate for large matrices
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    if ops >= MatmulConfig::default().accelerate_threshold {
        if let Ok(()) = accelerate_backend::gemm_accelerate(GemmParams {
            a_data,
            a_shape: [a_shape[0], a_shape[1]],
            transpose_left: true,
            b_data: grad_output,
            b_shape: [grad_shape[0], grad_shape[1]],
            transpose_right: false,
            c_data: b_grad,
            alpha: 1.0,
        }) {
            return;
        }
    }

    // CPU fallback
    cpu_gemm::matmul_backward_right(a_data, a_shape, grad_output, grad_shape, b_grad);
}

/// Check if Apple Accelerate framework is available at runtime
pub fn has_accelerate_support() -> bool {
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    {
        // Accelerate is always available on macOS when compiled with the feature
        true
    }
    #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
    {
        false
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, target_os = "macos", feature = "accelerate"))]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], epsilon: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < epsilon)
    }

    #[test]
    fn test_accelerate_available() {
        assert!(
            has_accelerate_support(),
            "Accelerate should be available on macOS with accelerate feature"
        );
    }

    #[test]
    fn test_accelerate_matmul_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = vec![2, 3];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_shape = vec![3, 2];

        let result = matmul(&a, &a_shape, &b, &b_shape);

        let expected = vec![22.0, 28.0, 49.0, 64.0];
        assert!(
            approx_eq(&result, &expected, 1e-5),
            "Expected {:?}, got {:?}",
            expected,
            result
        );
    }

    #[test]
    fn test_accelerate_matmul_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let a_shape = vec![2, 2];
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let i_shape = vec![2, 2];

        let result = matmul(&a, &a_shape, &identity, &i_shape);
        assert!(approx_eq(&result, &a, 1e-5));
    }

    #[test]
    fn test_accelerate_vs_cpu_consistency() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let a_shape = vec![3, 3];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let b_shape = vec![3, 3];

        let cpu_result = cpu_gemm::matmul(&a, &a_shape, &b, &b_shape);

        let config = MatmulConfig {
            accelerate_threshold: 0,
        };
        let accelerate_result = matmul_with_config(&a, &a_shape, &b, &b_shape, &config);

        assert!(
            approx_eq(&accelerate_result, &cpu_result, 1e-4),
            "Accelerate result differs from CPU: {:?} vs {:?}",
            accelerate_result,
            cpu_result
        );
    }

    #[test]
    fn test_accelerate_matmul_large_matrix() {
        let size = 200;
        let a: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
        let a_shape = vec![size, size];
        let b: Vec<f32> = (0..size * size).map(|i| ((i + 1) % 10) as f32).collect();
        let b_shape = vec![size, size];

        let cpu_result = cpu_gemm::matmul(&a, &a_shape, &b, &b_shape);

        let config = MatmulConfig {
            accelerate_threshold: 0,
        };
        let accelerate_result = matmul_with_config(&a, &a_shape, &b, &b_shape, &config);

        assert!(approx_eq(&accelerate_result, &cpu_result, 1e-3));
    }

    #[test]
    fn test_accelerate_backward_left() {
        let grad_output = vec![1.0, 2.0, 3.0, 4.0];
        let grad_shape = vec![2, 2];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let b_shape = vec![2, 2];

        let mut a_grad = vec![0.0; 4];
        matmul_backward_left(&grad_output, &grad_shape, &b, &b_shape, &mut a_grad);

        let expected = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(&a_grad, &expected, 1e-5));
    }

    #[test]
    fn test_accelerate_backward_right() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let a_shape = vec![2, 2];
        let grad_output = vec![1.0, 2.0, 3.0, 4.0];
        let grad_shape = vec![2, 2];

        let mut b_grad = vec![0.0; 4];
        matmul_backward_right(&a, &a_shape, &grad_output, &grad_shape, &mut b_grad);

        let expected = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(&b_grad, &expected, 1e-5));
    }

    #[test]
    fn test_accelerate_transpose_operations() {
        // Test with transposed operations
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = vec![2, 3];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_shape = vec![3, 2];

        let config = MatmulConfig {
            accelerate_threshold: 0,
        };

        // Test backward passes which use transpose operations internally
        let mut a_grad = vec![0.0; 6];
        let grad_output = vec![1.0, 1.0, 1.0, 1.0];
        let grad_shape = vec![2, 2];

        matmul_backward_left(&grad_output, &grad_shape, &b, &b_shape, &mut a_grad);

        // Verify result makes sense
        assert!(a_grad.iter().all(|&x| x.is_finite()));
        assert!(a_grad.iter().any(|&x| x != 0.0));
    }
}

#[cfg(test)]
mod non_accelerate_tests {
    use super::*;

    #[test]
    fn test_has_accelerate_support_reports_correctly() {
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        {
            assert!(has_accelerate_support());
        }

        #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
        {
            assert!(!has_accelerate_support());
        }
    }

    #[test]
    fn test_matmul_always_works() {
        // This test works regardless of Accelerate availability
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = matmul(&a, &[2, 2], &b, &[2, 2]);

        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
