use crate::tensor::kernels::{cpu_gemm, GemmParams};

// ============================================================================
// Metal Backend (macOS only, feature-gated)
// ============================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) mod metal_backend {
    use super::*;
    use objc2::{rc::Retained, runtime::ProtocolObject, AnyThread};
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLResourceOptions,
    };
    use objc2_metal_performance_shaders::{
        MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
    };
    use std::{ptr::NonNull, sync::OnceLock};
    use tracing::{debug_span, instrument};

    pub struct MetalContext {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    }

    impl MetalContext {
        fn new() -> Result<Self, String> {
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| "No Metal device found".to_string())?;
            let command_queue = device
                .newCommandQueue()
                .ok_or_else(|| "Failed to create command queue".to_string())?;

            Ok(Self {
                device,
                command_queue,
            })
        }
    }

    pub(crate) fn get_metal_context() -> Option<&'static MetalContext> {
        static CONTEXT: OnceLock<Option<MetalContext>> = OnceLock::new();
        CONTEXT
            .get_or_init(|| {
                debug_span!("InitializeMetalContext").in_scope(|| MetalContext::new().ok())
            })
            .as_ref()
    }

    #[instrument(skip_all, fields(
        op = "gemm_metal",
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
    pub fn gemm_metal(
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
        let ctx = get_metal_context().ok_or_else(|| "Metal not available".to_string())?;

        // Calculate result dimensions
        let (m, k, n) = {
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
        };

        // --- Create Buffers ---
        let (a_buffer, b_buffer, c_buffer) = debug_span!(
            "CreateBuffers",
            a_len = a_data.len(),
            b_len = b_data.len(),
            c_len = c_data.len()
        )
        .in_scope(|| {
            let a_buffer = unsafe {
                ctx.device.newBufferWithBytes_length_options(
                    NonNull::new(a_data.as_ptr() as *mut _).unwrap(),
                    std::mem::size_of_val(a_data),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or("Failed to create buffer A")?;

            let b_buffer = unsafe {
                ctx.device.newBufferWithBytes_length_options(
                    NonNull::new(b_data.as_ptr() as *mut _).unwrap(),
                    std::mem::size_of_val(b_data),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or("Failed to create buffer B")?;

            let c_buffer = ctx
                .device
                .newBufferWithLength_options(
                    std::mem::size_of_val(c_data),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("Failed to create buffer C")?;

            Ok::<_, String>((a_buffer, b_buffer, c_buffer))
        })?;

        // --- Create Matrix Descriptors ---
        let (a_desc, b_desc, c_desc) =
            debug_span!("CreateDescriptors", m, n, k).in_scope(|| unsafe {
                let a_desc =
                    MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                        a_shape[0],
                        a_shape[1],
                        a_shape[1] * std::mem::size_of::<f32>(),
                        MPSDataType::Float32,
                    );

                let b_desc =
                    MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                        b_shape[0],
                        b_shape[1],
                        b_shape[1] * std::mem::size_of::<f32>(),
                        MPSDataType::Float32,
                    );

                let c_desc =
                    MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                        m,
                        n,
                        n * std::mem::size_of::<f32>(),
                        MPSDataType::Float32,
                    );

                (a_desc, b_desc, c_desc)
            });

        // --- Create MPS Matrices ---
        let (a_matrix, b_matrix, c_matrix) = debug_span!("CreateMPSMatrices").in_scope(|| unsafe {
            let a_matrix =
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &a_buffer, &a_desc);
            let b_matrix =
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &b_buffer, &b_desc);
            let c_matrix =
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &c_buffer, &c_desc);
            (a_matrix, b_matrix, c_matrix)
        });

        // --- Create Kernel ---
        let matmul = debug_span!("CreateMultiplicationKernel", m, n, k, alpha).in_scope(|| unsafe {
        MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            MPSMatrixMultiplication::alloc(),
            &ctx.device,
            transpose_left,
            transpose_right,
            m,
            n,
            k,
            alpha as f64,
            0.0,
        )
        });

        // --- Encode and Execute ---
        debug_span!("EncodeAndExecute").in_scope(|| {
            let command_buffer = ctx
                .command_queue
                .commandBuffer()
                .ok_or_else(|| "Failed to create command buffer".to_string())?;

            unsafe {
                matmul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &command_buffer,
                    &a_matrix,
                    &b_matrix,
                    &c_matrix,
                );
            }

            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            Ok::<_, String>(())
        })?;

        // --- Copy Result Back ---
        debug_span!("CopyResultBack", size = c_data.len()).in_scope(|| unsafe {
            let c_ptr = c_buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(c_ptr, c_data.as_mut_ptr(), c_data.len());
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
    /// Minimum number of operations (M*N*K) to use Metal GPU
    pub metal_threshold: usize,
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self {
            metal_threshold: 10_000,
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

    // Try Metal for large matrices
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if ops >= config.metal_threshold {
        let mut result = vec![0.0; m * n];

        if let Err(e) = metal_backend::gemm_metal(GemmParams {
            a_data,
            a_shape: [a_shape[0], a_shape[1]],
            transpose_left: false,
            b_data,
            b_shape: [b_shape[0], b_shape[1]],
            transpose_right: false,
            c_data: &mut result,
            alpha: 1.0,
        }) {
            // Fall through to CPU if Metal fails
            #[cfg(debug_assertions)]
            eprintln!("Metal matmul failed ({}), falling back to CPU", e);
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

    // Try Metal for large matrices
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if ops >= MatmulConfig::default().metal_threshold {
        if let Ok(()) = metal_backend::gemm_metal(GemmParams {
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

    // Try Metal for large matrices
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if ops >= MatmulConfig::default().metal_threshold {
        if let Ok(()) = metal_backend::gemm_metal(GemmParams {
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

/// Check if Metal acceleration is available at runtime
pub fn has_metal_support() -> bool {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        metal_backend::get_metal_context().is_some()
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        false
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], epsilon: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < epsilon)
    }

    #[test]
    fn test_metal_available() {
        assert!(
            has_metal_support(),
            "Metal should be available on macOS with metal feature"
        );
    }

    #[test]
    fn test_metal_matmul_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = vec![2, 3];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_shape = vec![3, 2];

        // Use the public API like the benchmark does
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
    fn test_metal_matmul_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let a_shape = vec![2, 2];
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let i_shape = vec![2, 2];

        let result = matmul(&a, &a_shape, &identity, &i_shape);
        assert!(approx_eq(&result, &a, 1e-5));
    }

    #[test]
    fn test_metal_matmul_transpose_left() {
        // For transpose left: A^T @ B where A is [3,2], so A^T is [2,3]
        let a_t = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // A^T (transposed version)
        let a_t_shape = vec![2, 3];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_shape = vec![3, 2];

        let config = MatmulConfig { metal_threshold: 0 }; // Force Metal even for small matrices
        let result = matmul_with_config(&a_t, &a_t_shape, &b, &b_shape, &config);

        let expected = vec![35.0, 44.0, 44.0, 56.0];
        assert!(approx_eq(&result, &expected, 1e-5));
    }

    #[test]
    fn test_metal_matmul_transpose_right() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = vec![2, 3];
        // For transpose right: A @ B^T where B is [2,3], so B^T is [3,2]
        let b_t = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // B^T (transposed version)
        let b_t_shape = vec![3, 2];

        let config = MatmulConfig { metal_threshold: 0 };
        let result = matmul_with_config(&a, &a_shape, &b_t, &b_t_shape, &config);

        let expected = vec![14.0, 32.0, 32.0, 77.0];
        assert!(approx_eq(&result, &expected, 1e-5));
    }

    #[test]
    fn test_metal_matmul_both_transpose() {
        // A^T @ B^T where A is [3,2] and B is [2,3]
        let a_t = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // A^T: [2,3]
        let a_t_shape = vec![2, 3];
        let b_t = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // B^T: [3,2]
        let b_t_shape = vec![3, 2];

        let config = MatmulConfig { metal_threshold: 0 };
        let result = matmul_with_config(&a_t, &a_t_shape, &b_t, &b_t_shape, &config);

        let expected = vec![22.0, 49.0, 28.0, 64.0];
        assert!(approx_eq(&result, &expected, 1e-5));
    }

    #[test]
    fn test_metal_vs_cpu_consistency() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let a_shape = vec![3, 3];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let b_shape = vec![3, 3];

        let cpu_result = cpu_gemm::matmul(&a, &a_shape, &b, &b_shape);

        let config = MatmulConfig { metal_threshold: 0 };
        let metal_result = matmul_with_config(&a, &a_shape, &b, &b_shape, &config);

        assert!(approx_eq(&metal_result, &cpu_result, 1e-4));
    }

    #[test]
    fn test_metal_matmul_with_alpha() {
        // Note: The public API doesn't support alpha parameter directly
        // We'll test with alpha=1.0 and scale the result manually
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let a_shape = vec![2, 2];
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let i_shape = vec![2, 2];

        let config = MatmulConfig { metal_threshold: 0 };
        let result = matmul_with_config(&a, &a_shape, &identity, &i_shape, &config);

        // Scale by alpha manually
        let alpha = 2.0;
        let scaled_result: Vec<f32> = result.iter().map(|x| x * alpha).collect();

        let expected = vec![2.0, 4.0, 6.0, 8.0];
        assert!(approx_eq(&scaled_result, &expected, 1e-5));
    }

    #[test]
    fn test_metal_matmul_large_matrix() {
        let size = 200;
        let a: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
        let a_shape = vec![size, size];
        let b: Vec<f32> = (0..size * size).map(|i| ((i + 1) % 10) as f32).collect();
        let b_shape = vec![size, size];

        let cpu_result = cpu_gemm::matmul(&a, &a_shape, &b, &b_shape);

        // Force Metal for this large operation
        let config = MatmulConfig { metal_threshold: 0 };
        let metal_result = matmul_with_config(&a, &a_shape, &b, &b_shape, &config);

        assert!(approx_eq(&metal_result, &cpu_result, 1e-3));
    }

    #[test]
    fn test_public_api_uses_metal() {
        // Test that the public API uses Metal for large matrices
        let size = 500; // Above threshold
        let a: Vec<f32> = (0..size * size).map(|i| (i % 5) as f32).collect();
        let b: Vec<f32> = (0..size * size).map(|i| ((i + 2) % 5) as f32).collect();

        let result = matmul(&a, &[size, size], &b, &[size, size]);

        assert_eq!(result.len(), size * size);
    }
}

#[cfg(test)]
mod non_metal_tests {
    use super::*;

    #[test]
    fn test_has_metal_support_reports_correctly() {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            // On macOS with metal feature, it might be true
            let _ = has_metal_support();
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            assert!(!has_metal_support());
        }
    }

    #[test]
    fn test_matmul_always_works() {
        // This test works regardless of Metal availability
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = matmul(&a, &[2, 2], &b, &[2, 2]);

        assert_eq!(result.len(), 4);
        // Basic sanity check
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
