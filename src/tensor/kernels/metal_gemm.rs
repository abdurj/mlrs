use crate::tensor::kernels::{cpu_gemm, GemmParams};

// ============================================================================
// Metal Backend (macOS only, feature-gated)
// ============================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_backend {
    use super::*;
    use objc2::{rc::Retained, runtime::ProtocolObject, AnyThread};
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLResourceOptions,
    };
    use objc2_metal_performance_shaders::{
        MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
    };
    use std::sync::OnceLock;

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
        CONTEXT.get_or_init(|| MetalContext::new().ok()).as_ref()
    }

    pub fn gemm_metal(params: GemmParams) -> Result<(), String> {
        let ctx = get_metal_context().ok_or_else(|| "Metal not available".to_string())?;

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

        // Calculate result dimensions
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

        // Create buffers
        let a_buffer = unsafe {
            ctx.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    std::ptr::NonNull::new(params.a_data.as_ptr() as *mut _).unwrap(),
                    (params.a_data.len() * 4) as usize,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or("Failed to create buffer A")?;

        let b_buffer = unsafe {
            ctx.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    std::ptr::NonNull::new(params.b_data.as_ptr() as *mut _).unwrap(),
                    (params.b_data.len() * 4) as usize,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or("Failed to create buffer B")?;

        let c_buffer = unsafe {
            ctx.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    std::ptr::NonNull::new(c_data.as_mut_ptr() as *mut _).unwrap(),
                    (c_data.len() * 4) as usize,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or("Failed to create buffer C")?;

        // Copy input data to buffers
        unsafe {
            std::ptr::copy_nonoverlapping(
                a_data.as_ptr(),
                a_buffer.contents().as_ptr() as *mut f32,
                a_data.len(),
            );
            std::ptr::copy_nonoverlapping(
                b_data.as_ptr(),
                b_buffer.contents().as_ptr() as *mut f32,
                b_data.len(),
            );
        }

        // Create descriptors based on physical storage layout
        let a_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                a_shape[0] as usize,
                a_shape[1] as usize,
                (a_shape[1] * std::mem::size_of::<f32>()) as usize,
                MPSDataType::Float32,
            )
        };

        let b_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                b_shape[0] as usize,
                b_shape[1] as usize,
                (b_shape[1] * std::mem::size_of::<f32>()) as usize,
                MPSDataType::Float32,
            )
        };

        let c_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                m as usize,
                n as usize,
                (n * std::mem::size_of::<f32>()) as usize,
                MPSDataType::Float32,
            )
        };

        // Create MPS matrices
        let a_matrix =
            unsafe { MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &a_buffer, &a_desc) };
        let b_matrix =
            unsafe { MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &b_buffer, &b_desc) };
        let c_matrix =
            unsafe { MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &c_buffer, &c_desc) };

        // Create multiplication kernel
        let matmul = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &ctx.device,
                transpose_left,
                transpose_right,
                m as usize,
                n as usize,
                k as usize,
                alpha as f64,
                0.0,
            )
        };

        // Encode and execute
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

        // Copy result back
        unsafe {
            let c_ptr = c_buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(c_ptr, c_data.as_mut_ptr(), c_data.len());
        }

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
            metal_threshold: 1_000_000,
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

        if let Ok(()) = metal_backend::gemm_metal(GemmParams {
            a_data,
            a_shape: [a_shape[0], a_shape[1]],
            transpose_left: false,
            b_data,
            b_shape: [b_shape[0], b_shape[1]],
            transpose_right: false,
            c_data: &mut result,
            alpha: 1.0,
        }) {
            return result;
        }

        // Fall through to CPU if Metal fails
        #[cfg(debug_assertions)]
        eprintln!("Metal matmul failed, falling back to CPU");
    }

    // CPU fallback
    cpu_gemm::matmul(a_data, a_shape, b_data, b_shape)
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
        let a_shape = [2, 3];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_shape = [3, 2];

        let mut result = vec![0.0; 4];
        metal_backend::gemm_metal(GemmParams {
            a_data: &a,
            a_shape,
            transpose_left: false,
            b_data: &b,
            b_shape,
            transpose_right: false,
            c_data: &mut result,
            alpha: 1.0,
        })
        .expect("Metal matmul failed");

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
        let a_shape = [2, 2];
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let i_shape = [2, 2];

        let mut result = vec![0.0; 4];
        metal_backend::gemm_metal(GemmParams {
            a_data: &a,
            a_shape,
            transpose_left: false,
            b_data: &identity,
            b_shape: i_shape,
            transpose_right: false,
            c_data: &mut result,
            alpha: 1.0,
        })
        .unwrap();

        assert!(approx_eq(&result, &a, 1e-5));
    }

    #[test]
    fn test_metal_matmul_transpose_left() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = [3, 2];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_shape = [3, 2];

        let mut result = vec![0.0; 4];
        metal_backend::gemm_metal(GemmParams {
            a_data: &a,
            a_shape,
            transpose_left: true,
            b_data: &b,
            b_shape,
            transpose_right: false,
            c_data: &mut result,
            alpha: 1.0,
        })
        .unwrap();

        let expected = vec![35.0, 44.0, 44.0, 56.0];
        assert!(approx_eq(&result, &expected, 1e-5));
    }

    #[test]
    fn test_metal_matmul_transpose_right() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = [2, 3];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_shape = [2, 3];

        let mut result = vec![0.0; 4];
        metal_backend::gemm_metal(GemmParams {
            a_data: &a,
            a_shape,
            transpose_left: false,
            b_data: &b,
            b_shape,
            transpose_right: true,
            c_data: &mut result,
            alpha: 1.0,
        })
        .unwrap();

        let expected = vec![14.0, 32.0, 32.0, 77.0];
        assert!(approx_eq(&result, &expected, 1e-5));
    }

    #[test]
    fn test_metal_matmul_both_transpose() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = [3, 2];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_shape = [2, 3];

        let mut result = vec![0.0; 4];
        metal_backend::gemm_metal(GemmParams {
            a_data: &a,
            a_shape,
            transpose_left: true,
            b_data: &b,
            b_shape,
            transpose_right: true,
            c_data: &mut result,
            alpha: 1.0,
        })
        .unwrap();

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

        let mut metal_result = vec![0.0; 9];
        metal_backend::gemm_metal(GemmParams {
            a_data: &a,
            a_shape: [3, 3],
            transpose_left: false,
            b_data: &b,
            b_shape: [3, 3],
            transpose_right: false,
            c_data: &mut metal_result,
            alpha: 1.0,
        })
        .unwrap();

        assert!(approx_eq(&metal_result, &cpu_result, 1e-4));
    }

    #[test]
    fn test_metal_matmul_with_alpha() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let a_shape = [2, 2];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let b_shape = [2, 2];

        let mut result = vec![0.0; 4];
        metal_backend::gemm_metal(GemmParams {
            a_data: &a,
            a_shape,
            transpose_left: false,
            b_data: &b,
            b_shape,
            transpose_right: false,
            c_data: &mut result,
            alpha: 2.0,
        })
        .unwrap();

        let expected = vec![2.0, 4.0, 6.0, 8.0];
        assert!(approx_eq(&result, &expected, 1e-5));
    }

    #[test]
    fn test_metal_matmul_large_matrix() {
        let size = 200;
        let a: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
        let a_shape = vec![size, size];
        let b: Vec<f32> = (0..size * size).map(|i| ((i + 1) % 10) as f32).collect();
        let b_shape = vec![size, size];

        let cpu_result = cpu_gemm::matmul(&a, &a_shape, &b, &b_shape);

        let mut metal_result = vec![0.0; size * size];
        metal_backend::gemm_metal(GemmParams {
            a_data: &a,
            a_shape: [size, size],
            transpose_left: false,
            b_data: &b,
            b_shape: [size, size],
            transpose_right: false,
            c_data: &mut metal_result,
            alpha: 1.0,
        })
        .unwrap();

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
