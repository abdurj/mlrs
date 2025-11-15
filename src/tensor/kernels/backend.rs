//! Backend abstraction layer for matrix multiplication
//!
//! This module provides a unified interface for selecting and using different
//! GEMM backends: CPU (naive), Apple Accelerate (AMX), and Metal (GPU).
//!
//! The backend selection can be controlled via:
//! 1. Runtime configuration (MatmulConfig with backend preference)
//! 2. Matrix size heuristics (automatic selection based on operation count)
//! 3. Environment variables (for easy experimentation)

use std::str::FromStr;

use super::{cpu_gemm, GemmParams};

#[cfg(all(target_os = "macos", feature = "accelerate"))]
use super::amx_gemm;

#[cfg(all(target_os = "macos", feature = "metal"))]
use super::metal_gemm;

/// Available backends for matrix multiplication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Pure CPU implementation (always available)
    Cpu,
    /// Apple Accelerate framework with AMX (macOS only)
    Accelerate,
    /// Metal GPU acceleration (macOS only)
    Metal,
    /// Automatically select best backend based on matrix size
    Auto,
}

impl Backend {
    /// Check if this backend is available on the current platform
    pub fn is_available(&self) -> bool {
        match self {
            Backend::Cpu => true,
            Backend::Accelerate => {
                #[cfg(all(target_os = "macos", feature = "accelerate"))]
                {
                    amx_gemm::has_accelerate_support()
                }
                #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
                {
                    false
                }
            }
            Backend::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    metal_gemm::has_metal_support()
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    false
                }
            }
            Backend::Auto => true, // Auto is always available (falls back to CPU)
        }
    }

    /// Get a human-readable name for this backend
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Accelerate => "Accelerate (AMX)",
            Backend::Metal => "Metal (GPU)",
            Backend::Auto => "Auto",
        }
    }

    /// Select the best backend based on matrix size and available backends
    fn select_for_size(&self, ops: usize) -> Backend {
        match self {
            Backend::Auto => {
                // Heuristics for automatic backend selection:
                // - Very small matrices (< 1000 ops): CPU (overhead of acceleration not worth it)
                // - Small matrices (< 100k ops): Accelerate if available (low overhead)
                // - Medium matrices (100k - 10M ops): Accelerate preferred
                // - Large matrices (> 10M ops): Metal or Accelerate, both are good

                if ops < 1000 {
                    Backend::Cpu
                } else if ops < 10_000_000 {
                    // Prefer Accelerate for small to medium matrices (lower overhead than Metal)
                    #[cfg(all(target_os = "macos", feature = "accelerate"))]
                    if amx_gemm::has_accelerate_support() {
                        return Backend::Accelerate;
                    }

                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    if metal_gemm::has_metal_support() {
                        return Backend::Metal;
                    }

                    Backend::Cpu
                } else {
                    // For very large matrices, Metal and Accelerate are both excellent
                    // Try Metal first (may have edge for very large matrices)
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    if metal_gemm::has_metal_support() {
                        return Backend::Metal;
                    }

                    #[cfg(all(target_os = "macos", feature = "accelerate"))]
                    if amx_gemm::has_accelerate_support() {
                        return Backend::Accelerate;
                    }

                    Backend::Cpu
                }
            }
            // For explicit backend selection, return as-is
            _ => *self,
        }
    }
}

/// Error type for backend parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseBackendError;

impl std::fmt::Display for ParseBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid backend name")
    }
}

impl std::error::Error for ParseBackendError {}

impl FromStr for Backend {
    type Err = ParseBackendError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Backend::Cpu),
            "accelerate" | "amx" => Ok(Backend::Accelerate),
            "metal" | "gpu" => Ok(Backend::Metal),
            "auto" => Ok(Backend::Auto),
            _ => Err(ParseBackendError),
        }
    }
}

/// Configuration for matrix multiplication backend selection
#[derive(Debug, Clone)]
pub struct MatmulConfig {
    /// Preferred backend (can be Backend::Auto for automatic selection)
    pub backend: Backend,
    /// Minimum operations for using acceleration (only used with Backend::Auto)
    pub acceleration_threshold: usize,
    /// Enable debug logging of backend selection
    pub debug_backend_selection: bool,
}

impl Default for MatmulConfig {
    fn default() -> Self {
        // Check environment variable for backend override
        let backend = std::env::var("RUST_NN_BACKEND")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(Backend::Auto);

        let debug_backend_selection = std::env::var("RUST_NN_DEBUG_BACKEND")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        Self {
            backend,
            acceleration_threshold: 1000,
            debug_backend_selection,
        }
    }
}

impl MatmulConfig {
    /// Create a config with a specific backend
    pub fn with_backend(backend: Backend) -> Self {
        Self {
            backend,
            ..Default::default()
        }
    }

    /// Create a config that always uses CPU
    pub fn cpu_only() -> Self {
        Self::with_backend(Backend::Cpu)
    }

    /// Create a config that prefers Accelerate
    pub fn prefer_accelerate() -> Self {
        Self::with_backend(Backend::Accelerate)
    }

    /// Create a config that prefers Metal
    pub fn prefer_metal() -> Self {
        Self::with_backend(Backend::Metal)
    }
}

/// Execute GEMM with the specified configuration
fn gemm_with_backend(params: GemmParams, config: &MatmulConfig) -> Result<(), String> {
    // Calculate operation count for auto-selection
    let m = if params.transpose_left {
        params.a_shape[1]
    } else {
        params.a_shape[0]
    };
    let k = if params.transpose_left {
        params.a_shape[0]
    } else {
        params.a_shape[1]
    };
    let n = if params.transpose_right {
        params.b_shape[0]
    } else {
        params.b_shape[1]
    };
    let ops = m * k * n;

    // Select backend
    let selected_backend = if ops < config.acceleration_threshold {
        Backend::Cpu
    } else {
        config.backend.select_for_size(ops)
    };

    // Debug logging
    if config.debug_backend_selection {
        eprintln!(
            "[GEMM] {}x{}x{} ({} ops) -> {}",
            m,
            k,
            n,
            ops,
            selected_backend.name()
        );
    }

    // Execute with selected backend
    match selected_backend {
        Backend::Cpu => {
            cpu_gemm::gemm_core(params);
            Ok(())
        }
        Backend::Accelerate => {
            #[cfg(all(target_os = "macos", feature = "accelerate"))]
            {
                amx_gemm::accelerate_backend::gemm_accelerate(params)
            }
            #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
            {
                Err("Accelerate backend not available".to_string())
            }
        }
        Backend::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                metal_gemm::metal_backend::gemm_metal(params)
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Err("Metal backend not available".to_string())
            }
        }
        Backend::Auto => {
            unreachable!("Auto backend should be resolved by select_for_size")
        }
    }
}

/// Matrix multiplication: C = A @ B
pub fn matmul(a_data: &[f32], a_shape: &[usize], b_data: &[f32], b_shape: &[usize]) -> Vec<f32> {
    matmul_with_config(
        a_data,
        a_shape,
        b_data,
        b_shape,
        &MatmulConfig::prefer_accelerate(),
    )
}

/// Matrix multiplication with custom configuration
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
    let n = b_shape[1];
    let mut result = vec![0.0; m * n];

    let gemm_result = gemm_with_backend(
        GemmParams {
            a_data,
            a_shape: [a_shape[0], a_shape[1]],
            transpose_left: false,
            b_data,
            b_shape: [b_shape[0], b_shape[1]],
            transpose_right: false,
            c_data: &mut result,
            alpha: 1.0,
        },
        config,
    );

    // If backend fails, fall back to CPU
    if let Err(e) = gemm_result {
        #[cfg(debug_assertions)]
        eprintln!("Backend failed ({}), falling back to CPU", e);

        result.fill(0.0); // Clear partial results
        cpu_gemm::gemm_core(GemmParams {
            a_data,
            a_shape: [a_shape[0], a_shape[1]],
            transpose_left: false,
            b_data,
            b_shape: [b_shape[0], b_shape[1]],
            transpose_right: false,
            c_data: &mut result,
            alpha: 1.0,
        });
    }

    result
}

/// Computes gradient with respect to the left matrix (A)
pub fn matmul_backward_left(
    grad_output: &[f32],
    grad_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    a_grad: &mut [f32],
) {
    matmul_backward_left_with_config(
        grad_output,
        grad_shape,
        b_data,
        b_shape,
        a_grad,
        &MatmulConfig::prefer_accelerate(),
    )
}

/// Computes gradient with respect to the left matrix with custom config
pub fn matmul_backward_left_with_config(
    grad_output: &[f32],
    grad_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    a_grad: &mut [f32],
    config: &MatmulConfig,
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

    let gemm_result = gemm_with_backend(
        GemmParams {
            a_data: grad_output,
            a_shape: [grad_shape[0], grad_shape[1]],
            transpose_left: false,
            b_data,
            b_shape: [b_shape[0], b_shape[1]],
            transpose_right: true,
            c_data: a_grad,
            alpha: 1.0,
        },
        config,
    );

    if gemm_result.is_err() {
        cpu_gemm::matmul_backward_left(grad_output, grad_shape, b_data, b_shape, a_grad);
    }
}

/// Computes gradient with respect to the right matrix (B)
pub fn matmul_backward_right(
    a_data: &[f32],
    a_shape: &[usize],
    grad_output: &[f32],
    grad_shape: &[usize],
    b_grad: &mut [f32],
) {
    matmul_backward_right_with_config(
        a_data,
        a_shape,
        grad_output,
        grad_shape,
        b_grad,
        &MatmulConfig::prefer_accelerate(),
    )
}

/// Computes gradient with respect to the right matrix with custom config
pub fn matmul_backward_right_with_config(
    a_data: &[f32],
    a_shape: &[usize],
    grad_output: &[f32],
    grad_shape: &[usize],
    b_grad: &mut [f32],
    config: &MatmulConfig,
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

    let gemm_result = gemm_with_backend(
        GemmParams {
            a_data,
            a_shape: [a_shape[0], a_shape[1]],
            transpose_left: true,
            b_data: grad_output,
            b_shape: [grad_shape[0], grad_shape[1]],
            transpose_right: false,
            c_data: b_grad,
            alpha: 1.0,
        },
        config,
    );

    if gemm_result.is_err() {
        cpu_gemm::matmul_backward_right(a_data, a_shape, grad_output, grad_shape, b_grad);
    }
}

/// Get list of available backends on this platform
pub fn available_backends() -> Vec<Backend> {
    let mut backends = vec![Backend::Cpu]; // CPU always available

    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    if amx_gemm::has_accelerate_support() {
        backends.push(Backend::Accelerate);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    if metal_gemm::has_metal_support() {
        backends.push(Backend::Metal);
    }

    backends
}

/// Print information about available backends
pub fn print_backend_info() {
    println!("Available GEMM backends:");
    for backend in available_backends() {
        println!("  ✓ {}", backend.name());
    }

    println!("\nTo select a specific backend, set RUST_NN_BACKEND environment variable:");
    println!("  export RUST_NN_BACKEND=cpu        # Force CPU");
    println!("  export RUST_NN_BACKEND=accelerate # Force Apple Accelerate (AMX)");
    println!("  export RUST_NN_BACKEND=metal      # Force Metal (GPU)");
    println!("  export RUST_NN_BACKEND=auto       # Automatic selection (default)");

    println!("\nTo debug backend selection:");
    println!("  export RUST_NN_DEBUG_BACKEND=1");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_availability() {
        // CPU should always be available
        assert!(Backend::Cpu.is_available());

        // Auto should always be available
        assert!(Backend::Auto.is_available());

        // Platform-specific backends
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        assert!(Backend::Accelerate.is_available());

        #[cfg(all(target_os = "macos", feature = "metal"))]
        assert!(Backend::Metal.is_available());
    }

    #[test]
    fn test_backend_from_string() {
        assert_eq!(Backend::from_str("cpu"), Ok(Backend::Cpu));
        assert_eq!(Backend::from_str("CPU"), Ok(Backend::Cpu));
        assert_eq!(Backend::from_str("accelerate"), Ok(Backend::Accelerate));
        assert_eq!(Backend::from_str("AMX"), Ok(Backend::Accelerate));
        assert_eq!(Backend::from_str("metal"), Ok(Backend::Metal));
        assert_eq!(Backend::from_str("GPU"), Ok(Backend::Metal));
        assert_eq!(Backend::from_str("auto"), Ok(Backend::Auto));
        assert_eq!(Backend::from_str("invalid"), Err(ParseBackendError));
    }

    #[test]
    fn test_matmul_with_different_backends() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = vec![2, 3];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_shape = vec![3, 2];

        let expected = vec![22.0, 28.0, 49.0, 64.0];

        // Test CPU backend
        let config = MatmulConfig::cpu_only();
        let result = matmul_with_config(&a, &a_shape, &b, &b_shape, &config);
        assert_eq!(result, expected);

        // Test Accelerate if available
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        {
            let config = MatmulConfig::prefer_accelerate();
            let result = matmul_with_config(&a, &a_shape, &b, &b_shape, &config);
            for (r, e) in result.iter().zip(expected.iter()) {
                assert!((r - e).abs() < 1e-5);
            }
        }

        // Test Metal if available
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let config = MatmulConfig::prefer_metal();
            let result = matmul_with_config(&a, &a_shape, &b, &b_shape, &config);
            for (r, e) in result.iter().zip(expected.iter()) {
                assert!((r - e).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_auto_backend_selection() {
        // Test that Auto backend selects appropriately
        let config = MatmulConfig::default();

        // Small matrix should use CPU
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let result = matmul_with_config(&a, &[2, 2], &b, &[2, 2], &config);
        assert_eq!(result.len(), 4);
    }
}
