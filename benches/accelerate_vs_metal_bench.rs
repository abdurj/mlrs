/*
Apple Accelerate (AMX) vs Metal GPU Benchmark
==============================================

This benchmark compares the performance of different GEMM backends on macOS:
- CPU (Naive): Pure Rust implementation
- Accelerate (AMX): Apple's optimized BLAS using AMX instructions on Apple Silicon
- Metal (GPU): Metal Performance Shaders using the GPU

The Accelerate framework leverages Apple's Matrix coprocessor (AMX) which provides
exceptional matrix multiplication performance on M1/M2/M3 chips.
*/

use rust_nn::tensor::Tensor;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
enum Backend {
    CpuNaive,
    Accelerate,
    Metal,
}

impl Backend {
    fn name(&self) -> &'static str {
        match self {
            Backend::CpuNaive => "CPU (Naive)",
            Backend::Accelerate => "Accelerate (AMX)",
            Backend::Metal => "Metal (GPU)",
        }
    }
}

struct BenchmarkResult {
    backend: String,
    latency_ms: f64,
    speedup: f64,
}

fn matmul_cpu_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

fn matmul(a: &Tensor, b: &Tensor, backend: Backend) -> Tensor {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    assert_eq!(
        b.shape[0], k,
        "Incompatible matrix shapes for multiplication"
    );

    match backend {
        Backend::CpuNaive => {
            let mut result = Tensor::zeros(vec![m, n]);
            result.data = matmul_cpu_naive(&a.data, &b.data, m, k, n);
            result
        }
        Backend::Accelerate => {
            #[cfg(all(target_os = "macos", feature = "accelerate"))]
            {
                use rust_nn::tensor::kernels::amx_gemm;
                let mut result = Tensor::zeros(vec![m, n]);
                result.data = amx_gemm::matmul(&a.data, &a.shape, &b.data, &b.shape);
                result
            }
            #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
            {
                panic!("Accelerate backend not available on this platform");
            }
        }
        Backend::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                use rust_nn::tensor::kernels::metal_gemm;
                let mut result = Tensor::zeros(vec![m, n]);
                result.data = metal_gemm::matmul(&a.data, &a.shape, &b.data, &b.shape);
                result
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                panic!("Metal backend not available on this platform");
            }
        }
    }
}

fn benchmark_backend(a: &Tensor, b: &Tensor, backend: Backend, iterations: usize) -> f64 {
    // Warm-up (important for GPU to allocate resources)
    for _ in 0..3 {
        let _ = matmul(a, b, backend);
    }

    // Actual benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = matmul(a, b, backend);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn benchmark_shape(
    m: usize,
    k: usize,
    n: usize,
    iterations: usize,
    backends: &[Backend],
) -> Vec<BenchmarkResult> {
    println!("  Generating random matrices ({} x {} x {})...", m, k, n);
    let a = Tensor::randn(vec![m, k]);
    let b = Tensor::randn(vec![k, n]);

    let mut results = Vec::new();

    for &backend in backends {
        print!("  Benchmarking {}... ", backend.name());
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let latency_ms = benchmark_backend(&a, &b, backend, iterations);
        println!("done ({:.2} ms)", latency_ms);
        results.push(BenchmarkResult {
            backend: backend.name().to_string(),
            latency_ms,
            speedup: 0.0, // Will be calculated later
        });
    }

    // Calculate speedup relative to first backend (baseline)
    let baseline_latency = results[0].latency_ms;
    for result in &mut results {
        result.speedup = baseline_latency / result.latency_ms;
    }

    results
}

fn print_results_table(
    shape_name: &str,
    m: usize,
    k: usize,
    n: usize,
    results: &[BenchmarkResult],
) {
    println!("\n{} ({}x{}x{})", shape_name, m, k, n);
    println!("{}", "=".repeat(80));
    println!(
        "{:<20} {:>15} {:>15} {:>15}",
        "Backend", "Latency (ms)", "Speedup", "GFLOPS"
    );
    println!("{}", "-".repeat(80));

    let ops = 2.0 * m as f64 * k as f64 * n as f64;

    for result in results {
        let gflops = ops / (result.latency_ms / 1000.0) / 1e9;
        println!(
            "{:<20} {:>15.3} {:>15.2}x {:>15.2}",
            result.backend, result.latency_ms, result.speedup, gflops
        );
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║         Apple Accelerate (AMX) vs Metal GPU Benchmark                     ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    // Check available backends
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    {
        use rust_nn::tensor::kernels::amx_gemm;
        println!("✓ Accelerate (AMX) backend: Available");
        if amx_gemm::has_accelerate_support() {
            println!("  → Apple AMX acceleration enabled");
        }
    }
    #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
    {
        println!("✗ Accelerate (AMX) backend: Not available");
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use rust_nn::tensor::kernels::metal_gemm;
        println!("✓ Metal (GPU) backend: Available");
        if metal_gemm::has_metal_support() {
            println!("  → Metal GPU acceleration enabled");
        }
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        println!("✗ Metal (GPU) backend: Not available");
    }

    println!("\n{}", "─".repeat(80));

    // Define backends to benchmark
    let mut backends = vec![Backend::CpuNaive];

    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    backends.push(Backend::Accelerate);

    #[cfg(all(target_os = "macos", feature = "metal"))]
    backends.push(Backend::Metal);

    if backends.len() == 1 {
        println!("\n⚠️  WARNING: Only CPU backend available!");
        println!("To enable all backends, run:");
        println!("  cargo bench --bench accelerate_vs_metal_bench --features metal,accelerate");
        println!();
    }

    // Benchmark configurations: (name, m, k, n, iterations)
    let benchmarks = vec![
        // Small matrices (where overhead matters)
        ("Small Square", 64, 64, 64, 100),
        ("Small Batch", 32, 256, 256, 50),
        // Medium matrices
        ("Medium Square", 256, 256, 256, 50),
        ("Medium Wide", 128, 512, 256, 30),
        // Large matrices (where acceleration really shines)
        ("Large Square", 512, 512, 512, 20),
        ("Large Wide", 256, 1024, 512, 15),
        ("Very Large Square", 1024, 1024, 1024, 10),
        ("Huge Square", 2048, 2048, 2048, 5),
        // Deep learning typical shapes
        ("DL Forward Pass", 128, 4096, 1024, 10),
        ("DL Backward Pass", 4096, 128, 1024, 10),
        ("DL Weight Grad", 4096, 4096, 128, 10),
        // Extreme sizes for stress testing
        ("Stress Test", 4096, 4096, 4096, 3),
    ];

    for (name, m, k, n, iterations) in benchmarks {
        println!("\n{}", "─".repeat(80));
        println!("Running: {}", name);
        let results = benchmark_shape(m, k, n, iterations, &backends);
        print_results_table(name, m, k, n, &results);
    }

    println!("\n{}", "═".repeat(80));
    println!("Benchmark complete!");

    // Summary
    println!("\n📊 Summary:");
    println!("  • CPU backend: Always available, good for small matrices");

    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    println!("  • Accelerate (AMX): Excellent for all sizes, low overhead");

    #[cfg(all(target_os = "macos", feature = "metal"))]
    println!("  • Metal (GPU): Best for very large matrices, has launch overhead");

    println!("\n💡 Recommendations:");
    println!("  • Small matrices (< 128x128): Use Accelerate or CPU");
    println!("  • Medium matrices (128-1024): Use Accelerate (best balance)");
    println!("  • Large matrices (> 1024x1024): Both Accelerate and Metal are excellent");
    println!("  • Very large matrices (> 2048x2048): Metal may have slight edge");
}
