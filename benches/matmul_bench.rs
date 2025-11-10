use rust_nn::tensor::Tensor;
use std::time::Instant;

#[derive(Clone)]
enum GemmVariant {
    Naive,
    NaiveAccum,
    CacheAware,
    Tiling1D,
    Tiling,
    Metal,
}

impl GemmVariant {
    fn name(&self) -> &'static str {
        match self {
            GemmVariant::Naive => "Naive",
            GemmVariant::NaiveAccum => "NaiveAccum",
            GemmVariant::CacheAware => "CacheAware",
            GemmVariant::Tiling1D => "Tiling1D",
            GemmVariant::Tiling => "Tiling",
            GemmVariant::Metal => "Metal",
        }
    }
}

struct BenchmarkResult {
    variant: String,
    latency_ms: f64,
    speedup: f64,
}

// Type alias for kernel functions
type GemmKernel = fn(&[f32], &[f32], &mut [f32], usize, usize, usize);

fn gemm_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn gemm_naive_accum(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += sum;
        }
    }
}

fn gemm_cache_aware(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}

fn gemm_tiling_1d(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let tile_size = 64;

    // iterate over k in tiles
    for kk in (0..k).step_by(tile_size) {
        let k_end = usize::min(kk + tile_size, k);

        // standard matmul for the tile
        for i in 0..m {
            for p in kk..k_end {
                for j in 0..n {
                    c[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
    }
}

fn gemm_tiling(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let tile_m = 64;
    let tile_n = 64;
    let tile_k = 64;

    for ii in (0..m).step_by(tile_m) {
        let i_end = usize::min(ii + tile_m, m);
        for jj in (0..n).step_by(tile_n) {
            let j_end = usize::min(jj + tile_n, n);
            for kk in (0..k).step_by(tile_k) {
                let k_end = usize::min(kk + tile_k, k);

                // Compute the tile with cache-aware order
                for i in ii..i_end {
                    for p in kk..k_end {
                        let a_val = a[i * k + p];
                        for j in jj..j_end {
                            c[i * n + j] += a_val * b[p * n + j];
                        }
                    }
                }
            }
        }
    }
}

fn matmul(a: &Tensor, b: &Tensor, variant: &GemmVariant) -> Tensor {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    assert_eq!(
        b.shape[0], k,
        "Incompatible matrix shapes for multiplication"
    );

    let mut c = Tensor::zeros(vec![m, n]);

    match variant {
        GemmVariant::Metal => {
            // Use Metal implementation
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                use rust_nn::tensor::kernels::metal_gemm;
                let result = metal_gemm::matmul(&a.data, &a.shape, &b.data, &b.shape);
                c.data = result;
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                panic!("Metal variant not available on this platform");
            }
        }
        _ => {
            let kernel: GemmKernel = match variant {
                GemmVariant::Naive => gemm_naive,
                GemmVariant::NaiveAccum => gemm_naive_accum,
                GemmVariant::CacheAware => gemm_cache_aware,
                GemmVariant::Tiling1D => gemm_tiling_1d,
                GemmVariant::Tiling => gemm_tiling,
                GemmVariant::Metal => unreachable!(),
            };

            kernel(&a.data, &b.data, &mut c.data, m, k, n);
        }
    }

    c
}

fn benchmark_variant(a: &Tensor, b: &Tensor, variant: &GemmVariant, iterations: usize) -> f64 {
    // Warm-up
    for _ in 0..3 {
        let _ = matmul(a, b, variant);
    }

    // Actual benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = matmul(a, b, variant);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn benchmark_shape(
    m: usize,
    k: usize,
    n: usize,
    iterations: usize,
    variants: &[GemmVariant],
) -> Vec<BenchmarkResult> {
    let a = Tensor::randn(vec![m, k]);
    let b = Tensor::randn(vec![k, n]);

    let mut results = Vec::new();

    for variant in variants {
        let latency_ms = benchmark_variant(&a, &b, variant, iterations);
        results.push(BenchmarkResult {
            variant: variant.name().to_string(),
            latency_ms,
            speedup: 0.0, // Will be calculated later
        });
    }

    // Calculate speedup relative to first variant (baseline)
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
    println!("{}", "=".repeat(70));
    println!(
        "{:<15} {:>15} {:>15} {:>15}",
        "Variant", "Latency (ms)", "Speedup", "GFLOPS"
    );
    println!("{}", "-".repeat(70));

    let ops = 2.0 * m as f64 * k as f64 * n as f64;

    for result in results {
        let gflops = ops / (result.latency_ms / 1000.0) / 1e9;
        println!(
            "{:<15} {:>15.3} {:>15.2}x {:>15.2}",
            result.variant, result.latency_ms, result.speedup, gflops
        );
    }
}

fn main() {
    println!("Matrix Multiplication Benchmarks");
    println!("=================================\n");

    // Define CPU variants to benchmark
    let cpu_variants = vec![
        GemmVariant::Naive,
        GemmVariant::NaiveAccum,
        GemmVariant::CacheAware,
        GemmVariant::Tiling1D,
        GemmVariant::Tiling,
    ];

    // Add Metal variant if available
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let all_variants = {
        let mut variants = cpu_variants.clone();
        variants.push(GemmVariant::Metal);
        variants
    };
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    let all_variants = cpu_variants.clone();

    // Define benchmark configurations: (name, m, k, n, iterations, use_metal)
    let benchmarks = vec![
        ("Small Square", 32, 32, 32, 100, false),
        ("Medium Square", 256, 256, 256, 50, false),
        ("Large Square", 1024, 1024, 1024, 10, true),
        ("MNIST Batch (32)", 32, 784, 128, 50, false),
        ("MNIST Batch (64)", 64, 784, 256, 50, false),
        ("CIFAR-10 Batch", 128, 3072, 512, 20, true),
        ("Tall Matrix", 1000, 100, 10, 50, false),
        ("Wide Matrix", 10, 100, 1000, 50, false),
        ("Very Large Square", 2048, 2048, 2048, 3, true),
    ];

    for (name, m, k, n, iterations, use_metal) in benchmarks {
        let variants = if use_metal {
            &all_variants
        } else {
            &cpu_variants
        };
        let results = benchmark_shape(m, k, n, iterations, variants);
        print_results_table(name, m, k, n, &results);
    }

    println!("\n{}", "=".repeat(70));
    println!("Benchmark complete!");
}
