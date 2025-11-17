/*
Metal vs CPU Matrix Multiplication Benchmarks
==============================================


Running: Large Square (1024x1024x1024)
  Generating random matrices...
  Benchmarking CPU Cache-Aware... done
  Benchmarking CPU Tiled... done
  Benchmarking Metal GPU... done

Large Square (1024x1024x1024)
================================================================================
Variant                 Latency (ms)         Speedup          GFLOPS
--------------------------------------------------------------------------------
CPU Cache-Aware              147.411            1.00x           14.57
CPU Tiled                    359.215            0.41x            5.98
Metal GPU                      5.683           25.94x          377.88

Running: Very Large Square (2048x2048x2048)
  Generating random matrices...
  Benchmarking CPU Cache-Aware... done
  Benchmarking CPU Tiled... done
  Benchmarking Metal GPU... done

Very Large Square (2048x2048x2048)
================================================================================
Variant                 Latency (ms)         Speedup          GFLOPS
--------------------------------------------------------------------------------
CPU Cache-Aware             1207.284            1.00x           14.23
CPU Tiled                   2838.431            0.43x            6.05
Metal GPU                     17.111           70.56x         1004.05

Running: Huge Square (4096x4096x4096)
  Generating random matrices...
  Benchmarking CPU Cache-Aware... done
  Benchmarking CPU Tiled... done
  Benchmarking Metal GPU... done

Huge Square (4096x4096x4096)
================================================================================
Variant                 Latency (ms)         Speedup          GFLOPS
--------------------------------------------------------------------------------
CPU Cache-Aware            11204.396            1.00x           12.27
CPU Tiled                  24154.752            0.46x            5.69
Metal GPU                    211.020           53.10x          651.31

Running: Large Batch GEMM (256x4096x4096)
  Generating random matrices...
  Benchmarking CPU Cache-Aware... done
  Benchmarking CPU Tiled... done
  Benchmarking Metal GPU... done

Large Batch GEMM (256x4096x4096)
================================================================================
Variant                 Latency (ms)         Speedup          GFLOPS
--------------------------------------------------------------------------------
CPU Cache-Aware              731.814            1.00x           11.74
CPU Tiled                   1480.309            0.49x            5.80
Metal GPU                     10.876           67.29x          789.83

Running: Wide Matrix (512x8192x512)
  Generating random matrices...
  Benchmarking CPU Cache-Aware... done
  Benchmarking CPU Tiled... done
  Benchmarking Metal GPU... done

Wide Matrix (512x8192x512)
================================================================================
Variant                 Latency (ms)         Speedup          GFLOPS
--------------------------------------------------------------------------------
CPU Cache-Aware              366.708            1.00x           11.71
CPU Tiled                    681.671            0.54x            6.30
Metal GPU                      7.186           51.03x          597.72

Running: Tall Matrix (8192x512x512)
  Generating random matrices...
  Benchmarking CPU Cache-Aware... done
  Benchmarking CPU Tiled... done
  Benchmarking Metal GPU... done

Tall Matrix (8192x512x512)
================================================================================
Variant                 Latency (ms)         Speedup          GFLOPS
--------------------------------------------------------------------------------
CPU Cache-Aware              316.134            1.00x           13.59
CPU Tiled                    709.828            0.45x            6.05
Metal GPU                     14.588           21.67x          294.42

Running: Deep Learning (Forward) (128x4096x1024)
  Generating random matrices...
  Benchmarking CPU Cache-Aware... done
  Benchmarking CPU Tiled... done
  Benchmarking Metal GPU... done

Deep Learning (Forward) (128x4096x1024)
================================================================================
Variant                 Latency (ms)         Speedup          GFLOPS
--------------------------------------------------------------------------------
CPU Cache-Aware               79.929            1.00x           13.43
CPU Tiled                    170.594            0.47x            6.29
Metal GPU                      3.171           25.20x          338.57

Running: Deep Learning (Backward) (4096x128x1024)
  Generating random matrices...
  Benchmarking CPU Cache-Aware... done
  Benchmarking CPU Tiled... done
  Benchmarking Metal GPU... done

Deep Learning (Backward) (4096x128x1024)
================================================================================
Variant                 Latency (ms)         Speedup          GFLOPS
--------------------------------------------------------------------------------
CPU Cache-Aware               76.040            1.00x           14.12
CPU Tiled                    168.552            0.45x            6.37
Metal GPU                      5.847           13.00x          183.64

*/

use rust_nn::tensor::Tensor;
use std::time::Instant;

#[derive(Clone)]
enum GemmVariant {
    CacheAware,
    Tiling,
    Metal,
}

impl GemmVariant {
    fn name(&self) -> &'static str {
        match self {
            GemmVariant::CacheAware => "CPU Cache-Aware",
            GemmVariant::Tiling => "CPU Tiled",
            GemmVariant::Metal => "Metal GPU",
        }
    }
}

struct BenchmarkResult {
    variant: String,
    latency_ms: f64,
    speedup: f64,
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
        GemmVariant::CacheAware => {
            gemm_cache_aware(&a.data, &b.data, &mut c.data, m, k, n);
        }
        GemmVariant::Tiling => {
            gemm_tiling(&a.data, &b.data, &mut c.data, m, k, n);
        }
    }

    c
}

fn benchmark_variant(a: &Tensor, b: &Tensor, variant: &GemmVariant, iterations: usize) -> f64 {
    // Warm-up
    for _ in 0..2 {
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
    println!("  Generating random matrices...");
    let a = Tensor::randn(vec![m, k]);
    let b = Tensor::randn(vec![k, n]);

    let mut results = Vec::new();

    for variant in variants {
        print!("  Benchmarking {}... ", variant.name());
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let latency_ms = benchmark_variant(&a, &b, variant, iterations);
        println!("done");
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
    println!("{}", "=".repeat(80));
    println!(
        "{:<20} {:>15} {:>15} {:>15}",
        "Variant", "Latency (ms)", "Speedup", "GFLOPS"
    );
    println!("{}", "-".repeat(80));

    let ops = 2.0 * m as f64 * k as f64 * n as f64;

    for result in results {
        let gflops = ops / (result.latency_ms / 1000.0) / 1e9;
        println!(
            "{:<20} {:>15.3} {:>15.2}x {:>15.2}",
            result.variant, result.latency_ms, result.speedup, gflops
        );
    }
}

fn main() {
    println!("Metal vs CPU Matrix Multiplication Benchmarks");
    println!("==============================================\n");

    // Define variants to benchmark
    let variants = vec![
        GemmVariant::CacheAware,
        GemmVariant::Tiling,
        #[cfg(all(target_os = "macos", feature = "metal"))]
        GemmVariant::Metal,
    ];

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        println!("WARNING: Metal support not enabled. Only CPU variants will be tested.");
        println!("Enable with: cargo run --release --features metal --bin metal_bench\n");
    }

    // Define benchmark configurations: (name, m, k, n, iterations)
    // Focus on larger matrices where GPU acceleration matters
    let benchmarks = vec![
        ("Large Square", 1024, 1024, 1024, 10),
        ("Very Large Square", 2048, 2048, 2048, 5),
        ("Huge Square", 4096, 4096, 4096, 3),
        ("Large Batch GEMM", 256, 4096, 4096, 5),
        ("Wide Matrix", 512, 8192, 512, 5),
        ("Tall Matrix", 8192, 512, 512, 5),
        ("Deep Learning (Forward)", 128, 4096, 1024, 10),
        ("Deep Learning (Backward)", 4096, 128, 1024, 10),
    ];

    for (name, m, k, n, iterations) in benchmarks {
        println!("\nRunning: {} ({}x{}x{})", name, m, k, n);
        let results = benchmark_shape(m, k, n, iterations, &variants);
        print_results_table(name, m, k, n, &results);
    }

    println!("\n{}", "=".repeat(80));
    println!("Benchmark complete!");

    #[cfg(all(target_os = "macos", feature = "metal"))]
    println!("\nNote: Metal GPU acceleration was used for large matrix operations.");
}
