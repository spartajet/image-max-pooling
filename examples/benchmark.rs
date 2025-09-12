use std::time::Instant;

use anyhow::Result;
use image_max_polling::{max_pooling_simd, min_pooling_simd, supports_avx2};

fn benchmark_pooling(
    data: &[u8],
    width: usize,
    factor: usize,
    iterations: usize,
) -> (std::time::Duration, std::time::Duration) {
    // Warm up
    let _ = max_pooling_simd(data, width, factor);
    let _ = min_pooling_simd(data, width, factor);

    // Benchmark max pooling
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = max_pooling_simd(data, width, factor);
    }
    let max_elapsed = start.elapsed();

    // Benchmark min pooling
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = min_pooling_simd(data, width, factor);
    }
    let min_elapsed = start.elapsed();

    (max_elapsed, min_elapsed)
}

fn main() -> Result<()> {
    println!("=== Image Pooling Performance Benchmark ===");
    println!("AVX2 supported: {}", supports_avx2());
    println!();

    let iterations = 100;
    let test_cases = vec![
        (512, 512, 2, "512x512, 2x2 pooling"),
        (1024, 1024, 2, "1024x1024, 2x2 pooling"),
        (1920, 1080, 4, "1920x1080, 4x4 pooling"),
        (1920, 1080, 8, "1920x1080, 8x8 pooling"),
        (4096, 4096, 4, "4096x4096, 4x4 pooling"),
    ];

    for (width, height, factor, description) in test_cases {
        // Generate test data
        let data_size = width * height;
        let data: Vec<u8> = (0..data_size).map(|i| ((i * 37) % 256) as u8).collect();

        println!("Testing: {}", description);
        println!(
            "Input size: {}x{} = {:.1} MB",
            width,
            height,
            data_size as f64 / (1024.0 * 1024.0)
        );

        let (max_time, min_time) = benchmark_pooling(&data, width, factor, iterations);

        let max_avg = max_time.as_nanos() as f64 / iterations as f64 / 1_000_000.0;
        let min_avg = min_time.as_nanos() as f64 / iterations as f64 / 1_000_000.0;

        println!("Max pooling avg: {:.2} ms", max_avg);
        println!("Min pooling avg: {:.2} ms", min_avg);

        if max_avg > min_avg {
            let speedup = max_avg / min_avg;
            println!("Min pooling is {:.2}x faster", speedup);
        } else {
            let speedup = min_avg / max_avg;
            println!("Max pooling is {:.2}x faster", speedup);
        }

        // Calculate throughput
        let max_throughput = (data_size as f64 / 1024.0 / 1024.0) / (max_avg / 1000.0);
        let min_throughput = (data_size as f64 / 1024.0 / 1024.0) / (min_avg / 1000.0);

        println!("Max pooling throughput: {:.1} MB/s", max_throughput);
        println!("Min pooling throughput: {:.1} MB/s", min_throughput);
        println!();
    }

    // Test different pooling factors
    println!("=== Pooling Factor Performance Comparison ===");
    let width = 1920;
    let height = 1080;
    let data_size = width * height;
    let data: Vec<u8> = (0..data_size).map(|i| ((i * 37) % 256) as u8).collect();

    let factors = vec![2, 4, 8, 16];

    for factor in factors {
        println!("Pooling factor: {}x{}", factor, factor);
        let (max_time, min_time) = benchmark_pooling(&data, width, factor, 50);

        let max_avg = max_time.as_nanos() as f64 / 50.0 / 1_000_000.0;
        let min_avg = min_time.as_nanos() as f64 / 50.0 / 1_000_000.0;

        println!("  Max pooling: {:.2} ms", max_avg);
        println!("  Min pooling: {:.2} ms", min_avg);

        // Calculate output size
        let output_width = width / factor;
        let output_height = height / factor;
        println!("  Output size: {}x{}", output_width, output_height);
        println!();
    }

    // Memory efficiency test
    println!("=== Memory Efficiency Test ===");
    let test_sizes = vec![(256, 256), (512, 512), (1024, 1024), (2048, 2048)];

    for (width, height) in test_sizes {
        let data_size = width * height;
        let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();

        let factor = 4;
        let iterations = 10;

        let (max_time, min_time) = benchmark_pooling(&data, width, factor, iterations);

        let max_avg = max_time.as_nanos() as f64 / iterations as f64 / 1_000_000.0;
        let min_avg = min_time.as_nanos() as f64 / iterations as f64 / 1_000_000.0;

        let mb_size = data_size as f64 / (1024.0 * 1024.0);

        println!("Size: {}x{} ({:.1} MB)", width, height, mb_size);
        println!(
            "  Max pooling: {:.2} ms ({:.1} MB/s)",
            max_avg,
            mb_size / (max_avg / 1000.0)
        );
        println!(
            "  Min pooling: {:.2} ms ({:.1} MB/s)",
            min_avg,
            mb_size / (min_avg / 1000.0)
        );
    }

    Ok(())
}
