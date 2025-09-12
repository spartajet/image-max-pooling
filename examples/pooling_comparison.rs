use std::time::Instant;

use anyhow::Result;
use image::{EncodableLayout, GrayImage, ImageReader};
use image_max_polling::{max_pooling_simd, min_pooling_simd, supports_avx2};

fn main() -> Result<()> {
    let support_avx2 = supports_avx2();
    println!("AVX2 supported: {support_avx2}");

    let image = ImageReader::open("test_image/surface.jpeg")?
        .decode()?
        .to_luma8();
    println!("read image ok");
    let image_width = image.width();
    let image_height = image.height();
    println!("Original image size: {}x{}", image_width, image_height);

    let image_data = image.as_bytes();
    let factor = 8;

    // Max pooling
    let start = Instant::now();
    let (max_width, max_height, max_output) =
        max_pooling_simd(image_data, image_width as usize, factor);
    let max_elapsed = start.elapsed();
    println!("Max pooling time elapsed: {:?}", max_elapsed);
    println!("Max pooling output size: {}x{}", max_width, max_height);

    // Min pooling
    let start = Instant::now();
    let (min_width, min_height, min_output) =
        min_pooling_simd(image_data, image_width as usize, factor);
    let min_elapsed = start.elapsed();
    println!("Min pooling time elapsed: {:?}", min_elapsed);
    println!("Min pooling output size: {}x{}", min_width, min_height);

    // Save results
    let max_result_image =
        GrayImage::from_vec(max_width as u32, max_height as u32, max_output).unwrap();
    max_result_image.save("test_image/max_pooling_result.png")?;
    println!("Max pooling result saved to: test_image/max_pooling_result.png");

    let min_result_image =
        GrayImage::from_vec(min_width as u32, min_height as u32, min_output).unwrap();
    min_result_image.save("test_image/min_pooling_result.png")?;
    println!("Min pooling result saved to: test_image/min_pooling_result.png");

    // Performance comparison
    println!("\n=== Performance Comparison ===");
    println!("Max pooling: {:?}", max_elapsed);
    println!("Min pooling: {:?}", min_elapsed);

    if max_elapsed > min_elapsed {
        let ratio = max_elapsed.as_nanos() as f64 / min_elapsed.as_nanos() as f64;
        println!("Min pooling is {:.2}x faster than max pooling", ratio);
    } else {
        let ratio = min_elapsed.as_nanos() as f64 / max_elapsed.as_nanos() as f64;
        println!("Max pooling is {:.2}x faster than min pooling", ratio);
    }

    Ok(())
}
