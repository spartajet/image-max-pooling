use std::time::Instant;

use anyhow::Result;
use image::{EncodableLayout, GrayImage, ImageReader};
use image_max_polling::{max_pooling_simd, supports_avx2};

fn main() -> Result<()> {
    let support_avx2 = supports_avx2();
    println!("AVX2 supported: {support_avx2}");

    let image = ImageReader::open("test_image/surface.jpeg")?
        .decode()?
        .to_luma8();
    println!("read image ok");
    let image_width = image.width();
    let image_height = image.height();

    let image_data = image.as_bytes();
    let factor = 8;

    let start = Instant::now();
    let (new_width, new_height, output) =
        max_pooling_simd(image_data, image_width as usize, factor);

    let elapse = start.elapsed();

    println!("time elapsed:{elapse:?}");

    let result_image = GrayImage::from_vec(new_width as u32, new_height as u32, output).unwrap();

    result_image.save("test_image/result.png")?;

    Ok(())
}
