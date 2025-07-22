# Image Max Polling with SIMD Acceleration

A high-performance Rust library for maximum pooling operations on images, leveraging SIMD instructions (AVX2/NEON) and parallel processing for accelerated performance.

## Features

- **SIMD Optimization**: Utilizes AVX2 (x86-64) or NEON (ARM) intrinsics for vectorized processing.
- **Parallel Execution**: Multi-threaded processing via Rayon for scalable performance.
- **Dynamic CPU Detection**: Runtime checks for AVX2 support (x86-64 only).
- **Image Compatibility**: Works seamlessly with the `image` crate for input/output.

## Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
image-max-polling = "0.1"
```

## Usage

### Basic Example
Process an image with 8x8 max pooling:
```rust
use image_max_polling::{max_pooling_simd, supports_avx2};
use image::{GrayImage, ImageReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("AVX2 supported: {}", supports_avx2());

    let img = ImageReader::open("input.png")?.decode()?.to_luma8();
    let pooled = max_pooling_simd(img.as_bytes(), img.width() as usize, 8);

    GrayImage::from_vec(
        img.width() / 8,
        img.height() / 8,
        pooled,
    )?.save("output.png")?;

    Ok(())
}
```

### Advanced: Custom Pooling Factor
```rust
let pooled_data = max_pooling_simd(&pixels, width, pooling_factor);
```



## Requirements

- **Rust**: 1.60+ (2024 edition)
- **CPU**:
  - x86-64 with AVX2 **OR**
  - ARMv8 with NEON
- **OS**: Linux/macOS/Windows

## Testing

Run examples with:
```sh
cargo run --example max_pooling
```

## License

MIT License - See [LICENSE](LICENSE) for details.
