# Image Max & Min Pooling with SIMD Acceleration

A high-performance Rust library for maximum and minimum pooling operations on images, leveraging SIMD instructions (AVX2/NEON) and parallel processing for accelerated performance.

## Features

- **SIMD Optimization**: Utilizes AVX2 (x86-64) or NEON (ARM) intrinsics for vectorized processing
- **Dual Pooling Operations**: Supports both maximum and minimum pooling
- **Parallel Execution**: Multi-threaded processing via Rayon for scalable performance
- **Dynamic CPU Detection**: Runtime checks for AVX2 support (x86-64 only)
- **Image Compatibility**: Works seamlessly with the `image` crate for input/output

## Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
image-max-polling = "0.1"
```

## Usage

### Maximum Pooling Example
Process an image with 8x8 max pooling:
```rust
use image_max_polling::{max_pooling_simd, supports_avx2};
use image::{GrayImage, ImageReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("AVX2 supported: {}", supports_avx2());

    let img = ImageReader::open("input.png")?.decode()?.to_luma8();
    let (new_width, new_height, pooled_data) = max_pooling_simd(
        img.as_bytes(), 
        img.width() as usize, 
        8
    );

    let result = GrayImage::from_vec(
        new_width as u32,
        new_height as u32,
        pooled_data,
    ).unwrap();
    
    result.save("max_output.png")?;
    Ok(())
}
```

### Minimum Pooling Example
Process an image with 8x8 min pooling:
```rust
use image_max_polling::{min_pooling_simd, supports_avx2};
use image::{GrayImage, ImageReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("AVX2 supported: {}", supports_avx2());

    let img = ImageReader::open("input.png")?.decode()?.to_luma8();
    let (new_width, new_height, pooled_data) = min_pooling_simd(
        img.as_bytes(), 
        img.width() as usize, 
        8
    );

    let result = GrayImage::from_vec(
        new_width as u32,
        new_height as u32,
        pooled_data,
    ).unwrap();
    
    result.save("min_output.png")?;
    Ok(())
}
```

### Pooling Comparison
Compare max and min pooling results:
```rust
use image_max_polling::{max_pooling_simd, min_pooling_simd};

let image_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
let width = 3;
let factor = 3;

// Maximum pooling - extracts brightest features
let (_, _, max_result) = max_pooling_simd(&image_data, width, factor);
println!("Max pooling result: {:?}", max_result); // [9]

// Minimum pooling - extracts darkest features  
let (_, _, min_result) = min_pooling_simd(&image_data, width, factor);
println!("Min pooling result: {:?}", min_result); // [1]
```

## API Reference

### Functions

#### `max_pooling_simd(input: &[u8], width: usize, factor: usize) -> (usize, usize, Vec<u8>)`
Performs maximum pooling operation using SIMD acceleration.
- **Parameters**:
  - `input`: Input image data as a flat byte array (row-major order)
  - `width`: Width of the input image in pixels
  - `factor`: Pooling factor (e.g., 2 for 2x2 pooling)
- **Returns**: Tuple of (output_width, output_height, output_data)

#### `min_pooling_simd(input: &[u8], width: usize, factor: usize) -> (usize, usize, Vec<u8>)`
Performs minimum pooling operation using SIMD acceleration.
- **Parameters**: Same as `max_pooling_simd`
- **Returns**: Same as `max_pooling_simd`

#### `supports_avx2() -> bool`
Checks if the current CPU supports AVX2 instructions.

## Performance

Both pooling operations are optimized with:
- **SIMD Instructions**: Process 32 bytes (256 bits) in parallel
- **Multi-threading**: Parallel row processing using Rayon
- **Cache-friendly**: Efficient memory access patterns

Typical performance on modern CPUs:
- **Max Pooling**: ~50-100ms for 1920x1080 images with 8x8 pooling
- **Min Pooling**: Similar performance to max pooling

## Examples

Run the provided examples:

```bash
# Basic max pooling
cargo run --example max_polling

# Basic min pooling  
cargo run --example min_pooling

# Compare both operations
cargo run --example pooling_comparison
```

## Use Cases

### Maximum Pooling
- **Feature extraction**: Highlights prominent features
- **Noise reduction**: Preserves strong signals
- **Downsampling**: Maintains important visual information
- **CNN preprocessing**: Standard in deep learning pipelines

### Minimum Pooling
- **Edge detection**: Emphasizes dark boundaries
- **Background extraction**: Identifies darker regions
- **Texture analysis**: Highlights fine dark details
- **Medical imaging**: Useful for detecting dark anomalies

## Requirements

- **Rust**: 1.60+ (2024 edition)
- **CPU**:
  - x86-64 with AVX2 **OR**
  - ARMv8 with NEON
- **OS**: Linux/macOS/Windows

## Testing

Run tests with:
```bash
cargo test
```

Run benchmarks with:
```bash
cargo test --release
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.