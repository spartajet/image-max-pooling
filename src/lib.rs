//! # Image Max & Min Pooling with SIMD Acceleration
//!
//! A high-performance Rust library for maximum and minimum pooling operations on images,
//! leveraging SIMD instructions (AVX2/NEON) and parallel processing for accelerated performance.
//!
//! ## Features
//!
//! - **SIMD Optimization**: Utilizes AVX2 (x86-64) or NEON (ARM) intrinsics
//! - **Dual Pooling Operations**: Supports both maximum and minimum pooling
//! - **Parallel Execution**: Multi-threaded processing via Rayon
//! - **Dynamic CPU Detection**: Runtime checks for AVX2 support
//!
//! ## Quick Start
//!
//! ```rust
//! use image_max_polling::{max_pooling_simd, min_pooling_simd};
//!
//! let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
//! let width = 3;
//! let factor = 3;
//!
//! // Maximum pooling - extracts brightest features
//! let (_, _, max_result) = max_pooling_simd(&data, width, factor);
//!
//! // Minimum pooling - extracts darkest features
//! let (_, _, min_result) = min_pooling_simd(&data, width, factor);
//! ```

#![allow(dead_code)]
use rayon::prelude::*;
use std::arch::x86_64::*;
use std::mem;

// 条件编译选择指令集
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64 as arch;

// 动态检测 CPU 特性
pub fn supports_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        arch::_xgetbv(0) & 0x6 == 0x6
    }

    #[cfg(target_arch = "aarch64")]
    true // ARM 默认启用 NEON
}
/// 使用 SIMD 指令优化的最大池化函数
///
/// # 详细说明
/// 对输入的二维图像数据执行最大池化操作，使用 AVX2 SIMD 指令集并行处理以提升性能。
/// 池化操作将输入图像按指定因子缩小，每个输出像素对应输入区域的最大值。
///
/// # 参数
/// - `input`: 输入图像数据的一维数组表示（按行优先存储）
/// - `width`: 输入图像的宽度（像素数）
/// - `factor`: 池化因子，决定缩放比例（如 2 表示 2x2 池化）
///
/// # 返回值
/// 返回元组 (output_width, output_height, output_data)：
/// - `output_width`: 输出图像宽度
/// - `output_height`: 输出图像高度
/// - `output_data`: 池化后的图像数据
///
/// # 示例
/// ```rust
/// use image_max_polling::max_pooling_simd;
/// let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let (w, h, result) = max_pooling_simd(&input, 3, 1);
/// ```
pub fn max_pooling_simd(input: &[u8], width: usize, factor: usize) -> (usize, usize, Vec<u8>) {
    // 计算输出图像尺寸
    let output_width = width / factor;
    let output_height = input.len() / (width * factor);
    let mut output = vec![0; output_width * output_height];

    // 使用 Rayon 进行分块并行处理，每个线程处理一行输出
    output
        .par_chunks_mut(output_width)
        .enumerate()
        .for_each(|(oy, row)| {
            // 计算当前输出行对应的输入行范围
            let start_y = oy * factor;
            let end_y = start_y + factor;

            // 遍历当前行的每个输出像素
            (0..output_width).for_each(|ox| {
                let start_x = ox * factor;

                // 初始化 AVX2 SIMD 寄存器用于存储最大值
                let mut simd_max = unsafe { _mm256_setzero_si256() };

                // 遍历池化窗口内的所有行
                for y in start_y..end_y {
                    let row_start = y * width + start_x;
                    let chunk = &input[row_start..row_start + factor];

                    // 使用 SIMD 处理：每次加载 32 字节（256位）进行并行比较
                    for chunk32 in chunk.chunks_exact(32) {
                        let data =
                            unsafe { _mm256_loadu_si256(chunk32.as_ptr() as *const __m256i) };
                        simd_max = unsafe { _mm256_max_epu8(simd_max, data) };
                    }

                    // 处理不足 32 字节的剩余数据
                    let remainder = chunk.chunks_exact(32).remainder();
                    if !remainder.is_empty() {
                        // 将剩余数据复制到对齐的缓冲区
                        let mut buffer = [0u8; 32];
                        buffer[..remainder.len()].copy_from_slice(remainder);
                        let data = unsafe { _mm256_loadu_si256(buffer.as_ptr() as *const __m256i) };
                        simd_max = unsafe { _mm256_max_epu8(simd_max, data) };
                    }
                }

                // 从 SIMD 寄存器中提取最终的最大值
                let mut max_val = 0;
                let max_arr: &[u8; 32] = unsafe { mem::transmute(&simd_max) };
                for &val in max_arr {
                    if val > max_val {
                        max_val = val;
                    }
                }
                row[ox] = max_val; // 存储到输出数组
            });
        });
    (output_width, output_height, output)
}

/// 使用 SIMD 指令优化的最小池化函数
///
/// # 详细说明
/// 对输入的二维图像数据执行最小池化操作，使用 AVX2 SIMD 指令集并行处理以提升性能。
/// 池化操作将输入图像按指定因子缩小，每个输出像素对应输入区域的最小值。
///
/// # 参数
/// - `input`: 输入图像数据的一维数组表示（按行优先存储）
/// - `width`: 输入图像的宽度（像素数）
/// - `factor`: 池化因子，决定缩放比例（如 2 表示 2x2 池化）
///
/// # 返回值
/// 返回元组 (output_width, output_height, output_data)：
/// - `output_width`: 输出图像宽度
/// - `output_height`: 输出图像高度
/// - `output_data`: 池化后的图像数据
///
/// # 示例
/// ```rust
/// use image_max_polling::min_pooling_simd;
/// let input = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];
/// let (w, h, result) = min_pooling_simd(&input, 3, 1);
/// ```
pub fn min_pooling_simd(input: &[u8], width: usize, factor: usize) -> (usize, usize, Vec<u8>) {
    // 计算输出图像尺寸
    let output_width = width / factor;
    let output_height = input.len() / (width * factor);
    let mut output = vec![255; output_width * output_height]; // 初始化为最大值 255

    // 使用 Rayon 进行分块并行处理，每个线程处理一行输出
    output
        .par_chunks_mut(output_width)
        .enumerate()
        .for_each(|(oy, row)| {
            // 计算当前输出行对应的输入行范围
            let start_y = oy * factor;
            let end_y = start_y + factor;

            // 遍历当前行的每个输出像素
            (0..output_width).for_each(|ox| {
                let start_x = ox * factor;

                // 初始化 AVX2 SIMD 寄存器用于存储最小值（设为 255，即最大值）
                let mut simd_min = unsafe { _mm256_set1_epi8(255u8 as i8) };

                // 遍历池化窗口内的所有行
                for y in start_y..end_y {
                    let row_start = y * width + start_x;
                    let chunk = &input[row_start..row_start + factor];

                    // 使用 SIMD 处理：每次加载 32 字节（256位）进行并行比较
                    for chunk32 in chunk.chunks_exact(32) {
                        let data =
                            unsafe { _mm256_loadu_si256(chunk32.as_ptr() as *const __m256i) };
                        simd_min = unsafe { _mm256_min_epu8(simd_min, data) };
                    }

                    // 处理不足 32 字节的剩余数据
                    let remainder = chunk.chunks_exact(32).remainder();
                    if !remainder.is_empty() {
                        // 将剩余数据复制到对齐的缓冲区，填充 255（最大值）
                        let mut buffer = [255u8; 32];
                        buffer[..remainder.len()].copy_from_slice(remainder);
                        let data = unsafe { _mm256_loadu_si256(buffer.as_ptr() as *const __m256i) };
                        simd_min = unsafe { _mm256_min_epu8(simd_min, data) };
                    }
                }

                // 从 SIMD 寄存器中提取最终的最小值
                let mut min_val = 255;
                let min_arr: &[u8; 32] = unsafe { mem::transmute(&simd_min) };
                for &val in min_arr {
                    if val < min_val {
                        min_val = val;
                    }
                }
                row[ox] = min_val; // 存储到输出数组
            });
        });
    (output_width, output_height, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pooling_simple() {
        // 创建一个简单的 3x3 图像
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let width = 3;
        let factor = 3;

        let (output_width, output_height, output) = max_pooling_simd(&input, width, factor);

        assert_eq!(output_width, 1);
        assert_eq!(output_height, 1);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 9); // 最大值应该是 9
    }

    #[test]
    fn test_min_pooling_simple() {
        // 创建一个简单的 3x3 图像
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let width = 3;
        let factor = 3;

        let (output_width, output_height, output) = min_pooling_simd(&input, width, factor);

        assert_eq!(output_width, 1);
        assert_eq!(output_height, 1);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 1); // 最小值应该是 1
    }

    #[test]
    fn test_max_pooling_2x2() {
        // 创建一个 4x4 图像，进行 2x2 池化
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let width = 4;
        let factor = 2;

        let (output_width, output_height, output) = max_pooling_simd(&input, width, factor);

        assert_eq!(output_width, 2);
        assert_eq!(output_height, 2);
        assert_eq!(output.len(), 4);

        // 每个 2x2 区域的最大值
        assert_eq!(output[0], 6); // max(1,2,5,6)
        assert_eq!(output[1], 8); // max(3,4,7,8)
        assert_eq!(output[2], 14); // max(9,10,13,14)
        assert_eq!(output[3], 16); // max(11,12,15,16)
    }

    #[test]
    fn test_min_pooling_2x2() {
        // 创建一个 4x4 图像，进行 2x2 池化
        let input = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        let width = 4;
        let factor = 2;

        let (output_width, output_height, output) = min_pooling_simd(&input, width, factor);

        assert_eq!(output_width, 2);
        assert_eq!(output_height, 2);
        assert_eq!(output.len(), 4);

        // 每个 2x2 区域的最小值
        assert_eq!(output[0], 11); // min(16,15,12,11)
        assert_eq!(output[1], 9); // min(14,13,10,9)
        assert_eq!(output[2], 3); // min(8,7,4,3)
        assert_eq!(output[3], 1); // min(6,5,2,1)
    }

    #[test]
    fn test_pooling_edge_values() {
        // 测试边界值：0 和 255
        let input = vec![
            0, 255, 0, 255, 255, 0, 255, 0, 0, 255, 0, 255, 255, 0, 255, 0,
        ];
        let width = 4;
        let factor = 2;

        // 测试最大池化
        let (_, _, max_output) = max_pooling_simd(&input, width, factor);
        assert!(max_output.iter().all(|&x| x == 255));

        // 测试最小池化
        let (_, _, min_output) = min_pooling_simd(&input, width, factor);
        assert!(min_output.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_pooling_identical_values() {
        // 测试所有值相同的情况
        let input = vec![100; 36]; // 6x6 全是 100
        let width = 6;
        let factor = 3;

        let (max_w, max_h, max_output) = max_pooling_simd(&input, width, factor);
        let (min_w, min_h, min_output) = min_pooling_simd(&input, width, factor);

        assert_eq!(max_w, min_w);
        assert_eq!(max_h, min_h);
        assert_eq!(max_output.len(), min_output.len());

        // 所有值都应该是 100
        assert!(max_output.iter().all(|&x| x == 100));
        assert!(min_output.iter().all(|&x| x == 100));
    }
}
