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
