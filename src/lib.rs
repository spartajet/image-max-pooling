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

/// SIMD 加速的最大值池化
pub fn max_pooling_simd(input: &[u8], width: usize, factor: usize) -> (usize, usize, Vec<u8>) {
    let output_width = width / factor;
    let output_height = input.len() / (width * factor);
    let mut output = vec![0; output_width * output_height];

    // 分块并行处理
    output
        .par_chunks_mut(output_width)
        .enumerate()
        .for_each(|(oy, row)| {
            let start_y = oy * factor;
            let end_y = start_y + factor;

            (0..output_width).for_each(|ox| {
                let start_x = ox * factor;
                // let end_x = start_x + factor;

                // 使用 SIMD 计算局部最大值
                let mut simd_max = unsafe { _mm256_setzero_si256() };
                for y in start_y..end_y {
                    let row_start = y * width + start_x;
                    let chunk = &input[row_start..row_start + factor];

                    // 每 32 字节一组处理
                    for chunk32 in chunk.chunks_exact(32) {
                        let data =
                            unsafe { _mm256_loadu_si256(chunk32.as_ptr() as *const __m256i) };
                        simd_max = unsafe { _mm256_max_epu8(simd_max, data) };
                    }

                    // 处理剩余不足 32 字节的部分
                    let remainder = chunk.chunks_exact(32).remainder();
                    if !remainder.is_empty() {
                        let mut buffer = [0u8; 32];
                        buffer[..remainder.len()].copy_from_slice(remainder);
                        let data = unsafe { _mm256_loadu_si256(buffer.as_ptr() as *const __m256i) };
                        simd_max = unsafe { _mm256_max_epu8(simd_max, data) };
                    }
                }

                // 提取 SIMD 结果中的最大值
                let mut max_val = 0;
                let max_arr: &[u8; 32] = unsafe { mem::transmute(&simd_max) };
                for &val in max_arr {
                    if val > max_val {
                        max_val = val;
                    }
                }
                row[ox] = max_val;
            });
        });
    (output_width, output_height, output)
}
