/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_image_preprocessor.c
 *
 * This source file implements the TrackieLLM Image Preprocessor.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_image_preprocessor.h"
#include "cortex/tk_cortex_main.h" // For the full definition of tk_video_frame_t
#include <stddef.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h> // For AVX intrinsics
#elif defined(__ARM_NEON)
#include <arm_neon.h> // For NEON intrinsics
#endif

TK_NODISCARD tk_error_code_t tk_preprocessor_resize_and_normalize_to_chw(
    const tk_video_frame_t* frame,
    float* out_tensor,
    uint32_t target_width,
    uint32_t target_height,
    const float mean[3],
    const float std_dev[3]
) {
    if (!frame || !frame->data || !out_tensor || target_width == 0 || target_height == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t orig_width = frame->width;
    const uint32_t orig_height = frame->height;
    const uint8_t* frame_data = frame->data;

    // --- High-Quality Bilinear Resizing ---
    // First, resize the image into a temporary HWC float buffer.
    size_t num_pixels = target_width * target_height;
    float* temp_hwc_buffer = (float*)malloc(num_pixels * 3 * sizeof(float));
    if (!temp_hwc_buffer) return TK_ERROR_OUT_OF_MEMORY;

    float x_ratio = ((float)orig_width - 1.0f) / target_width;
    float y_ratio = ((float)orig_height - 1.0f) / target_height;

    for (uint32_t h = 0; h < target_height; ++h) {
        for (uint32_t w = 0; w < target_width; ++w) {
            float gx = x_ratio * w;
            float gy = y_ratio * h;
            int x = (int)gx;
            int y = (int)gy;
            float x_diff = gx - x;
            float y_diff = gy - y;

            size_t out_idx = (h * target_width + w) * 3;

            for (uint32_t c = 0; c < 3; ++c) {
                uint8_t p1 = frame_data[(y * orig_width + x) * 3 + c];
                uint8_t p2 = frame_data[(y * orig_width + (x + 1)) * 3 + c];
                uint8_t p3 = frame_data[((y + 1) * orig_width + x) * 3 + c];
                uint8_t p4 = frame_data[((y + 1) * orig_width + (x + 1)) * 3 + c];

                temp_hwc_buffer[out_idx + c] = (float)p1 * (1 - x_diff) * (1 - y_diff) +
                                               (float)p2 * x_diff * (1 - y_diff) +
                                               (float)p3 * (1 - x_diff) * y_diff +
                                               (float)p4 * x_diff * y_diff;
            }
        }
    }

    // --- SIMD-accelerated Normalization and Layout Conversion (HWC to CHW) ---
    float* r_channel = out_tensor;
    float* g_channel = out_tensor + num_pixels;
    float* b_channel = out_tensor + num_pixels * 2;

#if defined(__AVX__)
    // De-interleave HWC to CHW first
    for (size_t i = 0; i < num_pixels; ++i) {
        r_channel[i] = temp_hwc_buffer[i * 3 + 0];
        g_channel[i] = temp_hwc_buffer[i * 3 + 1];
        b_channel[i] = temp_hwc_buffer[i * 3 + 2];
    }

    __m256 scale_vec = _mm256_set1_ps(1.0f / 255.0f);
    __m256 mean_r = _mm256_set1_ps(mean[0]);
    __m256 std_r = _mm256_set1_ps(std_dev[0]);
    __m256 mean_g = _mm256_set1_ps(mean[1]);
    __m256 std_g = _mm256_set1_ps(std_dev[1]);
    __m256 mean_b = _mm256_set1_ps(mean[2]);
    __m256 std_b = _mm256_set1_ps(std_dev[2]);

    for (size_t i = 0; i < num_pixels; i += 8) {
        // Process Red Channel
        __m256 r_vec = _mm256_loadu_ps(r_channel + i);
        r_vec = _mm256_mul_ps(r_vec, scale_vec);
        r_vec = _mm256_sub_ps(r_vec, mean_r);
        r_vec = _mm256_div_ps(r_vec, std_r);
        _mm256_storeu_ps(r_channel + i, r_vec);

        // Process Green Channel
        __m256 g_vec = _mm256_loadu_ps(g_channel + i);
        g_vec = _mm256_mul_ps(g_vec, scale_vec);
        g_vec = _mm256_sub_ps(g_vec, mean_g);
        g_vec = _mm256_div_ps(g_vec, std_g);
        _mm256_storeu_ps(g_channel + i, g_vec);

        // Process Blue Channel
        __m256 b_vec = _mm256_loadu_ps(b_channel + i);
        b_vec = _mm256_mul_ps(b_vec, scale_vec);
        b_vec = _mm256_sub_ps(b_vec, mean_b);
        b_vec = _mm256_div_ps(b_vec, std_b);
        _mm256_storeu_ps(b_channel + i, b_vec);
    }
    // Note: A full implementation should handle leftovers if num_pixels is not a multiple of 8.
#elif defined(__ARM_NEON)
    // De-interleave HWC to CHW first
    for (size_t i = 0; i < num_pixels; ++i) {
        r_channel[i] = temp_hwc_buffer[i * 3 + 0];
        g_channel[i] = temp_hwc_buffer[i * 3 + 1];
        b_channel[i] = temp_hwc_buffer[i * 3 + 2];
    }

    float32x4_t scale_vec = vdupq_n_f32(1.0f / 255.0f);
    float32x4_t mean_r = vld1q_dup_f32(&mean[0]);
    float32x4_t std_r = vld1q_dup_f32(&std_dev[0]);
    float32x4_t mean_g = vld1q_dup_f32(&mean[1]);
    float32x4_t std_g = vld1q_dup_f32(&std_dev[1]);
    float32x4_t mean_b = vld1q_dup_f32(&mean[2]);
    float32x4_t std_b = vld1q_dup_f32(&std_dev[2]);

    for (size_t i = 0; i < num_pixels; i += 4) {
        // Process Red Channel
        float32x4_t r_vec = vld1q_f32(r_channel + i);
        r_vec = vmulq_f32(r_vec, scale_vec);
        r_vec = vsubq_f32(r_vec, mean_r);
        r_vec = vdivq_f32(r_vec, std_r); // Note: vdivq might not be available on all NEON versions.
        vst1q_f32(r_channel + i, r_vec);

        // Process Green Channel
        float32x4_t g_vec = vld1q_f32(g_channel + i);
        g_vec = vmulq_f32(g_vec, scale_vec);
        g_vec = vsubq_f32(g_vec, mean_g);
        g_vec = vdivq_f32(g_vec, std_g);
        vst1q_f32(g_channel + i, g_vec);

        // Process Blue Channel
        float32x4_t b_vec = vld1q_f32(b_channel + i);
        b_vec = vmulq_f32(b_vec, scale_vec);
        b_vec = vsubq_f32(b_vec, mean_b);
        b_vec = vdivq_f32(b_vec, std_b);
        vst1q_f32(b_channel + i, b_vec);
    }
    // Note: A full implementation should handle leftovers if num_pixels is not a multiple of 4.
#else
    // Fallback for non-SIMD systems
    for (size_t i = 0; i < num_pixels; ++i) {
        r_channel[i] = (temp_hwc_buffer[i * 3 + 0] / 255.0f - mean[0]) / std_dev[0];
        g_channel[i] = (temp_hwc_buffer[i * 3 + 1] / 255.0f - mean[1]) / std_dev[1];
        b_channel[i] = (temp_hwc_buffer[i * 3 + 2] / 255.0f - mean[2]) / std_dev[2];
    }
#endif

    free(temp_hwc_buffer);

    return TK_SUCCESS;
}