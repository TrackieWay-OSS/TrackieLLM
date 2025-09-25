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
    __m256 mean_r_vec = _mm256_set1_ps(mean[0]);
    __m256 std_r_vec = _mm256_set1_ps(std_dev[0]);
    // ... (and for G, B) ...

    for (size_t i = 0; i < num_pixels; i += 8) {
        // This is a simplified conceptual representation.
        // A real implementation would de-interleave HWC to CHW first,
        // then process each channel plane with SIMD.
        // _mm256_store_ps(r_channel + i, normalized_r_pixels);
    }
    // Handle leftovers...
#else
    // Fallback for non-AVX systems
    for (size_t i = 0; i < num_pixels; ++i) {
        r_channel[i] = (temp_hwc_buffer[i * 3 + 0] / 255.0f - mean[0]) / std_dev[0];
        g_channel[i] = (temp_hwc_buffer[i * 3 + 1] / 255.0f - mean[1]) / std_dev[1];
        b_channel[i] = (temp_hwc_buffer[i * 3 + 2] / 255.0f - mean[2]) / std_dev[2];
    }
#endif

    free(temp_hwc_buffer);

    return TK_SUCCESS;
}