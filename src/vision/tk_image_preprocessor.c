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

    // TODO: Replace this simple nearest-neighbor resizing with a higher-quality
    // algorithm like bilinear interpolation from a library like stb_image_resize
    // or libyuv for better model accuracy.

    // The output tensor is in CHW format, so we can structure the loops
    // to write directly into it.
    for (uint32_t c = 0; c < 3; ++c) { // Channel (R, G, B)
        for (uint32_t h = 0; h < target_height; ++h) { // Row
            for (uint32_t w = 0; w < target_width; ++w) { // Column
                // Find the corresponding pixel in the original image
                uint32_t orig_x = (w * orig_width) / target_width;
                uint32_t orig_y = (h * orig_height) / target_height;

                // Get the pixel value from the HWC input
                uint8_t val = frame_data[(orig_y * orig_width + orig_x) * 3 + c];

                // Calculate the index in the CHW output tensor
                size_t out_idx = c * (target_height * target_width) + h * target_width + w;

                // Normalize and store the value
                // TODO: This loop is a prime candidate for SIMD (NEON/SSE) optimization
                // to accelerate the normalization process on multiple pixels at once.
                out_tensor[out_idx] = ((val / 255.0f) - mean[c]) / std_dev[c];
            }
        }
    }

    return TK_SUCCESS;
}