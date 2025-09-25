/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_image_preprocessor.h
 *
 * This header file defines the public API for the TrackieLLM Image Preprocessor.
 * This module centralizes common, CPU-intensive image manipulation tasks
 * required by various vision models, such as resizing, normalization, and
 * data layout conversion.
 *
 * Centralizing this logic avoids code duplication and provides a single
 * point for future performance optimizations (e.g., using SIMD instructions
 * or dedicated libraries like libyuv).
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_VISION_TK_IMAGE_PREPROCESSOR_H
#define TRACKIELLM_VISION_TK_IMAGE_PREPROCESSOR_H

#include <stdint.h>
#include "utils/tk_error_handling.h"

// Forward-declare the video frame struct
struct tk_video_frame_s;
typedef struct tk_video_frame_s tk_video_frame_t;

/**
 * @brief Resizes and normalizes an image for model input.
 *
 * This function takes a raw image frame, resizes it to the target dimensions
 * using a basic nearest-neighbor algorithm, and then normalizes the pixel
 * values. It converts the data layout from HWC (Height-Width-Channel) to
 * CHW (Channel-Height-Width) as required by many deep learning frameworks.
 *
 * @param[in] frame The source video frame (must be RGB8 format).
 * @param[out] out_tensor A pointer to the float buffer where the resulting
 *                        CHW tensor will be stored. The buffer must be
 *                        pre-allocated by the caller to be at least
 *                        `target_width * target_height * 3 * sizeof(float)`.
 * @param[in] target_width The desired width of the output tensor.
 * @param[in] target_height The desired height of the output tensor.
 * @param[in] mean A 3-element array with the mean values for R, G, B channels.
 * @param[in] std_dev A 3-element array with the std deviation values for R, G, B.
 *
 * @return TK_SUCCESS on successful preprocessing.
 * @return TK_ERROR_INVALID_ARGUMENT if any pointer is NULL or dimensions are zero.
 */
TK_NODISCARD tk_error_code_t tk_preprocessor_resize_and_normalize_to_chw(
    const tk_video_frame_t* frame,
    float* out_tensor,
    uint32_t target_width,
    uint32_t target_height,
    const float mean[3],
    const float std_dev[3]
);

#endif // TRACKIELLM_VISION_TK_IMAGE_PREPROCESSOR_H