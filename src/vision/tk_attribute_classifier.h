/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_attribute_classifier.h
 *
 * This header file defines the public API for the Attribute Classifier module.
 * This module uses classical computer vision techniques to determine high-level
 * attributes of detected objects, such as dominant color or state (e.g., open/closed),
 * without requiring a heavy deep learning model.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_VISION_TK_ATTRIBUTE_CLASSIFIER_H
#define TRACKIELLM_VISION_TK_ATTRIBUTE_CLASSIFIER_H

#include "tk_vision_pipeline.h" // For tk_video_frame_t and tk_rect_t
#include "utils/tk_error_handling.h"

/**
 * @brief Determines the dominant color name from a region of an image.
 *
 * This function analyzes the pixels within the given bounding box of a frame
 * and returns a string representing the most prominent color.
 *
 * @param[in] frame The full video frame.
 * @param[in] bbox The bounding box of the object to analyze.
 * @param[out] out_color_name A pointer to a char* that will be allocated and
 *                            filled with the name of the dominant color (e.g., "red").
 *                            The caller is responsible for freeing this memory.
 *
 * @return TK_SUCCESS on successful classification.
 * @return TK_ERROR_INVALID_ARGUMENT if inputs are invalid.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 */
TK_NODISCARD tk_error_code_t tk_classify_dominant_color(
    const tk_video_frame_t* frame,
    const tk_rect_t* bbox,
    char** out_color_name
);

/**
 * @brief Determines if a door is open or closed based on line detection.
 *
 * @param[in] frame The full video frame.
 * @param[in] bbox The bounding box of the door object.
 * @param[out] out_state_name A pointer to a char* that will be allocated and
 *                            filled with the state (e.g., "open", "closed").
 *
 * @return TK_SUCCESS on successful classification.
 */
TK_NODISCARD tk_error_code_t tk_classify_door_state(
    const tk_video_frame_t* frame,
    const tk_rect_t* bbox,
    char** out_state_name
);

#endif // TRACKIELLM_VISION_TK_ATTRIBUTE_CLASSIFIER_H