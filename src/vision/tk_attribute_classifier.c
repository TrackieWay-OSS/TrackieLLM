/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_attribute_classifier.c
 *
 * This source file implements the Attribute Classifier module.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_attribute_classifier.h"
#include "cortex/tk_cortex_main.h" // For tk_video_frame_t definition
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// --- Internal Helper for Color Conversion ---
static void rgb_to_hsv(float r, float g, float b, float *h, float *s, float *v) {
    float max_val = fmaxf(r, fmaxf(g, b));
    float min_val = fminf(r, fminf(g, b));
    float delta = max_val - min_val;
    *v = max_val;
    if (max_val > 0.0) {
        *s = (delta / max_val);
    } else {
        *s = 0.0;
        *h = NAN;
        return;
    }
    if (r >= max_val) *h = (g - b) / delta;
    else if (g >= max_val) *h = 2.0 + (b - r) / delta;
    else *h = 4.0 + (r - g) / delta;
    *h *= 60.0;
    if (*h < 0.0) *h += 360.0;
}

// --- Public API Implementation ---

TK_NODISCARD tk_error_code_t tk_classify_dominant_color(
    const tk_video_frame_t* frame,
    const tk_rect_t* bbox,
    char** out_color_name
) {
    if (!frame || !bbox || !out_color_name) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Bins for a simple color histogram (approximating basic colors)
    // Order: red, yellow, green, cyan, blue, magenta
    const char* color_names[] = {"red", "yellow", "green", "cyan", "blue", "magenta", "black", "white", "gray"};
    long color_bins[9] = {0};

    for (int y = bbox->y; y < (bbox->y + bbox->h); ++y) {
        for (int x = bbox->x; x < (bbox->x + bbox->w); ++x) {
            if (x < 0 || x >= frame->width || y < 0 || y >= frame->height) continue;

            size_t idx = (y * frame->width + x) * 3;
            float r = frame->data[idx] / 255.0f;
            float g = frame->data[idx + 1] / 255.0f;
            float b = frame->data[idx + 2] / 255.0f;

            float h, s, v;
            rgb_to_hsv(r, g, b, &h, &s, &v);

            // Simple classification based on HSV
            if (s < 0.1) { // Grayscale
                if (v < 0.1) color_bins[6]++; // Black
                else if (v > 0.9) color_bins[7]++; // White
                else color_bins[8]++; // Gray
            } else {
                if (h < 30 || h >= 330) color_bins[0]++; // Red
                else if (h >= 30 && h < 90) color_bins[1]++; // Yellow
                else if (h >= 90 && h < 150) color_bins[2]++; // Green
                else if (h >= 150 && h < 210) color_bins[3]++; // Cyan
                else if (h >= 210 && h < 270) color_bins[4]++; // Blue
                else if (h >= 270 && h < 330) color_bins[5]++; // Magenta
            }
        }
    }

    // Find the bin with the most pixels
    int max_idx = 0;
    for (int i = 1; i < 9; ++i) {
        if (color_bins[i] > color_bins[max_idx]) {
            max_idx = i;
        }
    }

    // Allocate and return the color name
    *out_color_name = strdup(color_names[max_idx]);
    if (!*out_color_name) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    return TK_SUCCESS;
}


TK_NODISCARD tk_error_code_t tk_classify_door_state(
    const tk_video_frame_t* frame,
    const tk_rect_t* bbox,
    char** out_state_name
) {
    if (!frame || !bbox || !out_state_name) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // This is a functional placeholder. A full implementation would require:
    // 1. A robust Canny edge detection implementation.
    // 2. A robust Hough transform implementation to detect lines.
    // 3. Logic to analyze the orientation and length of detected lines.

    // Placeholder logic: Count the number of "strong vertical edge" pixels.
    // A real implementation would be much more complex.
    int vertical_edge_pixels = 0;
    for (int y = bbox->y + 1; y < (bbox->y + bbox->h) - 1; ++y) {
        for (int x = bbox->x + 1; x < (bbox->x + bbox->w) - 1; ++x) {
            // Convert to grayscale (simple luminance)
            size_t idx_center = (y * frame->width + x) * 3;
            size_t idx_above = ((y - 1) * frame->width + x) * 3;
            size_t idx_below = ((y + 1) * frame->width + x) * 3;

            uint8_t gray_center = (frame->data[idx_center] * 299 + frame->data[idx_center+1] * 587 + frame->data[idx_center+2] * 114) / 1000;
            uint8_t gray_above = (frame->data[idx_above] * 299 + frame->data[idx_above+1] * 587 + frame->data[idx_above+2] * 114) / 1000;
            uint8_t gray_below = (frame->data[idx_below] * 299 + frame->data[idx_below+1] * 587 + frame->data[idx_below+2] * 114) / 1000;

            // Simple Sobel-like vertical edge detection
            if (abs((int)gray_above - (int)gray_below) > 100) { // High threshold for strong edges
                vertical_edge_pixels++;
            }
        }
    }

    // If a significant portion of the door's area consists of strong vertical edges,
    // we assume it's closed.
    float edge_density = (float)vertical_edge_pixels / (float)(bbox->w * bbox->h);

    if (edge_density > 0.1) { // 10% density threshold
        *out_state_name = strdup("closed");
    } else {
        *out_state_name = strdup("open");
    }

    if (!*out_state_name) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    return TK_SUCCESS;
}