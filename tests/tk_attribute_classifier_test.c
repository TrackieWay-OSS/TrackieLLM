/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * This file is part of the TrackieLLM project.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "vision/tk_attribute_classifier.h"
#include "cortex/tk_cortex_main.h" // For tk_video_frame_t
#include <assert.h>
#include <string.h>
#include <stdlib.h>

// A simple test runner framework
#define RUN_TEST(test) \
    printf("Running test: %s... ", #test); \
    test(); \
    printf("OK\n");

void test_classify_red_color() {
    // 1. Setup: Create a synthetic image that is pure red
    const int width = 100;
    const int height = 100;
    uint8_t* image_data = (uint8_t*)malloc(width * height * 3);
    assert(image_data);

    for (int i = 0; i < width * height * 3; i += 3) {
        image_data[i] = 255; // R
        image_data[i+1] = 0;   // G
        image_data[i+2] = 0;   // B
    }

    tk_video_frame_t frame = {
        .width = width,
        .height = height,
        .data = image_data,
    };

    tk_rect_t bbox = { .x = 10, .y = 10, .w = 80, .h = 80 };
    char* color_name = NULL;

    // 2. Act: Run the classifier
    tk_error_code_t err = tk_classify_dominant_color(&frame, &bbox, &color_name);

    // 3. Assert: Check the results
    assert(err == TK_SUCCESS);
    assert(color_name != NULL);
    assert(strcmp(color_name, "red") == 0);

    // 4. Teardown
    free(color_name);
    free(image_data);
}

int main() {
    printf("--- Starting Attribute Classifier Tests ---\n");
    RUN_TEST(test_classify_red_color);
    printf("--- All Attribute Classifier Tests Passed ---\n");
    return 0;
}