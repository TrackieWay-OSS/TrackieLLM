/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_vision_pipeline.c
 *
 * This source file implements the TrackieLLM Vision Pipeline.
 * It provides the core logic for loading models, processing video frames,
 * and fusing sensor data to create a semantic understanding of the environment.
 *
 * The implementation is designed to be modular, efficient, and robust,
 * handling model loading, inference execution, and memory management with care.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_vision_pipeline.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <pthread.h> // For mutex support

// Assume these headers exist and provide the necessary functionality
#include "vision/tk_object_detector.h"
#include "vision/tk_depth_midas.h"
#include "vision/tk_text_recognition.hpp"
#include "vision/tk_attribute_classifier.h" // For attribute classification
#include "utils/tk_logging.h"
#include "memory/tk_memory_pool.h"

// Opaque structure for the vision pipeline
struct tk_vision_pipeline_s {
    tk_vision_backend_e backend;
    int gpu_device_id;
    
    // Model instances
    tk_object_detector_t* object_detector;
    tk_depth_midas_t* depth_estimator;
    tk_text_recognizer_t* text_recognizer;
    
    // Configuration (initial and runtime)
    float object_confidence_threshold;
    uint32_t max_detected_objects;
    float focal_length_x;
    float focal_length_y;

    // Runtime updatable state
    bool is_object_detection_enabled;
    bool is_depth_estimation_enabled;

    // Thread-safety
    pthread_mutex_t config_mutex;
};

// Internal helper functions
static tk_error_code_t load_models(tk_vision_pipeline_t* pipeline, const tk_vision_pipeline_config_t* config);
static void unload_models(tk_vision_pipeline_t* pipeline);
static tk_error_code_t perform_object_detection(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result);
static tk_error_code_t perform_depth_estimation(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result);
static tk_error_code_t perform_ocr(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, const tk_rect_t* ocr_roi, tk_vision_result_t* result);
static tk_error_code_t fuse_object_depth(tk_vision_pipeline_t* pipeline, tk_vision_result_t* result, const tk_video_frame_t* frame);

//------------------------------------------------------------------------------
// Pipeline Lifecycle Management
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_vision_pipeline_create(tk_vision_pipeline_t** out_pipeline, const tk_vision_pipeline_config_t* config) {
    if (!out_pipeline || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_vision_pipeline_t* pipeline = (tk_vision_pipeline_t*)calloc(1, sizeof(tk_vision_pipeline_t));
    if (!pipeline) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    pipeline->backend = config->backend;
    pipeline->gpu_device_id = config->gpu_device_id;
    pipeline->object_confidence_threshold = config->object_confidence_threshold;
    pipeline->max_detected_objects = config->max_detected_objects;
    // Initialize camera parameters from config
    pipeline->focal_length_x = config->focal_length_x;
    pipeline->focal_length_y = config->focal_length_y;

    // Initialize the mutex
    if (pthread_mutex_init(&pipeline->config_mutex, NULL) != 0) {
        free(pipeline);
        return TK_ERROR_INTERNAL_FAILURE; // Or a more specific error for mutex failure
    }

    tk_error_code_t err = load_models(pipeline, config);
    if (err != TK_SUCCESS) {
        pthread_mutex_destroy(&pipeline->config_mutex);
        free(pipeline);
        return err;
    }

    // Default to enabled if models loaded successfully
    pipeline->is_object_detection_enabled = (pipeline->object_detector != NULL);
    pipeline->is_depth_estimation_enabled = (pipeline->depth_estimator != NULL);

    *out_pipeline = pipeline;
    tk_log_info("Vision pipeline created successfully with focal lengths: fx=%.2f, fy=%.2f", 
                pipeline->focal_length_x, pipeline->focal_length_y);
    return TK_SUCCESS;
}

void tk_vision_pipeline_destroy(tk_vision_pipeline_t** pipeline) {
    if (pipeline && *pipeline) {
        tk_log_info("Destroying vision pipeline");
        unload_models(*pipeline);
        pthread_mutex_destroy(&(*pipeline)->config_mutex);
        free(*pipeline);
        *pipeline = NULL;
    }
}

//------------------------------------------------------------------------------
// Runtime Configuration Management
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_vision_pipeline_update_config(tk_vision_pipeline_t* pipeline, const tk_vision_runtime_config_t* config) {
    if (!pipeline || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_log_info("Updating vision pipeline runtime configuration.");

    pthread_mutex_lock(&pipeline->config_mutex);

    // Update simple boolean flags
    pipeline->is_object_detection_enabled = config->enable_object_detection;
    pipeline->is_depth_estimation_enabled = config->enable_depth_estimation;

    // Update parameters that require calls to sub-modules
    if (pipeline->object_detector) {
        // This function will be added to tk_object_detector.h/c
        tk_object_detector_update_thresholds(
            pipeline->object_detector,
            config->object_confidence_threshold,
            config->iou_threshold
        );
    }

    // Store the new confidence threshold in the pipeline as well, for consistency
    pipeline->object_confidence_threshold = config->object_confidence_threshold;

    pthread_mutex_unlock(&pipeline->config_mutex);

    tk_log_info("Runtime configuration updated successfully.");

    return TK_SUCCESS;
}


//------------------------------------------------------------------------------
// Core Processing Function
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_vision_pipeline_process_frame(
    tk_vision_pipeline_t* pipeline,
    const tk_video_frame_t* video_frame,
    tk_vision_analysis_flags_t analysis_flags,
    const tk_rect_t* ocr_roi,
    uint64_t timestamp_ns,
    tk_vision_result_t** out_result
) {
    if (!pipeline || !video_frame || !out_result) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_vision_result_t* result = (tk_vision_result_t*)calloc(1, sizeof(tk_vision_result_t));
    if (!result) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    result->source_frame_timestamp_ns = timestamp_ns;
    result->valid_analyses_mask = TK_VISION_RESULT_NONE;

    tk_error_code_t err = TK_SUCCESS;

    // Check which analyses are enabled in a thread-safe manner
    pthread_mutex_lock(&pipeline->config_mutex);
    bool object_detection_enabled = pipeline->is_object_detection_enabled;
    bool depth_estimation_enabled = pipeline->is_depth_estimation_enabled;
    pthread_mutex_unlock(&pipeline->config_mutex);

    // --- Perform Object Detection ---
    if (object_detection_enabled && (analysis_flags & TK_VISION_ANALYZE_OBJECT_DETECTION) && pipeline->object_detector) {
        err = perform_object_detection(pipeline, video_frame, result);
        if (err == TK_SUCCESS) {
            tk_log_debug("Object detection completed, found %zu objects", result->object_count);
            result->valid_analyses_mask |= TK_VISION_RESULT_OBJECT_DETECTION;
        } else {
            tk_log_warning("Object detection failed with error %d, continuing...", err);
        }
    }

    // --- Perform Depth Estimation ---
    if (depth_estimation_enabled && (analysis_flags & TK_VISION_ANALYZE_DEPTH_ESTIMATION) && pipeline->depth_estimator) {
        err = perform_depth_estimation(pipeline, video_frame, result);
        if (err == TK_SUCCESS) {
            tk_log_debug("Depth estimation completed");
            result->valid_analyses_mask |= TK_VISION_RESULT_DEPTH_ESTIMATION;
        } else {
            tk_log_warning("Depth estimation failed with error %d, continuing...", err);
        }
    }

    // --- Perform OCR ---
    bool ocr_requested_by_flag = (analysis_flags & TK_VISION_ANALYZE_OCR);
    bool ocr_requested_by_roi = (ocr_roi != NULL);
    bool ocr_triggered_by_detection = false;

    if (!ocr_requested_by_flag && !ocr_requested_by_roi && (result->valid_analyses_mask & TK_VISION_RESULT_OBJECT_DETECTION)) {
        for (size_t i = 0; i < result->object_count; ++i) {
            if (strstr(result->objects[i].label, "sign") || strstr(result->objects[i].label, "text")) {
                ocr_triggered_by_detection = true;
                break;
            }
        }
    }

    if ((ocr_requested_by_flag || ocr_requested_by_roi || ocr_triggered_by_detection) && pipeline->text_recognizer) {
        // Pass the explicit ROI if provided
        err = perform_ocr(pipeline, video_frame, ocr_roi, result);
        if (err == TK_SUCCESS) {
            tk_log_debug("OCR completed, found %zu text blocks", result->text_block_count);
            result->valid_analyses_mask |= TK_VISION_RESULT_OCR;
        } else {
            tk_log_warning("OCR failed with error %d, continuing...", err);
        }
    }

    // --- Perform Navigation Analysis ---
    bool can_analyze_nav = (result->valid_analyses_mask & TK_VISION_RESULT_DEPTH_ESTIMATION);
    if ((analysis_flags & TK_VISION_ANALYZE_NAVIGATION_CUES) && can_analyze_nav) {
        CNavigationCues* nav_cues = tk_vision_rust_analyze_navigation(result->depth_map);
        if (nav_cues) {
            tk_log_debug("Navigation analysis completed. Found %zu vertical changes.", nav_cues->vertical_changes_count);
            result->valid_analyses_mask |= TK_VISION_RESULT_NAVIGATION_CUES;
            // In a full implementation, this data would be attached to the result.
            tk_vision_rust_free_navigation_cues(nav_cues);
        } else {
            tk_log_warning("Navigation analysis failed or returned no data.");
        }
    }

    // --- Perform Sensor Fusion ---
    bool can_fuse = (result->valid_analyses_mask & TK_VISION_RESULT_OBJECT_DETECTION) &&
                    (result->valid_analyses_mask & TK_VISION_RESULT_DEPTH_ESTIMATION);
    if ((analysis_flags & TK_VISION_ANALYZE_FUSION_DISTANCE) && can_fuse) {
        err = fuse_object_depth(pipeline, result, video_frame);
        if (err == TK_SUCCESS) {
            tk_log_debug("Object-depth fusion completed");
            result->valid_analyses_mask |= TK_VISION_RESULT_FUSION_DISTANCE;

            // --- Build Scene Graph ---
            // This is done only if fusion was successful and it was requested.
            if ((analysis_flags & TK_VISION_ANALYZE_SCENE_GRAPH) && nav_cues) {
                // The C `tk_vision_object_t` is not directly compatible with Rust's `EnrichedObject`.
                // A temporary conversion is needed. For now, we assume they are compatible
                // as a simplification, but this would need a proper conversion function.
                result->serialized_scene_graph = tk_vision_rust_build_scene_graph(
                    (const EnrichedObject*)result->objects,
                    result->object_count,
                    &nav_cues->point_cloud,
                    result->depth_map->width,
                    result->depth_map->height
                );
                if (result->serialized_scene_graph) {
                    tk_log_debug("Scene graph built successfully.");
                } else {
                    tk_log_warning("Scene graph construction failed.");
                }
            }

        } else {
            tk_log_warning("Object-depth fusion failed with error %d, continuing...", err);
        }
    }

    *out_result = result;
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Result Data Management
//------------------------------------------------------------------------------

void tk_vision_result_destroy(tk_vision_result_t** result) {
    if (result && *result) {
        // Free objects
        if ((*result)->objects) {
            for (size_t i = 0; i < (*result)->object_count; ++i) {
                if ((*result)->objects[i].label) {
                    free((void*)((*result)->objects[i].label));
                }
                if ((*result)->objects[i].recognized_text) {
                    free((*result)->objects[i].recognized_text);
                }
                if ((*result)->objects[i].attributes) {
                    free((*result)->objects[i].attributes);
                }
            }
            free((*result)->objects);
        }

        // Free text blocks
        if ((*result)->text_blocks) {
            for (size_t i = 0; i < (*result)->text_block_count; ++i) {
                if ((*result)->text_blocks[i].text) {
                    free((void*)((*result)->text_blocks[i].text));
                }
            }
            free((*result)->text_blocks);
        }

        // Free depth map
        if ((*result)->depth_map) {
            if ((*result)->depth_map->data) {
                free((*result)->depth_map->data);
            }
            free((*result)->depth_map);
        }

        // Free scene graph string
        if ((*result)->serialized_scene_graph) {
            tk_vision_rust_free_string((*result)->serialized_scene_graph);
        }

        free(*result);
        *result = NULL;
    }
}

//------------------------------------------------------------------------------
// Internal Helper Functions
//------------------------------------------------------------------------------

// A placeholder for COCO class labels. A real implementation would load this from a file.
const char* COCO_LABELS[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};
const size_t COCO_CLASS_COUNT = 80;


static tk_error_code_t load_models(tk_vision_pipeline_t* pipeline, const tk_vision_pipeline_config_t* config) {
    tk_error_code_t err;

    if (!config->object_detection_model_path || !config->depth_estimation_model_path || !config->tesseract_data_path) {
        tk_log_error("Missing required model paths in configuration");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_log_info("Loading vision models...");

    // --- Load Object Detector ---
    tk_object_detector_config_t detector_config = {
        .backend = config->backend,
        .gpu_device_id = config->gpu_device_id,
        .model_path = config->object_detection_model_path,
        .input_width = 640, // Common for YOLOv5
        .input_height = 640,
        .class_labels = COCO_LABELS,
        .class_count = COCO_CLASS_COUNT,
        .confidence_threshold = config->object_confidence_threshold,
        .iou_threshold = 0.5f // A common default for NMS
    };
    err = tk_object_detector_create(&pipeline->object_detector, &detector_config);
    if (err != TK_SUCCESS) {
        tk_log_warning("Failed to create object detector (error %d). Object detection will be disabled.", err);
        pipeline->object_detector = NULL; // Ensure it's null
    } else {
        tk_log_debug("Object detection model loaded successfully");
    }

    // --- Load Depth Estimator ---
    tk_depth_estimator_config_t depth_config = {
        .backend = config->backend,
        .gpu_device_id = config->gpu_device_id,
        .model_path = config->depth_estimation_model_path,
        .input_width = 256, // Common for MiDaS DPT-SwinV2-Tiny
        .input_height = 256
    };
    err = tk_depth_estimator_create(&pipeline->depth_estimator, &depth_config);
    if (err != TK_SUCCESS) {
        tk_log_warning("Failed to create depth estimator (error %d). Depth estimation will be disabled.", err);
        pipeline->depth_estimator = NULL; // Ensure it's null
    } else {
        tk_log_debug("Depth estimation model loaded successfully");
    }

    // --- Load Text Recognizer ---
    tk_ocr_config_t ocr_config = {
        .data_path = config->tesseract_data_path,
        .language = TK_OCR_LANG_ENGLISH, // Default to English
        .engine_mode = TK_OCR_ENGINE_DEFAULT,
        .psm = TK_OCR_PSM_AUTO,
        .dpi = 300, // A reasonable default DPI
        .num_threads = 2, // Default number of threads
    };
    err = tk_text_recognition_create(&pipeline->text_recognizer, &ocr_config);
    if (err != TK_SUCCESS) {
        tk_log_warning("Failed to create text recognizer (error %d). OCR will be disabled.", err);
        pipeline->text_recognizer = NULL; // Ensure it's null
    } else {
        tk_log_debug("Text recognizer initialized successfully");
    }

    return TK_SUCCESS;
}

static void unload_models(tk_vision_pipeline_t* pipeline) {
    if (pipeline->object_detector) {
        tk_object_detector_destroy(&pipeline->object_detector);
    }
    if (pipeline->depth_estimator) {
        tk_depth_estimator_destroy(&pipeline->depth_estimator);
    }
    if (pipeline->text_recognizer) {
        tk_text_recognition_destroy(&pipeline->text_recognizer);
    }
}

static tk_error_code_t perform_object_detection(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result) {
    tk_detection_result_t* detections = NULL;
    size_t detection_count = 0;

    tk_error_code_t err = tk_object_detector_detect(pipeline->object_detector, frame, &detections, &detection_count);
    if (err != TK_SUCCESS) {
        return err;
    }

    if (detection_count > 0) {
        result->objects = (tk_vision_object_t*)calloc(detection_count, sizeof(tk_vision_object_t));
        if (!result->objects) {
            tk_object_detector_free_results(&detections);
            return TK_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < detection_count; ++i) {
            result->objects[i].class_id = detections[i].class_id;
            result->objects[i].label = detections[i].label; // Direct pointer, no copy needed as labels are static const
            result->objects[i].confidence = detections[i].confidence;
            result->objects[i].bbox = detections[i].bbox;
            result->objects[i].distance_meters = 0.0f; // Will be populated by fusion
            result->objects[i].width_meters = 0.0f;    // Will be populated by fusion
            result->objects[i].height_meters = 0.0f;   // Will be populated by fusion
            result->objects[i].is_partially_occluded = false; // Will be populated by fusion
            result->objects[i].recognized_text = NULL; // Will be populated by OCR
            result->objects[i].attributes = NULL;      // Will be populated by attribute classifier

            // --- Classify Attributes ---
            char* color_name = NULL;
            char* door_state = NULL;
            char temp_attributes[128] = {0};

            if (tk_classify_dominant_color(frame, &detections[i].bbox, &color_name) == TK_SUCCESS) {
                strncat(temp_attributes, "color:", sizeof(temp_attributes) - strlen(temp_attributes) - 1);
                strncat(temp_attributes, color_name, sizeof(temp_attributes) - strlen(temp_attributes) - 1);
                free(color_name);
            }

            if (strstr(detections[i].label, "door") != NULL) {
                if (tk_classify_door_state(frame, &detections[i].bbox, &door_state) == TK_SUCCESS) {
                    if (strlen(temp_attributes) > 0) {
                        strncat(temp_attributes, ",", sizeof(temp_attributes) - strlen(temp_attributes) - 1);
                    }
                    strncat(temp_attributes, "state:", sizeof(temp_attributes) - strlen(temp_attributes) - 1);
                    strncat(temp_attributes, door_state, sizeof(temp_attributes) - strlen(temp_attributes) - 1);
                    free(door_state);
                }
            }

            if (strlen(temp_attributes) > 0) {
                result->objects[i].attributes = strdup(temp_attributes);
            }
        }
        result->object_count = detection_count;
    }

    tk_object_detector_free_results(&detections);
    return TK_SUCCESS;
}

static tk_error_code_t perform_depth_estimation(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result) {
    // The estimate function allocates the depth map, which will be owned by the result struct.
    tk_error_code_t err = tk_depth_estimator_estimate(pipeline->depth_estimator, frame, &result->depth_map);
    if (err != TK_SUCCESS) {
        return err;
    }

    // The 'result->depth_map' is now populated and owned by the result.
    // It will be freed in tk_vision_result_destroy.

    return TK_SUCCESS;
}

static tk_error_code_t perform_ocr(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, const tk_rect_t* ocr_roi, tk_vision_result_t* result) {
    tk_log_debug("Performing OCR.");

    // Base image parameters for the whole frame
    tk_ocr_image_params_t image_params = {
        .image_data = frame->data,
        .width = frame->width,
        .height = frame->height,
        .channels = 3, // Assuming RGB8
        .stride = frame->width * 3,
        .psm = TK_OCR_PSM_AUTO,
    };

    tk_ocr_result_t* ocr_result = NULL;
    tk_error_code_t err = TK_SUCCESS;

    if (ocr_roi) {
        // --- Case 1: Targeted OCR on a specific Region of Interest ---
        tk_log_debug("Performing OCR on specified ROI: [x:%d, y:%d, w:%d, h:%d]", ocr_roi->x, ocr_roi->y, ocr_roi->w, ocr_roi->h);
        err = tk_text_recognition_process_region(
            pipeline->text_recognizer, &image_params,
            ocr_roi->x, ocr_roi->y, ocr_roi->w, ocr_roi->h,
            &ocr_result
        );
    } else {
        // --- Case 2: General OCR on the whole frame ---
        // This is triggered by the TK_VISION_ANALYZE_OCR flag or by detecting text-like objects.
        tk_log_debug("Performing OCR on the full frame.");
        err = tk_text_recognition_process_image(pipeline->text_recognizer, &image_params, &ocr_result);
    }

    if (err != TK_SUCCESS) {
        tk_log_warn("Tesseract image processing failed with error %d", err);
        return err;
    }

    if (ocr_result && ocr_result->block_count > 0) {
        // Convert Tesseract's result blocks into our pipeline's result blocks
        result->text_block_count = ocr_result->block_count;
        result->text_blocks = (tk_vision_text_block_t*)calloc(result->text_block_count, sizeof(tk_vision_text_block_t));
        if (!result->text_blocks) {
            tk_text_recognition_free_result(&ocr_result);
            return TK_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < ocr_result->block_count; ++i) {
            result->text_blocks[i].text = strdup(ocr_result->blocks[i].text);
            result->text_blocks[i].confidence = ocr_result->blocks[i].confidence;
            result->text_blocks[i].bbox = ocr_result->blocks[i].bbox;
        }
    }

    tk_text_recognition_free_result(&ocr_result);
    return TK_SUCCESS;
}

// --- FFI Declarations for Rust Fusion Library ---

// Define the C-compatible structs that Rust will return
typedef struct {
    uint32_t class_id;
    float confidence;
    tk_rect_t bbox;
    float distance_meters;
    float width_meters;
    float height_meters;
    bool is_partially_occluded;
} EnrichedObject;

typedef struct {
    const EnrichedObject* objects;
    size_t count;
} CFusedResult;

// Declare the external Rust functions
extern CFusedResult* tk_vision_rust_fuse_data(
    const tk_detection_result_t* detections_ptr,
    size_t detection_count,
    const tk_vision_depth_map_t* depth_map_ptr,
    uint32_t frame_width,
    uint32_t frame_height,
    float focal_length_x,
    float focal_length_y
);

extern void tk_vision_rust_free_fused_result(CFusedResult* result_ptr);

// --- FFI Declarations for Rust Navigation Analysis Library ---

// C-compatible mirror of the GroundPlaneStatus enum in Rust
typedef enum {
    C_GROUND_PLANE_STATUS_UNKNOWN,
    C_GROUND_PLANE_STATUS_FLAT,
    C_GROUND_PLANE_STATUS_OBSTACLE,
    C_GROUND_PLANE_STATUS_HOLE,
    C_GROUND_PLANE_STATUS_RAMP_UP,
    C_GROUND_PLANE_STATUS_RAMP_DOWN,
} CGroundPlaneStatus;

// C-compatible mirror of the VerticalChange struct in Rust
typedef struct {
    float height_m;
    CGroundPlaneStatus status;
    uint32_t grid_x;
    uint32_t grid_y;
} CVerticalChange;

// C-compatible mirror of the NavigationCues struct in Rust
typedef struct {
    const CGroundPlaneStatus* traversability_grid;
    size_t grid_size;
    uint32_t grid_width;
    uint32_t grid_height;
    const CVerticalChange* detected_vertical_changes;
    size_t vertical_changes_count;
} CNavigationCues;

// Declare the external Rust functions for navigation analysis
extern CNavigationCues* tk_vision_rust_analyze_navigation(const tk_vision_depth_map_t* depth_map_ptr);
extern void tk_vision_rust_free_navigation_cues(CNavigationCues* cues_ptr);

// C-compatible mirror of nalgebra::Point3<f32>
typedef struct {
    float x, y, z;
} CPoint3D;

// C-compatible mirror of the CPointCloud struct in Rust
typedef struct {
    const CPoint3D* points;
    size_t count;
} CPointCloud;

// --- FFI Declarations for Rust Scene Graph Library ---
extern char* tk_vision_rust_build_scene_graph(const EnrichedObject* enriched_objects_ptr, size_t object_count, const CPointCloud* point_cloud, uint32_t depth_map_width, uint32_t depth_map_height);
extern void tk_vision_rust_free_string(char* s);


/**
 * @brief Calls the Rust library to fuse object detection and depth data.
 *
 * This function acts as a bridge to the high-performance, safe Rust implementation
 * of the fusion logic. It passes pointers to the raw C data, and the Rust side
 * returns a new structure with the calculated distances and sizes.
 */
static tk_error_code_t fuse_object_depth(tk_vision_pipeline_t* pipeline, tk_vision_result_t* result, const tk_video_frame_t* frame) {
    if (!result->depth_map || result->object_count == 0) {
        return TK_SUCCESS; // Nothing to fuse
    }

    // The Rust function needs an array of tk_detection_result_t, but our pipeline
    // result has tk_vision_object_t. For this call, we'll create a temporary
    // array of the required input type.
    tk_detection_result_t* raw_detections = (tk_detection_result_t*)malloc(result->object_count * sizeof(tk_detection_result_t));
    if (!raw_detections) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < result->object_count; ++i) {
        raw_detections[i].class_id = result->objects[i].class_id;
        raw_detections[i].label = result->objects[i].label;
        raw_detections[i].confidence = result->objects[i].confidence;
        raw_detections[i].bbox = result->objects[i].bbox;
    }

    // Call the external Rust function
    CFusedResult* fused_result = tk_vision_rust_fuse_data(
        raw_detections,
        result->object_count,
        result->depth_map,
        frame->width,
        frame->height,
        pipeline->focal_length_x,
        pipeline->focal_length_y
    );

    free(raw_detections); // Free the temporary array

    if (!fused_result) {
        tk_log_error("Rust fusion function returned null. Fusion failed.");
        return TK_ERROR_INFERENCE_FAILED; // Or a more specific error
    }

    // The number of objects should match
    if (fused_result->count != result->object_count) {
        tk_log_warn("Mismatch in object count between C (%zu) and Rust (%zu) layers.", result->object_count, fused_result->count);
    }

    // Copy the fused data (distance, width, height) back into our main result struct
    for (size_t i = 0; i < result->object_count && i < fused_result->count; ++i) {
        result->objects[i].distance_meters = fused_result->objects[i].distance_meters;
        result->objects[i].width_meters = fused_result->objects[i].width_meters;
        result->objects[i].height_meters = fused_result->objects[i].height_meters;
        result->objects[i].is_partially_occluded = fused_result->objects[i].is_partially_occluded;

        tk_log_debug("Object %zu (%s) fused distance: %.2fm, size: %.2fx%.2fm, occluded: %s",
                     i, result->objects[i].label,
                     result->objects[i].distance_meters,
                     result->objects[i].width_meters, result->objects[i].height_meters,
                     result->objects[i].is_partially_occluded ? "yes" : "no");
    }

    // IMPORTANT: Free the memory allocated by the Rust library
    tk_vision_rust_free_fused_result(fused_result);

    return TK_SUCCESS;
}
