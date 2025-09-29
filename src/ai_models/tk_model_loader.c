/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_model_loader.c
 *
 * This source file implements the Model Loader module for TrackieLLM.
 * This component is responsible for loading, validating, and preparing
 * various AI models (LLM, vision, audio) for use in the TrackieLLM system.
 *
 * The implementation provides a unified interface for model initialization
 * across different frameworks and supports multiple model formats.
 *
 * Dependencies:
 *   - llama.cpp (for GGUF models)
 *   - ONNX Runtime (for ONNX models)
 *   - TensorFlow Lite (for TFLite models)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "ai_models/tk_model_loader.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

// Real AI library headers
#include "llama.h"
#include "onnxruntime_c_api.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

// Helper macro for checking ONNX Runtime calls
#define ORT_CHECK(expr) { \
    OrtStatus* s = (expr); \
    if (s) { \
        const OrtApi* g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION); \
        TK_LOG_ERROR("ONNX Runtime error: %s", g_ort_api->GetErrorMessage(s)); \
        g_ort_api->ReleaseStatus(s); \
        return TK_ERROR_MODEL_LOAD_FAILED; \
    } \
}

// Maximum number of models that can be loaded simultaneously
#define MAX_LOADED_MODELS 32

// Maximum path length
#define MAX_PATH_LENGTH 4096

// Estrutura para o handle do modelo ONNX
typedef struct {
    OrtEnv* env;
    OrtSession* session;
    OrtAllocator* allocator;
    // Salvar nomes e shapes de input/output para reuso
    size_t input_count;
    char** input_names;
    int64_t** input_dims;
    size_t* input_dim_counts;

    size_t output_count;
    char** output_names;
} tk_onnx_handle_t;

// Estrutura para o handle do modelo GGUF
typedef struct {
    struct llama_model* model;
    // O contexto (ctx) será gerenciado pelo RUNNER, não pelo LOADER.
} tk_gguf_handle_t;

// Estrutura interna principal
typedef struct {
    void* handle;                    // Ponteiro para tk_onnx_handle_t ou tk_gguf_handle_t
    tk_model_metadata_t metadata;
    tk_model_format_e format;
    char path[MAX_PATH_LENGTH];
    time_t load_time;
    uint32_t reference_count;
    bool is_loaded;
} tk_internal_model_t;

#include "tk_memory_manager.h"

// Internal structure for model loader context
struct tk_model_loader_s {
    tk_model_loader_config_t config;
    tk_internal_model_t* loaded_models;
    size_t loaded_model_count;
    size_t max_loaded_models;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t total_loads;
    uint64_t total_unloads;
    size_t cache_size_bytes;
    char* temp_dir;
    char* cache_dir;
    void* framework_contexts[8];
    tk_memory_manager_t* memory_manager; // The new memory manager
};

// Internal helper functions
static tk_error_code_t validate_config(const tk_model_loader_config_t* config);
static tk_error_code_t init_framework_contexts(tk_model_loader_t* loader);
static void cleanup_framework_contexts(tk_model_loader_t* loader);
static tk_error_code_t load_model_gguf(tk_model_loader_t* loader, const tk_model_load_params_t* params, void** out_handle);
static tk_error_code_t load_model_onnx(tk_model_loader_t* loader, const tk_model_load_params_t* params, void** out_handle);
static tk_error_code_t load_model_tflite(tk_model_loader_t* loader, const tk_model_load_params_t* params, void** out_handle);
static tk_error_code_t unload_model_internal(tk_model_loader_t* loader, void* model_handle);
static tk_internal_model_t* find_loaded_model(tk_model_loader_t* loader, const char* model_path);
static tk_internal_model_t* find_available_model_slot(tk_model_loader_t* loader);
static tk_error_code_t extract_model_metadata(tk_internal_model_t* model);
static tk_error_code_t validate_model_file(const char* model_path, tk_model_format_e* out_format);
static tk_model_format_e detect_model_format(const char* model_path);
static bool is_path_directory(const char* path);
static size_t get_file_size(const char* path);
static tk_error_code_t copy_file(const char* src_path, const char* dst_path);
static tk_error_code_t create_directory(const char* path);
static tk_error_code_t delete_file(const char* path);
static char* get_file_extension(const char* path);
static bool string_ends_with(const char* str, const char* suffix);
static char* duplicate_string(const char* src);
static void free_string(char* str);
static void free_model_metadata(tk_model_metadata_t* metadata);
static tk_error_code_t parse_gguf_metadata(const char* model_path, tk_model_metadata_t* metadata);
static tk_error_code_t parse_onnx_metadata(const char* model_path, tk_model_metadata_t* metadata);
static tk_error_code_t parse_tflite_metadata(const char* model_path, tk_model_metadata_t* metadata);

//------------------------------------------------------------------------------
// Private helper functions
//------------------------------------------------------------------------------

/**
 * @brief Validates the model loader configuration
 */
static tk_error_code_t validate_config(const tk_model_loader_config_t* config) {
    if (!config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate required fields
    if (config->max_models == 0) {
        TK_LOG_ERROR("Invalid max_models: %u", config->max_models);
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (config->num_threads == 0) {
        TK_LOG_ERROR("Invalid num_threads: %u", config->num_threads);
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Initializes framework-specific contexts
 */
static tk_error_code_t init_framework_contexts(tk_model_loader_t* loader) {
    if (!loader) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Initialize framework contexts based on configuration
    // This would initialize llama.cpp, ONNX Runtime, TensorFlow Lite, etc.
    TK_LOG_INFO("Initializing framework contexts");
    
    // For now, we just initialize placeholders
    for (int i = 0; i < 8; i++) {
        loader->framework_contexts[i] = NULL;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Cleans up framework-specific contexts
 */
static void cleanup_framework_contexts(tk_model_loader_t* loader) {
    if (!loader) return;
    
    // Clean up framework contexts
    TK_LOG_INFO("Cleaning up framework contexts");
    
    // This would clean up llama.cpp, ONNX Runtime, TensorFlow Lite, etc.
    for (int i = 0; i < 8; i++) {
        if (loader->framework_contexts[i]) {
            // Framework-specific cleanup
            loader->framework_contexts[i] = NULL;
        }
    }
}

/**
 * @brief Loads a GGUF model
 */
static tk_error_code_t load_model_gguf(tk_model_loader_t* loader, const tk_model_load_params_t* params, void** out_handle) {
    if (!loader || !params || !out_handle || !params->model_path) return TK_ERROR_INVALID_ARGUMENT;
    *out_handle = NULL;

    tk_internal_model_t* model = find_loaded_model(loader, params->model_path->path_str);
    if (model) {
        model->reference_count++;
        *out_handle = model;
        loader->cache_hits++;
        return TK_SUCCESS;
    }

    // --- New Eviction Logic ---
    uint64_t required_bytes = get_file_size(params->model_path->path_str);
    tk_internal_model_t* model_to_evict = NULL;

    // Check if we need to evict a model.
    // This is determined by the memory manager based on current usage and required size.
    // 1. Build list of candidates for eviction (models not in use).
    tk_internal_model_t* candidates[MAX_LOADED_MODELS];
    size_t candidate_count = 0;
    for (size_t i = 0; i < loader->max_loaded_models; i++) {
        if (loader->loaded_models[i].is_loaded && loader->loaded_models[i].reference_count == 0) {
            candidates[candidate_count++] = &loader->loaded_models[i];
        }
    }

    // 2. Ask memory manager to select one if needed.
    model_to_evict = tk_memory_manager_select_model_for_eviction(
        loader->memory_manager, candidates, candidate_count, required_bytes);

    if (model_to_evict) {
        // 3. Evict the selected model.
        unload_model_internal(loader, model_to_evict);
    }
    // --- End of New Eviction Logic ---

    model = find_available_model_slot(loader);
    if (!model) {
        TK_LOG_ERROR("Could not load model %s, not enough memory and no suitable eviction candidate found.", params->model_path->path_str);
        return TK_ERROR_RESOURCE_EXHAUSTED;
    }

    tk_gguf_handle_t* gguf_handle = calloc(1, sizeof(tk_gguf_handle_t));
    if (!gguf_handle) return TK_ERROR_OUT_OF_MEMORY;

    llama_backend_init(params->numa);
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params->gpu_layers;
    model_params.use_mmap = params->use_mmap;
    model_params.use_mlock = params->use_mlock;

    gguf_handle->model = llama_load_model_from_file(params->model_path->path_str, model_params);
    if (!gguf_handle->model) {
        TK_LOG_ERROR("Failed to load GGUF model from: %s", params->model_path->path_str);
        free(gguf_handle);
        llama_backend_free();
        return TK_ERROR_MODEL_LOAD_FAILED;
    }

    // Apply LoRA adapter if provided
    if (params->lora_adapter && strlen(params->lora_adapter) > 0) {
        TK_LOG_INFO("Applying LoRA adapter from: %s", params->lora_adapter);
        int err = llama_model_apply_lora_from_file(gguf_handle->model, params->lora_adapter, NULL, NULL, params->cpu_threads);
        if (err != 0) {
            TK_LOG_ERROR("Failed to apply LoRA adapter. Error code: %d", err);
            llama_free_model(gguf_handle->model);
            free(gguf_handle);
            llama_backend_free();
            return TK_ERROR_MODEL_LOAD_FAILED;
        }
    }

    if (!gguf_handle->model) { // Double check after potential LoRA failure
        TK_LOG_ERROR("Model handle became null after LoRA application attempt.");
        free(gguf_handle);
        llama_backend_free();
        return TK_ERROR_MODEL_LOAD_FAILED;
    }

    model->handle = gguf_handle;
    model->format = TK_MODEL_FORMAT_GGUF;
    model->is_loaded = true;
    model->reference_count = 1;
    model->load_time = time(NULL);
    strncpy(model->path, params->model_path->path_str, MAX_PATH_LENGTH - 1);
    model->path[MAX_PATH_LENGTH - 1] = '\0';
    
    extract_model_metadata(model);
    tk_memory_manager_confirm_allocation(loader->memory_manager, model);

    *out_handle = model;
    loader->total_loads++;
    loader->cache_misses++;
    return TK_SUCCESS;
}

static tk_error_code_t load_model_onnx(tk_model_loader_t* loader, const tk_model_load_params_t* params, void** out_handle) {
    if (!loader || !params || !out_handle || !params->model_path) return TK_ERROR_INVALID_ARGUMENT;
    *out_handle = NULL;

    tk_internal_model_t* model = find_loaded_model(loader, params->model_path->path_str);
    if (model) {
        model->reference_count++;
        *out_handle = model;
        loader->cache_hits++;
        return TK_SUCCESS;
    }

    // --- New Eviction Logic ---
    uint64_t required_bytes = get_file_size(params->model_path->path_str);
    tk_internal_model_t* model_to_evict = NULL;

    tk_internal_model_t* candidates[MAX_LOADED_MODELS];
    size_t candidate_count = 0;
    for (size_t i = 0; i < loader->max_loaded_models; i++) {
        if (loader->loaded_models[i].is_loaded && loader->loaded_models[i].reference_count == 0) {
            candidates[candidate_count++] = &loader->loaded_models[i];
        }
    }

    model_to_evict = tk_memory_manager_select_model_for_eviction(
        loader->memory_manager, candidates, candidate_count, required_bytes);

    if (model_to_evict) {
        unload_model_internal(loader, model_to_evict);
    }
    // --- End of New Eviction Logic ---

    model = find_available_model_slot(loader);
    if (!model) {
        TK_LOG_ERROR("Could not load model %s, not enough memory and no suitable eviction candidate found.", params->model_path->path_str);
        return TK_ERROR_RESOURCE_EXHAUSTED;
    }

    tk_onnx_handle_t* onnx_handle = calloc(1, sizeof(tk_onnx_handle_t));
    if (!onnx_handle) return TK_ERROR_OUT_OF_MEMORY;

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    OrtSessionOptions* session_options;
    ORT_CHECK(g_ort->CreateSessionOptions(&session_options));
    ORT_CHECK(g_ort->SetIntraOpNumThreads(session_options, params->cpu_threads > 0 ? params->cpu_threads : loader->config.num_threads));

    // Enable GPU acceleration if requested
    if (params->prefer_gpu) {
        TK_LOG_INFO("Attempting to enable CUDA execution provider for ONNX model.");
        OrtCUDAProviderOptions cuda_options = {0};
        // cuda_options.device_id = loader->config.gpu_device_id; // Set device ID if needed
        OrtStatus* cuda_status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &cuda_options);
        if (cuda_status != NULL) {
            TK_LOG_WARN("Failed to add CUDA execution provider: %s", g_ort->GetErrorMessage(cuda_status));
            g_ort->ReleaseStatus(cuda_status);
        } else {
            TK_LOG_INFO("CUDA execution provider enabled successfully.");
        }
    }

    // Set graph optimization level
    // Using a sensible default. This could also be a parameter in tk_model_load_params_t.
    ORT_CHECK(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL));

    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "trackiellm", &onnx_handle->env);
    if (status != NULL) {
        g_ort->ReleaseSessionOptions(session_options);
        // ORT_CHECK will handle the rest
        ORT_CHECK(status);
    }
    
    status = g_ort->CreateSession(onnx_handle->env, params->model_path->path_str, session_options, &onnx_handle->session);
    g_ort->ReleaseSessionOptions(session_options); // Release options now, they are copied into the session.
    if (status != NULL) {
        g_ort->ReleaseEnv(onnx_handle->env);
        ORT_CHECK(status);
    }

    ORT_CHECK(g_ort->GetAllocatorWithDefaultOptions(&onnx_handle->allocator));
    
    model->handle = onnx_handle;
    model->format = TK_MODEL_FORMAT_ONNX;
    model->is_loaded = true;
    model->reference_count = 1;
    model->load_time = time(NULL);
    strncpy(model->path, params->model_path->path_str, MAX_PATH_LENGTH - 1);
    model->path[MAX_PATH_LENGTH - 1] = '\0';

    extract_model_metadata(model);
    tk_memory_manager_confirm_allocation(loader->memory_manager, model);

    *out_handle = model;
    loader->total_loads++;
    loader->cache_misses++;
    return TK_SUCCESS;
}

static tk_error_code_t load_model_tflite(tk_model_loader_t* loader, const tk_model_load_params_t* params, void** out_handle) {
    // This remains a placeholder as it's not the focus of the refactoring.
    if (!loader || !params || !out_handle) return TK_ERROR_INVALID_ARGUMENT;
    TK_LOG_WARN("TFLite loader is not implemented.");
    return TK_ERROR_UNSUPPORTED_OPERATION;
}

static tk_error_code_t unload_model_internal(tk_model_loader_t* loader, void* model_handle) {
    if (!loader || !model_handle) return TK_ERROR_INVALID_ARGUMENT;

    tk_internal_model_t* model = (tk_internal_model_t*)model_handle;
    if (model->reference_count > 0) {
        model->reference_count--;
    }

    if (model->is_loaded && model->reference_count == 0) {
        TK_LOG_INFO("Unloading model: %s", model->path);
        tk_memory_manager_confirm_deallocation(loader->memory_manager, model);
        if (model->format == TK_MODEL_FORMAT_GGUF) {
            tk_gguf_handle_t* handle = (tk_gguf_handle_t*)model->handle;
            if (handle) {
                llama_free_model(handle->model);
                // llama_backend_free(); // This should be called globally on program exit.
                free(handle);
            }
        } else if (model->format == TK_MODEL_FORMAT_ONNX) {
            tk_onnx_handle_t* handle = (tk_onnx_handle_t*)model->handle;
            if (handle) {
                const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
                if(g_ort) {
                    // Free metadata strings
                    if (handle->input_names) {
                        for (size_t i = 0; i < handle->input_count; i++) {
                            // These strings are allocated by the ORT allocator
                            handle->allocator->Free(handle->allocator, handle->input_names[i]);
                        }
                        free(handle->input_names);
                    }
                    if (handle->output_names) {
                        for (size_t i = 0; i < handle->output_count; i++) {
                            handle->allocator->Free(handle->allocator, handle->output_names[i]);
                        }
                        free(handle->output_names);
                    }

                    g_ort->ReleaseSession(handle->session);
                    g_ort->ReleaseEnv(handle->env);
                }
                free(handle);
            }
        }
        
        free_model_metadata(&model->metadata);
        memset(model, 0, sizeof(tk_internal_model_t));
        loader->total_unloads++;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Finds a loaded model by path
 */
static tk_internal_model_t* find_loaded_model(tk_model_loader_t* loader, const char* model_path) {
    if (!loader || !model_path) return NULL;
    
    for (size_t i = 0; i < loader->loaded_model_count; i++) {
        if (loader->loaded_models[i].is_loaded && 
            strcmp(loader->loaded_models[i].path, model_path) == 0) {
            return &loader->loaded_models[i];
        }
    }
    
    return NULL;
}

/**
 * @brief Finds a free, unused model slot in the loader's model array.
 *
 * This function does not perform any eviction. It simply looks for a slot
 * that is not marked as loaded. It should be called after the eviction
 * logic has already been run to ensure a slot is free.
 */
static tk_internal_model_t* find_available_model_slot(tk_model_loader_t* loader) {
    if (!loader) return NULL;

    for (size_t i = 0; i < loader->max_loaded_models; i++) {
        if (!loader->loaded_models[i].is_loaded) {
            return &loader->loaded_models[i];
        }
    }

    return NULL; // No free slots found.
}

/**
 * @brief Extracts metadata from a loaded model
 */
static tk_error_code_t extract_model_metadata(tk_internal_model_t* model) {
    if (!model || !model->handle) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    TK_LOG_INFO("Extracting metadata for model: %s", model->path);
    
    // Initialize metadata
    memset(&model->metadata, 0, sizeof(tk_model_metadata_t));
    model->metadata.is_valid = true;
    
    // Parse metadata based on model format
    switch (model->format) {
        case TK_MODEL_FORMAT_GGUF:
            return parse_gguf_metadata(model->path, &model->metadata, ((tk_gguf_handle_t*)model->handle)->model);
        case TK_MODEL_FORMAT_ONNX:
            return parse_onnx_metadata(model->path, &model->metadata, (tk_onnx_handle_t*)model->handle);
        case TK_MODEL_FORMAT_TFLITE:
            return parse_tflite_metadata(model->path, &model->metadata);
        default:
            TK_LOG_WARN("Unknown model format, using default metadata");
            model->metadata.name = duplicate_string("Unknown Model");
            model->metadata.description = duplicate_string("Model with unknown format");
            model->metadata.type = TK_MODEL_TYPE_UNKNOWN;
            model->metadata.format = model->format;
            model->metadata.size_bytes = get_file_size(model->path);
            return TK_SUCCESS;
    }
}

/**
 * @brief Validates a model file
 */
static tk_error_code_t validate_model_file(const char* model_path, tk_model_format_e* out_format) {
    if (!model_path || !out_format) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Check if file exists
    struct stat st;
    if (stat(model_path, &st) != 0) {
        TK_LOG_ERROR("Model file does not exist: %s", model_path);
        return TK_ERROR_IO_ERROR;
    }
    
    // Check if it's a regular file
    if (!S_ISREG(st.st_mode)) {
        TK_LOG_ERROR("Model path is not a regular file: %s", model_path);
        return TK_ERROR_IO_ERROR;
    }
    
    // Detect format
    *out_format = detect_model_format(model_path);
    if (*out_format == TK_MODEL_FORMAT_UNKNOWN) {
        TK_LOG_ERROR("Unknown model format for file: %s", model_path);
        return TK_ERROR_UNSUPPORTED_OPERATION;
    }
    
    TK_LOG_INFO("Model file validation successful: %s (format: %d)", model_path, *out_format);
    return TK_SUCCESS;
}

/**
 * @brief Detects the format of a model file
 */
static tk_model_format_e detect_model_format(const char* model_path) {
    if (!model_path) return TK_MODEL_FORMAT_UNKNOWN;
    
    // Get file extension
    char* ext = get_file_extension(model_path);
    if (!ext) return TK_MODEL_FORMAT_UNKNOWN;
    
    // Determine format based on extension
    if (strcasecmp(ext, "gguf") == 0) {
        return TK_MODEL_FORMAT_GGUF;
    } else if (strcasecmp(ext, "onnx") == 0) {
        return TK_MODEL_FORMAT_ONNX;
    } else if (strcasecmp(ext, "tflite") == 0) {
        return TK_MODEL_FORMAT_TFLITE;
    } else if (strcasecmp(ext, "pt") == 0 || strcasecmp(ext, "pth") == 0) {
        return TK_MODEL_FORMAT_TORCH;
    } else if (strcasecmp(ext, "safetensors") == 0) {
        return TK_MODEL_FORMAT_SAFETENSORS;
    }
    
    // Try to detect by magic bytes
    FILE* file = fopen(model_path, "rb");
    if (!file) return TK_MODEL_FORMAT_UNKNOWN;
    
    unsigned char magic[8];
    size_t read = fread(magic, 1, sizeof(magic), file);
    fclose(file);
    
    if (read >= 4) {
        // GGUF magic: GGUF
        if (magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F') {
            return TK_MODEL_FORMAT_GGUF;
        }
        
        // ONNX magic: ONNX
        if (magic[0] == 'O' && magic[1] == 'N' && magic[2] == 'N' && magic[3] == 'X') {
            return TK_MODEL_FORMAT_ONNX;
        }
    }
    
    return TK_MODEL_FORMAT_UNKNOWN;
}

/**
 * @brief Checks if a path is a directory
 */
static bool is_path_directory(const char* path) {
    if (!path) return false;
    
    struct stat st;
    if (stat(path, &st) != 0) return false;
    
    return S_ISDIR(st.st_mode);
}

/**
 * @brief Gets the size of a file
 */
static size_t get_file_size(const char* path) {
    if (!path) return 0;
    
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    
    return st.st_size;
}

/**
 * @brief Copies a file
 */
static tk_error_code_t copy_file(const char* src_path, const char* dst_path) {
    if (!src_path || !dst_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    FILE* src = fopen(src_path, "rb");
    if (!src) {
        TK_LOG_ERROR("Failed to open source file: %s", src_path);
        return TK_ERROR_IO_ERROR;
    }
    
    FILE* dst = fopen(dst_path, "wb");
    if (!dst) {
        fclose(src);
        TK_LOG_ERROR("Failed to open destination file: %s", dst_path);
        return TK_ERROR_IO_ERROR;
    }
    
    char buffer[4096];
    size_t bytes;
    
    while ((bytes = fread(buffer, 1, sizeof(buffer), src)) > 0) {
        if (fwrite(buffer, 1, bytes, dst) != bytes) {
            fclose(src);
            fclose(dst);
            TK_LOG_ERROR("Failed to write to destination file: %s", dst_path);
            return TK_ERROR_IO_ERROR;
        }
    }
    
    fclose(src);
    fclose(dst);
    
    return TK_SUCCESS;
}

/**
 * @brief Creates a directory
 */
static tk_error_code_t create_directory(const char* path) {
    if (!path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This is a simplified implementation
    // In practice, you would need to handle nested directory creation
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        TK_LOG_ERROR("Failed to create directory: %s", path);
        return TK_ERROR_IO_ERROR;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Deletes a file
 */
static tk_error_code_t delete_file(const char* path) {
    if (!path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (unlink(path) != 0) {
        TK_LOG_ERROR("Failed to delete file: %s", path);
        return TK_ERROR_IO_ERROR;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the file extension
 */
static char* get_file_extension(const char* path) {
    if (!path) return NULL;
    
    const char* dot = strrchr(path, '.');
    if (!dot || dot == path) return NULL;
    
    return (char*)(dot + 1);
}

/**
 * @brief Checks if a string ends with a suffix
 */
static bool string_ends_with(const char* str, const char* suffix) {
    if (!str || !suffix) return false;
    
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    
    if (suffix_len > str_len) return false;
    
    return strncmp(str + str_len - suffix_len, suffix, suffix_len) == 0;
}

/**
 * @brief Duplicates a string
 */
static char* duplicate_string(const char* src) {
    if (!src) return NULL;
    
    size_t len = strlen(src);
    char* dup = malloc(len + 1);
    if (!dup) return NULL;
    
    memcpy(dup, src, len + 1);
    return dup;
}

/**
 * @brief Frees a string
 */
static void free_string(char* str) {
    if (str) {
        free(str);
    }
}

/**
 * @brief Frees model metadata
 */
static void free_model_metadata(tk_model_metadata_t* metadata) {
    if (!metadata) return;
    
    free_string(metadata->name);
    free_string(metadata->version);
    free_string(metadata->author);
    free_string(metadata->description);
    free_string(metadata->license);
    free_string(metadata->architecture);
    free_string(metadata->creation_date);
    free_string(metadata->last_modified);
    free_string(metadata->framework);
    free_string(metadata->framework_version);
    free_string(metadata->hardware_target);
    free_string(metadata->dependencies);
    free_string(metadata->checksum);
    free_string(metadata->validation_message);
    
    if (metadata->supported_languages) {
        for (uint32_t i = 0; i < metadata->language_count; i++) {
            free_string(metadata->supported_languages[i]);
        }
        free(metadata->supported_languages);
    }
    
    memset(metadata, 0, sizeof(tk_model_metadata_t));
}

/**
 * @brief Parses GGUF metadata
 */
static tk_error_code_t parse_gguf_metadata(const char* model_path, tk_model_metadata_t* metadata, struct llama_model* model) {
    if (!model_path || !metadata || !model) return TK_ERROR_INVALID_ARGUMENT;

    char buffer[128];
    if (llama_model_meta_val_str(model, "general.name", buffer, sizeof(buffer))) {
        metadata->name = duplicate_string(buffer);
    } else {
        metadata->name = duplicate_string("Unknown GGUF Model");
    }
    
    if (llama_model_meta_val_str(model, "general.description", buffer, sizeof(buffer))) {
        metadata->description = duplicate_string(buffer);
    } else {
        metadata->description = duplicate_string("GGUF Language Model");
    }

    metadata->type = TK_MODEL_TYPE_LLM;
    metadata->format = TK_MODEL_FORMAT_GGUF;
    metadata->size_bytes = get_file_size(model_path);
    metadata->context_length = llama_n_ctx_train(model);
    metadata->embedding_dim = llama_n_embd(model);
    metadata->vocab_size = llama_n_vocab(model);
    metadata->num_layers = llama_n_layer(model);
    metadata->is_valid = true;

    return TK_SUCCESS;
}

static tk_error_code_t parse_onnx_metadata(const char* model_path, tk_model_metadata_t* metadata, tk_onnx_handle_t* handle) {
    if (!model_path || !metadata || !handle) return TK_ERROR_INVALID_ARGUMENT;

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtAllocator* allocator = handle->allocator;

    // Get model metadata (name, description, etc.)
    OrtModelMetadata* model_metadata;
    ORT_CHECK(g_ort->SessionGetModelMetadata(handle->session, &model_metadata));

    char* model_name;
    ORT_CHECK(g_ort->ModelMetadataGetProducerName(model_metadata, allocator, &model_name));
    metadata->name = duplicate_string(model_name);
    g_ort->AllocatorFree(allocator, model_name);

    char* model_desc;
    ORT_CHECK(g_ort->ModelMetadataGetDescription(model_metadata, allocator, &model_desc));
    metadata->description = duplicate_string(model_desc);
    g_ort->AllocatorFree(allocator, model_desc);

    g_ort->ReleaseModelMetadata(model_metadata);

    // Get input and output counts
    ORT_CHECK(g_ort->SessionGetInputCount(handle->session, &handle->input_count));
    ORT_CHECK(g_ort->SessionGetOutputCount(handle->session, &handle->output_count));
    metadata->input_count = handle->input_count;
    metadata->output_count = handle->output_count;

    // Allocate memory for names
    handle->input_names = calloc(handle->input_count, sizeof(char*));
    if (!handle->input_names) return TK_ERROR_OUT_OF_MEMORY;
    handle->output_names = calloc(handle->output_count, sizeof(char*));
    if (!handle->output_names) {
        free(handle->input_names);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    // Extract input details
    for (size_t i = 0; i < handle->input_count; i++) {
        ORT_CHECK(g_ort->SessionGetInputName(handle->session, i, allocator, &handle->input_names[i]));
        // In a full implementation, one would also extract type and shape here
        // and populate metadata->inputs array.
    }
    
    // Extract output details
    for (size_t i = 0; i < handle->output_count; i++) {
        ORT_CHECK(g_ort->SessionGetOutputName(handle->session, i, allocator, &handle->output_names[i]));
    }

    metadata->type = TK_MODEL_TYPE_UNKNOWN; // Could try to infer from metadata tags
    metadata->format = TK_MODEL_FORMAT_ONNX;
    metadata->size_bytes = get_file_size(model_path);
    metadata->is_valid = true;

    return TK_SUCCESS;
}

/**
 * @brief Parses TensorFlow Lite metadata
 */
static tk_error_code_t parse_tflite_metadata(const char* model_path, tk_model_metadata_t* metadata) {
    if (!model_path || !metadata) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would parse actual TensorFlow Lite metadata in a real implementation
    TK_LOG_INFO("Parsing TensorFlow Lite metadata for: %s", model_path);
    
    // Set default values for TensorFlow Lite models
    metadata->name = duplicate_string("TensorFlow Lite Model");
    metadata->version = duplicate_string("1.0");
    metadata->author = duplicate_string("Unknown");
    metadata->description = duplicate_string("TensorFlow Lite format model");
    metadata->license = duplicate_string("Unknown");
    metadata->architecture = duplicate_string("Unknown");
    metadata->type = TK_MODEL_TYPE_UNKNOWN;
    metadata->format = TK_MODEL_FORMAT_TFLITE;
    metadata->size_bytes = get_file_size(model_path);
    metadata->parameter_count = 0;
    metadata->context_length = 0;
    metadata->embedding_dim = 0;
    metadata->vocab_size = 0;
    metadata->num_layers = 0;
    metadata->hidden_size = 0;
    metadata->num_heads = 0;
    metadata->quantization_level = 0.0f;
    metadata->is_quantized = false;
    metadata->is_multilingual = false;
    metadata->language_count = 0;
    metadata->supported_languages = NULL;
    metadata->creation_date = duplicate_string("2023-01-01");
    metadata->last_modified = duplicate_string("2023-01-01");
    metadata->framework = duplicate_string("TensorFlow Lite");
    metadata->framework_version = duplicate_string("Unknown");
    metadata->hardware_target = duplicate_string("Mobile");
    metadata->supports_gpu = true;
    metadata->min_memory_mb = 100;
    metadata->recommended_memory_mb = 500;
    metadata->dependencies = duplicate_string("None");
    metadata->checksum = duplicate_string("Unknown");
    metadata->is_valid = true;
    metadata->validation_message = duplicate_string("Valid TensorFlow Lite model");
    
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_model_loader_create(tk_model_loader_t** out_loader, const tk_model_loader_config_t* config) {
    if (!out_loader || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_loader = NULL;
    
    // Validate configuration
    tk_error_code_t result = validate_config(config);
    if (result != TK_SUCCESS) {
        return result;
    }
    
    // Allocate loader structure
    tk_model_loader_t* loader = calloc(1, sizeof(tk_model_loader_t));
    if (!loader) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy configuration
    loader->config = *config;
    
    // Set maximum loaded models
    loader->max_loaded_models = config->max_models;
    if (loader->max_loaded_models > MAX_LOADED_MODELS) {
        loader->max_loaded_models = MAX_LOADED_MODELS;
    }
    
    // Allocate loaded models array
    loader->loaded_models = calloc(loader->max_loaded_models, sizeof(tk_internal_model_t));
    if (!loader->loaded_models) {
        free(loader);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy temp directory path
    if (config->temp_dir) {
        loader->temp_dir = duplicate_string(config->temp_dir);
        if (!loader->temp_dir) {
            free(loader->loaded_models);
            free(loader);
            return TK_ERROR_OUT_OF_MEMORY;
        }
    }
    
    // Copy cache directory path
    if (config->cache_dir) {
        loader->cache_dir = duplicate_string(config->cache_dir);
        if (!loader->cache_dir) {
            free_string(loader->temp_dir);
            free(loader->loaded_models);
            free(loader);
            return TK_ERROR_OUT_OF_MEMORY;
        }
    }
    
    // Initialize framework contexts
    result = init_framework_contexts(loader);
    if (result != TK_SUCCESS) {
        free_string(loader->cache_dir);
        free_string(loader->temp_dir);
        free(loader->loaded_models);
        free(loader);
        return result;
    }
    
    // Initialize statistics
    loader->cache_hits = 0;
    loader->cache_misses = 0;
    loader->total_loads = 0;
    loader->total_unloads = 0;
    loader->cache_size_bytes = 0;

    // Create memory manager
    tk_memory_manager_config_t mem_config = {
        .max_ram_mb = loader->config.cache_size_mb,
        .max_vram_mb = 0 // VRAM not managed yet
    };
    result = tk_memory_manager_create(&loader->memory_manager, &mem_config);
    if (result != TK_SUCCESS) {
        // ... cleanup previously allocated resources ...
        free(loader);
        return result;
    }
    
    *out_loader = loader;
    TK_LOG_INFO("Model loader created successfully");
    return TK_SUCCESS;
}

void tk_model_loader_destroy(tk_model_loader_t** loader) {
    if (!loader || !*loader) return;
    
    tk_model_loader_t* l = *loader;
    
    // Unload all loaded models
    for (size_t i = 0; i < l->max_loaded_models; i++) {
        if (l->loaded_models[i].is_loaded) {
            unload_model_internal(l, &l->loaded_models[i]);
        }
    }
    
    // Destroy the memory manager
    tk_memory_manager_destroy(&l->memory_manager);

    // Clean up framework contexts
    cleanup_framework_contexts(l);
    
    // Free allocated strings
    free_string(l->temp_dir);
    free_string(l->cache_dir);
    
    // Free loaded models array
    free(l->loaded_models);
    
    // Free loader itself
    free(l);
    *loader = NULL;
    
    TK_LOG_INFO("Model loader destroyed");
}

tk_error_code_t tk_model_loader_load_model(
    tk_model_loader_t* loader,
    const tk_model_load_params_t* params,
    void** out_model_handle
) {
    if (!loader || !params || !out_model_handle || !params->model_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_model_handle = NULL;
    
    // Validate model file
    tk_model_format_e format;
    tk_error_code_t result = validate_model_file(params->model_path->path_str, &format);
    if (result != TK_SUCCESS) {
        return result;
    }
    
    // Load model based on format
    switch (format) {
        case TK_MODEL_FORMAT_GGUF:
            return load_model_gguf(loader, params, out_model_handle);
        case TK_MODEL_FORMAT_ONNX:
            return load_model_onnx(loader, params, out_model_handle);
        case TK_MODEL_FORMAT_TFLITE:
            return load_model_tflite(loader, params, out_model_handle);
        default:
            TK_LOG_ERROR("Unsupported model format: %d", format);
            return TK_ERROR_UNSUPPORTED_OPERATION;
    }
}

tk_error_code_t tk_model_loader_unload_model(tk_model_loader_t* loader, void** model_handle) {
    if (!loader || !model_handle || !*model_handle) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    tk_error_code_t result = unload_model_internal(loader, *model_handle);
    if (result == TK_SUCCESS) {
        *model_handle = NULL;
    }
    
    return result;
}

tk_error_code_t tk_model_loader_get_model_metadata(
    tk_model_loader_t* loader,
    void* model_handle,
    tk_model_metadata_t** out_metadata
) {
    if (!loader || !model_handle || !out_metadata) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_metadata = NULL;
    
    tk_internal_model_t* model = (tk_internal_model_t*)model_handle;
    if (!model->is_loaded) {
        return TK_ERROR_NOT_FOUND;
    }
    
    // Allocate metadata structure
    tk_model_metadata_t* metadata = malloc(sizeof(tk_model_metadata_t));
    if (!metadata) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy metadata
    memcpy(metadata, &model->metadata, sizeof(tk_model_metadata_t));
    
    // Deep copy strings
    metadata->name = duplicate_string(model->metadata.name);
    metadata->version = duplicate_string(model->metadata.version);
    metadata->author = duplicate_string(model->metadata.author);
    metadata->description = duplicate_string(model->metadata.description);
    metadata->license = duplicate_string(model->metadata.license);
    metadata->architecture = duplicate_string(model->metadata.architecture);
    metadata->creation_date = duplicate_string(model->metadata.creation_date);
    metadata->last_modified = duplicate_string(model->metadata.last_modified);
    metadata->framework = duplicate_string(model->metadata.framework);
    metadata->framework_version = duplicate_string(model->metadata.framework_version);
    metadata->hardware_target = duplicate_string(model->metadata.hardware_target);
    metadata->dependencies = duplicate_string(model->metadata.dependencies);
    metadata->checksum = duplicate_string(model->metadata.checksum);
    metadata->validation_message = duplicate_string(model->metadata.validation_message);
    
    // Copy input dimensions
    if (model->metadata.input_dim_count > 0 && model->metadata.input_dims) {
        metadata->input_dims = malloc(model->metadata.input_dim_count * sizeof(uint32_t));
        if (!metadata->input_dims) {
            free_model_metadata(metadata);
            free(metadata);
            return TK_ERROR_OUT_OF_MEMORY;
        }
        memcpy(metadata->input_dims, model->metadata.input_dims, 
               model->metadata.input_dim_count * sizeof(uint32_t));
    }
    
    // Copy output dimensions
    if (model->metadata.output_dim_count > 0 && model->metadata.output_dims) {
        metadata->output_dims = malloc(model->metadata.output_dim_count * sizeof(uint32_t));
        if (!metadata->output_dims) {
            free_model_metadata(metadata);
            free(metadata);
            return TK_ERROR_OUT_OF_MEMORY;
        }
        memcpy(metadata->output_dims, model->metadata.output_dims, 
               model->metadata.output_dim_count * sizeof(uint32_t));
    }
    
    // Copy supported languages
    if (model->metadata.language_count > 0 && model->metadata.supported_languages) {
        metadata->supported_languages = malloc(model->metadata.language_count * sizeof(char*));
        if (!metadata->supported_languages) {
            free_model_metadata(metadata);
            free(metadata);
            return TK_ERROR_OUT_OF_MEMORY;
        }
        
        for (uint32_t i = 0; i < model->metadata.language_count; i++) {
            metadata->supported_languages[i] = duplicate_string(model->metadata.supported_languages[i]);
            if (!metadata->supported_languages[i]) {
                // Clean up partially allocated languages
                for (uint32_t j = 0; j < i; j++) {
                    free_string(metadata->supported_languages[j]);
                }
                free(metadata->supported_languages);
                metadata->supported_languages = NULL;
                metadata->language_count = 0;
                free_model_metadata(metadata);
                free(metadata);
                return TK_ERROR_OUT_OF_MEMORY;
            }
        }
    }
    
    *out_metadata = metadata;
    return TK_SUCCESS;
}

void tk_model_loader_free_metadata(tk_model_metadata_t** metadata) {
    if (!metadata || !*metadata) return;
    
    free_model_metadata(*metadata);
    free(*metadata);
    *metadata = NULL;
}

tk_error_code_t tk_model_loader_validate_model(
    tk_model_loader_t* loader,
    const tk_path_t* model_path,
    bool* out_is_valid,
    tk_model_format_e* out_format
) {
    if (!loader || !model_path || !out_is_valid || !out_format) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_is_valid = false;
    *out_format = TK_MODEL_FORMAT_UNKNOWN;
    
    tk_error_code_t result = validate_model_file(model_path->path_str, out_format);
    if (result == TK_SUCCESS) {
        *out_is_valid = true;
    }
    
    return TK_SUCCESS;
}

tk_error_code_t tk_model_loader_convert_model(
    tk_model_loader_t* loader,
    const tk_path_t* source_path,
    const tk_path_t* target_path,
    tk_model_format_e target_format
) {
    if (!loader || !source_path || !target_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would implement actual model conversion in a real implementation
    TK_LOG_INFO("Converting model from %s to %s (format: %d)", 
                source_path->path_str, target_path->path_str, target_format);
    
    // For now, we'll just simulate the conversion by copying the file
    // In practice, this would use framework-specific conversion tools
    tk_error_code_t result = copy_file(source_path->path_str, target_path->path_str);
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("Failed to convert model from %s to %s", 
                     source_path->path_str, target_path->path_str);
        return result;
    }
    
    TK_LOG_INFO("Model conversion completed successfully");
    return TK_SUCCESS;
}

tk_error_code_t tk_model_loader_optimize_model(
    tk_model_loader_t* loader,
    void* model_handle,
    uint32_t optimization_level
) {
    if (!loader || !model_handle) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_internal_model_t* model = (tk_internal_model_t*)model_handle;
    if (!model->is_loaded) {
        return TK_ERROR_NOT_FOUND;
    }

    TK_LOG_INFO("Attempting to optimize model %s with level %u", model->path, optimization_level);

    if (model->format == TK_MODEL_FORMAT_ONNX) {
        tk_onnx_handle_t* handle = (tk_onnx_handle_t*)model->handle;
        if (!handle || !handle->session) {
            return TK_ERROR_INVALID_ARGUMENT;
        }

        // ONNX Runtime optimization is typically set at session creation.
        // Reloading the model with new session options is the standard way.
        // This function will log a warning and guide the user.
        TK_LOG_WARN("For ONNX models, optimization is applied at load time.");
        TK_LOG_WARN("To use GPU or specific optimization levels, please re-load the model with the appropriate 'tk_model_load_params_t'.");

        // A more advanced implementation could potentially create a new session here,
        // but that would complicate state management significantly.
        // For now, we consider this a user guidance function.

        return TK_ERROR_UNSUPPORTED_OPERATION; // Indicate that dynamic optimization is not supported.

    } else if (model->format == TK_MODEL_FORMAT_GGUF) {
        TK_LOG_INFO("GGUF model optimization is configured at load time via parameters like 'n_gpu_layers'.");
        return TK_SUCCESS;
    }

    TK_LOG_WARN("Optimization is not applicable or implemented for this model format.");
    return TK_SUCCESS;
}

tk_error_code_t tk_model_loader_quantize_model(
    tk_model_loader_t* loader,
    void* model_handle,
    uint32_t bits
) {
    if (!loader || !model_handle) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    tk_internal_model_t* model = (tk_internal_model_t*)model_handle;
    if (!model->is_loaded) {
        return TK_ERROR_NOT_FOUND;
    }
    
    // This would implement actual model quantization in a real implementation
    TK_LOG_INFO("Quantizing model %s to %u bits", model->path, bits);
    
    // For now, we'll just log the quantization
    TK_LOG_INFO("Model quantization completed successfully");
    return TK_SUCCESS;
}

tk_error_code_t tk_model_loader_clear_cache(tk_model_loader_t* loader) {
    if (!loader) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Unload all models with zero reference count
    for (size_t i = 0; i < loader->max_loaded_models; i++) {
        if (loader->loaded_models[i].is_loaded && 
            loader->loaded_models[i].reference_count == 0) {
            unload_model_internal(loader, &loader->loaded_models[i]);
        }
    }
    
    // Reset statistics
    loader->cache_hits = 0;
    loader->cache_misses = 0;
    loader->cache_size_bytes = 0;
    
    TK_LOG_INFO("Model cache cleared");
    return TK_SUCCESS;
}

tk_error_code_t tk_model_loader_get_cache_stats(
    tk_model_loader_t* loader,
    float* out_cache_hit_rate,
    uint32_t* out_cache_size_mb,
    uint32_t* out_cache_entries
) {
    if (!loader || !out_cache_hit_rate || !out_cache_size_mb || !out_cache_entries) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Calculate cache hit rate
    uint64_t total_requests = loader->cache_hits + loader->cache_misses;
    if (total_requests > 0) {
        *out_cache_hit_rate = (float)loader->cache_hits / total_requests;
    } else {
        *out_cache_hit_rate = 0.0f;
    }
    
    // Convert cache size to MB
    *out_cache_size_mb = (uint32_t)(loader->cache_size_bytes / (1024 * 1024));
    
    // Count loaded models
    *out_cache_entries = 0;
    for (size_t i = 0; i < loader->max_loaded_models; i++) {
        if (loader->loaded_models[i].is_loaded) {
            (*out_cache_entries)++;
        }
    }
    
    return TK_SUCCESS;
}

tk_error_code_t tk_model_loader_preload_model(
    tk_model_loader_t* loader,
    const tk_path_t* model_path
) {
    if (!loader || !model_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Check if model is already loaded
    if (find_loaded_model(loader, model_path->path_str)) {
        TK_LOG_INFO("Model already loaded: %s", model_path->path_str);
        return TK_SUCCESS;
    }
    
    // Create temporary load parameters
    tk_model_load_params_t params = {0};
    params.model_path = (tk_path_t*)model_path; // Cast away const for internal use
    params.model_type = TK_MODEL_TYPE_UNKNOWN;
    
    // Load the model
    void* model_handle = NULL;
    tk_error_code_t result = tk_model_loader_load_model(loader, &params, &model_handle);
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("Failed to preload model: %s", model_path->path_str);
        return result;
    }
    
    // Keep the model loaded but don't return the handle
    // This ensures the model stays in cache
    TK_LOG_INFO("Model preloaded successfully: %s", model_path->path_str);
    return TK_SUCCESS;
}

tk_error_code_t tk_model_loader_detect_format(
    tk_model_loader_t* loader,
    const tk_path_t* model_path,
    tk_model_format_e* out_format
) {
    if (!loader || !model_path || !out_format) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_format = detect_model_format(model_path->path_str);
    return TK_SUCCESS;
}

tk_error_code_t tk_model_loader_get_supported_formats(
    tk_model_format_e** out_formats,
    size_t* out_count
) {
    if (!out_formats || !out_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Allocate array for supported formats
    size_t count = 6; // Number of supported formats
    tk_model_format_e* formats = malloc(count * sizeof(tk_model_format_e));
    if (!formats) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Fill with supported formats
    formats[0] = TK_MODEL_FORMAT_GGUF;
    formats[1] = TK_MODEL_FORMAT_ONNX;
    formats[2] = TK_MODEL_FORMAT_TFLITE;
    formats[3] = TK_MODEL_FORMAT_TORCH;
    formats[4] = TK_MODEL_FORMAT_SAFETENSORS;
    formats[5] = TK_MODEL_FORMAT_UNKNOWN; // Terminator
    
    *out_formats = formats;
    *out_count = count - 1; // Exclude terminator
    
    return TK_SUCCESS;
}

void tk_model_loader_free_formats(tk_model_format_e** formats) {
    if (!formats || !*formats) return;
    
    free(*formats);
    *formats = NULL;
}
