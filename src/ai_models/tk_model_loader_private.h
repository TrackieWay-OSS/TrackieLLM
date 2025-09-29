/*
 * Copyright (C) 2025 TrackieWay-OSS
 *
 * This file is part of TrackieLLM.
 *
 * This is a private header for the AI Models C library. It should not be
 * included by external applications. It exposes internal data structures
 * needed by different implementation files within the library.
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef TK_MODEL_LOADER_PRIVATE_H
#define TK_MODEL_LOADER_PRIVATE_H

#include "tk_model_loader.h"
#include "onnxruntime_c_api.h"
#include "llama.h"
#include <time.h>
#include <stdbool.h>

#define MAX_PATH_LENGTH 4096

// Estrutura para o handle do modelo ONNX
typedef struct {
    OrtEnv* env;
    OrtSession* session;
    OrtAllocator* allocator;
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
} tk_gguf_handle_t;

// Estrutura interna principal que representa um modelo carregado
typedef struct {
    void* handle;
    tk_model_metadata_t metadata;
    tk_model_format_e format;
    char path[MAX_PATH_LENGTH];
    time_t load_time;
    uint32_t reference_count;
    bool is_loaded;
} tk_internal_model_t;


#endif // TK_MODEL_LOADER_PRIVATE_H