/*
 * Copyright (C) 2025 TrackieWay-OSS
 *
 * This file is part of TrackieLLM.
 *
 * TrackieLLM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TrackieLLM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with TrackieLLM. If not, see <https://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include "tk_onnx_runner.h"
#include "tk_model_loader_private.h" // Access internal model structure
#include "utils/tk_logging.h"
#include <stdlib.h>
#include <string.h>

// Helper macro for checking ONNX Runtime calls
#define ORT_RUNNER_CHECK(expr) { \
    OrtStatus* s = (expr); \
    if (s) { \
        const OrtApi* g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION); \
        TK_LOG_ERROR("ONNX Runtime error in runner: %s", g_ort_api->GetErrorMessage(s)); \
        g_ort_api->ReleaseStatus(s); \
        return TK_ERROR_INFERENCE_FAILED; \
    } \
}

struct tk_onnx_runner_s {
    OrtSession* session;
    OrtAllocator* allocator;
    size_t input_count;
    char** input_names;
    size_t output_count;
    char** output_names;
};

tk_error_code_t tk_onnx_runner_create(tk_onnx_runner_t** out_runner, void* model_handle) {
    if (!out_runner || !model_handle) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    *out_runner = NULL;

    tk_internal_model_t* internal_model = (tk_internal_model_t*)model_handle;
    if (internal_model->format != TK_MODEL_FORMAT_ONNX) {
        TK_LOG_ERROR("Model handle is not for an ONNX model.");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_onnx_handle_t* onnx_handle = (tk_onnx_handle_t*)internal_model->handle;
    if (!onnx_handle || !onnx_handle->session) {
        TK_LOG_ERROR("Invalid ONNX handle or session in the provided model.");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_onnx_runner_t* runner = calloc(1, sizeof(tk_onnx_runner_t));
    if (!runner) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    runner->session = onnx_handle->session; // We will use the session from the loader
    runner->allocator = onnx_handle->allocator;
    runner->input_count = onnx_handle->input_count;
    runner->output_count = onnx_handle->output_count;

    // The model loader should have already extracted these names.
    // We just need to copy the pointers. The loader owns the memory.
    // The runner borrows these pointers. The model loader owns them.
    // The Rust wrapper MUST ensure the Model outlives the OnnxRunner.
    runner->input_names = onnx_handle->input_names;
    runner->output_names = onnx_handle->output_names;

    *out_runner = runner;
    TK_LOG_INFO("ONNX runner created successfully.");
    return TK_SUCCESS;
}

void tk_onnx_runner_destroy(tk_onnx_runner_t** runner) {
    if (!runner || !*runner) {
        return;
    }

    // The session and other resources are owned by the tk_model_loader.
    // The runner only holds references, so we just free the runner struct itself.
    free(*runner);
    *runner = NULL;
    TK_LOG_INFO("ONNX runner destroyed.");
}

static size_t get_element_size_bytes(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return sizeof(float);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return sizeof(uint8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return sizeof(int8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return sizeof(int16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return sizeof(int32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return sizeof(int64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return sizeof(bool);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2; // __fp16
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return sizeof(double);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return sizeof(uint32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return sizeof(uint64_t);
        // String, Complex types are not handled here as they are more complex.
        default: return 0;
    }
}

tk_error_code_t tk_tensor_create_from_raw(
    tk_tensor_t** out_tensor,
    void* data,
    const int64_t* shape,
    size_t shape_len,
    ONNXTensorElementDataType type
) {
    if (!out_tensor) return TK_ERROR_INVALID_ARGUMENT;
    *out_tensor = NULL;

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtAllocator* allocator;
    ORT_RUNNER_CHECK(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    size_t element_count = 1;
    for(size_t i = 0; i < shape_len; ++i) element_count *= shape[i];

    size_t element_size = get_element_size_bytes(type);
    if (element_size == 0) {
        TK_LOG_ERROR("Unsupported or invalid tensor element data type: %d", type);
        return TK_ERROR_INVALID_ARGUMENT;
    }
    size_t data_byte_len = element_count * element_size;

    OrtMemoryInfo* memory_info;
    ORT_RUNNER_CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    OrtValue* ort_value = NULL;
    ORT_RUNNER_CHECK(g_ort->CreateTensorWithDataAsOrtValue(memory_info, data, data_byte_len, shape, shape_len, type, &ort_value));
    g_ort->ReleaseMemoryInfo(memory_info);

    tk_tensor_t* tensor = calloc(1, sizeof(tk_tensor_t));
    if (!tensor) {
        g_ort->ReleaseValue(ort_value);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    tensor->ort_value = ort_value;
    *out_tensor = tensor;

    return TK_SUCCESS;
}

void tk_tensor_destroy(tk_tensor_t** tensor) {
    if (!tensor || !*tensor) return;
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if ((*tensor)->ort_value) {
        g_ort->ReleaseValue((*tensor)->ort_value);
    }
    free(*tensor);
    *tensor = NULL;
}

tk_error_code_t tk_onnx_runner_run(
    tk_onnx_runner_t* runner,
    const tk_tensor_t* const* inputs,
    size_t input_count,
    tk_tensor_t*** outputs,
    size_t* output_count
) {
    if (!runner || !inputs || !outputs || !output_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    if (input_count != runner->input_count) {
        TK_LOG_ERROR("Mismatched input count: model expects %zu, but got %zu.", runner->input_count, input_count);
        return TK_ERROR_INVALID_ARGUMENT;
    }
    *outputs = NULL;
    *output_count = 0;

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // Prepare input OrtValue pointers
    OrtValue* input_ort_values[input_count];
    for (size_t i = 0; i < input_count; ++i) {
        input_ort_values[i] = inputs[i]->ort_value;
    }

    // Prepare output OrtValue pointers
    OrtValue* output_ort_values[runner->output_count];
    for (size_t i = 0; i < runner->output_count; ++i) {
        output_ort_values[i] = NULL;
    }

    ORT_RUNNER_CHECK(g_ort->Run(
        runner->session,
        NULL, // Default run options
        (const char* const*)runner->input_names,
        (const OrtValue* const*)input_ort_values,
        runner->input_count,
        (const char* const*)runner->output_names,
        runner->output_count,
        output_ort_values
    ));

    // Allocate and wrap the output tensors
    tk_tensor_t** out_tensors = calloc(runner->output_count, sizeof(tk_tensor_t*));
    if (!out_tensors) {
        // Cleanup allocated OrtValues
        for (size_t i = 0; i < runner->output_count; ++i) {
            if (output_ort_values[i]) g_ort->ReleaseValue(output_ort_values[i]);
        }
        return TK_ERROR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < runner->output_count; ++i) {
        out_tensors[i] = calloc(1, sizeof(tk_tensor_t));
        if (!out_tensors[i]) {
            // Cleanup everything allocated so far
            for(size_t j = 0; j < i; ++j) {
                g_ort->ReleaseValue(out_tensors[j]->ort_value);
                free(out_tensors[j]);
            }
            free(out_tensors);
            for (size_t k = 0; k < runner->output_count; ++k) {
                if (output_ort_values[k]) g_ort->ReleaseValue(output_ort_values[k]);
            }
            return TK_ERROR_OUT_OF_MEMORY;
        }
        out_tensors[i]->ort_value = output_ort_values[i];
    }

    *outputs = out_tensors;
    *output_count = runner->output_count;

    return TK_SUCCESS;
}

void tk_onnx_runner_free_outputs(tk_tensor_t** outputs, size_t output_count) {
    // This function is now only responsible for freeing the array of pointers,
    // as the Tensors themselves are owned by the Rust `Tensor` struct which will
    // call tk_tensor_destroy on each one.
    if (!outputs) return;
    free(outputs);
}