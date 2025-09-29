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

#ifndef TK_ONNX_RUNNER_H
#define TK_ONNX_RUNNER_H

#include "utils/tk_error_handling.h"
#include "tk_model_loader.h" // For tk_internal_model_t
#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle for an ONNX runner.
 */
typedef struct tk_onnx_runner_s tk_onnx_runner_t;

/**
 * @brief Represents a tensor, wrapping an OrtValue.
 * This structure allows passing tensor data across the FFI boundary.
 */
typedef struct tk_tensor_s {
    OrtValue* ort_value;
} tk_tensor_t;

/**
 * @brief Creates a new ONNX runner instance.
 *
 * The runner is created from a model that has already been loaded by the
 * tk_model_loader. This function extracts the ONNX session and prepares for
 * inference.
 *
 * @param[out] out_runner Pointer to receive the created runner handle.
 * @param[in] model_handle A handle to the loaded model obtained from `tk_model_loader_load_model`.
 *                         This handle must point to a valid ONNX model.
 * @return `TK_SUCCESS` on success, or an error code otherwise.
 */
tk_error_code_t tk_onnx_runner_create(tk_onnx_runner_t** out_runner, void* model_handle);

/**
 * @brief Destroys an ONNX runner instance and frees its resources.
 *
 * @param[in,out] runner Pointer to the runner handle to be destroyed.
 *                       The pointer will be set to NULL after destruction.
 */
void tk_onnx_runner_destroy(tk_onnx_runner_t** runner);

/**
 * @brief Runs inference on the ONNX model.
 *
 * @param[in] runner The ONNX runner instance.
 * @param[in] inputs An array of pointers to input tensors (`tk_tensor_t`).
 * @param[in] input_count The number of input tensors.
 * @param[out] outputs A pointer to receive an array of output tensor pointers.
 *                     The caller is responsible for freeing this array and the
 *                     tensors within it using `tk_onnx_runner_free_outputs`.
 * @param[out] output_count A pointer to receive the number of output tensors.
 * @return `TK_SUCCESS` on success, or an error code if inference fails.
 */
tk_error_code_t tk_onnx_runner_run(
    tk_onnx_runner_t* runner,
    const tk_tensor_t* const* inputs,
    size_t input_count,
    tk_tensor_t*** outputs,
    size_t* output_count
);

/**
 * @brief Creates a new tensor from raw data for input into the model.
 *
 * @param[out] out_tensor Pointer to receive the created tensor handle.
 * @param[in] data Pointer to the raw data buffer.
 * @param[in] shape An array of integers representing the tensor's shape.
 * @param[in] shape_len The number of dimensions in the shape array.
 * @param[in] type The ONNX data type of the tensor.
 * @return `TK_SUCCESS` on success.
 */
tk_error_code_t tk_tensor_create_from_raw(
    tk_tensor_t** out_tensor,
    void* data,
    const int64_t* shape,
    size_t shape_len,
    ONNXTensorElementDataType type
);

/**
 * @brief Destroys a tensor instance.
 *
 * @param[in,out] tensor Pointer to the tensor handle to be destroyed.
 */
void tk_tensor_destroy(tk_tensor_t** tensor);

/**
 * @brief Frees the memory allocated for output tensors from a run.
 *
 * @param[in] outputs The array of output tensors to free.
 * @param[in] output_count The number of tensors in the array.
 */
void tk_onnx_runner_free_outputs(tk_tensor_t** outputs, size_t output_count);

#ifdef __cplusplus
}
#endif

#endif // TK_ONNX_RUNNER_H