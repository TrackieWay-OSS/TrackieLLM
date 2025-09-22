#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: src/ai_models/CMakeLists.txt
#
# This script builds the C-language components of the ai_models module.
# It compiles the model loader and runner into a static library, which is then
# linked against by the Rust FFI wrapper.
#
# SPDX-License-Identifier: AGPL-3.0 license
#

# Define the static library target for the C components of ai_models.
add_library(trackie_ai_models_c STATIC
    tk_model_loader.c
    tk_runner_lifecycle.c
    tk_runner_streaming.c
    tk_runner_helpers.c
)

# The C code needs to be able to find the llama.h header.
# The llama.cpp project is located at ../llama.cpp from this directory.
target_include_directories(trackie_ai_models_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/llama.cpp/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/llama.cpp/ggml/include>
)

# Link our C library against the llama.cpp library.
# We assume that the root CMakeLists.txt has already added the llama.cpp
# subdirectory and that it creates a target named 'llama'.
target_link_libraries(trackie_ai_models_c PRIVATE llama)

# Link against ONNX Runtime if it has been found by the root script.
# We assume the root script creates an imported target for it.
if(ONNX_RUNTIME_FOUND)
    target_link_libraries(trackie_ai_models_c PRIVATE ONNX_RUNTIME::onnxruntime)
endif()

# This is a common pattern to ensure other targets that link against this
# one automatically get the correct include directories.
target_include_directories(trackie_ai_models_c PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
