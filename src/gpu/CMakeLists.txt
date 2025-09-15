#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: src/gpu/CMakeLists.txt
#
# This script builds the individual GPU accelerator libraries for the TrackieLLM
# project. It conditionally compiles each library based on the options selected
# in the root CMakeLists.txt file (e.g., TK_ENABLE_CUDA).
#
# The libraries are named according to the convention:
# tlib<type>_<spec>-<version>.bin
# where <type> is 'acc' (AI) or 'acg' (Graphics), and <spec> is a short
# identifier like 'cd' (CUDA) or 'vlk' (Vulkan).
#
# SPDX-License-Identifier: AGPL-3.0 license
#

# --- Helper Macro for setting library properties ---
macro(set_binary_properties target_name)
    set_target_properties(${target_name} PROPERTIES
        PREFIX "tlib"
        SUFFIX ".bin"
    )
    # Set the output name explicitly to match the user's request format
    # e.g., tlibacc_cd-v1.0.0.bin
    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME "${target_name}"
    )
endmacro()


#--------------------------------------------------------------------------------
# AI Accelerators (acc)
#--------------------------------------------------------------------------------

# CUDA (NVIDIA)
if(TK_ENABLE_CUDA)
    set(target_name "acc_cd-v${PROJECT_VERSION}")
    add_library(${target_name} SHARED
        cuda/tk_cuda_dispatch.cu
        cuda/tk_cuda_kernels.cu
        cuda/tk_cuda_math_helpers.cu
        extensions/cuda/tk_cuda_image_ops.cu
        extensions/cuda/tk_cuda_tensor_ops.cu
    )
    target_link_libraries(${target_name} PRIVATE ${CUDA_LIBRARIES})
    set_binary_properties(${target_name})
    message(STATUS "Configured AI Accelerator: tlib${target_name}.bin")
endif()

# ROCm (AMD)
if(TK_ENABLE_ROCM)
    set(target_name "acc_ro-v${PROJECT_VERSION}")
    add_library(${target_name} SHARED
        rocm/tk_rocm_dispatch.cpp
        rocm/tk_rocm_kernels.cpp
        extensions/rocm/tk_rocm_image_ops.cpp
        extensions/rocm/tk_rocm_tensor_ops.cpp
    )
    set_binary_properties(${target_name})
    message(STATUS "Configured AI Accelerator: tlib${target_name}.bin")
endif()

# Android NNAPI
if(TK_ENABLE_NNAPI AND ANDROID)
    set(target_name "acc_nn-v${PROJECT_VERSION}")
    add_library(${target_name} SHARED
        nnapi/tk_nnapi_dispatch.cpp
    )
    target_link_libraries(${target_name} PRIVATE ${NEURALNETWORKS_LIBRARY})
    set_binary_properties(${target_name})
    message(STATUS "Configured AI Accelerator: tlib${target_name}.bin")
endif()


#--------------------------------------------------------------------------------
# Graphics Accelerators (acg) and Unified Targets
#--------------------------------------------------------------------------------

# Metal (Apple) - Unified Compute and Graphics
if(TK_ENABLE_METAL)
    set(target_name "acc_mt-v${PROJECT_VERSION}") # Naming it as an ACC, as it's a primary role
    add_library(${target_name} SHARED
        metal/tk_metal_dispatch.mm
        metal/tk_metal_helpers.mm
        extensions/metal/tk_metal_image_ops.mm
        extensions/metal/tk_metal_tensor_ops.mm
    )
    # Link against all required frameworks for both Compute and Graphics
    target_link_libraries(${target_name} PRIVATE "-framework Metal" "-framework Foundation" "-framework QuartzCore")
    target_sources(${target_name} PRIVATE
        metal/tk_metal_kernels.metal
        extensions/metal/tk_metal_extension_kernels.metal
    )
    set_binary_properties(${target_name})
    message(STATUS "Configured Unified Accelerator (ACC/ACG): tlib${target_name}.bin")
endif()

# Vulkan
if(TK_ENABLE_VULKAN)
    set(target_name "acg_vlk-v${PROJECT_VERSION}")
    add_library(${target_name} SHARED
        vulkan/tk_vulkan_dispatch.cpp
        vulkan/tk_vulkan_helpers.cpp
        vulkan/tk_vulkan_kernels.cpp
    )
    target_link_libraries(${target_name} PRIVATE Vulkan::Vulkan)
    set_binary_properties(${target_name})
    message(STATUS "Configured Graphics Accelerator: tlib${target_name}.bin")
endif()

# OpenGL ES
if(TK_ENABLE_GLES)
    set(target_name "acg_gles-v${PROJECT_VERSION}")
    add_library(${target_name} SHARED
        gles/tk_gles_dispatch.cpp
        gles/tk_gles_helpers.cpp
        gles/tk_gles_kernels.cpp
    )
    target_link_libraries(${target_name} PRIVATE ${EGL_LIBRARIES} ${GLESv3_LIBRARIES})
    set_binary_properties(${target_name})
    message(STATUS "Configured Graphics Accelerator: tlib${target_name}.bin")
endif()

# OpenCL
if(TK_ENABLE_OPENCL)
    set(target_name "acg_ocl-v${PROJECT_VERSION}")
    add_library(${target_name} SHARED
        opencl/tk_opencl_dispatch.cpp
        opencl/tk_opencl_helpers.cpp
        opencl/tk_opencl_kernels.cpp
    )
    target_link_libraries(${target_name} PRIVATE OpenCL::OpenCL)
    set_binary_properties(${target_name})
    message(STATUS "Configured Graphics Accelerator: tlib${target_name}.bin")
endif()
