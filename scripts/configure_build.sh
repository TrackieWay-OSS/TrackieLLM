#!/bin/bash

#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: scripts/configure_build.sh
#
# This script provides a text-based user interface (TUI) to interactively
# configure and build the TrackieLLM project. It allows developers to easily
# select which AI and Graphics accelerator backends to include in the build
# without manually setting CMake flags.
#
# SPDX-License-Identifier: AGPL-3.0 license
#

# --- Helper Functions ---
function print_header() {
    echo "================================================="
    echo "  TrackieLLM Interactive Build Configurator"
    echo "================================================="
    echo
}

function print_menu() {
    local title="$1"
    shift
    local options=("$@")
    echo "### $title ###"
    for i in "${!options[@]}"; do
        echo "[$((i+1))] ${options[$i]}"
    done
    echo "[0] None"
}

# --- Main Script ---
print_header

# --- AI Accelerator Selection ---
acc_options=("CUDA (NVIDIA)" "ROCm (AMD)" "Metal (Apple)" "NNAPI (Android)")
print_menu "AI Accelerators (acc)" "${acc_options[@]}"
read -p "Select AI accelerator: " acc_choice

# --- Graphics Accelerator Selection ---
echo
acg_options=("Vulkan" "Metal (Apple)" "OpenGL ES" "OpenCL")
print_menu "Graphics Accelerators (acg)" "${acg_options[@]}"
read -p "Select Graphics accelerator: " acg_choice

# --- Build Flags ---
# Start with all backends disabled to ensure a clean configuration
CMAKE_FLAGS="-DTK_ENABLE_CUDA=OFF -DTK_ENABLE_ROCM=OFF -DTK_ENABLE_METAL=OFF -DTK_ENABLE_NNAPI=OFF"
CMAKE_FLAGS="$CMAKE_FLAGS -DTK_ENABLE_VULKAN=OFF -DTK_ENABLE_GLES=OFF -DTK_ENABLE_OPENCL=OFF"
SELECTED_ACC="None"
SELECTED_ACG="None"

# Process AI choice
case $acc_choice in
    1) CMAKE_FLAGS="$CMAKE_FLAGS -DTK_ENABLE_CUDA=ON"; SELECTED_ACC="CUDA" ;;
    2) CMAKE_FLAGS="$CMAKE_FLAGS -DTK_ENABLE_ROCM=ON"; SELECTED_ACC="ROCm" ;;
    3) CMAKE_FLAGS="$CMAKE_FLAGS -DTK_ENABLE_METAL=ON"; SELECTED_ACC="Metal" ;;
    4) CMAKE_FLAGS="$CMAKE_FLAGS -DTK_ENABLE_NNAPI=ON"; SELECTED_ACC="NNAPI" ;;
    *) ;;
esac

# Process Graphics choice
case $acg_choice in
    1) CMAKE_FLAGS="$CMAKE_FLAGS -DTK_ENABLE_VULKAN=ON"; SELECTED_ACG="Vulkan" ;;
    2) # If Metal was already chosen for ACC, this is redundant but harmless
       CMAKE_FLAGS="$CMAKE_FLAGS -DTK_ENABLE_METAL=ON"; SELECTED_ACG="Metal" ;;
    3) CMAKE_FLAGS="$CMAKE_FLAGS -DTK_ENABLE_GLES=ON"; SELECTED_ACG="OpenGL ES" ;;
    4) CMAKE_FLAGS="$CMAKE_FLAGS -DTK_ENABLE_OPENCL=ON"; SELECTED_ACG="OpenCL" ;;
    *) ;;
esac

# --- Summary and Execution ---
echo
echo "--- Configuration Summary ---"
echo "AI Accelerator:      $SELECTED_ACC"
echo "Graphics Accelerator:  $SELECTED_ACG"
echo "-----------------------------"
echo
echo "The following CMake command will be executed from the current directory:"
# Use sed to format the flags for readability
FORMATTED_FLAGS=$(echo $CMAKE_FLAGS | sed 's/ -D/\n  -D/g')
echo -e "cmake $FORMATTED_FLAGS\n  .."
echo

read -p "Press Enter to continue or Ctrl+C to cancel."

# Run cmake
# The '..' assumes this script is run from a 'build' subdirectory.
cmake $CMAKE_FLAGS ..
CMAKE_EXIT_CODE=$?

# Check if CMake succeeded before attempting to build
if [ $CMAKE_EXIT_CODE -eq 0 ]; then
    echo
    echo "--------------------------------"
    echo "CMake configuration successful."
    echo "Starting build..."
    echo "--------------------------------"
    # Use all available processor cores for a faster build
    make -j$(nproc)
    BUILD_EXIT_CODE=$?

    if [ $BUILD_EXIT_CODE -eq 0 ]; then
        echo
        echo "Build successful."
    else
        echo
        echo "Build failed. Please check the compilation errors above."
    fi
else
    echo
    echo "CMake configuration failed. Build process aborted."
fi
