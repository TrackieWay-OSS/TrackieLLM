/*
 * Copyright (C) 2025 TrackieWay-OSS
 *
 * This file is part of TrackieLLM.
 *
 * This file provides a safe Rust wrapper for the ONNX C++ backend.
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

use super::{ffi, loader::Model, AiModelsError};
use std::ptr::{null_mut, NonNull};
use std::marker::PhantomData;

// Re-export the data type enum for convenience.
pub use super::ffi::ONNXTensorElementDataType;

/// A safe wrapper for a `tk_tensor_t` pointer.
/// It owns the C-level tensor and will free it on Drop.
pub struct Tensor {
    ptr: NonNull<ffi::tk_tensor_t>,
}

impl Tensor {
    /// Creates a new input tensor from a raw data slice.
    pub fn new<T>(data: &mut [T], shape: &[i64], data_type: ONNXTensorElementDataType) -> Result<Self, AiModelsError> {
        let mut ptr = null_mut();
        let code = unsafe {
            ffi::tk_tensor_create_from_raw(
                &mut ptr,
                data.as_mut_ptr() as *mut std::ffi::c_void,
                shape.as_ptr(),
                shape.len(),
                data_type,
            )
        };

        if code != ffi::TK_SUCCESS {
            return Err(AiModelsError::OnnxRunnerFailed(format!(
                "FFI call to tk_tensor_create_from_raw failed with code {}", code
            )));
        }

        Ok(Self {
            ptr: NonNull::new(ptr).expect("C API returned null tensor on success"),
        })
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { ffi::tk_tensor_destroy(&mut self.ptr.as_ptr()) };
    }
}


/// A handle to the C-level `tk_onnx_runner_t`.
#[derive(Debug)]
struct RunnerHandle(NonNull<ffi::tk_onnx_runner_t>);
unsafe impl Send for RunnerHandle {}
unsafe impl Sync for RunnerHandle {}

use std::sync::Arc;

/// A safe, high-level runner for ONNX-based models, using the C++ backend.
#[derive(Debug)]
pub struct OnnxRunner {
    handle: RunnerHandle,
    // Keep a strong reference to the model to ensure it outlives the runner.
    _model: Arc<Model>,
}

impl OnnxRunner {
    pub fn new(model: Arc<Model>) -> Result<Self, AiModelsError> {
        let mut ptr = null_mut();
        let code = unsafe { ffi::tk_onnx_runner_create(&mut ptr, model.handle.as_ptr()) };
        if code != ffi::TK_SUCCESS {
            return Err(AiModelsError::OnnxRunnerFailed(format!(
                "FFI call to tk_onnx_runner_create failed with code {}", code
            )));
        }
        Ok(Self {
            handle: RunnerHandle(NonNull::new(ptr).expect("C API returned null pointer on success")),
            _model: model,
        })
    }

    /// Runs inference on the loaded ONNX model.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, AiModelsError> {
        let input_ptrs: Vec<*const ffi::tk_tensor_t> = inputs.iter().map(|t| t.ptr.as_ptr()).collect();
        let mut output_ptrs = null_mut();
        let mut output_count = 0;

        let code = unsafe {
            ffi::tk_onnx_runner_run(
                self.handle.0.as_ptr(),
                input_ptrs.as_ptr(),
                input_ptrs.len(),
                &mut output_ptrs,
                &mut output_count,
            )
        };

        if code != ffi::TK_SUCCESS {
            return Err(AiModelsError::InferenceFailed(format!(
                "ONNX inference failed with code {}", code
            )));
        }

        // The C side allocates an array of pointers, and a tensor for each pointer.
        // We need to take ownership of these and manage their lifecycle.
        let outputs = unsafe {
            let slice = std::slice::from_raw_parts_mut(output_ptrs, output_count);
            slice.iter_mut().map(|ptr| {
                Tensor { ptr: NonNull::new(*ptr).unwrap() }
            }).collect()
        };

        // Now that the Vec<Tensor> owns the individual tensors, we can free the
        // outer array that held the pointers.
        unsafe {
            ffi::tk_onnx_runner_free_outputs(output_ptrs, output_count);
        }

        Ok(outputs)
    }
}

impl Drop for OnnxRunner {
    fn drop(&mut self) {
        unsafe { ffi::tk_onnx_runner_destroy(&mut self.handle.0.as_ptr()) };
    }
}