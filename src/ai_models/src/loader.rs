/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/ai_models/loader.rs
 *
 * This file provides a safe Rust interface for the C-based model loader API.
 * It defines the `ModelLoader` struct, which manages the lifecycle of the
 * underlying `tk_model_loader_t`, and the `Model` struct, which is a safe
 * RAII wrapper around a loaded model handle.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::{ffi, AiModelsError};
use internal_tools::fs_utils::Path as TkPath;
use std::ptr::null_mut;

/// A safe, RAII-compliant wrapper for a loaded model handle.
///
/// When this struct is dropped, it automatically calls the C API to unload
/// the model, ensuring that resources are always cleaned up.
#[derive(Debug)]
pub struct Model {
    /// Opaque handle to the loaded model context in the C layer.
    pub(crate) handle: *mut std::ffi::c_void,
    /// A raw pointer to the loader that owns this model.
    /// This is necessary to call the unload function on drop.
    loader_ptr: *mut ffi::tk_model_loader_t,
}

impl Drop for Model {
    /// Unloads the model from the C API when the `Model` struct goes out of scope.
    fn drop(&mut self) {
        if !self.handle.is_null() && !self.loader_ptr.is_null() {
            log::debug!("Dropping Model, unloading from C API.");
            unsafe {
                // The C API expects a pointer to the handle pointer.
                let mut handle_ptr = self.handle;
                ffi::tk_model_loader_unload_model(self.loader_ptr, &mut handle_ptr);
                // The C function should nullify the handle, but we do it here too for safety.
                self.handle = null_mut();
            }
        }
    }
}

/// A safe wrapper for the `tk_model_loader_t` C API.
///
/// This struct manages the creation and destruction of the C loader and provides
/// safe methods for loading models into memory.
pub struct ModelLoader {
    ptr: *mut ffi::tk_model_loader_t,
}

impl ModelLoader {
    /// Creates a new `ModelLoader` with default settings.
    pub fn new() -> Result<Self, AiModelsError> {
        let config = ffi::tk_model_loader_config_t {
            max_models: 16,
            num_threads: 4,
            ..Default::default()
        };

        let mut ptr = null_mut();
        let code = unsafe { ffi::tk_model_loader_create(&mut ptr, &config) };
        if code != ffi::TK_SUCCESS || ptr.is_null() {
            return Err(AiModelsError::Ffi(format!(
                "tk_model_loader_create failed with code {}",
                code
            )));
        }
        log::info!("ModelLoader created successfully.");
        Ok(Self { ptr })
    }

    /// Loads a model from the specified path.
    ///
    /// This function calls the C API to load a model and wraps the resulting
    /// handle in a safe `Model` struct that manages its lifetime.
    pub fn load_model(&self, path: &TkPath) -> Result<Model, AiModelsError> {
        let params = ffi::tk_model_load_params_t {
            model_path: path.as_ptr() as *mut _,
            // Other parameters can be set here if needed.
            // For now, we rely on the C API's defaults.
            ..Default::default() // This requires deriving Default for the C struct
        };

        let mut model_handle = null_mut();
        let code = unsafe {
            ffi::tk_model_loader_load_model(self.ptr, &params, &mut model_handle)
        };

        if code != ffi::TK_SUCCESS || model_handle.is_null() {
            return Err(AiModelsError::ModelLoadFailed {
                path: path.to_string(),
                reason: format!("FFI call to tk_model_loader_load_model failed with code {}", code),
            });
        }

        log::info!("Model loaded successfully from path: {}", path.to_string());
        Ok(Model {
            handle: model_handle,
            loader_ptr: self.ptr,
        })
    }
}

impl Drop for ModelLoader {
    /// Destroys the underlying C model loader.
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            log::debug!("Dropping ModelLoader, destroying C loader instance.");
            unsafe { ffi::tk_model_loader_destroy(&mut self.ptr) };
        }
    }
}

// The Default trait is now derived for the FFI struct in lib.rs,
// so this manual implementation is no longer needed.
