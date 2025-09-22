/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/ai_models/loader.rs
 *
 * This file provides a safe Rust interface for the C-based model loader API
 * defined in `tk_model_loader.h`.
 *
 * ## Safety and Invariants
 *
 * This module encapsulates the `unsafe` FFI calls required to interact with the
 * C model loader.
 *
 * - **Resource Management**: `ModelLoader` and `Model` are RAII guards. Their
 *   `Drop` implementations ensure that the corresponding C resources
 *   (`tk_model_loader_t*` and `tk_internal_model_t*`) are always released,
 *   preventing memory leaks.
 * - **Pointer Validity**: The C API returns raw pointers that can be null on
 *   failure. The `new` and `load_model` functions immediately check for null
 *   and convert it into a Rust `Result`, ensuring that `ModelLoader` and `Model`
 *   never contain null pointers.
 *
 * ## Auditing Guide
 *
 * To audit `unsafe` code:
 * 1. Check `ModelLoader::new` and `load_model` to ensure null checks after FFI
 *    calls are present and correct.
 * 2. Verify that the `Drop` implementations correctly call the corresponding
 *    `destroy`/`unload` FFI functions on their non-null pointers.
 */

use super::{ffi, AiModelsError};
use internal_tools::fs_utils::Path as TkPath;
use std::ptr::{null_mut, NonNull};

/// A safe, RAII-compliant wrapper for a loaded model handle.
///
/// When this struct is dropped, it automatically calls the C API to unload
/// the model, ensuring that resources are always cleaned up.
#[derive(Debug)]
pub struct Model {
    /// Opaque handle to the loaded model context in the C layer.
    pub(crate) handle: NonNull<std::ffi::c_void>,
    /// A raw pointer to the loader that owns this model.
    loader_ptr: *mut ffi::tk_model_loader_t,
}

impl Drop for Model {
    /// Unloads the model from the C API when the `Model` struct goes out of scope.
    fn drop(&mut self) {
        if !self.loader_ptr.is_null() {
            log::debug!("Dropping Model, unloading from C API.");
            // Unsafe FFI call. Safe because we know the loader and handle are valid.
            unsafe {
                let mut handle_ptr = self.handle.as_ptr();
                ffi::tk_model_loader_unload_model(self.loader_ptr, &mut handle_ptr);
            }
        }
    }
}

/// A safe wrapper for the `tk_model_loader_t` C API.
///
/// This struct manages the creation and destruction of the C loader and provides
/// safe methods for loading models into memory.
pub struct ModelLoader {
    ptr: NonNull<ffi::tk_model_loader_t>,
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
        // Unsafe FFI call.
        let code = unsafe { ffi::tk_model_loader_create(&mut ptr, &config) };

        if code != ffi::TK_SUCCESS {
            return Err(AiModelsError::Ffi(format!(
                "tk_model_loader_create failed with code {}", code
            )));
        }

        // Convert raw pointer to NonNull, panicking if it's null.
        let non_null_ptr = NonNull::new(ptr).expect("C API returned null on success");
        log::info!("ModelLoader created successfully.");
        Ok(Self { ptr: non_null_ptr })
    }

    /// Loads a model from the specified path.
    pub fn load_model(&self, path: &TkPath, params: &ffi::tk_model_load_params_t) -> Result<Model, AiModelsError> {
        let mut model_handle = null_mut();
        // Unsafe FFI call. Safe because loader pointer is valid and path is valid.
        let code = unsafe {
            ffi::tk_model_loader_load_model(self.ptr.as_ptr(), params, &mut model_handle)
        };

        if code != ffi::TK_SUCCESS {
            return Err(AiModelsError::ModelLoadFailed {
                path: path.to_string(),
                reason: format!("FFI call to tk_model_loader_load_model failed with code {}", code),
            });
        }

        let non_null_handle = NonNull::new(model_handle).expect("C API returned null on success");
        log::info!("Model loaded successfully from path: {}", path.to_string());
        Ok(Model {
            handle: non_null_handle,
            loader_ptr: self.ptr.as_ptr(),
        })
    }
}

impl Drop for ModelLoader {
    /// Destroys the underlying C model loader.
    fn drop(&mut self) {
        log::debug!("Dropping ModelLoader, destroying C loader instance.");
        // Unsafe FFI call. Safe because we know the pointer is valid.
        unsafe { ffi::tk_model_loader_destroy(&mut self.ptr.as_ptr()) };
    }
}
