/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/ai_models/lib.rs
 *
 * This file is the main library entry point for the 'ai_models' crate. This
 * crate is the core of the application's intelligence, providing a unified
 * Rust interface for loading, managing, and running various AI models.
 *
 * It acts as a safe, high-level abstraction layer over the comprehensive C APIs
 * for model loading and GGUF-based inference. For ONNX models, it re-exports
 * a pure-Rust implementation based on the `tract-onnx` crate.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TABLE OF CONTENTS ===
// 1. Crate-Level Attributes & Documentation
// 2. FFI Bindings Module (private)
// 3. Public Module Declarations
// 4. Core Public Data Structures
// 5. Core Public Error Type
// 6. Public Prelude
// ===========================

#![allow(unsafe_code)] // Unsafe code is necessary for FFI and is audited.
#![allow(missing_docs)] // TODO: Remove this allow once public API is stable.
#![deny(warnings)]

//! # TrackieLLM AI Models Crate
//!
//! Provides high-level, safe Rust APIs for interacting with AI models.
//!
//! ## Design Philosophy
//!
//! This crate follows a "safe over unsafe" design pattern. All `unsafe` FFI
//! calls are encapsulated within higher-level functions that return `Result`
//! types and manage resource lifetimes automatically using the `Drop` trait.
//! This prevents common C errors like memory leaks, use-after-free, and null
//! pointer dereferences from propagating into the Rust part of the application.
//!
//! For ONNX models, we opt for a pure-Rust implementation using `tract-onnx`
//! to maximize safety and simplify the build process. For GGUF models, we use
//! a carefully managed FFI bridge to the highly optimized `llama.cpp` library.
//!
//! ## Auditing Unsafe Code
//!
//! To audit the `unsafe` code in this crate, focus on the following areas:
//!
//! 1.  **`loader.rs`**: Check that `NonNull` is used correctly to wrap pointers
//!     returned from the C API and that `Drop` implementations correctly call
//!     the corresponding C `destroy` functions.
//! 2.  **`gguf_runner.rs`**: Scrutinize the `unsafe impl Send` block and its safety
//!     justification. Ensure that the `RunnerHandle` is only ever used in a way
//!     that upholds the invariants described (i.e., behind a `Mutex`).
//! 3.  **`ffi` module**: Verify that all `extern "C"` function signatures and
//!     `#[repr(C)]` struct definitions exactly match their counterparts in the
//!     C header files (`.h`). Any mismatch can lead to undefined behavior.
//!
//! ### Potential Memory Leaks
//!
//! Leaks can occur if the `Drop` implementation for a Rust wrapper is not
//! called (e.g., due to `std::mem::forget`) or if the C API allocates memory
//! that is not properly owned and freed by a Rust object. The primary defense
//! is the RAII pattern implemented in `loader.rs` and `gguf_runner.rs`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

// --- FFI Bindings Module ---
mod ffi {
    #![allow(non_camel_case_types, non_snake_case, dead_code)]

    pub type tk_error_code_t = i32;
    pub const TK_SUCCESS: tk_error_code_t = 0;
    pub const TK_ERROR_INVALID_ARGUMENT: i32 = -1;
    pub const TK_ERROR_OUT_OF_MEMORY: i32 = -2;
    pub const TK_ERROR_MODEL_LOAD_FAILED: i32 = -10;
    pub const TK_ERROR_INFERENCE_FAILED: i32 = -11;

    // --- Opaque Types ---
    pub enum tk_model_loader_s {}
    pub type tk_model_loader_t = tk_model_loader_s;
    pub enum tk_llm_runner_s {}
    pub type tk_llm_runner_t = tk_llm_runner_s;
    pub enum llama_model {}
    pub enum tk_path_s {}
    pub type tk_path_t = tk_path_s;

    // --- Structs & Enums ---
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub enum tk_model_format_e {
        TK_MODEL_FORMAT_UNKNOWN = 0,
        TK_MODEL_FORMAT_GGUF,
        TK_MODEL_FORMAT_ONNX,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct tk_model_loader_config_t {
        pub max_models: u32,
        pub num_threads: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct tk_model_load_params_t {
        pub model_path: *mut tk_path_t,
        pub model_type: u32,
        pub force_reload: bool,
        pub gpu_layers: u32,
        pub cpu_threads: u32,
        pub use_mmap: bool,
        pub use_mlock: bool,
        pub numa: bool,
        pub seed: u32,
    }

    #[repr(C)]
    pub struct tk_llm_config_t {
        pub context_size: u32,
        pub system_prompt: *const std::os::raw::c_char,
        pub random_seed: u32,
    }

    #[repr(C)]
    pub struct tk_gguf_handle_t {
        pub model: *mut llama_model,
    }
    
    extern "C" {
        // tk_model_loader.h
        pub fn tk_model_loader_create(out_loader: *mut *mut tk_model_loader_t, config: *const tk_model_loader_config_t) -> tk_error_code_t;
        pub fn tk_model_loader_destroy(loader: *mut *mut tk_model_loader_t);
        pub fn tk_model_loader_load_model(loader: *mut tk_model_loader_t, params: *const tk_model_load_params_t, out_model_handle: *mut *mut std::ffi::c_void) -> tk_error_code_t;
        pub fn tk_model_loader_unload_model(loader: *mut tk_model_loader_t, model_handle: *mut *mut std::ffi::c_void) -> tk_error_code_t;

        // tk_model_runner.h
        pub fn tk_llm_runner_create(out_runner: *mut *mut tk_llm_runner_t, model: *mut llama_model, config: *const tk_llm_config_t) -> tk_error_code_t;
        pub fn tk_llm_runner_destroy(runner: *mut *mut tk_llm_runner_t);
        pub fn tk_llm_runner_prepare_generation(runner: *mut tk_llm_runner_t, prompt: *const std::os::raw::c_char, use_tool_grammar: bool) -> tk_error_code_t;
        pub fn tk_llm_runner_generate_next_token(runner: *mut tk_llm_runner_t) -> *const std::os::raw::c_char;
    }
}


// --- Public Module Declarations ---

pub mod loader;
pub mod gguf_runner;
pub mod onnx_runner;


// --- Core Public Data Structures ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters_json_schema: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmToolCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub enum LlmResult {
    Text(String),
    ToolCall(LlmToolCall),
}

pub struct LlmConfig<'a> {
    pub context_size: u32,
    pub system_prompt: &'a str,
    pub random_seed: u32,
}


// --- Core Public Error Type ---

#[derive(Debug, Error)]
pub enum AiModelsError {
    #[error("FFI call failed: {0}")]
    Ffi(String),

    #[error("Invalid C-style string: {0}")]
    InvalidCString(#[from] std::ffi::NulError),
    
    #[error("Invalid UTF-8 sequence in string from FFI: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),
    
    #[error("JSON operation failed: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Model load failed for path '{path}': {reason}")]
    ModelLoadFailed {
        path: String,
        reason: String,
    },

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Failed to load GGUF model: {0}")]
    GgufLoadFailed(String),

    #[error("GGUF inference failed: {0}")]
    GgufInferenceFailed(String),

    #[error("An error originating from the `tract-onnx` library: {0}")]
    TractError(#[from] tract_onnx::prelude::TractError),
}


// --- Public Prelude ---

pub mod prelude {
    pub use super::{
        gguf_runner::GgufRunner,
        loader::{Model, ModelLoader},
        onnx_runner::{OnnxConfig, OnnxRunner, Tensor, ExecutionProvider},
        AiModelsError, LlmConfig, LlmResult, LlmToolCall, ToolDefinition,
    };
}
