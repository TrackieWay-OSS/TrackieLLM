/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/ai_models/gguf_runner.rs
 *
 * This file implements a safe Rust wrapper for the GGUF-based LLM runner
 * defined in `tk_model_runner.h`. It provides a high-level, idiomatic Rust
 * interface for stateful, tool-using conversational AI, with a focus on
 * a streaming-first API.
 *
 * ## Safety and Invariants
 *
 * This module contains `unsafe` code due to its interaction with the C FFI.
 * The primary safety concerns are pointer validity, resource management (memory),
 * and thread safety.
 *
 * - **Resource Management**: The `LlmContext` struct is an RAII guard. It owns the
 *   `tk_llm_runner_t` pointer and is responsible for its destruction via the
 *   `Drop` trait. This ensures that the C context is always freed when the
 *   `GgufRunner` goes out of scope.
 *
 * - **Pointer Validity**: The `RunnerHandle` struct uses `NonNull`, which ensures
 *   the wrapped pointer is never null. This turns potential null pointer
 *   dereferences into panics at the boundary, which is safer.
 *
 * - **Thread Safety**: The raw `*mut tk_llm_runner_t` pointer is not `Send`.
 *   We wrap it in `RunnerHandle` and implement `unsafe impl Send`. The justification
 *   for this is documented on the implementation itself. The core guarantee is
 *   that the `GgufRunner` will be managed by the `ModelService`, which places
 *   it behind a `tokio::sync::Mutex`. This external lock ensures that only one
 *   thread can access the C context at a time, making our use of the pointer safe.
 *
 * ## Auditing Guide
 *
 * To audit the `unsafe` code in this file:
 * 1. Verify that every `unsafe` block is as small as possible.
 * 2. Scrutinize the `LlmContext::new` function to ensure it correctly checks for
 *    a null pointer from the FFI call before creating the `NonNull` handle.
 * 3. Review the `Drop` implementation for `LlmContext` to confirm that
 *    `tk_llm_runner_destroy` is always called on a valid, non-null pointer.
 * 4. Examine the `stream_response` method. The `context_handle.as_ptr()` call is
 *    safe because the `LlmContext` (and thus the `GgufRunner`) is alive for the
 *    `'static` lifetime of the `spawn_blocking` task due to the design of the
 *    `ModelService` (which holds it in an `Arc`).
 */

use super::{ffi, loader::Model, AiModelsError, LlmConfig};
use futures::stream::{self, Stream, StreamExt};
use std::ffi::{CStr, CString};
use std::ptr::{null_mut, NonNull};

/// A handle to the C-level `tk_llm_runner_t`.
///
/// This wrapper uses `NonNull` to enforce that the pointer can never be null,
/// providing a small but important safety guarantee over a raw `*mut T`.
#[derive(Clone, Copy)]
struct RunnerHandle(NonNull<ffi::tk_llm_runner_t>);

/// ## `Send` Trait Safety Justification
///
/// We are marking `RunnerHandle` as `Send` so that it can be moved into the
/// `tokio::task::spawn_blocking` closure. This is a manual guarantee to the Rust
/// compiler that we are handling thread safety correctly.
///
/// 1.  **C-Side Guarantees**: The `llama.cpp` library, and by extension our C API,
///     is NOT inherently thread-safe. A single `llama_context` (which is inside
///     our `tk_llm_runner_t`) must not be accessed by multiple threads concurrently.
///
/// 2.  **Synchronization Mechanism**: The required synchronization is NOT handled
///     within this struct. It is an architectural invariant provided by the `ModelService`.
///     The `ModelService` holds the `GgufRunner` (which owns this handle) inside an
///     `Arc<Mutex<...>>`.
///
/// 3.  **Conclusion**: Any code that uses the `GgufRunner` must first acquire a
///     lock on the `Mutex`. This ensures that only one thread at a time can call
///     methods like `stream_response`. Because the `spawn_blocking` closure
///     takes ownership of the `GgufRunner` temporarily (via the `MutexGuard`),
///     we can guarantee that the `*mut tk_llm_runner_t` is only ever used by one
///     thread at a time.
unsafe impl Send for RunnerHandle {}

/// A low-level RAII wrapper for the `tk_llm_runner_t` handle.
struct LlmContext {
    handle: RunnerHandle,
}

impl LlmContext {
    fn new(model: &Model, config: &LlmConfig) -> Result<Self, AiModelsError> {
        let c_system_prompt = CString::new(config.system_prompt)?;
        let c_config = ffi::tk_llm_config_t {
            context_size: config.context_size,
            system_prompt: c_system_prompt.as_ptr(),
            random_seed: config.random_seed,
        };

        // Unsafe block to cast the generic model handle to the expected GGUF handle type.
        let gguf_handle = unsafe {
            &*(model.handle.as_ptr() as *const ffi::tk_gguf_handle_t)
        };

        let mut ptr = null_mut();
        // Unsafe FFI call to create the runner.
        let code = unsafe {
            ffi::tk_llm_runner_create(&mut ptr, gguf_handle.model, &c_config)
        };

        if code != ffi::TK_SUCCESS {
            return Err(AiModelsError::GgufLoadFailed(format!(
                "FFI call to tk_llm_runner_create failed with code {}", code
            )));
        }

        let handle = RunnerHandle(NonNull::new(ptr).expect("C API returned null pointer on success"));
        Ok(Self { handle })
    }
}

impl Drop for LlmContext {
    fn drop(&mut self) {
        unsafe { ffi::tk_llm_runner_destroy(&mut self.handle.0.as_ptr()) };
    }
}

/// A safe, high-level runner for GGUF-based Large Language Models.
pub struct GgufRunner {
    context: LlmContext,
}

impl GgufRunner {
    pub fn new(model: &Model, config: &LlmConfig) -> Result<Self, AiModelsError> {
        let context = LlmContext::new(model, config)?;
        Ok(Self { context })
    }

    pub fn prepare_response(&mut self, prompt: &str, use_tool_grammar: bool) -> Result<(), AiModelsError> {
        let c_prompt = CString::new(prompt)?;
        let code = unsafe {
            ffi::tk_llm_runner_prepare_generation(self.context.handle.0.as_ptr(), c_prompt.as_ptr(), use_tool_grammar)
        };
        if code != ffi::TK_SUCCESS {
            return Err(AiModelsError::GgufInferenceFailed("Failed to prepare generation".to_string()));
        }
        Ok(())
    }

    pub fn stream_response(&mut self) -> impl Stream<Item = Result<String, AiModelsError>> + '_ {
        // The handle is copied here. Since it's just a pointer wrapper, this is cheap.
        // The safety of using it in another thread is guaranteed by the `Send` impl.
        let handle = self.context.handle;
        stream::unfold((), move |()| async move {
            let token_result = tokio::task::spawn_blocking(move || {
                let c_str = unsafe { ffi::tk_llm_runner_generate_next_token(handle.0.as_ptr()) };
                if c_str.is_null() {
                    return Ok(None);
                }
                let token = unsafe { CStr::from_ptr(c_str) }
                    .to_str()
                    .map_err(AiModelsError::from)?
                    .to_owned();
                Ok(Some(token))
            })
            .await
            .unwrap();

            match token_result {
                Ok(Some(token)) => Some((Ok(token), ())),
                Ok(None) => None,
                Err(e) => Some((Err(e), ())),
            }
        })
    }

    pub async fn generate_response_full(&mut self, prompt: &str, use_tool_grammar: bool) -> Result<String, AiModelsError> {
        self.prepare_response(prompt, use_tool_grammar)?;
        let mut full_text = String::new();
        let mut stream = self.stream_response();
        while let Some(token_result) = stream.next().await {
            full_text.push_str(&token_result?);
        }
        Ok(full_text)
    }
}
