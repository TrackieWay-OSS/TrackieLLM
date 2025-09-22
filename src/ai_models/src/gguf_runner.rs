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
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::{
    ffi, loader::Model, AiModelsError, LlmConfig, LlmResult, LlmToolCall, ToolDefinition,
};
use futures::stream::{self, Stream};
use std::ffi::{CStr, CString};
use std::pin::Pin;
use std::ptr::null_mut;

/// A low-level RAII wrapper for the `tk_llm_runner_t` handle.
struct LlmContext {
    ptr: *mut ffi::tk_llm_runner_t,
}

impl LlmContext {
    /// Creates a new `LlmContext` from a loaded model handle.
    fn new(model: &Model, config: &LlmConfig) -> Result<Self, AiModelsError> {
        let c_system_prompt = CString::new(config.system_prompt)?;
        let c_config = ffi::tk_llm_config_t {
            context_size: config.context_size,
            system_prompt: c_system_prompt.as_ptr(),
            random_seed: config.random_seed,
        };

        // The model handle from the loader is a void*. We need to cast it
        // to the concrete handle type to get the llama_model*.
        // This is inherently unsafe but required by the C API design.
        let gguf_handle = unsafe {
            &*(model.handle as *const crate::loader::tk_gguf_handle_t)
        };

        let mut ptr = null_mut();
        let code = unsafe {
            ffi::tk_llm_runner_create(&mut ptr, gguf_handle.model, &c_config)
        };

        if code != ffi::TK_SUCCESS || ptr.is_null() {
            return Err(AiModelsError::GgufLoadFailed(format!(
                "FFI call to tk_llm_runner_create failed with code {}",
                code
            )));
        }
        Ok(Self { ptr })
    }
}

impl Drop for LlmContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::tk_llm_runner_destroy(&mut self.ptr) };
        }
    }
}

/// A safe, high-level runner for GGUF-based Large Language Models.
pub struct GgufRunner {
    context: LlmContext,
}

impl GgufRunner {
    /// Creates a new `GgufRunner` from a previously loaded model.
    pub fn new(model: &Model, config: &LlmConfig) -> Result<Self, AiModelsError> {
        let context = LlmContext::new(model, config)?;
        Ok(Self { context })
    }

    /// Prepares the runner for a new generation by processing the initial prompt.
    pub fn prepare_response(
        &mut self,
        prompt: &str,
        use_tool_grammar: bool,
    ) -> Result<(), AiModelsError> {
        let c_prompt = CString::new(prompt)?;
        let code = unsafe {
            ffi::tk_llm_runner_prepare_generation(self.context.ptr, c_prompt.as_ptr(), use_tool_grammar)
        };
        if code != ffi::TK_SUCCESS {
            return Err(AiModelsError::GgufInferenceFailed(
                "Failed to prepare generation".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns a stream that yields the next token of the response on each iteration.
    pub fn stream_response(&mut self) -> impl Stream<Item = Result<String, AiModelsError>> + '_ {
        stream::unfold(self, |runner| async {
            let context_ptr = runner.context.ptr;
            // The FFI call is blocking, so we move it to a blocking-safe thread.
            let token_result = tokio::task::spawn_blocking(move || {
                let c_str = unsafe { ffi::tk_llm_runner_generate_next_token(context_ptr) };
                if c_str.is_null() {
                    return Ok(None);
                }
                let token = unsafe { CStr::from_ptr(c_str) }.to_str()?.to_owned();
                Ok(Some(token))
            })
            .await
            .unwrap(); // Propagate panics from the blocking task.

            match token_result {
                Ok(Some(token)) => Some((Ok(token), runner)),
                Ok(None) => None, // End of stream
                Err(e) => Some((Err(e.into()), runner)),
            }
        })
    }

    // The following functions are placeholders for the full API and would need to be implemented.
    // For now, they are commented out to allow the project to compile.
    /*
    pub async fn generate_response_full(
        &mut self,
        user_transcription: &str,
        vision_context: Option<&str>,
        available_tools: &[ToolDefinition],
    ) -> Result<LlmResult, AiModelsError> {
        // 1. Build the full prompt string here in Rust.
        let prompt = self.build_prompt_string(user_transcription, vision_context, available_tools)?;

        // 2. Prepare the C runner for generation.
        self.prepare_response(&prompt, !available_tools.is_empty())?;
        
        // 3. Collect the full response from the stream.
        let mut full_text = String::new();
        let mut stream = self.stream_response();
        while let Some(token_result) = stream.next().await {
            full_text.push_str(&token_result?);
        }

        // 4. Parse the full response to determine if it's a tool call or text.
        self.parse_full_response(&full_text)
    }

    fn build_prompt_string(...) -> Result<String, AiModelsError> {
        // ... logic to construct the prompt string ...
        Ok("...".to_string())
    }

    fn parse_full_response(&self, response: &str) -> Result<LlmResult, AiModelsError> {
        // ... logic to check for tool call JSON and parse it ...
        // This would replace the C-side `parse_tool_call`.
        if response.contains("tool_call") {
            // Parse JSON...
            Ok(LlmResult::ToolCall(...))
        } else {
            Ok(LlmResult::Text(response.to_string()))
        }
    }
    */
}

impl From<std::str::Utf8Error> for AiModelsError {
    fn from(err: std::str::Utf8Error) -> Self {
        AiModelsError::InvalidUtf8(err)
    }
}
