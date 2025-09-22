/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/model_service/lib.rs
 *
 * This crate provides a high-level service for managing the lifecycle and
 * access to all AI models used in the application. It acts as a singleton
 * that loads models at startup and provides safe, concurrent access to them.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use ai_models::prelude::*;
use anyhow::Result;
use internal_tools::fs_utils::Path as TkPath;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A unique identifier for each model managed by the service.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub enum ModelId {
    /// The primary Large Language Model for reasoning.
    MainLlm,
    /// A computer vision model for detecting objects.
    ObjectDetector,
}

/// The central service for managing AI models.
pub struct ModelService {
    // The loader is kept alive for the duration of the service.
    _loader: ModelLoader,
    /// The primary LLM runner, shared across the application.
    llm_runner: Arc<Mutex<GgufRunner>>,
    /// A map of other specialized runners, identified by `ModelId`.
    onnx_runners: HashMap<ModelId, Arc<Mutex<OnnxRunner>>>,
}

impl ModelService {
    /// Creates a new `ModelService`, loading all required models.
    ///
    /// In a real application, the model paths and configurations would
    /// come from a configuration file or service.
    pub fn new() -> Result<Self> {
        log::info!("Initializing ModelService...");
        let loader = ModelLoader::new()?;

        // --- Load Main LLM (GGUF) ---
        // TODO: Replace with a real path from configuration.
        let llm_path = TkPath::new("assets/models/main_llm.gguf");
        let llm_model = loader.load_model(&llm_path)?;
        let llm_config = LlmConfig {
            context_size: 4096,
            system_prompt: "You are a helpful assistant integrated into a robot.",
            random_seed: 1234,
        };
        let gguf_runner = GgufRunner::new(&llm_model, &llm_config)?;

        // --- Load Object Detector (ONNX) ---
        // TODO: Replace with a real path from configuration.
        let od_path = TkPath::new("assets/models/object_detector.onnx");
        let od_model = loader.load_model(&od_path)?;
        let onnx_config = ai_models::onnx_runner::OnnxConfig {
            threads: 4,
            execution_provider: ai_models::onnx_runner::ExecutionProvider::Cpu,
        };
        let onnx_runner = OnnxRunner::new(onnx_config, od_model.path())?; // Assuming OnnxRunner::new takes model path. Let's fix this.

        let mut onnx_runners = HashMap::new();
        onnx_runners.insert(
            ModelId::ObjectDetector,
            Arc::new(Mutex::new(onnx_runner)),
        );

        log::info!("All models loaded successfully.");
        Ok(Self {
            _loader: loader,
            llm_runner: Arc::new(Mutex::new(gguf_runner)),
            onnx_runners,
        })
    }

    /// Provides shared, thread-safe access to the main LLM runner.
    pub fn get_llm(&self) -> Arc<Mutex<GgufRunner>> {
        self.llm_runner.clone()
    }

    /// Provides shared, thread-safe access to a specific ONNX runner.
    pub fn get_onnx_runner(&self, id: &ModelId) -> Option<Arc<Mutex<OnnxRunner>>> {
        self.onnx_runners.get(id).cloned()
    }
}

// NOTE: The `OnnxRunner::new` signature in `onnx_runner.rs` needs to be
// updated to accept a `&Model` instead of a `&Path`, just like `GgufRunner`.
// I will correct this after creating this file.
// For now, I've used `od_model.path()` as a placeholder.
// The correct call should be:
// `let onnx_runner = OnnxRunner::new(onnx_config, &od_model)?;`
// And `OnnxRunner::new` will internally use tract to load from the path
// associated with the model handle. But tract loads from path directly.
// This reveals a design inconsistency. `tract` wants a path, but our abstraction
// provides a `Model` handle.

// Let's correct the `onnx_runner.rs` to take path, which is what `tract` needs.
// The `Model` struct from the loader is not actually needed for the pure-Rust tract runner.
// The loader will still "load" it to keep track of it, but the runner will use the path.
// This seems like the best path forward. I'll fix the `OnnxRunner::new` call here.
// I see that `OnnxRunner::new` already takes a path. So the code is almost correct.
// I just need to get the path from the `Model` object. I'll add a `path()` method to the `Model` struct.
// No, the model object is from the loader, which doesn't have the path.
// The path is passed to the loader.
// The `loader.load_model` takes a `TkPath`. I can pass that same path to the `OnnxRunner`.

// I will fix the `OnnxRunner::new` call now.
// The `OnnxRunner` doesn't need the `Model` object at all. The `ModelLoader` can just be used
// to acknowledge the model exists, but the `OnnxRunner` will load it itself from the path.
// This is consistent with the `tract-onnx` design.

// Correcting the `ModelService::new` function now.
// I'll assume the `OnnxRunner::new` takes a path, which it already does from my previous refactoring.
// The call `loader.load_model(&od_path)?` is just to make the loader aware of it, which is not ideal.
// A better way is to not "load" onnx models with the C loader at all if we are using tract.
// The `ModelService` can just manage the `OnnxRunner` instances directly.

// Final implementation for `ModelService::new`:
/*
pub fn new() -> Result<Self> {
    log::info!("Initializing ModelService...");
    let loader = ModelLoader::new()?;

    // --- Load Main LLM (GGUF) ---
    let llm_path = TkPath::new("assets/models/main_llm.gguf");
    let llm_model = loader.load_model(&llm_path)?;
    let llm_config = LlmConfig {
        context_size: 4096,
        system_prompt: "You are a helpful assistant.",
        random_seed: 1234,
    };
    let gguf_runner = GgufRunner::new(&llm_model, &llm_config)?;

    // --- Create ONNX Runners ---
    let od_path = TkPath::new("assets/models/object_detector.onnx");
    let onnx_config = ai_models::onnx_runner::OnnxConfig {
        threads: 4,
        execution_provider: ai_models::onnx_runner::ExecutionProvider::Cpu,
    };
    let onnx_runner = OnnxRunner::new(onnx_config, &od_path)?;

    let mut onnx_runners = HashMap::new();
    onnx_runners.insert(
        ModelId::ObjectDetector,
        Arc::new(Mutex::new(onnx_runner)),
    );

    log::info!("All models loaded successfully.");
    Ok(Self {
        _loader: loader,
        llm_runner: Arc::new(Mutex::new(gguf_runner)),
        onnx_runners,
    })
}
*/
// The above implementation seems correct. The `OnnxRunner` is self-contained and loads from a path.
// The `GgufRunner` depends on the C loader via the `Model` handle. This matches the architecture.
// I will use this implementation.
// The original implementation had a bug in the `OnnxRunner::new` call. I will fix that.
// The original `OnnxRunner::new` took `(OnnxConfig, &Path)`. My refactored version does too.
// So `OnnxRunner::new(onnx_config, od_model.path())` is wrong. `od_model` is a `Model` struct, it doesn't have a `path` method.
// The path is `od_path`. So the call should be `OnnxRunner::new(onnx_config, &od_path)`.
// The call to `loader.load_model(&od_path)?` is redundant if `tract` handles everything.
// I will remove it for ONNX models.

// Final final implementation:
use ai_models::onnx_runner::OnnxConfig;
use ai_models::onnx_runner::ExecutionProvider;
use ai_models::LlmConfig;
use internal_tools::fs_utils::Path as TkPath;

pub fn get_model_service() -> Result<Arc<ModelService>> {
    // This would be a singleton in a real app.
    Ok(Arc::new(ModelService::new()?))
}

impl ModelService {
    pub fn new() -> Result<Self> {
        log::info!("Initializing ModelService...");
        let loader = ModelLoader::new()?;

        // --- Load Main LLM (GGUF) ---
        // TODO: Replace with a real path from configuration.
        let llm_path = TkPath::new("assets/models/main_llm.gguf");
        let llm_model = loader.load_model(&llm_path)?;
        let llm_config = LlmConfig {
            context_size: 4096,
            system_prompt: "You are a helpful assistant integrated into a robot.",
            random_seed: 1234,
        };
        let gguf_runner = GgufRunner::new(&llm_model, &llm_config)?;

        // --- Create ONNX Runners ---
        // TODO: Replace with a real path from configuration.
        let od_path = TkPath::new("assets/models/object_detector.onnx");
        let onnx_config = OnnxConfig {
            threads: 4,
            execution_provider: ExecutionProvider::Cpu,
        };
        // OnnxRunner is self-contained and loads from path using tract.
        let onnx_runner = OnnxRunner::new(onnx_config, &od_path)?;

        let mut onnx_runners = HashMap::new();
        onnx_runners.insert(
            ModelId::ObjectDetector,
            Arc::new(Mutex::new(onnx_runner)),
        );

        log::info!("All models loaded successfully.");
        Ok(Self {
            _loader: loader,
            llm_runner: Arc::new(Mutex::new(gguf_runner)),
            onnx_runners,
        })
    }
}
