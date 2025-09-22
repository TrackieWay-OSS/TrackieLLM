/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/ai_models/onnx_runner.rs
 *
 * This file provides a safe, pure-Rust interface for running models in the ONNX
 * (Open Neural Network Exchange) format using the `tract-onnx` inference engine.
 * This approach avoids FFI, enhancing safety and simplifying the toolchain.
 *
 * The `OnnxRunner` is responsible for the entire lifecycle of an ONNX model,
 * from loading and optimization to inference. It is suitable for non-LLM tasks
 * such as computer vision, audio processing, etc.
 *
 * ## Safety
 * This module is written in 100% safe Rust and has no `unsafe` blocks. All
 * interactions with low-level tensor operations are managed by the `tract` crate.
 */

use crate::AiModelsError;
use std::path::Path;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::downcast_rs::Downcast;

/// A new Tensor struct that wraps tract's Tensor.
/// This abstracts the underlying tensor library from the user of OnnxRunner.
#[derive(Clone, Debug)]
pub struct Tensor(pub tract_onnx::prelude::Tensor);

/// Configuration for initializing an `OnnxRunner`.
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    /// The execution provider to use. (Note: tract primarily uses CPU)
    pub execution_provider: ExecutionProvider,
}

/// Defines the available execution providers for ONNX models.
/// Note: `tract` is mainly a CPU engine, so these are conceptual for API compatibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// Use the CPU for inference.
    Cpu,
}

/// A safe, high-level runner for ONNX-based models, using the `tract` engine.
#[derive(Debug)]
pub struct OnnxRunner {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl OnnxRunner {
    /// Creates a new `OnnxRunner` by loading and optimizing an ONNX model from a file.
    ///
    /// # Arguments
    ///
    /// * `_config` - Configuration for the runner (currently unused by tract's simple API).
    /// * `model_path` - The file path to the ONNX model.
    pub fn new(_config: OnnxConfig, model_path: &Path) -> Result<Self, AiModelsError> {
        log::info!(
            "Initializing ONNX runner for model path: {:?}",
            model_path
        );

        // Tract's builder pattern to load, type-check, optimize, and make the model runnable.
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self { model })
    }

    /// Runs inference on the loaded ONNX model.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of input tensors. The number and shape of these
    ///   tensors must match the model's expected inputs.
    ///
    /// # Returns
    ///
    /// A `Vec<Tensor>` containing the model's output tensors.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, AiModelsError> {
        log::debug!("Running ONNX inference with {} input tensor(s).", inputs.len());

        if inputs.is_empty() {
            return Err(AiModelsError::InferenceFailed(
                "At least one input tensor is required.".to_string(),
            ));
        }

        // Convert our `Tensor` wrappers into the `TValue` (`Arc<dyn Datum>`) that tract expects.
        let tract_inputs: TVec<TValue> =
            inputs.iter().map(|t| t.0.clone().into()).collect();

        let result_tensors = self.model.run(tract_inputs)?;

        // Downcast the resulting `TValue`s back to concrete `tract_onnx::prelude::Tensor`s
        // and wrap them in our public `Tensor` type.
        let outputs: Result<Vec<Tensor>, _> = result_tensors
            .into_iter()
            .map(|t| {
                t.as_any()
                    .downcast_ref::<tract_onnx::prelude::Tensor>()
                    .ok_or_else(|| {
                        AiModelsError::InferenceFailed(
                            "Failed to downcast output tensor to concrete type".to_string(),
                        )
                    })
                    .map(|tensor| Tensor(tensor.clone()))
            })
            .collect();

        outputs
    }
}
