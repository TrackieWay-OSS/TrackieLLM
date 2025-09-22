use ai_models::prelude::*;
use internal_tools::fs_utils::Path as TkPath;

#[test]
fn test_model_loader_creation() {
    let loader = ModelLoader::new();
    assert!(loader.is_ok(), "ModelLoader::new() should not fail");
}

#[test]
fn test_gguf_model_load_failure() {
    let loader = ModelLoader::new().unwrap();
    let non_existent_path = TkPath::from_str("/tmp/non_existent_model.gguf").unwrap();

    let result = loader.load_model(&non_existent_path);

    assert!(result.is_err(), "Loading a non-existent model should fail");

    if let Err(AiModelsError::ModelLoadFailed { path, .. }) = result {
        assert_eq!(path, non_existent_path.to_string());
    } else {
        panic!("Expected ModelLoadFailed error, but got {:?}", result);
    }
}

#[test]
fn test_onnx_runner_creation_failure() {
    let non_existent_path_str = "/tmp/non_existent_model.onnx";
    let non_existent_path = std::path::Path::new(non_existent_path_str);

    let config = ai_models::onnx_runner::OnnxConfig {
        threads: 1,
        execution_provider: ai_models::onnx_runner::ExecutionProvider::Cpu,
    };

    let result = OnnxRunner::new(config, non_existent_path);

    assert!(result.is_err(), "Creating an OnnxRunner with a non-existent model should fail");

    if let Err(AiModelsError::TractError(_)) = result {
        // Correct error type, as tract will fail to open the file.
    } else {
        panic!("Expected TractError, but got {:?}", result);
    }
}
