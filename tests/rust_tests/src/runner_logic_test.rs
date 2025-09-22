//! Tests for the GgufRunner logic, using a mocked FFI layer.

// Import the mock FFI functions into a module named `ffi` so that the
// `ai_models` crate's `use super::ffi` statements resolve to our mock.
use super::mock_ffi as ffi;

// We need to bring the real `ai_models` types into scope.
use ai_models::prelude::*;
use ai_models::loader::Model; // Assuming Model is public
use std::ptr::NonNull;

#[tokio::test]
async fn test_gguf_runner_streaming_success() {
    // --- Setup ---
    // Use a dummy pointer for the Model handle, since the mock FFI doesn't use it.
    let dummy_handle = NonNull::new(1 as *mut _).unwrap();
    let model = Model {
        handle: dummy_handle,
        // The loader pointer is not used in the runner, so we can null it out.
        loader_ptr: std::ptr::null_mut(),
    };

    let config = LlmConfig {
        context_size: 1024,
        system_prompt: "System",
        random_seed: 0,
    };

    // --- Act ---
    let mut runner = GgufRunner::new(&model, &config).expect("Runner creation should succeed with mock");

    runner.prepare_response("prompt", false).expect("Prepare should succeed");

    let mut stream = runner.stream_response();

    // --- Assert ---
    // The mock FFI is configured to return "hello" 3 times.
    assert_eq!(stream.next().await.unwrap().unwrap(), "hello");
    assert_eq!(stream.next().await.unwrap().unwrap(), "hello");
    assert_eq!(stream.next().await.unwrap().unwrap(), "hello");

    // The stream should now be empty.
    assert!(stream.next().await.is_none());
}
