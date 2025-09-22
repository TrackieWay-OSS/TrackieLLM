//! Mock implementation of the C FFI layer for testing purposes.

use std::os::raw::{c_char, c_void};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, Ordering};

// --- Mock State & Constants ---

pub const TK_SUCCESS: i32 = 0;
pub const TK_ERROR_INVALID_ARGUMENT: i32 = -1;

// A global atomic flag to control the behavior of the mock functions.
pub static MOCK_SHOULD_SUCCEED: AtomicBool = AtomicBool::new(true);

// --- Mock Structs (Opaque pointers) ---

pub enum tk_model_loader_s {}
pub type tk_model_loader_t = tk_model_loader_s;
pub enum tk_llm_runner_s {}
pub type tk_llm_runner_t = tk_llm_runner_s;
pub enum llama_model {}

// --- Mock FFI Functions ---

#[no_mangle]
pub unsafe extern "C" fn tk_model_loader_create(
    out_loader: *mut *mut tk_model_loader_t,
    _config: *const c_void,
) -> i32 {
    if MOCK_SHOULD_SUCCEED.load(Ordering::SeqCst) {
        *out_loader = 1 as *mut tk_model_loader_t; // Dummy non-null pointer
        TK_SUCCESS
    } else {
        *out_loader = null_mut();
        TK_ERROR_INVALID_ARGUMENT
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_model_loader_destroy(_loader: *mut *mut tk_model_loader_t) {}

#[no_mangle]
pub unsafe extern "C" fn tk_model_loader_load_model(
    _loader: *mut tk_model_loader_t,
    _params: *const c_void,
    out_model_handle: *mut *mut c_void,
) -> i32 {
    if MOCK_SHOULD_SUCCEED.load(Ordering::SeqCst) {
        *out_model_handle = 2 as *mut c_void; // Dummy non-null pointer
        TK_SUCCESS
    } else {
        *out_model_handle = null_mut();
        TK_ERROR_INVALID_ARGUMENT
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_model_loader_unload_model(_loader: *mut tk_model_loader_t, _handle: *mut *mut c_void) {}

#[no_mangle]
pub unsafe extern "C" fn tk_llm_runner_create(
    out_runner: *mut *mut tk_llm_runner_t,
    _model: *mut llama_model,
    _config: *const c_void,
) -> i32 {
    if MOCK_SHOULD_SUCCEED.load(Ordering::SeqCst) {
        *out_runner = 3 as *mut tk_llm_runner_t; // Dummy non-null pointer
        TK_SUCCESS
    } else {
        *out_runner = null_mut();
        TK_ERROR_INVALID_ARGUMENT
    }
}

#[no_mangle]
pub unsafe extern "C" fn tk_llm_runner_destroy(_runner: *mut *mut tk_llm_runner_t) {}

#[no_mangle]
pub unsafe extern "C" fn tk_llm_runner_prepare_generation(
    _runner: *mut tk_llm_runner_t,
    _prompt: *const c_char,
    _use_tool_grammar: bool,
) -> i32 {
    if MOCK_SHOULD_SUCCEED.load(Ordering::SeqCst) {
        TK_SUCCESS
    } else {
        TK_ERROR_INVALID_ARGUMENT
    }
}

// A static buffer to hold the mock token string.
static MOCK_TOKEN: &[u8] = b"hello\0";

#[no_mangle]
pub unsafe extern "C" fn tk_llm_runner_generate_next_token(_runner: *mut tk_llm_runner_t) -> *const c_char {
    if MOCK_SHOULD_SUCCEED.load(Ordering::SeqCst) {
        // Return the mock token a few times, then stop.
        static mut COUNTER: i32 = 0;
        if COUNTER < 3 {
            COUNTER += 1;
            MOCK_TOKEN.as_ptr() as *const c_char
        } else {
            COUNTER = 0; // Reset for next test
            null()
        }
    } else {
        null()
    }
}
