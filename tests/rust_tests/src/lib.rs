#[cfg(test)]
mod model_loading_test;

#[cfg(test)]
#[path = "."]
mod runner_tests {
    // This will be the mock FFI layer
    mod mock_ffi;

    // This test will use the mock FFI
    mod runner_logic_test;
}
