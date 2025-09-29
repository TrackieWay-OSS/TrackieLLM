/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_runner_helpers.c
 *
 * Implements helper functions for the LLM runner, such as history management
 * and file reading.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_runner_private.h"

// --- Private Helper Functions ---

static char* duplicate_string(const char* src) {
    if (!src) return NULL;
    size_t len = strlen(src);
    char* dup = malloc(len + 1);
    if (!dup) return NULL;
    memcpy(dup, src, len + 1);
    return dup;
}

static void free_string(char* str) {
    if (str) free(str);
}

void clear_history(tk_llm_runner_t* runner) {
    if (!runner) return;
    for (size_t i = 0; i < runner->history_count; i++) {
        free_string(runner->history[i].role);
        free_string(runner->history[i].content);
    }
    runner->history_count = 0;
}

tk_error_code_t init_history(tk_llm_runner_t* runner) {
    if (!runner) return TK_ERROR_INVALID_ARGUMENT;
    runner->history_capacity = 32;
    runner->history = calloc(runner->history_capacity, sizeof(tk_llm_history_entry_t));
    if (!runner->history) return TK_ERROR_OUT_OF_MEMORY;
    runner->history_count = 0;
    return TK_SUCCESS;
}

char* read_grammar_file(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        TK_LOG_ERROR("Could not open grammar file: %s", file_path);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (length <= 0) {
        fclose(file);
        return NULL;
    }
    char* buffer = malloc(length + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }
    if (fread(buffer, 1, length, file) != (size_t)length) {
        fclose(file);
        free(buffer);
        return NULL;
    }
    buffer[length] = '\0';
    fclose(file);
    return buffer;
}


// --- Public Helper Functions ---

tk_error_code_t tk_llm_runner_add_tool_response(
    tk_llm_runner_t* runner,
    const char* tool_name,
    const char* tool_output
) {
    if (!runner || !tool_name || !tool_output) return TK_ERROR_INVALID_ARGUMENT;

    // 1. Format the tool response into a string for the LLM.
    // The exact format is crucial and should match the prompt engineering strategy.
    // We use a simple structured format here. Max size for safety.
    char formatted_response[4096];
    int written = snprintf(formatted_response, sizeof(formatted_response),
                           "[TOOL_RESULT] name: \"%s\", output: %s [/TOOL_RESULT]",
                           tool_name, tool_output);
    if (written < 0 || (size_t)written >= sizeof(formatted_response)) {
        TK_LOG_ERROR("Failed to format tool response or output was too long.");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    TK_LOG_INFO("Injecting tool response into context: %s", formatted_response);

    // 2. Tokenize the formatted response.
    int32_t n_ctx = llama_n_ctx(runner->ctx);
    llama_token* tokens = malloc(sizeof(llama_token) * n_ctx);
    if (!tokens) return TK_ERROR_OUT_OF_MEMORY;

    int n_tokens = llama_tokenize(runner->model, formatted_response, strlen(formatted_response), tokens, n_ctx, false, true);
    if (n_tokens < 0) {
        free(tokens);
        TK_LOG_ERROR("Failed to tokenize tool response.");
        return TK_ERROR_INFERENCE_FAILED;
    }

    // 3. Decode the tokens into the current context.
    if (llama_decode(runner->ctx, llama_batch_get_one(tokens, n_tokens, runner->n_past, 0))) {
        free(tokens);
        TK_LOG_ERROR("llama_decode failed on tool response.");
        return TK_ERROR_INFERENCE_FAILED;
    }

    // 4. Update the context position.
    runner->n_past += n_tokens;
    free(tokens);

    // The runner is now ready to generate the next text token based on the tool's output.
    runner->is_processing = true;

    return TK_SUCCESS;
}

tk_error_code_t tk_llm_runner_reset_context(tk_llm_runner_t* runner) {
    if (!runner) return TK_ERROR_INVALID_ARGUMENT;

    clear_history(runner);
    if (runner->ctx) {
        llama_kv_cache_clear(runner->ctx);
    }
    runner->n_past = 0;

    return TK_SUCCESS;
}

void tk_llm_result_destroy(tk_llm_result_t** result) {
    // This function is now mostly obsolete as the Rust side will be responsible
    // for parsing the full streamed response and creating its own result types.
    if (!result || !*result) return;
    free(*result);
    *result = NULL;
}
