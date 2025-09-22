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
    // This function will need a more complex implementation to fit the streaming model,
    // likely by adding a specific entry type to the history.
    TK_LOG_INFO("Tool response received for '%s'. Not yet added to history.", tool_name);
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
