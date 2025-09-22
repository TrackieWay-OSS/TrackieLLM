/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_runner_private.h
 *
 * Private header for the LLM runner C implementation. This should not be
 * included by any files outside of the ai_models C implementation.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_AI_MODELS_TK_RUNNER_PRIVATE_H
#define TRACKIELLM_AI_MODELS_TK_RUNNER_PRIVATE_H

#include "ai_models/tk_model_runner.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"
#include "llama.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// --- Internal Structs ---

typedef struct {
    char* role;
    char* content;
} tk_llm_history_entry_t;

struct tk_llm_runner_s {
    struct llama_context* ctx;
    struct llama_model* model; // Non-owning pointer
    struct llama_sampling_context* sctx;
    struct llama_grammar* grammar;

    tk_llm_config_t config;
    char* system_prompt;
    tk_llm_history_entry_t* history;
    size_t history_count;
    size_t history_capacity;

    bool is_processing;
    int32_t n_past;
};

// --- Forward declarations for helper functions ---

// from tk_runner_helpers.c
char* read_grammar_file(const char* file_path);
tk_error_code_t init_history(tk_llm_runner_t* runner);
void clear_history(tk_llm_runner_t* runner);

#endif // TRACKIELLM_AI_MODELS_TK_RUNNER_PRIVATE_H
