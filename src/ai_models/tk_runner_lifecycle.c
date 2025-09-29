/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_runner_lifecycle.c
 *
 * Implements the lifecycle management (create/destroy) for the LLM runner.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_runner_private.h"

// --- Public API Implementation ---

#include "tk_model_loader_private.h" // For tk_internal_model_t

tk_error_code_t tk_llm_runner_create(
    tk_llm_runner_t** out_runner,
    void* model_handle,
    const tk_llm_config_t* config
) {
    if (!out_runner || !model_handle || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_internal_model_t* internal_model = (tk_internal_model_t*)model_handle;
    if (internal_model->format != TK_MODEL_FORMAT_GGUF) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    tk_gguf_handle_t* gguf_handle = (tk_gguf_handle_t*)internal_model->handle;
    struct llama_model* model = gguf_handle->model;

    tk_llm_runner_t* runner = calloc(1, sizeof(tk_llm_runner_t));
    if (!runner) return TK_ERROR_OUT_OF_MEMORY;

    runner->model = model; // Non-owning pointer
    runner->config = *config;
    if (config->system_prompt) {
        char* p_prompt = strdup(config->system_prompt);
        if(!p_prompt) {
            free(runner);
            return TK_ERROR_OUT_OF_MEMORY;
        }
        runner->system_prompt = p_prompt;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config->context_size > 0 ? config->context_size : 4096;
    ctx_params.seed = config->random_seed;

    runner->ctx = llama_new_context_with_model(model, ctx_params);
    if (!runner->ctx) {
        TK_LOG_ERROR("Failed to create LLM context");
        free(runner->system_prompt);
        free(runner);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }

    char* grammar_str = read_grammar_file("src/ai_models/grammars/tool_call.gbnf");
    if (grammar_str) {
        const char* rules[] = { grammar_str };
        struct llama_grammar_params grammar_params = {
            .n_rules = 1,
            .start_rule_index = 0,
            .rules = rules
        };
        runner->grammar = llama_grammar_init_from_params(grammar_params);
        free(grammar_str);
        if (!runner->grammar) {
            TK_LOG_WARN("Failed to compile GBNF grammar for tool calls.");
        }
    } else {
        TK_LOG_WARN("Could not read tool call grammar file.");
    }

    struct llama_sampling_params sparams = llama_sampling_default_params();
    runner->sctx = llama_sampling_init(sparams);
    if (!runner->sctx) {
        if(runner->grammar) llama_grammar_free(runner->grammar);
        llama_free(runner->ctx);
        free(runner->system_prompt);
        free(runner);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    init_history(runner);

    *out_runner = runner;
    return TK_SUCCESS;
}

void tk_llm_runner_destroy(tk_llm_runner_t** runner) {
    if (!runner || !*runner) return;

    tk_llm_runner_t* r = *runner;

    if (r->grammar) llama_grammar_free(r->grammar);
    if (r->sctx) llama_sampling_free(r->sctx);
    if (r->ctx) llama_free(r->ctx);

    free(r->system_prompt);
    clear_history(r);
    free(r->history);
    free(r);

    *runner = NULL;
}
