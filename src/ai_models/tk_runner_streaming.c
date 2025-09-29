/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_runner_streaming.c
 *
 * Implements the streaming inference API for the LLM runner.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_runner_private.h"

tk_error_code_t tk_llm_runner_prepare_generation(
    tk_llm_runner_t* runner,
    const char* prompt,
    bool use_tool_grammar
) {
    if (!runner || !prompt) return TK_ERROR_INVALID_ARGUMENT;

    int32_t n_ctx = llama_n_ctx(runner->ctx);
    llama_token* tokens = malloc(sizeof(llama_token) * n_ctx);
    if (!tokens) return TK_ERROR_OUT_OF_MEMORY;

    int n_tokens = llama_tokenize(runner->model, prompt, strlen(prompt), tokens, n_ctx, true, false);
    if (n_tokens < 0) {
        free(tokens);
        TK_LOG_ERROR("Failed to tokenize prompt.");
        return TK_ERROR_INFERENCE_FAILED;
    }

    llama_kv_cache_clear(runner->ctx);
    runner->n_past = 0;

    if (llama_decode(runner->ctx, llama_batch_get_one(tokens, n_tokens, runner->n_past, 0))) {
        free(tokens);
        TK_LOG_ERROR("llama_decode failed on prompt.");
        return TK_ERROR_INFERENCE_FAILED;
    }
    runner->n_past += n_tokens;
    free(tokens);

    llama_sampling_reset(runner->sctx);
    if (use_tool_grammar && runner->grammar) {
        llama_sampling_set_grammar(runner->sctx, runner->grammar);
    } else {
        llama_sampling_set_grammar(runner->sctx, NULL);
    }

    runner->is_processing = true;
    return TK_SUCCESS;
}

// Special pointer value to indicate a tool call has completed.
// The Rust side will check for this specific address.
#define TK_TOOL_CALL_TOKEN ((const char*)1)

const char* tk_llm_runner_generate_next_token(tk_llm_runner_t* runner) {
    if (!runner || !runner->is_processing) return NULL;

    llama_token id = llama_sampling_sample(runner->sctx, runner->ctx, NULL, 0);
    llama_sampling_accept(runner->sctx, runner->ctx, id, true);

    if (id == llama_token_eos(runner->model)) {
        runner->is_processing = false;
        return NULL; // End of Stream
    }

    // Check if the grammar rule for a tool call has just been completed.
    if (runner->grammar != NULL && llama_sampling_prev_token_is_term(runner->sctx)) {
        TK_LOG_INFO("Tool call grammar rule completed. Signaling tool call.");
        // We don't advance the context yet. The tool call content is in the sampling context.
        // The Rust side will now read the tool call string, and then call add_tool_response.
        runner->is_processing = false; // Pause processing until tool response is added
        return TK_TOOL_CALL_TOKEN;
    }

    if (llama_decode(runner->ctx, llama_batch_get_one(&id, 1, runner->n_past, 0))) {
        TK_LOG_ERROR("llama_decode failed during token generation.");
        runner->is_processing = false;
        return NULL;
    }
    runner->n_past++;

    return llama_token_to_piece(runner->ctx, id);
}
