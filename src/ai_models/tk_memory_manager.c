/*
 * Copyright (C) 2025 TrackieWay-OSS
 *
 * This file is part of TrackieLLM.
 *
 * This file implements the Memory Manager module.
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include "tk_memory_manager.h"
#include "tk_model_loader_private.h" // For tk_internal_model_t definition
#include "utils/tk_logging.h"
#include <stdlib.h>

struct tk_memory_manager_s {
    tk_memory_manager_config_t config;
    uint64_t current_ram_usage_bytes;
    uint64_t current_vram_usage_bytes;
};

tk_error_code_t tk_memory_manager_create(tk_memory_manager_t** out_manager, const tk_memory_manager_config_t* config) {
    if (!out_manager || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_memory_manager_t* manager = calloc(1, sizeof(tk_memory_manager_t));
    if (!manager) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    manager->config = *config;
    manager->current_ram_usage_bytes = 0;
    manager->current_vram_usage_bytes = 0;

    *out_manager = manager;
    TK_LOG_INFO("Memory manager created. RAM limit: %llu MB, VRAM limit: %llu MB",
                manager->config.max_ram_mb, manager->config.max_vram_mb);
    return TK_SUCCESS;
}

void tk_memory_manager_destroy(tk_memory_manager_t** manager) {
    if (!manager || !*manager) {
        return;
    }
    free(*manager);
    *manager = NULL;
    TK_LOG_INFO("Memory manager destroyed.");
}

struct tk_internal_model_t* tk_memory_manager_select_model_for_eviction(
    tk_memory_manager_t* manager,
    struct tk_internal_model_t** candidates,
    size_t candidate_count,
    uint64_t required_bytes
) {
    if (!manager || !candidates) {
        return NULL;
    }

    uint64_t max_ram_bytes = manager->config.max_ram_mb * 1024 * 1024;
    uint64_t available_bytes = (max_ram_bytes > manager->current_ram_usage_bytes)
                               ? max_ram_bytes - manager->current_ram_usage_bytes
                               : 0;

    if (required_bytes <= available_bytes) {
        TK_LOG_INFO("No eviction needed. Required: %llu, Available: %llu", required_bytes, available_bytes);
        return NULL; // Enough space, no eviction needed.
    }

    TK_LOG_INFO("Memory pressure detected. Required: %llu, Available: %llu. Searching for eviction candidates.",
                required_bytes, available_bytes);

    struct tk_internal_model_t* best_candidate = NULL;
    time_t oldest_time = time(NULL);

    for (size_t i = 0; i < candidate_count; i++) {
        struct tk_internal_model_t* candidate = candidates[i];
        uint64_t candidate_size = candidate->metadata.size_bytes;

        // Check if this candidate is large enough to satisfy the requirement
        if (available_bytes + candidate_size >= required_bytes) {
            // It's a potential candidate. Is it better than the current best?
            // "Better" means it's the least recently used (oldest).
            if (candidate->load_time < oldest_time) {
                oldest_time = candidate->load_time;
                best_candidate = candidate;
            }
        }
    }

    if (best_candidate) {
        TK_LOG_INFO("Selected model for eviction: %s (Size: %llu, Loaded: %ld)",
                    best_candidate->path, best_candidate->metadata.size_bytes, best_candidate->load_time);
    } else {
        TK_LOG_ERROR("Could not find a suitable model to evict to free %llu bytes.", required_bytes);
    }

    return best_candidate;
}

void tk_memory_manager_confirm_allocation(tk_memory_manager_t* manager, struct tk_internal_model_t* model_handle) {
    if (!manager || !model_handle) {
        return;
    }
    // Assuming all models are loaded into RAM for now. VRAM logic would be more complex.
    uint64_t model_size = model_handle->metadata.size_bytes;
    manager->current_ram_usage_bytes += model_size;
    TK_LOG_INFO("Confirmed allocation for model '%s' (%llu bytes). Current RAM usage: %llu bytes",
                model_handle->path, model_size, manager->current_ram_usage_bytes);
}

void tk_memory_manager_confirm_deallocation(tk_memory_manager_t* manager, struct tk_internal_model_t* model_handle) {
    if (!manager || !model_handle) {
        return;
    }
    uint64_t model_size = model_handle->metadata.size_bytes;
    if (manager->current_ram_usage_bytes >= model_size) {
        manager->current_ram_usage_bytes -= model_size;
    } else {
        manager->current_ram_usage_bytes = 0;
    }
    TK_LOG_INFO("Confirmed deallocation for model '%s' (%llu bytes). Current RAM usage: %llu bytes",
                model_handle->path, model_size, manager->current_ram_usage_bytes);
}