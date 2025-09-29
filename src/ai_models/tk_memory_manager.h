/*
 * Copyright (C) 2025 TrackieWay-OSS
 *
 * This file is part of TrackieLLM.
 *
 * This file defines the public API for the Memory Manager module, which is
 * responsible for monitoring system and GPU memory and making intelligent
 * decisions about model allocation and eviction.
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef TK_MEMORY_MANAGER_H
#define TK_MEMORY_MANAGER_H

#include "utils/tk_error_handling.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for the memory manager.
typedef struct tk_memory_manager_s tk_memory_manager_t;

// Forward declaration for the internal model representation.
// This avoids a circular dependency with tk_model_loader_private.h
struct tk_internal_model_t;

/**
 * @brief Configuration for the memory manager.
 */
typedef struct {
    uint64_t max_ram_mb;        // Maximum RAM to be managed (in MB).
    uint64_t max_vram_mb;       // Maximum VRAM to be managed (in MB).
    // Eviction policy can be added here, e.g., "LRU", "LFU", "priority_based"
} tk_memory_manager_config_t;

/**
 * @brief Creates a new memory manager instance.
 *
 * @param[out] out_manager Pointer to receive the created manager handle.
 * @param[in] config Configuration for the memory manager.
 * @return `TK_SUCCESS` on success.
 */
tk_error_code_t tk_memory_manager_create(tk_memory_manager_t** out_manager, const tk_memory_manager_config_t* config);

/**
 * @brief Destroys a memory manager instance.
 *
 * @param[in,out] manager Pointer to the manager handle to be destroyed.
 */
void tk_memory_manager_destroy(tk_memory_manager_t** manager);

/**
 * @brief Selects a model to evict to make space for a new model.
 *
 * This function implements the core eviction strategy. It first checks if
 * an eviction is necessary by comparing the required bytes with available memory.
 * If an eviction is needed, it analyzes the list of candidate models (those
 * with a reference count of zero) and selects the best one to remove.
 *
 * The strategy is a hybrid of **Least Recently Used (LRU)** and a "best-fit"
 * approach. It identifies the subset of candidates that are large enough to
 * free the required space and, from that subset, returns the one that was
 * loaded the longest time ago (the LRU candidate).
 *
 * @param[in] manager The memory manager instance.
 * @param[in] candidates An array of pointers to model handles that can be evicted.
 * @param[in] candidate_count The number of models in the candidate list.
 * @param[in] required_bytes The size of the new model that needs to be loaded.
 * @return A pointer to the `tk_internal_model_t` handle of the model to evict.
 *         Returns `NULL` if no eviction is necessary.
 *         Returns `NULL` if no single candidate is large enough to free the
 *         required space.
 */
struct tk_internal_model_t* tk_memory_manager_select_model_for_eviction(
    tk_memory_manager_t* manager,
    struct tk_internal_model_t** candidates,
    size_t candidate_count,
    uint64_t required_bytes
);

/**
 * @brief Notifies the memory manager that a model has been loaded.
 *
 * @param[in] manager The memory manager instance.
 * @param[in] model_handle Handle to the model that was loaded.
 */
void tk_memory_manager_confirm_allocation(tk_memory_manager_t* manager, struct tk_internal_model_t* model_handle);

/**
 * @brief Notifies the memory manager that a model has been unloaded.
 *
 * @param[in] manager The memory manager instance.
 * @param[in] model_handle Handle to the model that was unloaded.
 */
void tk_memory_manager_confirm_deallocation(tk_memory_manager_t* manager, struct tk_internal_model_t* model_handle);


#ifdef __cplusplus
}
#endif

#endif // TK_MEMORY_MANAGER_H