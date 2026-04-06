#pragma once
/*
 * bridge_internal.hpp - private header shared between bridge_*.cpp files.
 *
 * NOT part of the public API (not included by ion7_bridge.h).
 * Used only by bridge_core.cpp, bridge_utils.cpp, etc. to share:
 *   - the 3 log globals
 *   - the unified log callback declaration
 *   - the ctx_mem() inline helper
 *
 * Do not include this from Lua-facing code or external consumers.
 */

#include "llama.h"

/* ── Log globals (defined once in bridge_core.cpp) ─────────────────────── */
extern int   g_log_level;
extern FILE* g_log_file;
extern int   g_log_timestamps;

/* ── Unified log callback (defined in bridge_core.cpp) ─────────────────── */
void ion7_log_dispatch(enum ggml_log_level level, const char* text, void* ud);

/* ── KV cache memory accessor ───────────────────────────────────────────── */
static inline llama_memory_t ctx_mem(struct llama_context* ctx)
{
    return llama_get_memory(ctx);
}
