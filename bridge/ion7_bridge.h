/*
 * ion7_bridge.h - Public API of the ion7-core C bridge.
 *
 * Copyright (C) 2026 Ion7 Project Contributors
 * SPDX-License-Identifier: MIT
 *
 * This file is part of ion7-core.
 *
 * ion7-core is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ion7-core is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with ion7-core. If not, see <https://www.gnu.org/licenses/>.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * STABILITY CONTRACT - ion7-core v1.x
 *
 * Every function declared here is stable across minor versions.
 * Breaking changes require a major version bump (2.0).
 * Downstream libraries (ion7-llm, ion7-embed, ...) depend on this.
 *
 * Architecture:
 *   The bridge wraps two categories of llama.cpp functions:
 *
 *   1. PARAM CONSTRUCTORS - llama_*_params structs change between llama.cpp
 *      releases; the bridge absorbs those changes so LuaJIT FFI does not.
 *
 *   2. ERGONOMIC HELPERS - multi-out-param functions and batch operations
 *      that are awkward to call directly from LuaJIT FFI.
 *
 *   Everything else is called directly via cdef in ffi/types.lua.
 *
 * Naming convention : ion7_<noun>_<verb>
 * All parameters    : primitive C types (int, float, char*, size_t).
 * Linkage           : pure C - no C++, no STL, no exceptions.
 * ──────────────────────────────────────────────────────────────────────────
 */

#ifndef ION7_BRIDGE_H
#define ION7_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "llama.h"

/* =========================================================================
 * Version & info
 * ======================================================================= */

/** Returns the bridge version string, e.g. "1.0.0". */
const char* ion7_bridge_version(void);

/** Returns the llama.cpp system info string (CPU features, backends, etc.). */
const char* ion7_llama_info(void);

/* =========================================================================
 * Backend lifecycle
 * ======================================================================= */

/** Initialise the llama.cpp + ggml backend. Call once at startup. */
void ion7_backend_init(void);

/** Release all global llama.cpp resources. Call once at shutdown. */
void ion7_backend_free(void);

/**
 * Set the minimum log verbosity for llama.cpp internal messages.
 * @param level  0=silent  1=error  2=warn  3=info  4=debug
 */
void ion7_set_log_level(int level);

/* =========================================================================
 * Capability detection
 * ======================================================================= */

int ion7_supports_mmap          (void); /**< 1 if mmap loading is supported. */
int ion7_supports_mlock         (void); /**< 1 if mlock is supported. */
int ion7_supports_gpu_offload   (void); /**< 1 if GPU offload is available. */
int ion7_supports_rpc           (void); /**< 1 if the RPC backend is available. */
int ion7_max_devices            (void); /**< Maximum number of GPU devices. */
int ion7_max_parallel_sequences (void); /**< Maximum parallel sequences. */

/* =========================================================================
 * Model loading & management
 * ======================================================================= */

/**
 * Load a GGUF model from disk.
 *
 * @param path          Path to the .gguf file.
 * @param n_gpu_layers  Layers to offload: 0=CPU only, -1=all layers.
 * @param use_mmap      Map the file into memory instead of reading it.
 * @param use_mlock     Lock mapped pages in RAM to prevent swapping.
 * @param vocab_only    Load vocabulary only, skip model weights.
 * @return              Opaque model pointer, or NULL on failure.
 */
struct llama_model* ion7_model_load(const char* path, int32_t n_gpu_layers,
                                     int use_mmap, int use_mlock, int vocab_only);

/**
 * Load a multi-shard model from an explicit list of GGUF paths.
 * Use when shards do not follow the standard auto-discovery naming pattern.
 * Paths must be provided in correct shard order.
 *
 * @param paths         Array of GGUF file paths.
 * @param n_paths       Number of paths.
 * @param n_gpu_layers  Layers to offload: 0=CPU only, -1=all layers.
 */
struct llama_model* ion7_model_load_splits(const char** paths, size_t n_paths,
                                            int32_t n_gpu_layers);

/**
 * Load a model from an already-open file descriptor.
 * The caller owns the fd; this function does not close it.
 */
struct llama_model* ion7_model_load_fd(int fd, int n_gpu_layers);

/** Free a model and release all associated resources. Safe to call with NULL. */
void ion7_model_free(struct llama_model* model);

/** Save a model to a GGUF file. */
void ion7_model_save(const struct llama_model* model, const char* path);

/**
 * Quantize a GGUF model file.
 *
 * @param path_in   Source GGUF path.
 * @param path_out  Destination GGUF path.
 * @param ftype     Target quantization type (llama_ftype enum value).
 * @param n_threads Worker thread count. 0 = use hardware concurrency.
 * @param pure      Quantize all tensors uniformly (no mixed precision).
 * @param allow_req Allow re-quantizing already-quantized tensors.
 * @param dry_run   Compute the quantization plan without writing output.
 * @return          0 on success, non-zero on failure.
 */
int ion7_model_quantize(const char* path_in, const char* path_out,
                         int ftype, int n_threads,
                         int pure, int allow_req, int dry_run);

/* ---- Model introspection ------------------------------------------------ */

int         ion7_model_desc         (const struct llama_model* m, char* buf, size_t sz);
uint64_t    ion7_model_size         (const struct llama_model* m);
uint64_t    ion7_model_n_params     (const struct llama_model* m);
int32_t     ion7_model_n_ctx_train  (const struct llama_model* m);
int32_t     ion7_model_n_embd       (const struct llama_model* m);
int32_t     ion7_model_n_embd_inp   (const struct llama_model* m);
int32_t     ion7_model_n_embd_out   (const struct llama_model* m);
int32_t     ion7_model_n_layer      (const struct llama_model* m);
int32_t     ion7_model_n_head       (const struct llama_model* m);
int32_t     ion7_model_n_head_kv    (const struct llama_model* m);
int32_t     ion7_model_n_swa        (const struct llama_model* m);
int32_t     ion7_model_n_cls_out    (const struct llama_model* m);
int         ion7_model_has_encoder  (const struct llama_model* m);
int         ion7_model_has_decoder  (const struct llama_model* m);
int         ion7_model_is_recurrent (const struct llama_model* m);
int         ion7_model_is_hybrid    (const struct llama_model* m);
int         ion7_model_is_diffusion (const struct llama_model* m);
float       ion7_model_rope_freq_scale_train(const struct llama_model* m);

/**
 * Returns the model's RoPE type as a string.
 * @return One of: "none" | "norm" | "neox" | "mrope" | "imrope" | "vision" | "unknown"
 */
const char* ion7_model_rope_type          (const struct llama_model* m);

/** Returns a classifier output label by index. NULL if i >= n_cls_out. */
const char* ion7_model_cls_label          (const struct llama_model* m, uint32_t i);

/** Returns the decoder start token ID, or -1 if not applicable. */
int32_t     ion7_model_decoder_start_token(const struct llama_model* m);

/**
 * Returns the model's chat template string for the given variant name.
 * @param name  Template variant (e.g. "tool_use"), or NULL for the default.
 * @return      Template string, or NULL if none is embedded in the model.
 */
const char* ion7_model_chat_template(const struct llama_model* m, const char* name);

/** Look up a GGUF metadata value by key. Returns bytes written, or -1. */
int ion7_model_meta_val    (const struct llama_model* m, const char* key,
                              char* buf, size_t sz);

/** Returns the number of GGUF metadata entries. */
int ion7_model_meta_count  (const struct llama_model* m);

/** Returns the metadata key at index i. Returns bytes written, or -1. */
int ion7_model_meta_key_at (const struct llama_model* m, int32_t i,
                              char* buf, size_t sz);

/** Returns the metadata value at index i. Returns bytes written, or -1. */
int ion7_model_meta_val_at (const struct llama_model* m, int32_t i,
                              char* buf, size_t sz);

/* =========================================================================
 * VRAM auto-fit
 * ======================================================================= */

/**
 * Probe available device memory and compute the maximum n_gpu_layers and
 * n_ctx that fit without OOM. Not thread-safe; call before loading any model.
 *
 * @param path          Path to the GGUF file (used for size estimation).
 * @param n_gpu_layers  Output: recommended GPU layer count.
 * @param n_ctx         In/out: requested context size, may be reduced to fit.
 * @param n_ctx_min     Minimum acceptable context size.
 * @return              0=success, 1=cannot fit, 2=error.
 */
int ion7_params_fit(const char* path, int32_t* n_gpu_layers,
                     uint32_t* n_ctx, uint32_t n_ctx_min);

/* =========================================================================
 * Context creation
 * ======================================================================= */

/**
 * Create an inference context.
 *
 * @param type_k  K cache element type (GGML_TYPE_* integer value).
 *                1=F16 (default), 30=BF16, 8=Q8_0, 2=Q4_0, 3=Q4_1,
 *                6=Q5_0, 7=Q5_1, 20=IQ4_NL. 0 = llama.cpp default.
 * @param type_v  V cache element type. Same values as type_k.
 *                Q4_K/Q5_K for the V cache require the library to be
 *                compiled with GGML_CUDA_FA_ALL_QUANTS.
 */
struct llama_context* ion7_context_create(
    struct llama_model* model,
    uint32_t n_ctx,
    uint32_t n_batch,
    uint32_t n_ubatch,
    uint32_t n_seq_max,
    int32_t  n_threads,
    int32_t  n_threads_batch,
    int      flash_attn,
    int      offload_kqv,
    int      op_offload,
    int      no_perf,
    int      type_k,
    int      type_v,
    int      swa_full,
    int      kv_unified);

/**
 * Create an embedding context with output pooling.
 *
 * @param pooling  Pooling strategy (llama_pooling_type):
 *                 0=none, 1=mean, 2=cls, 3=last, 4=rank.
 */
struct llama_context* ion7_embedding_context_create(
    struct llama_model* model,
    uint32_t n_ctx,
    uint32_t n_seq_max,
    int32_t  n_threads,
    int      pooling);

/** Free a context. Safe to call with NULL. */
void ion7_context_free(struct llama_context* ctx);

/* =========================================================================
 * KV cache management
 * ======================================================================= */

/** Clear the entire KV cache across all sequences. */
void ion7_kv_clear(struct llama_context* ctx);

/**
 * Remove KV entries for seq_id in position range [p0, p1).
 * Negative p0 means from position 0; negative p1 means to end of cache.
 * Negative seq_id affects all sequences.
 */
void ion7_kv_seq_rm (struct llama_context* ctx,
                      int32_t seq_id, int32_t p0, int32_t p1);

/**
 * Copy KV entries from src to dst for positions in [p0, p1).
 * Used to fork sequences for parallel decoding or beam search.
 */
void ion7_kv_seq_cp (struct llama_context* ctx,
                      int32_t src, int32_t dst, int32_t p0, int32_t p1);

/** Discard all KV entries that do not belong to seq_id. */
void ion7_kv_seq_keep(struct llama_context* ctx, int32_t seq_id);

/**
 * Shift KV positions in [p0, p1) by delta for seq_id.
 * Only valid when llama_memory_can_shift() returns true.
 */
void ion7_kv_seq_shift(struct llama_context* ctx,
                         int32_t seq_id, int32_t p0, int32_t p1, int32_t delta);

/* =========================================================================
 * State persistence
 * ======================================================================= */

/** Returns the byte size of the full context state snapshot. */
size_t ion7_state_size(struct llama_context* ctx);

/** Serialise the full context state into buf. Returns bytes written. */
size_t ion7_state_get(struct llama_context* ctx, uint8_t* buf, size_t sz);

/** Restore context state from a buffer. Returns 1 on success. */
int    ion7_state_set(struct llama_context* ctx, const uint8_t* buf, size_t sz);

/** Save state to a file. Returns 1 on success. */
int    ion7_state_save_file(struct llama_context* ctx, const char* path,
                              const llama_token* tokens, size_t n);

/** Load state from a file. Returns 1 on success. */
int    ion7_state_load_file(struct llama_context* ctx, const char* path,
                              llama_token* out, size_t cap, size_t* n_out);

/** Returns the byte size of a single sequence's state. */
size_t ion7_state_seq_size(struct llama_context* ctx, int32_t seq_id);

/** Save a single sequence's state to a file. Returns 1 on success. */
int    ion7_state_seq_save(struct llama_context* ctx, const char* path,
                             int32_t seq_id);

/** Load a single sequence's state from a file. Returns 1 on success. */
int    ion7_state_seq_load(struct llama_context* ctx, const char* path,
                             int32_t dest_seq_id);

/* =========================================================================
 * LoRA adapters
 * ======================================================================= */

/** Load a LoRA adapter from a GGUF file. Returns NULL on failure. */
struct llama_adapter_lora* ion7_lora_load(struct llama_model* model,
                                           const char* path);

/** Free a LoRA adapter. Safe to call with NULL. */
void ion7_lora_free(struct llama_adapter_lora* adapter);

/**
 * Apply a LoRA adapter to a context.
 * @param scale  Blend strength: 1.0=full effect, 0.0=no effect.
 * @return       0 on success.
 */
int ion7_lora_apply(struct llama_context* ctx,
                     struct llama_adapter_lora* adapter, float scale);

/** Remove all LoRA adapters from a context. Returns 0 on success. */
int ion7_lora_remove(struct llama_context* ctx);

/** Read a metadata value from a LoRA adapter. Returns bytes written, or -1. */
int ion7_lora_meta_val(const struct llama_adapter_lora* a,
                        const char* key, char* buf, size_t sz);

/* =========================================================================
 * Performance monitoring
 * ======================================================================= */

/** Print a performance summary to stderr. */
void ion7_perf_print(struct llama_context* ctx);

/** Reset all performance counters. */
void ion7_perf_reset(struct llama_context* ctx);

/**
 * Read all performance counters in one call. Pass NULL for unused fields.
 *
 * @param t_load_ms    Model load time (ms).
 * @param t_p_eval_ms  Prompt processing time (ms).
 * @param t_eval_ms    Token generation time (ms).
 * @param n_p_eval     Prompt tokens processed.
 * @param n_eval       Tokens generated.
 * @param n_reused     KV cache positions reused.
 */
void ion7_perf_get(struct llama_context* ctx,
                    double*  t_load_ms,   double*  t_p_eval_ms,
                    double*  t_eval_ms,   int32_t* n_p_eval,
                    int32_t* n_eval,      int32_t* n_reused);

/* =========================================================================
 * Chat template
 * ======================================================================= */

/**
 * Apply a Jinja-like chat template to a message array.
 *
 * @param tmpl     Template string. NULL uses the model's embedded template.
 * @param chat     Array of llama_chat_message structs.
 * @param n_msg    Number of messages.
 * @param add_ass  Append the assistant turn prefix to the output.
 * @param buf      Output buffer.
 * @param length   Output buffer size in bytes.
 * @return         Total bytes needed. If > length, resize and call again.
 */
int32_t ion7_chat_apply_template(const char* tmpl,
                                  const llama_chat_message* chat, size_t n_msg,
                                  bool add_ass, char* buf, int32_t length);

/* =========================================================================
 * Threadpool
 * ======================================================================= */

typedef struct ggml_threadpool ion7_threadpool_t;

/**
 * Create a CPU threadpool with the given number of worker threads.
 * A shared threadpool amortises thread creation cost across multiple contexts.
 */
ion7_threadpool_t* ion7_threadpool_create(int n_threads);

/** Free a threadpool. Must be detached from all contexts first. */
void ion7_threadpool_free      (ion7_threadpool_t* tp);

/** Returns the number of worker threads in the pool. */
int  ion7_threadpool_n_threads (ion7_threadpool_t* tp);

/** Pause all worker threads. */
void ion7_threadpool_pause (ion7_threadpool_t* tp);

/** Resume paused worker threads. */
void ion7_threadpool_resume(ion7_threadpool_t* tp);

/**
 * Attach a threadpool to a context.
 * @param tp        Primary pool (generation).
 * @param tp_batch  Batch pool (prompt processing). NULL reuses tp.
 */
void ion7_threadpool_attach(struct llama_context* ctx,
                              ion7_threadpool_t* tp,
                              ion7_threadpool_t* tp_batch);

/** Detach the threadpool from a context. */
void ion7_threadpool_detach(struct llama_context* ctx);

/* =========================================================================
 * Custom samplers
 * ======================================================================= */

/** Callback signatures for custom sampler implementations. */
typedef void (*ion7_sampler_apply_fn) (llama_token_data_array* cur_p, void* ud);
typedef void (*ion7_sampler_accept_fn)(llama_token token,              void* ud);
typedef void (*ion7_sampler_reset_fn) (void*                           ud);
typedef void (*ion7_sampler_free_fn)  (void*                           ud);

/**
 * Create a custom llama_sampler driven by caller-supplied callbacks.
 *
 * The sampler is owned by the chain after being added to it.
 * Do not free it manually once it has been added to a chain.
 *
 * @param name      Name shown in performance output.
 * @param apply_fn  Required. Modify or select from cur_p candidates.
 * @param accept_fn Called after the chain accepts a token. May be NULL.
 * @param reset_fn  Called on chain reset. May be NULL.
 * @param free_fn   Called when the sampler is freed. May be NULL.
 * @param userdata  Passed to every callback. May be NULL.
 * @return          Sampler pointer, or NULL on allocation failure.
 */
struct llama_sampler* ion7_sampler_create(
    const char*             name,
    ion7_sampler_apply_fn   apply_fn,
    ion7_sampler_accept_fn  accept_fn,
    ion7_sampler_reset_fn   reset_fn,
    ion7_sampler_free_fn    free_fn,
    void*                   userdata);

/* =========================================================================
 * Diagnostics
 * ======================================================================= */

/** Print bridge and llama.cpp struct sizes to stderr (ABI validation). */
void ion7_print_struct_sizes(void);

#ifdef __cplusplus
}
#endif
#endif /* ION7_BRIDGE_H */
