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
uint32_t    ion7_model_n_cls_out    (const struct llama_model* m);
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

/**
 * Create an inference context with an eval callback.
 *
 * Same parameters as ion7_context_create, plus:
 * @param cb_eval           Called for every tensor computed during a forward
 *                          pass.  When ask=true: return true to compute the
 *                          tensor, false to skip.  When ask=false: tensor data
 *                          is ready and can be inspected / copied.
 * @param cb_eval_user_data Opaque pointer forwarded to cb_eval as ud.
 *                          May be NULL when the callback uses Lua closures.
 *
 * Use ion7_tensor_* helpers to safely inspect tensors inside the callback.
 * Useful for activation extraction (control vector building).
 */
struct llama_context* ion7_context_create_with_cb(
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
    int      kv_unified,
    void*    cb_eval,
    void*    cb_eval_user_data);

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

/* =========================================================================
 * ── C++ EXTENSIONS (libcommon — bridge v1.1) ─────────────────────────────
 * Requires libcommon. Available when ION7_BRIDGE_VERSION >= "1.1.0".
 * ======================================================================= */

/* ── Chat Templates (Jinja2 native) ────────────────────────────────────── */

/**
 * Opaque handle to a common_chat_templates instance.
 * Wraps the Jinja2 template engine embedded in the GGUF.
 */
typedef struct common_chat_templates ion7_chat_templates_t;

/**
 * Initialise chat templates from a loaded model.
 * @param tmpl_override  Override the model's embedded template. NULL = use model's.
 * @return               Opaque handle, or NULL on failure. Free with ion7_chat_templates_free.
 */
ion7_chat_templates_t* ion7_chat_templates_init(
    const struct llama_model* model,
    const char*               tmpl_override);

/** Free a chat templates handle. Safe to call with NULL. */
void ion7_chat_templates_free(ion7_chat_templates_t* t);

/**
 * Returns 1 if the model's Jinja2 template supports the enable_thinking parameter.
 * (Qwen3/3.5, DeepSeek-R1, and similar reasoning models return 1.)
 */
int ion7_chat_templates_support_thinking(const ion7_chat_templates_t* t);

/**
 * Apply a Jinja2 chat template with full parameter support.
 *
 * @param roles             Array of n_msgs role strings ("system","user","assistant").
 * @param contents          Array of n_msgs content strings.
 * @param n_msgs            Number of messages.
 * @param add_ass           1 = append assistant generation prefix.
 * @param enable_thinking   -1 = model default, 0 = disable thinking, 1 = enable thinking.
 * @param buf               Output buffer. May be NULL for size query.
 * @param buf_len           Size of buf in bytes.
 * @return                  Total bytes needed (including NUL). Resize and retry if > buf_len.
 */
int32_t ion7_chat_templates_apply(
    ion7_chat_templates_t* t,
    const char**           roles,
    const char**           contents,
    size_t                 n_msgs,
    int                    add_ass,
    int                    enable_thinking,
    char*                  buf,
    int32_t                buf_len);

/* ── Reasoning Budget ──────────────────────────────────────────────────── */

/**
 * Create a reasoning-budget sampler that hard-limits the token count inside
 * <think>...</think> blocks for Qwen3/3.5 and compatible models.
 *
 * Insert into a sampler chain via llama_sampler_chain_add().
 * The chain takes ownership; do not free manually after adding.
 *
 * @param model     Loaded model (used to tokenize delimiters).
 * @param n_budget  Max tokens inside the block.
 *                  0 = completely disable thinking (pre-fills empty block).
 *                 -1 = unlimited (passthrough, useful for dynamic control).
 * @return          llama_sampler*, or NULL on failure.
 */
struct llama_sampler* ion7_reasoning_budget_init(
    const struct llama_model* model,
    int32_t                   n_budget);

/* ── Training (llama_opt API) ──────────────────────────────────────────── */

/** Opaque training state handle returned by ion7_opt_init. */
typedef struct ion7_opt_state ion7_opt_state_t;

/**
 * Initialise training on a context.
 *
 * Must be called before ion7_opt_epoch. The context must have been created
 * with F32 or BF16 KV cache (quantized KV is not supported during training).
 *
 * @param optimizer  0 = AdamW (recommended), 1 = SGD.
 * @param lr         Learning rate (e.g. 1e-4 for LoRA, 1e-5 for full fine-tune).
 * @return           Opaque state handle. Free with ion7_opt_free when done.
 */
ion7_opt_state_t* ion7_opt_init(
    struct llama_context* ctx,
    struct llama_model*   model,
    int                   optimizer,
    float                 lr);

/** Release training state. */
void ion7_opt_free(ion7_opt_state_t* state);

/**
 * Create a training dataset from a pre-tokenised corpus.
 *
 * @param tokens    Flat array of llama_token (entire corpus).
 * @param n_tokens  Total token count.
 * @param n_ctx     Sequence length / context window.
 * @return          Dataset handle. Free with ion7_opt_dataset_free.
 */
ggml_opt_dataset_t ion7_opt_dataset_create(
    struct llama_context* ctx,
    const llama_token*    tokens,
    int64_t               n_tokens,
    int64_t               n_ctx);

/** Free a dataset. */
void ion7_opt_dataset_free(ggml_opt_dataset_t dataset);

/**
 * Run one training epoch over the dataset.
 *
 * @param val_split  Fraction of dataset held out for validation (0.0 = none).
 * @return           Average training loss, or -1.0 on error.
 */
float ion7_opt_epoch(
    struct llama_context* ctx,
    ggml_opt_dataset_t    dataset,
    float                 val_split);

/* =========================================================================
 * ── Context warmup (JIT GPU kernel compilation) ──────────────────────────
 * ======================================================================= */

/**
 * Warm up the context by running a single dummy decode.
 * Forces GPU backends to JIT-compile shaders before the real first decode,
 * cutting time-to-first-token by 2-3x on CUDA/Metal/Vulkan.
 * The KV cache is cleared after the warmup decode.
 * Call once after ion7_context_create, before any real decode.
 */
void ion7_context_warmup(struct llama_context* ctx);

/* =========================================================================
 * ── common_sampler - advanced sampling (DRY, XTC, grammar_lazy, ...) ────
 * ======================================================================= */

/**
 * Flat parameter struct for the common_sampler.
 * Declare with ffi.new("ion7_csampler_params_t") and set fields directly.
 */
typedef struct {
    uint32_t seed;              /**< LLAMA_DEFAULT_SEED (0xFFFFFFFF) for random.      */
    int32_t  top_k;             /**< Keep top-k tokens. <= 0 = disabled.              */
    float    top_p;             /**< Nucleus sampling threshold. 1.0 = disabled.      */
    float    min_p;             /**< Min-p threshold. 0.0 = disabled.                 */
    float    xtc_probability;   /**< XTC apply probability. 0.0 = disabled.           */
    float    xtc_threshold;     /**< XTC removal threshold (default 0.1).             */
    float    temp;              /**< Sampling temperature (default 0.8).              */
    float    repeat_penalty;    /**< Repetition penalty. 1.0 = disabled.              */
    float    freq_penalty;      /**< Frequency penalty.  0.0 = disabled.              */
    float    pres_penalty;      /**< Presence penalty.   0.0 = disabled.              */
    int32_t  repeat_last_n;     /**< Window for repeat penalties. -1 = full ctx.      */
    float    dry_mult;          /**< DRY multiplier. 0.0 = disabled.                  */
    float    dry_base;          /**< DRY base (default 1.75).                         */
    int32_t  dry_allowed_len;   /**< DRY allowed sequence length (default 2).         */
    int32_t  dry_last_n;        /**< DRY penalty window. -1 = full ctx.               */
    int32_t  mirostat;          /**< 0 = off, 1 = Mirostat v1, 2 = Mirostat v2.      */
    float    mirostat_tau;      /**< Mirostat target entropy (default 5.0).           */
    float    mirostat_eta;      /**< Mirostat learning rate (default 0.1).            */
    int32_t  grammar_lazy;      /**< 0 = apply grammar always, 1 = lazy (CRANE).     */
} ion7_csampler_params_t;

/** Opaque handle to a common_sampler instance. */
typedef struct ion7_csampler ion7_csampler_t;

/**
 * Create a common_sampler.
 *
 * @param model            Loaded model (used for vocab and grammar compilation).
 * @param params           Numeric sampling params. NULL = use all defaults.
 * @param grammar          GBNF grammar string, or NULL for no grammar.
 * @param trigger_words    Words that activate grammar_lazy. NULL = none.
 * @param n_triggers       Length of trigger_words array.
 * @param logit_bias_ids   Token IDs for manual logit bias. NULL = none.
 * @param logit_bias_val   Corresponding bias values (added to raw logits).
 * @param n_logit_bias     Length of both logit bias arrays.
 * @return                 Opaque handle. Free with ion7_csampler_free.
 */
ion7_csampler_t* ion7_csampler_init(
    const struct llama_model*     model,
    const ion7_csampler_params_t* params,
    const char*                   grammar,
    const char**                  trigger_words,
    int                           n_triggers,
    const int32_t*                logit_bias_ids,
    const float*                  logit_bias_val,
    int                           n_logit_bias);

/** Free a common_sampler. Safe to call with NULL. */
void ion7_csampler_free(ion7_csampler_t* s);

/**
 * Sample a token from context logits at batch index idx.
 * @param grammar_first  1 = apply grammar before other samplers (more correct, slower).
 * @return               Sampled token ID.
 */
int32_t ion7_csampler_sample(ion7_csampler_t* s, struct llama_context* ctx,
                              int idx, int grammar_first);

/** Notify the sampler that token was accepted (updates grammar, DRY, mirostat state). */
void ion7_csampler_accept(ion7_csampler_t* s, int32_t token);

/**
 * Combined sample + accept in one bridge call - preferred hot-path variant.
 * Equivalent to ion7_csampler_sample() followed by ion7_csampler_accept(),
 * but with a single FFI boundary crossing per generated token.
 */
int32_t ion7_csampler_sample_accept(ion7_csampler_t* s, struct llama_context* ctx, int idx, int grammar_first);

/** Reset all sampler state (grammar automaton, DRY history, mirostat). */
void ion7_csampler_reset(ion7_csampler_t* s);

/** Returns the last accepted token, or -1 if no token has been accepted yet. */
int32_t ion7_csampler_last(const ion7_csampler_t* s);

/** Returns the effective RNG seed used by this sampler instance. */
uint32_t ion7_csampler_get_seed(const ion7_csampler_t* s);

/* =========================================================================
 * ── Speculative decoding (n-gram / draft model) ───────────────────────────
 * ======================================================================= */

/* 1:1 with common_speculative_type enum in common/common.h.
 * Never remap these - breaking changes are intentional when llama.cpp changes.  */
#define ION7_SPEC_NONE          0  /**< No speculative decoding.                 */
#define ION7_SPEC_DRAFT         1  /**< Separate lighter draft model (ctx_dft).  */
#define ION7_SPEC_EAGLE3        2  /**< EAGLE-3 draft heads on target model.     */
#define ION7_SPEC_NGRAM_SIMPLE  3  /**< Simple n-gram from recent context.       */
#define ION7_SPEC_NGRAM_MAP_K   4  /**< N-gram map with prediction statistics.   */
#define ION7_SPEC_NGRAM_MAP_K4V 5  /**< N-gram map with k + 4 m-gram values.     */
#define ION7_SPEC_NGRAM_MOD     6  /**< N-gram with modular prediction.          */
#define ION7_SPEC_NGRAM_CACHE   7  /**< LRU n-gram cache (≈ Cacheback paper).    */

/** Opaque handle to a common_speculative instance. */
typedef struct ion7_speculative ion7_speculative_t;

/**
 * Create a speculative decoder.
 *
 * @param ctx_tgt   Main (target) context. Always required.
 * @param ctx_dft   Draft model context. Required for ION7_SPEC_DRAFT; NULL otherwise.
 * @param type      ION7_SPEC_* constant.
 * @param n_draft   Max draft tokens generated per step (default: 16).
 * @param ngram_min Minimum n-gram size for n-gram types (default: 1).
 * @param ngram_max Maximum n-gram size for n-gram types (default: 4).
 * @return          Opaque handle, or NULL on failure. Free with ion7_speculative_free.
 */
ion7_speculative_t* ion7_speculative_init(
    struct llama_context* ctx_tgt,
    struct llama_context* ctx_dft,
    int type, int n_draft, int ngram_min, int ngram_max);

/** Free a speculative decoder. Safe to call with NULL. */
void ion7_speculative_free(ion7_speculative_t* spec);

/**
 * Notify the speculative decoder of the full current prompt.
 * Call before the generation loop and whenever the prompt changes.
 */
void ion7_speculative_begin(ion7_speculative_t* spec,
                             const int32_t* prompt, int n_prompt);

/**
 * Generate draft tokens from the speculative decoder.
 *
 * @param prompt      Full current token sequence (prompt + accepted tokens so far).
 * @param n_prompt    Length of prompt.
 * @param last_token  Last token accepted by the target model.
 * @param out_draft   Buffer receiving draft token IDs.
 * @param max_draft   Capacity of out_draft. Typically equals n_draft from init.
 * @return            Number of tokens written to out_draft.
 */
int ion7_speculative_draft(ion7_speculative_t* spec,
                            const int32_t* prompt, int n_prompt,
                            int32_t last_token,
                            int32_t* out_draft, int max_draft);

/**
 * Report how many consecutive draft tokens the target model accepted.
 * Must be called after each verification step.
 */
void ion7_speculative_accept(ion7_speculative_t* spec, int n_accepted);

/** Print acceptance rate and effective speedup to stderr. */
void ion7_speculative_stats(const ion7_speculative_t* spec);

/* =========================================================================
 * ── Chat output parsing (tool calls + thinking extraction) ───────────────
 * ======================================================================= */

/**
 * Parse raw model output into structured content, thinking block, and tool calls.
 *
 * Buffers that are too small are NUL-terminated at their capacity limit.
 * Returns 0 = ok, -1 = parse error, 1 = one or more buffers truncated.
 *
 * @param t               Chat templates handle (same as used for apply_template).
 * @param text            Raw model output string (NUL-terminated).
 * @param enable_thinking  1 = thinking was enabled, 0 = disabled, -1 = auto-detect.
 * @param content_buf     Out: main response text.
 * @param content_len     Capacity of content_buf.
 * @param thinking_buf    Out: contents of <think>…</think> block. Empty string if none.
 * @param thinking_len    Capacity of thinking_buf.
 * @param tools_buf       Out: tool calls as a JSON array string. "[]" if none.
 * @param tools_len       Capacity of tools_buf.
 * @param out_has_tools   Out: set to 1 if at least one tool call was parsed.
 */
int ion7_chat_parse(
    ion7_chat_templates_t* t,
    const char*            text,
    int                    enable_thinking,
    char*   content_buf,  int32_t content_len,
    char*   thinking_buf, int32_t thinking_len,
    char*   tools_buf,    int32_t tools_len,
    int*    out_has_tools);

/* =========================================================================
 * ── UTF-8 streaming helpers ───────────────────────────────────────────────
 * ======================================================================= */

/**
 * Returns the expected byte length of a UTF-8 sequence from its first byte.
 * @return 1-4 for valid leading bytes; 0 for continuation bytes or invalid input.
 */
int ion7_utf8_seq_len(uint8_t first_byte);

/**
 * Returns 1 if buf ends on a complete UTF-8 character boundary, 0 otherwise.
 * Use before passing streamed bytes to Lua to avoid splitting multibyte characters.
 */
int ion7_utf8_is_complete(const char* buf, size_t len);

/* =========================================================================
 * ── JSON Schema → GBNF grammar ───────────────────────────────────────────
 * ======================================================================= */

/**
 * Convert a JSON Schema (draft-07) to a GBNF grammar string.
 *
 * The C++ implementation handles $ref, anyOf, allOf, oneOf, and format
 * constraints more robustly than the pure-Lua ion7-grammar module.
 *
 * @param schema_json  NUL-terminated JSON Schema string.
 * @param out          Output buffer (NUL-terminated GBNF).
 * @param out_len      Capacity of out.
 * @return             Bytes needed (including NUL). -1 on parse error.
 *                     If return > out_len, resize and call again.
 */
int ion7_json_schema_to_grammar(const char* schema_json, char* out, size_t out_len);

/* =========================================================================
 * ── Partial regex matching (streaming stop-string detection) ─────────────
 * ======================================================================= */

/** Opaque compiled regex handle. */
typedef struct ion7_regex ion7_regex_t;

/** Compile a regex pattern. Returns NULL on invalid pattern. */
ion7_regex_t* ion7_regex_new(const char* pattern);

/** Free a compiled regex. Safe to call with NULL. */
void ion7_regex_free(ion7_regex_t* r);

/**
 * Match the pattern against text.
 *
 * When partial=1, also returns 1 for text that is a valid prefix of a future
 * full match - used in streaming to decide whether to buffer more tokens.
 *
 * @return  0 = no match (can stop buffering),
 *          1 = partial prefix match (keep buffering),
 *          2 = full match (trigger stop).
 */
int ion7_regex_search(ion7_regex_t* r, const char* text, size_t len, int partial);

/* =========================================================================
 * ── Control vectors (activation steering) ───────────────────────────────
 * ======================================================================= */

/**
 * Apply a control vector to the context.
 *
 * A control vector additively steers model behaviour at inference time
 * without modifying model weights. Effective immediately on next decode.
 *
 * @param data      Float buffer of size n_embd × (il_end - il_start + 1).
 *                  Layout: one n_embd-length vector per layer, layer il_start first.
 * @param len       Total number of floats in data.
 * @param n_embd    Model embedding dimension.
 * @param il_start  First layer to apply (inclusive, 1-based).
 * @param il_end    Last layer to apply  (inclusive, 1-based).
 * @return          0 on success, non-zero on error.
 */
int ion7_cvec_apply(struct llama_context* ctx,
                     const float* data, size_t len,
                     int32_t n_embd, int32_t il_start, int32_t il_end);

/** Remove the currently applied control vector. No-op if none is active. */
void ion7_cvec_clear(struct llama_context* ctx);

/* =========================================================================
 * ── Tensor inspection (for use inside cb_eval callbacks) ─────────────────
 * ======================================================================= */

/** Name of a GGML tensor (empty string if t is NULL). */
const char* ion7_tensor_name(void* t);

/** GGML type enum value of the tensor (-1 if t is NULL). */
int ion7_tensor_type(void* t);

/** Size along dimension dim (0..3). Returns 0 for out-of-range dim or NULL. */
int64_t ion7_tensor_ne(void* t, int dim);

/** Total byte size of the tensor (0 if t is NULL). */
size_t ion7_tensor_nbytes(void* t);

/**
 * Copy tensor data into a pre-allocated F32 CPU buffer.
 *
 * Handles tensors on any backend (CUDA, Metal, CPU) via
 * ggml_backend_tensor_get, and converts F16/BF16 → F32 automatically.
 *
 * @param dst        Destination float buffer (CPU).
 * @param dst_count  Capacity of dst in floats.
 * @return           Number of floats written, or -1 on error / unsupported type.
 */
int ion7_tensor_copy_f32(void* t, float* dst, size_t dst_count);

/* =========================================================================
 * ── NUMA topology ────────────────────────────────────────────────────────
 * ======================================================================= */

/**
 * Configure NUMA memory policy. Call before ion7_backend_init for best effect.
 *
 * @param strategy  0=disabled, 1=distribute (round-robin across nodes),
 *                  2=isolate (local node only - best for single-socket),
 *                  3=numactl (follow numactl configuration),
 *                  4=mirror (replicate on all nodes).
 */
void ion7_numa_init(int strategy);

/** Returns 1 if NUMA was initialised with a non-disabled strategy. */
int ion7_is_numa(void);

/* =========================================================================
 * ── CPU capability detection ─────────────────────────────────────────────
 * ======================================================================= */

/** CPU SIMD/ISA capability flags. All fields: 1 = supported, 0 = not supported. */
typedef struct {
    /* x86 */
    int sse3, ssse3, avx, avx2, avx_vnni, bmi2, f16c, fma;
    int avx512, avx512_vbmi, avx512_vnni, avx512_bf16, amx_int8;
    /* ARM */
    int neon, arm_fma, fp16_va, dotprod, matmul_int8;
    int sve;      /**< 1 if SVE is available.                       */
    int sve_cnt;  /**< SVE vector register size in bytes (e.g. 16). */
    int sme;
    /* Other ISAs */
    int riscv_v;
    int rvv_vlen; /**< RISC-V V vector register length in bits.     */
    int vsx;
    int wasm_simd;
} ion7_cpu_caps_t;

/** Fill *out with the capabilities of the current CPU. Always succeeds. */
void ion7_cpu_caps(ion7_cpu_caps_t* out);

/* =========================================================================
 * ── Log routing ──────────────────────────────────────────────────────────
 * ======================================================================= */

/**
 * Redirect llama.cpp log output to a file.
 * @param path  Destination file path. NULL or "" = restore stderr (default).
 */
void ion7_log_to_file(const char* path);

/**
 * Toggle ISO-8601 timestamp prefix on each log line.
 * @param enable  1 = prefix lines with timestamp, 0 = no prefix (default).
 */
void ion7_log_set_timestamps(int enable);

/* =========================================================================
 * ── Base64 encode / decode (for multimodal image data) ───────────────────
 * ======================================================================= */

/**
 * Encode binary data to Base64.
 * Required output capacity: ceil(len / 3) * 4 + 1 bytes.
 * @return  Number of Base64 characters written (excluding NUL), or -1 if out_len too small.
 */
int ion7_base64_encode(const uint8_t* data, size_t len, char* out, size_t out_len);

/**
 * Decode a Base64 string to binary.
 * Required output capacity: floor(src_len / 4) * 3 + 3 bytes.
 * @return  Number of bytes decoded, or -1 on invalid input or buffer too small.
 */
int ion7_base64_decode(const char* src, size_t src_len, uint8_t* out, size_t out_len);

/* =========================================================================
 * ── JSON utilities (nlohmann/json, already linked via libcommon) ──────────
 * All three functions are string-in / string-out: no C++ types cross the
 * boundary, making them trivial to call from LuaJIT FFI.
 * ======================================================================= */

/**
 * Validate a JSON string.
 * @return  1 if the string is valid JSON, 0 otherwise.
 */
int ion7_json_validate(const char* json_str);

/**
 * Pretty-print a JSON string with 2-space indentation.
 *
 * @param json_str  Input JSON (compact or already formatted).
 * @param out       Output buffer. May be NULL for size query.
 * @param out_len   Capacity of out.
 * @return          Bytes needed (including NUL), or -1 on parse error.
 *                  If return > out_len, resize and call again.
 */
int ion7_json_format(const char* json_str, char* out, size_t out_len);

/**
 * Merge two JSON objects using RFC 7396 JSON Merge Patch semantics.
 * Keys in overlay overwrite matching keys in base; null values delete keys.
 * Useful for building tool schemas and config objects dynamically.
 *
 * @param base     Base JSON object string.
 * @param overlay  Patch JSON object string (applied on top of base).
 * @param out      Output buffer for the merged JSON.
 * @param out_len  Capacity of out.
 * @return         Bytes needed (including NUL), or -1 on parse error.
 */
int ion7_json_merge(const char* base, const char* overlay,
                     char* out, size_t out_len);

/* =========================================================================
 * ── Logprob / Entropy (compute-intensive - C faster than Lua for n_vocab) ─
 * ======================================================================= */

/**
 * Compute the log-probability of token_id at batch position idx.
 *
 * Runs log-softmax over all logits in C - ~151k floats for Qwen3.5.
 * Doing this in Lua costs ~50x more due to tonumber() conversion overhead.
 *
 * @param ctx       Inference context (must have logits enabled for idx).
 * @param idx       Batch position (usually 0 after decode_single).
 * @param token_id  Token whose probability to compute.
 * @return          Log-probability (negative, closer to 0 = more likely).
 */
float ion7_logprob(struct llama_context* ctx, int32_t idx, int32_t token_id);

/**
 * Compute the Shannon entropy (in nats) of the logit distribution at idx.
 *
 * @param ctx  Inference context.
 * @param idx  Batch position.
 * @return     Entropy in nats (>= 0).
 */
float ion7_entropy(struct llama_context* ctx, int32_t idx);

#ifdef __cplusplus
}
#endif
#endif /* ION7_BRIDGE_H */
